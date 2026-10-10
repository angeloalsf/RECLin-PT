"""O treino de ponta a ponta em CPU, com o modelo minúsculo e partições pequenas:
retomada exata, reprodutibilidade, separação entre treino e TEST e os scripts.

Configuração pequena e determinística: as primeiras 4 linhas do TRAIN (1.232
candidatos), 2 do DEV (548) e 2 do TEST (630), 3 épocas, lotes de 32 (39 passos
por época), `max_length` 64, `lr` 1e-3. Tudo no mesmo processo e no mesmo ambiente:
as comparações são de igualdade exata (tensores com `torch.equal`, arquivos
byte a byte).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reclin import particoes  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.execucao import diretorio, predicoes  # noqa: E402
from reclin.treino import checkpoint, classificador  # noqa: E402
from reclin.util.caminhos import CODIGO  # noqa: E402

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"
SCRIPTS = CODIGO / "scripts"
CFG = Config(epochs=3, batch_size=32, max_length=64, lr=1e-3, seed=42)
PASSOS_POR_EPOCA = 39


@pytest.fixture(scope="module")
def pequenas(subconjunto):
    return subconjunto(4, 2, 2)


def treinar(raiz, nome, pasta_particoes, config=CFG, **kw):
    return classificador.treinar_execucao(raiz, nome, config, modelo=MODELO_MINUSCULO,
                                          pasta_particoes=pasta_particoes, **kw)


@pytest.fixture(scope="module")
def continua(tmp_path_factory, pequenas):
    raiz = tmp_path_factory.mktemp("continua")
    registro = treinar(raiz, "c", pequenas)
    assert registro["concluido"] and registro["passos"]["por_epoca"] == PASSOS_POR_EPOCA
    classificador.avaliar_test(raiz / "c", pasta_particoes=pequenas)
    return raiz / "c"


def iguais(a, b) -> bool:
    """Igualdade exata de estruturas com tensores e arrays."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(iguais(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(iguais(x, y) for x, y in zip(a, b))
    return a == b


def sem_duracao(historico):
    return [{k: v for k, v in h.items() if k != "duration_s"} for h in historico]


def conferir_igual_a_continua(pasta: Path, continua: Path) -> None:
    a, b = checkpoint.carregar_estado(continua), checkpoint.carregar_estado(pasta)
    for chave in ("modelo", "otimizador", "agendador", "rng", "gerador_atual", "gerador_inicio_epoca",
                  "posicao", "melhor_f1", "soma_loss_epoca"):
        assert iguais(a[chave], b[chave]), chave
    assert sem_duracao(a["historico"]) == sem_duracao(b["historico"])
    for arquivo in ("model.safetensors", "tokenizer.json"):
        assert (continua / "checkpoints/melhor_modelo" / arquivo).read_bytes() == \
            (pasta / "checkpoints/melhor_modelo" / arquivo).read_bytes(), arquivo
    assert (continua / "predicoes_dev.json").read_bytes() == (pasta / "predicoes_dev.json").read_bytes()
    metricas = [json.loads((p / "metricas.json").read_text("utf-8"))["dev"] for p in (continua, pasta)]
    assert metricas[0] == metricas[1]


# --------------------------------------------------------------------------- #
# Retomada exata                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("parar,descricao", [
    (PASSOS_POR_EPOCA + 7, "meio da época 2"),
    (PASSOS_POR_EPOCA, "fim da época 1"),
    (1, "primeiro passo"),
])
def test_interrompida_e_retomada_igual_a_continua(parar, descricao, tmp_path, pequenas, continua):
    parcial = treinar(tmp_path, "r", pequenas, parar_apos_passo=parar)
    assert not parcial["concluido"] and parcial["posicao"]["passo_global"] == parar
    assert not (tmp_path / "r" / "predicoes_dev.json").exists()
    final = treinar(tmp_path, "r", pequenas, retomar=True)
    assert final["concluido"]
    conferir_igual_a_continua(tmp_path / "r", continua)
    classificador.avaliar_test(tmp_path / "r", pasta_particoes=pequenas)
    assert (tmp_path / "r/predicoes_test.json").read_bytes() == (continua / "predicoes_test.json").read_bytes()


def test_varias_interrupcoes_e_checkpoints_periodicos(tmp_path, pequenas, continua):
    assert not treinar(tmp_path, "v", pequenas, parar_apos_passo=5, checkpoint_a_cada=7)["concluido"]
    assert not treinar(tmp_path, "v", pequenas, retomar=True, parar_apos_passo=2 * PASSOS_POR_EPOCA + 3,
                       checkpoint_a_cada=7)["concluido"]
    final = treinar(tmp_path, "v", pequenas, retomar=True, checkpoint_a_cada=7)
    assert final["concluido"]
    conferir_igual_a_continua(tmp_path / "v", continua)
    tipos = [(s["tipo"], s["status"]) for s in final["sessoes"]]
    assert tipos == [("nova", "interrompido"), ("retomada", "interrompido"), ("retomada", "concluido")]
    assert final["sessoes"][1]["a_partir_de"]["passo_global"] == 5
    assert final["sessoes"][2]["a_partir_de"] == {
        "epoca": 3, "passo_na_epoca": 3, "passo_global": 2 * PASSOS_POR_EPOCA + 3,
        "epoca_concluida": False, "passos_por_epoca": PASSOS_POR_EPOCA, "epocas": 3}


def test_checkpoint_guarda_a_posicao_no_meio_da_epoca(tmp_path, pequenas):
    treinar(tmp_path, "p", pequenas, parar_apos_passo=PASSOS_POR_EPOCA + 7)
    estado = checkpoint.carregar_estado(tmp_path / "p")
    assert estado["posicao"] == {"epoca": 2, "passo_na_epoca": 7, "passo_global": PASSOS_POR_EPOCA + 7,
                                 "epoca_concluida": False, "passos_por_epoca": PASSOS_POR_EPOCA,
                                 "epocas": 3}
    assert estado["agendador"]["last_epoch"] == PASSOS_POR_EPOCA + 7
    assert len(estado["historico"]) == 1 and estado["soma_loss_epoca"] > 0
    assert not torch.equal(estado["gerador_inicio_epoca"], estado["gerador_atual"])
    resumo = json.loads((tmp_path / "p/checkpoints/ultimo.json").read_text(encoding="utf-8"))
    assert resumo["posicao"] == estado["posicao"]


@pytest.mark.parametrize("batch_size", [28, 32])     # 1.232 = 44 × 28 (lotes completos) e 38 × 32 + 16
def test_gerador_avanca_como_uma_passada_completa(batch_size, tmp_path, pequenas):
    """Ao fim de uma época, o gerador do DataLoader está onde uma passada
    completa pelo loader (como a do legado, até o StopIteration) o deixa. Com
    lotes todos completos, parar no último lote sem esgotar o iterador
    deixaria de consumir o último sorteio do RandomSampler."""
    from reclin import entrada, modelos
    cfg = dataclasses.replace(CFG, epochs=1, batch_size=batch_size)
    treinar(tmp_path, "g", pequenas, config=cfg)
    gravado = checkpoint.carregar_estado(tmp_path / "g")["gerador_atual"]
    tok = modelos.carregar_tokenizer(MODELO_MINUSCULO)
    ex = entrada.exemplos(particoes.ler_particao(pequenas, "train"))
    loader = entrada.criar_loader(tok, ex, max_length=cfg.max_length, batch_size=batch_size,
                                  embaralhar=True, seed=cfg.seed)
    for _ in loader:
        pass
    assert torch.equal(gravado, loader.generator.get_state())


# --------------------------------------------------------------------------- #
# Reprodutibilidade                                                           #
# --------------------------------------------------------------------------- #
def test_mesma_configuracao_e_semente_mesmo_resultado(tmp_path, pequenas, continua):
    treinar(tmp_path, "d", pequenas)
    conferir_igual_a_continua(tmp_path / "d", continua)


def test_outra_semente_outro_resultado(tmp_path, pequenas, continua):
    treinar(tmp_path, "s", pequenas, config=dataclasses.replace(CFG, seed=43))
    a = checkpoint.carregar_estado(continua)["modelo"]
    b = checkpoint.carregar_estado(tmp_path / "s")["modelo"]
    assert not all(torch.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- #
# Recusas                                                                     #
# --------------------------------------------------------------------------- #
def test_recusas_de_criacao_e_retomada(tmp_path, pequenas, continua):
    with pytest.raises(classificador.ErroExecucao, match="já existe"):
        treinar(continua.parent, "c", pequenas)
    with pytest.raises(classificador.ErroExecucao, match="não há execução"):
        treinar(tmp_path, "nada", pequenas, retomar=True)
    with pytest.raises(classificador.ErroExecucao, match="já foi concluído"):
        treinar(continua.parent, "c", pequenas, retomar=True)
    treinar(tmp_path, "x", pequenas, parar_apos_passo=3)
    with pytest.raises(classificador.ErroExecucao, match="outra configuração"):
        treinar(tmp_path, "x", pequenas, config=dataclasses.replace(CFG, lr=2e-3), retomar=True)
    with pytest.raises(ValueError, match="positivo"):
        treinar(tmp_path, "y", pequenas, checkpoint_a_cada=0)


# --------------------------------------------------------------------------- #
# Separação entre treino e TEST                                               #
# --------------------------------------------------------------------------- #
def test_treino_nao_le_o_test(tmp_path, subconjunto, monkeypatch, continua):
    lidas = []
    original = particoes.ler_particao
    monkeypatch.setattr(particoes, "ler_particao", lambda p, n: lidas.append(n) or original(p, n))
    sem_test = subconjunto(4, 2, 2, sem_test=True)
    registro = treinar(tmp_path, "t", sem_test)
    assert registro["concluido"] and lidas == ["train", "dev"]
    pasta = tmp_path / "t"
    assert not (pasta / "predicoes_test.json").exists() and not (pasta / diretorio.TRILHA).exists()
    assert set(json.loads((pasta / "metricas.json").read_text(encoding="utf-8"))) == {"dev"}
    assert registro["test_avaliado"] is False
    assert set(diretorio.ler_config(pasta)["conjuntos"]) == {"dev"}
    # o resultado é o mesmo da execução com o arquivo do TEST presente
    assert (pasta / "predicoes_dev.json").read_bytes() == (continua / "predicoes_dev.json").read_bytes()


def test_o_comando_de_treino_nao_tem_opcao_de_test():
    import importlib.util
    spec = importlib.util.spec_from_file_location("treinar_script", SCRIPTS / "treinar.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    destinos = {a.dest for a in modulo.montar_parser()._actions}
    assert not any("test" in d for d in destinos), destinos


def hashes_do_treino(pasta: Path) -> dict[str, str]:
    arquivos = sorted(p for p in (pasta / "checkpoints").rglob("*") if p.is_file())
    arquivos += [pasta / "predicoes_dev.json"]
    return {p.relative_to(pasta).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in arquivos}


def test_avaliar_test_e_separado_e_nao_muda_o_treino(tmp_path, pequenas):
    treinar(tmp_path, "a", pequenas, parar_apos_passo=10)
    with pytest.raises(classificador.ErroExecucao, match="não terminou"):
        classificador.avaliar_test(tmp_path / "a", pasta_particoes=pequenas)
    treinar(tmp_path, "a", pequenas, retomar=True)
    pasta = tmp_path / "a"
    antes, treino_antes = hashes_do_treino(pasta), json.loads((pasta / "treino.json").read_text("utf-8"))
    resultado = classificador.avaliar_test(pasta, pasta_particoes=pequenas)
    assert hashes_do_treino(pasta) == antes
    treino_depois = json.loads((pasta / "treino.json").read_text("utf-8"))
    assert {k for k in treino_antes if treino_antes[k] != treino_depois[k]} == {"test_avaliado"}
    assert resultado["trilha"]["eval_index_for_config"] == 1
    sidecar = predicoes.ler(pasta / "predicoes_test.json")
    assert len(sidecar["y_true"]) == diretorio.ler_config(pasta)["conjuntos"]["test"]["n_candidatos"]
    assert set(json.loads((pasta / "metricas.json").read_text("utf-8"))) == {"dev", "test"}

    with pytest.raises(classificador.ErroExecucao, match="já foi avaliado"):
        classificador.avaliar_test(pasta, pasta_particoes=pequenas)
    primeira = (pasta / "predicoes_test.json").read_bytes()
    segunda = classificador.avaliar_test(pasta, pasta_particoes=pequenas, reavaliar=True)
    assert segunda["trilha"]["eval_index_for_config"] == 2 and segunda["trilha"]["eval_index"] == 2
    assert (pasta / "predicoes_test.json").read_bytes() == primeira


# --------------------------------------------------------------------------- #
# Scripts e compatibilidade com as etapas anteriores                          #
# --------------------------------------------------------------------------- #
def rodar(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True)


def test_scripts_de_ponta_a_ponta(tmp_path, pequenas, continua):
    base = ["--saida", tmp_path, "--particoes", pequenas, "--modelo", MODELO_MINUSCULO,
            "--epochs", 3, "--batch-size", 32, "--max-length", 64, "--lr", "1e-3"]
    r = rodar("treinar.py", "--nome", "s", *base, "--parar-apos-passo", 30)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "--retomar" in r.stdout
    r = rodar("treinar.py", "--nome", "s", *base, "--retomar")
    assert r.returncode == 0, r.stdout + r.stderr
    r = rodar("avaliar_test.py", "--execucao", tmp_path / "s", "--particoes", pequenas)
    assert r.returncode == 0, r.stdout + r.stderr
    for arquivo in ("predicoes_dev.json", "predicoes_test.json"):
        assert (tmp_path / "s" / arquivo).read_bytes() == (continua / arquivo).read_bytes(), arquivo
    # a avaliação e a comparação das etapas anteriores leem a execução nova
    assert rodar("avaliar.py", "--execucao", tmp_path / "s").returncode == 0
    r = rodar("comparar.py", "par", "--a", tmp_path / "s/predicoes_test.json",
              "--b", continua / "predicoes_test.json", "--saida", tmp_path / "cmp.json", "--n-boot", 20)
    assert r.returncode == 0, r.stdout + r.stderr
    mcnemar = json.loads((tmp_path / "cmp.json").read_text("utf-8"))["mcnemar"]
    assert mcnemar["b_only_a_correct"] == mcnemar["c_only_b_correct"] == 0


@pytest.mark.parametrize("argumentos,mensagem", [
    (["--lr", "0"], "lr precisa ser positivo"),
    (["--warmup-ratio", "1.5"], "warmup_ratio"),
    (["--checkpoint-a-cada", "0"], "checkpoint-a-cada precisa ser positivo"),
    (["--modelo", "nao/existe"], "não é uma pasta"),
    (["--retomar"], "não há execução"),
])
def test_treinar_recusa_configuracao_invalida(argumentos, mensagem, tmp_path, pequenas):
    base = {"--modelo": str(MODELO_MINUSCULO)}
    for i in range(0, len(argumentos), 2):
        if argumentos[i] in base:
            base.pop(argumentos[i])
    extras = [x for par in base.items() for x in par]
    r = rodar("treinar.py", "--nome", "e", "--saida", tmp_path, "--particoes", pequenas, *extras, *argumentos)
    assert r.returncode == 1, r.stdout + r.stderr
    assert mensagem in r.stdout


def test_treinar_rejeita_opcoes_invalidas_no_argparse(tmp_path):
    r = rodar("treinar.py", "--nome", "e", "--saida", tmp_path, "--encoder", "gpt")
    assert r.returncode == 2 and "invalid choice" in r.stderr


def test_avaliar_test_recusa_pasta_que_nao_e_execucao(tmp_path):
    r = rodar("avaliar_test.py", "--execucao", tmp_path)
    assert r.returncode == 1 and "não é uma execução de treino" in r.stdout
