"""As estratégias treinadas de ponta a ponta em CPU, com o modelo minúsculo:
retomada, separação entre estratégias, sidecars no conjunto completo, TEST
separado do treino e os comandos.

Partições pequenas (as primeiras 30/10/10 linhas: 20/28/60 pares restritos)
com o léxico congelado das partições completas; 3 épocas, lotes de 4 (5
passos por época no espaço restrito), `max_length` 64, `lr` 1e-3. Os testes
dos comandos usam as partições completas, com 1 época.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from reclin import particoes, tarefa  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.estrategias import baseline, pair_aware, restrito  # noqa: E402
from reclin.execucao import diretorio, predicoes  # noqa: E402
from reclin.negacao import lexico as modulo_lexico  # noqa: E402
from reclin.tarefa import NOREL  # noqa: E402
from reclin.treino import checkpoint, classificador  # noqa: E402
from reclin.util.caminhos import CODIGO, PARTICOES  # noqa: E402

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"
SCRIPTS = CODIGO / "scripts"
PEQUENA = {"epochs": 3, "batch_size": 4, "max_length": 64, "lr": 1e-3, "seed": 42}
CONFIGS = {"restrito": restrito.ConfigRestrito(**PEQUENA), "pair_aware": pair_aware.ConfigPairAware(**PEQUENA)}
MODULOS = {"restrito": restrito, "pair_aware": pair_aware}
PASSOS = 5


@pytest.fixture(scope="module")
def lexico():
    return modulo_lexico.carregar_congelado(particoes.ler_particao(PARTICOES, "train"))


@pytest.fixture(scope="module")
def pequenas(subconjunto):
    return subconjunto(30, 10, 10)


def treinar(raiz, nome, qual, pasta_particoes, lexico, config=None, **kw):
    config = config or CONFIGS[qual]
    return classificador.treinar_execucao(raiz, nome, config, modelo=MODELO_MINUSCULO,
                                          pasta_particoes=pasta_particoes,
                                          montagem=MODULOS[qual].montagem(config, lexico=lexico), **kw)


def avaliar(pasta, qual, pasta_particoes, lexico, **kw):
    return classificador.avaliar_test(pasta, pasta_particoes=pasta_particoes,
                                      montagem=MODULOS[qual].montagem(CONFIGS[qual], lexico=lexico), **kw)


@pytest.fixture(scope="module")
def continuas(tmp_path_factory, pequenas, lexico):
    raiz = tmp_path_factory.mktemp("continuas")
    for qual in MODULOS:
        registro = treinar(raiz, qual, qual, pequenas, lexico)
        assert registro["concluido"] and registro["passos"]["por_epoca"] == PASSOS
        avaliar(raiz / qual, qual, pequenas, lexico)
    return raiz


def iguais(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(iguais(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(iguais(x, y) for x, y in zip(a, b))
    try:
        import numpy as np
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
    except ImportError:
        pass
    return a == b


# --------------------------------------------------------------------------- #
# Retomada                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("qual", ["restrito", "pair_aware"])
@pytest.mark.parametrize("parar", [PASSOS + 2, PASSOS])
def test_interrompida_e_retomada_igual_a_continua(qual, parar, tmp_path, pequenas, lexico, continuas):
    assert not treinar(tmp_path, "r", qual, pequenas, lexico, parar_apos_passo=parar)["concluido"]
    assert treinar(tmp_path, "r", qual, pequenas, lexico, retomar=True)["concluido"]
    a, b = checkpoint.carregar_estado(continuas / qual), checkpoint.carregar_estado(tmp_path / "r")
    for chave in ("modelo", "otimizador", "agendador", "rng", "gerador_atual", "posicao", "melhor_f1"):
        assert iguais(a[chave], b[chave]), chave
    avaliar(tmp_path / "r", qual, pequenas, lexico)
    for arquivo in ("predicoes_dev.json", "predicoes_test.json",
                    "checkpoints/melhor_modelo/model.safetensors"):
        assert (continuas / qual / arquivo).read_bytes() == (tmp_path / "r" / arquivo).read_bytes(), arquivo
    if qual == "pair_aware":
        cab = "checkpoints/melhor_modelo/pair_aware_head.safetensors"
        assert (continuas / qual / cab).read_bytes() == (tmp_path / "r" / cab).read_bytes()


# --------------------------------------------------------------------------- #
# O que cada estratégia grava                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("qual", ["restrito", "pair_aware"])
def test_sidecars_no_conjunto_completo_com_os_indices(qual, pequenas, lexico, continuas):
    pasta = continuas / qual
    for particao, extras in (("dev", ["best_dev_macro_f1_history", "dev_macro_f1_recomputed",
                                      "dev_negation_of_f1_recomputed"]), ("test", [])):
        s = predicoes.ler(pasta / f"predicoes_{particao}.json")
        cands = tarefa.candidatos(particoes.ler_particao(pequenas, particao))
        indices = MODULOS[qual].selecionar(cands, lexico)
        assert s["y_true"] == tarefa.y_true(cands)                     # o y_true do conjunto completo
        assert s["restricted_indices"] == indices and s["n_restrito"] == len(indices)
        fora = [i for i in range(len(cands)) if i not in set(indices)]
        assert all(s["y_pred"][i] == NOREL and s["probs"][i] == [0.0, 0.0, 1.0] for i in fora)
        esperadas = (["model", "seed", "labels", "y_true", "y_pred", "espaco"]
                     + (["cabeca"] if qual == "pair_aware" else [])
                     + ["encoder", "best_epoch", "restored_best_state", "split", "n_restrito",
                        "restricted_indices"] + extras + ["probs"])
        assert list(s) == esperadas


@pytest.mark.parametrize("qual", ["restrito", "pair_aware"])
def test_historico_e_registros(qual, continuas):
    treino = json.loads((continuas / qual / "treino.json").read_text("utf-8"))
    assert set(treino["dev_history"][0]) == {"epoch", "train_loss", "dev_loss", "dev_macro_f1",
                                             "dev_negation_of_f1", "dev_restrito_macro_f1",
                                             "dev_restrito_negation_of_f1", "duration_s"}
    assert treino["subconjunto"]["train"]["n_restrito"] == 20 and treino["n_candidatos"]["train"] == 5310
    config = diretorio.ler_config(continuas / qual)
    assert config["estrategia"] == qual and config["config"]["estrategia_config"]["lexico"]["lexico_sha1"] == \
        modulo_lexico.CONGELADO["lexico_sha1"]
    metricas = json.loads((continuas / qual / "metricas.json").read_text("utf-8"))
    assert set(metricas) == {"dev", "dev_restrito", "test", "test_restrito"}
    if qual == "pair_aware":
        assert set(treino["descricao"]["marcadores_ausentes"]) == {"train", "dev"}
        assert set(treino["descricao_test"]["marcadores_ausentes"]) == {"test"}
        assert treino["descricao"]["cabeca"]["mlp_hidden"] == 16


def test_baseline_e_o_classificador_da_etapa_5(tmp_path, pequenas):
    config = Config(**PEQUENA)
    classificador.treinar_execucao(tmp_path, "c", config, modelo=MODELO_MINUSCULO, pasta_particoes=pequenas)
    classificador.treinar_execucao(tmp_path, "b", config, modelo=MODELO_MINUSCULO, pasta_particoes=pequenas,
                                   montagem=baseline.montagem(config))
    assert (tmp_path / "c/predicoes_dev.json").read_bytes() == (tmp_path / "b/predicoes_dev.json").read_bytes()
    assert iguais(checkpoint.carregar_estado(tmp_path / "c")["modelo"],
                  checkpoint.carregar_estado(tmp_path / "b")["modelo"])
    assert diretorio.ler_config(tmp_path / "b")["estrategia"] == "baseline"


# --------------------------------------------------------------------------- #
# Separação entre estratégias                                                 #
# --------------------------------------------------------------------------- #
def test_uma_estrategia_nao_retoma_a_execucao_de_outra(tmp_path, pequenas, lexico):
    treinar(tmp_path, "x", "restrito", pequenas, lexico, parar_apos_passo=3)
    with pytest.raises(classificador.ErroExecucao, match="outra configuração"):
        treinar(tmp_path, "x", "pair_aware", pequenas, lexico, retomar=True)
    with pytest.raises(classificador.ErroExecucao, match="outra configuração"):
        classificador.treinar_execucao(tmp_path, "x", Config(**PEQUENA), modelo=MODELO_MINUSCULO,
                                       pasta_particoes=pequenas, montagem=baseline.montagem(), retomar=True)
    outro_lexico = {k: v for k, v in lexico.items() if k != "sem"}
    with pytest.raises(classificador.ErroExecucao, match="estrategia_config"):
        treinar(tmp_path, "x", "restrito", pequenas, outro_lexico, retomar=True)
    with pytest.raises(classificador.ErroExecucao, match="outra configuração"):
        treinar(tmp_path, "x", "restrito", pequenas, lexico, retomar=True,
                config=dataclasses.replace(CONFIGS["restrito"], seed=43))
    with pytest.raises(classificador.ErroExecucao, match="já existe"):
        treinar(tmp_path, "x", "pair_aware", pequenas, lexico)
    assert treinar(tmp_path, "x", "restrito", pequenas, lexico, retomar=True)["concluido"]


def test_config_de_outra_estrategia_e_recusada(tmp_path, pequenas, lexico):
    with pytest.raises(classificador.ErroExecucao, match="usa ConfigPairAware"):
        treinar(tmp_path, "y", "pair_aware", pequenas, lexico, config=CONFIGS["restrito"])


def test_avaliar_test_com_a_montagem_de_outra_estrategia(continuas, pequenas, lexico):
    with pytest.raises(classificador.ErroExecucao, match="é da estratégia 'restrito'"):
        classificador.avaliar_test(continuas / "restrito", pasta_particoes=pequenas, reavaliar=True,
                                   montagem=pair_aware.montagem(CONFIGS["pair_aware"], lexico=lexico))
    with pytest.raises(classificador.ErroExecucao, match="é da estratégia 'pair_aware'"):
        classificador.avaliar_test(continuas / "pair_aware", pasta_particoes=pequenas, reavaliar=True)


@pytest.mark.parametrize("qual", ["restrito", "pair_aware"])
def test_treino_nao_le_o_test(qual, tmp_path, subconjunto, lexico, monkeypatch, continuas):
    lidas = []
    original = particoes.ler_particao
    monkeypatch.setattr(particoes, "ler_particao", lambda p, n: lidas.append(n) or original(p, n))
    sem_test = subconjunto(30, 10, 10, sem_test=True)
    assert treinar(tmp_path, "t", qual, sem_test, lexico)["concluido"]
    assert lidas == ["train", "dev"]
    assert not (tmp_path / "t" / "predicoes_test.json").exists()
    assert (tmp_path / "t/predicoes_dev.json").read_bytes() == (continuas / qual / "predicoes_dev.json").read_bytes()


# --------------------------------------------------------------------------- #
# Comandos                                                                    #
# --------------------------------------------------------------------------- #
def rodar(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def pelos_comandos(tmp_path_factory):
    """Restrito (interrompido e retomado) e Pair-Aware pelos scripts, nas
    partições completas, 1 época, e o TEST de cada um."""
    raiz = tmp_path_factory.mktemp("comandos")
    base = ["--saida", raiz, "--modelo", MODELO_MINUSCULO, "--epochs", 1, "--lr", "1e-3"]
    saidas = [rodar("treinar.py", "--estrategia", "restrito", *base, "--parar-apos-passo", 40),
              rodar("treinar.py", "--estrategia", "restrito", *base, "--retomar"),
              rodar("treinar.py", "--estrategia", "pair_aware", *base),
              rodar("avaliar_test.py", "--execucao", raiz / "restrito_biobertpt_seed42"),
              rodar("avaliar_test.py", "--execucao", raiz / "pairaware_biobertpt_seed42")]
    return raiz, saidas


def test_comandos_de_ponta_a_ponta(pelos_comandos):
    raiz, saidas = pelos_comandos
    assert [s.returncode for s in saidas] == [3, 0, 0, 0, 0], [s.stdout[-1500:] for s in saidas]
    for nome in ("restrito_biobertpt_seed42", "pairaware_biobertpt_seed42"):
        assert (raiz / nome / "predicoes_test.json").is_file()
        assert len(predicoes.ler(raiz / nome / "predicoes_test.json")["y_true"]) == 19210
    restr = predicoes.ler(raiz / "restrito_biobertpt_seed42/predicoes_test.json")
    assert restr["n_restrito"] == 800                       # o espaço restrito do TEST (dados.json)


def test_avaliacao_e_comparacao_das_etapas_anteriores(pelos_comandos, tmp_path):
    raiz, _ = pelos_comandos
    assert rodar("avaliar.py", "--execucao", raiz / "pairaware_biobertpt_seed42").returncode == 0
    r = rodar("comparar.py", "par", "--a", raiz / "pairaware_biobertpt_seed42/predicoes_test.json",
              "--b", raiz / "restrito_biobertpt_seed42/predicoes_test.json", "--saida", tmp_path / "c.json",
              "--n-boot", 20)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "c.json").read_text("utf-8"))["n_test"] == 19210


@pytest.mark.parametrize("argumentos,mensagem", [
    (["--estrategia", "pair_aware", "--mlp-hidden", "0"], "mlp_hidden precisa ser positivo"),
    (["--estrategia", "pair_aware", "--head-dropout", "1.5"], "head_dropout"),
    (["--estrategia", "restrito", "--lr", "0"], "lr precisa ser positivo"),
])
def test_treinar_recusa_configuracao_invalida(argumentos, mensagem, tmp_path):
    r = rodar("treinar.py", "--saida", tmp_path, "--modelo", MODELO_MINUSCULO, *argumentos)
    assert r.returncode == 1 and mensagem in r.stdout, r.stdout + r.stderr


def test_restrito_recusa_particoes_com_outro_lexico(tmp_path, pequenas):
    """O restrito e a Pair-Aware usam o léxico congelado: um TRAIN que induz
    outro léxico é recusado, em vez de treinar num espaço diferente."""
    r = rodar("treinar.py", "--estrategia", "restrito", "--saida", tmp_path, "--particoes", pequenas,
              "--modelo", MODELO_MINUSCULO)
    assert r.returncode == 1 and "deixariam de ser comparáveis" in r.stdout


def test_opcoes_da_cabeca_so_na_pair_aware(tmp_path):
    r = rodar("treinar.py", "--estrategia", "baseline", "--saida", tmp_path, "--mlp-hidden", "8")
    assert r.returncode == 2 and "--mlp-hidden" in r.stderr
    r = rodar("treinar.py", "--estrategia", "x")
    assert r.returncode == 2 and "invalid choice" in r.stderr


def test_avaliar_test_de_estrategia_desconhecida(tmp_path):
    (tmp_path / "e").mkdir()
    (tmp_path / "e/config.json").write_text(json.dumps({"config": {"estrategia": "two_stage"}}), "utf-8")
    (tmp_path / "e/treino.json").write_text("{}", "utf-8")
    r = rodar("avaliar_test.py", "--execucao", tmp_path / "e")
    assert r.returncode == 1 and "desconhecida" in r.stdout
