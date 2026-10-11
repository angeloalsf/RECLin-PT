"""Two-Stage, baseline calibrado, proveniência e validação técnica de ponta a
ponta em CPU, com o modelo minúsculo (A3–A14, A17, A19–A21).

Validação técnica, não avaliação científica: o modelo minúsculo é aleatório e
os números não dizem nada sobre desempenho.

Nenhum destes testes lê o TEST real. As partições são pequenas (as 6
primeiras linhas do TRAIN e as 4 primeiras do DEV congelados) e a partição
"test" delas é um TEST SINTÉTICO: as linhas 5 a 8 do DEV congelado, com o seu
próprio MANIFEST. Ele existe só para exercitar o caminho da avaliação do TEST
(recusas, avaliação única, trilha). O treino e a escolha de τ e de t rodam numa
cópia sem o arquivo do TEST.

Os pesos de partida são uma cópia do modelo minúsculo com um `PROVENIENCIA.json`
sintético (commit fictício), como a pasta que `scripts/pretreino.py baixar`
produz para o BioBERTpt.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from reclin import entrada, modelos, particoes, pretreino  # noqa: E402
from reclin.avaliacao import limiar as busca_limiar  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.estrategias import baseline, baseline_calibrado, two_stage  # noqa: E402
from reclin.estrategias.two_stage import composicao, inicializacao, treino  # noqa: E402
from reclin.execucao import diretorio, predicoes, validacao  # noqa: E402
from reclin.tarefa import NOREL  # noqa: E402
from reclin.treino import checkpoint, classificador, laco  # noqa: E402
from reclin.util.caminhos import CODIGO, PARTICOES  # noqa: E402
from reclin.util.io import gravar_json, ler_json  # noqa: E402

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"
RESULTADOS_LEGADO = CODIGO / "testes" / "referencia" / "resultados_legado"
SCRIPTS = CODIGO / "scripts"
COMMIT = "0" * 40
PEQ = {"batch_size": 16, "max_length": 64, "lr": 1e-3}
CLI_PEQ = ["--batch-size", 16, "--max-length", 64, "--lr", "1e-3", "--threads", 1]
PASSOS = {1: 104, 2: 6}            # passos por época de cada estágio nestas partições


def rodar(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True)


def config(estagio, **kw):
    return two_stage.CONFIGS[estagio](**{**PEQ, "epochs": 2, **kw})


@pytest.fixture(scope="module")
def pre(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("pretreino") / "biobertpt-minusculo"
    shutil.copytree(MODELO_MINUSCULO, pasta)
    pretreino.gravar(pasta, repositorio="teste/minusculo", commit=COMMIT)
    return pasta


@pytest.fixture(scope="module")
def part(tmp_path_factory):
    """TRAIN e DEV pequenos e um TEST SINTÉTICO (linhas do DEV), com MANIFEST."""
    pasta = tmp_path_factory.mktemp("particoes_two_stage")
    train = (PARTICOES / "train.jsonl").read_bytes().splitlines(keepends=True)
    dev = (PARTICOES / "dev.jsonl").read_bytes().splitlines(keepends=True)
    (pasta / "train.jsonl").write_bytes(b"".join(train[:6]))
    (pasta / "dev.jsonl").write_bytes(b"".join(dev[:4]))
    (pasta / "test.jsonl").write_bytes(b"".join(dev[4:8]))          # TEST sintético
    particoes.gravar_manifesto(pasta, particoes.calcular_manifesto(
        pasta, seed=42, origem={"path": "TRAIN/DEV congelados; test sintético (linhas 5-8 do DEV)", "sha256": None}))
    return pasta


@pytest.fixture(scope="module")
def part_sem_test(part, tmp_path_factory):
    """As mesmas partições, com o mesmo MANIFEST, sem o arquivo do TEST."""
    pasta = tmp_path_factory.mktemp("particoes_sem_test")
    for nome in ("train.jsonl", "dev.jsonl", "MANIFEST.json"):
        shutil.copy(part / nome, pasta / nome)
    return pasta


@pytest.fixture(scope="module")
def estagios(tmp_path_factory, pre, part_sem_test):
    raiz = tmp_path_factory.mktemp("estagios")
    for estagio in (1, 2):
        treino.treinar_estagio(raiz, f"e{estagio}", estagio, config(estagio), modelo=pre,
                               pasta_particoes=part_sem_test)
    return raiz


@pytest.fixture(scope="module")
def composta(estagios, part_sem_test, tmp_path_factory):
    raiz = tmp_path_factory.mktemp("composicao")
    resultado = composicao.compor_dev(estagios / "e1", estagios / "e2", raiz, "comp", pasta_particoes=part_sem_test)
    return raiz / "comp", resultado


@pytest.fixture(scope="module")
def base(tmp_path_factory, pre, part_sem_test):
    """Baseline do código novo, 1 época, com os pesos com proveniência."""
    raiz = tmp_path_factory.mktemp("baseline")
    cfg = Config(**PEQ, epochs=1)
    classificador.treinar_execucao(raiz, "baseline_minusculo", cfg, modelo=pre, pasta_particoes=part_sem_test,
                                   montagem=baseline.montagem(cfg))
    return raiz / "baseline_minusculo"


def _bytes(pasta, *nomes):
    return {n: (pasta / n).read_bytes() for n in nomes}


# --------------------------------------------------------------------------- #
# A3, A4 — inicialização independente a partir dos pesos originais            #
# --------------------------------------------------------------------------- #
def test_inicializacao_independente_dos_dois_estagios(estagios, pre):
    t1, t2 = ler_json(estagios / "e1" / "treino.json"), ler_json(estagios / "e2" / "treino.json")
    i1, i2 = t1["inicializacao"], t2["inicializacao"]
    for i in (i1, i2):
        assert i["encoder"]["iguais_ao_checkpoint"] is True and i["encoder"]["n_tensores"] == 39
        assert i["cabeca"]["nova"] is True and i["cabeca"]["formato"] == [2, 16]
        assert i["marcadores"]["ids"] == [118, 119, 120, 121] and i["marcadores"]["vocabulario_original"] == 118
        assert i["pooler"]["novo"] is False
    # os dois partem do mesmo encoder (o do arquivo de pesos do commit)…
    assert i1["encoder"]["sha256"] == i2["encoder"]["sha256"]
    assert i1["sha256_encoder_completo_inicial"] == i2["sha256_encoder_completo_inicial"]
    # …com a mesma cabeça inicial (mesmo formato, mesma semente: instâncias separadas)…
    assert i1["cabeca"]["sha256"] == i2["cabeca"]["sha256"]
    # …e cada um é ajustado de forma independente
    assert t1["final"]["encoder_mudou_no_treino"] and t2["final"]["encoder_mudou_no_treino"]
    assert t1["final"]["sha256_encoder_completo_final"] != t2["final"]["sha256_encoder_completo_final"]
    for t in (t1, t2):
        assert ler_json(estagios / t["nome"] / "config.json")["config"]["proveniencia_pesos"] == pretreino.conferir(pre)


def test_verificar_inicializacao_pelo_comando(pre, estagios, tmp_path):
    r = rodar("two_stage.py", "verificar-inicializacao", "--modelo", pre, "--saida", tmp_path / "r.json")
    assert r.returncode == 0, r.stdout + r.stderr
    relatorio = ler_json(tmp_path / "r.json")
    for estagio in (1, 2):
        registrado = ler_json(estagios / f"e{estagio}" / "treino.json")["inicializacao"]
        assert relatorio[f"estagio{estagio}"] == registrado    # o que o treino verificou é o que o comando mostra
    assert relatorio["proveniencia_pesos"]["commit"] == COMMIT


def test_verificacao_da_carga_acusa_divergencias(pre):
    from reclin.treino import reprodutibilidade
    reprodutibilidade.fixar_sementes(42)
    tokenizer, rede, _, _ = treino.carregar_modelo(pre, 1)
    info = {"missing_keys": ["classifier.weight", "classifier.bias"], "unexpected_keys": [], "mismatched_keys": []}
    inicializacao.verificar_carga(pre, rede, info, tokenizer)
    with pytest.raises(inicializacao.ErroInicializacao, match="chaves novas inesperadas"):
        inicializacao.verificar_carga(pre, rede, {**info, "missing_keys": info["missing_keys"]
                                                  + ["bert.encoder.layer.0.output.dense.weight"]}, tokenizer)
    with pytest.raises(inicializacao.ErroInicializacao, match="chaves novas inesperadas"):
        inicializacao.verificar_carga(pre, rede, {**info, "missing_keys": []}, tokenizer)
    with torch.no_grad():
        rede.bert.encoder.layer[0].output.dense.weight[0, 0] += 1.0
    with pytest.raises(inicializacao.ErroInicializacao, match="não é o do pré-treino"):
        inicializacao.verificar_carga(pre, rede, info, tokenizer)


def _copia_com_proveniencia(origem, destino, alterar=None):
    shutil.copytree(origem, destino, ignore=shutil.ignore_patterns(pretreino.ARQUIVO))
    if alterar:
        alterar(destino)
    pretreino.gravar(destino, repositorio="teste/minusculo", commit=COMMIT)
    return destino


def test_recusa_pontos_de_partida_ajustados(estagios, pre, part_sem_test, tmp_path, base):
    melhor_e1 = estagios / "e1" / "checkpoints" / "melhor_modelo"

    def config_com(campo, valor):
        def alterar(pasta):
            c = json.loads((pasta / "config.json").read_text())
            c[campo] = valor
            (pasta / "config.json").write_text(json.dumps(c))
        return alterar

    casos = {
        "melhor_modelo do estágio 1": melhor_e1,
        "melhor_modelo do baseline": base / "checkpoints" / "melhor_modelo",
        "cópia do melhor_modelo fora da execução": _copia_com_proveniencia(melhor_e1, tmp_path / "copia_e1"),
        "rótulos da tarefa": _copia_com_proveniencia(pre, tmp_path / "rotulos",
                                                     config_com("id2label", {"0": "negation_of", "1": "x"})),
        "cabeça de classificação": _copia_com_proveniencia(pre, tmp_path / "arq",
                                                           config_com("architectures",
                                                                      ["BertForSequenceClassification"])),
        "sem PROVENIENCIA.json": MODELO_MINUSCULO,
        "identificador do Hub": "pucpr/biobertpt-all",
    }
    for descricao, origem in casos.items():
        with pytest.raises((inicializacao.ErroInicializacao, pretreino.ErroProveniencia)):
            inicializacao.recusar_origem(origem)
        with pytest.raises(classificador.ErroExecucao):
            treino.treinar_estagio(tmp_path / "x", "n", 2, config(2), modelo=origem, pasta_particoes=part_sem_test)
        assert not (tmp_path / "x" / "n").exists(), descricao         # recusado antes de criar a execução


# --------------------------------------------------------------------------- #
# A5, A6, A7 — independência, retomada, seleção                               #
# --------------------------------------------------------------------------- #
ARQUIVOS_ESTAGIO = ("predicoes_dev_estagio.json", "metricas_estagio.json")


def _resumo(pasta):
    t = ler_json(pasta / "treino.json")
    estado = checkpoint.carregar_estado(pasta)
    return {"historico": [{k: v for k, v in h.items() if k != "duration_s"} for h in t["dev_history"]],
            "melhor_epoca": t["melhor_epoca"], "final": t["final"], "arquivos": _bytes(pasta, *ARQUIVOS_ESTAGIO),
            "pesos": {k: v.clone() for k, v in estado["modelo"].items()}}


def _iguais(a, b):
    assert a["historico"] == b["historico"] and a["melhor_epoca"] == b["melhor_epoca"]
    assert a["final"] == b["final"] and a["arquivos"] == b["arquivos"]
    assert a["pesos"].keys() == b["pesos"].keys()
    assert all(torch.equal(a["pesos"][k], b["pesos"][k]) for k in a["pesos"])


def test_estagio2_nao_depende_do_estagio1(estagios, pre, part_sem_test, tmp_path):
    treino.treinar_estagio(tmp_path, "e2_sozinho", 2, config(2), modelo=pre, pasta_particoes=part_sem_test)
    _iguais(_resumo(estagios / "e2"), _resumo(tmp_path / "e2_sozinho"))
    with pytest.raises(classificador.ErroExecucao, match="outra configuração"):
        treino.treinar_estagio(estagios, "e1", 2, config(2), modelo=pre, pasta_particoes=part_sem_test, retomar=True)
    with pytest.raises(classificador.ErroExecucao, match="já foi concluído"):
        treino.treinar_estagio(estagios, "e1", 1, config(1), modelo=pre, pasta_particoes=part_sem_test, retomar=True)


@pytest.mark.parametrize("estagio", [pytest.param(1, marks=pytest.mark.lento), 2])
@pytest.mark.parametrize("parada", ["meio", "fim"])
def test_retomada_exata(estagios, pre, part_sem_test, tmp_path, estagio, parada):
    passo = PASSOS[estagio] // 2 if parada == "meio" else PASSOS[estagio]
    r = treino.treinar_estagio(tmp_path, "s", estagio, config(estagio), modelo=pre, pasta_particoes=part_sem_test,
                               parar_apos_passo=passo)
    assert not r["concluido"]
    r = treino.treinar_estagio(tmp_path, "s", estagio, config(estagio), modelo=pre, pasta_particoes=part_sem_test,
                               retomar=True)
    assert r["concluido"] and [s["tipo"] for s in r["sessoes"]] == ["nova", "retomada"]
    _iguais(_resumo(estagios / f"e{estagio}"), _resumo(tmp_path / "s"))


def test_recarga_do_melhor_modelo_reproduz_as_predicoes(estagios, part_sem_test):
    for estagio in (1, 2):
        pasta = estagios / f"e{estagio}"
        gravado = ler_json(pasta / "predicoes_dev_estagio.json")
        tokenizer, rede = modelos.recarregar(pasta / "checkpoints" / "melhor_modelo")
        dev = entrada.exemplos(particoes.ler_particao(part_sem_test, "dev"))
        from reclin.estrategias.two_stage import estagios as modulo_estagios
        _, ex = modulo_estagios.exemplos_do_estagio(estagio, dev)
        loader = entrada.criar_loader(tokenizer, ex, max_length=64, batch_size=16, embaralhar=False, seed=42)
        preds, probs = laco.prever(rede, loader, "cpu", com_probs=True)
        assert preds == gravado["y_pred"] and predicoes.arredondar_probs(probs) == gravado["probs"]


def test_selecao_pelo_campo_do_estagio(estagios):
    for estagio in (1, 2):
        t = ler_json(estagios / f"e{estagio}" / "treino.json")
        campo = two_stage.CAMPO_SELECAO[estagio]
        assert t["campo_selecao"] == campo and all(campo in h for h in t["dev_history"])
        assert (t["melhor_epoca"], t["melhor_valor_selecao"]) == laco.melhor_epoca(t["dev_history"], campo)
        valores = [h[campo] for h in t["dev_history"]]
        assert t["melhor_epoca"] == valores.index(max(valores)) + 1                 # primeiro máximo
        registro = ler_json(estagios / f"e{estagio}" / "config.json")["config"]
        assert registro["estrategia_config"]["campo_selecao"] == campo
        assert t["n_exemplos"]["train"] == (1652 if estagio == 1 else 91)
        assert len(t["pesos_classe"]) == 2


# --------------------------------------------------------------------------- #
# A8–A12 — τ pelo fluxo real, avaliação do TEST, isolamento, compatibilidade  #
# --------------------------------------------------------------------------- #
def test_busca_de_tau_pelo_fluxo_real(composta, estagios, part_sem_test):
    pasta, resultado = composta
    congelado = ler_json(pasta / "limiar.json")
    assert congelado["limiar"] == resultado["tau"] and len(congelado["tabela"]) == 19
    assert congelado["configuracao_busca"]["grade"] == list(busca_limiar.GRADE)
    e1, e2 = composicao.ler_estagio(estagios / "e1", 1), composicao.ler_estagio(estagios / "e2", 2)
    pipeline = composicao.Pipeline(e1, e2, "cpu")
    dev = entrada.exemplos(particoes.ler_particao(part_sem_test, "dev"))
    probs1 = pipeline.probabilidades_estagio1(dev)
    for linha in congelado["tabela"]:
        y_pred, roteados, _ = pipeline.fluxo(dev, probs1, linha["limiar"])
        assert busca_limiar.pontuar(dev.rotulos, y_pred)["contagens"] == linha["contagens"]
        assert linha["n_roteados"] == len(roteados)
        assert all(y_pred[i] == NOREL for i in range(len(y_pred)) if i not in set(roteados))
    sidecar = diretorio.ler_predicoes(pasta, "dev")
    y_pred, roteados, preds2 = pipeline.fluxo(dev, probs1, resultado["tau"])
    assert sidecar["y_pred"] == y_pred and sidecar["indices_estagio2"] == roteados
    assert sidecar["preds_estagio2"] == preds2 and sidecar["probs_estagio1"] == probs1
    assert sidecar["labels"] == ["negation_of", "associated_with", "no_relation"] and "probs" not in sidecar
    assert all(p[1] == round(p[1], 4) for p in probs1)                              # 4 casas
    assert "dev_diagnostico_estagios" in diretorio.ler_metricas(pasta)


def _copiar_execucao(pasta, destino):
    shutil.copytree(pasta, destino)
    return destino


def test_avaliacao_do_test_recusa_vinculos_incompativeis(composta, estagios, part, tmp_path):
    pasta, _ = composta
    # limiar editado
    c = _copiar_execucao(pasta, tmp_path / "c1")
    registro = ler_json(c / "limiar.json")
    registro["limiar"] = 0.05 if registro["limiar"] != 0.05 else 0.1
    gravar_json(c / "limiar.json", registro)
    with pytest.raises(classificador.ErroExecucao, match="editado"):
        composicao.avaliar_test(c, estagio1=estagios / "e1", estagio2=estagios / "e2", pasta_particoes=part)
    # melhor_modelo de um estágio alterado depois do τ
    e1 = _copiar_execucao(estagios / "e1", tmp_path / "e1_alterado")
    with open(e1 / "checkpoints" / "melhor_modelo" / "config.json", "a") as f:
        f.write(" ")
    with pytest.raises(classificador.ErroExecucao, match="mudou depois do treino"):
        composicao.avaliar_test(pasta, estagio1=e1, estagio2=estagios / "e2", pasta_particoes=part)
    # estágios trocados
    with pytest.raises(classificador.ErroExecucao):
        composicao.avaliar_test(pasta, estagio1=estagios / "e2", estagio2=estagios / "e1", pasta_particoes=part)
    # outro DEV
    outra = tmp_path / "outra"
    outra.mkdir()
    for nome in ("train.jsonl", "test.jsonl"):
        shutil.copy(part / nome, outra / nome)
    (outra / "dev.jsonl").write_bytes(b"".join((PARTICOES / "dev.jsonl").read_bytes()
                                               .splitlines(keepends=True)[:3]))
    particoes.gravar_manifesto(outra, particoes.calcular_manifesto(outra, seed=42, origem={"path": "x", "sha256": None}))
    with pytest.raises(classificador.ErroExecucao, match="não vale"):
        composicao.avaliar_test(pasta, pasta_particoes=outra)
    assert not (pasta / "predicoes_test.json").exists()


def test_avaliacao_do_test_uma_vez(composta, part, part_sem_test):
    pasta, _ = composta
    with pytest.raises(classificador.ErroExecucao, match="ausente"):   # sem o arquivo do TEST, não há avaliação
        composicao.avaliar_test(pasta, pasta_particoes=part_sem_test)
    r = composicao.avaliar_test(pasta, pasta_particoes=part)
    assert r["trilha"]["eval_index_for_config"] == 1
    sidecar = diretorio.ler_predicoes(pasta, "test")
    predicoes.validar(sidecar)
    assert sidecar["limiar_tau"] == ler_json(pasta / "limiar.json")["limiar"]
    with pytest.raises(classificador.ErroExecucao, match="já foi avaliado"):
        composicao.avaliar_test(pasta, pasta_particoes=part)
    r = composicao.avaliar_test(pasta, pasta_particoes=part, reavaliar=True)
    assert r["trilha"]["eval_index_for_config"] == 2
    assert "test" in diretorio.ler_metricas(pasta)


def test_treino_e_escolha_de_tau_nao_leem_o_test(estagios, composta, part_sem_test):
    assert not (part_sem_test / "test.jsonl").exists()
    for nome in ("e1", "e2"):
        assert ler_json(estagios / nome / "treino.json")["concluido"]
    assert (composta[0] / "limiar.json").is_file()


def test_avaliacao_e_comparacao_existentes_aceitam_o_two_stage(composta, base, part, tmp_path):
    pasta, _ = composta
    if not (pasta / "predicoes_test.json").exists():
        composicao.avaliar_test(pasta, pasta_particoes=part)
    base = _copiar_execucao(base, tmp_path / "baseline_minusculo")      # a original segue sem TEST avaliado
    classificador.avaliar_test(base, pasta_particoes=part, montagem=baseline.montagem())
    r = rodar("avaliar.py", "--execucao", pasta)
    assert r.returncode == 0, r.stdout + r.stderr
    r = rodar("comparar.py", "par", "--a", pasta / "predicoes_test.json", "--b", base / "predicoes_test.json",
              "--saida", tmp_path / "comp.json", "--n-boot", 200)
    assert r.returncode == 0, r.stdout + r.stderr
    relatorio = ler_json(tmp_path / "comp.json")
    assert "paired_bootstrap" in relatorio and "mcnemar" in relatorio


# --------------------------------------------------------------------------- #
# A13, A14 — só entradas do código novo; sementes                             #
# --------------------------------------------------------------------------- #
def test_entradas_do_legado_sao_recusadas(estagios, tmp_path, part_sem_test):
    with pytest.raises(classificador.ErroExecucao, match="código novo"):
        composicao.compor_dev(RESULTADOS_LEGADO, estagios / "e2", tmp_path, "c", pasta_particoes=part_sem_test)
    with pytest.raises(baseline_calibrado.ErroCalibracao, match="código novo"):
        baseline_calibrado.calibrar_dev(RESULTADOS_LEGADO, tmp_path, "c")
    r = rodar("baseline_calibrado.py", "calibrar-dev", "--base", RESULTADOS_LEGADO, "--saida", tmp_path)
    assert r.returncode == 1
    r = rodar("pretreino.py", "conferir", "--execucoes", RESULTADOS_LEGADO)
    assert r.returncode == 1 and "código novo" in r.stdout


def test_sementes_divergentes_sao_recusadas(estagios, pre, part_sem_test, tmp_path):
    treino.treinar_estagio(tmp_path, "e2_s43", 2, config(2, seed=43, epochs=1), modelo=pre,
                           pasta_particoes=part_sem_test)
    with pytest.raises(classificador.ErroExecucao, match="seed"):
        composicao.compor_dev(estagios / "e1", tmp_path / "e2_s43", tmp_path, "c", pasta_particoes=part_sem_test)
    assert not (tmp_path / "c").exists()


# --------------------------------------------------------------------------- #
# A17 — proveniência nas execuções                                            #
# --------------------------------------------------------------------------- #
def test_proveniencia_registrada_e_conferida_no_treino(base, pre, part_sem_test, tmp_path):
    registro = ler_json(base / "config.json")["config"]
    assert registro["proveniencia_pesos"] == pretreino.conferir(pre)
    alterado = _copiar_execucao(pre, tmp_path / "pre_alterado")
    with open(alterado / "tokenizer_config.json", "a") as f:
        f.write(" ")
    cfg = Config(**PEQ, epochs=1)
    with pytest.raises(classificador.ErroExecucao, match="não conferem"):
        classificador.treinar_execucao(tmp_path, "b", cfg, modelo=alterado, pasta_particoes=part_sem_test,
                                       montagem=baseline.montagem(cfg))
    assert not (tmp_path / "b").exists()
    # retomar com outra cópia (mesma pasta, manifesto de outro commit) é recusado
    outra = _copiar_execucao(pre, tmp_path / "pre_outra")
    classificador.treinar_execucao(tmp_path, "r", cfg, modelo=outra, pasta_particoes=part_sem_test,
                                   montagem=baseline.montagem(cfg), parar_apos_passo=5)
    pretreino.gravar(outra, repositorio="teste/minusculo", commit="1" * 40)
    with pytest.raises(classificador.ErroExecucao, match="outra configuração"):
        classificador.treinar_execucao(tmp_path, "r", cfg, modelo=outra, pasta_particoes=part_sem_test,
                                       montagem=baseline.montagem(cfg), retomar=True)


def test_sem_marca_nem_manifesto_o_config_nao_muda(tmp_path, part_sem_test):
    cfg = Config(**PEQ, epochs=1)
    classificador.treinar_execucao(tmp_path, "b", cfg, modelo=MODELO_MINUSCULO, pasta_particoes=part_sem_test,
                                   montagem=baseline.montagem(cfg), parar_apos_passo=1)
    registro = ler_json(tmp_path / "b" / "config.json")["config"]
    assert set(registro) == {"estrategia", "config", "modelo", "particoes", "estrategia_config"}


def _execucao_sintetica(raiz, nome, estrategia, cfg, prov, **extra):
    registro = {"estrategia": estrategia, "config": cfg.como_dict(), "modelo": "x", "particoes": {},
                "estrategia_config": {}, **({"proveniencia_pesos": prov} if prov else {}), **extra}
    gravar_json(raiz / nome / "config.json", {"nome": nome, "estrategia": estrategia, "config": registro,
                                              "conjuntos": {}})
    return raiz / nome


def test_conferencia_final_das_execucoes(tmp_path, pre):
    prov = pretreino.conferir(pre)
    outro = {**prov, "commit": "1" * 40}
    from reclin.estrategias import pair_aware, restrito
    ok = [_execucao_sintetica(tmp_path, "baseline_biobertpt_seed42", "baseline", Config(), prov),
          _execucao_sintetica(tmp_path, "restrito_biobertpt_seed42", "restrito", restrito.ConfigRestrito(), prov),
          _execucao_sintetica(tmp_path, "pairaware_biobertpt_seed42", "pair_aware", pair_aware.ConfigPairAware(),
                              prov),
          _execucao_sintetica(tmp_path, "twostage_e1_biobertpt_seed42", "two_stage_e1", two_stage.ConfigEstagio1(),
                              prov),
          _execucao_sintetica(tmp_path, "twostage_e2_biobertpt_seed42", "two_stage_e2", two_stage.ConfigEstagio2(),
                              prov)]
    gravar_json(tmp_path / "twostage_biobertpt_seed42" / "config.json", {
        "nome": "twostage_biobertpt_seed42", "estrategia": "two_stage", "conjuntos": {},
        "config": {"estrategia": "two_stage", "estagios": {"1": {"nome": "twostage_e1_biobertpt_seed42"},
                                                            "2": {"nome": "twostage_e2_biobertpt_seed42"}}}})
    gravar_json(tmp_path / "baseline_calibrado_biobertpt_seed42" / "config.json", {
        "nome": "baseline_calibrado_biobertpt_seed42", "estrategia": "baseline_calibrado", "conjuntos": {},
        "config": {"estrategia": "baseline_calibrado", "base": {"nome": "baseline_biobertpt_seed42"}}})
    derivadas = [tmp_path / "twostage_biobertpt_seed42", tmp_path / "baseline_calibrado_biobertpt_seed42"]
    r = rodar("pretreino.py", "conferir", "--execucoes", *ok, *derivadas)
    assert r.returncode == 0, r.stdout + r.stderr
    casos = {
        "outro commit": _execucao_sintetica(tmp_path, "b_outro", "baseline", Config(), outro),
        "sem proveniência": _execucao_sintetica(tmp_path, "b_sem", "baseline", Config(), None),
        "épocas fora do protocolo, sem marca": _execucao_sintetica(tmp_path, "b_ep", "baseline",
                                                                   Config(epochs=1), prov),
        "validação técnica": _execucao_sintetica(tmp_path, "b_val", "baseline", Config(epochs=1), prov,
                                                 validacao=validacao.bloco(1, 3)),
        "outra semente": _execucao_sintetica(tmp_path, "b_s43", "baseline", Config(seed=43), prov),
    }
    for descricao, pasta in casos.items():
        r = rodar("pretreino.py", "conferir", "--execucoes", *ok, pasta)
        assert r.returncode == 1, descricao
    r = rodar("pretreino.py", "conferir", "--execucoes", ok[0], derivadas[0])          # sem os estágios
    assert r.returncode == 1 and "depende de" in r.stdout


# --------------------------------------------------------------------------- #
# A19 — baseline calibrado                                                    #
# --------------------------------------------------------------------------- #
def test_baseline_calibrado_de_ponta_a_ponta(base, part, tmp_path):
    base_local = _copiar_execucao(base, tmp_path / "baseline_minusculo")
    r = baseline_calibrado.calibrar_dev(base_local, tmp_path, "cal")
    pasta = tmp_path / "cal"
    congelado = ler_json(pasta / "limiar.json")
    assert congelado["limiar"] == r["t"] and congelado["configuracao_busca"]["grade"] == list(busca_limiar.GRADE)
    sidecar_base = diretorio.ler_predicoes(base_local, "dev")
    assert diretorio.ler_predicoes(pasta, "dev")["y_pred"] == baseline_calibrado.prever(sidecar_base["probs"], r["t"])
    with pytest.raises(baseline_calibrado.ErroCalibracao, match="ainda não foi avaliado"):
        baseline_calibrado.aplicar_test(pasta)
    classificador.avaliar_test(base_local, pasta_particoes=part, montagem=baseline.montagem())
    resultado = baseline_calibrado.aplicar_test(pasta)
    sidecar_test = diretorio.ler_predicoes(base_local, "test")
    assert diretorio.ler_predicoes(pasta, "test")["y_pred"] == baseline_calibrado.prever(sidecar_test["probs"],
                                                                                         resultado["t"])
    with pytest.raises(baseline_calibrado.ErroCalibracao, match="já foi avaliado"):
        baseline_calibrado.aplicar_test(pasta)
    assert baseline_calibrado.aplicar_test(pasta, reavaliar=True)["trilha"]["eval_index_for_config"] == 2
    # predições do DEV da base alteradas depois de t: recusado
    c = _copiar_execucao(pasta, tmp_path / "cal2")
    b = _copiar_execucao(base_local, tmp_path / "base2")
    sidecar = ler_json(b / "predicoes_dev.json")
    sidecar["probs"][0][0] = 0.9999
    gravar_json(b / "predicoes_dev.json", sidecar)
    with pytest.raises(baseline_calibrado.ErroCalibracao, match="não vale"):
        baseline_calibrado.aplicar_test(c, base=b, reavaliar=True)
    r = rodar("baseline_calibrado.py", "aplicar-test", "--help")
    assert "--limiar" not in r.stdout and "--t " not in r.stdout                      # não há t manual


# --------------------------------------------------------------------------- #
# A20, A21 — rodada de validação técnica pelos comandos                       #
# --------------------------------------------------------------------------- #
def test_rodada_de_validacao_tecnica(pre, part, part_sem_test, estagios, composta, tmp_path):
    saida = tmp_path / "validacao"
    comum = ["--saida", saida, "--particoes", part_sem_test, "--modelo", pre, *CLI_PEQ]
    r = rodar("treinar.py", "--estrategia", "baseline", "--validacao-epocas", 1, *comum)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "o TEST não é avaliado" in r.stdout
    for estagio, epocas in ((1, 1), (2, 2)):
        r = rodar("two_stage.py", "treinar", "--estagio", estagio, "--validacao-epocas", epocas, *comum)
        assert r.returncode == 0, r.stdout + r.stderr
    nomes = {"baseline": "baseline_validacao_1ep", 1: "twostage_e1_validacao_1ep", 2: "twostage_e2_validacao_2ep"}
    for chave, protocolo, epocas in (("baseline", 3, 1), (1, 3, 1), (2, 10, 2)):
        registro = ler_json(saida / nomes[chave] / "config.json")["config"]
        assert registro["validacao"]["epocas"] == epocas and registro["validacao"]["epocas_protocolo"] == protocolo
        assert registro["config"]["epochs"] == epocas
        assert len(ler_json(saida / nomes[chave] / "treino.json")["dev_history"]) == epocas
    r = rodar("two_stage.py", "compor-dev", "--estagio1", saida / nomes[1], "--estagio2", saida / nomes[2],
              "--saida", saida, "--particoes", part_sem_test, "--threads", 1)
    assert r.returncode == 0, r.stdout + r.stderr
    r = rodar("baseline_calibrado.py", "calibrar-dev", "--base", saida / nomes["baseline"], "--saida", saida)
    assert r.returncode == 0, r.stdout + r.stderr
    composicao_val, calibrado_val = saida / "twostage_validacao", saida / "baseline_calibrado_validacao_1ep"
    assert validacao.da_execucao(composicao_val) and validacao.da_execucao(calibrado_val)
    assert ler_json(composicao_val / "limiar.json")["validacao"]
    assert not (part_sem_test / "test.jsonl").exists()

    recusas = {
        "TEST do baseline": ("avaliar_test.py", "--execucao", saida / nomes["baseline"], "--particoes", part),
        "TEST do Two-Stage": ("two_stage.py", "avaliar-test", "--execucao", composicao_val, "--particoes", part),
        "TEST do calibrado": ("baseline_calibrado.py", "aplicar-test", "--execucao", calibrado_val),
        "checagem final": ("pretreino.py", "conferir", "--execucoes", saida / nomes["baseline"], saida / nomes[1]),
        "comparação": ("comparar.py", "par", "--a", composicao_val / "predicoes_dev.json",
                       "--b", saida / nomes["baseline"] / "predicoes_dev.json", "--saida", tmp_path / "x.json"),
        "mistura na composição": ("two_stage.py", "compor-dev", "--estagio1", saida / nomes[1],
                                  "--estagio2", estagios / "e2", "--saida", saida, "--nome", "misturada",
                                  "--particoes", part_sem_test),
        "τ da fase científica com estágio de validação": ("two_stage.py", "avaliar-test", "--execucao", composta[0],
                                                          "--estagio1", saida / nomes[1], "--particoes", part),
        "--epochs junto com --validacao-epocas": ("treinar.py", "--estrategia", "baseline", "--validacao-epocas", 1,
                                                  "--epochs", 1, "--nome", "z", *comum),
    }
    for descricao, comando in recusas.items():
        r = rodar(*comando)
        assert r.returncode in (1, 2), f"{descricao}: {r.stdout}{r.stderr}"
    for pasta in (saida / nomes["baseline"], composicao_val, calibrado_val):
        assert not (pasta / "predicoes_test.json").exists()
    assert not (saida / "misturada").exists() and not (saida / "z").exists()
    assert os.path.isdir(saida)
