"""Portão da etapa 6: baseline, restrito e Pair-Aware iguais aos do legado.

Referências (todas geradas pelo código do legado; nenhuma foi alterada):

* `dados.json` — o espaço restrito de cada partição completa: SHA-256 dos
  índices e das janelas, tamanho e resumo (contagens, tetos de recall); os
  pesos `balanced` do TRAIN restrito;
* `resultados_legado/finetuning_restrito/restrito_biobertpt_seed{42..46}` e
  `resultados_legado/pair_aware/pairaware_biobertpt_seed42` — as execuções
  oficiais (GPU), com os `restricted_indices` do DEV e (restrito) do TEST;
* `treino/restrito_minusculo_seed42.*` (10 épocas, partições completas) e
  `treino/pairaware_minusculo_seed42.*` (3 épocas) — o legado treinando o
  modelo minúsculo em CPU: histórico, sidecars do DEV e do TEST (com os
  índices e as `probs`), métricas no conjunto completo e no subconjunto,
  pesos de classe, configuração da cabeça e marcadores ausentes;
* `treino/baseline_pequeno_seed42.*` — o baseline (já usado na etapa 5), agora
  pela estratégia `baseline`.

Os índices e as janelas são conferidos em qualquer ambiente; a reprodução
numérica dos treinos, só no ambiente das referências (ver
`test_treino_legado.py`), com `RECLIN_FORCAR_EQUIVALENCIA=1` para forçar.
Os sidecars são comparados byte a byte, com o campo `model` (o caminho do
modelo minúsculo na máquina que gerou as referências) igualado.
"""
from __future__ import annotations

import json
import os
import platform

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from reclin import entrada, particoes, tarefa  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.estrategias import baseline, pair_aware, restrito  # noqa: E402
from reclin.execucao import subconjunto  # noqa: E402
from reclin.negacao import lexico as modulo_lexico  # noqa: E402
from reclin.tarefa import LABELS  # noqa: E402
from reclin.treino import classificador, laco  # noqa: E402
from reclin.util.caminhos import CODIGO, PARTICOES  # noqa: E402
from reclin.util.io import ler_json, sha256_json  # noqa: E402

REFERENCIA = CODIGO / "testes" / "referencia"
TREINO = REFERENCIA / "treino"
RESULTADOS = REFERENCIA / "resultados_legado"
MODELO_MINUSCULO = REFERENCIA / "modelo_minusculo"

ESPERADO = {"sistema": "Linux", "arquitetura": "x86_64", "torch": "2.11.0", "transformers": "5.16.1"}
ATUAL = {"sistema": platform.system(), "arquitetura": platform.machine(),
         "torch": torch.__version__.split("+")[0], "transformers": transformers.__version__}
EXATO = pytest.mark.skipif(
    ATUAL != ESPERADO and os.environ.get("RECLIN_FORCAR_EQUIVALENCIA") != "1",
    reason=f"reprodução numérica do legado: referências geradas em {ESPERADO}, este ambiente é "
           f"{ATUAL} (RECLIN_FORCAR_EQUIVALENCIA=1 roda assim mesmo)")


@pytest.fixture(scope="module")
def lexico(documentos):
    return modulo_lexico.carregar_congelado(documentos["train"])


@pytest.fixture(scope="module")
def espaco(documentos, lexico):
    """Por partição: candidatos, índices do restrito e exemplos completos."""
    saida = {}
    for nome in ("train", "dev", "test"):
        cands = tarefa.candidatos(documentos[nome])
        saida[nome] = {"cands": cands, "indices": restrito.selecionar(cands, lexico),
                       "exemplos": entrada.exemplos(documentos[nome])}
    return saida


# --------------------------------------------------------------------------- #
# Índices e janelas do espaço restrito                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nome", ["train", "dev", "test"])
def test_indices_e_janelas_do_restrito(nome, espaco, referencia_dados):
    ref = referencia_dados["espaco_restrito"][nome]
    e = espaco[nome]
    assert len(e["indices"]) == ref["n_restrito"]
    assert sha256_json(e["indices"]) == ref["indices_sha256"]
    assert sha256_json(e["exemplos"].subconjunto(e["indices"]).textos) == ref["janelas_sha256"]
    assert subconjunto.resumo(e["indices"], e["exemplos"].rotulos) == ref["resumo"]


@pytest.mark.parametrize("nome", ["train", "dev", "test"])
def test_pair_aware_seleciona_os_mesmos_pares(nome, espaco, lexico):
    assert pair_aware.selecionar(espaco[nome]["cands"], lexico) == espaco[nome]["indices"]


def test_pesos_balanced_do_restrito(espaco, referencia_dados):
    e = espaco["train"]
    pesos = laco.pesos_balanced(e["exemplos"].subconjunto(e["indices"]).rotulos)
    assert dict(zip(LABELS, pesos)) == referencia_dados["pesos_balanced"]["espaco_restrito"]


EXECUCOES_OFICIAIS = ([(f"finetuning_restrito/restrito_biobertpt_seed{s}", p)
                       for s in range(42, 47) for p in ("dev", "test")]
                      + [("pair_aware/pairaware_biobertpt_seed42", "dev")])


@pytest.mark.parametrize("execucao,particao", EXECUCOES_OFICIAIS)
def test_indices_iguais_aos_das_execucoes_oficiais(execucao, particao, espaco):
    sufixo = ".dev_preds.json" if particao == "dev" else ".preds.json"
    ref = ler_json(RESULTADOS / f"{execucao}{sufixo}")
    assert ref["restricted_indices"] == espaco[particao]["indices"]
    assert ref["n_restrito"] == len(espaco[particao]["indices"])
    assert ref["y_true"] == espaco[particao]["exemplos"].rotulos


# --------------------------------------------------------------------------- #
# Reprodução dos treinos minúsculos                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def uma_thread():
    antes = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(antes)


def treinar_e_avaliar(raiz, nome, config, montagem, pasta_particoes=PARTICOES):
    treino = classificador.treinar_execucao(raiz, nome, config, modelo=MODELO_MINUSCULO, montagem=montagem,
                                            pasta_particoes=pasta_particoes, dispositivo="cpu")
    classificador.avaliar_test(raiz / nome, montagem=montagem, pasta_particoes=pasta_particoes,
                               dispositivo="cpu")
    return treino


def sem_duracao(historico):
    return [{k: v for k, v in h.items() if k != "duration_s"} for h in historico]


def conferir_sidecars_byte_a_byte(pasta, nome_ref):
    for sufixo, arquivo in ((".dev_preds.json", "predicoes_dev.json"), (".preds.json", "predicoes_test.json")):
        bruto = (TREINO / f"{nome_ref}{sufixo}").read_bytes()
        novo = ler_json(pasta / arquivo)
        novo["model"] = json.loads(bruto)["model"]
        assert json.dumps(novo, ensure_ascii=False).encode("utf-8") == bruto, sufixo


def conferir_metricas(m, ref_metricas):
    assert (m["macro_f1"], m["micro_f1"], m["weighted_f1"], m["mcc"]) == \
        (ref_metricas["test_macro_f1"], ref_metricas["test_micro_f1"], ref_metricas["test_weighted_f1"],
         ref_metricas["test_mcc"])
    assert m["f1_per_class"] == ref_metricas["test_f1_per_class"]
    assert m["classification_report"] == ref_metricas["sklearn_report"]
    assert m["confusion_matrix"] == ref_metricas["confusion_matrix"]


def conferir_restrito(pasta, treino, ref, config_ref_extra=()):
    """O que o restrito e a Pair-Aware têm em comum."""
    assert sem_duracao(treino["dev_history"]) == sem_duracao(ref["dev_history"])
    assert treino["n_params"] == ref["n_params"]
    assert treino["melhor_epoca"] == ref["instrumentation"]["best_epoch"]
    assert treino["n_candidatos"] == {k: ref["n_candidates"][k] for k in ("train", "dev")}
    metricas = ler_json(pasta / "metricas.json")
    conferir_metricas(metricas["test"], ref)
    esp = ref["restricted_space"]
    assert {p: treino["subconjunto"][p] for p in ("train", "dev")} == \
        {p: esp["por_split"][p] for p in ("train", "dev")}
    teste = ler_json(pasta / "predicoes_test.json")
    assert subconjunto.resumo(teste["restricted_indices"], teste["y_true"]) == esp["por_split"]["test"]
    assert {l: round(p, 6) for l, p in zip(LABELS, treino["pesos_classe"])} == esp["pesos_classe_efetivos"]
    registro = json.loads((pasta / "config.json").read_text("utf-8"))["config"]
    lex = registro["estrategia_config"]["lexico"]
    assert (lex["formas"], lex["lexico_sha1"], lex["min_freq"]) == \
        (esp["lexico"]["formas"], esp["lexico"]["sha1"], esp["lexico"]["min_freq"])
    config = registro["config"]
    for campo in ("max_gap", "ctx_chars", "max_length", "epochs", "batch_size", "lr", "weight_decay",
                  "warmup_ratio", "class_weight"):
        assert config[campo] == ref["config"][campo], campo
    if esp.get("test_restrito"):
        r = metricas["test_restrito"]
        assert (r["macro_f1"], r["f1_per_class"], r["confusion_matrix"]["matrix"]) == \
            (esp["test_restrito"]["macro_f1"], esp["test_restrito"]["f1_per_class"],
             esp["test_restrito"]["confusion_matrix"])
    return metricas


@EXATO
@pytest.mark.lento
def test_restrito_minusculo(tmp_path, lexico, uma_thread):
    """10 épocas no espaço restrito das partições completas (cerca de 2,5 min)."""
    config = restrito.ConfigRestrito(lr=1e-3, seed=42)
    treino = treinar_e_avaliar(tmp_path, "r", config, restrito.montagem(config, lexico=lexico))
    ref = ler_json(TREINO / "restrito_minusculo_seed42.json")
    conferir_restrito(tmp_path / "r", treino, ref)
    conferir_sidecars_byte_a_byte(tmp_path / "r", "restrito_minusculo_seed42")


@EXATO
def test_pair_aware_minusculo(tmp_path, lexico, uma_thread):
    """3 épocas no espaço restrito das partições completas (cerca de 1 min)."""
    config = pair_aware.ConfigPairAware(epochs=3, lr=1e-3, seed=42)
    treino = treinar_e_avaliar(tmp_path, "p", config, pair_aware.montagem(config, lexico=lexico))
    ref = ler_json(TREINO / "pairaware_minusculo_seed42.json")
    metricas = conferir_restrito(tmp_path / "p", treino, ref)
    conferir_sidecars_byte_a_byte(tmp_path / "p", "pairaware_minusculo_seed42")
    pa = ref["pair_aware"]
    final = ler_json(tmp_path / "p" / "treino.json")
    assert final["descricao"]["cabeca"] == pa["cabeca"]
    assert final["descricao"]["n_params_cabeca"] == pa["n_params_cabeca"]
    ausentes = {**final["descricao"]["marcadores_ausentes"], **final["descricao_test"]["marcadores_ausentes"]}
    assert ausentes == pa["marcadores_ausentes"]
    assert (metricas["dev"]["macro_f1"], metricas["dev"]["f1_per_class"],
            metricas["dev"]["confusion_matrix"]) == (ref["dev_best_macro_f1"], ref["dev_best_f1_per_class"],
                                                    ref["dev_best_confusion_matrix"])
    r = metricas["dev_restrito"]
    assert (r["macro_f1"], r["f1_per_class"], r["confusion_matrix"]["matrix"]) == \
        (pa["dev_restrito"]["macro_f1"], pa["dev_restrito"]["f1_per_class"], pa["dev_restrito"]["confusion_matrix"])


@EXATO
def test_baseline_pequeno_pela_estrategia(tmp_path, subconjunto, uma_thread):
    """O baseline do legado sobre as primeiras 30/10/10 linhas, 3 épocas, pela
    estratégia `baseline` (cerca de 1 min)."""
    pasta_particoes = subconjunto(30, 10, 10)
    config = Config(epochs=3, batch_size=64, max_length=128, lr=1e-3, seed=42)
    treino = treinar_e_avaliar(tmp_path, "b", config, baseline.montagem(config), pasta_particoes)
    ref = ler_json(TREINO / "baseline_pequeno_seed42.json")
    assert sem_duracao(treino["dev_history"]) == sem_duracao(ref["dev_history"])
    assert treino["n_params"] == ref["n_params"] and treino["melhor_epoca"] == ref["instrumentation"]["best_epoch"]
    conferir_metricas(ler_json(tmp_path / "b" / "metricas.json")["test"], ref)
    conferir_sidecars_byte_a_byte(tmp_path / "b", "baseline_pequeno_seed42")
