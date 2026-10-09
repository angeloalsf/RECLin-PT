"""Portão da etapa 2 (1/2): as métricas recalculadas dos sidecars do legado são
as que o legado registrou.

Fonte: `referencia/resultados_legado/` (cópia sem alteração de `results/` do
legado, commit a5f055c). Para cada sidecar, `reclin.avaliacao.metricas`
recalcula as métricas a partir de `y_true` e `y_pred` e compara com o registro
do legado. A comparação é de IGUALDADE EXATA de floats, sem tolerância, exceto
onde indicado com a justificativa:

* o F1 do `FASE2_test_summary.json`, que o legado calculou como 2·P·R/(P+R):
  confere-se que essa fórmula, aplicada à precisão e ao recall recalculados,
  dá exatamente o valor gravado (a divergência é da fórmula, não das
  predições);
* as trilhas `test_evals`, que o legado gravou arredondadas a 6 casas:
  compara-se com o valor recalculado arredondado a 6 casas.
"""
from __future__ import annotations

import pytest

from reclin.avaliacao import agregacao, metricas
from reclin.execucao import predicoes as sidecar
from reclin.execucao import trilha as modulo_trilha
from reclin.tarefa import LABELS, NEG
from reclin.util.caminhos import CODIGO
from reclin.util.io import ler_json

RESULTADOS_LEGADO = CODIGO / "testes" / "referencia" / "resultados_legado"

BASELINES = [f"baseline_{e}_seed{s}" for e in ("biobertpt", "bertimbau") for s in (42, 43)]
RESTRITOS = [f"finetuning_restrito/restrito_biobertpt_seed{s}" for s in range(42, 47)]
PAIR_AWARE = "pair_aware/pairaware_biobertpt_seed42"
TREINADAS = BASELINES + RESTRITOS                         # TEST avaliado
COM_DEV = TREINADAS + [PAIR_AWARE]                        # DEV da melhor época
FASE2 = [f"filtro_{e}_seed{s}" for e in ("biobertpt", "bertimbau") for s in (42, 43)] + ["regra_pura"]

SIDECARS = ([(n + ".preds.json", "test") for n in TREINADAS + FASE2]
            + [(n + ".dev_preds.json", "dev") for n in COM_DEV])


def avaliar(pred: dict) -> dict:
    return metricas.avaliar(pred["y_true"], pred["y_pred"])


def subconjunto(pred: dict) -> dict:
    """O sidecar restrito ao subespaço da execução (`restricted_indices`)."""
    idx = pred["restricted_indices"]
    return {"y_true": [pred["y_true"][i] for i in idx], "y_pred": [pred["y_pred"][i] for i in idx]}


def melhor_epoca(historico: list[dict]) -> int:
    """Critério do laço de treino do legado: a primeira época com o maior
    macro-F1 no DEV (`relation_extraction.best_epoch_from_history`)."""
    melhor, epoca = -1.0, None
    for e in historico:
        if e["dev_macro_f1"] > melhor:
            melhor, epoca = e["dev_macro_f1"], e["epoch"]
    return epoca


def test_lista_cobre_todos_os_sidecars_versionados():
    versionados = sorted(p.relative_to(RESULTADOS_LEGADO).as_posix()
                         for p in RESULTADOS_LEGADO.rglob("*preds.json"))
    assert versionados == sorted(nome for nome, _ in SIDECARS)


@pytest.mark.parametrize("nome,particao", SIDECARS)
def test_sidecar_pertence_ao_conjunto_de_referencia(nome, particao, legado, referencia_dados):
    pred = legado(nome)
    sidecar.validar(pred)
    ref = referencia_dados["particoes"][particao]
    conjunto = {"particao": particao, "n_candidatos": ref["n_candidatos"],
                "y_true_sha256": ref["y_true_sha256"]}
    assert sidecar.conferir_conjunto(pred, conjunto) == []


@pytest.mark.parametrize("nome", TREINADAS)
def test_metricas_do_test_iguais_ao_json(nome, legado):
    registro = legado(nome + ".json")
    m = avaliar(legado(nome + ".preds.json"))
    assert m["macro_f1"] == registro["test_macro_f1"]
    assert m["micro_f1"] == registro["test_micro_f1"]
    assert m["weighted_f1"] == registro["test_weighted_f1"]
    assert m["mcc"] == registro["test_mcc"]
    assert m["f1_per_class"] == registro["test_f1_per_class"]
    assert m["classification_report"] == registro["sklearn_report"]
    assert m["confusion_matrix"] == registro["confusion_matrix"]
    assert registro["n_candidates"]["test"] == m["n"]
    assert registro["n_candidates"]["dev"] == len(legado(nome + ".dev_preds.json")["y_true"])


@pytest.mark.parametrize("nome", COM_DEV)
def test_metricas_do_dev_iguais_ao_historico(nome, legado):
    registro, pred = legado(nome + ".json"), legado(nome + ".dev_preds.json")
    epoca = melhor_epoca(registro["dev_history"])
    assert epoca == registro["instrumentation"]["best_epoch"] == pred["best_epoch"]
    historico = next(e for e in registro["dev_history"] if e["epoch"] == epoca)
    m = avaliar(pred)

    assert m["macro_f1"] == historico["dev_macro_f1"]
    assert m["f1_per_class"]["negation_of"] == historico["dev_negation_of_f1"]
    assert pred["dev_macro_f1_recomputed"] == m["macro_f1"]
    assert pred["best_dev_macro_f1_history"] == historico["dev_macro_f1"]
    if "restricted_indices" in pred:                      # restrito e Pair-Aware
        r = metricas.avaliar(**subconjunto(pred))
        assert r["macro_f1"] == historico["dev_restrito_macro_f1"]
        assert r["f1_per_class"]["negation_of"] == historico["dev_restrito_negation_of_f1"]
        assert pred["dev_negation_of_f1_recomputed"] == m["f1_per_class"]["negation_of"]
        assert registro["instrumentation"]["dev_macro_f1_recomputed"] == m["macro_f1"]
    if "dev_best_macro_f1" in registro:                   # Pair-Aware
        assert registro["dev_best_macro_f1"] == m["macro_f1"]
        assert registro["dev_best_f1_per_class"] == m["f1_per_class"]
        assert registro["dev_best_confusion_matrix"] == m["confusion_matrix"]


@pytest.mark.parametrize("nome", RESTRITOS)
def test_metricas_do_subespaco_restrito(nome, legado):
    registrado = legado(nome + ".json")["restricted_space"]["test_restrito"]
    pred = legado(nome + ".preds.json")
    r = metricas.avaliar(**subconjunto(pred))
    assert r["macro_f1"] == registrado["macro_f1"]
    assert r["f1_per_class"] == registrado["f1_per_class"]
    assert r["confusion_matrix"]["matrix"] == registrado["confusion_matrix"]


def test_pair_aware_nao_avaliou_o_test(legado):
    registro = legado(PAIR_AWARE + ".json")
    assert registro["test_avaliado"] is False and registro["test_macro_f1"] is None


def _sistemas_fase2(legado):
    return legado("FASE2_test_summary.json")["systems"]


@pytest.mark.parametrize("indice", range(9))
def test_resumo_da_fase2(indice, legado):
    """Contagens, precisão, recall e macro-F1 exatos; o F1 gravado é o da
    fórmula 2·P·R/(P+R) do legado aplicada à P e à R recalculadas."""
    sistema = _sistemas_fase2(legado)[indice]
    pred = legado(sistema["path"])
    r = metricas.resumo_alvo(pred["y_true"], pred["y_pred"], NEG)
    for campo in ("tp", "fp", "fn", "precision", "recall", "macro_f1"):
        assert r[campo] == sistema[campo], campo
    p, rec = r["precision"], r["recall"]
    assert sistema["f1"] == (2 * p * rec / (p + rec) if p + rec else 0.0)
    assert r["f1"] == 2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"])


def test_resumo_da_fase2_tem_os_nove_sistemas(legado):
    ordem = [f"{s}_{e}_seed{n}" for s in ("filtro",) for n in (42, 43) for e in ("biobertpt", "bertimbau")]
    ordem += ["regra_pura"] + [f"baseline_{e}_seed{n}" for n in (42, 43) for e in ("biobertpt", "bertimbau")]
    assert [s["path"] for s in _sistemas_fase2(legado)] == [n + ".preds.json" for n in ordem]
    assert legado("FASE2_test_summary.json")["n_test"] == 19210


TRILHAS = [n + ".test_evals.jsonl" for n in BASELINES + RESTRITOS + ["regra_pura"]]


def _sidecar_da_linha(trilha: str, linha: dict) -> str:
    """Qual sidecar cada linha de trilha avaliou."""
    base = trilha.removesuffix(".test_evals.jsonl")
    if linha["model"].startswith("filtro("):
        return base.replace("baseline_", "filtro_") + ".preds.json"
    return base + ".preds.json"


@pytest.mark.parametrize("trilha", TRILHAS)
def test_trilhas_test_evals(trilha):
    linhas = modulo_trilha.ler(RESULTADOS_LEGADO / trilha)
    assert linhas
    for linha in linhas:
        pred = ler_json(RESULTADOS_LEGADO / _sidecar_da_linha(trilha, linha))
        m = avaliar(pred)
        assert linha["n_test"] == m["n"]
        assert linha["test_macro_f1"] == round(m["macro_f1"], 6)
        assert linha["test_negation_of_f1"] == round(m["f1_per_class"]["negation_of"], 6)
        if "config" in linha:                     # linhas da fase 2 trazem a configuração
            assert modulo_trilha.sha1_config(linha["config"]) == linha["config_sha1"]


def test_agregacao_igual_ao_summary_by_seed(legado):
    resumo = legado("summary_by_seed.json")
    assert resumo["seeds"] == [42, 43]
    for encoder in ("biobertpt", "bertimbau"):
        por_semente = {s: avaliar(legado(f"baseline_{encoder}_seed{s}.preds.json")) for s in (42, 43)}
        registrado = resumo["models"][encoder]
        assert agregacao.metricas_agregadas(por_semente) == registrado["metrics"]
        assert registrado["model"] == legado(f"baseline_{encoder}_seed42.json")["model"]
    assert set(registrado["metrics"]) == {"macro_f1"} | {f"f1_{r}" for r in LABELS}
