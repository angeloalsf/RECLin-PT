"""Métricas, significância, protocolo e agregação (`reclin.avaliacao`) sobre exemplos pequenos."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from reclin.avaliacao import agregacao, metricas, protocolo, significancia
from reclin.tarefa import ASSOC, LABELS, NEG, NOREL

# 8 exemplos: 3 negation_of, 2 associated_with, 3 no_relation
Y_TRUE = [NEG, NEG, NEG, ASSOC, ASSOC, NOREL, NOREL, NOREL]
Y_PRED = [NEG, NEG, NOREL, ASSOC, NEG, NOREL, NOREL, ASSOC]


def test_matriz_de_confusao():
    assert metricas.matriz_confusao(Y_TRUE, Y_PRED) == [[2, 0, 1], [1, 1, 0], [0, 1, 2]]
    with pytest.raises(ValueError):
        metricas.matriz_confusao([0, 1], [0])


def test_metricas_por_classe_pelas_contagens():
    m = metricas.avaliar(Y_TRUE, Y_PRED)
    neg = m["por_classe"]["negation_of"]
    assert (neg["tp"], neg["fp"], neg["fn"], neg["support"]) == (2, 1, 1, 3)
    assert neg["precision"] == 2 / 3 and neg["recall"] == 2 / 3
    assert neg["f1"] == 4 / 6                                  # 2TP / (2TP + FP + FN)
    assocc = m["por_classe"]["associated_with"]
    assert (assocc["precision"], assocc["recall"], assocc["f1"]) == (1 / 2, 1 / 2, 2 / 4)
    assert m["f1_per_class"] == {r: m["por_classe"][r]["f1"] for r in LABELS}


def test_medias_e_acuracia():
    m = metricas.avaliar(Y_TRUE, Y_PRED)
    f1 = [m["f1_per_class"][r] for r in LABELS]
    assert m["macro_f1"] == float(np.mean(f1))
    assert m["weighted_f1"] == float(np.average(f1, weights=[3, 2, 3]))
    assert m["micro_f1"] == m["classification_report"]["accuracy"] == 5 / 8
    rel = m["classification_report"]
    assert set(rel) == set(LABELS) | {"accuracy", "macro avg", "weighted avg"}
    assert rel["negation_of"]["support"] == 3.0 and rel["macro avg"]["support"] == 8.0


def test_divisao_por_zero_vale_zero():
    # nenhuma predição de associated_with e nenhum exemplo verdadeiro dela
    m = metricas.avaliar([NEG, NOREL], [NEG, NOREL])
    assocc = m["por_classe"]["associated_with"]
    assert (assocc["precision"], assocc["recall"], assocc["f1"]) == (0.0, 0.0, 0.0)
    assert m["f1_per_class"]["negation_of"] == 1.0


def test_mcc():
    assert metricas.avaliar([0, 1, 2, 2], [0, 1, 2, 2])["mcc"] == 1.0
    assert metricas.avaliar([0, 1, 2, 2], [2, 2, 2, 2])["mcc"] == 0.0     # sempre a mesma classe
    # 2 classes: coincide com a fórmula binária (TP·TN − FP·FN) / sqrt(...)
    tp, tn, fp, fn = 3, 4, 1, 2
    y_true = [0] * tp + [2] * tn + [2] * fp + [0] * fn
    y_pred = [0] * tp + [2] * tn + [0] * fp + [2] * fn
    binario = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert metricas.avaliar(y_true, y_pred)["mcc"] == pytest.approx(binario, rel=1e-15)


def test_resumo_alvo():
    r = metricas.resumo_alvo(Y_TRUE, Y_PRED)
    assert set(r) == {"tp", "fp", "fn", "precision", "recall", "f1", "macro_f1"}
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)


# --------------------------------------------------------------------------- #
def _pred(y_true, y_pred, model="m", seed=1):
    return {"model": model, "seed": seed, "labels": list(LABELS), "y_true": y_true, "y_pred": y_pred}


def test_f1_classe_vetorizado_igual_ao_das_metricas():
    f1 = significancia.f1_classe(np.array(Y_TRUE), np.array(Y_PRED), NEG)
    assert f1 == metricas.avaliar(Y_TRUE, Y_PRED)["f1_per_class"]["negation_of"]


def test_mcnemar():
    a = np.array([True, True, False, True])
    b = np.array([True, False, False, False])
    r = significancia.mcnemar_exato(a, b)
    assert (r["b_only_a_correct"], r["c_only_b_correct"], r["n_discordant"]) == (2, 0, 2)
    assert r["p_value"] == 0.5                                 # 2 · (1/2)²
    sem = significancia.mcnemar_exato(a, a)
    assert sem["p_value"] == 1.0 and sem["note"] == "sem discordancias"


def test_bootstrap_deterministico_pela_semente():
    yt, ya, yb = np.array(Y_TRUE * 5), np.array(Y_PRED * 5), np.array(Y_TRUE * 5)
    r1 = significancia.bootstrap_pareado(yt, ya, yb, NEG, n_boot=200, seed=7)
    r2 = significancia.bootstrap_pareado(yt, ya, yb, NEG, n_boot=200, seed=7)
    r3 = significancia.bootstrap_pareado(yt, ya, yb, NEG, n_boot=200, seed=8)
    assert r1 == r2 and r1 != r3
    assert r1["observed_diff"] == pytest.approx(4 / 6 - 1.0)
    assert r1["ci95_low"] <= r1["mean_diff"] <= r1["ci95_high"] and r1["n_boot"] == 200
    assert 0.0 <= r1["p_value"] <= 1.0


def test_comparar_exige_o_mesmo_conjunto():
    a = _pred(Y_TRUE, Y_PRED)
    with pytest.raises(ValueError, match="mesmo conjunto"):
        significancia.comparar(a, _pred(Y_TRUE[:-1] + [NEG], Y_PRED), seed=1, n_boot=10)
    with pytest.raises(ValueError, match="rótulos"):
        significancia.comparar(a, dict(a, labels=list(reversed(LABELS))), seed=1, n_boot=10)


def test_relatorio_no_formato_do_legado():
    r = significancia.comparar(_pred(Y_TRUE, Y_PRED, "A"), _pred(Y_TRUE, Y_TRUE, "B"), seed=3, n_boot=50)
    assert set(r) == {"model_a", "model_b", "n_test", "target_class", "accuracy", "target_f1",
                      "mcnemar", "paired_bootstrap"}
    texto = significancia.relatorio_json(r)
    assert not texto.endswith("\n") and texto.startswith('{\n  "accuracy"')
    assert json.loads(texto) == r
    assert significancia.significativo({"paired_bootstrap": {"ci95_low": 0.1, "ci95_high": 0.2}})
    assert not significancia.significativo({"paired_bootstrap": {"ci95_low": -0.1, "ci95_high": 0.2}})


def test_gravar_relatorio_sem_quebra_final(tmp_path):
    r = significancia.comparar(_pred(Y_TRUE, Y_PRED), _pred(Y_TRUE, Y_TRUE), seed=3, n_boot=20)
    destino = tmp_path / "s.json"
    significancia.gravar_relatorio(destino, r)
    assert destino.read_bytes() == significancia.relatorio_json(r).encode("utf-8")


# --------------------------------------------------------------------------- #
def test_regra_de_semente():
    assert protocolo.semente_bootstrap(42, 43) == 42
    assert protocolo.semente_bootstrap(None, 43) == 43
    with pytest.raises(ValueError):
        protocolo.semente_bootstrap(None, None)


def test_protocolo_do_tcc():
    comparacoes = protocolo.comparacoes_tcc()
    assert len(comparacoes) == 26 == len({c.arquivo for c in comparacoes})
    fase1 = comparacoes[:2]
    assert [c.arquivo for c in fase1] == ["significance_biobertpt_vs_bertimbau_seed42.json",
                                          "significance_biobertpt_vs_bertimbau_seed43.json"]
    regra_vs_baseline = [c for c in comparacoes if c.a == "regra_pura"]
    assert len(regra_vs_baseline) == 4 and all(c.seed == c.seed_b for c in regra_vs_baseline)
    assert sum(1 for c in comparacoes if c.b == "regra_pura") == 4
    assert sum(1 for c in comparacoes if c.a.startswith("filtro_") and c.b.startswith("baseline_")) == 16


# --------------------------------------------------------------------------- #
def test_agregar_media_e_desvio_populacional():
    r = agregacao.agregar({42: 0.6, 43: 0.8})
    assert r["mean"] == pytest.approx(0.7) and r["pstdev"] == pytest.approx(0.1)
    assert r["n"] == 2 and r["by_seed"] == {"42": 0.6, "43": 0.8}
    with pytest.raises(ValueError):
        agregacao.agregar({42: 0.6})


def test_metricas_agregadas():
    por_semente = {s: metricas.avaliar(Y_TRUE, Y_PRED if s == 42 else Y_TRUE) for s in (42, 43)}
    r = agregacao.metricas_agregadas(por_semente)
    assert set(r) == {"macro_f1"} | {f"f1_{rotulo}" for rotulo in LABELS}
    assert r["f1_negation_of"]["by_seed"] == {"42": 4 / 6, "43": 1.0}
