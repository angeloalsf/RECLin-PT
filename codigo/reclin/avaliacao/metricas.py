"""Métricas de classificação sobre os três rótulos da tarefa.

Tudo é calculado a partir da matriz de confusão, com as fórmulas por
contagens:

* precisão = TP / (TP + FP)
* recall   = TP / (TP + FN)
* F1       = 2·TP / (2·TP + FP + FN)

valendo 0 quando o denominador é zero (o `zero_division=0` do sklearn). As
médias seguem o sklearn: macro é a média simples entre as classes, weighted é
ponderada pelo suporte (número de exemplos verdadeiros da classe) e micro, com
os três rótulos, é a acurácia. O MCC é o multiclasse (Gorodkin), calculado
sobre a matriz de confusão.

As operações em ponto flutuante são as mesmas do scikit-learn 1.6.1, que
produziu os JSONs dos treinos do legado: os testes de equivalência
(`testes/equivalencia/test_metricas_legado.py`) conferem igualdade exata com
eles, campo a campo. Em particular, as médias são calculadas com o numpy
(`np.mean`, `np.average`), como no sklearn, e não com o `sum()` do Python: a
partir do Python 3.12, `sum()` de floats usa soma compensada e pode diferir da
soma sequencial na última casa decimal (aconteceu no macro-F1 do baseline
BioBERTpt semente 43: 0.7127856751095188 contra 0.7127856751095191).

Origem no legado: dez implementações. O sklearn em
`relation_extraction.run` e `train_restrito.metricas`; fórmulas manuais em
`significance.f1_for_class`, `run_fase2_test.f1_class`,
`make_rule_baseline.score`, `criterio_parada.prf`, `_artifacts.class_metrics`
e outras. Uma delas (`make_rule_baseline.score`) calculava o F1 como
2·P·R / (P + R), que difere de 2·TP / (2·TP + FP + FN) na última casa decimal;
é a fórmula do `FASE2_test_summary.json` (ver a nota da etapa 2).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from reclin.tarefa import LABELS, NEG


def matriz_confusao(y_true: Sequence[int], y_pred: Sequence[int],
                    n_rotulos: int = len(LABELS)) -> list[list[int]]:
    """Linhas = rótulo verdadeiro, colunas = rótulo predito, na ordem dos ids."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true tem {len(y_true)} elementos e y_pred {len(y_pred)}")
    matriz = [[0] * n_rotulos for _ in range(n_rotulos)]
    for t, p in zip(y_true, y_pred):
        matriz[t][p] += 1
    return matriz


def _dividir(numerador: float, denominador: float) -> float:
    return numerador / denominador if denominador else 0.0


def metricas_classe(matriz: list[list[int]], classe: int) -> dict[str, Any]:
    """Contagens, precisão, recall e F1 de uma classe."""
    tp = matriz[classe][classe]
    suporte = sum(matriz[classe])                       # verdadeiros da classe
    preditos = sum(linha[classe] for linha in matriz)  # preditos como a classe
    return {
        "tp": tp, "fp": preditos - tp, "fn": suporte - tp, "support": suporte,
        "precision": _dividir(tp, preditos),
        "recall": _dividir(tp, suporte),
        "f1": _dividir(2.0 * tp, 1.0 * suporte + preditos),
    }


def mcc(matriz: list[list[int]]) -> float:
    """Coeficiente de correlação de Matthews multiclasse."""
    n_rotulos = len(matriz)
    verdadeiros = [float(sum(matriz[i])) for i in range(n_rotulos)]
    preditos = [float(sum(matriz[i][j] for i in range(n_rotulos))) for j in range(n_rotulos)]
    corretos = float(sum(matriz[i][i] for i in range(n_rotulos)))
    n = sum(preditos)
    cov_ytyp = corretos * n - sum(t * p for t, p in zip(verdadeiros, preditos))
    cov_ypyp = n ** 2 - sum(p * p for p in preditos)
    cov_ytyt = n ** 2 - sum(t * t for t in verdadeiros)
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    return cov_ytyp / math.sqrt(cov_ytyt * cov_ypyp)


def avaliar(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Todas as métricas de um par (y_true, y_pred) de ids de rótulos.

    As chaves de `classification_report` reproduzem o `output_dict` do
    `sklearn.metrics.classification_report`, que o legado gravava em
    `sklearn_report`.
    """
    matriz = matriz_confusao(y_true, y_pred)
    por_classe = {rotulo: metricas_classe(matriz, i) for i, rotulo in enumerate(LABELS)}
    n = len(y_true)
    acertos = sum(matriz[i][i] for i in range(len(LABELS)))

    suportes = np.array([por_classe[r]["support"] for r in LABELS])

    def media(campo: str) -> float:
        return float(np.mean(np.array([por_classe[r][campo] for r in LABELS])))

    def ponderada(campo: str) -> float:
        return float(np.average(np.array([por_classe[r][campo] for r in LABELS]),
                                weights=suportes))

    relatorio: dict[str, Any] = {
        r: {"precision": m["precision"], "recall": m["recall"], "f1-score": m["f1"],
            "support": float(m["support"])}
        for r, m in por_classe.items()}
    relatorio["accuracy"] = acertos / n
    relatorio["macro avg"] = {"precision": media("precision"), "recall": media("recall"),
                              "f1-score": media("f1"), "support": float(n)}
    relatorio["weighted avg"] = {"precision": ponderada("precision"),
                                 "recall": ponderada("recall"),
                                 "f1-score": ponderada("f1"), "support": float(n)}
    return {
        "n": n,
        "macro_f1": media("f1"),
        "micro_f1": (2.0 * acertos) / (2.0 * n),
        "weighted_f1": ponderada("f1"),
        "mcc": mcc(matriz),
        "f1_per_class": {r: por_classe[r]["f1"] for r in LABELS},
        "por_classe": por_classe,
        "classification_report": relatorio,
        "confusion_matrix": {"labels": list(LABELS), "matrix": matriz},
    }


def resumo_alvo(y_true: Sequence[int], y_pred: Sequence[int], classe: int = NEG) -> dict[str, Any]:
    """Contagens, P, R e F1 da classe-alvo e o macro-F1: o resumo que as
    tabelas da fase 2 usam."""
    m = avaliar(y_true, y_pred)
    alvo = m["por_classe"][LABELS[classe]]
    return {k: alvo[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")} | {
        "macro_f1": m["macro_f1"]}
