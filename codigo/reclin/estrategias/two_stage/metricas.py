"""Métricas binárias de cada estágio do Two-Stage.

As métricas da saída final (3 rótulos) são as de `avaliacao.metricas`; estas
servem só aos estágios, para escolher o checkpoint de cada um e para os
diagnósticos rotulados por estágio (nunca apresentados como resultado da
arquitetura).

* `ap(y_bin, escores)` — precisão média (average precision) da classe
  positiva, como em `sklearn.metrics.average_precision_score`: os escores
  distintos, em ordem decrescente, são os limiares; AP = Σ (R_n − R_{n−1})·P_n.
  Escores empatados entram juntos no mesmo limiar. Sem positivos, 0.
* `binarias(y_true, y_pred, positivo)` — contagens, precisão, recall e F1 da
  classe `positivo` e o macro-F1 das duas classes (as fórmulas de
  `avaliacao.metricas`).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from reclin.avaliacao import metricas


def ap(y_bin: Sequence[int], escores: Sequence[float]) -> float:
    """Average precision da classe positiva (`y_bin == 1`)."""
    y = np.asarray(y_bin, dtype=np.int64)
    s = np.asarray(escores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"{y.size} rótulos e {s.size} escores")
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    ordem = np.argsort(-s, kind="mergesort")
    y, s = y[ordem], s[ordem]
    # último índice de cada grupo de escores iguais (os limiares)
    fim_grupo = np.r_[np.nonzero(np.diff(s))[0], y.size - 1]
    tp = np.cumsum(y)[fim_grupo]
    preditos = fim_grupo + 1
    precisao = tp / preditos
    recall = tp / n_pos
    recall_anterior = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_anterior) * precisao))


def binarias(y_true: Sequence[int], y_pred: Sequence[int], positivo: int = 0) -> dict[str, Any]:
    """Métricas de um estágio (2 classes, ids 0 e 1)."""
    matriz = metricas.matriz_confusao(y_true, y_pred, n_rotulos=2)
    por_classe = [metricas.metricas_classe(matriz, c) for c in (0, 1)]
    alvo = por_classe[positivo]
    return {"n": len(y_true), "tp": alvo["tp"], "fp": alvo["fp"], "fn": alvo["fn"],
            "precision": alvo["precision"], "recall": alvo["recall"], "f1": alvo["f1"],
            "macro_f1": float(np.mean([c["f1"] for c in por_classe])), "matriz": matriz}
