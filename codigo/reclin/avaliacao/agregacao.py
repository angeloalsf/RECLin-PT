"""Agregação de uma métrica entre sementes.

Para cada métrica: média e desvio-padrão POPULACIONAL (`statistics.pstdev`),
o número de sementes e os valores por semente que entraram na conta.

Por que `pstdev` e não `stdev`: é o que o legado usou nos números já
discutidos, e com duas sementes a escolha não é neutra (`stdev` dá exatamente
√2 vezes `pstdev`). Com duas sementes, nenhum dos dois estima a variância de
inicialização com credibilidade: o desvio é dispersão observada, não intervalo
de confiança.

Origem no legado: `scripts/aggregate_seeds.py` (`results/summary_by_seed.json`).
Os rótulos de apresentação dos modelos ("BioBERTpt (clínico)") ficam para os
relatórios.
"""
from __future__ import annotations

import statistics
from typing import Any, Mapping

from reclin.tarefa import LABELS


def agregar(por_semente: Mapping[int, float]) -> dict[str, Any]:
    """Média, desvio populacional e valores de uma métrica, na ordem das
    sementes recebidas."""
    if len(por_semente) < 2:
        raise ValueError("a agregação precisa de ao menos duas sementes")
    valores = list(por_semente.values())
    return {"mean": statistics.mean(valores), "pstdev": statistics.pstdev(valores),
            "n": len(valores), "by_seed": {str(s): v for s, v in por_semente.items()}}


def metricas_agregadas(metricas_por_semente: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Agrega o macro-F1 e o F1 de cada classe de um sistema, a partir das
    métricas de `metricas.avaliar` de cada semente."""
    saida = {"macro_f1": agregar({s: m["macro_f1"] for s, m in metricas_por_semente.items()})}
    for rotulo in LABELS:
        saida[f"f1_{rotulo}"] = agregar(
            {s: m["f1_per_class"][rotulo] for s, m in metricas_por_semente.items()})
    return saida
