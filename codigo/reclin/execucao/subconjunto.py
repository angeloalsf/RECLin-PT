"""Predições sobre um subconjunto dos candidatos, levadas de volta ao conjunto completo.

Uma estratégia pode treinar, escolher a época e prever só sobre parte dos
candidatos de uma partição (o fine-tuning restrito e a Pair-Aware usam os
pares cujo e1 é pista). Para que as predições sejam comparáveis com as de
qualquer outra execução, elas voltam ao conjunto completo, na mesma ordem e
com o mesmo `y_true`:

* `y_true` nunca é remapeado: é o do conjunto completo. As relações fora do
  subconjunto continuam lá e viram falso negativo (o teto de recall);
* fora do subconjunto, `y_pred` é `no_relation` e `probs` é `[0, 0, 1]` (a
  decisão é da seleção, não do modelo); dentro, vale a predição do modelo.

`indices` são as posições do subconjunto no conjunto completo, em ordem
crescente e sem repetição — a ordem em que os exemplos foram montados.

Origem no legado: `remapear`, `PROBS_FORA` e `EspacoRestrito.resumo` de
`src/finetuning_restrito/restricted_space.py`. O critério de seleção (e1 no
léxico) é das estratégias, não deste módulo.
"""
from __future__ import annotations

from typing import Any, Sequence

from reclin.tarefa import LABELS, NOREL

PROBS_FORA = (0.0, 0.0, 1.0)


def conferir_indices(indices: Sequence[int], n_completo: int) -> None:
    """Falha se os índices não forem posições crescentes e distintas do conjunto."""
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("índices do subconjunto fora de ordem ou repetidos")
    if indices and (indices[0] < 0 or indices[-1] >= n_completo):
        raise ValueError(f"índices do subconjunto fora de 0..{n_completo - 1}")


def remapear(indices: Sequence[int], n_completo: int, y_pred: Sequence[int],
             probs: Sequence[Sequence[float]] | None = None
             ) -> tuple[list[int], list[list[float]] | None]:
    """Predições (e probabilidades) do subconjunto no conjunto completo."""
    conferir_indices(indices, n_completo)
    if len(y_pred) != len(indices):
        raise ValueError(f"{len(y_pred)} predições para {len(indices)} posições do subconjunto")
    if probs is not None and len(probs) != len(indices):
        raise ValueError(f"{len(probs)} linhas de probs para {len(indices)} posições do subconjunto")
    completo = [NOREL] * n_completo
    for i, y in zip(indices, y_pred):
        completo[i] = int(y)
    probs_completo = None
    if probs is not None:
        probs_completo = [list(PROBS_FORA) for _ in range(n_completo)]
        for i, p in zip(indices, probs):
            probs_completo[i] = [float(x) for x in p]
    return completo, probs_completo


def resumo(indices: Sequence[int], y_true: Sequence[int]) -> dict[str, Any]:
    """Tamanho, contagens por rótulo e teto de recall do subconjunto."""
    conferir_indices(indices, len(y_true))
    completo = dict.fromkeys(LABELS, 0)
    sub = dict.fromkeys(LABELS, 0)
    for y in y_true:
        completo[LABELS[y]] += 1
    for i in indices:
        sub[LABELS[y_true[i]]] += 1

    def teto(rotulo: str) -> float | None:
        return sub[rotulo] / completo[rotulo] if completo[rotulo] else None

    n, k = len(y_true), len(indices)
    return {
        "n_completo": n,
        "n_restrito": k,
        "fracao": round(k / n, 6) if n else None,
        "contagens_completo": completo,
        "contagens_restrito": sub,
        "teto_recall_negation_of": teto("negation_of"),
        "teto_recall_associated_with": teto("associated_with"),
        "razao_no_relation_por_negation_of": (round(sub["no_relation"] / sub["negation_of"], 4)
                                              if sub["negation_of"] else None),
    }
