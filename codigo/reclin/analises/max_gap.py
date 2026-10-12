"""Sensibilidade ao limite de distância entre as entidades (`max_gap`).

Para cada limite da lista: quantos pares candidatos são gerados (por rótulo) e
que fração das relações anotadas de cada tipo vira candidato. A enumeração dos
pares é a da tarefa (`tarefa.iter_candidate_pairs`), sem nenhuma lógica
paralela: o limite só muda o parâmetro `max_gap`.

O `max_gap` da tarefa (25) já está fixado e congelado nas partições e nos
conjuntos de referência; esta análise só o descreve, não o escolhe. Por isso
usa TRAIN e DEV por padrão (o script só inclui o TEST quando pedido).

Origem no legado (só a finalidade): `analyze_max_gap.py`.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from reclin.tarefa import LABELS, MAX_GAP, Documento, iter_candidate_pairs

LIMITES: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200)


def sensibilidade(documentos: Iterable[Documento], limites: Sequence[int] = LIMITES) -> dict[str, Any]:
    docs = list(documentos)
    limites = sorted(set(int(x) for x in limites))
    if not limites or limites[0] < 0:
        raise ValueError("os limites precisam ser inteiros não negativos")
    total_relacoes: Counter[str] = Counter()
    for doc in docs:
        total_relacoes.update(r["type"] for r in doc["relations"])
    linhas = []
    for limite in limites:
        por_rotulo: Counter[str] = Counter()
        for doc in docs:
            for c in iter_candidate_pairs(doc, max_gap=limite):
                por_rotulo[c["label"]] += 1
        cobertura = {t: {"cobertas": por_rotulo.get(t, 0), "total": total_relacoes[t],
                         "fracao": por_rotulo.get(t, 0) / total_relacoes[t] if total_relacoes[t] else None}
                     for t in sorted(total_relacoes)}
        relacoes = sum(por_rotulo.get(r, 0) for r in LABELS if r != "no_relation")
        linhas.append({"max_gap": limite, "n_candidatos": sum(por_rotulo.values()),
                       "candidatos_por_rotulo": {r: por_rotulo.get(r, 0) for r in LABELS},
                       "cobertura": cobertura,
                       "no_relation_por_relacao": (por_rotulo.get("no_relation", 0) / relacoes
                                                   if relacoes else None),
                       "da_tarefa": limite == MAX_GAP})
    return {"n_documentos": len(docs), "relacoes_anotadas": dict(sorted(total_relacoes.items())),
            "max_gap_da_tarefa": MAX_GAP, "linhas": linhas}
