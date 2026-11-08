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


def distancias_anotadas(documentos: Iterable[Documento],
                        limites: Sequence[int] = (20, 25, 50)) -> dict[str, Any]:
    """Distribuição de `entity_gap` nas relações anotadas, por tipo.

    `entity_gap` é simétrico: caracteres entre o fim do primeiro span **no
    texto** e o início do segundo, qualquer que seja o e1; 0 quando os spans se
    tocam ou se sobrepõem. Para cada limite, conta as relações com distância
    acima dele (as que nunca viram candidato com aquele `max_gap`).
    """
    from reclin.tarefa import entity_gap
    por_tipo: dict[str, list[int]] = {}
    for doc in documentos:
        m = {e["id"]: e for e in doc["entities"]}
        for r in doc["relations"]:
            por_tipo.setdefault(r["type"], []).append(entity_gap(m[r["e1_id"]], m[r["e2_id"]]))
    out: dict[str, Any] = {}
    for tipo, v in sorted(por_tipo.items()):
        v.sort()
        n = len(v)

        def q(p: float) -> int:
            return v[min(n - 1, int(p * n))]

        out[tipo] = {"n": n, "mediana": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99),
                     "max": v[-1], "ate_1": sum(1 for g in v if g <= 1),
                     "acima_de": {str(lim): sum(1 for g in v if g > lim) for lim in limites}}
    total = sum(d["n"] for d in out.values())
    out["todas"] = {"n": total, "acima_de": {str(lim): sum(d["acima_de"][str(lim)] for d in out.values())
                                             for lim in limites}}
    out["todas"]["fracao_acima_de"] = {k: (x / total if total else None)
                                       for k, x in out["todas"]["acima_de"].items()}
    return out
