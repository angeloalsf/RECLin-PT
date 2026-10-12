"""Estatísticas descritivas das partições congeladas.

Por partição: documentos, tamanho dos textos, entidades (e por tipo),
relações anotadas por tipo, pares candidatos por rótulo (com o `max_gap` da
tarefa), cobertura das relações anotadas pelos candidatos e a distribuição da
distância (`tarefa.entity_gap`) entre as entidades das relações anotadas.

Cobertura de um tipo = pares candidatos com aquele rótulo ÷ relações anotadas
daquele tipo (a mesma definição de `analises.max_gap`). Ela fica abaixo de 1
por três motivos, contados à parte: a distância acima de `max_gap`, entidades
com o mesmo span (o par não é candidato) e o mesmo par (e1, e2) com mais de
uma relação anotada (o candidato leva um rótulo só, o da última anotação).
Cada relação conta em um motivo só, na ordem acima.

Só descreve os dados: não depende de nenhum modelo nem de resultado algum. O
TEST entra porque o TCC descreve o corpus inteiro; os números dele são
contagens de anotação (as relações por tipo já estão no MANIFEST), não
avaliação.

Origem no legado (só a finalidade): as estatísticas de `make_tcc_eda.py`.
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from reclin.tarefa import LABELS, MAX_GAP, Documento, entity_gap, iter_candidate_pairs

# limites superiores (inclusivos) das faixas de distância entre entidades
FAIXAS_GAP: tuple[int, ...] = (0, 5, 10, 25, 50, 100)


def faixa(gap: int, faixas: Sequence[int] = FAIXAS_GAP) -> str:
    """Rótulo da faixa de `gap`: '0', '1-5', ..., '>100'."""
    anterior = -1
    for limite in faixas:
        if gap <= limite:
            return str(limite) if anterior + 1 == limite else f"{anterior + 1}-{limite}"
        anterior = limite
    return f">{faixas[-1]}"


def rotulos_faixas(faixas: Sequence[int] = FAIXAS_GAP) -> list[str]:
    rotulos, anterior = [], -1
    for limite in faixas:
        rotulos.append(str(limite) if anterior + 1 == limite else f"{anterior + 1}-{limite}")
        anterior = limite
    return rotulos + [f">{faixas[-1]}"]


def _resumo(valores: list[int]) -> dict[str, Any]:
    if not valores:
        return {"n": 0, "media": None, "mediana": None, "min": None, "max": None}
    return {"n": len(valores), "media": statistics.mean(valores), "mediana": statistics.median(valores),
            "min": min(valores), "max": max(valores)}


def estatisticas_particao(documentos: Iterable[Documento], max_gap: int = MAX_GAP) -> dict[str, Any]:
    docs = list(documentos)
    tipos_entidade: Counter[str] = Counter()
    relacoes: Counter[str] = Counter()
    candidatos: Counter[str] = Counter()
    motivos: Counter[str] = Counter()
    gaps_relacoes: dict[str, Counter[str]] = {}
    gaps_valores: dict[str, list[int]] = {}
    for doc in docs:
        entidades = {e["id"]: e for e in doc["entities"]}
        tipos_entidade.update(e["type"] for e in doc["entities"])
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            candidatos[c["label"]] += 1
        por_par = Counter((r["e1_id"], r["e2_id"]) for r in doc["relations"])
        for r in doc["relations"]:
            relacoes[r["type"]] += 1
            e1, e2 = entidades.get(r["e1_id"]), entidades.get(r["e2_id"])
            if e1 is None or e2 is None:
                motivos["entidade_inexistente"] += 1
                continue
            gap = entity_gap(e1, e2)
            gaps_valores.setdefault(r["type"], []).append(gap)
            gaps_relacoes.setdefault(r["type"], Counter())[faixa(gap)] += 1
            if gap > max_gap:
                motivos["acima_de_max_gap"] += 1
            elif (e1["start"], e1["end"]) == (e2["start"], e2["end"]):
                motivos["mesmo_span"] += 1
            elif por_par[(r["e1_id"], r["e2_id"])] > 1:
                motivos["par_com_mais_de_uma_relacao"] += 1
    tamanhos = [len(d["text"]) for d in docs]
    tipos_rel = sorted(relacoes)
    return {
        "n_documentos": len(docs),
        "caracteres": _resumo(tamanhos),
        "n_entidades": sum(tipos_entidade.values()),
        "entidades_por_tipo": dict(sorted(tipos_entidade.items(), key=lambda kv: (-kv[1], kv[0]))),
        "relacoes_por_tipo": {t: relacoes[t] for t in tipos_rel},
        "candidatos_por_rotulo": {r: candidatos.get(r, 0) for r in LABELS},
        "n_candidatos": sum(candidatos.values()),
        "max_gap": max_gap,
        "cobertura_relacoes": {t: {"cobertas": candidatos.get(t, 0), "total": relacoes[t],
                                   "fracao": candidatos.get(t, 0) / relacoes[t] if relacoes[t] else None}
                               for t in tipos_rel},
        "relacoes_fora_dos_candidatos": {m: motivos.get(m, 0) for m in
                                         ("acima_de_max_gap", "mesmo_span", "par_com_mais_de_uma_relacao",
                                          "entidade_inexistente")},
        "gap_relacoes": {t: {"faixas": {f: gaps_relacoes.get(t, Counter()).get(f, 0) for f in rotulos_faixas()},
                             "resumo": _resumo(gaps_valores.get(t, []))}
                         for t in tipos_rel},
    }


def estatisticas(por_particao: Mapping[str, Iterable[Documento]], max_gap: int = MAX_GAP) -> dict[str, Any]:
    """`estatisticas_particao` de cada partição e o total."""
    saida = {p: estatisticas_particao(docs, max_gap) for p, docs in por_particao.items()}
    saida["_total"] = {
        "n_documentos": sum(s["n_documentos"] for s in saida.values()),
        "n_entidades": sum(s["n_entidades"] for s in saida.values()),
        "n_candidatos": sum(s["n_candidatos"] for s in saida.values()),
        "relacoes_por_tipo": dict(sum((Counter(s["relacoes_por_tipo"]) for s in saida.values()), Counter())),
        "candidatos_por_rotulo": {r: sum(s["candidatos_por_rotulo"][r] for s in list(saida.values()))
                                  for r in LABELS},
    }
    return saida
