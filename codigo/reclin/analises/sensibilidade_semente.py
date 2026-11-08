"""Sensibilidade da divisão 800/100/100 à semente do particionamento.

Roda o **mesmo algoritmo** de `particoes.particionar` (estratificação pela
presença de `negation_of`, 80/10/10 por documento) com outras sementes e conta,
em cada partição, documentos e relações por tipo. Não grava partições e não
muda as congeladas: a semente da tarefa continua 42. A saída é só agregada.

Serve para dizer o quanto a partição da semente 42 é típica: por exemplo, em
que percentil fica o número de `negation_of` do DEV entre as sementes testadas.
Usa as anotações das três partições só como contagem descritiva; nada é
escolhido com ela.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from reclin.particoes import PARTICOES, SEED, particionar
from reclin.tarefa import Documento

TIPOS = ("negation_of", "associated_with")


def contar(particoes: dict[str, list[Documento]]) -> dict[str, dict[str, int]]:
    out = {}
    for nome in PARTICOES:
        docs = particoes[nome]
        rel = Counter(r["type"] for d in docs for r in d["relations"])
        out[nome] = {"documentos": len(docs),
                     "documentos_com_negation_of": sum(1 for d in docs
                                                       if any(r["type"] == "negation_of" for r in d["relations"])),
                     **{t: rel.get(t, 0) for t in TIPOS}}
    return out


def _resumo(valores: Sequence[int], valor_referencia: int) -> dict[str, Any]:
    v = sorted(valores)
    n = len(v)

    def q(p: float) -> int:
        return v[min(n - 1, int(p * n))]

    return {"min": v[0], "p5": q(0.05), "mediana": q(0.5), "p95": q(0.95), "max": v[-1],
            "semente_de_referencia": valor_referencia,
            "percentil_da_referencia": sum(1 for x in v if x <= valor_referencia) / n}


def sensibilidade(documentos: Iterable[Documento], sementes: Sequence[int] = tuple(range(200)),
                  referencia: int = SEED) -> dict[str, Any]:
    """Contagens por semente e o resumo de cada contagem, com a posição da semente de referência."""
    docs = list(documentos)
    sementes = sorted(set(int(s) for s in sementes) | {referencia})
    por_semente = {s: contar(particionar(docs, seed=s)) for s in sementes}
    ref = por_semente[referencia]
    resumo = {nome: {campo: _resumo([por_semente[s][nome][campo] for s in sementes], ref[nome][campo])
                     for campo in ("documentos_com_negation_of", *TIPOS)}
              for nome in PARTICOES}
    return {"n_documentos": len(docs), "n_sementes": len(sementes), "sementes": [sementes[0], sementes[-1]],
            "semente_de_referencia": referencia, "referencia": ref, "resumo": resumo,
            "por_semente": {str(s): por_semente[s] for s in sementes}}
