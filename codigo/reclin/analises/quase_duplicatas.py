"""Detector de documentos quase duplicados entre (e dentro de) partições.

Só **identifica e registra**: não muda partições, não remove documentos e não
grava texto. A saída tem só ids, partições, similaridades e contagens.

Normalização: minúsculas e **sem nenhum espaço em branco** (espaço, quebra de
linha, tabulação). É a normalização que revela as cópias do SemClinBr que
diferem só em espaços. Similaridade: Jaccard entre os conjuntos de *shingles*
de `k` caracteres do texto normalizado. Um par é registrado quando o Jaccard é
pelo menos `limiar`; para não comparar textos de tamanhos muito diferentes, o
par só é avaliado se a razão entre os números de *shingles* é pelo menos
`razao_minima` (um par com Jaccard >= `limiar` sempre passa nesse filtro
quando `razao_minima <= limiar`).

Para cada par, também são contadas (sem texto):

* as diferenças entre os textos originais, por classe de caractere (espaço,
  letra, dígito, pontuação), pelo `difflib`;
* a concordância das anotações: entidades em comum (texto da entidade sem
  caixa + tipo) e relações em comum (textos dos dois argumentos + tipo), sem
  offsets, porque os espaços deslocam as posições. É uma estimativa
  indicativa: entidades com fronteiras diferentes não casam.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from itertools import combinations
from typing import Any, Iterable

from reclin.tarefa import Documento

LIMIAR = 0.8
K = 8
MIN_SHINGLES = 10


def normalizar(texto: str) -> str:
    return re.sub(r"\s+", "", texto).lower()


def shingles(texto: str, k: int = K) -> frozenset[str]:
    t = normalizar(texto)
    if len(t) < k:
        return frozenset({t}) if t else frozenset()
    return frozenset(t[i:i + k] for i in range(len(t) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _classe(c: str) -> str:
    if c.isspace():
        return "espaco"
    if c.isdigit():
        return "digito"
    if c.isalpha():
        return "letra"
    return "pontuacao"


def diferencas(texto_a: str, texto_b: str) -> dict[str, Any]:
    """Caracteres diferentes entre os dois textos, por classe (sem o texto)."""
    sm = difflib.SequenceMatcher(None, texto_a, texto_b, autojunk=False)
    classes: Counter[str] = Counter()
    blocos = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        blocos += 1
        classes.update(_classe(c) for c in texto_a[i1:i2] + texto_b[j1:j2])
    return {"blocos": blocos, "caracteres_por_classe": dict(sorted(classes.items())),
            "so_espacos": bool(classes) and set(classes) == {"espaco"}}


def _assinatura(doc: Documento) -> tuple[Counter, Counter]:
    m = {e["id"]: e for e in doc["entities"]}
    ents = Counter((e["text"].lower(), e["type"]) for e in doc["entities"])
    rels = Counter((m[r["e1_id"]]["text"].lower(), m[r["e2_id"]]["text"].lower(), r["type"])
                   for r in doc["relations"] if r["e1_id"] in m and r["e2_id"] in m)
    return ents, rels


def concordancia(doc_a: Documento, doc_b: Documento) -> dict[str, Any]:
    """Contagens das anotações dos dois documentos e do que coincide (sem texto)."""
    ea, ra = _assinatura(doc_a)
    eb, rb = _assinatura(doc_b)
    comum_r = ra & rb
    tipos = sorted({k[2] for k in ra} | {k[2] for k in rb})
    return {
        "entidades": {"a": sum(ea.values()), "b": sum(eb.values()), "em_comum": sum((ea & eb).values())},
        "relacoes": {"a": sum(ra.values()), "b": sum(rb.values()), "em_comum": sum(comum_r.values())},
        "relacoes_por_tipo": {t: {"a": sum(v for k, v in ra.items() if k[2] == t),
                                  "b": sum(v for k, v in rb.items() if k[2] == t),
                                  "em_comum": sum(v for k, v in comum_r.items() if k[2] == t)}
                              for t in tipos}}


def detectar(particoes: dict[str, Iterable[Documento]], *, limiar: float = LIMIAR, k: int = K,
             razao_minima: float | None = None, min_shingles: int = MIN_SHINGLES) -> dict[str, Any]:
    """Pares de documentos com Jaccard >= `limiar`, entre e dentro das partições.

    `particoes` é `{nome: documentos}`. Documentos com menos de `min_shingles`
    *shingles* são comparados só pela igualdade do texto normalizado.
    """
    if not 0 < limiar <= 1:
        raise ValueError(f"limiar fora de (0, 1]: {limiar}")
    razao_minima = limiar if razao_minima is None else razao_minima
    docs = [(nome, d) for nome, ds in particoes.items() for d in ds]
    ids = [d["doc_id"] for _, d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("doc_id repetido entre as partições")
    sh = [shingles(d["text"], k) for _, d in docs]
    norm = [normalizar(d["text"]) for _, d in docs]
    pares = []
    for i, j in combinations(range(len(docs)), 2):
        a, b = sh[i], sh[j]
        curtos = len(a) < min_shingles or len(b) < min_shingles
        if curtos:
            if norm[i] != norm[j] or not norm[i]:
                continue
            sim = 1.0
        else:
            if min(len(a), len(b)) / max(len(a), len(b)) < razao_minima:
                continue
            sim = jaccard(a, b)
            if sim < limiar:
                continue
        (pa, da), (pb, db) = docs[i], docs[j]
        pares.append({
            "doc_a": da["doc_id"], "particao_a": pa, "doc_b": db["doc_id"], "particao_b": pb,
            "entre_particoes": pa != pb, "jaccard": round(sim, 4),
            "identicos_sem_espacos": norm[i] == norm[j],
            "identicos_byte_a_byte": da["text"] == db["text"],
            "tamanhos": [len(da["text"]), len(db["text"])],
            "diferencas": diferencas(da["text"], db["text"]),
            "anotacao": concordancia(da, db)})
    pares.sort(key=lambda p: (-p["jaccard"], p["doc_a"], p["doc_b"]))
    entre = Counter(f"{p['particao_a']}-{p['particao_b']}" for p in pares if p["entre_particoes"])
    return {"parametros": {"normalizacao": "minúsculas, sem espaços em branco", "k": k,
                           "limiar_jaccard": limiar, "razao_minima": razao_minima,
                           "min_shingles": min_shingles},
            "n_documentos": len(docs), "n_pares": len(pares),
            "n_pares_identicos_sem_espacos": sum(p["identicos_sem_espacos"] for p in pares),
            "pares_entre_particoes": dict(sorted(entre.items())), "pares": pares}


def documentos_de_test_com_copia_no_treino(resultado: dict[str, Any]) -> list[str]:
    """Ids do TEST com cópia (idêntica sem espaços) no TRAIN, a partir de `detectar`."""
    ids = set()
    for p in resultado["pares"]:
        if p["identicos_sem_espacos"] and {p["particao_a"], p["particao_b"]} == {"train", "test"}:
            ids.add(p["doc_a"] if p["particao_a"] == "test" else p["doc_b"])
    return sorted(ids, key=lambda x: (int(x) if x.isdigit() else 0, x))
