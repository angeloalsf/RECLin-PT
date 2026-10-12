"""Alinhamento das entidades com o texto do documento.

Uma entidade anotada traz `start`, `end` e `text`. Quatro situações:

* `exato` — `texto[start:end] == text`;
* `equivalente` — os offsets cobrem o mesmo texto a menos de espaços e de
  entidades HTML (`&gt;`, por exemplo): `35 , 7ºC` contra `35,7ºC`;
* `deslocado` — o texto da entidade aparece perto de `start` (dentro de uma
  janela), mas não nos offsets: deriva de offsets na conversão do corpus; o
  deslocamento da ocorrência mais próxima é registrado;
* `ausente` — nenhuma das anteriores.

Nada é corrigido: a análise só descreve, porque as janelas marcadas dos
modelos usam os offsets como estão (`entrada.janela_marcada`), e qualquer
correção mudaria os dados congelados da tarefa.

Origem no legado (só a finalidade): `align_entities`, usado pela auditoria dos
marcadores e pela dos falsos positivos.
"""
from __future__ import annotations

import html
from collections import Counter
from typing import Any, Iterable

from reclin.tarefa import LABELS, MAX_GAP, Documento, Entidade, iter_candidate_pairs

EXATO, EQUIVALENTE, DESLOCADO, AUSENTE = "exato", "equivalente", "deslocado", "ausente"
STATUS = (EXATO, EQUIVALENTE, DESLOCADO, AUSENTE)
JANELA = 50


def _sem_espacos(texto: str) -> str:
    return "".join(html.unescape(texto).split())


def alinhar_entidade(texto: str, entidade: Entidade, janela: int = JANELA) -> dict[str, Any]:
    """`{"status", "delta"}`: `delta` é o deslocamento da ocorrência mais
    próxima (positivo = mais adiante no texto); None se exata ou ausente."""
    inicio, fim, alvo = entidade["start"], entidade["end"], entidade["text"]
    if texto[inicio:fim] == alvo:
        return {"status": EXATO, "delta": None}
    if _sem_espacos(texto[inicio:fim]) == _sem_espacos(alvo) and alvo.strip():
        return {"status": EQUIVALENTE, "delta": None}
    if not alvo:
        return {"status": AUSENTE, "delta": None}
    lo, hi = max(0, inicio - janela), min(len(texto), fim + janela)
    melhor = None
    pos = texto.find(alvo, lo)
    while pos != -1 and pos + len(alvo) <= hi:
        delta = pos - inicio
        if melhor is None or (abs(delta), delta) < (abs(melhor), melhor):
            melhor = delta
        pos = texto.find(alvo, pos + 1)
    if melhor is None:
        return {"status": AUSENTE, "delta": None}
    return {"status": DESLOCADO, "delta": melhor}


def auditar(documentos: Iterable[Documento], janela: int = JANELA, exemplos: int = 10) -> dict[str, Any]:
    """Contagem por status, histograma dos deslocamentos e alguns exemplos."""
    status: Counter[str] = Counter()
    deltas: Counter[int] = Counter()
    casos = []
    n_docs = 0
    for doc in documentos:
        n_docs += 1
        for ent in doc["entities"]:
            r = alinhar_entidade(doc["text"], ent, janela)
            status[r["status"]] += 1
            if r["status"] != EXATO:
                if r["delta"] is not None:
                    deltas[r["delta"]] += 1
                if len(casos) < exemplos:
                    casos.append({"doc_id": doc["doc_id"], "id": ent["id"], "start": ent["start"],
                                  "end": ent["end"], "text": ent["text"],
                                  "no_texto": doc["text"][ent["start"]:ent["end"]], **r})
    total = sum(status.values())
    return {"n_documentos": n_docs, "n_entidades": total,
            "status": {s: status.get(s, 0) for s in STATUS},
            "fracao_exata": status.get(EXATO, 0) / total if total else None,
            "deslocamentos": {str(d): n for d, n in sorted(deltas.items())},
            "exemplos": casos, "janela": janela}


def impacto_nos_pares(documentos: Iterable[Documento], max_gap: int | None = None,
                      janela: int = JANELA) -> dict[str, Any]:
    """Pares candidatos (por rótulo) com ao menos uma entidade cujo texto não
    está nos próprios offsets (`deslocado` ou `ausente`): são os pares cuja
    janela marcada envolve um trecho diferente do texto anotado da entidade."""
    total: Counter[str] = Counter()
    afetados: Counter[str] = Counter()
    for doc in documentos:
        status = {e["id"]: alinhar_entidade(doc["text"], e, janela)["status"] for e in doc["entities"]}
        for c in iter_candidate_pairs(doc, max_gap=MAX_GAP if max_gap is None else max_gap):
            total[c["label"]] += 1
            if status[c["e1"]["id"]] in (DESLOCADO, AUSENTE) or status[c["e2"]["id"]] in (DESLOCADO, AUSENTE):
                afetados[c["label"]] += 1
    return {r: {"pares": total.get(r, 0), "afetados": afetados.get(r, 0),
                "fracao": afetados.get(r, 0) / total[r] if total.get(r) else None} for r in LABELS}


def perfil_da_deriva(documentos: Iterable[Documento], janela: int = JANELA) -> dict[str, Any]:
    """Por documento, a sequência dos deslocamentos das entidades na ordem do
    texto (0 para as alinhadas): sem deriva, deriva monótona (só cresce em
    módulo, como um deslocamento acumulado) ou não monótona. Uma deriva
    acumulada é o padrão esperado quando caracteres são removidos do texto
    depois de calculados os offsets; a busca do texto na janela pode achar uma
    ocorrência vizinha errada em entidades curtas e repetidas, o que aparece
    como não monótona."""
    perfil: Counter[str] = Counter()
    maximos = []
    for doc in documentos:
        seq = []
        for e in sorted(doc["entities"], key=lambda e: (e["start"], e["end"], e["id"])):
            r = alinhar_entidade(doc["text"], e, janela)
            if r["status"] in (EXATO, EQUIVALENTE):
                seq.append(0)
            elif r["status"] == DESLOCADO:
                seq.append(r["delta"])
        if not seq or all(d == 0 for d in seq):
            perfil["sem_deriva"] += 1
            continue
        perfil["monotona" if all(b <= a for a, b in zip(seq, seq[1:])) else "nao_monotona"] += 1
        maximos.append(max(abs(d) for d in seq))
    maximos.sort()
    return {"documentos": {k: perfil.get(k, 0) for k in ("sem_deriva", "monotona", "nao_monotona")},
            "deriva_maxima": {"mediana": maximos[len(maximos) // 2] if maximos else None,
                              "max": maximos[-1] if maximos else None},
            "texto_com_quebra_de_linha": sum(1 for d in documentos if "\n" in d["text"])
            if isinstance(documentos, list) else None}
