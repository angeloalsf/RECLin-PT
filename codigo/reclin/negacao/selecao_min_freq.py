"""Seleção do `min_freq` do léxico de pistas só com o TRAIN (pendência D4).

Implementa o procedimento pré-registrado em
`docs/protocolo/d4-min-freq.md`: validação cruzada em 5 partes por
documento, estratificada pela presença de `negation_of`; em cada parte, o
léxico induzido das outras quatro é avaliado como detector de `negation_of`
nos pares candidatos da parte retida ("o `e1` é pista"); VP, FP e FN somados;
F1 = 2·VP / (2·VP + FP + FN); maior F1, empate → maior `min_freq`.

Nenhum modelo, nenhuma predição, nenhum resultado histórico: só os documentos
do TRAIN recebidos de quem chama. O módulo não lê arquivos.
"""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Any, Iterable, Sequence

from reclin.negacao import lexico
from reclin.tarefa import MAX_GAP, Documento, iter_candidate_pairs
from reclin.util.io import sha256_json

# As constantes do pré-registro (bloco JSON de docs/protocolo/d4-min-freq.md).
CANDIDATOS: tuple[int, ...] = (1, 2, 3, 5, 10)
N_PARTES = 5
SEMENTE = 42
MAX_GAP_SELECAO = MAX_GAP
METRICA = "F1 = 2VP/(2VP+FP+FN), contagens somadas nas partes"
DESEMPATE = "maior min_freq"
ESTRATIFICACAO = "presença de negation_of no documento"


def _chave(doc: Documento) -> tuple[int, str]:
    doc_id = doc["doc_id"]
    return (int(doc_id) if doc_id.isdigit() else 0, doc_id)


def _tem_negacao(doc: Documento) -> bool:
    return any(r["type"] == "negation_of" for r in doc["relations"])


def atribuir_partes(documentos: Iterable[Documento], n_partes: int = N_PARTES,
                    semente: int = SEMENTE) -> dict[str, int]:
    """`doc_id` → parte (0..n_partes-1). Documentos ordenados pelo `doc_id`,
    separados em com e sem `negation_of`, cada grupo embaralhado (o "com"
    primeiro) pelo mesmo `random.Random(semente)`; o i-ésimo de cada grupo vai
    para a parte `i mod n_partes`."""
    docs = sorted(documentos, key=_chave)
    ids = [d["doc_id"] for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("doc_id repetido")
    com = [d["doc_id"] for d in docs if _tem_negacao(d)]
    sem = [d["doc_id"] for d in docs if not _tem_negacao(d)]
    rng = random.Random(semente)
    rng.shuffle(com)
    rng.shuffle(sem)
    partes: dict[str, int] = {}
    for grupo in (com, sem):
        for i, doc_id in enumerate(grupo):
            partes[doc_id] = i % n_partes
    return partes


def contar(documentos: Iterable[Documento], lex: Iterable[str], max_gap: int = MAX_GAP_SELECAO) -> dict[str, int]:
    """VP, FP e FN do detector "o `e1` é pista" nos pares candidatos."""
    conjunto = set(lex)
    vp = fp = fn = 0
    for doc in documentos:
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            pista = lexico.e_pista(c["e1"], conjunto)
            neg = c["label"] == "negation_of"
            vp += pista and neg
            fp += pista and not neg
            fn += neg and not pista
    return {"vp": vp, "fp": fp, "fn": fn}


def f1(contagem: dict[str, int]) -> Fraction:
    """F1 exato (fração); 0 quando não há positivos nem preditos."""
    den = 2 * contagem["vp"] + contagem["fp"] + contagem["fn"]
    return Fraction(2 * contagem["vp"], den) if den else Fraction(0)


def _descrever(contagem: dict[str, int]) -> dict[str, Any]:
    vp, fp, fn = contagem["vp"], contagem["fp"], contagem["fn"]
    exato = f1(contagem)
    return {**contagem, "f1": float(exato), "f1_fracao": f"{exato.numerator}/{exato.denominator}",
            "precisao": vp / (vp + fp) if vp + fp else None,
            "cobertura": vp / (vp + fn) if vp + fn else None}


def escolher(resultados: dict[int, dict[str, int]]) -> int:
    """Maior F1 exato; empate → maior `min_freq`."""
    return max(resultados, key=lambda m: (f1(resultados[m]), m))


def selecionar(documentos_train: Sequence[Documento], *, candidatos: Sequence[int] = CANDIDATOS,
               n_partes: int = N_PARTES, semente: int = SEMENTE,
               max_gap: int = MAX_GAP_SELECAO) -> dict[str, Any]:
    """O procedimento completo (ver o módulo). Devolve as contagens por valor
    e por parte, o valor escolhido e o léxico final (induzido do TRAIN inteiro)."""
    docs = list(documentos_train)
    partes = atribuir_partes(docs, n_partes, semente)
    por_parte = {k: [d for d in docs if partes[d["doc_id"]] == k] for k in range(n_partes)}
    resto = {k: [d for d in docs if partes[d["doc_id"]] != k] for k in range(n_partes)}

    somas: dict[int, dict[str, int]] = {}
    detalhes: dict[str, Any] = {}
    for m in candidatos:
        total = {"vp": 0, "fp": 0, "fn": 0}
        linhas = []
        for k in range(n_partes):
            lex = lexico.induzir(resto[k], min_freq=m, max_gap=max_gap)
            c = contar(por_parte[k], lex, max_gap)
            for chave in total:
                total[chave] += c[chave]
            linhas.append({"parte": k, "n_formas": len(lex), "lexico_sha1": lexico.lexico_sha1(lex), **_descrever(c)})
        somas[m] = total
        detalhes[str(m)] = {"total": _descrever(total), "partes": linhas}

    escolhido = escolher(somas)
    final = lexico.induzir(docs, min_freq=escolhido, max_gap=max_gap)
    ordem = sorted(partes, key=lambda i: (int(i) if i.isdigit() else 0, i))
    return {
        "candidatos": list(candidatos), "n_partes": n_partes, "semente": semente, "max_gap": max_gap,
        "metrica": METRICA, "desempate": DESEMPATE, "estratificacao": ESTRATIFICACAO,
        "n_documentos": len(docs),
        "documentos_por_parte": [sum(1 for v in partes.values() if v == k) for k in range(n_partes)],
        "com_negation_of_por_parte": [sum(1 for d in por_parte[k] if _tem_negacao(d)) for k in range(n_partes)],
        "atribuicao_sha256": sha256_json([[i, partes[i]] for i in ordem]),
        "resultados": detalhes,
        "ordem_por_f1": sorted(somas, key=lambda m: (f1(somas[m]), m), reverse=True),
        "min_freq_escolhido": escolhido,
        "lexico_final": {"min_freq": escolhido, "max_gap": max_gap, "n_formas": len(final),
                         "lexico_sha1": lexico.lexico_sha1(final), "formas": final},
    }
