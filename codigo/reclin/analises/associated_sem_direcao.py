"""Avaliação secundária de `associated_with` sem considerar a direção do par.

O SemClinBr anota `associated_with` com direção, e a tarefa classifica pares
ordenados. Parte dos falsos positivos dessa classe são pares cuja direção
inversa está anotada (`reclin.analises.auditoria_execucoes`). Esta análise mede
quanto do erro de `associated_with` é só de direção: funde os dois sentidos de
cada par de entidades e avalia a classe sobre **pares não ordenados**.

Definição (por par não ordenado {a, b} do mesmo documento):

* referência positiva: algum dos sentidos é `associated_with` na anotação;
* predição positiva: o modelo previu `associated_with` em algum dos sentidos.

Só entram pares em que os dois sentidos são candidatos (o gerador enumera pares
ordenados dentro da mesma janela, então isso vale para todos; a função confere
e conta os que não tiverem o inverso). `negation_of` não é avaliada aqui: a
direção faz parte da sua definição (pista → conceito negado).

É uma análise descritiva e secundária. Não muda a tarefa, as métricas
principais nem o critério de seleção. Não lê o TEST.
"""
from __future__ import annotations

from typing import Any, Sequence

from reclin.tarefa import ASSOC, LABELS, Candidato


def _prf(vp: int, fp: int, fn: int) -> dict[str, Any]:
    p = vp / (vp + fp) if vp + fp else 0.0
    r = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * vp / (2 * vp + fp + fn) if 2 * vp + fp + fn else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": vp, "fp": fp, "fn": fn}


def avaliar(candidatos: Sequence[Candidato], y_pred: Sequence[int]) -> dict[str, Any]:
    """`associated_with` com direção (como na tarefa) e sem direção, no mesmo conjunto."""
    if len(candidatos) != len(y_pred):
        raise ValueError(f"{len(candidatos)} candidatos e {len(y_pred)} predições")
    rotulo_assoc = LABELS[ASSOC]
    ouro = [c["label"] == rotulo_assoc for c in candidatos]
    previsto = [int(p) == ASSOC for p in y_pred]

    vp = sum(o and p for o, p in zip(ouro, previsto))
    com_direcao = _prf(vp, sum(p and not o for o, p in zip(ouro, previsto)),
                       sum(o and not p for o, p in zip(ouro, previsto)))

    pares: dict[tuple[str, str, str], list[Any]] = {}       # {doc, a, b} -> [ouro, previsto, n_sentidos]
    for c, o, p in zip(candidatos, ouro, previsto):
        a, b = sorted((c["e1"]["id"], c["e2"]["id"]))
        reg = pares.setdefault((c["doc_id"], a, b), [False, False, 0])
        reg[0], reg[1], reg[2] = reg[0] or o, reg[1] or p, reg[2] + 1
    vp = sum(1 for o, p, _ in pares.values() if o and p)
    sem_direcao = _prf(vp, sum(1 for o, p, _ in pares.values() if p and not o),
                       sum(1 for o, p, _ in pares.values() if o and not p))
    return {
        "com_direcao": {**com_direcao, "n_pares_ordenados": len(candidatos), "suporte": sum(ouro)},
        "sem_direcao": {**sem_direcao, "n_pares_nao_ordenados": len(pares),
                        "suporte": sum(1 for o, _, _ in pares.values() if o),
                        "pares_sem_o_sentido_inverso": sum(1 for _, _, n in pares.values() if n != 2)},
        "diferenca_f1": sem_direcao["f1"] - com_direcao["f1"],
    }
