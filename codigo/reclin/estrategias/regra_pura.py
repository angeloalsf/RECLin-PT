"""Regra pura: `negation_of` por pista lexical e distância, sem modelo.

É o piso de `negation_of`: o que se consegue sem rede neural, só com o léxico
de pistas (`reclin.negacao.lexico`) e a distância em caracteres entre os spans
(`tarefa.entity_gap`). Quatro variantes:

* R1 — todo par cujo e1 é pista → `negation_of`;
* R2 — R1, mas só o alvo mais próximo de cada pista (um alvo por e1);
* R3 — R1 com `gap <= max_target_gap`;
* R4 — R2 e R3 juntas: o mais próximo entre os que passam no limiar.

"Mais próximo" desempata por menor `entity_gap`, depois menor `e2.start`,
depois menor `e2.id` como inteiro — determinístico, independente da ordem de
iteração. Tudo o que a regra não marca é `no_relation`; ela nunca prevê
`associated_with`. As predições cobrem o conjunto completo de candidatos da
partição, na ordem de `tarefa.iter_candidate_pairs`, e por isso podem ser
pareadas com as de qualquer outra execução. Não tem semente (`seed = null`).

Calibração, só no DEV (`calibrar`), com o critério fixado no legado antes de
rodar: varre R1 a R4 e, para R3 e R4, o limiar em `GAPS` (R1 e R2 aparecem uma
vez, com a janela de candidatos); escolhe o maior F1 de `negation_of`; empate
pela regra de menor índice, depois pelo menor limiar. O léxico é dado de
fora — no legado, o de `min_freq` escolhido pela calibração do filtro (3), que
é o léxico congelado da etapa 3. Resultado do legado: R3 com `gap <= 1`.

O F1 do critério é o de `avaliacao.metricas.resumo_alvo_pr` (2·P·R/(P+R),
como no legado).

Origem no legado: `predict_rule` e `RULES` (`scripts/make_rule_baseline.py`),
a varredura da regra (`scripts/calibrate_cue_filter.py`) e o sidecar da regra
(`scripts/run_fase2_test.py`).
"""
from __future__ import annotations

from typing import Any, Collection, Sequence

from reclin.avaliacao import metricas
from reclin.execucao import predicoes as sidecar
from reclin.negacao.lexico import e_pista
from reclin.tarefa import MAX_GAP, NEG, NOREL, Candidato, entity_gap, y_true

NOME = "regra_pura"
REGRAS = ("R1", "R2", "R3", "R4")
GAPS = (0, 1, 2, 3, 5, 10, 25)


def prever(candidatos: Sequence[Candidato], lexico: Collection[str], regra: str,
           max_target_gap: int) -> list[int]:
    """Aplica a regra e devolve `y_pred` (ids de `tarefa.LABELS`) para todos os
    candidatos. `max_target_gap` só é usado por R3 e R4."""
    if regra not in REGRAS:
        raise ValueError(f"regra desconhecida: {regra!r}; opções: {REGRAS}")
    mais_proximo = regra in ("R2", "R4")
    com_limiar = regra in ("R3", "R4")

    marcados = []     # (índice, doc_id, e1.id, gap, e2.start, e2.id)
    for i, c in enumerate(candidatos):
        if not e_pista(c["e1"], lexico):
            continue
        gap = entity_gap(c["e1"], c["e2"])
        if com_limiar and gap > max_target_gap:
            continue
        marcados.append((i, c["doc_id"], c["e1"]["id"], gap, c["e2"]["start"], c["e2"]["id"]))

    if mais_proximo:
        melhor: dict[tuple[str, str], tuple[tuple[int, int, int], int]] = {}
        for i, doc_id, e1_id, gap, e2_start, e2_id in marcados:
            chave, ordem = (doc_id, e1_id), (gap, e2_start, int(e2_id))
            if chave not in melhor or ordem < melhor[chave][0]:
                melhor[chave] = (ordem, i)
        positivos = {i for _, i in melhor.values()}
    else:
        positivos = {m[0] for m in marcados}

    y = [NOREL] * len(candidatos)
    for i in positivos:
        y[i] = NEG
    return y


def nome_modelo(regra: str, max_target_gap: int, min_freq: int) -> str:
    return f"regra {regra} (gap<={max_target_gap}, min_freq={min_freq})"


def montar_predicoes(candidatos: Sequence[Candidato], lexico: Collection[str], *,
                     regra: str, max_target_gap: int, min_freq: int,
                     max_gap: int = MAX_GAP) -> dict[str, Any]:
    """Sidecar da regra no formato do legado, com o extra `postproc` (os
    parâmetros da regra)."""
    return sidecar.montar(
        model=nome_modelo(regra, max_target_gap, min_freq), seed=None,
        y_true=y_true(candidatos), y_pred=prever(candidatos, lexico, regra, max_target_gap),
        extra={"postproc": {"kind": "pure_rule", "rule": regra,
                            "max_target_gap": max_target_gap, "min_freq": min_freq,
                            "lexicon_size": len(lexico), "max_gap": max_gap,
                            "calibrated_on": "dev"}})


def calibrar(candidatos_dev: Sequence[Candidato], y_true_dev: Sequence[int],
             lexico: Collection[str], *, gaps: Sequence[int] = GAPS,
             max_gap: int = MAX_GAP) -> dict[str, Any]:
    """Escolhe a regra e o limiar pelo F1 de `negation_of` no DEV. Devolve o
    escolhido e a varredura completa (contagens, P, R e F1 de cada combinação)."""
    varredura = []
    for regra in REGRAS:
        for g in (gaps if regra in ("R3", "R4") else (max_gap,)):
            r = metricas.resumo_alvo_pr(y_true_dev, prever(candidatos_dev, lexico, regra, g))
            varredura.append({"regra": regra, "gap": g, **r})
    melhor = max(varredura, key=lambda r: (r["f1"], -REGRAS.index(r["regra"]), -r["gap"]))
    return {"regra": melhor["regra"], "gap": melhor["gap"], "varredura": varredura}
