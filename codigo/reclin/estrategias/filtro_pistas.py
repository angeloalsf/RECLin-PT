"""Filtro de pistas: pós-processamento das predições de outra execução.

Toda predição `negation_of` cujo e1 não é pista de negação do léxico
(`reclin.negacao.lexico.e_pista`) é rebaixada para `no_relation`; as demais
predições ficam como estão. O filtro não treina nada e não muda o conjunto de
candidatos: as predições filtradas têm o mesmo `y_true` da execução de base e
podem ser pareadas com ela. Como só remove `negation_of`, a cobertura do léxico
(fração dos pares `negation_of` cujo e1 é pista) é o teto do recall.

Calibração, só no DEV (`calibrar_min_freq` e `calibrar_porta_gap`), com o
critério fixado no legado antes de rodar:

1. `min_freq` do léxico, entre `MIN_FREQS`: maior média do F1 de
   `negation_of` do filtro sobre as execuções de base; empate pelo menor
   `min_freq` (o léxico de maior cobertura). Resultado do legado: 3.
2. Porta de gap (filtro seguido de `porta_gap`), entre `GAPS_PORTA`: maior
   média do F1; empate pelo menor limiar. Resultado do legado: 25, igual à
   janela de candidatos, ou seja, a porta desligada — a variante não foi
   adotada.

O F1 do critério é o de `avaliacao.metricas.resumo_alvo_pr` (2·P·R/(P+R),
como no legado), e a média entre execuções usa `math.fsum`, que dá o mesmo
resultado em qualquer versão do Python (o legado usava `sum()` no Python 3.13).

Origem no legado: `apply_cue_filter` (`src/negation_lexicon.py`), a varredura
de `min_freq` e `gap_gate` (`scripts/calibrate_cue_filter.py`) e o sidecar do
filtro (`scripts/run_fase2_test.py`).
"""
from __future__ import annotations

import math
from typing import Any, Collection, Mapping, Sequence

from reclin.avaliacao import metricas
from reclin.execucao import predicoes as sidecar
from reclin.negacao.lexico import e_pista
from reclin.tarefa import MAX_GAP, NEG, NOREL, Candidato, entity_gap, y_true

NOME = "filtro_pistas"
MIN_FREQS = (1, 2, 3, 5, 10)
GAPS_PORTA = (0, 1, 2, 3, 5, 10, 25)


def filtrar(candidatos: Sequence[Candidato], y_pred: Sequence[int],
            lexico: Collection[str]) -> list[int]:
    """Rebaixa para `no_relation` cada `negation_of` cujo e1 não é pista."""
    if len(candidatos) != len(y_pred):
        raise ValueError(f"{len(candidatos)} candidatos e {len(y_pred)} predições: "
                         "as predições não são deste conjunto de candidatos")
    return [NOREL if p == NEG and not e_pista(c["e1"], lexico) else p
            for c, p in zip(candidatos, y_pred)]


def porta_gap(candidatos: Sequence[Candidato], y_pred: Sequence[int],
              max_target_gap: int) -> list[int]:
    """Rebaixa para `no_relation` cada `negation_of` com `entity_gap` acima do
    limiar. Variante testada na calibração (filtro + porta), não adotada."""
    return [NOREL if p == NEG and entity_gap(c["e1"], c["e2"]) > max_target_gap else p
            for c, p in zip(candidatos, y_pred)]


def nome_execucao(nome_base: str) -> str:
    """`baseline_biobertpt_seed42` → `filtro_biobertpt_seed42`; outras bases
    ganham só o prefixo (`restrito_x_seed42` → `filtro_restrito_x_seed42`)."""
    return "filtro_" + nome_base.removeprefix("baseline_")


def montar_predicoes(base: Mapping[str, Any], candidatos: Sequence[Candidato],
                     lexico: Collection[str], *, base_preds: str, min_freq: int,
                     max_gap: int = MAX_GAP) -> dict[str, Any]:
    """Sidecar do filtro aplicado às predições `base` (um sidecar), no formato
    do legado: `model` = `filtro(<model da base> s<seed>)`, a mesma semente, o
    mesmo `y_true`, e os extras `base_preds` (de onde vieram as predições) e
    `postproc` (os parâmetros do filtro)."""
    sidecar.validar(dict(base))
    if list(base["y_true"]) != y_true(candidatos):
        raise ValueError(f"{base_preds} não pertence a este conjunto de candidatos "
                         "(y_true diferente)")
    return sidecar.montar(
        model=f"filtro({base['model']} s{base['seed']})", seed=base["seed"],
        y_true=base["y_true"], y_pred=filtrar(candidatos, base["y_pred"], lexico),
        extra={"base_preds": base_preds,
               "postproc": {"kind": "cue_filter", "min_freq": min_freq,
                            "lexicon_size": len(lexico), "max_gap": max_gap,
                            "demote_to": "no_relation", "calibrated_on": "dev"}})


# --------------------------------------------------------------------------- #
# Calibração no DEV                                                           #
# --------------------------------------------------------------------------- #
def _media(valores: Sequence[float]) -> float:
    return math.fsum(valores) / len(valores)


def calibrar_min_freq(contagem_train: Mapping[str, int], candidatos_dev: Sequence[Candidato],
                      y_true_dev: Sequence[int], predicoes_dev: Mapping[str, Sequence[int]],
                      min_freqs: Sequence[int] = MIN_FREQS) -> dict[str, Any]:
    """Escolhe `min_freq` pelo F1 médio do filtro no DEV.

    `contagem_train` é a frequência das formas de e1 em `negation_of` no TRAIN
    (`negacao.lexico.contar_formas`); `predicoes_dev`, o `y_pred` do DEV de cada
    execução de base, pelo nome. Devolve o escolhido e a varredura completa
    (tamanho do léxico, cobertura no DEV e o resumo de cada execução)."""
    n_gold = sum(1 for t in y_true_dev if t == NEG)
    varredura = []
    for mf in min_freqs:
        lex = {forma for forma, n in contagem_train.items() if n >= mf}
        cobertos = sum(1 for c, t in zip(candidatos_dev, y_true_dev)
                       if t == NEG and e_pista(c["e1"], lex))
        por_execucao = {nome: metricas.resumo_alvo_pr(y_true_dev, filtrar(candidatos_dev, y, lex))
                        for nome, y in predicoes_dev.items()}
        varredura.append({"min_freq": mf, "n_formas": len(lex),
                          "cobertura": cobertos / n_gold if n_gold else None,
                          "por_execucao": por_execucao,
                          "f1_medio": _media([r["f1"] for r in por_execucao.values()])})
    melhor = max(varredura, key=lambda r: (r["f1_medio"], -r["min_freq"]))
    return {"min_freq": melhor["min_freq"], "varredura": varredura}


def calibrar_porta_gap(candidatos_dev: Sequence[Candidato], y_true_dev: Sequence[int],
                       predicoes_dev: Mapping[str, Sequence[int]], lexico: Collection[str],
                       gaps: Sequence[int] = GAPS_PORTA) -> dict[str, Any]:
    """Escolhe o limiar da porta de gap aplicada depois do filtro, pelo F1
    médio no DEV, com o léxico já calibrado."""
    varredura = []
    for g in gaps:
        por_execucao = {
            nome: metricas.resumo_alvo_pr(
                y_true_dev, porta_gap(candidatos_dev, filtrar(candidatos_dev, y, lexico), g))
            for nome, y in predicoes_dev.items()}
        varredura.append({"gap": g, "por_execucao": por_execucao,
                          "f1_medio": _media([r["f1"] for r in por_execucao.values()])})
    melhor = max(varredura, key=lambda r: (r["f1_medio"], -r["gap"]))
    return {"gap": melhor["gap"], "varredura": varredura}
