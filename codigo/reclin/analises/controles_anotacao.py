"""Controles diagnósticos informados pelas anotações (tipos semânticos do SemClinBr).

**Não são sistemas do RECLin-PT.** As quatro configurações experimentais
(baseline, Pair-Aware linear, Two-Stage e Pair-Aware MLP) não leem tipos
semânticos: os marcadores de entidade são sem tipo (`entrada.janela_marcada`).
Estes controles fazem o oposto, de propósito: usam o tipo `Negation` que a
anotação atribui às pistas. Servem para medir quanto da tarefa a própria
anotação já entrega, ou seja, o tamanho do atalho que um modelo teria se os
tipos entrassem como atributo de entrada. Não treinam, não escolhem parâmetros
e não mudam dados, partições nem modelos.

Dois controles:

* `controle_tipo` — **identificação de pistas pelo tipo.** No nível da
  entidade, prevê "pista" para toda entidade cujo tipo inclui `Negation`.
  Verdade: as entidades que são e1 de ao menos uma `negation_of` anotada (a
  convenção da tarefa: e1 nega e2). No nível do par, sobre os candidatos da
  tarefa, prevê `negation_of` para todo par cujo e1 tem o tipo `Negation` e
  cujo e2 não tem.
* `controle_tipo_posicao` — **tipo + posição.** Para cada entidade com o tipo
  `Negation`, toma como alvo a entidade seguinte no texto: a primeira, na ordem
  (start, end, id), que começa depois do fim da pista e não tem o tipo
  `Negation`. Prevê `negation_of` para o par (pista, alvo) se ele for um
  candidato da tarefa (`entity_gap` <= `max_gap`); todos os outros candidatos
  ficam `no_relation`.

Os dois são avaliados sobre o mesmo conjunto de candidatos dos modelos
(`tarefa.candidatos`, `max_gap` 25), com as fórmulas de
`avaliacao.metricas.resumo_alvo` (TP, FP, FN, precisão, recall, F1 de
`negation_of`). O recall também é dado sobre as relações anotadas (o teto da
janela entra como falso negativo).

O TEST é recusado: os controles medem a anotação de TRAIN e DEV. O tipo é lido
do campo `type` das entidades, que é a cópia do atributo `tag` do XML
(`corpus/conversao.py`): tipos UMLS e os rótulos do corpus, unidos por `|`.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from reclin.avaliacao.metricas import resumo_alvo
from reclin.tarefa import LABEL2ID, MAX_GAP, NEG, NOREL, Documento, Entidade, candidatos, entity_gap

TIPO_PISTA = "Negation"
PARTICOES_PERMITIDAS = ("train", "dev")
NATUREZA = ("controle diagnóstico informado pelas anotações (tipos semânticos do SemClinBr); "
            "não é configuração experimental do RECLin-PT")


def tipos(entidade: Entidade) -> frozenset[str]:
    """Os rótulos do campo `type` (o `tag` do XML), separados por `|`."""
    return frozenset(t.strip() for t in entidade["type"].split("|") if t.strip())


def e_pista_pelo_tipo(entidade: Entidade) -> bool:
    return TIPO_PISTA in tipos(entidade)


def conferir_particao(particao: str) -> None:
    if particao not in PARTICOES_PERMITIDAS:
        raise ValueError(f"controle diagnóstico só em {PARTICOES_PERMITIDAS}; {particao!r} recusada "
                         "(o TEST não é avaliado nesta etapa)")


def _alvo(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """TP, FP, FN, P, R e F1 de `negation_of` (as fórmulas de `metricas.resumo_alvo`)."""
    if not y_true:                       # nenhum candidato: nada previsto, nada a acertar
        return _prf(0, 0, 0)
    alvo = resumo_alvo(y_true, y_pred)
    return {k: alvo[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0}


def pistas_anotadas(doc: Documento) -> set[str]:
    """Ids das entidades que são e1 de alguma `negation_of` anotada."""
    return {r["e1_id"] for r in doc["relations"] if r["type"] == "negation_of"}


# --------------------------------------------------------------------------- #
# Controle 1: identificação de pistas pelo tipo                                #
# --------------------------------------------------------------------------- #
def controle_tipo(documentos: Iterable[Documento], max_gap: int = MAX_GAP) -> dict[str, Any]:
    docs = list(documentos)
    tp = fp = fn = 0
    for doc in docs:
        gold = pistas_anotadas(doc)
        for e in doc["entities"]:
            pred, verdadeiro = e_pista_pelo_tipo(e), e["id"] in gold
            tp += pred and verdadeiro
            fp += pred and not verdadeiro
            fn += verdadeiro and not pred
    entidade = _prf(tp, fp, fn) | {
        "previsao": f"entidade com o tipo {TIPO_PISTA}",
        "verdade": "entidade que é e1 de ao menos uma negation_of anotada",
        "n_entidades": sum(len(d["entities"]) for d in docs)}

    cands = candidatos(docs, max_gap=max_gap)
    y_true = [LABEL2ID[c["label"]] for c in cands]
    y_pred = [NEG if e_pista_pelo_tipo(c["e1"]) and not e_pista_pelo_tipo(c["e2"]) else NOREL
              for c in cands]
    par = _alvo(y_true, y_pred) | {
        "previsao": f"negation_of para todo candidato com e1 {TIPO_PISTA} e e2 sem {TIPO_PISTA}",
        "n_candidatos": len(cands), "max_gap": max_gap}
    return {"nivel_entidade": entidade, "nivel_par": par}


# --------------------------------------------------------------------------- #
# Controle 2: tipo + posição (pista -> entidade seguinte)                     #
# --------------------------------------------------------------------------- #
def alvo_seguinte(pista: Entidade, entidades: Sequence[Entidade]) -> Entidade | None:
    """A primeira entidade, na ordem (start, end, id), que começa no fim da pista
    ou depois e não tem o tipo `Negation`; None se não houver."""
    for e in sorted(entidades, key=lambda x: (x["start"], x["end"], x["id"])):
        if e["id"] != pista["id"] and e["start"] >= pista["end"] and not e_pista_pelo_tipo(e):
            return e
    return None


def predicoes_tipo_posicao(documentos: Iterable[Documento], max_gap: int = MAX_GAP
                           ) -> tuple[list[int], list[int], dict[str, int]]:
    """`y_true` e `y_pred` alinhados com `tarefa.candidatos`, e contagens da regra."""
    y_true: list[int] = []
    y_pred: list[int] = []
    cont: Counter[str] = Counter(dict.fromkeys(("pistas", "pistas_sem_entidade_seguinte",
                                                "alvo_fora_da_janela", "pares_previstos"), 0))
    for doc in documentos:
        escolhidos = set()
        for e in doc["entities"]:
            if not e_pista_pelo_tipo(e):
                continue
            cont["pistas"] += 1
            alvo = alvo_seguinte(e, doc["entities"])
            if alvo is None:
                cont["pistas_sem_entidade_seguinte"] += 1
            elif entity_gap(e, alvo) > max_gap:
                cont["alvo_fora_da_janela"] += 1
            else:
                escolhidos.add((e["id"], alvo["id"]))
        for c in candidatos([doc], max_gap=max_gap):
            y_true.append(LABEL2ID[c["label"]])
            y_pred.append(NEG if (c["e1"]["id"], c["e2"]["id"]) in escolhidos else NOREL)
        cont["pares_previstos"] += len(escolhidos)
    return y_true, y_pred, dict(cont)


def controle_tipo_posicao(documentos: Iterable[Documento], max_gap: int = MAX_GAP) -> dict[str, Any]:
    docs = list(documentos)
    y_true, y_pred, cont = predicoes_tipo_posicao(docs, max_gap=max_gap)
    alvo = _alvo(y_true, y_pred)
    anotadas = sum(1 for d in docs for r in d["relations"] if r["type"] == "negation_of")
    return alvo | {
        "previsao": (f"negation_of para (pista, entidade seguinte): pista = entidade com o tipo {TIPO_PISTA}; "
                     f"alvo = primeira entidade sem {TIPO_PISTA} que começa depois do fim da pista"),
        "recall_sobre_anotadas": alvo["tp"] / anotadas if anotadas else 0.0,
        "negation_of_anotadas": anotadas, "n_candidatos": len(y_true), "max_gap": max_gap,
        "contagens_da_regra": cont}


# --------------------------------------------------------------------------- #
# Relatório                                                                   #
# --------------------------------------------------------------------------- #
def relatorio(particoes: dict[str, list[Documento]], max_gap: int = MAX_GAP) -> dict[str, Any]:
    """Os dois controles por partição (só TRAIN e DEV). Só números agregados."""
    for nome in particoes:
        conferir_particao(nome)
    return {"natureza": NATUREZA, "tipo_pista": TIPO_PISTA, "max_gap": max_gap,
            "particoes": {nome: {"controle_tipo": controle_tipo(docs, max_gap=max_gap),
                                 "controle_tipo_posicao": controle_tipo_posicao(docs, max_gap=max_gap)}
                          for nome, docs in particoes.items()}}
