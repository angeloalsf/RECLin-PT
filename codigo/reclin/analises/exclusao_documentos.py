"""Análise de sensibilidade com documentos excluídos da avaliação (preparada, não executada no TEST).

Recalcula as métricas de uma execução **a partir das predições já gravadas**
(`predicoes_<particao>.json`), sobre os candidatos da partição menos os dos
documentos excluídos. Não roda modelo, não muda a partição, não muda as
predições e não grava nada por conta própria.

Uso previsto (pré-registrado em `docs/protocolo/fase-biobertpt-seed42.md`,
emenda E2): depois que o TEST de uma configuração for avaliado, uma vez, pelo
comando próprio, relatar também as métricas sem os documentos do TEST que têm
cópia no TRAIN (`DOCUMENTOS_TEST_COM_COPIA_NO_TREINO`). A análise principal
continua sendo a do TEST completo. Nesta etapa só o DEV pode ser usado; o TEST
exige `permitir_test=True`, que o script só passa com `--test-autorizado`.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from reclin.avaliacao.metricas import avaliar, resumo_alvo
from reclin.tarefa import LABEL2ID, MAX_GAP, Documento, candidatos

# As três notas do TEST que são cópias (diferentes só em espaços) de notas do
# TRAIN (9635, 9636, 9637), achadas pelo detector `analises.quase_duplicatas`.
DOCUMENTOS_TEST_COM_COPIA_NO_TREINO: tuple[str, ...] = ("9534", "9535", "9536")


def indices_mantidos(documentos: Iterable[Documento], excluir: Sequence[str],
                     max_gap: int = MAX_GAP) -> tuple[list[int], list[int], dict[str, int]]:
    """Posições dos candidatos mantidos, `y_true` do conjunto completo e contagens."""
    excluir = set(excluir)
    docs = list(documentos)
    presentes = {d["doc_id"] for d in docs}
    ausentes = excluir - presentes
    if ausentes:
        raise ValueError(f"documentos a excluir que não estão na partição: {sorted(ausentes)}")
    mantidos, y_true = [], []
    for i, c in enumerate(candidatos(docs, max_gap=max_gap)):
        y_true.append(LABEL2ID[c["label"]])
        if c["doc_id"] not in excluir:
            mantidos.append(i)
    return mantidos, y_true, {"candidatos": len(y_true), "mantidos": len(mantidos),
                              "excluidos": len(y_true) - len(mantidos),
                              "documentos_excluidos": len(excluir)}


def metricas_excluindo(predicoes: dict[str, Any], documentos: Iterable[Documento], *,
                       particao: str, excluir: Sequence[str] = DOCUMENTOS_TEST_COM_COPIA_NO_TREINO,
                       max_gap: int = MAX_GAP, permitir_test: bool = False) -> dict[str, Any]:
    """Métricas com e sem os documentos de `excluir`, a partir de um sidecar de predições."""
    if particao == "test" and not permitir_test:
        raise PermissionError("o TEST não é usado nesta etapa: a análise de sensibilidade no TEST só roda "
                              "depois da avaliação autorizada do TEST (permitir_test=True)")
    if particao not in ("dev", "test"):
        raise ValueError(f"partição de avaliação desconhecida: {particao!r}")
    mantidos, y_true, cont = indices_mantidos(documentos, excluir, max_gap)
    if list(predicoes["y_true"]) != y_true:
        raise ValueError("o y_true das predições não é o dos candidatos desta partição "
                         "(partição, versão dos dados ou max_gap diferentes)")
    y_pred = list(predicoes["y_pred"])
    if len(y_pred) != len(y_true):
        raise ValueError(f"{len(y_pred)} predições para {len(y_true)} candidatos")
    sub_t = [y_true[i] for i in mantidos]
    sub_p = [y_pred[i] for i in mantidos]
    alvo = ("tp", "fp", "fn", "precision", "recall", "f1")
    completo, sem = avaliar(y_true, y_pred), avaliar(sub_t, sub_p)
    return {"particao": particao, "documentos_excluidos": sorted(set(excluir)), "contagens": cont,
            "completo": {"negation_of": {k: resumo_alvo(y_true, y_pred)[k] for k in alvo},
                         "macro_f1": completo["macro_f1"], "mcc": completo["mcc"]},
            "sem_excluidos": {"negation_of": {k: resumo_alvo(sub_t, sub_p)[k] for k in alvo},
                              "macro_f1": sem["macro_f1"], "mcc": sem["mcc"]}}
