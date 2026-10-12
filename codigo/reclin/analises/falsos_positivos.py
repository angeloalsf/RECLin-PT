"""Auditoria dos falsos positivos de `negation_of` de uma execução.

Um falso positivo é um par predito `negation_of` cujo rótulo anotado é outro.
Para cada um: a origem (`associated_with` ou `no_relation` anotado), a
distância entre as entidades, a forma normalizada do e1, o tipo do e2, se o e1
é pista (quando quem chama fornece `e_pista`), o alinhamento dos offsets e um
trecho do texto. O resumo agrega por origem, por pista, por faixa de distância
e pelas formas de e1 mais frequentes.

As predições vêm de uma execução do inventário (`execucao.inventario`), já
conferidas: os candidatos são os do conjunto de referência da partição, na
mesma ordem do `y_true` do sidecar (conferido aqui também). Por padrão, o
DEV; o TEST só quando a execução teve o TEST avaliado pelo comando próprio
(isso é análise descritiva de uma avaliação já feita, não uma nova).

Origem no legado (só a finalidade): `audit_residual_fp.py`.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from reclin.analises.alinhamento import alinhar_entidade
from reclin.analises.corpus import faixa, rotulos_faixas
from reclin.tarefa import ASSOC, ID2LABEL, NEG, NOREL, Candidato, Documento, entity_gap, y_true

CONTEXTO = 60


def _normalizar(texto: str) -> str:
    return " ".join(texto.lower().split())


def _trecho(texto: str, c: Candidato, contexto: int = CONTEXTO) -> str:
    ini = max(0, min(c["e1"]["start"], c["e2"]["start"]) - contexto)
    fim = min(len(texto), max(c["e1"]["end"], c["e2"]["end"]) + contexto)
    return " ".join(texto[ini:fim].split())


def auditar(candidatos: Sequence[Candidato], sidecar: Mapping[str, Any],
            documentos: Mapping[str, Documento], *,
            e_pista: Callable[[dict], bool] | None = None, top: int = 20,
            exemplos: int = 50) -> dict[str, Any]:
    """Falsos positivos de `negation_of` de `sidecar` sobre `candidatos`
    (`documentos`: doc_id → documento, para o texto)."""
    if sidecar["y_true"] != y_true(candidatos):
        raise ValueError("o y_true do sidecar não é o dos candidatos: as predições são de outro conjunto")
    fps = []
    for c, yt, yp in zip(candidatos, sidecar["y_true"], sidecar["y_pred"]):
        if yp != NEG or yt == NEG:
            continue
        texto = documentos[c["doc_id"]]["text"]
        gap = entity_gap(c["e1"], c["e2"])
        fps.append({
            "doc_id": c["doc_id"], "e1": c["e1"]["text"], "e2": c["e2"]["text"],
            "forma_e1": _normalizar(c["e1"]["text"]), "tipo_e2": c["e2"]["type"],
            "origem": ID2LABEL[yt], "gap": gap, "faixa_gap": faixa(gap),
            "e1_pista": e_pista(c["e1"]) if e_pista else None,
            "alinhamento": [alinhar_entidade(texto, c["e1"])["status"], alinhar_entidade(texto, c["e2"])["status"]],
            "trecho": _trecho(texto, c)})
    tp = sum(1 for yt, yp in zip(sidecar["y_true"], sidecar["y_pred"]) if yt == yp == NEG)
    resumo = {
        "n_candidatos": len(candidatos), "tp": tp, "fp": len(fps),
        "por_origem": {ID2LABEL[r]: sum(1 for f in fps if f["origem"] == ID2LABEL[r]) for r in (ASSOC, NOREL)},
        "por_faixa_gap": {f: sum(1 for x in fps if x["faixa_gap"] == f) for f in rotulos_faixas()},
        "formas_e1": Counter(f["forma_e1"] for f in fps).most_common(top),
        "tipos_e2": Counter(f["tipo_e2"] for f in fps).most_common(top),
        "alinhamento_nao_exato": sum(1 for f in fps if f["alinhamento"] != ["exato", "exato"]),
    }
    if e_pista is not None:
        resumo["e1_pista"] = {"sim": sum(1 for f in fps if f["e1_pista"]),
                              "nao": sum(1 for f in fps if not f["e1_pista"])}
    return {"resumo": resumo, "exemplos": fps[:exemplos]}
