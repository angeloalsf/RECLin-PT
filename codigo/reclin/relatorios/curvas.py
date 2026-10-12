"""Curvas por época das execuções treinadas: o histórico do DEV gravado pelo
treino (`treino.json`, `dev_history`), com a época escolhida marcada.

Só o DEV: é o que o treino mede a cada época. A métrica de seleção de cada
execução é a gravada por ela (`campo_selecao` no `treino.json` dos estágios do
Two-Stage; `dev_macro_f1` nas demais), e a época marcada é a gravada em
`melhor_epoca` — esta rotina não reescolhe nada.

Origem no legado (só a finalidade): `make_tcc_curves.py`.
"""
from __future__ import annotations

from typing import Any, Sequence

from reclin.execucao import inventario as inv
from reclin.relatorios import fase
from reclin.relatorios.formatacao import Numero
from reclin.relatorios.tabelas import ErroTabela, Tabela

CAMPO_PADRAO = "dev_macro_f1"


def historico(ex: inv.Execucao) -> dict[str, Any]:
    """Épocas, perdas, métrica de seleção e época escolhida de uma execução."""
    if not ex.treino:
        raise ErroTabela(f"{ex.nome} não tem treino.json")
    campo = ex.treino.get("campo_selecao", CAMPO_PADRAO)
    linhas = ex.treino.get("dev_history") or []
    if not linhas or campo not in linhas[0]:
        raise ErroTabela(f"{ex.nome}: histórico sem o campo de seleção {campo!r}")
    return {"execucao": ex.nome, "campo_selecao": campo, "melhor_epoca": ex.treino.get("melhor_epoca"),
            "epocas": [int(h["epoch"]) for h in linhas],
            "train_loss": [h.get("train_loss") for h in linhas], "dev_loss": [h.get("dev_loss") for h in linhas],
            "selecao": [h[campo] for h in linhas],
            "dev_negation_of_f1": [h.get("dev_negation_of_f1") for h in linhas]}


def tabelas(inventario: inv.Inventario, *, natureza: str = "cientifica",
            treinadas: Sequence[tuple[str, str]] | None = None) -> tuple[list[Tabela], list[dict[str, Any]]]:
    """Uma tabela por execução treinada (pendente quando ausente) e os
    históricos, para as figuras."""
    saida, historicos = [], []
    for rotulo, nome in (treinadas or fase.treinadas()):
        chave = "curva_" + nome
        ex = inventario.execucoes.get(nome)
        colunas = ["Época", "Perda (TRAIN)", "Perda (DEV)", "Métrica de seleção (DEV)", "Escolhida"]
        if ex is None or ex.natureza != natureza or not ex.treino:
            motivo = (f"{nome} não encontrada" if ex is None else
                      f"{nome} é de natureza {ex.natureza!r}" if ex.natureza != natureza else f"{nome} sem treino")
            saida.append(Tabela(chave, f"Curva por época: {rotulo}", colunas, [], natureza, pendencias=[motivo]))
            continue
        if not ex.valida:
            raise ErroTabela(f"{nome} tem problemas de proveniência: " + "; ".join(ex.problemas))
        h = historico(ex)
        linhas = []
        for i, epoca in enumerate(h["epocas"]):
            k = f"curva.{nome}.epoca{epoca}"
            linhas.append([Numero(epoca, inteiro=True),
                           Numero(h["train_loss"][i], chave=f"{k}.train_loss"),
                           Numero(h["dev_loss"][i], chave=f"{k}.dev_loss"),
                           Numero(h["selecao"][i], chave=f"{k}.selecao"),
                           "sim" if epoca == h["melhor_epoca"] else ""])
        t = Tabela(chave, f"Curva por época: {rotulo} ({h['campo_selecao']})", colunas, linhas, natureza,
                   origens=[ex.origem()], notas=[f"Métrica de seleção: {h['campo_selecao']}; época escolhida: "
                                                f"{h['melhor_epoca']} (primeiro máximo)."])
        saida.append(t)
        historicos.append({"rotulo": rotulo, **h})
    return saida, historicos
