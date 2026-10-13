"""Comparação das estratégias no DEV, sobre o MESMO universo de candidatos.

Só lê `predicoes_dev.json` (sidecars já gravados) e calcula as métricas com `reclin.avaliacao.metricas.avaliar`;
não treina, não faz inferência, não baixa nada e **nunca lê o TEST** (a partição está fixa em `"dev"`).

Por que existe: o Baseline pontua os 19.064 candidatos do DEV; o Two-Stage compõe os dois estágios sobre o mesmo
DEV; o restrito treina (e seleciona pares) só onde `e1` é pista do léxico, mas **remapeia o resto para
`no_relation`** e grava `y_pred` no DEV completo. Logo os três sidecars têm o mesmo `y_true` e podem ser
comparados — desde que isso seja conferido, em vez de suposto. `comparar` recusa (e diz por quê) qualquer
sidecar cujo `y_true` ou `labels` difira do primeiro; nada é calculado sobre universos diferentes.
Métricas restritas ao subespaço treinado (ex.: `dev_restrito_*` do `treino.json`) NÃO entram aqui.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from reclin.avaliacao import metricas
from reclin.execucao import diretorio, predicoes
from reclin.tarefa import LABELS

PARTICAO = "dev"            # fixa: este módulo não compara o TEST
FOCO = ("negation_of", "associated_with", "no_relation")


class ErroComparacao(ValueError):
    """Sidecars que não podem ser comparados (universos diferentes)."""


def localizar(raizes: list[str | Path], nome: str) -> Path | None:
    """O `predicoes_dev.json` de `nome` na primeira raiz que o tiver (None se nenhuma)."""
    for raiz in raizes:
        try:
            return diretorio.caminho_predicoes(raiz, nome, PARTICAO)
        except FileNotFoundError:
            continue
    return None


def linha(nome: str, sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Métricas de uma estratégia no DEV completo, numa linha achatada."""
    m = metricas.avaliar(sidecar["y_true"], sidecar["y_pred"])
    saida: dict[str, Any] = {"execucao": nome, "n": m["n"], "macro_f1": m["macro_f1"],
                             "acuracia": m["classification_report"]["accuracy"], "mcc": m["mcc"]}
    for rotulo in LABELS:
        c = m["por_classe"][rotulo]
        saida.update({f"{rotulo}.precisao": c["precision"], f"{rotulo}.recall": c["recall"],
                      f"{rotulo}.f1": c["f1"], f"{rotulo}.suporte": c["support"]})
    saida["matriz_confusao"] = m["confusion_matrix"]["matrix"]
    return saida


def comparar(sidecars: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compara sidecars do DEV. `sidecars` mapeia nome → conteúdo de `predicoes_dev.json`.

    Levanta `ErroComparacao` se algum não for do mesmo conjunto que o primeiro (mesmos `labels` e `y_true`)."""
    if not sidecars:
        raise ErroComparacao("nenhuma predição para comparar")
    nomes = list(sidecars)
    referencia = sidecars[nomes[0]]
    for nome in nomes:
        predicoes.validar(dict(sidecars[nome]))
        if not predicoes.mesmo_conjunto(dict(referencia), dict(sidecars[nome])):
            raise ErroComparacao(f"{nome} não está sobre o mesmo conjunto de candidatos que {nomes[0]} "
                                 "(y_true/labels diferem): as métricas não são comparáveis")
    return {"particao": PARTICAO, "n": len(referencia["y_true"]), "labels": list(LABELS),
            "linhas": [linha(n, sidecars[n]) for n in nomes]}


def carregar_e_comparar(raizes: list[str | Path], nomes: list[str]) -> dict[str, Any]:
    """Procura cada nome nas raízes, compara os encontrados e lista os que não achou (sem inventar valores)."""
    achados, ausentes = {}, []
    for nome in nomes:
        caminho = localizar(raizes, nome)
        if caminho is None:
            ausentes.append(nome)
        else:
            achados[nome] = predicoes.ler(caminho)
    resultado = comparar(achados) if achados else {"particao": PARTICAO, "n": 0, "labels": list(LABELS), "linhas": []}
    resultado["ausentes"] = ausentes
    return resultado


def formatar(resultado: Mapping[str, Any]) -> str:
    """Tabela em texto: macro-F1, acurácia e P/R/F1/suporte de cada classe, mais as matrizes de confusão."""
    cab = ["execucao", "macro_f1", "acuracia"] + [f"{r}.{c}" for r in FOCO for c in ("precisao", "recall", "f1")]
    linhas = [" | ".join(cab)]
    for l in resultado["linhas"]:
        linhas.append(" | ".join([l["execucao"]] + [f"{l[c]:.4f}" for c in cab[1:]]))
    linhas.append("suporte (DEV, n=%d): " % resultado["n"] + ", ".join(
        f"{r}={int(resultado['linhas'][0][f'{r}.suporte'])}" for r in LABELS) if resultado["linhas"] else "sem linhas")
    for l in resultado["linhas"]:
        linhas.append(f"matriz de confusão de {l['execucao']} (linhas = verdadeiro, colunas = predito; ordem {list(LABELS)}):")
        linhas.extend("  " + str(lin) for lin in l["matriz_confusao"])
    if resultado.get("ausentes"):
        linhas.append("NÃO ENCONTRADAS (sem métricas, nada foi inventado): " + ", ".join(resultado["ausentes"]))
    return "\n".join(linhas)
