"""Formatação das tabelas: o modelo de célula e a saída em LaTeX, Markdown e CSV.

Uma tabela é uma lista de linhas de células. Uma célula é texto, um `Numero`
(valor, casas, chave no registro de valores) ou um dos marcadores `PENDENTE`
(o sistema ou a partição ainda não foi executado ou avaliado) e
`NAO_COMPARAVEL` (a métrica não se compara entre os sistemas, como o macro-F1
do restrito). Os marcadores nunca viram número.

LaTeX: vírgula decimal (ABNT), `booktabs`, cabeçalho de comentário com a
origem e o aviso de arquivo gerado. CSV: ponto decimal e valores completos
(sem arredondar), para reuso. Markdown: vírgula decimal, para o relatório.

Origem no legado (só a finalidade): `abnt_float`, `write_tex`, `ptbr`, `tt` e
os rótulos de `scripts/_artifacts.py`.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any, Sequence

from reclin.util.formatacao import fmt, inteiro, ptbr, sinal

PENDENTE = object()
NAO_COMPARAVEL = object()
TEXTO_PENDENTE = "pendente"
TEXTO_NAO_COMPARAVEL = "n/c"

_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


@dataclass(frozen=True)
class Numero:
    valor: float | int
    casas: int = 4
    chave: str | None = None
    com_sinal: bool = False
    inteiro: bool = False

    def texto(self, *, virgula: bool) -> str:
        if self.inteiro:
            return inteiro(int(self.valor), milhar="." if virgula else ",")
        if self.com_sinal:
            return sinal(self.valor, self.casas, virgula=virgula)
        return ptbr(self.valor, self.casas) if virgula else fmt(self.valor, self.casas)


def escapar_latex(texto: str) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in texto)


def tt(texto: str) -> str:
    return r"\texttt{" + escapar_latex(texto) + "}"


def texto_celula(celula: Any, *, virgula: bool) -> str:
    if celula is PENDENTE:
        return TEXTO_PENDENTE
    if celula is NAO_COMPARAVEL:
        return TEXTO_NAO_COMPARAVEL
    if isinstance(celula, Numero):
        return celula.texto(virgula=virgula)
    if celula is None:
        return "—"
    return str(celula)


def _latex_celula(celula: Any) -> str:
    if celula is PENDENTE:
        return r"\textit{" + TEXTO_PENDENTE + "}"
    if celula is NAO_COMPARAVEL:
        return TEXTO_NAO_COMPARAVEL
    if isinstance(celula, Numero):
        return celula.texto(virgula=True).replace("−", "$-$")
    return escapar_latex(texto_celula(celula, virgula=True))


def latex(colunas: Sequence[str], linhas: Sequence[Sequence[Any]], *, alinhamento: str | None = None,
          comentarios: Sequence[str] = ()) -> str:
    """Um `tabular` com `booktabs` (sem `table`, legenda nem rótulo: esses
    ficam no texto do TCC)."""
    alinhamento = alinhamento or ("l" + "r" * (len(colunas) - 1))
    partes = [f"% {c}" for c in comentarios]
    partes.append(r"\begin{tabular}{" + alinhamento + "}")
    partes.append(r"\toprule")
    partes.append(" & ".join(escapar_latex(c) for c in colunas) + r" \\")
    partes.append(r"\midrule")
    for linha in linhas:
        partes.append(" & ".join(_latex_celula(c) for c in linha) + r" \\")
    partes.append(r"\bottomrule")
    partes.append(r"\end{tabular}")
    return "\n".join(partes) + "\n"


def markdown(colunas: Sequence[str], linhas: Sequence[Sequence[Any]]) -> str:
    def limpo(texto: str) -> str:
        return texto.replace("|", "\\|")
    partes = ["| " + " | ".join(limpo(c) for c in colunas) + " |",
              "| " + " | ".join(["---"] * len(colunas)) + " |"]
    for linha in linhas:
        partes.append("| " + " | ".join(limpo(texto_celula(c, virgula=True)) for c in linha) + " |")
    return "\n".join(partes) + "\n"


def csv_texto(colunas: Sequence[str], linhas: Sequence[Sequence[Any]]) -> str:
    saida = io.StringIO()
    escritor = csv.writer(saida, lineterminator="\n")
    escritor.writerow(colunas)
    for linha in linhas:
        escritor.writerow([repr(float(c.valor)) if isinstance(c, Numero) and not c.inteiro
                           else (int(c.valor) if isinstance(c, Numero) else texto_celula(c, virgula=False))
                           for c in linha])
    return saida.getvalue()
