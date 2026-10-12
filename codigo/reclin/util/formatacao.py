"""Formatação de números para texto: casas fixas, vírgula decimal e sinal.

Sem conteúdo de domínio e sem LaTeX (o que é próprio das tabelas do TCC fica
em `reclin.relatorios.formatacao`). O arredondamento é o do `format` do
Python (meio-par sobre o valor binário), o mesmo em qualquer plataforma.

Origem no legado (só a finalidade; nada é importado): `fmt`, `d`, `sd`,
`ptbr` e `signed`, repetidos em vários scripts de relatório.
"""
from __future__ import annotations

import math

TRACO = "—"


def _finito(valor: float | None) -> bool:
    return valor is not None and not (isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)))


def fmt(valor: float | None, casas: int = 4) -> str:
    """`valor` com `casas` casas e ponto decimal; travessão se ausente."""
    if not _finito(valor):
        return TRACO
    return f"{float(valor):.{casas}f}"


def ptbr(valor: float | None, casas: int = 4) -> str:
    """Como `fmt`, com vírgula decimal (ABNT): 0,8123."""
    return fmt(valor, casas).replace(".", ",")


def sinal(valor: float | None, casas: int = 4, *, virgula: bool = True) -> str:
    """Com sinal explícito (+0,0123 / −0,0123); zero sai sem sinal. O menos é
    o sinal tipográfico (U+2212)."""
    if not _finito(valor):
        return TRACO
    texto = fmt(abs(valor), casas)
    if virgula:
        texto = texto.replace(".", ",")
    if float(texto.replace(",", ".")) == 0:
        return texto
    return ("+" if valor > 0 else "−") + texto


def porcentagem(valor: float | None, casas: int = 1, *, virgula: bool = True) -> str:
    """Fração em porcentagem: 0,9467 → 94,7%."""
    if not _finito(valor):
        return TRACO
    texto = fmt(100 * float(valor), casas)
    return (texto.replace(".", ",") if virgula else texto) + "%"


def inteiro(valor: int | None, *, milhar: str = ".") -> str:
    """Inteiro com separador de milhar (152.686)."""
    if valor is None:
        return TRACO
    return f"{int(valor):,}".replace(",", milhar)
