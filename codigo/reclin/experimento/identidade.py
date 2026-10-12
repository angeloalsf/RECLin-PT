"""Identidade própria da infraestrutura compartilhada (componente `infraestrutura`).

A infraestrutura (`reclin/experimento/**/*.py` e `scripts/experimento.py`) não
pertence a nenhum componente de treino e está fora da identidade de código v1
(`util.identidade_codigo`): editá-la não invalida treinos nem checkpoints.

Isto não a deixa sem identidade: o hash de cada arquivo (bytes, CRLF→LF, com
docstrings e comentários, como nos demais componentes) é calculado aqui e
gravado em `experimento.json` (`codigo.infraestrutura`), de modo que cada
execução diz com que versão da infraestrutura foi registrada, sincronizada ou
publicada. É informação de proveniência: **nunca bloqueia** uma retomada.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from reclin.util import identidade_codigo as ic

COMPONENTE = "infraestrutura"
ALGORITMO = "sha256-manifesto-v1"          # a mesma regra de bytes e de agregação da identidade de código
PADROES = ("reclin/experimento/**/*.py", "scripts/experimento.py")


def arquivos(raiz: str | Path) -> list[str]:
    raiz = Path(raiz)
    achados = {p.relative_to(raiz).as_posix() for padrao in PADROES for p in raiz.glob(padrao) if p.is_file()}
    return sorted(r for r in achados if "__pycache__" not in r.split("/"))


def calcular(raiz: str | Path) -> dict[str, Any]:
    """O registro de identidade da infraestrutura em `raiz` (a pasta `codigo/`)."""
    raiz = Path(raiz)
    manifesto = {r: ic._sha((raiz / r).read_bytes()) for r in arquivos(raiz)}  # noqa: SLF001
    return {"componente": COMPONENTE, "algoritmo": ALGORITMO, "identidade": ic.identidade_de(manifesto),
            "n_arquivos": len(manifesto), "arquivos": manifesto}
