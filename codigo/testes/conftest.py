"""Fixtures comuns.

Os testes ficam em duas pastas:

* `unidade/` — comportamento de cada módulo sobre exemplos pequenos;
* `equivalencia/` — o código novo contra as referências do legado em
  `referencia/` (geradas por `referencia/gerar_referencias.py`). São os
  portões de verificação das etapas da migração.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reclin import particoes
from reclin.util import caminhos
from reclin.util.io import ler_json

REFERENCIA = Path(__file__).resolve().parent / "referencia"
RESULTADOS_LEGADO = REFERENCIA / "resultados_legado"


@pytest.fixture(scope="session")
def referencia_dados() -> dict:
    return ler_json(REFERENCIA / "dados.json")


@pytest.fixture(scope="session")
def documentos() -> dict:
    """As partições congeladas versionadas em `codigo/dados/particoes/`."""
    return particoes.ler_particoes(caminhos.PARTICOES)


@pytest.fixture(scope="session")
def manifesto() -> dict:
    return particoes.ler_manifesto(caminhos.PARTICOES)


@pytest.fixture(scope="session")
def legado():
    """Lê (com cache) um arquivo de `referencia/resultados_legado/`, pelo
    caminho relativo a `results/` do legado."""
    cache: dict[str, object] = {}

    def ler(relativo: str):
        if relativo not in cache:
            cache[relativo] = ler_json(RESULTADOS_LEGADO / relativo)
        return cache[relativo]
    return ler
