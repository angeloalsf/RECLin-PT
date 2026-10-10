"""Fixtures comuns.

Os testes ficam em três pastas:

* `unidade/` — comportamento de cada módulo sobre exemplos pequenos;
* `integracao/` — o treino de ponta a ponta em CPU com o modelo minúsculo
  (retomada exata, reprodutibilidade, separação entre treino e TEST);
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


MODELO_MINUSCULO = REFERENCIA / "modelo_minusculo"


@pytest.fixture(scope="session")
def subconjunto(tmp_path_factory):
    """Fábrica de pastas de partições pequenas: as primeiras `n` linhas de cada
    partição congelada (os mesmos bytes), com um MANIFEST próprio. Com
    `sem_test=True`, o arquivo do TEST é apagado depois de calculado o
    MANIFEST (para provar que uma operação não o lê)."""
    def criar(train: int, dev: int, test: int, *, sem_test: bool = False) -> Path:
        pasta = tmp_path_factory.mktemp(f"particoes_{train}_{dev}_{test}")
        for nome, n in (("train", train), ("dev", dev), ("test", test)):
            linhas = (caminhos.PARTICOES / f"{nome}.jsonl").read_bytes().splitlines(keepends=True)[:n]
            (pasta / f"{nome}.jsonl").write_bytes(b"".join(linhas))
        particoes.gravar_manifesto(pasta, particoes.calcular_manifesto(
            pasta, seed=42, origem={"path": "subconjunto das partições congeladas", "sha256": None}))
        if sem_test:
            (pasta / "test.jsonl").unlink()
        return pasta
    return criar
