"""Testes de referência do legado (`testes/equivalencia/`): identificação e aviso.

Estes testes comparam o código novo com as referências geradas pelo legado
(`testes/referencia/`). A aprovação neles demonstra **compatibilidade técnica**
com a implementação anterior (portões das etapas 1 a 6), **não** desempenho
científico dos modelos atuais: os artefatos históricos que eles leem nunca
alimentam o inventário de execuções, as análises, a calibração, a seleção de
modelos nem os resultados novos (`reclin.execucao.inventario` os recusa).

Todo teste desta pasta recebe o marcador `referencia_legado` (para rodá-los à
parte: `-m referencia_legado`; para excluí-los: `-m "not referencia_legado"`),
e o resumo final do pytest separa a contagem deles com o aviso acima.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PASTA = Path(__file__).resolve().parent
MARCADOR = "referencia_legado"
AVISO = ("compatibilidade técnica com o legado (portões das etapas 1 a 6); "
         "não demonstram desempenho científico dos modelos atuais")


def _da_pasta(item) -> bool:
    try:
        Path(str(item.fspath)).resolve().relative_to(PASTA)
        return True
    except ValueError:
        return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _da_pasta(item):
            item.add_marker(getattr(pytest.mark, MARCADOR))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    contagem = {}
    for estado in ("passed", "failed", "skipped", "error"):
        relatorios = [r for r in terminalreporter.stats.get(estado, [])
                      if getattr(r, "when", "call") in ("call", "setup") and MARCADOR in getattr(r, "keywords", {})]
        contagem[estado] = len({r.nodeid for r in relatorios})
    if any(contagem.values()):
        terminalreporter.write_sep("-", "testes de referência do legado (testes/equivalencia/)")
        terminalreporter.write_line(f"{contagem['passed']} passaram, {contagem['failed']} falharam, "
                                    f"{contagem['skipped']} pulados, {contagem['error']} com erro: {AVISO}.")
