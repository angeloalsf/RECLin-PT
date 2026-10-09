"""O repositório como pacote entregue: estrutura, importações e pontos de entrada.

Estes testes conferem que uma cópia do projeto (por exemplo, a extraída de um
ZIP de entrega numa pasta limpa) tem tudo o que é preciso para rodar o que já
está implementado:

* os arquivos obrigatórios da raiz e de `codigo/`;
* as referências do legado completas e íntegras (SHA-256 de `referencias.json`);
* todos os módulos de `reclin`, descobertos automaticamente, importam sem
  efeitos colaterais (nada de logging configurado, arquivos criados,
  `os.environ` ou `sys.path` alterados);
* todos os scripts de `codigo/scripts/` respondem a `--help`.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from reclin.util import caminhos
from reclin.util.io import ler_json, sha256_arquivo

OBRIGATORIOS_RAIZ = ("README.md", "LICENSE", "CITATION.cff", ".gitattributes", ".gitignore")
OBRIGATORIOS_CODIGO = (
    "README.md", "pyproject.toml", "requirements.txt",
    "dados/particoes/train.jsonl", "dados/particoes/dev.jsonl", "dados/particoes/test.jsonl",
    "dados/particoes/MANIFEST.json",
    "docs/entregas/README.md",
    "testes/referencia/README.md", "testes/referencia/dados.json",
    "testes/referencia/referencias.json",
)
REFERENCIA = caminhos.CODIGO / "testes" / "referencia"

IMPORTA_TUDO = """
import importlib, logging, os, pkgutil, sys
ambiente, caminho = dict(os.environ), list(sys.path)
import reclin
modulos = sorted(m.name for m in pkgutil.walk_packages(reclin.__path__, "reclin."))
for nome in modulos:
    importlib.import_module(nome)
assert not logging.getLogger().handlers, "logging raiz configurado no import"
assert not logging.getLogger("reclin").handlers, "logger reclin configurado no import"
assert dict(os.environ) == ambiente, "os.environ alterado no import"
assert sys.path == caminho, "sys.path alterado no import"
print(" ".join(modulos))
"""


@pytest.mark.parametrize("nome", OBRIGATORIOS_RAIZ)
def test_arquivos_obrigatorios_da_raiz(nome):
    assert (caminhos.RAIZ / nome).is_file(), nome


@pytest.mark.parametrize("nome", OBRIGATORIOS_CODIGO)
def test_arquivos_obrigatorios_de_codigo(nome):
    assert (caminhos.CODIGO / nome).is_file(), nome


def test_particoes_protegidas_contra_crlf():
    regras = (caminhos.RAIZ / ".gitattributes").read_text(encoding="utf-8").split()
    assert "*.jsonl" in regras and "eol=lf" in regras


def test_referencias_completas_e_integras():
    registrados = ler_json(REFERENCIA / "referencias.json")["arquivos_sha256"]
    assert registrados, "referencias.json não lista arquivos"
    for relativo, sha in registrados.items():
        arquivo = REFERENCIA / relativo
        assert arquivo.is_file(), f"referência ausente: {relativo}"
        assert sha256_arquivo(arquivo) == sha, f"referência alterada: {relativo}"


def test_todos_os_modulos_importam_sem_efeito_colateral(tmp_path):
    """Num interpretador novo e numa pasta vazia, para pegar efeitos que um
    import anterior no processo do pytest esconderia."""
    antes = sorted(p.name for p in caminhos.CODIGO.iterdir())
    r = subprocess.run([sys.executable, "-c", IMPORTA_TUDO], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    modulos = set(r.stdout.split())
    assert {"reclin.tarefa", "reclin.particoes", "reclin.config",
            "reclin.util.io", "reclin.util.log", "reclin.util.caminhos"} <= modulos
    assert list(tmp_path.iterdir()) == [], "o import criou arquivos"
    assert sorted(p.name for p in caminhos.CODIGO.iterdir()) == antes


@pytest.mark.parametrize("script", sorted(p.name for p in (caminhos.CODIGO / "scripts").glob("*.py")))
def test_scripts_respondem_a_ajuda(script):
    r = subprocess.run([sys.executable, str(caminhos.CODIGO / "scripts" / script), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()
