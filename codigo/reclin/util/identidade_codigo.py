"""Identidade do código que produz as execuções, sem Git (pendência D6).

A identidade é um SHA-256 determinístico calculado a partir do conteúdo dos
arquivos de código e de configuração que podem mudar o que uma execução grava
(predições, métricas, checkpoints, `config.json`, trilha):

* incluídos: `reclin/**/*.py`, `scripts/*.py` e `pyproject.toml` (caminhos
  relativos a `codigo/`);
* excluídos, porque só leem execuções e não mudam o que elas gravam: as
  análises e os relatórios (`reclin/analises/`, `reclin/relatorios/`,
  `reclin/execucao/inventario.py`, `reclin/util/formatacao.py`,
  `scripts/analisar.py`, `scripts/gerar_tcc.py`, `scripts/conferir_tcc.py`);
* excluídos também, por não pertencerem a nenhum componente de treino: a
  infraestrutura compartilhada de execução (`reclin/experimento/` e
  `scripts/experimento.py`). Ela só obtém o código, confere o ambiente,
  copia execuções e registra proveniência em `experimento.json`; não grava
  nada do que esta identidade protege (`config.json`, `treino.json`, `checkpoints/`,
  predições, métricas) e nenhum componente a importa. Tem identidade própria
  (`reclin.experimento.identidade`), registrada, mas que nunca bloqueia
  uma retomada: ver `docs/arquitetura/identidade-por-componente.md`, seção 11;
* nunca entram: testes, documentação, notebooks, `Makefile`, partições,
  resultados, relatórios gerados, checkpoints, pesos, caches (`__pycache__`,
  `*.pyc`) e arquivos temporários.

Algoritmo (`sha256-manifesto-v1`):

1. lista os arquivos incluídos, em ordem do caminho POSIX relativo a `codigo/`;
2. para cada um, lê os bytes, troca `\\r\\n` por `\\n` (a mesma identidade no
   Windows e no Linux, qualquer que seja a conversão de fim de linha do
   checkout) e calcula o SHA-256;
3. o manifesto é `{caminho: sha256}`; a identidade é o SHA-256 do JSON do
   manifesto com chaves ordenadas e separadores `(",", ":")`, em UTF-8.

Cada execução grava, no `config.json` (chave `codigo`), o algoritmo, a
identidade e o manifesto inteiro: assim dá para conferir a consistência do
próprio registro (a identidade recalculada do manifesto gravado) e listar os
arquivos que mudaram em relação ao código atual.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from reclin.util import caminhos

ALGORITMO = "sha256-manifesto-v1"
PADROES = ("reclin/**/*.py", "scripts/*.py", "pyproject.toml")
EXCLUIDOS_PREFIXOS = ("reclin/analises/", "reclin/relatorios/", "reclin/experimento/")
EXCLUIDOS = frozenset({"reclin/execucao/inventario.py", "reclin/util/formatacao.py",
                       "scripts/analisar.py", "scripts/gerar_tcc.py", "scripts/conferir_tcc.py",
                       "scripts/experimento.py"})


class ErroCodigo(ValueError):
    """A execução não pode receber artefatos do código atual (outra versão ou sem identidade)."""


def incluido(relativo: str) -> bool:
    if "__pycache__" in relativo.split("/") or relativo.endswith(".pyc"):
        return False
    return relativo not in EXCLUIDOS and not relativo.startswith(EXCLUIDOS_PREFIXOS)


def arquivos(raiz: str | Path = caminhos.CODIGO) -> list[str]:
    """Os caminhos (POSIX, relativos a `raiz`) que entram na identidade."""
    raiz = Path(raiz)
    vistos = set()
    for padrao in PADROES:
        for p in raiz.glob(padrao):
            if p.is_file():
                vistos.add(p.relative_to(raiz).as_posix())
    return sorted(r for r in vistos if incluido(r))


def _sha(conteudo: bytes) -> str:
    return hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest()


def identidade_de(manifesto: Mapping[str, str]) -> str:
    texto = json.dumps(dict(manifesto), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def calcular(raiz: str | Path = caminhos.CODIGO) -> dict[str, Any]:
    """O registro de identidade do código em `raiz` (sem cache)."""
    raiz = Path(raiz)
    manifesto = {r: _sha((raiz / r).read_bytes()) for r in arquivos(raiz)}
    return {"algoritmo": ALGORITMO, "identidade": identidade_de(manifesto), "n_arquivos": len(manifesto),
            "arquivos": manifesto}


@lru_cache(maxsize=1)
def _atual() -> str:
    return json.dumps(calcular())


def atual() -> dict[str, Any]:
    """O registro do código instalado (calculado uma vez por processo)."""
    return json.loads(_atual())


def conferir_registro(registro: Mapping[str, Any] | None) -> list[str]:
    """Problemas de consistência de um registro gravado (vazio se confere):
    algoritmo conhecido e identidade igual à recalculada do manifesto."""
    if not registro:
        return ["sem identidade do código"]
    problemas = []
    if registro.get("algoritmo") != ALGORITMO:
        problemas.append(f"algoritmo de identidade desconhecido: {registro.get('algoritmo')!r}")
    manifesto = registro.get("arquivos")
    if not isinstance(manifesto, dict) or identidade_de(manifesto) != registro.get("identidade"):
        problemas.append("a identidade gravada não confere com o manifesto gravado")
    return problemas


def diferencas(registro: Mapping[str, Any], referencia: Mapping[str, Any]) -> list[str]:
    """Arquivos diferentes entre dois manifestos (incluídos, removidos ou alterados)."""
    a, b = registro.get("arquivos") or {}, referencia.get("arquivos") or {}
    return sorted(r for r in set(a) | set(b) if a.get(r) != b.get(r))
