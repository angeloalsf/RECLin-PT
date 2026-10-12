"""Identificadores de configuração e de execução.

* `config_id` — 12 caracteres hexadecimais do SHA-256 do JSON canônico da
  **chave de configuração**: estratégia, configuração efetiva, configuração
  própria da estratégia, pesos de origem (repositório, commit e hash do
  manifesto), dados (versão e SHA-256 das partições) e identidade do código.
  Não entra caminho local nenhum (o `modelo` do registro de treino é um
  caminho e é removido), de modo que a mesma configuração tem o mesmo
  `config_id` na máquina local, no Colab e em sessões diferentes;
* `run_id` — `<estrategia>__<config_id>__t<tentativa>`: uma tentativa de uma
  configuração. Reiniciar do zero a mesma configuração cria a tentativa
  seguinte; retomar um checkpoint **não** cria tentativa;
* os nomes oficiais do protocolo (`baseline_biobertpt_seed42`, …) continuam
  sendo os nomes das pastas das execuções oficiais; a ligação com o `run_id` é
  feita pelo registro (`alias_oficial` em `experimento.json`). Execuções
  exploratórias usam o `run_id` como nome da pasta e nunca um nome oficial.

A parte de código da chave é, por ora, a identidade global do código
(`util.identidade_codigo`, algoritmo `sha256-manifesto-v1`), com o campo
`escopo` dizendo isso. Quando a identidade passar a ser por componente, o
`escopo` muda e, por construção, o `config_id` também (nunca coincide por acidente).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

COMPRIMENTO_ID = 12
PADRAO_NOME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")
PADRAO_ESTRATEGIA = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
PADRAO_ID = re.compile(r"^[0-9a-f]{12}$")
PADRAO_RUN = re.compile(r"^([a-z][a-z0-9_]{0,40})__([0-9a-f]{12})__t([1-9][0-9]*)$")
CAMPOS_DA_CHAVE = ("estrategia", "config", "estrategia_config", "pesos", "dados", "codigo")
CAMINHO_ABSOLUTO = re.compile(r"^(/|[A-Za-z]:[\\/]|\\\\)")


class ErroId(ValueError):
    """Nome ou chave inválidos."""


def validar_nome(nome: str) -> str:
    """Nome seguro para pasta de execução (sem separadores, `..` nem espaços)."""
    if not isinstance(nome, str) or not PADRAO_NOME.fullmatch(nome) or ".." in nome:
        raise ErroId(f"nome de execução inválido: {nome!r} (use letras, números, '_', '-', '.', '@')")
    return nome


def canonico(objeto: Any) -> str:
    return json.dumps(objeto, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _caminhos_absolutos(valor: Any, onde: str = "") -> list[str]:
    achados: list[str] = []
    if isinstance(valor, str) and CAMINHO_ABSOLUTO.match(valor):
        achados.append(onde or "<raiz>")
    elif isinstance(valor, Mapping):
        for k, v in valor.items():
            achados += _caminhos_absolutos(v, f"{onde}.{k}" if onde else str(k))
    elif isinstance(valor, (list, tuple)):
        for i, v in enumerate(valor):
            achados += _caminhos_absolutos(v, f"{onde}[{i}]")
    return achados


def chave_configuracao(*, estrategia: str, config: Mapping[str, Any], estrategia_config: Mapping[str, Any] | None,
                       pesos: Mapping[str, Any], dados: Mapping[str, Any], codigo: Mapping[str, Any]) -> dict[str, Any]:
    """Monta e valida a chave de configuração.

    `config` é a configuração efetiva (`Config.como_dict()` do registro de
    treino); `pesos` é a proveniência (`repositorio`, `commit`, `hash_manifesto`);
    `dados` traz a versão e o SHA-256 das partições; `codigo` traz `escopo`,
    `algoritmo` e `identidade`. Falha se faltar um campo obrigatório ou se
    qualquer valor for um caminho absoluto."""
    if not PADRAO_ESTRATEGIA.fullmatch(estrategia or ""):
        raise ErroId(f"estratégia inválida: {estrategia!r}")
    exigidos = {"pesos": ("repositorio", "commit", "hash_manifesto"), "dados": ("versao", "particoes"),
                "codigo": ("escopo", "algoritmo", "identidade")}
    for campo, chaves in exigidos.items():
        valor = {"pesos": pesos, "dados": dados, "codigo": codigo}[campo]
        faltam = [c for c in chaves if not valor.get(c)]
        if faltam:
            raise ErroId(f"chave de configuração sem {campo}.{', '.join(faltam)}")
    limpa = {k: v for k, v in dict(config).items() if k != "modelo"}
    chave = {"estrategia": estrategia, "config": limpa,
             "estrategia_config": dict(estrategia_config) if estrategia_config else {},
             "pesos": {k: pesos[k] for k in exigidos["pesos"]},
             "dados": dict(dados), "codigo": {k: codigo[k] for k in exigidos["codigo"]}}
    absolutos = _caminhos_absolutos(chave)
    if absolutos:
        raise ErroId("a chave de configuração não pode conter caminhos absolutos: " + ", ".join(absolutos))
    return chave


def config_id(chave: Mapping[str, Any]) -> str:
    faltam = [c for c in CAMPOS_DA_CHAVE if c not in chave]
    if faltam:
        raise ErroId(f"chave de configuração incompleta (faltam {faltam}); use `chave_configuracao`")
    return hashlib.sha256(canonico(chave).encode("utf-8")).hexdigest()[:COMPRIMENTO_ID]


def run_id(estrategia: str, cid: str, tentativa: int) -> str:
    if not PADRAO_ESTRATEGIA.fullmatch(estrategia or ""):
        raise ErroId(f"estratégia inválida: {estrategia!r}")
    if not PADRAO_ID.fullmatch(cid or ""):
        raise ErroId(f"config_id inválido: {cid!r}")
    if not isinstance(tentativa, int) or tentativa < 1:
        raise ErroId(f"tentativa inválida: {tentativa!r} (inteiro ≥ 1)")
    return f"{estrategia}__{cid}__t{tentativa}"


def decompor_run_id(run: str) -> dict[str, Any]:
    m = PADRAO_RUN.fullmatch(run or "")
    if not m:
        raise ErroId(f"run_id inválido: {run!r}")
    return {"estrategia": m.group(1), "config_id": m.group(2), "tentativa": int(m.group(3))}


def proxima_tentativa(existentes: Iterable[str], estrategia: str, cid: str) -> int:
    """1 + a maior tentativa já registrada (em qualquer estado) dessa configuração.
    `existentes` são nomes/`run_id`s; os que não seguem o padrão são ignorados."""
    maior = 0
    for nome in existentes:
        m = PADRAO_RUN.fullmatch(nome or "")
        if m and m.group(1) == estrategia and m.group(2) == cid:
            maior = max(maior, int(m.group(3)))
    return maior + 1
