"""Tokenizer e modelo: carga, gravação, recarga e identidade.

* `carregar_tokenizer` — o tokenizer do checkpoint de pré-treino com os quatro
  marcadores de entidade (`entrada.MARCADORES`) acrescentados como tokens
  especiais.
* `carregar_classificador` — o encoder com a cabeça de classificação de
  sequência ([CLS] → 3 rótulos) e a matriz de embeddings redimensionada para o
  vocabulário com os marcadores. É o modelo do classificador da tarefa; uma
  estratégia com cabeça própria (a Pair-Aware) monta o seu.
* `salvar` / `recarregar` — gravação atômica no formato `save_pretrained` e
  recarga genérica por `type(modelo).from_pretrained`, sem conhecer a cabeça.
* `identidade` — o que identifica o par tokenizer + modelo de uma execução
  (vocabulário, configuração e número de parâmetros), conferido na retomada.

A ordem das chamadas importa para a reprodutibilidade: `from_pretrained`
inicializa a cabeça e `resize_token_embeddings` inicializa os embeddings dos
marcadores consumindo o gerador global do torch, depois de `fixar_sementes`.

O identificador do modelo é o do Hugging Face Hub (`config.ENCODERS`) ou o
caminho de uma pasta local com um checkpoint (por exemplo, o modelo minúsculo
das referências); nenhum caminho de máquina fica fixado aqui.

Origem no legado: o carregamento em `run()` de `src/relation_extraction.py`
(`AutoTokenizer`, `add_special_tokens`, `AutoModelForSequenceClassification`,
`resize_token_embeddings`), `save_best_model` e `load_best_state`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from reclin.entrada import MARCADORES
from reclin.tarefa import ID2LABEL, LABEL2ID, LABELS

log = logging.getLogger(__name__)


def carregar_tokenizer(modelo: str | Path) -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(modelo))
    acrescentados = tokenizer.add_special_tokens({"additional_special_tokens": list(MARCADORES)})
    log.info("Tokenizer %s: %d marcadores acrescentados, vocabulário de %d", modelo,
             acrescentados, len(tokenizer))
    return tokenizer


def carregar_classificador(modelo: str | Path, tokenizer: Any) -> Any:
    from transformers import AutoModelForSequenceClassification
    rede = AutoModelForSequenceClassification.from_pretrained(
        str(modelo), num_labels=len(LABELS), id2label=dict(ID2LABEL), label2id=dict(LABEL2ID))
    rede.resize_token_embeddings(len(tokenizer))
    return rede


def contar_parametros(modelo: Any) -> int:
    return int(sum(p.numel() for p in modelo.parameters()))


def salvar(pasta: str | Path, modelo: Any, tokenizer: Any) -> Path:
    """Grava modelo e tokenizer em `pasta` (formato `save_pretrained`) de forma
    atômica: escreve em `<pasta>.tmp` e só então substitui a anterior."""
    pasta = Path(pasta)
    tmp = pasta.with_name(pasta.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    modelo.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    if pasta.exists():
        shutil.rmtree(pasta)
    os.replace(tmp, pasta)
    return pasta


def recarregar(pasta: str | Path, classe: Any = None) -> tuple[Any, Any]:
    """Tokenizer e modelo gravados por `salvar`. `classe` é a do modelo
    (`type(modelo)`); sem ela, o classificador de sequência."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    classe = classe or AutoModelForSequenceClassification
    return AutoTokenizer.from_pretrained(str(pasta)), classe.from_pretrained(str(pasta))


def _sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def identidade(modelo_id: str | Path, tokenizer: Any, modelo: Any) -> dict[str, Any]:
    """O que identifica tokenizer e modelo de uma execução. Os hashes cobrem o
    vocabulário (com os marcadores) e a configuração da rede."""
    config = modelo.config.to_dict()
    for volatil in ("transformers_version", "_name_or_path", "name_or_path"):
        config.pop(volatil, None)
    return {
        "modelo": str(modelo_id),
        "classe": type(modelo).__name__,
        "tokenizer": type(tokenizer).__name__,
        "vocabulario": len(tokenizer),
        "vocabulario_sha256": _sha256(sorted(tokenizer.get_vocab().items())),
        "ids_marcadores": tokenizer.convert_tokens_to_ids(list(MARCADORES)),
        "config_sha256": _sha256(config),
        "n_parametros": contar_parametros(modelo),
    }
