#!/usr/bin/env python3
"""
Logging centralizado do pipeline RECLin-PT.

Uma unica funcao -- `get_logger(nome)` -- devolve um logger configurado para
escrever SIMULTANEAMENTE no terminal (stdout) e no arquivo `logs/pipeline.log`,
com o formato:

    [TIMESTAMP] [NIVEL] [MODULO] mensagem

Niveis usados no projeto: INFO (fluxo normal), WARNING (anomalias toleradas,
ex. erro de anotacao no SemClinBr) e ERROR (falhas que abortam a etapa).

O arquivo de log fica em <raiz-do-repo>/logs/pipeline.log, calculado a partir
da localizacao deste modulo (src/utils/logger.py -> parents[2] = raiz), de modo
que o destino e o mesmo independentemente do diretorio de trabalho. Em modo
'append': acumula as execucoes do pipeline numa trilha unica.

A configuracao e idempotente: chamar `get_logger` varias vezes com o mesmo nome
NAO duplica handlers (evita linhas repetidas no terminal).
"""
from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Raiz do repo: src/utils/logger.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG_FILE = _REPO_ROOT / "logs" / "pipeline.log"


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Devolve um logger com handlers de console e arquivo ja configurados.

    Parametros:
      name     -- aparece no campo [MODULO] do log (ex.: "baseline_biobertpt").
      level    -- nivel minimo capturado (default INFO).
      log_file -- caminho do arquivo de log (default <repo>/logs/pipeline.log).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Idempotencia: se ja tem handlers, nao adiciona de novo.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- console (stdout) ---
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # --- arquivo ---
    path = Path(log_file) if log_file is not None else _DEFAULT_LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Nao propaga para o root (evita duplicacao se o root tiver handler).
    logger.propagate = False
    return logger
