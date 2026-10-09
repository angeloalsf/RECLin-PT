"""Configuração do logging, chamada só pelos scripts.

Os módulos do pacote apenas obtêm seu logger com `logging.getLogger(__name__)`
(todos ficam sob o logger `reclin`) e nunca adicionam handlers. Quem decide
para onde as mensagens vão é o ponto de entrada, chamando `configurar` uma vez.

Origem no legado: `src/utils/logger.py`, que abria `logs/pipeline.log` no
momento do import. Esse efeito colateral obrigou o legado a criar os dois
módulos `_isolamento.py`, que desviavam os loggers antes de importar o núcleo;
aqui ele não existe, e os dois módulos deixam de ser necessários.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

FORMATO = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
FORMATO_DATA = "%Y-%m-%d %H:%M:%S"
RAIZ = "reclin"


def configurar(arquivo: str | Path | None = None, *, nivel: int = logging.INFO) -> logging.Logger:
    """Envia as mensagens do pacote para o terminal e, se pedido, para `arquivo`.

    Substitui handlers de uma configuração anterior, então chamar de novo (por
    exemplo, numa célula repetida do Colab) não duplica linhas. O arquivo é
    aberto em modo de acréscimo. Devolve o logger raiz do pacote.
    """
    logger = logging.getLogger(RAIZ)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(nivel)
    logger.propagate = False

    formatador = logging.Formatter(FORMATO, datefmt=FORMATO_DATA)
    terminal = logging.StreamHandler(sys.stdout)
    terminal.setFormatter(formatador)
    logger.addHandler(terminal)

    if arquivo is not None:
        arquivo = Path(arquivo)
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        em_arquivo = logging.FileHandler(arquivo, mode="a", encoding="utf-8")
        em_arquivo.setFormatter(formatador)
        logger.addHandler(em_arquivo)
    return logger
