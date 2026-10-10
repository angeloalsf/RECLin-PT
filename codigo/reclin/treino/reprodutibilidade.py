"""Reprodutibilidade: ambiente, sementes, geradores aleatórios e registro.

O que é fixado e por quê:

* `configurar_ambiente` — `CUBLAS_WORKSPACE_CONFIG=":4096:8"` (exigido pelo
  cuBLAS para GEMM determinística em GPU; precisa estar no ambiente antes de o
  contexto CUDA ser criado) e `PYTHONHASHSEED=0` (só vale para processos
  filhos: o do próprio processo é decidido na partida do interpretador). Usa
  `setdefault`: um valor exportado no shell prevalece. É chamada pelos
  scripts, nunca no import de um módulo.
* `fixar_sementes` — python, numpy, torch e CUDA com a mesma semente, cuDNN
  determinístico e `torch.use_deterministic_algorithms(True, warn_only=True)`:
  uma operação sem versão determinística vira aviso (que `coletar_avisos`
  registra), em vez de derrubar um treino de horas.
* `estados_rng` / `restaurar_rng` — os estados dos geradores globais, que o
  checkpoint guarda para que a retomada continue exatamente de onde parou.
* `ambiente` — versões, dispositivo, threads e variáveis relevantes, gravados
  com cada execução.

Garantia: no mesmo ambiente (mesmas versões, mesmo dispositivo, mesmo número de
threads), a mesma configuração e a mesma semente produzem os mesmos números,
bit a bit — conferido nos testes em CPU. Entre máquinas, sistemas, versões de
biblioteca ou entre CPU e GPU não há garantia de igualdade bit a bit.

Origem no legado: `set_all_seeds`, `collect_environment` e as variáveis no topo
de `src/relation_extraction.py`; `environment_extra` de
`src/finetuning_restrito/train_restrito.py`; `capturar_avisos` dos
`_isolamento.py`.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import platform
import random
import sys
import warnings
from typing import Any, Iterator

import numpy as np
import torch

VARIAVEIS_AMBIENTE = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
VARIAVEIS_REGISTRADAS = ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                         "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")


def configurar_ambiente() -> dict[str, str]:
    """Define as variáveis de `VARIAVEIS_AMBIENTE` que ainda não existirem e
    devolve os valores em vigor."""
    for nome, valor in VARIAVEIS_AMBIENTE.items():
        os.environ.setdefault(nome, valor)
    return {nome: os.environ[nome] for nome in VARIAVEIS_AMBIENTE}


def fixar_sementes(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def estados_rng() -> dict[str, Any]:
    """Estados dos geradores globais (python, numpy, torch e, se houver, CUDA)."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restaurar_rng(estados: dict[str, Any]) -> None:
    """Restaura os estados de `estados_rng`. Falha se o checkpoint tem estado de
    CUDA e esta máquina não tem GPU (ou o contrário): a retomada não seria a
    mesma execução."""
    random.setstate(estados["python"])
    np.random.set_state(estados["numpy"])
    torch.set_rng_state(estados["torch"])
    tem_cuda = torch.cuda.is_available()
    if (estados["cuda"] is not None) != tem_cuda:
        raise RuntimeError("o checkpoint foi gravado " + ("com" if estados["cuda"] is not None else "sem")
                           + " GPU e esta máquina está " + ("com" if tem_cuda else "sem")
                           + " GPU: a retomada não reproduziria a execução")
    if tem_cuda:
        torch.cuda.set_rng_state_all(estados["cuda"])


def _versao(modulo: str) -> str | None:
    try:
        return getattr(importlib.import_module(modulo), "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def ambiente() -> dict[str, Any]:
    """Versões e condições da execução. Tudo é melhor-esforço: o que não puder
    ser lido vira None."""
    def seguro(funcao):
        try:
            return funcao()
        except Exception:  # noqa: BLE001
            return None

    return {
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "versoes": {m: _versao(m) for m in ("torch", "transformers", "tokenizers", "safetensors",
                                            "huggingface_hub", "numpy", "scipy")},
        "cuda": seguro(lambda: torch.version.cuda),
        "cudnn": seguro(torch.backends.cudnn.version),
        "gpu": seguro(lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "threads_torch": torch.get_num_threads(),
        "pythonhashseed_valeu_no_processo": sys.flags.hash_randomization == 0,
        "variaveis": {v: os.environ.get(v) for v in VARIAVEIS_REGISTRADAS},
    }


@contextlib.contextmanager
def coletar_avisos() -> Iterator[list[str]]:
    """Registra as mensagens de aviso (uma vez cada) emitidas dentro do bloco —
    em especial as de operações sem versão determinística."""
    mensagens: list[str] = []
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("default")
        try:
            yield mensagens
        finally:
            for aviso in capturados:
                texto = f"{aviso.category.__name__}: {aviso.message}"
                if texto not in mensagens:
                    mensagens.append(texto)
