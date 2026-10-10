"""Baseline: o classificador da tarefa, sem nada específico de negação.

Todos os pares candidatos do conjunto de referência, a janela marcada com
`[E1]`/`[/E1]`/`[E2]`/`[/E2]`, o encoder (BioBERTpt ou BERTimbau) com a cabeça
de classificação de sequência ([CLS] → 3 rótulos) e a configuração base
(`config.Config`: 3 épocas, lotes de 64, `lr` 2e-5, pesos `balanced` do TRAIN,
melhor época pelo macro-F1 do DEV). É a pergunta do TCC: os dois encoders
passam exatamente por este código, e só o checkpoint de pré-treino muda.

Não usa o léxico, não seleciona candidatos e não remapeia nada: os sidecars
têm o formato dos baselines do legado (`baseline_<encoder>_seed<N>`).

Origem no legado: `src/baseline_biobertpt.py`, `src/baseline_bertimbau.py` e
`run()` de `src/relation_extraction.py`.
"""
from __future__ import annotations

from typing import Any

from reclin.config import Config
from reclin.treino.montagem import Montagem

NOME = "baseline"


def nome_execucao(config: Config) -> str:
    return f"baseline_{config.encoder}_seed{config.seed}"


def montagem(config: Config | None = None, **_: Any) -> Montagem:
    """A montagem do baseline: a do classificador da tarefa, com este nome."""
    return Montagem(estrategia=NOME, classe_config=Config)


def montagem_de_registro(registro: dict[str, Any]) -> Montagem:
    """A montagem de uma execução gravada (para avaliar o TEST)."""
    return montagem()
