"""Trilha das avaliações do TEST: uma linha por avaliação, nunca apagada.

Cada vez que o TEST de uma execução é avaliado, uma linha é acrescentada com a
data, o hash da configuração e as métricas principais. Dois contadores:

* `eval_index` — quantas avaliações a trilha tem, contando esta;
* `eval_index_for_config` — quantas tiveram o mesmo `config_sha1`.

Um `eval_index_for_config` maior que 1 é o caso que a análise de
multiplicidade precisa declarar: a mesma configuração teve o TEST medido mais
de uma vez. A trilha é registro, não trava: não impede reavaliar.

O `config_sha1` usa a receita do legado (SHA-1 do JSON com chaves ordenadas,
12 primeiros caracteres), de modo que configurações iguais têm o mesmo hash
nas trilhas antigas e nas novas. As métricas são arredondadas a 6 casas.

Origem no legado: três implementações com campos e hashes diferentes
(`relation_extraction.append_test_eval_log`, `train_restrito.append_test_eval_log`
e `run_fase2_test.append_eval`). `ler` aceita as linhas das três.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from reclin.util.io import ler_jsonl


def sha1_config(config: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def ler(caminho: str | Path) -> list[dict[str, Any]]:
    caminho = Path(caminho)
    return list(ler_jsonl(caminho)) if caminho.exists() else []


def registrar(caminho: str | Path, *, execucao: str, model: str, seed: int | None,
              config_sha1: str, n_test: int, macro_f1: float, negation_of_f1: float,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Acrescenta uma avaliação do TEST à trilha e devolve a linha gravada."""
    caminho = Path(caminho)
    anteriores = ler(caminho)
    linha = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "eval_index": len(anteriores) + 1,
        "eval_index_for_config": 1 + sum(1 for a in anteriores if a.get("config_sha1") == config_sha1),
        "execucao": execucao, "model": model, "seed": seed, "config_sha1": config_sha1,
        "n_test": int(n_test),
        "test_macro_f1": round(float(macro_f1), 6),
        "test_negation_of_f1": round(float(negation_of_f1), 6),
    }
    if extra:
        linha.update(extra)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(linha, ensure_ascii=False) + "\n")
    return linha


def avaliacoes_da_config(caminho: str | Path, config_sha1: str) -> int:
    """Quantas vezes a configuração já teve o TEST avaliado nesta trilha."""
    return sum(1 for a in ler(caminho) if a.get("config_sha1") == config_sha1)
