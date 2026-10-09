"""O sidecar de predições: um rótulo predito para cada candidato do conjunto.

Formato (o do legado, mantido para que as predições antigas e as novas sejam
lidas e comparadas da mesma forma):

    {"model": "pucpr/biobertpt-all", "seed": 42,
     "labels": ["negation_of", "associated_with", "no_relation"],
     "y_true": [...], "y_pred": [...],
     ...campos extras (split, best_epoch, ...),
     "probs": [[p_neg, p_assoc, p_norel], ...]}

`y_true` e `y_pred` são ids na ordem de `labels`, na ordem dos candidatos do
conjunto de referência. `probs` é opcional, arredondado a 4 casas. `seed` pode
ser `null` (a regra pura não tem semente). Gravado em JSON compacto, sem
quebra de linha final.

Origem no legado: `build_preds_payload` e `round_probs` de
`src/relation_extraction.py` (mesma ordem de chaves), `load_preds` de
`src/significance.py` e `conferir_y_true` de `restricted_space.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from reclin.tarefa import LABELS, ConjuntoReferencia
from reclin.util.io import gravar_texto, ler_json, sha256_json

CHAVES = ("model", "seed", "labels", "y_true", "y_pred")
CASAS_PROBS = 4


def arredondar_probs(probs: Iterable[Sequence[float]], casas: int = CASAS_PROBS) -> list[list[float]]:
    """Com 4 casas, cada sidecar do TEST fica em ~0,5 MB em vez de >10 MB; as
    linhas deixam de somar exatamente 1."""
    return [[round(float(p), casas) for p in linha] for linha in probs]


def montar(*, model: str, seed: int | None, y_true: Sequence[int], y_pred: Sequence[int],
           probs: Iterable[Sequence[float]] | None = None,
           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Monta um sidecar: as cinco chaves, os extras e, por último, `probs`."""
    predicoes: dict[str, Any] = {
        "model": model, "seed": seed, "labels": list(LABELS),
        "y_true": [int(i) for i in y_true], "y_pred": [int(i) for i in y_pred]}
    if extra:
        predicoes.update(extra)
    if probs is not None:
        predicoes["probs"] = arredondar_probs(probs)
    validar(predicoes)
    return predicoes


def validar(predicoes: dict[str, Any]) -> None:
    """Falha se o sidecar não tiver o formato esperado."""
    faltando = [c for c in CHAVES if c not in predicoes]
    if faltando:
        raise ValueError(f"sidecar sem as chaves {faltando}")
    if list(predicoes["labels"]) != list(LABELS):
        raise ValueError(f"rótulos {predicoes['labels']} diferentes de {list(LABELS)}")
    n = len(predicoes["y_true"])
    if len(predicoes["y_pred"]) != n:
        raise ValueError(f"y_true tem {n} elementos e y_pred {len(predicoes['y_pred'])}")
    ids = range(len(LABELS))
    for campo in ("y_true", "y_pred"):
        fora = [v for v in predicoes[campo] if v not in ids]
        if fora:
            raise ValueError(f"{campo} com ids fora de 0..{len(LABELS) - 1}: {fora[:5]}")
    if "probs" in predicoes:
        probs = predicoes["probs"]
        if len(probs) != n or any(len(linha) != len(LABELS) for linha in probs):
            raise ValueError("probs não tem uma linha de 3 valores por candidato")


def ler(caminho: str | Path) -> dict[str, Any]:
    predicoes = ler_json(caminho)
    validar(predicoes)
    return predicoes


def gravar(caminho: str | Path, predicoes: dict[str, Any]) -> None:
    validar(predicoes)
    gravar_texto(caminho, json.dumps(predicoes, ensure_ascii=False))


def conferir_conjunto(predicoes: dict[str, Any], conjunto: ConjuntoReferencia) -> list[str]:
    """Divergências entre o sidecar e o conjunto de referência (vazia se
    confere): mesmo número de candidatos e o mesmo `y_true`, elemento a
    elemento (pelo hash de `tarefa.conjunto_referencia`)."""
    divergencias = []
    if len(predicoes["y_true"]) != conjunto["n_candidatos"]:
        divergencias.append(f"{len(predicoes['y_true'])} predições para "
                            f"{conjunto['n_candidatos']} candidatos de {conjunto['particao']}")
    elif sha256_json(predicoes["y_true"]) != conjunto["y_true_sha256"]:
        divergencias.append(f"y_true diferente do conjunto de referência de {conjunto['particao']}")
    return divergencias


def mesmo_conjunto(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """As duas predições se referem ao mesmo conjunto e podem ser pareadas."""
    return a["labels"] == b["labels"] and a["y_true"] == b["y_true"]
