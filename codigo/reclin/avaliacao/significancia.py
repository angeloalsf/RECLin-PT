"""Significância estatística entre dois sistemas sobre o mesmo conjunto de teste.

Responde se a diferença observada entre dois sistemas é real ou pode ser ruído
da amostra. Usa as predições já gravadas: não treina nada.

Dois testes complementares:

1. **McNemar exato.** Olha só os exemplos em que os sistemas discordam no
   acerto: b = A acertou e B errou; c = A errou e B acertou. Sob a hipótese
   nula (mesma taxa de acerto), b ~ Binomial(b + c, 0,5); o p-valor é o do
   teste binomial bilateral exato (`scipy.stats.binomtest`). Mede se os padrões
   de erro globais diferem.
2. **Bootstrap pareado no F1 da classe-alvo** (`negation_of`). Reamostra os
   exemplos com reposição `n_boot` vezes e, em cada reamostra, recalcula
   F1_A − F1_B. Reporta a diferença observada e a média, o intervalo de 95%
   (percentis 2,5 e 97,5) e um p-valor bilateral (2 × a fração de reamostras
   com sinal contrário ao observado, limitado a 1).

O pareamento só é legítimo se as duas predições se referem ao mesmo conjunto:
mesmos rótulos e mesmo `y_true`, elemento a elemento. `comparar` falha se não.

Reprodutibilidade: o bootstrap usa `numpy.random.default_rng(seed)` e sorteia
os índices de cada reamostra com uma chamada `integers(0, n, n)` por iteração,
na mesma ordem do legado. Mudar isso (por exemplo, sortear tudo de uma vez)
muda a sequência e os números. A semente de cada comparação do TCC segue a
regra de `protocolo.semente_bootstrap`.

O relatório (`relatorio_json`) tem as chaves e a serialização do legado
(`json.dumps` com `indent=2`, chaves ordenadas, sem quebra de linha final):
`testes/equivalencia/test_comparacoes_legado.py` confere que as 26 comparações
gravadas no legado saem idênticas byte a byte.

Origem no legado: `src/significance.py`, chamado por `Makefile`, `run.sh`,
`scripts/run_fase2_significance.py` (por subprocesso) e três
`rodar_significance` (por `runpy`, trocando `sys.argv`). Aqui são funções.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reclin.util.io import gravar_texto

ALVO = "negation_of"
N_BOOT = 10_000


def f1_classe(y_true: np.ndarray, y_pred: np.ndarray, classe: int) -> float:
    """F1 de uma classe, vetorizado: 2·TP / (2·TP + FP + FN)."""
    verdadeiro = y_true == classe
    predito = y_pred == classe
    tp = int(np.sum(verdadeiro & predito))
    fp = int(np.sum(~verdadeiro & predito))
    fn = int(np.sum(verdadeiro & ~predito))
    denominador = 2 * tp + fp + fn
    return (2 * tp / denominador) if denominador > 0 else 0.0


def mcnemar_exato(acertos_a: np.ndarray, acertos_b: np.ndarray) -> dict[str, Any]:
    """McNemar exato (binomial) sobre os vetores booleanos de acerto."""
    from scipy.stats import binomtest

    b = int(np.sum(acertos_a & ~acertos_b))   # A certo, B errado
    c = int(np.sum(~acertos_a & acertos_b))   # A errado, B certo
    n = b + c
    if n == 0:
        return {"b_only_a_correct": b, "c_only_b_correct": c,
                "p_value": 1.0, "note": "sem discordancias"}
    p = binomtest(b, n, 0.5, alternative="two-sided").pvalue
    return {"b_only_a_correct": b, "c_only_b_correct": c,
            "n_discordant": n, "p_value": float(p)}


def bootstrap_pareado(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, classe: int,
                      n_boot: int, seed: int) -> dict[str, Any]:
    """Bootstrap pareado da diferença F1_A − F1_B na classe `classe`."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observada = f1_classe(y_true, y_a, classe) - f1_classe(y_true, y_b, classe)
    diferencas = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)              # uma reamostra, com reposição
        diferencas[k] = (f1_classe(y_true[idx], y_a[idx], classe)
                         - f1_classe(y_true[idx], y_b[idx], classe))
    baixo, alto = np.percentile(diferencas, [2.5, 97.5])
    if observada >= 0:
        p = 2.0 * float(np.mean(diferencas <= 0.0))
    else:
        p = 2.0 * float(np.mean(diferencas >= 0.0))
    return {"observed_diff": float(observada), "mean_diff": float(np.mean(diferencas)),
            "ci95_low": float(baixo), "ci95_high": float(alto),
            "p_value": min(1.0, p), "n_boot": int(n_boot)}


def comparar(pred_a: dict[str, Any], pred_b: dict[str, Any], *, seed: int,
             alvo: str = ALVO, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Compara duas predições (sidecars) do mesmo conjunto.

    Devolve o relatório com as chaves do legado: `model_a`, `model_b`,
    `n_test`, `target_class`, `accuracy`, `target_f1`, `mcnemar` e
    `paired_bootstrap`.
    """
    rotulos = pred_a["labels"]
    if pred_b["labels"] != rotulos:
        raise ValueError(f"rótulos divergentes: {rotulos} x {pred_b['labels']}")
    y_true = np.array(pred_a["y_true"])
    if len(pred_b["y_true"]) != len(y_true) or not np.array_equal(y_true, np.array(pred_b["y_true"])):
        raise ValueError(f"y_true de A e B diferem ({len(y_true)} x {len(pred_b['y_true'])} "
                         "exemplos): as predições não são do mesmo conjunto")
    y_a, y_b = np.array(pred_a["y_pred"]), np.array(pred_b["y_pred"])
    classe = rotulos.index(alvo)

    f1_a, f1_b = f1_classe(y_true, y_a, classe), f1_classe(y_true, y_b, classe)
    return {
        "model_a": pred_a.get("model"), "model_b": pred_b.get("model"),
        "n_test": int(len(y_true)), "target_class": alvo,
        "accuracy": {"a": float(np.mean(y_a == y_true)), "b": float(np.mean(y_b == y_true))},
        "target_f1": {"a": f1_a, "b": f1_b, "a_minus_b": f1_a - f1_b},
        "mcnemar": mcnemar_exato(y_a == y_true, y_b == y_true),
        "paired_bootstrap": bootstrap_pareado(y_true, y_a, y_b, classe, n_boot, seed),
    }


def significativo(relatorio: dict[str, Any]) -> bool:
    """O IC95 do bootstrap exclui o zero (alfa = 0,05)."""
    bs = relatorio["paired_bootstrap"]
    return bs["ci95_low"] > 0 or bs["ci95_high"] < 0


def relatorio_json(relatorio: dict[str, Any]) -> str:
    """Serialização do legado: indentação 2, chaves ordenadas, sem acentos
    escapados e sem quebra de linha no fim."""
    return json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True)


def gravar_relatorio(caminho: str | Path, relatorio: dict[str, Any]) -> None:
    gravar_texto(caminho, relatorio_json(relatorio))
