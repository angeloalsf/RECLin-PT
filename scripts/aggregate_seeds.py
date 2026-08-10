"""
Agrega as metricas de teste entre sementes, por modelo.

POR QUE ESTE SCRIPT EXISTE
--------------------------
A media e o desvio entre as sementes 42 e 43 eram calculados por um snippet
Python copiado do README e colado no terminal. Isso significava que (a) o
numero relatado no artigo dependia de quem colou o snippet e (b) nao havia
como versionar o resultado. Este script substitui o snippet: mesma conta,
executavel e com saida gravada em `results/`.

O QUE GERA
----------
    results/summary_by_seed.json

com, para cada modelo, a media e o desvio-padrao POPULACIONAL (`pstdev`,
como no snippet original) de:

  - `test_macro_f1`
  - `test_f1_per_class[classe]`, nas tres classes

mais os valores por semente que entraram na conta, para o numero ser
auditavel sem reabrir os quatro JSONs.

POR QUE `pstdev` E NAO `stdev`
------------------------------
E o que o snippet do README usava, e com n=2 a escolha nao e neutra:
`stdev` (amostral) da exatamente sqrt(2) vezes `pstdev`. Trocar aqui
mudaria silenciosamente os numeros ja discutidos no artigo. Com duas
sementes, nenhum dos dois estima variancia de inicializacao com
credibilidade -- e por isso que o desvio e reportado como dispersao
observada, nao como intervalo de confianca (ver "Limitacoes" no README).

USO
---
    python scripts/aggregate_seeds.py                    # sementes 42 e 43
    python scripts/aggregate_seeds.py --seeds 42 43 44
    python scripts/aggregate_seeds.py --check            # nao escreve nada
    python scripts/aggregate_seeds.py --out /tmp/s.json

Nao treina nada: so le `results/baseline_<modelo>_seed<N>.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

from _artifacts import (
    BASELINE_ORDER,
    CLASS_ORDER,
    DEFAULT_RESULTS_DIR,
    MissingResultError,
    display_path,
    label_for,
    load_baseline,
)

DEFAULT_SEEDS = [42, 43]
DEFAULT_OUT = DEFAULT_RESULTS_DIR / "summary_by_seed.json"


def metric_getters():
    """Metrica -> funcao que a extrai de um `baseline_*.json`.

    Ordem de saida: macro-F1 primeiro (a manchete), depois o F1 por classe
    na ordem canonica de `_artifacts.CLASS_ORDER`.
    """
    getters = [("macro_f1", lambda d: d["test_macro_f1"])]
    for name in CLASS_ORDER:
        getters.append(
            (f"f1_{name}", lambda d, name=name: d["test_f1_per_class"][name])
        )
    return getters


def aggregate(results_dir: Path, seeds: list[int]) -> dict:
    """Media/desvio por modelo e metrica, sobre as sementes pedidas."""
    summary = {"seeds": list(seeds), "models": {}}

    for slug in BASELINE_ORDER:
        per_seed = {seed: load_baseline(results_dir, slug, seed) for seed in seeds}

        metrics = {}
        for metric, getter in metric_getters():
            values = [getter(per_seed[seed]) for seed in seeds]
            metrics[metric] = {
                "mean": st.mean(values),
                # pstdev: desvio POPULACIONAL, igual ao snippet que este
                # script substituiu. Nao troque por stdev sem refazer os
                # numeros do artigo.
                "pstdev": st.pstdev(values),
                "n": len(values),
                "by_seed": {str(seed): value for seed, value in zip(seeds, values)},
            }

        summary["models"][slug] = {
            "label": label_for(per_seed[seeds[0]]),
            "model": per_seed[seeds[0]]["model"],
            "metrics": metrics,
        }

    return summary


def render(summary: dict) -> str:
    """Tabela de terminal, no formato `media ± desvio` do snippet antigo."""
    seeds = summary["seeds"]
    lines = [f"Agregacao sobre as sementes {seeds} (media ± desvio populacional)", ""]

    metrics = [name for name, _ in metric_getters()]
    width = max(len(m) for m in metrics) + 2

    for slug in BASELINE_ORDER:
        entry = summary["models"][slug]
        lines.append(f"{entry['label']}  [{slug}]")
        for metric in metrics:
            stats = entry["metrics"][metric]
            by_seed = " ".join(
                f"seed{seed}={stats['by_seed'][str(seed)]:.4f}" for seed in seeds
            )
            lines.append(
                f"  {metric:<{width}} {stats['mean']:.4f} ± {stats['pstdev']:.4f}"
                f" (n={stats['n']})   [{by_seed}]"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="so imprime a agregacao; nao grava o JSON")
    args = ap.parse_args()

    if len(args.seeds) < 2:
        print("ERRO: --seeds precisa de ao menos duas sementes para agregar.",
              file=sys.stderr)
        return 1

    try:
        summary = aggregate(args.results_dir, args.seeds)
    except (MissingResultError, ValueError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1

    print(render(summary), end="")

    if args.check:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nescrito: {display_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
