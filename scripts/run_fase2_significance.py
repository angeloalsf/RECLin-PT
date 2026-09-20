#!/usr/bin/env python3
"""
Roda `src/significance.py` de cada sistema da fase 2 contra os 4 baselines.

Nao gera predicao nenhuma; so compara os `preds.json` ja gravados por
`scripts/run_fase2_test.py`. A convencao de semente do bootstrap segue a regra
`SIGNIF_RULE` do Makefile, estendida para o caso cruzado: a semente do bootstrap
e a da execucao do lado A, e quando A nao tem semente (a regra pura), e a do lado
B. Assim a mesma comparacao sempre reproduz o mesmo intervalo.

`--from` e `--to` recortam a lista por indice, so para caber no limite de tempo
de cada chamada; a lista em si e fixa e ordenada.

Uso:
    python scripts/run_fase2_significance.py --list
    python scripts/run_fase2_significance.py --from 0 --to 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SYSTEMS = [
    ("filtro_biobertpt_seed42", 42),
    ("filtro_bertimbau_seed42", 42),
    ("filtro_biobertpt_seed43", 43),
    ("filtro_bertimbau_seed43", 43),
    ("regra_pura", None),
]
BASELINES = [
    ("baseline_biobertpt_seed42", 42),
    ("baseline_bertimbau_seed42", 42),
    ("baseline_biobertpt_seed43", 43),
    ("baseline_bertimbau_seed43", 43),
]
# Comparacao extra, fora das 20 pedidas: cada sistema filtrado contra a regra
# pura. Nao introduz sistema novo no TEST (todos ja foram registrados), e e o
# numero que sustenta a frase "o modelo decide o alvo, e essa decisao vale X".
EXTRA = [(s, "regra_pura", seed) for s, seed in SYSTEMS[:4]]


def jobs():
    out = []
    for a, sa in SYSTEMS:
        for b, sb in BASELINES:
            out.append((a, b, sa if sa is not None else sb))
    out.extend(EXTRA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=10**6)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    all_jobs = jobs()
    if args.list:
        for i, (a, b, s) in enumerate(all_jobs):
            print(f"{i:3d}  {a}  vs  {b}   (seed={s})")
        return 0

    for i, (a, b, seed) in enumerate(all_jobs):
        if not (args.lo <= i < args.hi):
            continue
        out = REPO / f"results/significance_{a}_vs_{b}.json"
        if out.is_file() and not args.force:
            print(f"[{i:3d}] ja existe, pulando: {out.name}")
            continue
        cmd = [sys.executable, str(REPO / "src/significance.py"),
               "--a", str(REPO / f"results/{a}.preds.json"),
               "--b", str(REPO / f"results/{b}.preds.json"),
               "--target", "negation_of", "--n-boot", str(args.n_boot),
               "--seed", str(seed), "--out", str(out)]
        print(f"[{i:3d}] {a} vs {b} (seed={seed})", flush=True)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
