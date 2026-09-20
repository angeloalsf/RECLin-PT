#!/usr/bin/env python3
"""
Baseline de REGRA PURA para `negation_of`, sem rede neural nenhuma.

POR QUE ESTE SCRIPT EXISTE
--------------------------
O Cap. 6 compara dois encoders entre si e nao compara nenhum deles contra o
baseline mais obvio de todos, que e nao usar modelo. Essa e a primeira conta que
um examinador atento faz, e descobri-la depois da defesa e pior do que
apresenta-la. Com a regra no texto o argumento fica mais forte, porque a
pergunta deixa de ser "qual encoder" e passa a ser "o que o encoder acrescenta
sobre uma regra fixa".

AS QUATRO REGRAS
----------------
Todas partem da mesma pista lexical de `src/negation_lexicon.py`, induzida so do
TRAIN. `gap` e `candidates.entity_gap`, a distancia em caracteres entre os spans.

  R1  todo par cujo e1 e pista  ->  `negation_of`
  R2  R1 + so o alvo mais proximo de cada pista (um alvo por pista)
  R3  R1 + `gap <= max_target_gap`
  R4  R1 + mais proximo + `gap <= max_target_gap`

Desempate de "mais proximo": menor `entity_gap`, depois menor `e2.start`, depois
menor `e2.id` como inteiro. Deterministico e independente da ordem de iteracao.

ESPACO DE SAIDA
---------------
O `preds.json` gerado vive no ESPACO COMPLETO de candidatos do split (19.210 no
test, 19.064 no dev), na mesma ordem de `iter_candidate_pairs`, com os mesmos
`labels` e o mesmo `y_true` dos sidecars das execucoes. Tudo que a regra nao
marca entra como `no_relation`. Isso e o que torna o arquivo pareavel por
`src/significance.py`, que aborta se os `y_true` de A e B divergirem. Regras que
produzem um espaco menor e sao comparadas nele foram a armadilha registrada na
migracao de `max_gap`; aqui o remapeamento e feito na origem.

A regra nunca prediz `associated_with`. O objetivo e ser o piso de
`negation_of`, nao um sistema completo de RE.

Uso:
    python scripts/make_rule_baseline.py --split test --rule R3 \
        --min-freq 2 --max-target-gap 1 \
        --out results/rule_baseline_test.preds.json
    python scripts/make_rule_baseline.py --split dev --all-rules   # so mede
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from candidates import entity_gap, iter_candidate_pairs  # noqa: E402
from negation_lexicon import (LABELS, induce_lexicon, is_cue,  # noqa: E402
                              read_jsonl)
from utils.logger import get_logger  # noqa: E402

log = get_logger("rule_baseline")

RULES = ("R1", "R2", "R3", "R4")
NEG = LABELS.index("negation_of")
NOR = LABELS.index("no_relation")


def build_candidates(split_path, max_gap):
    return [c for doc in read_jsonl(split_path)
            for c in iter_candidate_pairs(doc, max_gap=max_gap)]


def predict_rule(candidates, lexicon, rule, max_target_gap):
    """Aplica R1-R4 e devolve y_pred no espaco completo (ids de LABELS)."""
    if rule not in RULES:
        raise ValueError(f"regra desconhecida: {rule}")

    nearest = rule in ("R2", "R4")
    gated = rule in ("R3", "R4")

    y = [NOR] * len(candidates)
    hits = []  # (indice, doc_id, e1_id, gap, e2_start, e2_id)
    for i, c in enumerate(candidates):
        if not is_cue(c["e1"], lexicon):
            continue
        gap = entity_gap(c["e1"], c["e2"])
        if gated and gap > max_target_gap:
            continue
        hits.append((i, c["doc_id"], c["e1"]["id"], gap,
                     c["e2"]["start"], c["e2"]["id"]))

    if nearest:
        best: dict[tuple, tuple] = {}
        for h in hits:
            key = (h[1], h[2])                      # (doc_id, e1_id)
            order = (h[3], h[4], int(h[5]))         # gap, e2.start, e2.id
            if key not in best or order < best[key][0]:
                best[key] = (order, h[0])
        keep = {idx for _, idx in best.values()}
    else:
        keep = {h[0] for h in hits}

    for i in keep:
        y[i] = NEG
    return y


def score(y_true, y_pred, cls=NEG):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--max-gap", type=int, default=25,
                    help="janela do espaco de candidatos (igual ao pipeline)")
    ap.add_argument("--min-freq", type=int, default=2,
                    help="frequencia minima do lexico (escolher no DEV)")
    ap.add_argument("--max-target-gap", type=int, default=1,
                    help="limiar de gap de R3/R4 (escolher no DEV)")
    ap.add_argument("--rule", default="R3", choices=RULES)
    ap.add_argument("--all-rules", action="store_true",
                    help="mede R1-R4 e nao grava preds.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    splits = Path(args.splits_dir)
    lex = induce_lexicon(read_jsonl(splits / "train.jsonl"),
                         args.max_gap, args.min_freq)
    cands = build_candidates(splits / f"{args.split}.jsonl", args.max_gap)
    y_true = [LABELS.index(c["label"]) for c in cands]
    log.info("Split %s: %d candidatos, %d `negation_of` no gold",
             args.split, len(cands), sum(1 for y in y_true if y == NEG))

    rules = RULES if args.all_rules else (args.rule,)
    results = {}
    for rule in rules:
        y = predict_rule(cands, lex, rule, args.max_target_gap)
        m = score(y_true, y)
        results[rule] = m
        log.info("%s (min_freq=%d, gap<=%d): TP=%d FP=%d FN=%d  P=%.4f R=%.4f "
                 "F1=%.4f", rule, args.min_freq, args.max_target_gap,
                 m["tp"], m["fp"], m["fn"], m["precision"], m["recall"], m["f1"])
        last_y = y

    if args.all_rules:
        log.info("Modo --all-rules: nada foi gravado.")
        return 0

    if not args.out:
        log.error("--out e obrigatorio fora de --all-rules")
        return 2

    payload = {
        "model": f"rule:{args.rule}(min_freq={args.min_freq},"
                 f"gap<={args.max_target_gap})",
        "seed": None,
        "split": args.split,
        "labels": LABELS,
        "y_true": y_true,
        "y_pred": last_y,
        "rule": {"name": args.rule, "min_freq": args.min_freq,
                 "max_target_gap": args.max_target_gap,
                 "max_gap": args.max_gap, "lexicon_size": len(lex)},
        "negation_of": results[args.rule],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info("Predicoes da regra pura salvas em %s (%d exemplos)",
             out, len(last_y))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
