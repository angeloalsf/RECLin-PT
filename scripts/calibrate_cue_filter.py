#!/usr/bin/env python3
"""
Calibracao no DEV do filtro de pista e da regra pura. NUNCA le o TEST.

Le os `<out>.dev_preds.json` das 4 execucoes oficiais (predicoes do DEV geradas
com os pesos da MELHOR epoca) e varre os hiperparametros pos-hoc, escolhendo
cada um pelo F1 de `negation_of` no DEV. Grava `results/CALIBRACAO_filtro.md`.

CRITERIO DE DECISAO, FIXADO ANTES DE RODAR
------------------------------------------
1. `min_freq` do lexico: maximiza a MEDIA do F1 de `negation_of` no DEV sobre as
   4 execucoes. Empate resolve pelo MENOR `min_freq`, que e o lexico mais
   coberto e o teto de recall mais alto.
2. Regra pura (R1-R4 e o limiar de gap): maximiza o F1 de `negation_of` no DEV.
   Empate resolve pela regra de menor indice, depois pelo menor limiar.
3. Sistema combinado: mesma regra do item 1, sobre a media das 4 execucoes.
Nenhum desses numeros e do TEST, e nenhuma escolha e revisitada depois de o TEST
ser tocado.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from candidates import entity_gap, iter_candidate_pairs  # noqa: E402
from make_rule_baseline import RULES, predict_rule, score  # noqa: E402
from negation_lexicon import (LABELS, apply_cue_filter,  # noqa: E402
                              count_cue_forms, induce_lexicon, is_cue,
                              read_jsonl)
from utils.logger import get_logger  # noqa: E402

log = get_logger("calibrate")

NEG = LABELS.index("negation_of")
NOR = LABELS.index("no_relation")
MIN_FREQS = (1, 2, 3, 5, 10)
GAPS = (0, 1, 2, 3, 5, 10, 25)

RUNS = [
    ("BioBERTpt s42", "results/baseline_biobertpt_seed42.dev_preds.json"),
    ("BERTimbau s42", "results/baseline_bertimbau_seed42.dev_preds.json"),
    ("BioBERTpt s43", "results/baseline_biobertpt_seed43.dev_preds.json"),
    ("BERTimbau s43", "results/baseline_bertimbau_seed43.dev_preds.json"),
]


def gap_gate(candidates, y_pred, max_target_gap):
    """Rebaixa `negation_of` cujo gap excede o limiar. Complementa o filtro."""
    out = list(y_pred)
    for i, (c, p) in enumerate(zip(candidates, y_pred)):
        if p == NEG and entity_gap(c["e1"], c["e2"]) > max_target_gap:
            out[i] = NOR
    return out


def fmt(x, n=4):
    return f"{x:.{n}f}".replace(".", ",")


def mil(n):
    """Separador de milhar com ponto, no padrao do texto do TCC."""
    return f"{n:,}".replace(",", ".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gap", type=int, default=25)
    ap.add_argument("--out", default="results/CALIBRACAO_filtro.md")
    args = ap.parse_args()

    cands = [c for d in read_jsonl(REPO / "data/splits/dev.jsonl")
             for c in iter_candidate_pairs(d, max_gap=args.max_gap)]
    y_true = [LABELS.index(c["label"]) for c in cands]
    n_gold = sum(1 for y in y_true if y == NEG)
    log.info("DEV: %d candidatos, %d `negation_of`", len(cands), n_gold)

    preds = {}
    for name, path in RUNS:
        d = json.loads((REPO / path).read_text(encoding="utf-8"))
        if d["labels"] != LABELS or d["y_true"] != y_true:
            log.error("Sidecar %s nao pertence a este espaco de candidatos", path)
            return 2
        preds[name] = d["y_pred"]
    log.info("4 sidecars de DEV alinhados com o espaco reconstruido (0 divergencias)")

    train_docs = list(read_jsonl(REPO / "data/splits/train.jsonl"))
    counts = count_cue_forms(train_docs, args.max_gap)

    # --- 1. varredura de min_freq -------------------------------------------
    sweep = {}
    for mf in MIN_FREQS:
        lex = {f: n for f, n in counts.items() if n >= mf}
        cov = sum(1 for c, t in zip(cands, y_true)
                  if t == NEG and is_cue(c["e1"], lex))
        row = {"size": len(lex), "coverage": cov / n_gold, "runs": {}}
        for name in preds:
            row["runs"][name] = score(y_true, apply_cue_filter(cands, preds[name], lex))
        row["mean_f1"] = sum(r["f1"] for r in row["runs"].values()) / len(preds)
        sweep[mf] = row
        log.info("min_freq=%2d |lex|=%2d cobertura=%.4f  F1 medio=%.4f",
                 mf, len(lex), row["coverage"], row["mean_f1"])

    best_mf = max(MIN_FREQS, key=lambda m: (sweep[m]["mean_f1"], -m))
    lex_best = induce_lexicon(train_docs, args.max_gap, best_mf)
    log.info("ESCOLHA min_freq=%d (F1 medio no dev=%.4f)",
             best_mf, sweep[best_mf]["mean_f1"])

    # --- 2. regra pura -------------------------------------------------------
    rule_sweep = []
    for rule in RULES:
        gaps = GAPS if rule in ("R3", "R4") else (args.max_gap,)
        for g in gaps:
            m = score(y_true, predict_rule(cands, lex_best, rule, g))
            rule_sweep.append({"rule": rule, "gap": g, **m})
    best_rule = max(rule_sweep,
                    key=lambda r: (r["f1"], -RULES.index(r["rule"]), -r["gap"]))
    log.info("ESCOLHA regra pura: %s com gap<=%d (F1 no dev=%.4f)",
             best_rule["rule"], best_rule["gap"], best_rule["f1"])

    # --- 3. sistema combinado: filtro + porta de gap -------------------------
    comb_sweep = []
    for g in GAPS:
        runs = {}
        for name in preds:
            y = apply_cue_filter(cands, preds[name], lex_best)
            runs[name] = score(y_true, gap_gate(cands, y, g))
        comb_sweep.append({"gap": g, "runs": runs,
                           "mean_f1": sum(r["f1"] for r in runs.values()) / len(runs)})
    best_comb = max(comb_sweep, key=lambda r: (r["mean_f1"], -r["gap"]))
    log.info("ESCOLHA combinado: filtro + gap<=%d (F1 medio no dev=%.4f)",
             best_comb["gap"], best_comb["mean_f1"])

    payload = {
        "min_freq": best_mf, "rule": best_rule["rule"],
        "rule_gap": best_rule["gap"], "combined_gap": best_comb["gap"],
        "lexicon_size": len(lex_best),
    }
    (REPO / "results/CALIBRACAO_filtro.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")

    # --- relatorio -----------------------------------------------------------
    L = []
    A = L.append
    A("# Calibracao do filtro de pista e da regra pura — DEV")
    A("")
    A(f"`{datetime.now(timezone.utc).strftime('%d/%m/%Y')}` · "
      f"`max_gap = {args.max_gap}` · `split = dev` · "
      f"**o TEST nao foi lido por este script**")
    A("")
    A("Tudo aqui sai dos quatro `<out>.dev_preds.json` das execucoes oficiais de")
    A("17-20/09/2026, que guardam as predicoes do DEV feitas com os pesos da melhor")
    A("epoca. O espaco de candidatos do DEV foi reconstruido a partir de")
    A(f"`src/candidates.py` e conferido contra o `y_true` dos quatro sidecars, com")
    A(f"zero divergencias. Sao {mil(len(cands))} candidatos e {n_gold} pares `negation_of`")
    A("no gold.")
    A("")
    A("## Criterio, fixado antes de rodar")
    A("")
    A("1. `min_freq` do lexico maximiza a **media** do F1 de `negation_of` no DEV sobre")
    A("   as quatro execucoes. Empate resolve pelo menor `min_freq`, que e o lexico de")
    A("   maior cobertura e portanto o teto de recall mais alto.")
    A("2. A regra pura escolhe variante e limiar de gap pelo F1 de `negation_of` no DEV.")
    A("   Empate resolve pela regra de menor indice, depois pelo menor limiar.")
    A("3. O sistema combinado usa o mesmo criterio do item 1.")
    A("")
    A("Nenhuma dessas escolhas foi revisitada depois de o TEST ser tocado.")
    A("")
    A("## 1. Varredura de `min_freq`")
    A("")
    A("Cobertura e a fracao dos pares `negation_of` do DEV cujo e1 pertence ao lexico.")
    A("Ela e o teto de recall que o filtro impoe por construcao.")
    A("")
    A("| `min_freq` | formas | cobertura no dev | " +
      " | ".join(n for n, _ in RUNS) + " | media |")
    A("|---|---|---|" + "---|" * (len(RUNS) + 1))
    for mf in MIN_FREQS:
        r = sweep[mf]
        cells = " | ".join(fmt(r["runs"][n]["f1"]) for n, _ in RUNS)
        mark = "**" if mf == best_mf else ""
        A(f"| {mark}{mf}{mark} | {r['size']} | {fmt(r['coverage'])} | {cells} | "
          f"{mark}{fmt(r['mean_f1'])}{mark} |")
    A("")
    A(f"**Escolhido: `min_freq = {best_mf}`**, com {len(lex_best)} formas e cobertura")
    A(f"{fmt(sweep[best_mf]['coverage'])} no DEV.")
    A("")
    if best_mf != max(MIN_FREQS):
        A("Vale registrar onde o maximo caiu. A analise exploratoria de 09/09, feita")
        A("diretamente no TEST, encontrou ganho monotonico com lexicos cada vez mais")
        A(f"restritivos, com o melhor valor em `min_freq = {max(MIN_FREQS)}`. No DEV o maximo e")
        A(f"interior, em `min_freq = {best_mf}`, porque a cobertura comeca a cair antes de o")
        A("ganho de precisao compensar. A escolha honesta portanto nao e a que o TEST")
        A("premiaria, e a diferenca entre as duas e justamente o que a disciplina de")
        A("calibracao custa.")
        A("")
    A("Lexico resultante, com a frequencia no train:")
    A("")
    A("```")
    for form, n in lex_best.items():
        A(f"{n:>6}  {form}")
    A("```")
    A("")
    A("## 2. Regra pura (R1-R4)")
    A("")
    A(f"Lexico fixado em `min_freq = {best_mf}`. `gap` e `candidates.entity_gap`.")
    A("R1 e R2 nao tem limiar de gap, entao aparecem uma vez, com a janela do espaco")
    A("de candidatos.")
    A("")
    A("| regra | `gap <=` | TP | FP | FN | P | R | F1 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rule_sweep:
        mark = "**" if (r["rule"], r["gap"]) == (best_rule["rule"], best_rule["gap"]) else ""
        A(f"| {mark}{r['rule']}{mark} | {r['gap']} | {r['tp']} | {r['fp']} | {r['fn']} | "
          f"{fmt(r['precision'])} | {fmt(r['recall'])} | {mark}{fmt(r['f1'])}{mark} |")
    A("")
    A(f"**Escolhida: {best_rule['rule']} com `gap <= {best_rule['gap']}`**, "
      f"F1 = {fmt(best_rule['f1'])} no DEV.")
    A("")
    A("## 3. Sistema combinado: filtro + porta de gap")
    A("")
    A("O filtro de pista sozinho nao usa distancia. Esta secao mede o que acontece ao")
    A("aplicar tambem a condicao de gap de R3 sobre as predicoes ja filtradas, que e a")
    A("leitura literal de combinar o filtro com a regra pura. `gap <= 25` equivale a")
    A("desligar a porta, porque nenhum candidato excede a janela.")
    A("")
    A("| `gap <=` | " + " | ".join(n for n, _ in RUNS) + " | media |")
    A("|---|" + "---|" * (len(RUNS) + 1))
    for r in comb_sweep:
        mark = "**" if r["gap"] == best_comb["gap"] else ""
        cells = " | ".join(fmt(r["runs"][n]["f1"]) for n, _ in RUNS)
        A(f"| {mark}{r['gap']}{mark} | {cells} | {mark}{fmt(r['mean_f1'])}{mark} |")
    A("")
    A(f"**Escolhido: `gap <= {best_comb['gap']}`.**")
    if best_comb["gap"] >= args.max_gap:
        A("O limiar vencedor desliga a porta, ou seja, o DEV nao sustenta somar a")
        A("condicao de distancia ao filtro. O sistema combinado colapsa no filtro")
        A("sozinho e nao sera avaliado como um sistema separado no TEST.")
    A("")
    A("## Configuracao final levada ao TEST")
    A("")
    A("```json")
    A(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    A("```")
    A("")
    out = REPO / args.out
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("Relatorio de calibracao salvo em %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
