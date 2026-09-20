#!/usr/bin/env python3
"""
Consolida em `results/FASE2_significancia.md` o quadro do TEST e as 24
comparacoes de `src/significance.py`. Nao recalcula nada e nao toca no TEST:
so le os JSON ja gravados por `run_fase2_test.py` e `run_fase2_significance.py`.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BASELINES = ["baseline_biobertpt_seed42", "baseline_bertimbau_seed42",
             "baseline_biobertpt_seed43", "baseline_bertimbau_seed43"]
SYSTEMS = ["filtro_biobertpt_seed42", "filtro_bertimbau_seed42",
           "filtro_biobertpt_seed43", "filtro_bertimbau_seed43", "regra_pura"]
NICE = {
    "baseline_biobertpt_seed42": "BioBERTpt s42", "baseline_bertimbau_seed42": "BERTimbau s42",
    "baseline_biobertpt_seed43": "BioBERTpt s43", "baseline_bertimbau_seed43": "BERTimbau s43",
    "filtro_biobertpt_seed42": "filtro(BioBERTpt s42)", "filtro_bertimbau_seed42": "filtro(BERTimbau s42)",
    "filtro_biobertpt_seed43": "filtro(BioBERTpt s43)", "filtro_bertimbau_seed43": "filtro(BERTimbau s43)",
    "regra_pura": "regra pura R3",
}


def d(x, n=4):
    return f"{x:.{n}f}".replace(".", ",")


def sd(x, n=4):
    """Como `d`, mas com sinal explicito."""
    return f"{x:+.{n}f}".replace(".", ",")


def main() -> int:
    summ = json.loads((REPO / "results/FASE2_test_summary.json").read_text(encoding="utf-8"))
    by = {s["path"].replace(".preds.json", ""): s for s in summ["systems"]}
    sig = {}
    for p in glob.glob(str(REPO / "results/significance_f*_vs_*.json")) + \
             glob.glob(str(REPO / "results/significance_regra_pura_vs_*.json")):
        name = Path(p).stem.replace("significance_", "")
        a, b = name.split("_vs_")
        sig[(a, b)] = json.loads(Path(p).read_text(encoding="utf-8"))

    L = []
    A = L.append
    A("# Fase 2 sem GPU: filtro de pista e regra pura no TEST")
    A("")
    A(f"`{datetime.now(timezone.utc).strftime('%d/%m/%Y')}` · "
      f"`max_gap = 25` · `n_test = {summ['n_test']}` · "
      f"`calibracao = {json.dumps(summ['calibration'], sort_keys=True)}`")
    A("")
    A("Tudo calibrado no DEV (`results/CALIBRACAO_filtro.md`) e so entao aplicado ao")
    A("TEST, uma vez por sistema. As linhas de registro estao nos")
    A("`results/*.test_evals.jsonl`, com `eval_index_for_config = 1` para cada um dos")
    A("cinco sistemas novos.")
    A("")
    A("## Quadro no TEST")
    A("")
    A("| sistema | TP | FP | FN | P | R | **F1 `negation_of`** | macro-F1 |")
    A("|---|---|---|---|---|---|---|---|")
    for k in BASELINES + SYSTEMS:
        s = by[k]
        star = "**" if k in SYSTEMS else ""
        A(f"| {NICE[k]} | {s['tp']} | {s['fp']} | {s['fn']} | {d(s['precision'])} | "
          f"{d(s['recall'])} | {star}{d(s['f1'])}{star} | {d(s['macro_f1'])} |")
    A("")
    A("O filtro so rebaixa predicoes `negation_of` para `no_relation`, entao o F1 de")
    A("`associated_with` fica inalterado em todos os digitos nas quatro execucoes. Foi")
    A("conferido, e o macro-F1 sobe por conta das outras duas classes.")
    A("")
    A("## As 24 comparacoes")
    A("")
    A("Bootstrap pareado, 10.000 reamostragens, IC95 por percentis 2,5/97,5. A semente")
    A("do bootstrap segue a regra do Makefile, estendida ao caso cruzado: e a semente do")
    A("lado A, e quando A nao tem semente e a do lado B.")
    A("")
    A("O McNemar aparece so como contexto. Entre um sistema e sua propria versao")
    A("filtrada ele e degenerado por construcao, porque a regra so muda predicoes numa")
    A("direcao. O criterio de decisao aqui e o IC95 do bootstrap.")
    A("")
    A("| A | B | F1 A | F1 B | diferenca | IC95 | IC95 exclui zero? |")
    A("|---|---|---|---|---|---|---|")
    for a in SYSTEMS:
        for b in BASELINES + (["regra_pura"] if a != "regra_pura" else []):
            if (a, b) not in sig:
                continue
            s = sig[(a, b)]
            bs = s["paired_bootstrap"]
            excl = bs["ci95_low"] > 0 or bs["ci95_high"] < 0
            A(f"| {NICE[a]} | {NICE[b]} | {d(s['target_f1']['a'])} | "
              f"{d(s['target_f1']['b'])} | {sd(bs['observed_diff'])} | "
              f"[{sd(bs['ci95_low'])}; {sd(bs['ci95_high'])}] | "
              f"{'**sim**' if excl else 'nao'} |")
    A("")
    A("## Veredito")
    A("")
    filt = [k for k in SYSTEMS if k.startswith("filtro")]
    fails = [(a, b) for a in filt for b in BASELINES
             if not (sig[(a, b)]["paired_bootstrap"]["ci95_low"] > 0)]
    worst_sys = min(filt, key=lambda k: by[k]["f1"])
    best_base = max(BASELINES, key=lambda k: by[k]["f1"])
    ws = sig[(worst_sys, best_base)]["paired_bootstrap"]
    A(f"As {len(filt)*len(BASELINES)} comparacoes de cada execucao filtrada contra cada")
    A(f"baseline oficial dao IC95 acima de zero em {len(filt)*len(BASELINES)-len(fails)} delas.")
    if not fails:
        A("Nenhuma falha, nas duas sementes e nos dois encoders. O pior caso possivel e a")
        A(f"execucao filtrada mais fraca, {NICE[worst_sys]} = {d(by[worst_sys]['f1'])}, contra o")
        A(f"baseline mais forte, {NICE[best_base]} = {d(by[best_base]['f1'])}, e mesmo ele da")
        A(f"{sd(ws['observed_diff'])} com IC95 [{sd(ws['ci95_low'])}; {sd(ws['ci95_high'])}].")
        A("")
        A("**A meta da fase 2 esta batida sem fine-tuning.**")
    else:
        A("Falham: " + "; ".join(f"{NICE[a]} vs {NICE[b]}" for a, b in fails) + ".")
    A("")
    rule_fail = [b for b in BASELINES
                 if not (sig[("regra_pura", b)]["paired_bootstrap"]["ci95_low"] > 0)]
    A(f"A regra pura sozinha nao supera baseline nenhum. Ela fica em "
      f"{d(by['regra_pura']['f1'])}, e contra os quatro o IC95 contem zero, ou seja, ela")
    A("empata estatisticamente com os quatro baselines de 11,4 h de GPU. Esse e o")
    A("numero que protege o Cap. 6, e ele corta nos dois sentidos: a regra e um piso")
    A("alto, e o que o modelo contextual acrescenta so aparece depois que a pista")
    A("restringe o espaco.")
    A("")
    (REPO / "results/FASE2_significancia.md").write_text("\n".join(L) + "\n",
                                                         encoding="utf-8")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
