#!/usr/bin/env python3
"""
Gera as predicoes FINAIS no TEST dos sistemas da fase 2 e registra a avaliacao.

Este script e o unico ponto em que a fase 2 toca o TEST. Ele nao escolhe nada:
le a configuracao ja congelada em `results/CALIBRACAO_filtro.json`, produzida por
`scripts/calibrate_cue_filter.py` a partir do DEV. Se aquele arquivo nao existir,
aborta em vez de usar um default.

Sistemas gerados:
  * `filtro(<execucao>)` para as 4 execucoes oficiais de 17-20/09, com o
    `min_freq` calibrado no DEV;
  * `regra pura` R3 com o limiar de gap calibrado no DEV, sem modelo nenhum.

REGISTRO DE MULTIPLICIDADE
--------------------------
Cada sistema anexa uma linha ao `results/<execucao>.test_evals.jsonl` da execucao
de base, no mesmo formato de `relation_extraction.append_test_eval_log`, com dois
contadores:
  * `eval_index`, que conta TODAS as linhas do arquivo, mantendo a semantica que
    o arquivo ja tinha;
  * `eval_index_for_config`, que conta so as linhas com o mesmo `config_sha1`, e
    responde a pergunta que importa para multiplicidade, que e quantas vezes o
    TEST foi medido PARA AQUELA CONFIGURACAO.
O `config_sha1` da fase 2 e derivado da configuracao de pos-processamento somada
a identidade da execucao de base, entao e necessariamente diferente do hash do
baseline: sao configuracoes diferentes, e cada uma comeca a propria contagem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from candidates import iter_candidate_pairs  # noqa: E402
from make_rule_baseline import predict_rule, score  # noqa: E402
from negation_lexicon import (LABELS, apply_cue_filter,  # noqa: E402
                              induce_lexicon, read_jsonl)
from utils.logger import get_logger  # noqa: E402

log = get_logger("fase2_test")

NEG = LABELS.index("negation_of")

RUNS = [
    ("biobertpt", 42, "pucpr/biobertpt-all"),
    ("bertimbau", 42, "neuralmind/bert-base-portuguese-cased"),
    ("biobertpt", 43, "pucpr/biobertpt-all"),
    ("bertimbau", 43, "neuralmind/bert-base-portuguese-cased"),
]


def f1_class(y_true, y_pred, cls):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    den = 2 * tp + fp + fn
    return (2 * tp / den) if den else 0.0


def macro_f1(y_true, y_pred):
    return sum(f1_class(y_true, y_pred, i) for i in range(len(LABELS))) / len(LABELS)


def append_eval(out_path: Path, *, model, seed, cfg, n_test, macro, negf1):
    log_path = out_path.with_suffix(".test_evals.jsonl")
    digest = hashlib.sha1(
        json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    lines = []
    if log_path.is_file():
        lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    same_cfg = sum(1 for ln in lines if json.loads(ln).get("config_sha1") == digest)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "eval_index": len(lines) + 1,
        "eval_index_for_config": same_cfg + 1,
        "out": out_path.resolve().relative_to(REPO).as_posix(),
        "model": model,
        "seed": seed,
        "config_sha1": digest,
        "config": cfg,
        "n_test": int(n_test),
        "test_macro_f1": round(float(macro), 6),
        "test_negation_of_f1": round(float(negf1), 6),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("TEST registrado em %s: eval_index=%d, para esta config=%d "
             "(config_sha1=%s)", log_path.name, entry["eval_index"],
             entry["eval_index_for_config"], digest)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gap", type=int, default=25)
    ap.add_argument("--calib", default="results/CALIBRACAO_filtro.json")
    args = ap.parse_args()

    calib_path = REPO / args.calib
    if not calib_path.is_file():
        log.error("Configuracao calibrada ausente (%s). Rode primeiro "
                  "scripts/calibrate_cue_filter.py.", calib_path)
        return 2
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    log.info("Configuracao congelada no DEV: %s", calib)

    cands = [c for d in read_jsonl(REPO / "data/splits/test.jsonl")
             for c in iter_candidate_pairs(d, max_gap=args.max_gap)]
    y_true = [LABELS.index(c["label"]) for c in cands]
    log.info("TEST: %d candidatos, %d `negation_of` no gold",
             len(cands), sum(1 for y in y_true if y == NEG))

    lex = induce_lexicon(read_jsonl(REPO / "data/splits/train.jsonl"),
                         args.max_gap, calib["min_freq"])

    summary = []

    # --- 4 execucoes filtradas ----------------------------------------------
    for short, seed, model in RUNS:
        base = REPO / f"results/baseline_{short}_seed{seed}.json"
        preds_path = base.with_suffix(".preds.json")
        d = json.loads(preds_path.read_text(encoding="utf-8"))
        if d["labels"] != LABELS or d["y_true"] != y_true:
            log.error("%s nao pertence a este espaco de candidatos", preds_path)
            return 2
        y = apply_cue_filter(cands, d["y_pred"], lex)
        out = REPO / f"results/filtro_{short}_seed{seed}.preds.json"
        out.write_text(json.dumps({
            "model": f"filtro({model} s{seed})", "seed": seed,
            "labels": LABELS, "y_true": y_true, "y_pred": y,
            "base_preds": preds_path.name,
            "postproc": {"kind": "cue_filter", "min_freq": calib["min_freq"],
                         "lexicon_size": len(lex), "max_gap": args.max_gap,
                         "demote_to": "no_relation",
                         "calibrated_on": "dev"},
        }, ensure_ascii=False), encoding="utf-8")
        m, mac = score(y_true, y), macro_f1(y_true, y)
        cfg = {"system": "cue_filter", "base": preds_path.name,
               "min_freq": calib["min_freq"], "max_gap": args.max_gap,
               "demote_to": "no_relation", "calibrated_on": "dev"}
        append_eval(base, model=f"filtro({model})", seed=seed, cfg=cfg,
                    n_test=len(y_true), macro=mac, negf1=m["f1"])
        log.info("filtro(%s s%d): F1(negation_of)=%.4f  P=%.4f R=%.4f  macro=%.4f",
                 short, seed, m["f1"], m["precision"], m["recall"], mac)
        summary.append({"system": f"filtro({short} s{seed})", "path": out.name,
                        "seed": seed, **m, "macro_f1": mac})

    # --- regra pura ----------------------------------------------------------
    y = predict_rule(cands, lex, calib["rule"], calib["rule_gap"])
    out = REPO / "results/regra_pura.preds.json"
    out.write_text(json.dumps({
        "model": f"regra {calib['rule']} (gap<={calib['rule_gap']}, "
                 f"min_freq={calib['min_freq']})",
        "seed": None, "labels": LABELS, "y_true": y_true, "y_pred": y,
        "postproc": {"kind": "pure_rule", "rule": calib["rule"],
                     "max_target_gap": calib["rule_gap"],
                     "min_freq": calib["min_freq"], "lexicon_size": len(lex),
                     "max_gap": args.max_gap, "calibrated_on": "dev"},
    }, ensure_ascii=False), encoding="utf-8")
    m, mac = score(y_true, y), macro_f1(y_true, y)
    cfg = {"system": "pure_rule", "rule": calib["rule"],
           "max_target_gap": calib["rule_gap"], "min_freq": calib["min_freq"],
           "max_gap": args.max_gap, "calibrated_on": "dev"}
    append_eval(REPO / "results/regra_pura.json", model=f"regra {calib['rule']}",
                seed=None, cfg=cfg, n_test=len(y_true), macro=mac, negf1=m["f1"])
    log.info("regra pura %s (gap<=%d): F1(negation_of)=%.4f  P=%.4f R=%.4f  macro=%.4f",
             calib["rule"], calib["rule_gap"], m["f1"], m["precision"],
             m["recall"], mac)
    summary.append({"system": f"regra {calib['rule']}", "path": out.name,
                    "seed": None, **m, "macro_f1": mac})

    # --- baselines, so para o quadro (nenhuma avaliacao nova registrada) -----
    for short, seed, model in RUNS:
        d = json.loads((REPO / f"results/baseline_{short}_seed{seed}.preds.json")
                       .read_text(encoding="utf-8"))
        m = score(y_true, d["y_pred"])
        summary.append({"system": f"baseline {short} s{seed}",
                        "path": f"baseline_{short}_seed{seed}.preds.json",
                        "seed": seed, **m,
                        "macro_f1": macro_f1(y_true, d["y_pred"])})

    (REPO / "results/FASE2_test_summary.json").write_text(
        json.dumps({"calibration": calib, "n_test": len(y_true),
                    "systems": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log.info("Quadro salvo em results/FASE2_test_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
