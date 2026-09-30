#!/usr/bin/env python3
"""
Criterio de parada da Tarefa 2 (sanidade): o fine-tuning restrito supera o
BASELINE do mesmo encoder no DEV?

REGRA (fixada antes de rodar, nao revisitar depois de ver o resultado)
----------------------------------------------------------------------
Para cada semente da sanidade, compara o F1 de `negation_of` no DEV (19.064
pares, predicoes do restrito remapeadas ao espaco completo) com o F1 do baseline
original do MESMO encoder e da MESMA semente, lido do `.dev_preds.json` oficial.

    PASSA  <=>  F1_dev(restrito, s) > F1_dev(baseline, s)  para TODAS as sementes

Uma semente acima e outra abaixo = NAO PASSA. A comparacao e contra o baseline,
nunca contra o filtro da fase 2; o filtro aparece no relatorio so como contexto.
O TEST nao e lido por este script.

O QUE MAIS E REGISTRADO (informativo, nao entra na regra)
---------------------------------------------------------
* bootstrap pareado no DEV (src/significance.py inalterado, executado no mesmo
  processo para nao escrever em logs/pipeline.log);
* F1 do filtro da fase 2 no mesmo DEV (baseline + rebaixamento fora do lexico);
* confusao DENTRO do espaco restrito, em especial `negation_of` predito como
  `associated_with` (efeito possivel do peso 133 que o balanced da aos 17
  `associated_with` do train restrito);
* melhor epoca e se ela foi a ultima (sinal de subtreino);
* GPU gasta (soma de `duration_s` do dev_history).

Uso:
    python src/finetuning_restrito/criterio_parada.py --encoder bertimbau --seeds 42 43
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import runpy  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolamento  # noqa: E402
from _isolamento import REPO, RESULTS_DIR, SRC, exigir_saida_isolada  # noqa: E402

from relation_extraction import LABELS, read_jsonl  # noqa: E402
from restricted_space import (ASSOC, MIN_FREQ, NEG, NOREL,  # noqa: E402
                              carregar_espacos, carregar_lexico)

log = _isolamento.get()


def prf(y_true, y_pred, cls=NEG):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"f1": f, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn}


def rodar_significance(a, b, out, seed, n_boot):
    argv = sys.argv
    sys.argv = ["significance.py", "--a", str(a), "--b", str(b), "--target",
                "negation_of", "--n-boot", str(n_boot), "--seed", str(seed),
                "--out", str(out)]
    try:
        runpy.run_path(str(SRC / "significance.py"), run_name="__main__")
        code = 0
    except SystemExit as e:
        code = int(e.code or 0)
    finally:
        sys.argv = argv
    if code != 0:
        raise SystemExit(f"significance.py devolveu {code} para {a} vs {b}")
    return json.loads(Path(out).read_text(encoding="utf-8"))


def fmt(x, n=4):
    return f"{x:.{n}f}".replace(".", ",")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--encoder", required=True, choices=["biobertpt", "bertimbau"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="onde estao os restrito_<encoder>_seed<N>.* (Tarefa 2)")
    ap.add_argument("--baseline-dir", default=str(REPO / "results"))
    ap.add_argument("--splits-dir", default=str(REPO / "data" / "splits"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out-prefix", default=None,
                    help="default: <results-dir>/CRITERIO_PARADA_tarefa2_<encoder>")
    args = ap.parse_args()

    rdir, bdir = Path(args.results_dir), Path(args.baseline_dir)
    prefix = Path(args.out_prefix) if args.out_prefix else rdir / f"CRITERIO_PARADA_tarefa2_{args.encoder}"
    exigir_saida_isolada(prefix)
    sig_dir = prefix.parent / "significancia_dev"

    lex = carregar_lexico(read_jsonl(Path(args.splits_dir) / "train.jsonl"),
                          min_freq=MIN_FREQ, guarda=bdir / "CALIBRACAO_filtro.json")
    dv = carregar_espacos(args.splits_dir, lex, splits=("dev",))["dev"]
    dentro = set(dv.indices)

    linhas, gpu_s = [], 0.0
    for s in args.seeds:
        stem = f"restrito_{args.encoder}_seed{s}"
        rj = json.loads((rdir / f"{stem}.json").read_text(encoding="utf-8"))
        rd = json.loads((rdir / f"{stem}.dev_preds.json").read_text(encoding="utf-8"))
        bd = json.loads((bdir / f"baseline_{args.encoder}_seed{s}.dev_preds.json").read_text(encoding="utf-8"))
        assert rd["labels"] == LABELS == bd["labels"]
        assert rd["y_true"] == bd["y_true"] == dv.y_true_full, "y_true do dev diverge"
        assert len(rd["y_pred"]) == dv.n_full == 19064
        assert rd["restricted_indices"] == dv.indices, "espaco restrito do sidecar diverge"
        assert all(rd["y_pred"][i] == NOREL for i in range(dv.n_full) if i not in dentro)
        assert rj["seed"] == s and rj["encoder"] == args.encoder

        yt = rd["y_true"]
        m_r, m_b = prf(yt, rd["y_pred"]), prf(yt, bd["y_pred"])
        filt = [NOREL if (p == NEG and i not in dentro) else p for i, p in enumerate(bd["y_pred"])]
        m_f = prf(yt, filt)
        # confusao dentro do espaco restrito
        conf = [[0] * 3 for _ in range(3)]
        for i in dv.indices:
            conf[yt[i]][rd["y_pred"][i]] += 1
        sig = rodar_significance(rdir / f"{stem}.dev_preds.json",
                                 bdir / f"baseline_{args.encoder}_seed{s}.dev_preds.json",
                                 sig_dir / f"dev_sig_{stem}_vs_baseline_{args.encoder}_seed{s}.json",
                                 s, args.n_boot)
        hist = rj["dev_history"]
        dur = sum(h.get("duration_s", 0.0) for h in hist)
        gpu_s += dur
        linhas.append({
            "seed": s,
            "restrito": m_r, "baseline": m_b, "filtro_fase2_no_dev": m_f,
            "delta_f1_vs_baseline": m_r["f1"] - m_b["f1"],
            "supera_baseline": m_r["f1"] > m_b["f1"],
            "delta_f1_vs_filtro": m_r["f1"] - m_f["f1"],
            "bootstrap_dev_vs_baseline": sig["paired_bootstrap"],
            "confusao_dentro_do_espaco_restrito": {"labels": LABELS, "matriz": conf},
            "negation_of_predito_associated_with": conf[NEG][ASSOC],
            "best_epoch": rj["instrumentation"]["best_epoch"],
            "epochs": rj["config"]["epochs"],
            "best_epoch_e_a_ultima": rj["instrumentation"]["best_epoch"] == rj["config"]["epochs"],
            "dev_history_negation_of_f1": [round(h["dev_negation_of_f1"], 4) for h in hist],
            "duracao_treino_s": dur,
            "gpu": rj["environment"].get("gpu"),
            "avisos_capturados": len(rj["instrumentation"].get("avisos_capturados", [])),
        })

    passa = all(l["supera_baseline"] for l in linhas)
    media = sum(l["delta_f1_vs_baseline"] for l in linhas) / len(linhas)
    veredito = ("PASSA: o restrito supera o baseline no DEV em todas as sementes. "
                "Prosseguir para a Tarefa 3." if passa else
                "NAO PASSA: o restrito nao supera o baseline no DEV em todas as "
                "sementes. PARAR aqui; este e o veredito da frente.")
    rel = {"data": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "encoder": args.encoder,
           "seeds": args.seeds,
           "regra": "PASSA sse F1_dev(negation_of) restrito > baseline do mesmo "
                    "encoder e semente, para todas as sementes; dev remapeado, 19.064 pares",
           "teto_recall_dev": dv.teto_recall(), "n_restrito_dev": dv.n_restrito,
           "por_semente": linhas, "delta_medio_vs_baseline": media,
           "passa": passa, "veredito": veredito,
           "gpu_total_s": gpu_s}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(rel, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8", newline="\n")

    md = [f"# Criterio de parada da Tarefa 2 ({args.encoder})", "",
          f"`{rel['data']}` · DEV remapeado, 19.064 pares, 150 `negation_of` no gold · "
          f"teto de recall do espaco restrito {fmt(dv.teto_recall())} "
          f"({dv.n_restrito} pares restritos). O TEST nao foi lido.", "",
          "**Regra fixada antes de rodar:** passa se o F1 de `negation_of` do "
          "restrito superar o do baseline do mesmo encoder e da mesma semente, "
          "em todas as sementes.", "",
          "| semente | restrito F1 (P / R) | baseline F1 (P / R) | Δ | IC95 bootstrap dev | filtro fase 2 no dev | melhor época |",
          "|---|---|---|---|---|---|---|"]
    for l in linhas:
        r, b, f, bs = l["restrito"], l["baseline"], l["filtro_fase2_no_dev"], l["bootstrap_dev_vs_baseline"]
        md.append(f"| {l['seed']} | {fmt(r['f1'])} ({fmt(r['p'])} / {fmt(r['r'])}) | "
                  f"{fmt(b['f1'])} ({fmt(b['p'])} / {fmt(b['r'])}) | "
                  f"{'+' if l['delta_f1_vs_baseline'] >= 0 else ''}{fmt(l['delta_f1_vs_baseline'])} | "
                  f"[{fmt(bs['ci95_low'])}; {fmt(bs['ci95_high'])}] | {fmt(f['f1'])} | "
                  f"{l['best_epoch']} de {l['epochs']} |")
    md += ["", f"**Veredito: {veredito}**", "",
           f"Δ médio vs baseline: {fmt(media)}. GPU gasta na sanidade: "
           f"{fmt(gpu_s / 3600, 2)} h.", "",
           "## Diagnóstico (não entra na regra)", ""]
    for l in linhas:
        md.append(f"- semente {l['seed']}: {l['negation_of_predito_associated_with']} "
                  f"`negation_of` do dev preditos como `associated_with` dentro do "
                  f"espaço restrito; melhor época é a última: "
                  f"{'sim' if l['best_epoch_e_a_ultima'] else 'não'}; F1 por época "
                  f"{l['dev_history_negation_of_f1']}; {l['avisos_capturados']} aviso(s) "
                  f"de determinismo capturado(s).")
    md += ["", "O IC do bootstrap no dev e o filtro são contexto. A decisão usa "
           "só a comparação pontual acima, como fixado na tarefa."]
    prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    log.info("%s", veredito)
    log.info("Relatorio: %s(.json|.md)", prefix)
    return 0 if passa else 3


if __name__ == "__main__":
    raise SystemExit(main())
