#!/usr/bin/env python3
"""
Avaliacao final da frente (Tarefa 4). SO RODA SE A TAREFA 3 TIVER RODADO, e a
Tarefa 3 so roda se a Tarefa 2 passar em `criterio_parada.py`.

Para cada execucao restrita (encoder x semente), ja remapeada ao espaco completo
do TEST (19.210 pares), roda `src/significance.py` INALTERADO contra:

  (a) os 4 baselines oficiais (results/baseline_*_seed{42,43}.preds.json).
      Comparacao PRINCIPAL: decide se o fine-tuning "funcionou" no mesmo sentido
      em que o filtro funcionou.
  (b) as 4 execucoes filtradas da fase 2 (results/filtro_*_seed{42,43}.preds.json).
      Comparacao SECUNDARIA: so para escolher o que entregar se os dois baterem
      os baselines.

A metrica e o F1 de `negation_of` (bootstrap pareado). O McNemar que
`significance.py` tambem calcula mede acerto nas tres classes e NAO e
interpretavel aqui: o restrito devolve `no_relation` para quase todo
`associated_with` por construcao. Ele fica nos JSONs, marcado como tal.

AGREGACAO POR SEMENTE
---------------------
Para cada encoder e cada comparador: media das diferencas de F1 entre as N
sementes e IC95 t de Student (n-1 graus de liberdade). Esse IC mede a
variabilidade de SEMENTE do restrito com o comparador fixo; a variabilidade de
AMOSTRA DO TEST esta no bootstrap de cada par. Por isso o veredito exige as duas:

  (a) "supera com significancia" para um encoder  <=>
        IC95 entre sementes > 0 contra CADA um dos 4 baselines   E
        o pior caso (execucao restrita mais fraca do encoder x baseline mais
        forte) tem IC95 do bootstrap > 0 -- o mesmo criterio de pior caso usado
        na fase 2.
  (b) contra cada filtro: IC95 entre sementes > 0 = supera; < 0 = fica atras;
      contem 0 = empata.

Uso:
    python src/finetuning_restrito/avaliacao_final.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import runpy  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolamento  # noqa: E402
from _isolamento import REPO, RESULTS_DIR, SRC, exigir_saida_isolada  # noqa: E402

from relation_extraction import LABELS  # noqa: E402

log = _isolamento.get()
ENCODERS = ("biobertpt", "bertimbau")
BASE_SEEDS = (42, 43)


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


def ic_t(xs):
    """Media, desvio e IC95 t de Student de uma lista de diferencas."""
    from scipy.stats import t
    n = len(xs)
    m = statistics.fmean(xs)
    if n < 2:
        return {"n": n, "media": m, "dp": None, "ic95": [None, None]}
    sd = statistics.stdev(xs)
    h = t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return {"n": n, "media": m, "dp": sd, "ic95": [m - h, m + h]}


def classifica(ic):
    lo, hi = ic["ic95"]
    if lo is None:
        return "indefinido (n<2)"
    return "supera" if lo > 0 else ("fica atras" if hi < 0 else "empata")


def fmt(x, n=4, sinal=False):
    if x is None:
        return "—"
    s = f"{x:+.{n}f}" if sinal else f"{x:.{n}f}"
    return s.replace(".", ",")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=ENCODERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--baseline-dir", default=str(REPO / "results"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out-prefix", default=None,
                    help="default: <results-dir>/AVALIACAO_final")
    args = ap.parse_args()

    rdir, bdir = Path(args.results_dir), Path(args.baseline_dir)
    prefix = Path(args.out_prefix) if args.out_prefix else rdir / "AVALIACAO_final"
    exigir_saida_isolada(prefix)
    sig_dir = prefix.parent / "significancia"

    comparadores = {}
    for tipo, stem in (("baseline", "baseline_{e}_seed{s}"), ("filtro", "filtro_{e}_seed{s}")):
        for e in ENCODERS:
            for s in BASE_SEEDS:
                nome = stem.format(e=e, s=s)
                d = json.loads((bdir / f"{nome}.preds.json").read_text(encoding="utf-8"))
                comparadores[nome] = {"tipo": tipo, "encoder": e, "seed": s,
                                      "path": bdir / f"{nome}.preds.json",
                                      "y_true": d["y_true"]}
    y_true = next(iter(comparadores.values()))["y_true"]
    assert all(c["y_true"] == y_true for c in comparadores.values())

    execs, gpu_s = {}, 0.0
    for e in args.encoders:
        for s in args.seeds:
            stem = f"restrito_{e}_seed{s}"
            p = rdir / f"{stem}.preds.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            assert d["labels"] == LABELS and d["y_true"] == y_true and len(d["y_pred"]) == 19210
            rj = json.loads((rdir / f"{stem}.json").read_text(encoding="utf-8"))
            dur = sum(h.get("duration_s", 0.0) for h in rj["dev_history"])
            gpu_s += dur
            execs[stem] = {"encoder": e, "seed": s, "path": p, "duracao_s": dur,
                           "f1": rj["test_f1_per_class"]["negation_of"],
                           "best_epoch": rj["instrumentation"]["best_epoch"],
                           "gpu": rj["environment"].get("gpu")}

    pares = []
    for stem, ex in execs.items():
        for nome, c in comparadores.items():
            sig = rodar_significance(ex["path"], c["path"],
                                     sig_dir / f"sig_{stem}_vs_{nome}.json",
                                     ex["seed"], args.n_boot)
            bs = sig["paired_bootstrap"]
            pares.append({"restrito": stem, "encoder": ex["encoder"], "seed": ex["seed"],
                          "comparador": nome, "tipo": c["tipo"],
                          "f1_restrito": sig["target_f1"]["a"],
                          "f1_comparador": sig["target_f1"]["b"],
                          "diff": sig["target_f1"]["a_minus_b"],
                          "ic95_bootstrap": [bs["ci95_low"], bs["ci95_high"]],
                          "p_bootstrap": bs["p_value"],
                          "mcnemar_nao_interpretavel": sig["mcnemar"]})

    agreg = {}
    for e in args.encoders:
        f1s = [ex["f1"] for ex in execs.values() if ex["encoder"] == e]
        agreg[e] = {"f1_restrito": {"media": statistics.fmean(f1s),
                                    "dp": statistics.stdev(f1s) if len(f1s) > 1 else None,
                                    "min": min(f1s), "max": max(f1s), "n": len(f1s)},
                    "por_comparador": {}}
        for nome, c in comparadores.items():
            ps = [p for p in pares if p["encoder"] == e and p["comparador"] == nome]
            ic = ic_t([p["diff"] for p in ps])
            agreg[e]["por_comparador"][nome] = {
                "tipo": c["tipo"], **ic, "leitura": classifica(ic),
                "bootstrap_ic_acima_de_zero": sum(1 for p in ps if p["ic95_bootstrap"][0] > 0),
                "bootstrap_ic_abaixo_de_zero": sum(1 for p in ps if p["ic95_bootstrap"][1] < 0),
                "n": len(ps)}

    base_f1 = {n: next(p["f1_comparador"] for p in pares if p["comparador"] == n)
               for n in comparadores}
    mais_forte = max((n for n in comparadores if comparadores[n]["tipo"] == "baseline"),
                     key=lambda n: base_f1[n])
    veredito = {}
    for e in args.encoders:
        pc = agreg[e]["por_comparador"]
        pior = min((ex for ex in execs.values() if ex["encoder"] == e), key=lambda x: x["f1"])
        pior_par = next(p for p in pares if p["restrito"] == f"restrito_{e}_seed{pior['seed']}"
                        and p["comparador"] == mais_forte)
        a_ok = (all(pc[n]["leitura"] == "supera" for n in pc if pc[n]["tipo"] == "baseline")
                and pior_par["ic95_bootstrap"][0] > 0)
        leit_f = {n: pc[n]["leitura"] for n in pc if pc[n]["tipo"] == "filtro"}
        if all(v == "supera" for v in leit_f.values()):
            b = "supera o filtro (as 4 execucoes filtradas)"
        elif all(v == "fica atras" for v in leit_f.values()):
            b = "fica atras do filtro (as 4 execucoes filtradas)"
        else:
            b = "empata ou resultado misto contra o filtro: " + ", ".join(
                f"{n}={v}" for n, v in leit_f.items())
        veredito[e] = {"a_supera_baselines_com_significancia": a_ok,
                       "pior_caso": pior_par, "b_contra_filtro": b}

    base_gpu_s = 0.0
    for e in ENCODERS:
        for s in BASE_SEEDS:
            bj = json.loads((bdir / f"baseline_{e}_seed{s}.json").read_text(encoding="utf-8"))
            base_gpu_s += sum(h.get("duration_s", 0.0) for h in bj["dev_history"])

    rel = {"data": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "encoders": args.encoders,
           "seeds": args.seeds, "n_boot": args.n_boot,
           "execucoes": {k: {kk: (str(vv) if kk == "path" else vv) for kk, vv in v.items()}
                         for k, v in execs.items()},
           "comparadores_f1": base_f1, "baseline_mais_forte": mais_forte,
           "agregado": agreg, "pares": pares, "veredito": veredito,
           "custo": {"gpu_restrito_h": gpu_s / 3600,
                     "gpu_baselines_4_execucoes_h": base_gpu_s / 3600,
                     "gpu_filtro_h": 0.0},
           "nota_mcnemar": "McNemar mede acerto nas 3 classes; o restrito zera "
                           "associated_with por construcao. Nao interpretar."}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(rel, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8", newline="\n")

    md = ["# Avaliação final do fine-tuning restrito (Tarefa 4)", "",
          f"`{rel['data']}` · TEST remapeado, 19.210 pares · bootstrap {args.n_boot}× · "
          f"sementes {args.seeds}", ""]
    for e in args.encoders:
        a = agreg[e]
        md += [f"## {e}", "",
               f"F1 `negation_of` do restrito: média {fmt(a['f1_restrito']['media'])}, "
               f"dp {fmt(a['f1_restrito']['dp'])}, de {fmt(a['f1_restrito']['min'])} a "
               f"{fmt(a['f1_restrito']['max'])} (n={a['f1_restrito']['n']}).", "",
               "| comparador | F1 comparador | Δ médio | IC95 entre sementes | leitura | bootstrap IC>0 |",
               "|---|---|---|---|---|---|"]
        for n, c in a["por_comparador"].items():
            md.append(f"| {n} | {fmt(base_f1[n])} | {fmt(c['media'], sinal=True)} | "
                      f"[{fmt(c['ic95'][0], sinal=True)}; {fmt(c['ic95'][1], sinal=True)}] | "
                      f"{c['leitura']} | {c['bootstrap_ic_acima_de_zero']}/{c['n']} |")
        v = veredito[e]
        pp = v["pior_caso"]
        md += ["", f"- (a) supera os 4 baselines com significância: "
               f"**{'SIM' if v['a_supera_baselines_com_significancia'] else 'NÃO'}**. Pior caso "
               f"{pp['restrito']} vs {pp['comparador']}: Δ {fmt(pp['diff'], sinal=True)}, IC95 "
               f"[{fmt(pp['ic95_bootstrap'][0], sinal=True)}; {fmt(pp['ic95_bootstrap'][1], sinal=True)}].",
               f"- (b) contra o filtro da fase 2: {v['b_contra_filtro']}.", ""]
    md += ["## Custo", "",
           f"GPU do restrito: {fmt(gpu_s / 3600, 2)} h. GPU dos 4 baselines: "
           f"{fmt(base_gpu_s / 3600, 2)} h. Filtro: 0 h (pós-processamento).", "",
           "O McNemar fica nos JSONs de `significancia/`, mas não é interpretável "
           "aqui: ele mede acerto nas três classes e o restrito devolve "
           "`no_relation` para quase todo `associated_with` por construção."]
    prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    log.info("Avaliacao final gravada em %s(.json|.md)", prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
