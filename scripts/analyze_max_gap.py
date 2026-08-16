#!/usr/bin/env python3
"""
Diagnostico da janela `max_gap` da geracao de pares-candidatos.

PARA QUE SERVE
--------------
`src/candidates.py` so emite um par ordenado (e1, e2) quando `entity_gap(e1, e2)
<= max_gap`. Toda relacao gold que caia fora da janela e IRRECUPERAVEL por
qualquer classificador: ela nunca vira candidato. Do outro lado, alargar a janela
so cria pares negativos (`no_relation`). Este script mede os dois lados sobre o
corpus anotado e produz a figura que mostra o trade-off.

DEFINICAO DE `gap` (nao e "quantidade de entidades")
---------------------------------------------------
`entity_gap` (importado de `src/candidates.py`, nao reimplementado aqui) e a
DISTANCIA EM CARACTERES entre os dois spans: o numero de caracteres do texto que
ficam ENTRE o fim do span que comeca antes e o inicio do outro. E simetrica
(nao depende da ordem do par) e vale 0 para spans adjacentes, sobrepostos ou
aninhados. `max_gap=25` portanto significa "no maximo 25 caracteres de texto
separando as duas entidades" -- da ordem de 4 palavras, nao 25 entidades.

SO LEITURA
----------
Le apenas `--input` (default `data/processed/dataset.jsonl`, a saida do parser,
antes dos splits) e escreve apenas dentro de `--out-dir`. Nao toca em
`data/splits/`, `results/` nem em nenhuma config do pipeline.

USO
---
    python scripts/analyze_max_gap.py
    python scripts/analyze_max_gap.py --input data/processed/dataset.jsonl \
        --out-dir analysis/max_gap --gaps 1,2,3,5,8,10,15,20,30,50,75,100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from candidates import entity_gap, iter_candidate_pairs  # noqa: E402

RELATION_TYPES = ("associated_with", "negation_of")
PERCENTILES = (50, 75, 90, 95, 99)

DEFAULT_GAPS = (1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 75, 100, 150, 200)
CURRENT_MAX_GAP = 25  # o que o Makefile passa hoje (--max-gap 25)
# Era 20 ate esta analise; elevado para 25 para recuperar as 1.272 relacoes
# `associated_with` que 20 descartava. A figura/JSON ja gravados em
# analysis/max_gap/ sao os da rodada com current=20 (registro da decisao):
# para reproduzi-los, passe `--current 20`.

# Paleta: slots categoricos 1-3 (azul/laranja/verde-agua), validados para
# daltonismo em todos os pares. Series tambem sao rotuladas direto na figura,
# entao a identidade nunca depende so da cor.
C_AW = "#2a78d6"       # associated_with
C_NEG = "#eb6834"      # negation_of
C_ALL = "#1baf7a"      # geral / no_relation
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------- #
# Leitura                                                                      #
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# --------------------------------------------------------------------------- #
# Passo 2 -- distancia real nas relacoes gold                                  #
# --------------------------------------------------------------------------- #
def gold_gaps(docs: list[dict]) -> tuple[dict[str, np.ndarray], dict]:
    """Gap (em caracteres) de cada relacao anotada, por tipo.

    Devolve tambem o "accounting" de relacoes que o gerador nunca emitiria
    INDEPENDENTE da janela -- referencia quebrada, autopar, ou spans identicos
    (as tres exclusoes de `iter_candidate_pairs` que nao sao a janela). Elas nao
    entram na distribuicao porque nao dizem nada sobre a escolha de `max_gap`.
    """
    per_type: dict[str, list[int]] = {t: [] for t in RELATION_TYPES}
    other_types: dict[str, int] = {}
    acc = {"total": 0, "broken_ref": 0, "self_pair": 0, "same_span": 0,
           "duplicates": 0}

    for doc in docs:
        index = {e["id"]: e for e in doc["entities"]}
        seen: set[tuple[str, str, str]] = set()
        for rel in doc["relations"]:
            acc["total"] += 1
            key = (rel["e1_id"], rel["e2_id"], rel["type"])
            if key in seen:
                acc["duplicates"] += 1
            seen.add(key)

            e1, e2 = index.get(rel["e1_id"]), index.get(rel["e2_id"])
            if e1 is None or e2 is None:
                acc["broken_ref"] += 1
                continue
            if rel["e1_id"] == rel["e2_id"]:
                acc["self_pair"] += 1
                continue
            if (e1["start"], e1["end"]) == (e2["start"], e2["end"]):
                acc["same_span"] += 1
                continue
            if rel["type"] in per_type:
                per_type[rel["type"]].append(entity_gap(e1, e2))
            else:
                other_types[rel["type"]] = other_types.get(rel["type"], 0) + 1

    acc["other_types"] = other_types
    return {t: np.asarray(v, dtype=int) for t, v in per_type.items()}, acc


def describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    out = {"n": int(values.size), "mean": float(values.mean()),
           "std": float(values.std(ddof=0)), "min": int(values.min()),
           "max": int(values.max())}
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(values, p))
    return out


# --------------------------------------------------------------------------- #
# Passo 3 -- custo: quantos pares candidatos cada janela produz                #
# --------------------------------------------------------------------------- #
def candidate_gap_histogram(docs: list[dict]) -> np.ndarray:
    """Histograma dos gaps de TODOS os pares que o gerador consideraria.

    Aplica as mesmas exclusoes de `iter_candidate_pairs` menos a janela (pares
    ordenados, sem autopar, sem spans identicos). Com o histograma, o numero de
    candidatos de qualquer `max_gap` sai de uma soma cumulativa -- em vez de
    re-varrer o corpus O(n^2) uma vez por valor testado.
    """
    hist = np.zeros(1, dtype=np.int64)
    for doc in docs:
        ents = sorted(doc["entities"], key=lambda e: (e["start"], e["end"], e["id"]))
        gaps = []
        for i, e1 in enumerate(ents):
            for j, e2 in enumerate(ents):
                if i == j:
                    continue
                if (e1["start"], e1["end"]) == (e2["start"], e2["end"]):
                    continue
                gaps.append(entity_gap(e1, e2))
        if not gaps:
            continue
        local = np.bincount(np.asarray(gaps, dtype=int))
        if local.size > hist.size:
            local[: hist.size] += hist
            hist = local
        else:
            hist[: local.size] += local
    return hist


def cumulative(hist: np.ndarray, g: int) -> int:
    """Pares com gap <= g."""
    return int(hist[: g + 1].sum())


def sensitivity_table(gaps_by_type: dict[str, np.ndarray], hist: np.ndarray,
                      grid: list[int]) -> list[dict]:
    all_gaps = np.concatenate([v for v in gaps_by_type.values() if v.size])
    rows = []
    for g in grid:
        captured = int((all_gaps <= g).sum())
        pairs = cumulative(hist, g)
        no_rel = pairs - captured
        row = {
            "max_gap": g,
            "n_gold": int(all_gaps.size),
            "captured": captured,
            "lost": int(all_gaps.size - captured),
            "recall_ceiling_pct": 100.0 * captured / all_gaps.size,
            "n_candidates": pairs,
            "no_relation": no_rel,
            "neg_per_pos": no_rel / captured if captured else float("nan"),
        }
        for t, v in gaps_by_type.items():
            row[f"recall_{t}_pct"] = 100.0 * float((v <= g).mean()) if v.size else 0.0
            row[f"lost_{t}"] = int((v > g).sum())
        rows.append(row)

    # custo marginal: quantos negativos a mais por relacao gold a mais
    prev = None
    for row in rows:
        if prev is None:
            row["marginal_neg_per_gold"] = float("nan")
        else:
            d_gold = row["captured"] - prev["captured"]
            d_neg = row["no_relation"] - prev["no_relation"]
            row["marginal_neg_per_gold"] = d_neg / d_gold if d_gold else float("inf")
        prev = row
    return rows


# --------------------------------------------------------------------------- #
# Verificacao                                                                  #
# --------------------------------------------------------------------------- #
def verify_against_generator(docs: list[dict], grid: list[int],
                             hist: np.ndarray, n_docs: int = 60) -> list[str]:
    """Confere a contagem rapida (histograma) contra o gerador de verdade.

    Roda `iter_candidate_pairs` -- a funcao que o pipeline usa -- num
    subconjunto de documentos e exige contagem identica.
    """
    sample = docs[:n_docs]
    sample_hist = candidate_gap_histogram(sample)
    msgs = []
    for g in grid:
        if g > 100:
            continue
        truth = sum(1 for d in sample for _ in iter_candidate_pairs(d, max_gap=g))
        mine = cumulative(sample_hist, g)
        status = "ok" if truth == mine else "DIVERGENTE"
        msgs.append(f"  max_gap={g:<4d} gerador={truth:<8d} histograma={mine:<8d} {status}")
        if truth != mine:
            raise AssertionError(
                f"reimplementacao divergiu de iter_candidate_pairs em max_gap={g}")
    assert hist.size >= sample_hist.size
    return msgs


# --------------------------------------------------------------------------- #
# Passo 4 -- figura                                                            #
# --------------------------------------------------------------------------- #
def make_figure(gaps_by_type: dict[str, np.ndarray], rows: list[dict],
                out_path: Path, clip: int, dpi: int, current: int,
                fig_max_gap: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    plotted = [r for r in rows if r["max_gap"] <= fig_max_gap]
    x = [r["max_gap"] for r in plotted]
    x_right = fig_max_gap * 1.22   # margem para os rotulos diretos a direita

    fig = plt.figure(figsize=(10.0, 9.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.98, 0.95], hspace=0.46,
                          left=0.085, right=0.985, top=0.905, bottom=0.062)
    ax_h = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    ax_n = fig.add_subplot(gs[2], sharex=ax_r)

    for ax in (ax_h, ax_r, ax_n):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color("#c3c2b7")

    # ------------- (a) perfil da distancia nas relacoes anotadas ------------ #
    # Normalizado POR CLASSE: negation_of tem 1/6 do volume de
    # associated_with; em contagem bruta o perfil dela sumiria, e o ponto do
    # painel e justamente comparar os dois PERFIS.
    aw = gaps_by_type["associated_with"]
    neg = gaps_by_type["negation_of"]
    centers = np.arange(0, clip + 1)

    def share(values):
        counts = np.bincount(np.clip(values, 0, clip), minlength=clip + 1)
        return 100.0 * counts / values.size

    ax_h.bar(centers - 0.21, share(aw), width=0.40, color=C_AW,
             label=f"associated_with  (n={aw.size:,})".replace(",", "."))
    ax_h.bar(centers + 0.21, share(neg), width=0.40, color=C_NEG,
             label=f"negation_of  (n={neg.size:,})".replace(",", "."))

    ax_h.set_xlim(-0.8, clip + 0.8)
    ax_h.set_ylim(0, 88)

    ax_h.axvline(current, color=INK, linestyle="--", linewidth=1.5, zorder=1)
    ax_h.annotate(f"max_gap atual = {current}", xy=(current, 68),
                  xytext=(7, 0), textcoords="offset points",
                  color=INK, fontsize=8.5, fontweight="bold", va="center")

    for values, color, name, y0, ha, dx in (
            (neg, C_NEG, "negation_of", 80, "left", 7),
            (aw, C_AW, "associated_with", 55, "right", -7)):
        p90 = float(np.percentile(values, 90))
        ax_h.axvline(p90, color=color, linestyle=":", linewidth=1.8, zorder=1)
        ax_h.annotate(f"p90 {name} = {p90:.0f}", xy=(p90, y0),
                      xytext=(dx, 0), textcoords="offset points", ha=ha,
                      color=color, fontsize=8.5, fontweight="bold", va="center")

    ax_h.set_title("(a) Distância real entre as entidades das relações anotadas",
                   loc="left", fontsize=11.5, fontweight="bold", pad=8)
    ax_h.set_xlabel("gap = caracteres de texto entre os dois spans")
    ticks = [t for t in range(0, clip + 1, 5)]
    ax_h.set_xticks(ticks)
    ax_h.set_xticklabels([f"\u2265{clip}" if t == clip else str(t) for t in ticks])
    ax_h.set_ylabel("% das relações da classe")
    ax_h.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_h.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2,
                bbox_to_anchor=(1.0, 1.04))
    med_aw = float(np.median(aw))
    med_neg = float(np.median(neg))
    ax_h.annotate(
        f"mediana: {med_aw:.0f} caracteres (associated_with) · "
        f"{med_neg:.0f} caractere (negation_of)\n"
        f"a última barra (\u2265{clip}) agrega a cauda: {int((aw >= clip).sum())} "
        f"associated_with e {int((neg >= clip).sum())} negation_of",
        xy=(0.995, 0.42), xycoords="axes fraction", ha="right", va="top",
        fontsize=8.5, color=INK2,
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=3))

    # ------------- (b) teto de recall --------------------------------------- #
    series = (
        ("geral", [r["recall_ceiling_pct"] for r in plotted], C_ALL, 2.6, 0),
        ("associated_with", [r["recall_associated_with_pct"] for r in plotted],
         C_AW, 2.0, -13),
        ("negation_of", [r["recall_negation_of_pct"] for r in plotted],
         C_NEG, 2.0, 13),
    )
    for name, y, color, lw, dy in series:
        ax_r.plot(x, y, color=color, linewidth=lw, marker="o", markersize=4.5,
                  markeredgecolor=SURFACE, markeredgewidth=1.0, label=name,
                  zorder=3)
        ax_r.annotate(name, xy=(x[-1], y[-1]), xytext=(9, dy),
                      textcoords="offset points", color=color, fontsize=9,
                      fontweight="bold", va="center", zorder=4)

    cur = next(r for r in rows if r["max_gap"] == current)
    ax_r.axvline(current, color=INK, linestyle="--", linewidth=1.5, zorder=1)
    ax_r.annotate(f"max_gap={current}  →  teto de {cur['recall_ceiling_pct']:.1f}%\n"
                  f"{cur['lost']} relações anotadas ({cur['lost_associated_with']} "
                  f"associated_with, {cur['lost_negation_of']} negation_of)\n"
                  f"nunca chegam a virar candidato",
                  xy=(current, cur["recall_ceiling_pct"]), xytext=(16, -62),
                  textcoords="offset points", fontsize=9, color=INK,
                  fontweight="bold",
                  bbox=dict(facecolor=SURFACE, edgecolor="none", pad=3))
    ax_r.set_title("(b) Teto de recall — % das relações anotadas que a janela deixa virar candidato",
                   loc="left", fontsize=11.5, fontweight="bold", pad=8)
    ax_r.set_ylabel("% das relações gold capturadas")
    ax_r.set_ylim(30, 105)
    ax_r.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_r.legend(frameon=False, loc="lower right", fontsize=9, ncol=3,
                labelcolor=INK2, bbox_to_anchor=(0.995, -0.04))
    ax_r.tick_params(labelbottom=True)

    # ------------- (c) custo em candidatos negativos ------------------------ #
    y_neg = [r["no_relation"] for r in plotted]
    ax_n.plot(x, y_neg, color=C_ALL, linewidth=2.6, marker="o", markersize=4.5,
              markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
    ax_n.annotate("no_relation", xy=(x[-1], y_neg[-1]), xytext=(9, 0),
                  textcoords="offset points", color=C_ALL, fontsize=9,
                  fontweight="bold", va="center")
    ax_n.axvline(current, color=INK, linestyle="--", linewidth=1.5, zorder=1)
    ax_n.annotate(f"max_gap={current}\n{cur['no_relation']:,}".replace(",", ".") +
                  f" pares no_relation\n({cur['neg_per_pos']:.0f} por positivo capturado)",
                  xy=(0.28, 0.30), xycoords="axes fraction", ha="left",
                  va="top", fontsize=9, color=INK, fontweight="bold",
                  bbox=dict(facecolor=SURFACE, edgecolor="none", pad=3))
    last = plotted[-1]
    delta_rec = last["recall_ceiling_pct"] - cur["recall_ceiling_pct"]
    factor = last["no_relation"] / cur["no_relation"]
    ax_n.annotate(f"de max_gap={current} para {last['max_gap']}:\n"
                  f"negativos ×{factor:.1f}, por +{delta_rec:.1f} p.p. de teto de recall",
                  xy=(0.36, 0.95), xycoords="axes fraction", ha="left", va="top",
                  fontsize=9, color=INK2,
                  bbox=dict(facecolor=SURFACE, edgecolor="none", pad=3))
    ax_n.set_title("(c) Custo — pares no_relation gerados no corpus inteiro",
                   loc="left", fontsize=11.5, fontweight="bold", pad=8)
    ax_n.set_ylabel("pares no_relation")
    ax_n.set_xlabel("max_gap (caracteres entre os spans)")
    ax_n.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    ax_n.set_xlim(0, x_right)
    ax_n.set_xticks([t for t in (0, 10, 20, 30, 40, 50, 75, 100, 150, 200)
                     if t <= fig_max_gap])

    fig.text(0.085, 0.972, "Janela `max_gap`: o que ela captura x o que ela custa",
             ha="left", fontsize=14, fontweight="bold", color=INK)
    n_gold_fmt = f"{rows[0]['n_gold']:,}".replace(",", ".")
    fig.text(0.085, 0.947,
             f"corpus inteiro ({n_gold_fmt} relações anotadas) · gap = caracteres "
             "entre os spans, definição de src/candidates.entity_gap",
             ha="left", fontsize=9.5, color=INK2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Saida textual                                                                #
# --------------------------------------------------------------------------- #
def fmt_distribution(gaps_by_type: dict[str, np.ndarray]) -> str:
    all_gaps = np.concatenate([v for v in gaps_by_type.values() if v.size])
    header = (f"{'classe':<18} {'n':>6} {'média':>8} {'mediana':>8} {'p75':>7} "
              f"{'p90':>7} {'p95':>7} {'p99':>8} {'máx':>7}")
    lines = [header, "-" * len(header)]
    for name, values in (("GERAL", all_gaps), *gaps_by_type.items()):
        d = describe(values)
        lines.append(f"{name:<18} {d['n']:>6d} {d['mean']:>8.2f} {d['p50']:>8.1f} "
                     f"{d['p75']:>7.1f} {d['p90']:>7.1f} {d['p95']:>7.1f} "
                     f"{d['p99']:>8.1f} {d['max']:>7d}")
    return "\n".join(lines)


def fmt_sensitivity(rows: list[dict], current: int) -> str:
    header = (f"{'max_gap':>7} {'recall':>8} {'perdidas':>9} {'rec_assoc':>10} "
              f"{'rec_neg':>8} {'no_relation':>12} {'neg:pos':>8} {'Δneg/Δgold':>11}")
    lines = [header, "-" * len(header)]
    for r in rows:
        mark = "  <-- atual" if r["max_gap"] == current else ""
        marg = r["marginal_neg_per_gold"]
        marg_s = "-" if marg != marg else f"{marg:>11.0f}"  # NaN check
        lines.append(
            f"{r['max_gap']:>7d} {r['recall_ceiling_pct']:>7.2f}% {r['lost']:>9d} "
            f"{r['recall_associated_with_pct']:>9.2f}% {r['recall_negation_of_pct']:>7.2f}% "
            f"{r['no_relation']:>12,d} {r['neg_per_pos']:>8.1f} {marg_s:>11}{mark}"
            .replace(",", "."))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "data/processed/dataset.jsonl"),
                    help="corpus parseado (antes dos splits)")
    ap.add_argument("--out-dir", default=str(ROOT / "analysis/max_gap"),
                    help="pasta de saida da figura e do JSON (criada se faltar)")
    ap.add_argument("--gaps", default=",".join(str(g) for g in DEFAULT_GAPS),
                    help="valores de max_gap a testar, separados por virgula")
    ap.add_argument("--current", type=int, default=CURRENT_MAX_GAP,
                    help="valor em uso hoje, destacado na figura")
    ap.add_argument("--clip", type=int, default=30,
                    help="limite do eixo x do histograma (painel a); a cauda "
                         "acima disso e reportada em texto, nao empilhada")
    ap.add_argument("--fig-max-gap", type=int, default=100,
                    help="maior max_gap desenhado nos paineis (b) e (c); a "
                         "tabela impressa e o JSON usam --gaps inteiro")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-verify", action="store_true",
                    help="pula a conferencia contra iter_candidate_pairs")
    args = ap.parse_args()

    grid = sorted({int(g) for g in args.gaps.split(",") if g.strip()})
    out_dir = Path(args.out_dir)

    docs = read_jsonl(Path(args.input))
    n_ents = sum(len(d["entities"]) for d in docs)
    print(f"corpus: {len(docs)} documentos | {n_ents} entidades | "
          f"{sum(len(d['relations']) for d in docs)} relações anotadas")
    print(f"fonte:  {args.input}\n")

    print("== Passo 1 — definição de `gap` (de src/candidates.py) ==")
    print(f"  entity_gap = caracteres ENTRE os dois spans; 0 se adjacentes/sobrepostos.")
    print(f"  simétrica: entity_gap(a,b) == entity_gap(b,a). Não é contagem de entidades.")
    print(f"  amostra: entity_gap({{'start':0,'end':5}}, {{'start':12,'end':20}}) = "
          f"{entity_gap({'start': 0, 'end': 5}, {'start': 12, 'end': 20})}\n")

    gaps_by_type, acc = gold_gaps(docs)
    print("== Passo 2 — distribuição do gap nas relações anotadas ==")
    print(fmt_distribution(gaps_by_type))
    print(f"\n  relações lidas: {acc['total']} | ref. quebrada: {acc['broken_ref']} | "
          f"autopar: {acc['self_pair']} | spans idênticos: {acc['same_span']} | "
          f"duplicatas exatas: {acc['duplicates']}")
    if acc["other_types"]:
        print(f"  tipos fora de {RELATION_TYPES}: {acc['other_types']}")
    print()

    hist = candidate_gap_histogram(docs)
    rows = sensitivity_table(gaps_by_type, hist, grid)
    print("== Passo 3 — sensibilidade: teto de recall x max_gap ==")
    print(fmt_sensitivity(rows, args.current))
    print()

    if not args.no_verify:
        print("== Verificação contra src/candidates.iter_candidate_pairs (60 docs) ==")
        for line in verify_against_generator(docs, grid, hist):
            print(line)
        print()

    fig_path = out_dir / "max_gap_tradeoff.png"
    make_figure(gaps_by_type, rows, fig_path, args.clip, args.dpi, args.current,
                args.fig_max_gap)

    payload = {
        "source": str(args.input),
        "gap_definition": ("caracteres entre os spans (src/candidates.entity_gap); "
                           "simétrica; 0 para spans adjacentes ou sobrepostos"),
        "corpus": {"n_docs": len(docs), "n_entities": n_ents,
                   "n_relations": acc["total"]},
        "accounting": acc,
        "distribution": {
            "GERAL": describe(np.concatenate([v for v in gaps_by_type.values()])),
            **{t: describe(v) for t, v in gaps_by_type.items()},
        },
        "sensitivity": rows,
        "current_max_gap": args.current,
    }
    json_path = out_dir / "max_gap_analysis.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print(f"figura: {fig_path}")
    print(f"        {fig_path.with_suffix('.pdf')}")
    print(f"dados:  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
