#!/usr/bin/env python3
"""Estatísticas complementares do corpus (dados v2-texto-fonte), calculadas com as
funções do próprio repositório (`reclin.particoes`, `reclin.tarefa.entity_gap`,
`reclin.tarefa.iter_candidate_pairs`). Só descreve o corpus: não usa modelos nem
predições. Gera um JSON e as figuras de distância e de relações por documento.

Uso: PYTHONPATH=<repo>/codigo python3 eda_complementar.py <pasta particoes-v2> <saida_dir>
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from reclin import particoes
from reclin.tarefa import MAX_GAP, entity_gap, iter_candidate_pairs


def pct(valores, q):
    v = sorted(valores)
    k = (len(v) - 1) * q
    f, c = int(k), min(int(k) + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def main(pasta: Path, saida: Path) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    por = particoes.ler_particoes(pasta)
    docs = [d for nome in ("train", "dev", "test") for d in por[nome]]
    gaps = {"negation_of": [], "associated_with": []}
    rel_por_doc, atomicos = [], Counter()
    n_ent = 0
    for d in docs:
        ents = {e["id"]: e for e in d["entities"]}
        n_ent += len(ents)
        for e in d["entities"]:
            atomicos.update(t for t in e["type"].split("|"))
        rel_por_doc.append(len(d["relations"]))
        for r in d["relations"]:
            e1, e2 = ents.get(r["e1_id"]), ents.get(r["e2_id"])
            if e1 is None or e2 is None:
                continue
            gaps[r["type"]].append(entity_gap(e1, e2))
    resumo_gap = {t: {"n": len(v), "media": statistics.mean(v), "mediana": statistics.median(v),
                      "p75": pct(v, .75), "p95": pct(v, .95), "max": max(v),
                      "frac_ate_1": sum(1 for x in v if x <= 1) / len(v),
                      "frac_acima_max_gap": sum(1 for x in v if x > MAX_GAP) / len(v)} for t, v in gaps.items()}
    cand = {}
    for nome in ("train", "dev", "test"):
        c = Counter(x["label"] for d in por[nome] for x in iter_candidate_pairs(d, max_gap=MAX_GAP))
        cand[nome] = dict(c)
    res = {
        "n_documentos": len(docs), "n_entidades": n_ent,
        "n_relacoes": sum(rel_por_doc),
        "relacoes_por_documento": {"media": statistics.mean(rel_por_doc), "mediana": statistics.median(rel_por_doc),
                                   "p90": pct(rel_por_doc, .90), "max": max(rel_por_doc),
                                   "docs_sem_relacao": sum(1 for x in rel_por_doc if x == 0)},
        "distancia_relacoes": resumo_gap,
        "categorias_atomicas_top15": atomicos.most_common(15),
        "candidatos_por_particao": cand,
        "max_gap": MAX_GAP,
    }
    (saida / "eda_complementar.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
    cores = {"negation_of": "#2a78d6", "associated_with": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bins = list(range(0, 201, 5))
    for t in ("associated_with", "negation_of"):
        v = [min(x, 200) for x in gaps[t]]
        ax.hist(v, bins=bins, alpha=.75, label=t, color=cores[t], density=True)
    ax.axvline(MAX_GAP, color="black", ls="--", lw=1.2)
    ax.text(MAX_GAP + 3, ax.get_ylim()[1] * .85, "max_gap = 25", fontsize=10)
    ax.set_xlabel("distância entre as entidades (caracteres, recorte em 200)")
    ax.set_ylabel("densidade")
    ax.legend(frameon=False)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(saida / "eda_distancia_v2.png", dpi=200)
    plt.close(fig)
    print("gravado", saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
