"""
Regenera as figuras do artigo a partir de `results/*.json`.

POR QUE ESTE SCRIPT EXISTE
--------------------------
As tres figuras de `artigo-sbc/figs/` foram produzidas fora do repositorio,
em notebooks descartados. Nao havia caminho de `results/*.json` ate o PDF --
a maior lacuna de reprodutibilidade do projeto. Este script fecha esse
caminho.

O QUE GERA
----------
Em `artigo-sbc/figs_generated/`, em PDF (vetorial, mesmo formato dos
originais):

  f1_por_classe.pdf   -> Figura 1 (\\label{fig:f1classe})
                         barras agrupadas: F1 por classe, os dois baselines
  cm_biobertpt.pdf    -> Figura 2 (\\label{fig:cmbio})
                         matriz de confusao do BioBERTpt, normalizada por linha
  cm_bertimbau.pdf    -> Figura 3 (\\label{fig:cmber})
                         idem, BERTimbau

Normalizacao por linha: cada celula e a fracao das instancias de uma classe
verdadeira atribuida a cada classe predita, entao a diagonal e o recall por
classe. E a leitura declarada nas legendas do artigo.

Este script NUNCA escreve em `artigo-sbc/figs/`. A substituicao dos originais
e uma decisao manual.

ESTILO
------
Escala de cinza e fonte serifada, para o artigo continuar legivel impresso em
preto e branco (criterio dos PDFs originais). Fontes sao embutidas como
TrueType (fonttype 42), exigencia comum de submissao.

USO
---
    python scripts/make_figures.py                 # semente 42 (a do artigo)
    python scripts/make_figures.py --seed 43
    python scripts/make_figures.py --format png    # para inspecao rapida

Opcoes: --results-dir, --out-dir, --seed, --format, --dpi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display: roda em CI, servidor e Colab headless.

import matplotlib.pyplot as plt
import numpy as np

from _artifacts import (
    BASELINE_ORDER,
    CLASS_ORDER,
    DEFAULT_RESULTS_DIR,
    REPO_ROOT,
    MissingResultError,
    display_path,
    label_for,
    load_baselines,
    reorder,
    row_normalized,
    slug_for,
)

DEFAULT_OUT_DIR = REPO_ROOT / "artigo-sbc" / "figs_generated"

# Cinzas das barras: escuro para o encoder clinico, claro para o geral.
# Contraste suficiente para sobreviver a impressao em preto e branco.
BAR_COLORS = ["#3d3d3d", "#bfbfbf"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            # DejaVu Serif acompanha o matplotlib, entao a figura sai igual
            # em qualquer maquina, sem depender de fonte instalada.
            "font.serif": ["DejaVu Serif"],
            "font.size": 13,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def figure_f1_per_class(baselines: dict[str, dict], out_path: Path, dpi: int) -> None:
    """Figura 1: F1 por classe, barras agrupadas, dois baselines lado a lado."""
    positions = np.arange(len(CLASS_ORDER))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    for offset, slug in enumerate(BASELINE_ORDER):
        data = baselines[slug]
        scores = [data["test_f1_per_class"][name] for name in CLASS_ORDER]
        bars = ax.bar(
            positions + (offset - 0.5) * width,
            scores,
            width,
            label=label_for(data),
            color=BAR_COLORS[offset],
            edgecolor="black",
            linewidth=0.8,
        )
        # Rotulo em 2 casas sobre cada barra: e o que permite ler a figura
        # sem recorrer a Tabela 1.
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=12)

    ax.set_ylabel("F1")
    # Folga acima de 1,0 para os rótulos das barras mais altas (no_relation,
    # ~0,94) e para a legenda não cobrir nenhuma barra.
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks(positions)
    ax.set_xticklabels(CLASS_ORDER, rotation=20, ha="right")
    # 'upper left' fica sobre a classe mais baixa do gráfico (negation_of,
    # ~0,73), o único canto livre em todas as sementes rodadas.
    ax.legend(loc="upper left", framealpha=1.0, edgecolor="0.7", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def figure_confusion_matrix(data: dict, out_path: Path, dpi: int) -> None:
    """Figuras 2 e 3: matriz de confusao normalizada por linha."""
    confusion = data["confusion_matrix"]
    ordered = reorder(confusion["labels"], confusion["matrix"])
    normalized = np.array(row_normalized(ordered))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.imshow(normalized, cmap="Greys", vmin=0.0, vmax=1.0)

    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            value = normalized[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                # Texto branco sobre celula escura, preto sobre clara.
                color="white" if value > 0.5 else "black",
                fontsize=14,
            )

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Classe predita", labelpad=10)
    ax.set_ylabel("Classe verdadeira", labelpad=10)
    ax.set_title(label_for(data), pad=12)
    ax.tick_params(length=3)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera as figuras do artigo a partir de results/*.json.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="semente cujos resultados alimentam as figuras (padrão: 42).",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="formato de saída (padrão: pdf, o mesmo dos originais).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    try:
        baselines = load_baselines(args.results_dir, args.seed)
    except (MissingResultError, ValueError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []

    path = args.out_dir / f"f1_por_classe.{args.format}"
    figure_f1_per_class(baselines, path, args.dpi)
    written.append(path)

    for slug in BASELINE_ORDER:
        data = baselines[slug]
        path = args.out_dir / f"cm_{slug_for(data)}.{args.format}"
        figure_confusion_matrix(data, path, args.dpi)
        written.append(path)

    for path in written:
        print(f"escrito: {display_path(path)}")
    print(f"\n{len(written)} figura(s) da semente {args.seed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
