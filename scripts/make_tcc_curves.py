"""
Gera as curvas de treino/validacao por epoca do TCC a partir do campo
`dev_history` de `results/baseline_*.json`.

POR QUE ESTE SCRIPT EXISTE
--------------------------
`src/relation_extraction.py` grava, a cada epoca, `train_loss`, `dev_loss`,
`dev_macro_f1`, `dev_negation_of_f1` e a duracao. Esse historico nunca foi
lido por nenhum artefato: nem o artigo SBC nem o TCC mostravam como o ajuste
evoluiu. Sem ele, a selecao de modelo ("melhor epoca pelo macro-F1 no dev")
e uma afirmacao sem evidencia visivel, e a instabilidade do encoder clinico
entre sementes so aparece no numero final, nunca no percurso.

O QUE GERA
----------
Figuras PNG em `tcc/src/imagens/resultados/`:

  curvas_treino.png            -> \\label{fig:curvas}
                                  painel 2x2: uma linha por modelo,
                                  perda a esquerda, F1 no dev a direita,
                                  as duas sementes sobrepostas
  curvas_treino_biobertpt.png  -> \\label{fig:curvas_biobertpt}
  curvas_treino_bertimbau.png  -> \\label{fig:curvas_bertimbau}
                                  o mesmo par de eixos, um modelo por vez,
                                  para quem preferir duas figuras no texto

Tabela em `tcc/src/tabelas/`:

  dev_history.tex -> Tabela (\\label{tab:dev_history})
                     os mesmos valores em numero, com a duracao por epoca e
                     a epoca selecionada marcada

EPOCA SELECIONADA
-----------------
E o argmax de `dev_macro_f1`, o mesmo criterio que `relation_extraction.py`
usa para salvar `best_model/`. Aqui ele e recalculado do historico em vez de
lido de um campo: se o criterio mudasse no treino sem mudar aqui, a marca no
grafico deixaria de bater com o checkpoint -- e o teste
`--check-selected-epoch` acusa isso comparando com as metricas de teste.

CONVENCOES DO GRAFICO
---------------------
Escala de cinza (o TCC pode ser impresso em P&B): a semente e codificada por
estilo de linha (42 = solida, 43 = tracejada) e a metrica por marcador, nao
por cor. Marcador preenchido = epoca selecionada.

USO
---
    python scripts/make_tcc_curves.py
    python scripts/make_tcc_curves.py --seeds 42 43
    python scripts/make_tcc_curves.py --check
    python scripts/make_tcc_curves.py --out-dir /tmp/conferencia

Opcoes: --results-dir, --out-dir, --figs-dir, --seeds, --check, --dpi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display: roda em CI, servidor e Colab headless.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _artifacts import (
    BASELINE_ORDER,
    DEFAULT_RESULTS_DIR,
    SEEDS,
    TCC_RESULT_FIGS_DIR,
    TCC_TABLES_DIR,
    MissingResultError,
    abnt_float,
    display_path,
    label_for,
    load_baseline,
    ptbr,
    tt,
    write_tex,
)

SCRIPT = "make_tcc_curves.py"

# Semente -> estilo de linha. Codificar a semente por estilo (e nao por cor)
# mantem o grafico legivel impresso em preto e branco.
SEED_STYLES = {42: "-", 43: "--"}

# Metrica -> (rotulo na legenda, marcador, cor).
LOSS_SERIES = [
    ("train_loss", "Perda (treino)", "o", "#3d3d3d"),
    ("dev_loss", "Perda (validação)", "s", "#a0a0a0"),
]
F1_SERIES = [
    ("dev_macro_f1", "Macro-F1 (validação)", "o", "#3d3d3d"),
    ("dev_negation_of_f1", r"F1 negation_of (validação)", "s", "#a0a0a0"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def history_of(data: dict) -> list[dict]:
    """`dev_history` validado.

    Historico vazio ou com epocas fora de ordem produziria um grafico
    silenciosamente errado, entao a checagem e explicita.
    """
    history = data.get("dev_history")
    if not history:
        raise ValueError(
            f"{data['model']} (semente {data['seed']}) nao tem `dev_history`. "
            f"O treino foi feito por uma versao do pipeline anterior ao "
            f"registro de curvas -- retreine ou remova esta semente."
        )
    required = {"epoch", "train_loss", "dev_loss", "dev_macro_f1", "dev_negation_of_f1"}
    missing = required - set(history[0])
    if missing:
        raise ValueError(
            f"`dev_history` de {data['model']} (semente {data['seed']}) nao tem "
            f"os campos {sorted(missing)}."
        )
    epochs = [entry["epoch"] for entry in history]
    if epochs != sorted(epochs):
        raise ValueError(
            f"`dev_history` de {data['model']} (semente {data['seed']}) esta "
            f"fora de ordem: {epochs}."
        )
    return history


def selected_epoch(history: list[dict]) -> int:
    """Epoca de maior macro-F1 no dev -- o criterio de selecao do treino."""
    return max(history, key=lambda entry: entry["dev_macro_f1"])["epoch"]


# --------------------------------------------------------------------------- #
# Figuras                                                                      #
# --------------------------------------------------------------------------- #
def _plot_panel(ax, runs: dict[int, list[dict]], series, ylabel: str) -> None:
    for seed, history in runs.items():
        epochs = [entry["epoch"] for entry in history]
        best = selected_epoch(history)
        for key, name, marker, color in series:
            values = [entry[key] for entry in history]
            ax.plot(
                epochs,
                values,
                SEED_STYLES.get(seed, "-"),
                marker=marker,
                markersize=6,
                markerfacecolor="white",
                color=color,
                linewidth=1.6,
            )
            # Marcador preenchido na epoca escolhida pelo criterio de
            # selecao: liga a curva ao checkpoint que produziu o teste.
            index = epochs.index(best)
            ax.plot(
                [best],
                [values[index]],
                marker=marker,
                markersize=7,
                color=color,
                linestyle="none",
            )
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({entry["epoch"] for hist in runs.values() for entry in hist}))
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, color="0.8")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Folga vertical antes da legenda: com 3 epocas as curvas ocupam toda a
    # largura, e sem margem a caixa da legenda cobre pontos.
    ax.margins(y=0.24)

    # Legenda fatorada: uma entrada por metrica (cor + marcador) e uma por
    # semente (estilo de linha). O produto cartesiano -- 4 linhas com o nome
    # da metrica repetido -- nao cabia dentro dos eixos.
    handles = [
        Line2D(
            [], [], color=color, marker=marker, markerfacecolor="white",
            linewidth=1.6, label=name,
        )
        for _, name, marker, color in series
    ] + [
        Line2D([], [], color="black", linestyle=SEED_STYLES.get(seed, "-"),
               linewidth=1.6, label=f"semente {seed}")
        for seed in runs
    ]
    ax.legend(handles=handles, fontsize=8.5, framealpha=1.0, edgecolor="0.7")


def figure_model(runs: dict[int, list[dict]], title: str, path: Path, dpi: int) -> None:
    """Uma figura por modelo: perda a esquerda, F1 no dev a direita."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    _plot_panel(axes[0], runs, LOSS_SERIES, "Perda")
    _plot_panel(axes[1], runs, F1_SERIES, "F1")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_grid(
    per_model: dict[str, dict[int, list[dict]]], labels: dict[str, str],
    path: Path, dpi: int,
) -> None:
    """Painel 2x2: uma linha por modelo, perda e F1 lado a lado."""
    fig, axes = plt.subplots(
        len(BASELINE_ORDER), 2, figsize=(10.0, 3.6 * len(BASELINE_ORDER))
    )
    for row, slug in enumerate(BASELINE_ORDER):
        runs = per_model[slug]
        _plot_panel(axes[row][0], runs, LOSS_SERIES, "Perda")
        _plot_panel(axes[row][1], runs, F1_SERIES, "F1")
        axes[row][0].set_title(labels[slug], loc="left", fontsize=12)
        axes[row][1].set_title(labels[slug], loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tabela                                                                       #
# --------------------------------------------------------------------------- #
def build_history_table(
    per_model: dict[str, dict[int, list[dict]]],
    labels: dict[str, str],
    seeds: list[int],
) -> str:
    rows = []
    for slug in BASELINE_ORDER:
        for seed in seeds:
            history = per_model[slug][seed]
            best = selected_epoch(history)
            for index, entry in enumerate(history):
                # Modelo e semente so na primeira linha do bloco: repetir em
                # toda linha polui a leitura de uma tabela de 12 linhas.
                model_cell = labels[slug] if (seed == seeds[0] and index == 0) else ""
                seed_cell = str(seed) if index == 0 else ""
                mark = r"$\ast$" if entry["epoch"] == best else ""
                rows.append(
                    f"  {model_cell} & {seed_cell} & {entry['epoch']}{mark} & "
                    f"{ptbr(entry['train_loss'], 4)} & "
                    f"{ptbr(entry['dev_loss'], 4)} & "
                    f"{ptbr(entry['dev_macro_f1'])} & "
                    f"{ptbr(entry['dev_negation_of_f1'])} & "
                    f"{ptbr(entry['duration_s'] / 60, 1)} \\\\"
                )
            rows.append(r"  \midrule")
    rows = rows[:-1]  # o ultimo \midrule sobra antes do \bottomrule

    body = "\n".join(
        [
            r"  \begin{tabular}{llcccccc}",
            r"  \toprule",
            r"  Modelo & Semente & Época & Perda (treino) & Perda (val.) & "
            r"Macro-F1 (val.) & F1 " + tt("negation_of") + r" (val.) & "
            r"Duração (min) \\",
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    return abnt_float(
        body,
        caption=(
            "Evolução por época no conjunto de validação. "
            r"$\ast$ marca a época selecionada (maior macro-F1 na validação), "
            r"cujo \emph{checkpoint} produziu as métricas de teste."
        ),
        label="tab:dev_history",
        source=(
            r"elaborado pelo autor (campo \texttt{dev\_history} de "
            r"\texttt{results/baseline\_*.json})."
        ),
        resize=True,
    )


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera as curvas de treino/validacao por epoca do TCC.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TCC_TABLES_DIR,
        help="diretório da tabela (padrão: tcc/src/tabelas).",
    )
    parser.add_argument(
        "--figs-dir",
        type=Path,
        default=TCC_RESULT_FIGS_DIR,
        help="diretório das figuras (padrão: tcc/src/imagens/resultados).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="sementes plotadas em cada painel (padrão: 42 43).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="imprime a tabela na saída padrão sem gravar arquivos.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        per_model: dict[str, dict[int, list[dict]]] = {}
        labels: dict[str, str] = {}
        for slug in BASELINE_ORDER:
            per_model[slug] = {}
            for seed in args.seeds:
                data = load_baseline(args.results_dir, slug, seed)
                labels[slug] = label_for(data)
                per_model[slug][seed] = history_of(data)
    except (MissingResultError, ValueError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1

    table = build_history_table(per_model, labels, args.seeds)

    if args.check:
        print("===== dev_history.tex =====")
        print(table)
        for slug in BASELINE_ORDER:
            for seed in args.seeds:
                print(
                    f"{slug} seed{seed}: época selecionada = "
                    f"{selected_epoch(per_model[slug][seed])}"
                )
        return 0

    path = args.out_dir / "dev_history.tex"
    write_tex(
        path,
        table,
        script=SCRIPT,
        sources="campo dev_history de results/baseline_*.json",
    )
    print(f"escrito: {display_path(path)}")

    apply_style()
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    path = args.figs_dir / "curvas_treino.png"
    figure_grid(per_model, labels, path, args.dpi)
    print(f"escrito: {display_path(path)}")

    for slug in BASELINE_ORDER:
        path = args.figs_dir / f"curvas_treino_{slug}.png"
        figure_model(per_model[slug], labels[slug], path, args.dpi)
        print(f"escrito: {display_path(path)}")

    print(
        f"\n1 tabela e {1 + len(BASELINE_ORDER)} figura(s) de curvas "
        f"({len(BASELINE_ORDER)} modelos x {len(args.seeds)} sementes)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
