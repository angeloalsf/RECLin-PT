"""
Regenera a caracterizacao do corpus (Cap. Metodologia do TCC) a partir de
`data/`.

POR QUE ESTE SCRIPT EXISTE
--------------------------
As tabelas e figuras de EDA de `tcc/src/` foram herdadas de uma arvore de
scripts (`scripts/eda/*.py`, saida em `paper/tables/`) que nunca existiu
neste repositorio. Como consequencia, o Cap. de Metodologia descrevia um
corpus com 11.353 relacoes e uma janela `max_gap=200`, enquanto
`data/processed/dataset.jsonl` tem 11.458 relacoes e os experimentos de
`results/` rodavam com outra janela (20 a epoca, 25 desde 15/08/2026). Nao
havia caminho de `data/` ate o PDF do TCC. Este script fecha esse caminho: a
janela e sempre lida de `results/`, entao o texto nunca descreve uma
configuracao que nenhum experimento usou.

Nenhum valor daqui e transcrito: tudo e derivado de `data/processed/` e
`data/splits/` (conferidos contra `MANIFEST.json`), e a janela vem do campo
`config.max_gap` dos proprios JSONs de `results/`, nao de um default local.

O QUE GERA
----------
Fragmentos `.tex` em `tcc/src/tabelas/` (flutuantes ABNT completos, com
\\caption acima e Fonte abaixo -- e so dar \\input no capitulo):

  distribuicao_relacoes.tex  -> Tabela  (\\label{tab:distribuicao})
  distancia_por_tipo.tex     -> Tabela  (\\label{tab:distancia})
  particao_relacoes.tex      -> Tabela  (\\label{tab:particao})
  teto_recall.tex            -> Tabela  (\\label{tab:teto_recall})
  candidatos_por_particao.tex-> Tabela  (\\label{tab:candidatos})

Figuras PNG em `tcc/src/imagens/eda/`:

  01_reltype_distribution.png    -> \\label{fig:eda_reltype}
  02_distance_all_clip200.png    -> \\label{fig:eda_dist}
  03_relations_per_doc.png       -> \\label{fig:eda_relsdoc}
  04_entity_categories_atomic.png-> \\label{fig:eda_entcat}
  05_relations_by_split.png      -> \\label{fig:split_dist}

E um resumo em `results/tcc_eda.json` com as estatisticas citadas em PROSA
no capitulo (contagens de documento/entidade/relacao, medias, medianas,
percentis). Assim as frases do texto tambem tem um arquivo de origem
verificavel, e nao so as tabelas.

CONFERENCIA CRUZADA
-------------------
`--check-against-results` recomputa os pares candidatos de cada particao com
a janela lida de `results/` e compara com o campo `n_candidates` gravado
pelos experimentos. Se divergir, os splits em disco nao sao os que foram
treinados e o script para. Esta e a verificacao que amarra Metodologia e
Resultados.

USO
---
    python scripts/make_tcc_eda.py
    python scripts/make_tcc_eda.py --check-against-results
    python scripts/make_tcc_eda.py --check          # nao escreve; so imprime
    python scripts/make_tcc_eda.py --out-dir /tmp/conferencia

Opcoes: --data-dir, --results-dir, --out-dir, --figs-dir, --max-gap,
--check, --check-against-results, --dpi.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display: roda em CI, servidor e Colab headless.

import matplotlib.pyplot as plt
import numpy as np

from _artifacts import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    RELATION_TYPES,
    REPO_ROOT,
    SPLIT_LABELS,
    SPLIT_ORDER,
    TCC_EDA_FIGS_DIR,
    TCC_TABLES_DIR,
    MissingResultError,
    abnt_float,
    display_path,
    entity_gap,
    load_corpus,
    load_json,
    load_pipeline_config,
    load_splits,
    ptbr,
    ptbr_int,
    tt,
    write_tex,
)

SCRIPT = "make_tcc_eda.py"

# Percentis reportados na tabela de distancia. p95 e o que justifica a
# escolha da janela `max_gap` no texto, entao nao pode sair da tabela.
PERCENTILES = [25, 50, 75, 90, 95]

# Quantas categorias de entidade entram na figura 04. O corpus tem centenas
# de tipos atomicos; a cauda longa nao cabe nem informa.
TOP_ENTITY_CATEGORIES = 15

# Cinzas: escuro para a classe minoritaria (o foco do trabalho), claro para
# a majoritaria. Mesmo criterio de make_figures.py -- legivel em P&B.
GRAY_DARK = "#3d3d3d"
GRAY_LIGHT = "#bfbfbf"
SPLIT_COLORS = {"train": "#3d3d3d", "dev": "#8c8c8c", "test": "#d0d0d0"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 12,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


# --------------------------------------------------------------------------- #
# Estatisticas                                                                 #
# --------------------------------------------------------------------------- #
def atomic_entity_types(entity: dict) -> list[str]:
    """Tipos ATOMICOS de uma entidade.

    O SemClinBr anota tipos semanticos da UMLS e frequentemente empilha
    varios num unico atributo, separados por `|` (ex.:
    "Medical Device|Finding|Abbreviation"). Contar a string inteira
    produziria centenas de categorias mal povoadas; a figura 04 conta os
    tipos atomicos, entao uma entidade multi-tipo entra em cada um deles.
    """
    return [part.strip() for part in entity["type"].split("|") if part.strip()]


def corpus_stats(docs: list[dict]) -> dict:
    """Numeros citados em prosa na secao de caracterizacao do corpus."""
    per_doc = [len(doc["relations"]) for doc in docs]
    n_entities = sum(len(doc["entities"]) for doc in docs)
    by_type = collections.Counter(
        rel["type"] for doc in docs for rel in doc["relations"]
    )
    unknown = set(by_type) - set(RELATION_TYPES)
    if unknown:
        raise ValueError(
            f"tipos de relacao inesperados no corpus: {sorted(unknown)}. "
            f"Esperado apenas {RELATION_TYPES}. Atualize RELATION_TYPES em "
            f"scripts/_artifacts.py se o corpus mudou de verdade."
        )

    n_relations = sum(by_type.values())
    return {
        "n_docs": len(docs),
        "n_entities": n_entities,
        "entities_per_doc_mean": n_entities / len(docs),
        "n_relations": n_relations,
        "relations_per_doc_mean": n_relations / len(docs),
        "relations_per_doc_median": statistics.median(per_doc),
        "relations_per_doc_p90": float(np.percentile(per_doc, 90)),
        "relations_per_doc_max": max(per_doc),
        "docs_without_relation": sum(1 for count in per_doc if count == 0),
        "docs_without_relation_pct": 100 * sum(1 for c in per_doc if c == 0) / len(docs),
        "by_type": {name: by_type[name] for name in RELATION_TYPES},
        "by_type_pct": {
            name: 100 * by_type[name] / n_relations for name in RELATION_TYPES
        },
        "relations_per_doc": per_doc,
    }


def distance_stats(docs: list[dict]) -> dict[str, dict]:
    """Distancia em caracteres entre as entidades de cada relacao gold.

    E a evidencia empirica que justifica a janela `max_gap`: se o p95 de
    ambos os tipos cabe na janela, a perda de recall e marginal.
    """
    per_type: dict[str, list[int]] = {name: [] for name in RELATION_TYPES}
    for doc in docs:
        index = {entity["id"]: entity for entity in doc["entities"]}
        for rel in doc["relations"]:
            e1, e2 = index.get(rel["e1_id"]), index.get(rel["e2_id"])
            if e1 is None or e2 is None:
                # Relacao com referencia quebrada: o parser ja registra a
                # inconsistencia em log. Contar aqui inventaria distancia.
                continue
            per_type[rel["type"]].append(entity_gap(e1, e2))

    stats = {}
    for name, values in per_type.items():
        array = np.array(values)
        stats[name] = {
            "n": int(array.size),
            "mean": float(array.mean()),
            **{
                f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES
            },
            "values": values,
        }
    return stats


def split_stats(splits: dict[str, list[dict]]) -> dict[str, dict]:
    stats = {}
    for name in SPLIT_ORDER:
        docs = splits[name]
        by_type = collections.Counter(
            rel["type"] for doc in docs for rel in doc["relations"]
        )
        total = sum(by_type.values())
        stats[name] = {
            "n_docs": len(docs),
            **{key: by_type[key] for key in RELATION_TYPES},
            "total": total,
            "negation_pct": 100 * by_type["negation_of"] / total if total else 0.0,
        }
    return stats


def recall_ceiling(splits: dict[str, list[dict]], max_gap: int) -> dict[str, dict]:
    """Teto de recall imposto pela janela, por particao.

    Uma relacao gold cujo par ordenado fique alem de `max_gap` nunca e
    emitida como candidato e, portanto, e irrecuperavel por QUALQUER
    classificador. Separa-se dessa perda as DUPLICATAS de anotacao (mesmo
    par ordenado e mesmo tipo, anotado duas vezes no XML): o gerador as
    representa uma vez so, mas a relacao continua recuperavel -- logo nao
    reduzem o teto.
    """
    result = {}
    for name in SPLIT_ORDER:
        total = lost = duplicates = 0
        lost_by_type: collections.Counter = collections.Counter()
        for doc in splits[name]:
            index = {entity["id"]: entity for entity in doc["entities"]}
            seen: set[tuple[str, str, str]] = set()
            for rel in doc["relations"]:
                total += 1
                key = (rel["e1_id"], rel["e2_id"], rel["type"])
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                e1, e2 = index.get(rel["e1_id"]), index.get(rel["e2_id"])
                if e1 is None or e2 is None or entity_gap(e1, e2) > max_gap:
                    lost += 1
                    lost_by_type[rel["type"]] += 1
        result[name] = {
            "total": total,
            "lost": lost,
            "lost_pct": 100 * lost / total if total else 0.0,
            "duplicates": duplicates,
            "ceiling_pct": 100 * (total - lost) / total if total else 0.0,
            # Emite TODOS os tipos, inclusive os com zero perdas. Um tipo
            # ausente do Counter significa "nenhuma relacao perdida", fato
            # que o texto do TCC cita ("nenhuma das 152 relacoes do teste e
            # descartada"); grava-lo como 0 explicito permite conferi-lo em
            # `check_tcc_numbers.py` em vez de o caminho simplesmente sumir.
            "lost_by_type": {
                kind: lost_by_type.get(kind, 0) for kind in RELATION_TYPES
            },
        }
    return result


def candidate_stats(splits: dict[str, list[dict]], max_gap: int) -> dict[str, dict]:
    """Pares candidatos por particao, com a distribuicao de rotulos.

    Reimplementa `src/candidates.iter_candidate_pairs` sob as mesmas cinco
    propriedades (pares ordenados, sem autopares, sem spans identicos,
    janela respeitada, conferencia com o gold). `--check-against-results`
    confronta o total com o `n_candidates` gravado pelos experimentos.
    """
    result = {}
    for name in SPLIT_ORDER:
        counts: collections.Counter = collections.Counter()
        for doc in splits[name]:
            gold = {(r["e1_id"], r["e2_id"]): r["type"] for r in doc["relations"]}
            ents = sorted(
                doc["entities"], key=lambda e: (e["start"], e["end"], e["id"])
            )
            for i, e1 in enumerate(ents):
                for j, e2 in enumerate(ents):
                    if i == j:
                        continue
                    if (e1["start"], e1["end"]) == (e2["start"], e2["end"]):
                        continue
                    if entity_gap(e1, e2) > max_gap:
                        continue
                    counts[gold.get((e1["id"], e2["id"]), "no_relation")] += 1
        total = sum(counts.values())
        result[name] = {
            "total": total,
            **{key: counts[key] for key in (*RELATION_TYPES, "no_relation")},
            "no_relation_pct": 100 * counts["no_relation"] / total if total else 0.0,
        }
    return result


# --------------------------------------------------------------------------- #
# Tabelas                                                                      #
# --------------------------------------------------------------------------- #
def table_relation_distribution(stats: dict) -> str:
    rows = [
        rf"  {tt(name)} & {ptbr_int(stats['by_type'][name])} & "
        rf"{ptbr(stats['by_type_pct'][name], 2)} \\"
        for name in RELATION_TYPES
    ]
    body = "\n".join(
        [
            r"  \begin{tabular}{lrr}",
            r"  \toprule",
            r"  Tipo de relação & N & \% \\",
            r"  \midrule",
            *rows,
            r"  \midrule",
            rf"  \textbf{{Total}} & \textbf{{{ptbr_int(stats['n_relations'])}}} & "
            rf"\textbf{{100,00}} \\",
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    return abnt_float(
        body,
        caption="Distribuição dos tipos de relação no corpus parseado.",
        label="tab:distribuicao",
        source=r"elaborado pelo autor a partir do SemClinBr (\texttt{data/processed/dataset.jsonl}).",
    )


def table_distance(distances: dict[str, dict]) -> str:
    rows = []
    for name in RELATION_TYPES:
        entry = distances[name]
        cells = [ptbr(entry["mean"], 1)] + [
            ptbr(entry[f"p{p}"], 0) for p in PERCENTILES
        ]
        rows.append(
            rf"  {tt(name)} & {ptbr_int(entry['n'])} & " + " & ".join(cells) + r" \\"
        )
    header = "  Tipo & N & média & " + " & ".join(f"p{p}" for p in PERCENTILES) + r" \\"
    body = "\n".join(
        [
            r"  \begin{tabular}{lr" + "r" * (1 + len(PERCENTILES)) + "}",
            r"  \toprule",
            header,
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    return abnt_float(
        body,
        caption=(
            "Distância em caracteres entre as entidades de cada relação, "
            "por tipo."
        ),
        label="tab:distancia",
        source=r"elaborado pelo autor a partir do SemClinBr (\texttt{data/processed/dataset.jsonl}).",
        resize=True,
    )


def table_partition(splits: dict[str, dict], corpus: dict) -> str:
    rows = []
    for name in SPLIT_ORDER:
        entry = splits[name]
        rows.append(
            rf"  {SPLIT_LABELS[name]} & {ptbr_int(entry['n_docs'])} & "
            rf"{ptbr_int(entry['associated_with'])} & "
            rf"{ptbr_int(entry['negation_of'])} & "
            rf"{ptbr_int(entry['total'])} & {ptbr(entry['negation_pct'], 2)} \\"
        )
    totals = {
        key: sum(splits[name][key] for name in SPLIT_ORDER)
        for key in ("n_docs", "associated_with", "negation_of", "total")
    }
    body = "\n".join(
        [
            r"  \begin{tabular}{lrrrrr}",
            r"  \toprule",
            r"  Partição & Documentos & " + tt("associated_with") + " & "
            + tt("negation_of") + r" & Total relações & \% " + tt("negation_of")
            + r" \\",
            r"  \midrule",
            *rows,
            r"  \midrule",
            rf"  \textbf{{Total}} & \textbf{{{ptbr_int(totals['n_docs'])}}} & "
            rf"\textbf{{{ptbr_int(totals['associated_with'])}}} & "
            rf"\textbf{{{ptbr_int(totals['negation_of'])}}} & "
            rf"\textbf{{{ptbr_int(totals['total'])}}} & "
            rf"\textbf{{{ptbr(corpus['by_type_pct']['negation_of'], 2)}}} \\",
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    return abnt_float(
        body,
        caption=(
            "Distribuição real das relações por partição "
            r"(\emph{splits} congelados, semente 42)."
        ),
        label="tab:particao",
        source=r"elaborado pelo autor (\texttt{data/splits/MANIFEST.json}).",
        resize=True,
    )


def table_recall_ceiling(
    ceiling: dict[str, dict], splits: dict[str, dict], max_gap: int
) -> str:
    """Teto de recall com a perda DECOMPOSTA POR TIPO.

    A coluna agregada sozinha nao sustenta a afirmacao do texto: com
    `max_gap=25` a perda global e da ordem de 8 a 9%, mas ela e quase toda de
    `associated_with` -- em `negation_of`, a classe-alvo, e nula no teste e
    marginal nas demais particoes. Sem as colunas por tipo o leitor nao tem
    como verificar isso.
    """
    rows = []
    for name in SPLIT_ORDER:
        entry = ceiling[name]
        by_type = entry["lost_by_type"]
        cells = []
        for kind in RELATION_TYPES:
            lost_kind = by_type.get(kind, 0)
            gold_kind = splits[name][kind]
            cells.append(
                rf"{ptbr_int(lost_kind)} ({ptbr(100 * lost_kind / gold_kind, 2)}\%)"
                if gold_kind
                else "---"
            )
        rows.append(
            rf"  {SPLIT_LABELS[name]} & {ptbr_int(entry['total'])} & "
            + " & ".join(cells)
            + rf" & {ptbr_int(entry['duplicates'])} & "
            rf"{ptbr(entry['ceiling_pct'], 2)} \\"
        )
    total = sum(ceiling[name]["total"] for name in SPLIT_ORDER)
    lost = sum(ceiling[name]["lost"] for name in SPLIT_ORDER)
    duplicates = sum(ceiling[name]["duplicates"] for name in SPLIT_ORDER)
    total_cells = []
    for kind in RELATION_TYPES:
        lost_kind = sum(
            ceiling[name]["lost_by_type"].get(kind, 0) for name in SPLIT_ORDER
        )
        gold_kind = sum(splits[name][kind] for name in SPLIT_ORDER)
        total_cells.append(
            rf"\textbf{{{ptbr_int(lost_kind)}}} "
            rf"(\textbf{{{ptbr(100 * lost_kind / gold_kind, 2)}\%}})"
        )
    body = "\n".join(
        [
            r"  \begin{tabular}{lrrrrr}",
            r"  \toprule",
            rf"  Partição & Total ouro & Perdidas {tt('negation_of')} & "
            rf"Perdidas {tt('associated_with')} & Duplicatas colapsadas & "
            r"Teto recall (\%) \\",
            r"  \midrule",
            *rows,
            r"  \midrule",
            rf"  \textbf{{Total}} & \textbf{{{ptbr_int(total)}}} & "
            + " & ".join(total_cells)
            + rf" & \textbf{{{ptbr_int(duplicates)}}} & "
            rf"\textbf{{{ptbr(100 * (total - lost) / total, 2)}}} \\",
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    return abnt_float(
        body,
        caption=(
            rf"Teto de \emph{{recall}} imposto pela janela "
            rf"\texttt{{max\_gap}}$={max_gap}$ por partição."
        ),
        label="tab:teto_recall",
        source=r"elaborado pelo autor (\texttt{scripts/make\_tcc\_eda.py}).",
        resize=True,
    )


def quadro_sha256(data_dir: Path) -> str:
    """Quadro de hashes das particoes congeladas (Apendice A).

    Os hashes impressos no TCC divergiam dos de `MANIFEST.json` -- vinham de
    um `data/splits/SHA256SUMS.txt` que nao existe no repositorio. Gera-los
    aqui e a unica forma de o apendice de reprodutibilidade continuar
    verdadeiro depois de qualquer regeracao dos splits.

    E `quadro`, e nao `table`: hashes sao conteudo qualitativo/estruturado,
    nao dado numerico (convencao ABNT registrada no OUTLINE.md).
    """
    manifest = load_json(data_dir / "splits" / "MANIFEST.json")
    rows = [
        rf"  \texttt{{{name}.jsonl}} & "
        rf"\texttt{{{manifest['splits'][name]['sha256']}}} \\"
        for name in SPLIT_ORDER
    ]
    body = "\n".join(
        [
            r"  {\scriptsize",
            r"  \begin{tabular}{ll}",
            r"  \toprule",
            r"  Arquivo & SHA-256 \\",
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"  \end{tabular}}",
        ]
    )
    return abnt_float(
        body,
        caption="SHA-256 das partições congeladas do SemClinBr.",
        label="quad:sha256",
        source=(
            r"elaborado pelo autor (\texttt{data/splits/MANIFEST.json}, "
            r"gerado por \texttt{src/make\_splits.py})."
        ),
        env="quadro",
    )


def table_candidates(candidates: dict[str, dict], max_gap: int) -> str:
    rows = []
    for name in SPLIT_ORDER:
        entry = candidates[name]
        rows.append(
            rf"  {SPLIT_LABELS[name]} & {ptbr_int(entry['total'])} & "
            rf"{ptbr_int(entry['negation_of'])} & "
            rf"{ptbr_int(entry['associated_with'])} & "
            rf"{ptbr_int(entry['no_relation'])} & "
            rf"{ptbr(entry['no_relation_pct'], 2)} \\"
        )
    body = "\n".join(
        [
            r"  \begin{tabular}{lrrrrr}",
            r"  \toprule",
            r"  Partição & Candidatos & " + tt("negation_of") + " & "
            + tt("associated_with") + " & " + tt("no_relation")
            + r" & \% " + tt("no_relation") + r" \\",
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    return abnt_float(
        body,
        caption=(
            rf"Pares candidatos gerados por partição "
            rf"(\texttt{{max\_gap}}$={max_gap}$) e distribuição dos rótulos."
        ),
        label="tab:candidatos",
        source=(
            r"elaborado pelo autor (\texttt{src/candidates.py}); confere com o "
            r"campo \texttt{n\_candidates} de \texttt{results/baseline\_*.json}."
        ),
        resize=True,
    )


# --------------------------------------------------------------------------- #
# Figuras                                                                      #
# --------------------------------------------------------------------------- #
def figure_reltype(stats: dict, path: Path, dpi: int) -> None:
    counts = [stats["by_type"][name] for name in RELATION_TYPES]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(
        RELATION_TYPES,
        counts,
        color=[GRAY_DARK, GRAY_LIGHT],
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
    )
    for bar, name in zip(bars, RELATION_TYPES):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{counts[RELATION_TYPES.index(name)]:,}".replace(",", ".")
            + f"\n({stats['by_type_pct'][name]:.2f}%)".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("Relações anotadas")
    ax.set_ylim(0, max(counts) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_distance(
    distances: dict[str, dict], path: Path, dpi: int, clip: int, max_gap: int
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = np.linspace(0, clip, 41)
    for name, color in zip(RELATION_TYPES, [GRAY_DARK, GRAY_LIGHT]):
        values = np.clip(np.array(distances[name]["values"]), 0, clip)
        ax.hist(
            values,
            bins=bins,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            label=f"{name} (n={distances[name]['n']:,})".replace(",", "."),
        )
    # A janela e a decisao de projeto que esta figura justifica: tudo a
    # direita da linha e irrecuperavel por qualquer classificador.
    ax.axvline(max_gap, color="black", linestyle="--", linewidth=1.4)
    ax.annotate(
        rf"max_gap = {max_gap}",
        xy=(max_gap, ax.get_ylim()[1]),
        xytext=(6, -4),
        textcoords="offset points",
        va="top",
        fontsize=11,
    )
    ax.set_xlabel(f"Distância em caracteres (truncada em {clip})")
    ax.set_ylabel("Relações")
    ax.set_yscale("log")
    ax.legend(framealpha=1.0, edgecolor="0.7", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_relations_per_doc(stats: dict, path: Path, dpi: int) -> None:
    values = stats["relations_per_doc"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(
        values,
        bins=range(0, max(values) + 2),
        color=GRAY_LIGHT,
        edgecolor="black",
        linewidth=0.5,
    )
    median = stats["relations_per_doc_median"]
    ax.axvline(median, color="black", linestyle="--", linewidth=1.2)
    ax.text(
        median,
        ax.get_ylim()[1] * 0.92,
        f" mediana = {median:.0f}",
        fontsize=11,
        va="top",
    )
    ax.set_xlabel("Relações por documento")
    ax.set_ylabel("Documentos")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_entity_categories(docs: list[dict], path: Path, dpi: int) -> list[tuple]:
    counter: collections.Counter = collections.Counter()
    for doc in docs:
        for entity in doc["entities"]:
            counter.update(atomic_entity_types(entity))
    top = counter.most_common(TOP_ENTITY_CATEGORIES)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    names = [name for name, _ in top][::-1]
    counts = [count for _, count in top][::-1]
    bars = ax.barh(
        names, counts, color=GRAY_LIGHT, edgecolor="black", linewidth=0.7
    )
    ax.bar_label(
        bars,
        labels=[f"{c:,}".replace(",", ".") for c in counts],
        padding=3,
        fontsize=10,
    )
    ax.set_xlabel("Menções de entidade (tipos atômicos)")
    ax.set_xlim(0, max(counts) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return top


def figure_split_distribution(splits: dict[str, dict], path: Path, dpi: int) -> None:
    positions = np.arange(len(RELATION_TYPES))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, name in enumerate(SPLIT_ORDER):
        entry = splits[name]
        values = [entry[key] for key in RELATION_TYPES]
        bars = ax.bar(
            positions + (offset - 1) * width,
            values,
            width,
            # A virgula decimal e aplicada so ao numero: um `.replace` na
            # string inteira transformaria o "neg." do rotulo em "neg,".
            label=(
                f"{SPLIT_LABELS[name]} "
                f"({entry['negation_pct']:.1f}".replace(".", ",") + r"% neg.)"
            ),
            color=SPLIT_COLORS[name],
            edgecolor="black",
            linewidth=0.7,
        )
        ax.bar_label(
            bars,
            labels=[f"{v:,}".replace(",", ".") for v in values],
            padding=2,
            fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Relações (escala log)")
    ax.set_xticks(positions)
    ax.set_xticklabels(RELATION_TYPES)
    ax.legend(framealpha=1.0, edgecolor="0.7", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Conferencia cruzada com results/                                             #
# --------------------------------------------------------------------------- #
def check_against_results(
    candidates: dict[str, dict], results_dir: Path, seeds: list[int] | None = None
) -> None:
    """Confronta os candidatos recomputados com o que os experimentos viram.

    `n_candidates` e gravado por `src/relation_extraction.py` no momento do
    treino. Se os splits em disco tivessem mudado depois, este confronto
    falha -- e e a unica forma de detectar isso sem retreinar.
    """
    from _artifacts import BASELINE_ORDER, SEEDS, load_baseline

    expected = {name: candidates[name]["total"] for name in SPLIT_ORDER}
    for slug in BASELINE_ORDER:
        for seed in seeds or SEEDS:
            data = load_baseline(results_dir, slug, seed)
            observed = data["n_candidates"]
            for name in SPLIT_ORDER:
                if observed[name] != expected[name]:
                    raise ValueError(
                        f"{slug} seed{seed}: n_candidates[{name}]="
                        f"{observed[name]}, mas os splits em disco geram "
                        f"{expected[name]} com max_gap="
                        f"{data['config']['max_gap']}. Os splits nao sao os "
                        f"que foram treinados."
                    )
    print("conferencia OK: candidatos recomputados == n_candidates de results/")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera a caracterizacao do corpus do TCC a partir de data/.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TCC_TABLES_DIR,
        help="diretório das tabelas (padrão: tcc/src/tabelas).",
    )
    parser.add_argument(
        "--figs-dir",
        type=Path,
        default=TCC_EDA_FIGS_DIR,
        help="diretório das figuras (padrão: tcc/src/imagens/eda).",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=None,
        help=(
            "janela de geração de pares. Padrão: lida do campo config.max_gap "
            "dos resultados, para não divergir do que foi treinado."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="imprime as tabelas na saída padrão sem gravar arquivos.",
    )
    parser.add_argument(
        "--check-against-results",
        action="store_true",
        help="confere os candidatos recomputados contra results/*.json.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        config = load_pipeline_config(args.results_dir)
        max_gap = args.max_gap if args.max_gap is not None else config["max_gap"]
        docs = load_corpus(args.data_dir)
        splits = load_splits(args.data_dir)
    except (MissingResultError, ValueError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1

    corpus = corpus_stats(docs)
    distances = distance_stats(docs)
    per_split = split_stats(splits)
    ceiling = recall_ceiling(splits, max_gap)
    candidates = candidate_stats(splits, max_gap)

    if args.check_against_results:
        try:
            check_against_results(candidates, args.results_dir)
        except ValueError as error:
            print(f"ERRO: {error}", file=sys.stderr)
            return 1

    tables = {
        "distribuicao_relacoes.tex": (
            table_relation_distribution(corpus),
            "data/processed/dataset.jsonl",
        ),
        "distancia_por_tipo.tex": (
            table_distance(distances),
            "data/processed/dataset.jsonl",
        ),
        "particao_relacoes.tex": (
            table_partition(per_split, corpus),
            "data/splits/*.jsonl (conferidos contra MANIFEST.json)",
        ),
        "teto_recall.tex": (
            table_recall_ceiling(ceiling, per_split, max_gap),
            f"data/splits/*.jsonl, max_gap={max_gap} (de results/*.json)",
        ),
        "candidatos_por_particao.tex": (
            table_candidates(candidates, max_gap),
            f"data/splits/*.jsonl, max_gap={max_gap} (de results/*.json)",
        ),
        "sha256_splits.tex": (
            quadro_sha256(args.data_dir),
            "data/splits/MANIFEST.json",
        ),
    }

    if args.check:
        for name, (body, _) in tables.items():
            print(f"===== {name} =====")
            print(body)
        return 0

    for name, (body, source) in tables.items():
        path = args.out_dir / name
        write_tex(path, body, script=SCRIPT, sources=source)
        print(f"escrito: {display_path(path)}")

    apply_style()
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    figure_reltype(corpus, args.figs_dir / "01_reltype_distribution.png", args.dpi)
    figure_distance(
        distances,
        args.figs_dir / "02_distance_all_clip200.png",
        args.dpi,
        clip=200,
        max_gap=max_gap,
    )
    figure_relations_per_doc(
        corpus, args.figs_dir / "03_relations_per_doc.png", args.dpi
    )
    top_categories = figure_entity_categories(
        docs, args.figs_dir / "04_entity_categories_atomic.png", args.dpi
    )
    figure_split_distribution(
        per_split, args.figs_dir / "05_relations_by_split.png", args.dpi
    )
    for name in (
        "01_reltype_distribution.png",
        "02_distance_all_clip200.png",
        "03_relations_per_doc.png",
        "04_entity_categories_atomic.png",
        "05_relations_by_split.png",
    ):
        print(f"escrito: {display_path(args.figs_dir / name)}")

    # Resumo das estatisticas citadas em PROSA. Sem isso, frases como
    # "mediana de 10 relacoes por documento" nao teriam arquivo de origem.
    summary = {
        "max_gap": max_gap,
        "pipeline_config": config,
        "corpus": {k: v for k, v in corpus.items() if k != "relations_per_doc"},
        "distance_by_type": {
            name: {k: v for k, v in entry.items() if k != "values"}
            for name, entry in distances.items()
        },
        "splits": per_split,
        "recall_ceiling": ceiling,
        "candidates": candidates,
        "top_entity_categories": top_categories,
    }
    summary_path = args.results_dir / "tcc_eda.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"escrito: {display_path(summary_path)}")

    print(
        f"\n{len(tables)} tabela(s) e 5 figura(s) de EDA "
        f"(max_gap={max_gap}, {corpus['n_relations']} relações)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
