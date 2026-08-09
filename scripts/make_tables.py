"""
Regenera as tabelas do artigo a partir de `results/*.json`.

POR QUE ESTE SCRIPT EXISTE
--------------------------
As tabelas de `artigo-sbc/artigo.tex` eram digitadas a mao a partir dos
JSONs de resultado. Isso significava que (a) nao havia como verificar se o
que estava publicado correspondia ao que foi medido, e (b) rodar uma semente
nova exigia reeditar LaTeX na mao. Este script fecha as duas lacunas: os
numeros do artigo passam a ser derivados, nao transcritos.

O QUE GERA
----------
Um fragmento `.tex` por tabela, em `artigo-sbc/tables/` -- os arquivos que
`artigo.tex` puxa por `\\input{}`:

  tab_resultados.tex   -> Tabela 1 (\\label{tab:resultados})
                          metricas dos dois baselines no teste
  tab_signif.tex       -> Tabela 2 (\\label{tab:signif})
                          comparacao pareada BioBERTpt vs. BERTimbau

Cada fragmento e um ambiente `table` completo (com \\caption e \\label). O
`artigo.tex` os inclui por `\\input{tables/tab_resultados.tex}` e
`\\input{tables/tab_signif.tex}`, entao regerar aqui ja atualiza o artigo. Os
fragmentos usam `booktabs` (\\toprule/\\midrule/\\bottomrule), que ja e
carregado pelo preambulo do artigo.

Este script nao edita `artigo.tex` -- so reescreve os fragmentos que ele
inclui. Os \\label continuam sendo `tab:resultados` e `tab:signif`, entao as
remissoes `\\ref{}` no texto seguem valendo.

USO
---
    python scripts/make_tables.py                  # semente 42 (a do artigo),
                                                   # sobrescreve artigo-sbc/tables/
    python scripts/make_tables.py --seed 43
    python scripts/make_tables.py --check          # nao escreve; so imprime
    python scripts/make_tables.py --out-dir /tmp/tables_conferencia

Opcoes: --results-dir, --out-dir, --seed, --check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _artifacts import (
    BASELINE_ORDER,
    CLASS_ORDER,
    DEFAULT_RESULTS_DIR,
    MODEL_LABELS,
    REPO_ROOT,
    MissingResultError,
    display_path,
    label_for,
    load_baselines,
    load_significance,
    ptbr,
)

DEFAULT_OUT_DIR = REPO_ROOT / "artigo-sbc" / "tables"

HEADER = (
    "%% GERADO AUTOMATICAMENTE por scripts/make_tables.py -- nao edite a mao.\n"
    "%% Fonte: {sources}\n"
)


def _bold(text: str, is_best: bool) -> str:
    return rf"\textbf{{{text}}}" if is_best else text


def build_results_table(baselines: dict[str, dict], seed: int) -> str:
    """Tabela 1: metricas dos dois baselines no conjunto de teste.

    Colunas: Macro-F1, F1 por classe (3), MCC e acuracia. O melhor valor de
    cada coluna vai em negrito -- criterio declarado na propria legenda.
    """
    # Uma coluna = (cabecalho, funcao que extrai o valor de um baseline).
    columns = [
        (r"\textbf{Macro-F1}", lambda d: d["test_macro_f1"]),
        (r"\textbf{F1 neg.}", lambda d: d["test_f1_per_class"]["negation_of"]),
        (r"\textbf{F1 assoc.}", lambda d: d["test_f1_per_class"]["associated_with"]),
        (r"\textbf{F1 no\_rel.}", lambda d: d["test_f1_per_class"]["no_relation"]),
        (r"\textbf{MCC}", lambda d: d["test_mcc"]),
        (r"\textbf{Acur.}", lambda d: d["sklearn_report"]["accuracy"]),
    ]

    values = {
        slug: [getter(baselines[slug]) for _, getter in columns]
        for slug in BASELINE_ORDER
    }
    best = [
        max(values[slug][i] for slug in BASELINE_ORDER) for i in range(len(columns))
    ]

    rows = []
    for slug in BASELINE_ORDER:
        label = label_for(baselines[slug])
        cells = [
            # Comparacao no valor arredondado: se dois baselines empatam em
            # 3 casas, os dois ficam em negrito. Negritar so um deles seria
            # afirmar uma diferenca que a tabela nao mostra.
            _bold(ptbr(value), round(value, 3) >= round(best[i], 3))
            for i, value in enumerate(values[slug])
        ]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")

    header = (
        r"\textbf{Baseline} & "
        + " &\n".join(name for name, _ in columns)
        + r" \\"
    )

    return "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{Resultados dos \emph{{baselines}} no conjunto de teste "
            rf"(semente {seed}).",
            r"Negrito indica o melhor valor por coluna.}",
            r"\label{tab:resultados}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_significance_table(significance: dict, seed: int) -> str:
    """Tabela 2: comparacao pareada (McNemar + bootstrap) no F1 de negation_of."""
    target = significance["target_class"].replace("_", r"\_")
    label_a = MODEL_LABELS[significance["model_a"]].split(" (")[0]
    label_b = MODEL_LABELS[significance["model_b"]].split(" (")[0]

    f1 = significance["target_f1"]
    mcnemar = significance["mcnemar"]
    boot = significance["paired_bootstrap"]

    diff = f1["a_minus_b"]
    # `-` fora de modo matematico vira hifen no PDF; por isso os negativos
    # entram entre cifroes.
    diff_tex = rf"${ptbr(diff)}$" if diff < 0 else ptbr(diff)
    ci_low, ci_high = boot["ci95_low"], boot["ci95_high"]
    ci_tex = (
        rf"$[{ptbr(ci_low)};\,{'+' if ci_high >= 0 else ''}{ptbr(ci_high)}]$"
    )

    # Significancia pelo criterio declarado no artigo: o IC95 do bootstrap
    # pareado nao pode conter o zero.
    significant = not (ci_low <= 0.0 <= ci_high)
    verdict = r"\textbf{Sim}" if significant else r"\textbf{Não}"

    n_test = f"{significance['n_test']:,}".replace(",", ".")

    rows = [
        rf"F1 \texttt{{{target}}} --- {label_a} (A) & {ptbr(f1['a'])} \\",
        rf"F1 \texttt{{{target}}} --- {label_b} (B) & {ptbr(f1['b'])} \\",
        rf"Diferença observada (A $-$ B) & {diff_tex} \\",
        rf"McNemar: $b$ (só A acerta) / $c$ (só B acerta) & "
        rf"{mcnemar['b_only_a_correct']} / {mcnemar['c_only_b_correct']} \\",
        rf"McNemar: valor-$p$ (binomial exato) & {ptbr(mcnemar['p_value'], 2)} \\",
        rf"\emph{{Bootstrap}} pareado: IC95\% da diferença & {ci_tex} \\",
        rf"\emph{{Bootstrap}} pareado: valor-$p$ & {ptbr(boot['p_value'], 2)} \\",
        rf"\textbf{{Diferença significativa ($\alpha=0,05$)?}} & {verdict} \\",
    ]

    return "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{Comparação pareada {label_a} (clínico) vs.\ {label_b} "
            rf"(geral) no teste",
            rf"($n={n_test}$, semente {seed}).}}",
            r"\label{tab:signif}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"\textbf{Medida} & \textbf{Valor} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera as tabelas do artigo a partir de results/*.json.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "diretório de saída (padrão: artigo-sbc/tables, os fragmentos que "
            "o artigo inclui). Aponte para outro lugar se quiser só conferir."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="semente cujos resultados alimentam as tabelas (padrão: 42).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="imprime as tabelas na saída padrão sem gravar arquivos.",
    )
    args = parser.parse_args(argv)

    try:
        baselines = load_baselines(args.results_dir, args.seed)
        significance = load_significance(args.results_dir, args.seed)
    except (MissingResultError, ValueError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1

    sources = ", ".join(
        f"results/baseline_{slug}_seed{args.seed}.json" for slug in BASELINE_ORDER
    )

    tables = {
        "tab_resultados.tex": (
            build_results_table(baselines, args.seed),
            sources,
        ),
        "tab_signif.tex": (
            build_significance_table(significance, args.seed),
            f"results/significance_biobertpt_vs_bertimbau (semente {args.seed})",
        ),
    }

    if args.check:
        for name, (body, _) in tables.items():
            print(f"===== {name} =====")
            print(body)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, (body, source) in tables.items():
        path = args.out_dir / name
        path.write_text(HEADER.format(sources=source) + body, encoding="utf-8")
        print(f"escrito: {display_path(path)}")

    print(
        f"\n{len(tables)} tabela(s) da semente {args.seed}. "
        f"Classes na ordem {CLASS_ORDER}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
