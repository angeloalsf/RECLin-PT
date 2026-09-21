"""
Regenera as tabelas e figuras de RESULTADOS do TCC a partir de
`results/*.json`.

POR QUE ESTE SCRIPT EXISTE
--------------------------
`make_tables.py` e `make_figures.py` ja derivam os artefatos do artigo SBC
dos mesmos JSONs. O TCC precisa do mesmo conteudo sob outras convencoes:

  - legenda ACIMA e "Fonte:" ABAIXO (ABNT), contra legenda abaixo no SBC;
  - `\\input` de flutuante completo em `tcc/src/tabelas/`, com os `\\label`
    que os capitulos ja referenciam;
  - figuras em PNG (o TCC compila com `graphicx` sobre PNG em
    `imagens/`), contra PDF vetorial no artigo;
  - uma tabela a mais, de robustez a semente, que no artigo e so prosa.

O desenho de figura NAO e duplicado: `figure_f1_per_class` e
`figure_confusion_matrix` sao importados de `make_figures.py`, de modo que
a figura do TCC e literalmente a mesma do artigo, so que rasterizada. Se o
desenho mudar la, muda aqui.

O QUE GERA
----------
Fragmentos `.tex` em `tcc/src/tabelas/`:

  resultados.tex        -> Tabela (\\label{tab:resultados})
                           metricas dos dois baselines, semente de referencia
  significancia.tex     -> Tabela (\\label{tab:significancia})
                           McNemar + bootstrap pareado, semente de referencia
  significancia_seed43.tex -> Tabela (\\label{tab:significancia43})
                           o mesmo protocolo na semente 43
  robustez_semente.tex  -> Tabela (\\label{tab:robustez})
                           metricas por semente e amplitude, por modelo
  fase2_resultados.tex  -> Tabela (\\label{tab:fase2_resultados})
                           os nove sistemas da fase 2 (4 baselines, 4 execucoes
                           do RECLin-PT e a regra pura) na classe-alvo e em
                           macro-F1
  fase2_significancia.tex -> Tabela (\\label{tab:fase2_significancia})
                           as 16 comparacoes pareadas entre execucao filtrada e
                           baseline oficial

As duas ultimas leem `results/<sistema>.preds.json`, gravados por
`scripts/run_fase2_test.py`, e recalculam as metricas dos vetores de predicao.
Nao ha valor intermediario: `results/FASE2_test_summary.json` e registro da
execucao, nao fonte de numero publicado. Num `--results-dir` anterior a fase 2
elas sao puladas com aviso.

Figuras PNG em `tcc/src/imagens/resultados/`:

  f1_por_classe.png  -> \\label{fig:f1classe}
  cm_biobertpt.png   -> \\label{fig:cm_biobertpt}
  cm_bertimbau.png   -> \\label{fig:cm_bertimbau}

CRITERIO DE SIGNIFICANCIA
-------------------------
O mesmo do artigo: a diferenca e significativa quando o IC95\\% do bootstrap
pareado NAO contem o zero. O veredito e derivado do intervalo, nunca
digitado -- e o que faz a tabela da semente 43 dizer "Sim" e a da 42 dizer
"Nao" sem intervencao.

USO
---
    python scripts/make_tcc_artifacts.py
    python scripts/make_tcc_artifacts.py --seed 43
    python scripts/make_tcc_artifacts.py --check
    python scripts/make_tcc_artifacts.py --out-dir /tmp/conferencia

Opcoes: --results-dir, --out-dir, --figs-dir, --seed, --seeds, --check, --dpi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _artifacts import (
    BASELINE_ORDER,
    CLASS_ORDER,
    DEFAULT_RESULTS_DIR,
    FASE2_BASELINE_KEYS,
    FASE2_FILTER_KEYS,
    FASE2_RULE_KEY,
    FASE2_SYSTEM_KEYS,
    FASE2_TARGET_CLASS,
    MODEL_LABELS,
    REFERENCE_SEED,
    SEEDS,
    SLUG_MODELS,
    TCC_RESULT_FIGS_DIR,
    TCC_TABLES_DIR,
    MissingResultError,
    abnt_float,
    display_path,
    label_for,
    load_baseline,
    load_baselines,
    load_fase2_significance,
    load_fase2_systems,
    load_json,
    load_significance,
    ptbr,
    ptbr_int,
    signed,
    slug_for,
    tt,
    write_tex,
)

SCRIPT = "make_tcc_artifacts.py"


def _bold(text: str, is_best: bool) -> str:
    return rf"\textbf{{{text}}}" if is_best else text


# --------------------------------------------------------------------------- #
# Tabelas                                                                      #
# --------------------------------------------------------------------------- #
def build_results_table(baselines: dict[str, dict], seed: int) -> str:
    """Metricas dos dois baselines no teste, semente `seed`.

    Mesmas colunas de `make_tables.build_results_table` -- deliberadamente,
    para que TCC e artigo nao possam reportar conjuntos de metricas
    diferentes do mesmo experimento.
    """
    columns = [
        ("Macro-F1", lambda d: d["test_macro_f1"]),
        (r"F1$_{\text{neg}}$", lambda d: d["test_f1_per_class"]["negation_of"]),
        (
            r"F1$_{\text{assoc}}$",
            lambda d: d["test_f1_per_class"]["associated_with"],
        ),
        (
            r"F1$_{\text{no\_rel}}$",
            lambda d: d["test_f1_per_class"]["no_relation"],
        ),
        ("MCC", lambda d: d["test_mcc"]),
        ("Acurácia", lambda d: d["sklearn_report"]["accuracy"]),
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
        cells = [
            # Empate em 3 casas negrita os dois: negritar so um afirmaria uma
            # diferenca que a tabela nao mostra.
            _bold(ptbr(value), round(value, 3) >= round(best[i], 3))
            for i, value in enumerate(values[slug])
        ]
        rows.append(
            f"  {label_for(baselines[slug])} & " + " & ".join(cells) + r" \\"
        )

    n_test = baselines[BASELINE_ORDER[0]]["n_candidates"]["test"]
    body = "\n".join(
        [
            r"  \begin{tabular}{lcccccc}",
            r"  \toprule",
            r"  Modelo & " + " & ".join(name for name, _ in columns) + r" \\",
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    return abnt_float(
        body,
        caption=(
            rf"Resultados dos \emph{{baselines}} no conjunto de teste "
            rf"($n={ptbr_int(n_test)}$, semente {seed}). "
            rf"Negrito indica o melhor valor por coluna."
        ),
        label="tab:resultados",
        source=(
            rf"elaborado pelo autor "
            rf"(\texttt{{results/baseline\_*\_seed{seed}.json}})."
        ),
        resize=True,
    )


def build_significance_table(significance: dict, seed: int, label: str) -> str:
    """McNemar + bootstrap pareado no F1 da classe-alvo."""
    target = significance["target_class"]
    label_a = MODEL_LABELS[significance["model_a"]].split(" (")[0]
    label_b = MODEL_LABELS[significance["model_b"]].split(" (")[0]

    f1 = significance["target_f1"]
    mcnemar = significance["mcnemar"]
    boot = significance["paired_bootstrap"]

    diff = f1["a_minus_b"]
    # `-` fora de modo matematico vira hifen no PDF; negativos entram entre
    # cifroes.
    diff_tex = rf"${ptbr(diff)}$" if diff < 0 else ptbr(diff)
    ci_low, ci_high = boot["ci95_low"], boot["ci95_high"]
    ci_tex = rf"$[{ptbr(ci_low)};\,{'+' if ci_high >= 0 else ''}{ptbr(ci_high)}]$"

    # Valores-p muito pequenos (a semente 43 da ~1e-34) viram "< 0,001":
    # imprimir 0,00 sugeriria empate exato.
    def pvalue(value: float, decimals: int = 2) -> str:
        return r"$<0{,}001$" if value < 0.001 else ptbr(value, decimals)

    significant = not (ci_low <= 0.0 <= ci_high)
    verdict = r"\textbf{Sim}" if significant else r"\textbf{Não}"

    rows = [
        rf"  F1 {tt(target)} --- {label_a} (A) & {ptbr(f1['a'])} \\",
        rf"  F1 {tt(target)} --- {label_b} (B) & {ptbr(f1['b'])} \\",
        rf"  Diferença observada (A $-$ B) & {diff_tex} \\",
        rf"  McNemar: $b$ (só A acerta) / $c$ (só B acerta) & "
        rf"{ptbr_int(mcnemar['b_only_a_correct'])} / "
        rf"{ptbr_int(mcnemar['c_only_b_correct'])} \\",
        rf"  McNemar: valor-$p$ (binomial exato) & {pvalue(mcnemar['p_value'])} \\",
        rf"  \emph{{Bootstrap}} pareado: IC95\% da diferença & {ci_tex} \\",
        rf"  \emph{{Bootstrap}} pareado: valor-$p$ & "
        rf"{pvalue(boot['p_value'], 4)} \\",
        rf"  \textbf{{Diferença significativa ($\alpha=0{{,}}05$)?}} & {verdict} \\",
    ]

    body = "\n".join(
        [
            r"  \begin{tabular}{lc}",
            r"  \toprule",
            r"  Medida & Valor \\",
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    return abnt_float(
        body,
        caption=(
            rf"Comparação pareada {label_a} (clínico) \emph{{vs.}} {label_b} "
            rf"(geral) no conjunto de teste "
            rf"($n={ptbr_int(significance['n_test'])}$, semente {seed})."
        ),
        label=label,
        source=(
            rf"elaborado pelo autor (\texttt{{results/significance\_biobertpt"
            rf"\_vs\_bertimbau\_seed{seed}.json}})."
        ),
        resize=True,
    )


# Metricas da tabela de robustez, na ordem em que o capitulo as discute.
ROBUSTNESS_METRICS = [
    ("macro_f1", "Macro-F1"),
    ("f1_negation_of", r"F1 " + tt("negation_of")),
    ("f1_associated_with", r"F1 " + tt("associated_with")),
    ("f1_no_relation", r"F1 " + tt("no_relation")),
]


def build_robustness_table(summary: dict, seeds: list[int]) -> str:
    """Metricas por semente e amplitude observada, por modelo.

    A amplitude (max - min entre as sementes) e a coluna que sustenta o
    argumento central do capitulo: a variacao de inicializacao de um mesmo
    modelo supera a diferenca entre os dois modelos.
    """
    rows = []
    for slug in BASELINE_ORDER:
        entry = summary["models"][slug]
        for index, (key, name) in enumerate(ROBUSTNESS_METRICS):
            metric = entry["metrics"][key]
            by_seed = [metric["by_seed"][str(seed)] for seed in seeds]
            amplitude = max(by_seed) - min(by_seed)
            model_cell = entry["label"] if index == 0 else ""
            rows.append(
                f"  {model_cell} & {name} & "
                + " & ".join(ptbr(value) for value in by_seed)
                + rf" & {ptbr(metric['mean'])} & {ptbr(amplitude)} \\"
            )
        if slug != BASELINE_ORDER[-1]:
            rows.append(r"  \midrule")

    header = (
        r"  Modelo & Métrica & "
        + " & ".join(f"Semente {seed}" for seed in seeds)
        + r" & Média & Amplitude \\"
    )
    body = "\n".join(
        [
            r"  \begin{tabular}{ll" + "c" * (len(seeds) + 2) + "}",
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
            "Robustez à semente de treino: métricas de teste por semente, "
            "média e amplitude observada."
        ),
        label="tab:robustez",
        source=(
            r"elaborado pelo autor (\texttt{scripts/aggregate\_seeds.py} "
            r"$\rightarrow$ \texttt{results/summary\_by\_seed.json})."
        ),
        resize=True,
    )


# --------------------------------------------------------------------------- #
# Tabelas da fase 2 (filtro de pista lexical e regra pura)                     #
# --------------------------------------------------------------------------- #
# As duas tabelas desta secao descrevem sistemas que nao tem `baseline_*.json`:
# eles nascem das predicoes salvas, em `results/<chave>.preds.json`. Por isso
# nao passam por `load_baselines` e tem funcoes proprias, no mesmo padrao das
# demais: recebem dados ja carregados, devolvem o flutuante ABNT pronto e nao
# digitam nenhum numero.

# Por extenso na legenda, como o texto do capitulo os escreve. O dicionario
# existe so para as contagens que aparecem em legenda; o resto e derivado.
NUMBER_WORDS = {16: "dezesseis"}


def fase2_label(
    key: str, calib: dict | None = None, *, short: bool = False
) -> str:
    """Rotulo publicado de um sistema da fase 2, a partir da chave do arquivo.

    `short=True` devolve a forma usada na tabela de significancia, que tem duas
    colunas de sistema e precisa caber na largura do texto. `calib` so e exigido
    pela regra pura, cujo rotulo cita a variante e o limiar escolhidos no DEV.
    """
    if key == FASE2_RULE_KEY:
        if calib is None:
            raise ValueError(
                "o rotulo da regra pura depende de results/CALIBRACAO_filtro.json"
            )
        return (
            rf"Regra pura {calib['rule']} "
            rf"(\texttt{{gap}}$\leq {calib['rule_gap']}$)"
        )

    kind, slug, seed_part = key.split("_")
    seed = int(seed_part.removeprefix("seed"))
    full = MODEL_LABELS[SLUG_MODELS[slug]]
    # "BioBERTpt (clinico)" -> "BioBERTpt": a qualificacao do encoder ja foi
    # dada no Capitulo 6 e repeti-la em nove linhas so ocuparia largura.
    bare = full.split(" (")[0]
    semente = rf"s.~{seed}" if short else f"semente {seed}"
    if kind == "filtro":
        return f"RECLin-PT sobre {bare}, {semente}"
    return f"{full if not short else bare}, {semente}"


def build_fase2_results_table(systems: dict[str, dict], calib: dict) -> str:
    """Os nove sistemas da Secao 6.7 sob as mesmas quatro metricas.

    Precisao, recall e F1 sao da classe-alvo; o macro-F1 e das tres classes. Os
    quatro valores saem de `preds_metrics`, ou seja, dos vetores de predicao, e
    nao de um resumo intermediario.
    """
    columns = [
        (r"P$_{\text{neg}}$", "precision"),
        (r"R$_{\text{neg}}$", "recall"),
        (r"F1$_{\text{neg}}$", "f1"),
        ("Macro-F1", "macro_f1"),
    ]
    best = {
        key: max(systems[name][key] for name in FASE2_SYSTEM_KEYS)
        for _, key in columns
    }

    rows = []
    for group in (FASE2_BASELINE_KEYS, FASE2_FILTER_KEYS, [FASE2_RULE_KEY]):
        if rows:
            rows.append(r"  \midrule")
        for name in group:
            cells = [
                _bold(
                    ptbr(systems[name][key], 4),
                    round(systems[name][key], 4) >= round(best[key], 4),
                )
                for _, key in columns
            ]
            rows.append(
                f"  {fase2_label(name, calib)} & " + " & ".join(cells) + r" \\"
            )

    body = "\n".join(
        [
            r"  \begin{tabular}{lcccc}",
            r"  \toprule",
            r"  Sistema & " + " & ".join(name for name, _ in columns) + r" \\",
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    n_test = systems[FASE2_SYSTEM_KEYS[0]]["n"]
    return abnt_float(
        body,
        caption=(
            rf"Desempenho na classe {tt(FASE2_TARGET_CLASS)} e macro-F1 no "
            rf"conjunto de teste ($n={ptbr_int(n_test)}$) dos quatro "
            rf"\emph{{baselines}}, das quatro execuções do RECLin-PT e da regra "
            rf"pura {calib['rule']}. Negrito indica o melhor valor por coluna."
        ),
        label="tab:fase2_resultados",
        source=(
            r"elaborado pelo autor (\texttt{scripts/run\_fase2\_test.py} "
            r"$\rightarrow$ \texttt{results/*.preds.json})."
        ),
        resize=True,
    )


def build_fase2_significance_table(
    systems: dict[str, dict], comparisons: dict[tuple[str, str], dict]
) -> str:
    """Cada execucao filtrada contra cada baseline oficial, no F1 da classe-alvo.

    O McNemar nao entra: entre um sistema e a sua propria versao filtrada a
    regra so altera predicoes numa direcao, e a tabela de discordancias fica
    degenerada por construcao. A Secao 6.7 declara essa omissao.
    """
    rows = []
    n_boot = None
    for a_key in FASE2_FILTER_KEYS:
        if rows:
            rows.append(r"  \midrule")
        for b_key in FASE2_BASELINE_KEYS:
            data = comparisons[(a_key, b_key)]
            boot = data["paired_bootstrap"]
            n_boot = boot["n_boot"]

            # Guarda-corpo: o F1 impresso e o do sidecar, e o teste pareado
            # precisa ter medido o MESMO vetor. Se divergirem, um dos dois foi
            # regerado sozinho e a tabela afirmaria uma diferenca entre numeros
            # que nao se referem a mesma execucao.
            for role, key in (("a", a_key), ("b", b_key)):
                gravado = data["target_f1"][role]
                recalculado = systems[key]["f1"]
                if abs(gravado - recalculado) > 1e-9:
                    raise ValueError(
                        f"significance_{a_key}_vs_{b_key}.json diz F1({role})="
                        f"{gravado:.6f}, mas {key}.preds.json da "
                        f"{recalculado:.6f}. Regere a fase 2 inteira "
                        f"(run_fase2_test.py e run_fase2_significance.py)."
                    )

            diff = data["target_f1"]["a_minus_b"]
            rows.append(
                f"  {fase2_label(a_key, short=True)} & "
                f"{fase2_label(b_key, short=True)} & "
                f"{ptbr(data['target_f1']['a'], 4)} & "
                f"{ptbr(data['target_f1']['b'], 4)} & "
                rf"${signed(diff)}$ "
                rf"$[{signed(boot['ci95_low'])};\,{signed(boot['ci95_high'])}]$"
                r" \\"
            )

    body = "\n".join(
        [
            r"  \begin{tabular}{llccc}",
            r"  \toprule",
            r"  Sistema A & \emph{Baseline} B & F1 A & F1 B & "
            r"Diferença (A $-$ B) com IC95\% \\",
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"  \end{tabular}",
        ]
    )
    total = len(comparisons)
    return abnt_float(
        body,
        caption=(
            rf"As {NUMBER_WORDS.get(total, ptbr_int(total))} comparações "
            rf"pareadas entre cada execução do RECLin-PT e cada "
            rf"\emph{{baseline}} oficial, no F1 de {tt(FASE2_TARGET_CLASS)} "
            rf"sobre o conjunto de teste (\emph{{bootstrap}} pareado, "
            rf"{ptbr_int(n_boot)} reamostragens, IC95\% por percentis "
            rf"2,5/97,5)."
        ),
        label="tab:fase2_significancia",
        source=(
            r"elaborado pelo autor "
            r"(\texttt{results/significance\_filtro\_*\_vs\_baseline\_*.json})."
        ),
        resize=True,
    )


def build_fase2_tables(results_dir: Path) -> dict[str, tuple[str, str]]:
    """As duas tabelas da fase 2, prontas para entrar no dicionario de saida.

    Levanta `MissingResultError` quando o diretorio de resultados nao tem a fase
    2 (e o caso de `results/archive_*`, anteriores a ela).
    """
    systems = load_fase2_systems(results_dir)
    calib = load_json(results_dir / "CALIBRACAO_filtro.json")
    comparisons = {
        (a_key, b_key): load_fase2_significance(results_dir, a_key, b_key)
        for a_key in FASE2_FILTER_KEYS
        for b_key in FASE2_BASELINE_KEYS
    }
    return {
        "fase2_resultados.tex": (
            build_fase2_results_table(systems, calib),
            "results/*.preds.json + results/CALIBRACAO_filtro.json",
        ),
        "fase2_significancia.tex": (
            build_fase2_significance_table(systems, comparisons),
            "results/significance_filtro_*_vs_baseline_*.json",
        ),
    }


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera tabelas e figuras de resultados do TCC.",
    )
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
        default=TCC_RESULT_FIGS_DIR,
        help="diretório das figuras (padrão: tcc/src/imagens/resultados).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=REFERENCE_SEED,
        help=(
            f"semente de referência, a das tabelas principais e das figuras "
            f"(padrão: {REFERENCE_SEED})."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="sementes que entram na tabela de robustez (padrão: 42 43).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="imprime as tabelas na saída padrão sem gravar arquivos.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    other_seeds = [seed for seed in args.seeds if seed != args.seed]

    try:
        baselines = load_baselines(args.results_dir, args.seed)
        significance = load_significance(args.results_dir, args.seed)
        summary = load_json(args.results_dir / "summary_by_seed.json")
        others = {
            seed: load_significance(args.results_dir, seed) for seed in other_seeds
        }
        # Garante que a tabela de robustez nao cite uma semente que nao foi
        # treinada (summary_by_seed.json e regenerado a parte).
        for seed in args.seeds:
            load_baseline(args.results_dir, BASELINE_ORDER[0], seed)
    except (MissingResultError, ValueError, KeyError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1

    tables = {
        "resultados.tex": (
            build_results_table(baselines, args.seed),
            f"results/baseline_*_seed{args.seed}.json",
        ),
        "significancia.tex": (
            build_significance_table(significance, args.seed, "tab:significancia"),
            f"results/significance_biobertpt_vs_bertimbau_seed{args.seed}.json",
        ),
        "robustez_semente.tex": (
            build_robustness_table(summary, args.seeds),
            "results/summary_by_seed.json",
        ),
    }
    for seed, data in others.items():
        tables[f"significancia_seed{seed}.tex"] = (
            build_significance_table(data, seed, f"tab:significancia{seed}"),
            f"results/significance_biobertpt_vs_bertimbau_seed{seed}.json",
        )

    # Fase 2. Estas duas saem de `results/*.preds.json`, e nao dos
    # `baseline_*.json` carregados acima, entao tem bloco proprio. Um diretorio
    # anterior a fase 2 (`results/archive_*`) nao tem esses arquivos: nesse caso
    # as tabelas sao puladas com aviso, para que a comparacao entre rodadas
    # antigas continue possivel.
    try:
        tables.update(build_fase2_tables(args.results_dir))
    except MissingResultError as error:
        print(
            f"AVISO: tabelas da fase 2 nao geradas -- "
            f"{str(error).splitlines()[0]}\n"
            f"       Rode `python scripts/run_fase2_test.py` para produzi-las.",
            file=sys.stderr,
        )
    except (ValueError, KeyError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1

    if args.check:
        for name, (body, _) in tables.items():
            print(f"===== {name} =====")
            print(body)
        return 0

    for name, (body, source) in tables.items():
        path = args.out_dir / name
        write_tex(path, body, script=SCRIPT, sources=source)
        print(f"escrito: {display_path(path)}")

    # Desenho importado de make_figures.py: a figura do TCC e a mesma do
    # artigo, so que em PNG. Evita duas versoes da mesma matriz.
    from make_figures import apply_style, figure_confusion_matrix, figure_f1_per_class

    apply_style()
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    path = args.figs_dir / "f1_por_classe.png"
    figure_f1_per_class(baselines, path, args.dpi)
    print(f"escrito: {display_path(path)}")

    for slug in BASELINE_ORDER:
        data = baselines[slug]
        path = args.figs_dir / f"cm_{slug_for(data)}.png"
        figure_confusion_matrix(data, path, args.dpi)
        print(f"escrito: {display_path(path)}")

    print(
        f"\n{len(tables)} tabela(s) e 3 figura(s) "
        f"(referência: semente {args.seed}; robustez: {args.seeds}). "
        f"Classes na ordem {CLASS_ORDER}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
