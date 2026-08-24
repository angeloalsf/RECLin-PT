"""
Confere os numeros citados EM PROSA no TCC contra suas fontes.

POR QUE ESTE SCRIPT EXISTE
--------------------------
Tabelas e figuras sao geradas por `make_tcc_*.py` e, portanto, nao podem
divergir dos dados. As frases do corpo do texto, nao: "o BioBERTpt atinge
macro-F1 de 0,707" e digitada a mao e envelhece em silencio. Foi assim que
o capitulo de metodologia passou a descrever um corpus de 11.353 relacoes e
uma janela `max_gap=200` que nenhum experimento usou.

Este script fecha essa lacuna pelo unico caminho pratico: cada afirmacao
numerica do texto e registrada aqui junto com o caminho ate o valor de
origem, e a conferencia e executavel. Ele NAO analisa o LaTeX -- a lista
abaixo e mantida a mao. Ao editar uma frase com numero, atualize a entrada
correspondente; ao acrescentar uma, acrescente a entrada.

O QUE VERIFICA
--------------
1. Afirmacoes numericas do corpo do texto (lista `CLAIMS`).
2. A DIRECAO dos testes de significancia, e nao so os seus valores-p
   (`check_mcnemar_direction`): que o McNemar aponta para o BERTimbau nas
   duas sementes e que so o sinal da metrica-alvo inverte na semente 43.
   CLAIMS confere valores; um valor certo com a direcao descrita ao
   contrario na prosa passaria sem esta checagem -- e ja passou.
3. Que nenhum capitulo cita artefato de um pipeline que nao existe
   (`scripts/eda/`, `scripts/baselines/`, `paper/tables/`, `experiments/`).
4. Que nenhuma cifra da rodada anterior de `max_gap` sobreviveu na prosa
   (lista `STALE_NUMBERS`) -- CLAIMS confere um numero contra sua fonte, mas
   nao sabe em quantos arquivos o mesmo numero foi repetido.
5. Que todo `\\input{tabelas/...}` do texto aponta para arquivo existente.

USO
---
    python scripts/check_tcc_numbers.py
    python scripts/check_tcc_numbers.py --verbose   # imprime tambem os OK

Sai com codigo 1 na primeira divergencia, para poder rodar em CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _artifacts import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    REPO_ROOT,
    TCC_SRC,
    MissingResultError,
    load_json,
)

TEXT_DIRS = ["textuais", "pre_textuais", "apendices"]

# Caminhos de uma arvore de scripts que nunca existiu neste repositorio. O
# TCC os citava como "Fonte:" de tabelas e figuras; qualquer reaparicao e
# sinal de que um trecho antigo voltou.
GHOST_PATHS = [
    "scripts/eda",
    "scripts/baselines",
    "scripts/evaluation",
    "scripts/training",
    "scripts/splitting",
    "scripts/utils",
    "paper/tables",
    "experiments/results",
    "docs/plano",
    "split_stats.json",
    "SHA256SUMS",
]


# Numeros que pertenciam a rodada de `max_gap=20` e foram substituidos pelo
# retreino de 21/08/2026. Diferente de CLAIMS, que confere um numero contra sua
# fonte, esta lista procura o LITERAL no texto: serve para pegar a mesma cifra
# repetida num arquivo que ninguem lembrou de atualizar.
#
# Foi exatamente essa a falha que motivou a checagem: `93,06%` sobrevivia em
# `proposta_prototipo.tex` e o par `0,724 / 0,734` no `resumo.tex`, ambos fora
# de qualquer entrada de CLAIMS, porque CLAIMS confere o valor uma vez e nao
# sabe em quantos arquivos ele aparece.
#
# So vale para prosa (`textuais/`, `pre_textuais/`, `apendices/`). `tabelas/` e
# saida gerada e pode legitimamente conter qualquer numero.
STALE_NUMBERS: list[tuple[str, str]] = [
    (r"128\,380", "candidatos de treino de max_gap=20 (agora 152.686)"),
    (r"15\,994", "candidatos de validacao de max_gap=20 (agora 19.064)"),
    (r"16\,074", "candidatos de teste de max_gap=20 (agora 19.210)"),
    (r"14\,959", "no_relation no teste de max_gap=20 (agora 18.062)"),
    (r"93{,}06", "%% no_relation de max_gap=20 (agora 94,02)"),
    (r"89{,}36", "teto de recall do teste de max_gap=20 (agora 92,00)"),
    (r"12{,}02", "perda de associated_with de max_gap=20 (agora 9,11)"),
    (r"max\_gap}$=20$", "janela antiga; o valor ativo e 25"),
    (r"max\_gap}=20", "janela antiga; o valor ativo e 25"),
]


# Divergencias ja diagnosticadas, registradas em `tcc/OUTLINE.md` e pendentes
# de decisao editorial. Sao reportadas como AVISO a cada execucao, mas nao
# reprovam a conferencia -- do contrario o script ficaria permanentemente
# vermelho e deixaria de sinalizar regressoes novas.
#
# Vazio no momento: a divergencia de `sec:markers` (marcadores tipados com
# agregacao UMLS x marcadores de posicao) foi resolvida corrigindo o texto do
# TCC e do artigo SBC para descrever o que `build_marked_window` de fato faz.
KNOWN_PENDING: dict[tuple[str, str], str] = {}


def get(data: dict, path: str):
    """Navega um dicionario/lista por um caminho `a.b.c`.

    Um segmento numerico pode ser tanto indice de lista (`dev_history.1`)
    quanto chave de dicionario (`by_seed.43`, em que a semente e chave
    porque JSON so tem chaves de texto). A chave literal tem prioridade.
    """
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif part.isdigit():
            current = current[int(part)]
        else:
            raise KeyError(f"{part!r} (caminho completo: {path!r})")
    return current


# Cada afirmacao: (descricao, arquivo JSON, caminho no JSON, valor no texto,
# tolerancia). A tolerancia cobre o arredondamento da prosa -- "0,707" para
# 0,7065224987 e correto; 0,71 nao seria.
CLAIMS: list[tuple[str, str, str, float, float]] = [
    # --- Cap. Metodologia: caracterizacao do corpus -----------------------
    ("corpus: documentos", "tcc_eda.json", "corpus.n_docs", 1000, 0),
    ("corpus: entidades", "tcc_eda.json", "corpus.n_entities", 45508, 0),
    ("corpus: entidades/doc", "tcc_eda.json", "corpus.entities_per_doc_mean", 45.5, 0.05),
    ("corpus: relacoes", "tcc_eda.json", "corpus.n_relations", 11458, 0),
    ("corpus: relacoes/doc", "tcc_eda.json", "corpus.relations_per_doc_mean", 11.5, 0.05),
    ("corpus: mediana rel/doc", "tcc_eda.json", "corpus.relations_per_doc_median", 10, 0),
    ("corpus: p90 rel/doc", "tcc_eda.json", "corpus.relations_per_doc_p90", 24, 0),
    ("corpus: max rel/doc", "tcc_eda.json", "corpus.relations_per_doc_max", 65, 0),
    ("corpus: docs sem relacao", "tcc_eda.json", "corpus.docs_without_relation", 46, 0),
    ("corpus: %% docs sem relacao", "tcc_eda.json", "corpus.docs_without_relation_pct", 4.60, 0.005),
    ("corpus: %% associated_with", "tcc_eda.json", "corpus.by_type_pct.associated_with", 85.98, 0.005),
    ("corpus: %% negation_of", "tcc_eda.json", "corpus.by_type_pct.negation_of", 14.02, 0.005),
    # --- Cap. Metodologia: distancias -------------------------------------
    ("dist assoc: media", "tcc_eda.json", "distance_by_type.associated_with.mean", 11.8, 0.05),
    ("dist assoc: p50", "tcc_eda.json", "distance_by_type.associated_with.p50", 4, 0),
    ("dist assoc: p95", "tcc_eda.json", "distance_by_type.associated_with.p95", 43, 0),
    ("dist neg: media", "tcc_eda.json", "distance_by_type.negation_of.mean", 3.5, 0.05),
    ("dist neg: p50", "tcc_eda.json", "distance_by_type.negation_of.p50", 1, 0),
    ("dist neg: p75", "tcc_eda.json", "distance_by_type.negation_of.p75", 1, 0),
    ("dist neg: p95", "tcc_eda.json", "distance_by_type.negation_of.p95", 15, 0),
    # --- Cap. Metodologia: categorias de entidade -------------------------
    ("entidades: Abbreviation", "tcc_eda.json", "top_entity_categories.0.1", 12592, 0),
    ("entidades: Finding", "tcc_eda.json", "top_entity_categories.1.1", 6827, 0),
    ("entidades: Ther. or Prev. Procedure", "tcc_eda.json", "top_entity_categories.2.1", 4759, 0),
    ("entidades: Sign or Symptom", "tcc_eda.json", "top_entity_categories.3.1", 4616, 0),
    ("entidades: Negation (Apend. B)", "tcc_eda.json", "top_entity_categories.8.1", 2646, 0),
    # --- Cap. Metodologia: particoes --------------------------------------
    ("split treino: relacoes", "tcc_eda.json", "splits.train.total", 9221, 0),
    ("split validacao: relacoes", "tcc_eda.json", "splits.dev.total", 987, 0),
    ("split teste: relacoes", "tcc_eda.json", "splits.test.total", 1250, 0),
    ("split treino: %% negation_of", "tcc_eda.json", "splits.train.negation_pct", 14.09, 0.005),
    ("split validacao: %% negation_of", "tcc_eda.json", "splits.dev.negation_pct", 15.70, 0.005),
    ("split teste: %% negation_of", "tcc_eda.json", "splits.test.negation_pct", 12.16, 0.005),
    ("split teste: negation_of gold", "tcc_eda.json", "splits.test.negation_of", 152, 0),
    ("split teste: associated_with gold", "tcc_eda.json", "splits.test.associated_with", 1098, 0),
    # --- Cap. Metodologia: candidatos e teto de recall --------------------
    # Reancorado em 21/08/2026 para `max_gap=25` (retreino dos 4 experimentos).
    # Os valores de `max_gap=20` seguem em `results/archive_max_gap20/`: rodar
    # `check_tcc_numbers.py --results-dir results/archive_max_gap20` reprova de
    # proposito, e assim deve ser -- o texto descreve a rodada de 25.
    ("max_gap efetivo", "tcc_eda.json", "max_gap", 25, 0),
    ("candidatos treino", "tcc_eda.json", "candidates.train.total", 152686, 0),
    ("candidatos validacao", "tcc_eda.json", "candidates.dev.total", 19064, 0),
    ("candidatos teste", "tcc_eda.json", "candidates.test.total", 19210, 0),
    ("teste: no_relation", "tcc_eda.json", "candidates.test.no_relation", 18062, 0),
    ("teste: associated_with", "tcc_eda.json", "candidates.test.associated_with", 996, 0),
    ("teste: negation_of", "tcc_eda.json", "candidates.test.negation_of", 152, 0),
    ("teste: %% no_relation", "tcc_eda.json", "candidates.test.no_relation_pct", 94.02, 0.005),
    ("teto teste: perdidas negation_of", "tcc_eda.json", "recall_ceiling.test.lost_by_type.negation_of", 0, 0),
    ("teto teste: perdidas associated_with", "tcc_eda.json", "recall_ceiling.test.lost_by_type.associated_with", 100, 0),
    ("teto teste: %%", "tcc_eda.json", "recall_ceiling.test.ceiling_pct", 92.00, 0.005),
    ("teto treino: perdidas negation_of", "tcc_eda.json", "recall_ceiling.train.lost_by_type.negation_of", 27, 0),
    ("teto validacao: perdidas negation_of", "tcc_eda.json", "recall_ceiling.dev.lost_by_type.negation_of", 3, 0),
    # --- Cap. Metodologia / Apend. C: hiperparametros ---------------------
    ("hp: ctx_chars", "tcc_eda.json", "pipeline_config.ctx_chars", 128, 0),
    ("hp: max_length", "tcc_eda.json", "pipeline_config.max_length", 128, 0),
    ("hp: batch_size", "tcc_eda.json", "pipeline_config.batch_size", 64, 0),
    ("hp: epocas", "tcc_eda.json", "pipeline_config.epochs", 3, 0),
    ("hp: learning rate", "tcc_eda.json", "pipeline_config.lr", 2e-5, 0),
    # --- Cap. Experimentos: curvas ----------------------------------------
    ("curva bertimbau s43: dev_loss ep2", "baseline_bertimbau_seed43.json", "dev_history.1.dev_loss", 0.426, 0.0005),
    ("curva bertimbau s43: dev_loss ep3", "baseline_bertimbau_seed43.json", "dev_history.2.dev_loss", 0.474, 0.0005),
    ("curva bertimbau s43: macroF1 ep2", "baseline_bertimbau_seed43.json", "dev_history.1.dev_macro_f1", 0.654, 0.0005),
    ("curva bertimbau s43: macroF1 ep3", "baseline_bertimbau_seed43.json", "dev_history.2.dev_macro_f1", 0.680, 0.0005),
    ("curva biobertpt s43: F1neg ep1", "baseline_biobertpt_seed43.json", "dev_history.0.dev_negation_of_f1", 0.588, 0.0005),
    ("curva biobertpt s42: F1neg ep1", "baseline_biobertpt_seed42.json", "dev_history.0.dev_negation_of_f1", 0.436, 0.0005),
    ("curva bertimbau s42: F1neg ep1", "baseline_bertimbau_seed42.json", "dev_history.0.dev_negation_of_f1", 0.630, 0.0005),
    ("curva bertimbau s43: F1neg ep1", "baseline_bertimbau_seed43.json", "dev_history.0.dev_negation_of_f1", 0.573, 0.0005),
    # --- Cap. Experimentos: teste, semente 42 -----------------------------
    ("s42 biobertpt: macro-F1", "baseline_biobertpt_seed42.json", "test_macro_f1", 0.675, 0.0005),
    ("s42 bertimbau: macro-F1", "baseline_bertimbau_seed42.json", "test_macro_f1", 0.686, 0.0005),
    ("s42 biobertpt: F1 negation_of", "baseline_biobertpt_seed42.json", "test_f1_per_class.negation_of", 0.677, 0.0005),
    ("s42 bertimbau: F1 negation_of", "baseline_bertimbau_seed42.json", "test_f1_per_class.negation_of", 0.694, 0.0005),
    ("s42 biobertpt: recall negation_of", "baseline_biobertpt_seed42.json", "sklearn_report.negation_of.recall", 0.901, 0.0005),
    ("s42 bertimbau: recall negation_of", "baseline_bertimbau_seed42.json", "sklearn_report.negation_of.recall", 0.901, 0.0005),
    ("s42 biobertpt: F1 associated_with", "baseline_biobertpt_seed42.json", "test_f1_per_class.associated_with", 0.415, 0.0005),
    ("s42 bertimbau: F1 associated_with", "baseline_bertimbau_seed42.json", "test_f1_per_class.associated_with", 0.426, 0.0005),
    ("s42 biobertpt: n_params", "baseline_biobertpt_seed42.json", "n_params", 177_853_443, 500_000),
    ("s42 bertimbau: n_params", "baseline_bertimbau_seed42.json", "n_params", 108_928_515, 0),
    # --- Cap. Experimentos: significancia ---------------------------------
    ("s42 signif: diferenca", "significance_biobertpt_vs_bertimbau_seed42.json", "target_f1.a_minus_b", -0.017, 0.0005),
    ("s42 signif: McNemar p", "significance_biobertpt_vs_bertimbau_seed42.json", "mcnemar.p_value", 1.1e-5, 0.5e-5),
    ("s42 signif: b", "significance_biobertpt_vs_bertimbau_seed42.json", "mcnemar.b_only_a_correct", 540, 0),
    ("s42 signif: c", "significance_biobertpt_vs_bertimbau_seed42.json", "mcnemar.c_only_b_correct", 695, 0),
    ("s42 signif: discordantes", "significance_biobertpt_vs_bertimbau_seed42.json", "mcnemar.n_discordant", 1235, 0),
    ("s42 signif: IC baixo", "significance_biobertpt_vs_bertimbau_seed42.json", "paired_bootstrap.ci95_low", -0.053, 0.0005),
    ("s42 signif: IC alto", "significance_biobertpt_vs_bertimbau_seed42.json", "paired_bootstrap.ci95_high", 0.018, 0.0005),
    ("s42 signif: bootstrap p", "significance_biobertpt_vs_bertimbau_seed42.json", "paired_bootstrap.p_value", 0.345, 0.005),
    ("s42 signif: acuracia A", "significance_biobertpt_vs_bertimbau_seed42.json", "accuracy.a", 0.880, 0.0005),
    ("s42 signif: acuracia B", "significance_biobertpt_vs_bertimbau_seed42.json", "accuracy.b", 0.888, 0.0005),
    ("s43 signif: McNemar p", "significance_biobertpt_vs_bertimbau_seed43.json", "mcnemar.p_value", 2.7e-8, 0.5e-8),
    # `b` e `c` da semente 43 existem para tornar a DIRECAO do McNemar conferivel,
    # e nao so o seu valor-p. O texto (6.6, 7.1, 8.1 e o resumo) afirma que o
    # McNemar aponta para o BERTimbau (geral) nas DUAS sementes; quem sustenta essa
    # afirmacao e c > b, aqui e na semente 42. Ja houve uma versao do texto que
    # atribuia ao McNemar da semente 43 a direcao oposta -- a inversao de sinal e
    # do `paired_bootstrap` sobre o F1 de `negation_of`, nao do McNemar. Ver
    # `check_mcnemar_direction`, que confere a desigualdade explicitamente.
    ("s43 signif: b", "significance_biobertpt_vs_bertimbau_seed43.json", "mcnemar.b_only_a_correct", 442, 0),
    ("s43 signif: c", "significance_biobertpt_vs_bertimbau_seed43.json", "mcnemar.c_only_b_correct", 624, 0),
    ("s43 signif: discordantes", "significance_biobertpt_vs_bertimbau_seed43.json", "mcnemar.n_discordant", 1066, 0),
    ("s43 signif: acuracia A", "significance_biobertpt_vs_bertimbau_seed43.json", "accuracy.a", 0.898, 0.0005),
    ("s43 signif: acuracia B", "significance_biobertpt_vs_bertimbau_seed43.json", "accuracy.b", 0.908, 0.0005),
    ("s43 signif: IC baixo", "significance_biobertpt_vs_bertimbau_seed43.json", "paired_bootstrap.ci95_low", 0.016, 0.0005),
    ("s43 signif: IC alto", "significance_biobertpt_vs_bertimbau_seed43.json", "paired_bootstrap.ci95_high", 0.090, 0.0005),
    ("s43 signif: bootstrap p", "significance_biobertpt_vs_bertimbau_seed43.json", "paired_bootstrap.p_value", 0.0054, 0.00005),
    ("s43 signif: diferenca", "significance_biobertpt_vs_bertimbau_seed43.json", "target_f1.a_minus_b", 0.053, 0.0005),
    # --- Cap. Experimentos / Conclusao: robustez --------------------------
    ("s43 biobertpt: macro-F1", "summary_by_seed.json", "models.biobertpt.metrics.macro_f1.by_seed.43", 0.712, 0.0005),
    ("s43 biobertpt: F1 negation_of", "summary_by_seed.json", "models.biobertpt.metrics.f1_negation_of.by_seed.43", 0.754, 0.0005),
    ("s43 bertimbau: macro-F1", "summary_by_seed.json", "models.bertimbau.metrics.macro_f1.by_seed.43", 0.704, 0.0005),
    ("s43 bertimbau: F1 negation_of", "summary_by_seed.json", "models.bertimbau.metrics.f1_negation_of.by_seed.43", 0.702, 0.0005),
    # Medias entre as duas sementes, citadas em 6.6, na Discussao e na Conclusao.
    ("media biobertpt: macro-F1", "summary_by_seed.json", "models.biobertpt.metrics.macro_f1.mean", 0.694, 0.0005),
    ("media bertimbau: macro-F1", "summary_by_seed.json", "models.bertimbau.metrics.macro_f1.mean", 0.695, 0.0005),
    ("media biobertpt: F1 negation_of", "summary_by_seed.json", "models.biobertpt.metrics.f1_negation_of.mean", 0.715, 0.0005),
    ("media bertimbau: F1 negation_of", "summary_by_seed.json", "models.bertimbau.metrics.f1_negation_of.mean", 0.698, 0.0005),
]

# Amplitudes citadas em prosa ("0,070" no BioBERTpt, "0,008" no BERTimbau).
# Sao derivadas, nao um campo do JSON, entao vao num teste proprio.
AMPLITUDE_CLAIMS = [
    ("biobertpt", "f1_negation_of", 0.078),
    ("bertimbau", "f1_negation_of", 0.008),
    ("biobertpt", "macro_f1", 0.038),
    ("bertimbau", "macro_f1", 0.018),
]


def check_claims(results_dir: Path, verbose: bool) -> list[str]:
    failures = []
    cache: dict[str, dict] = {}
    for description, filename, path, expected, tolerance in CLAIMS:
        if filename not in cache:
            cache[filename] = load_json(results_dir / filename)
        actual = get(cache[filename], path)
        if abs(actual - expected) > tolerance:
            failures.append(
                f"{description}: texto diz {expected}, "
                f"{filename}:{path} tem {actual}"
            )
        elif verbose:
            print(f"  ok  {description}: {expected} ({actual})")

    summary = load_json(results_dir / "summary_by_seed.json")
    for slug, metric, expected in AMPLITUDE_CLAIMS:
        by_seed = summary["models"][slug]["metrics"][metric]["by_seed"]
        actual = max(by_seed.values()) - min(by_seed.values())
        if abs(actual - expected) > 0.0005:
            failures.append(
                f"amplitude {slug}/{metric}: texto diz {expected}, "
                f"summary_by_seed.json da {actual:.4f}"
            )
        elif verbose:
            print(f"  ok  amplitude {slug}/{metric}: {expected} ({actual:.4f})")
    return failures


def check_ghost_paths(tcc_src: Path, verbose: bool) -> tuple[list[str], list[str]]:
    """Procura citacoes a caminhos que nao existem no repositorio.

    Devolve (falhas, pendencias_conhecidas). O que esta em `KNOWN_PENDING`
    vira aviso, e nao erro: sao divergencias ja identificadas e registradas
    em `tcc/OUTLINE.md`, aguardando decisao editorial. Manter a checagem
    verde para todo o resto e o que a torna util; silenciar a pendencia
    seria o oposto do proposito deste script, por isso ela continua sendo
    impressa a cada execucao.
    """
    failures, pending = [], []
    for directory in TEXT_DIRS:
        for path in sorted((tcc_src / directory).glob("*.tex")):
            content = path.read_text(encoding="utf-8")
            # O LaTeX escapa `_` como `\_`; normaliza antes de procurar.
            flat = content.replace("\\_", "_")
            for ghost in GHOST_PATHS:
                for number, line in enumerate(flat.splitlines(), start=1):
                    if ghost not in line:
                        continue
                    location = f"{path.relative_to(tcc_src)}"
                    message = (
                        f"{location}:{number} cita '{ghost}', que nao existe "
                        f"no repositorio"
                    )
                    reason = KNOWN_PENDING.get((location, ghost))
                    (pending if reason else failures).append(
                        f"{message}\n      motivo: {reason}" if reason else message
                    )
    if verbose and not failures:
        print("  ok  nenhum caminho fantasma novo citado no texto")
    return failures, pending


def check_stale_numbers(tcc_src: Path, verbose: bool) -> list[str]:
    """Procura, na PROSA, cifras que o retreino tornou obsoletas."""
    failures = []
    for directory in TEXT_DIRS:
        for path in sorted((tcc_src / directory).glob("*.tex")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for literal, reason in STALE_NUMBERS:
                for number, line in enumerate(lines, start=1):
                    if literal in line:
                        failures.append(
                            f"{path.relative_to(tcc_src)}:{number} ainda cita "
                            f"'{literal}' -- {reason}"
                        )
    if verbose and not failures:
        print(f"  ok  nenhum dos {len(STALE_NUMBERS)} numeros obsoletos na prosa")
    return failures


def check_inputs(tcc_src: Path, verbose: bool) -> list[str]:
    failures = []
    pattern = re.compile(r"\\input\{(tabelas/[^}]+)\}")
    for directory in TEXT_DIRS:
        for path in sorted((tcc_src / directory).glob("*.tex")):
            for name in pattern.findall(path.read_text(encoding="utf-8")):
                target = tcc_src / (name if name.endswith(".tex") else name + ".tex")
                if not target.exists():
                    failures.append(
                        f"{path.relative_to(tcc_src)}: \\input{{{name}}} aponta "
                        f"para arquivo inexistente"
                    )
                elif verbose:
                    print(f"  ok  \\input{{{name}}}")
    return failures


def check_mcnemar_direction(results_dir: Path, verbose: bool) -> list[str]:
    """Confere a DIRECAO do McNemar, e nao so o seu valor-p.

    O texto afirma, em 6.6, 7.1, 8.1 e no resumo, que o McNemar aponta para o
    BERTimbau (modelo B, geral) nas DUAS sementes, e que o que inverte na
    semente 43 e apenas o `paired_bootstrap` sobre o F1 de `negation_of`. As
    entradas de CLAIMS conferem os valores de `b` e `c` um a um; esta checagem
    confere a RELACAO entre eles, que e o que a prosa de fato afirma, e o sinal
    do bootstrap em cada semente. Uma reexecucao dos experimentos que trocasse a
    direcao do McNemar reprovaria aqui mesmo que os literais de CLAIMS fossem
    atualizados junto.
    """
    failures = []
    bootstrap_esperado = {42: -1, 43: +1}  # sinal de a_minus_b por semente
    for seed in (42, 43):
        data = load_json(
            results_dir / f"significance_biobertpt_vs_bertimbau_seed{seed}.json"
        )
        b = data["mcnemar"]["b_only_a_correct"]   # so o BioBERTpt (A) acerta
        c = data["mcnemar"]["c_only_b_correct"]   # so o BERTimbau (B) acerta
        if not c > b:
            failures.append(
                f"direcao do McNemar na semente {seed}: o texto diz que o McNemar "
                f"aponta para o BERTimbau (geral) nas duas sementes, mas "
                f"c={c} nao supera b={b}"
            )
        elif verbose:
            print(f"  ok  McNemar s{seed} aponta para o BERTimbau: c={c} > b={b}")

        if data["accuracy"]["b"] <= data["accuracy"]["a"]:
            failures.append(
                f"direcao da acuracia na semente {seed}: o texto diz que a "
                f"acuracia acompanha o McNemar, mas o BERTimbau nao fica a frente"
            )
        elif verbose:
            print(f"  ok  acuracia s{seed} acompanha o McNemar (B > A)")

        diff = data["target_f1"]["a_minus_b"]
        esperado = bootstrap_esperado[seed]
        if (diff > 0) != (esperado > 0):
            failures.append(
                f"sinal da metrica-alvo na semente {seed}: o texto diz que a "
                f"diferenca e {'positiva' if esperado > 0 else 'negativa'} "
                f"(A - B), mas o JSON da {diff:+.4f}"
            )
        elif verbose:
            print(f"  ok  F1 negation_of s{seed}: sinal de A - B = {diff:+.4f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confere os numeros do TCC contra results/ e data/.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tcc-src", type=Path, default=TCC_SRC)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        ghost_failures, pending = check_ghost_paths(args.tcc_src, args.verbose)
        failures = (
            check_claims(args.results_dir, args.verbose)
            + check_mcnemar_direction(args.results_dir, args.verbose)
            + ghost_failures
            + check_stale_numbers(args.tcc_src, args.verbose)
            + check_inputs(args.tcc_src, args.verbose)
        )
    except (MissingResultError, KeyError, IndexError) as error:
        print(f"ERRO ao ler a fonte: {error}", file=sys.stderr)
        print(
            "  Rode `python scripts/make_tcc_eda.py` para (re)gerar "
            "results/tcc_eda.json.",
            file=sys.stderr,
        )
        return 1

    if pending:
        print(f"{len(pending)} pendência(s) conhecida(s) — ver tcc/OUTLINE.md:\n")
        for item in pending:
            print(f"  ! {item}\n")

    if failures:
        print(f"{len(failures)} divergência(s):\n", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(CLAIMS) + len(AMPLITUDE_CLAIMS)} afirmações numéricas do texto "
        f"conferem com results/; a direção do McNemar e o sinal da métrica-alvo "
        f"batem com o texto nas duas sementes; nenhum caminho fantasma novo; "
        f"nenhum dos {len(STALE_NUMBERS)} números obsoletos na prosa; todos os "
        f"\\input{{tabelas/...}} existem."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
