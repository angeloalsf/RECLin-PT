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
2. A DIRECAO e o VEREDITO dos testes de significancia, e nao so os seus
   valores-p (`check_significance_reading`): que nenhum dos dois testes
   rejeita H0 em qualquer das sementes, que a contagem do McNemar e a
   acuracia apontam para o BioBERTpt nas duas, e que so o sinal da
   metrica-alvo inverte na semente 43. CLAIMS confere valores; um valor
   certo com a direcao descrita ao contrario na prosa passaria sem esta
   checagem -- e ja passou.
3. Que nenhum capitulo cita artefato de um pipeline que nao existe
   (`scripts/eda/`, `scripts/baselines/`, `paper/tables/`, `experiments/`).
4. Que nenhuma cifra da rodada anterior de `max_gap` sobreviveu na prosa
   (lista `STALE_NUMBERS`) -- CLAIMS confere um numero contra sua fonte, mas
   nao sabe em quantos arquivos o mesmo numero foi repetido.
5. Que todo `\\input{tabelas/...}` do texto aponta para arquivo existente.
6. As cifras da fase 2 que nao sao campo de JSON: as metricas dos nove
   sistemas da Secao 6.7, recalculadas dos proprios sidecars de predicao
   (`check_fase2_systems`); as 16 comparacoes pareadas da tabela mais as 4 da
   regra pura, com as leituras que a prosa faz delas
   (`check_fase2_significance`); e o teto hipotetico da auditoria de falsos
   positivos, derivado dos TP/FN da execucao sorteada
   (`check_auditoria_teto`).
7. Duas afirmacoes que so o CORPUS sustenta, reproduzidas de `data/splits/` em
   vez de copiadas de um relatorio: a cobertura do lexico e o F1 da regra pura
   no DEV (`check_fase2_dev`).
8. As fracoes de entidades cujo offset recorta exatamente o texto anotado, que
   sustentam a ameaca a validade da Secao 7.4 (`check_offset_alignment`).

Os itens 1 a 5 leem so `results/` e `tcc/src/`. Os itens 6 a 8 leem tambem
`data/splits/`, e o 7 importa `src/negation_lexicon.py` para reproduzir a
inducao do lexico -- e o unico ponto deste script que depende de `src/`, e o
import e tardio por isso.

USO
---
    python scripts/check_tcc_numbers.py
    python scripts/check_tcc_numbers.py --verbose   # imprime tambem os OK
    python scripts/check_tcc_numbers.py --data-dir /outro/data

Sai com codigo 1 na primeira divergencia, para poder rodar em CI.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from _artifacts import (
    CLASS_ORDER,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    FASE2_BASELINE_KEYS,
    FASE2_FILTER_KEYS,
    FASE2_RULE_KEY,
    FASE2_TARGET_CLASS,
    REPO_ROOT,
    SPLIT_ORDER,
    TCC_SRC,
    MissingResultError,
    class_metrics,
    load_fase2_significance,
    load_fase2_systems,
    load_json,
    read_jsonl,
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
    # O exemplo de "macro-F1 sobe enquanto a perda de validacao piora" deixou de
    # existir no BERTimbau s43 na rodada de 17-20/09 (a perda passou a CAIR da
    # epoca 2 para a 3). O par citado em 6.2 e agora o BioBERTpt s42.
    ("curva biobertpt s42: dev_loss ep2", "baseline_biobertpt_seed42.json", "dev_history.1.dev_loss", 0.380, 0.0005),
    ("curva biobertpt s42: dev_loss ep3", "baseline_biobertpt_seed42.json", "dev_history.2.dev_loss", 0.426, 0.0005),
    ("curva biobertpt s42: macroF1 ep2", "baseline_biobertpt_seed42.json", "dev_history.1.dev_macro_f1", 0.654, 0.0005),
    ("curva biobertpt s42: macroF1 ep3", "baseline_biobertpt_seed42.json", "dev_history.2.dev_macro_f1", 0.673, 0.0005),
    ("curva biobertpt s43: F1neg ep1", "baseline_biobertpt_seed43.json", "dev_history.0.dev_negation_of_f1", 0.628, 0.0005),
    ("curva biobertpt s42: F1neg ep1", "baseline_biobertpt_seed42.json", "dev_history.0.dev_negation_of_f1", 0.643, 0.0005),
    ("curva bertimbau s42: F1neg ep1", "baseline_bertimbau_seed42.json", "dev_history.0.dev_negation_of_f1", 0.674, 0.0005),
    ("curva bertimbau s43: F1neg ep1", "baseline_bertimbau_seed43.json", "dev_history.0.dev_negation_of_f1", 0.525, 0.0005),
    # --- Cap. Experimentos: teste, semente 42 -----------------------------
    ("s42 biobertpt: macro-F1", "baseline_biobertpt_seed42.json", "test_macro_f1", 0.693, 0.0005),
    ("s42 bertimbau: macro-F1", "baseline_bertimbau_seed42.json", "test_macro_f1", 0.698, 0.0005),
    ("s42 biobertpt: F1 negation_of", "baseline_biobertpt_seed42.json", "test_f1_per_class.negation_of", 0.696, 0.0005),
    ("s42 bertimbau: F1 negation_of", "baseline_bertimbau_seed42.json", "test_f1_per_class.negation_of", 0.711, 0.0005),
    # O recall de `negation_of` deixou de ser identico entre os dois encoders na
    # rodada de 17-20/09; a palavra "identico" saiu de 6.3 junto com o 0,901.
    ("s42 biobertpt: recall negation_of", "baseline_biobertpt_seed42.json", "sklearn_report.negation_of.recall", 0.934, 0.0005),
    ("s42 bertimbau: recall negation_of", "baseline_bertimbau_seed42.json", "sklearn_report.negation_of.recall", 0.914, 0.0005),
    ("s42 biobertpt: precisao negation_of", "baseline_biobertpt_seed42.json", "sklearn_report.negation_of.precision", 0.555, 0.0005),
    ("s42 bertimbau: precisao negation_of", "baseline_bertimbau_seed42.json", "sklearn_report.negation_of.precision", 0.582, 0.0005),
    ("s42 biobertpt: F1 associated_with", "baseline_biobertpt_seed42.json", "test_f1_per_class.associated_with", 0.442, 0.0005),
    ("s42 bertimbau: F1 associated_with", "baseline_bertimbau_seed42.json", "test_f1_per_class.associated_with", 0.441, 0.0005),
    # Colunas que a Tabela 6 passou a atribuir ao encoder clinico e que 6.3 cita
    # uma a uma para mostrar que a tabela nao ordena os dois modelos.
    ("s42 biobertpt: F1 no_relation", "baseline_biobertpt_seed42.json", "test_f1_per_class.no_relation", 0.942, 0.0005),
    ("s42 bertimbau: F1 no_relation", "baseline_bertimbau_seed42.json", "test_f1_per_class.no_relation", 0.941, 0.0005),
    ("s42 biobertpt: MCC", "baseline_biobertpt_seed42.json", "test_mcc", 0.478, 0.0005),
    ("s42 bertimbau: MCC", "baseline_bertimbau_seed42.json", "test_mcc", 0.476, 0.0005),
    ("s42 biobertpt: n_params", "baseline_biobertpt_seed42.json", "n_params", 177_853_443, 500_000),
    ("s42 bertimbau: n_params", "baseline_bertimbau_seed42.json", "n_params", 108_928_515, 0),
    # --- Cap. Experimentos: significancia ---------------------------------
    ("s42 signif: diferenca", "significance_biobertpt_vs_bertimbau_seed42.json", "target_f1.a_minus_b", -0.015, 0.0005),
    ("s42 signif: McNemar p", "significance_biobertpt_vs_bertimbau_seed42.json", "mcnemar.p_value", 0.479, 0.0005),
    ("s42 signif: b", "significance_biobertpt_vs_bertimbau_seed42.json", "mcnemar.b_only_a_correct", 540, 0),
    ("s42 signif: c", "significance_biobertpt_vs_bertimbau_seed42.json", "mcnemar.c_only_b_correct", 516, 0),
    ("s42 signif: discordantes", "significance_biobertpt_vs_bertimbau_seed42.json", "mcnemar.n_discordant", 1056, 0),
    ("s42 signif: IC baixo", "significance_biobertpt_vs_bertimbau_seed42.json", "paired_bootstrap.ci95_low", -0.051, 0.0005),
    ("s42 signif: IC alto", "significance_biobertpt_vs_bertimbau_seed42.json", "paired_bootstrap.ci95_high", 0.021, 0.0005),
    ("s42 signif: bootstrap p", "significance_biobertpt_vs_bertimbau_seed42.json", "paired_bootstrap.p_value", 0.420, 0.0005),
    ("s42 signif: acuracia A", "significance_biobertpt_vs_bertimbau_seed42.json", "accuracy.a", 0.894, 0.0005),
    ("s42 signif: acuracia B", "significance_biobertpt_vs_bertimbau_seed42.json", "accuracy.b", 0.893, 0.0005),
    ("s43 signif: McNemar p", "significance_biobertpt_vs_bertimbau_seed43.json", "mcnemar.p_value", 0.142, 0.0005),
    # `b` e `c` da semente 43 existem para tornar a DIRECAO do McNemar conferivel,
    # e nao so o seu valor-p. Ate a rodada arquivada em `archive_pre_determinismo`
    # o texto (6.6, 7.1, 8.1 e o resumo) afirmava que o McNemar apontava para o
    # BERTimbau (geral) nas DUAS sementes. Na rodada de 17-20/09 a contagem
    # inverteu nas duas (b > c) e o valor-p subiu de ordens de 1e-5/1e-8 para
    # 0,479 e 0,142, de modo que o teste deixou de rejeitar H0. O texto passou a
    # afirmar o contrario, e quem sustenta essa afirmacao e b > c, aqui e na
    # semente 42. A inversao de SINAL da metrica-alvo continua sendo do
    # `paired_bootstrap` sobre o F1 de `negation_of`, nao do McNemar. Ver
    # `check_significance_reading`, que confere as desigualdades explicitamente.
    ("s43 signif: b", "significance_biobertpt_vs_bertimbau_seed43.json", "mcnemar.b_only_a_correct", 559, 0),
    ("s43 signif: c", "significance_biobertpt_vs_bertimbau_seed43.json", "mcnemar.c_only_b_correct", 510, 0),
    ("s43 signif: discordantes", "significance_biobertpt_vs_bertimbau_seed43.json", "mcnemar.n_discordant", 1069, 0),
    ("s43 signif: acuracia A", "significance_biobertpt_vs_bertimbau_seed43.json", "accuracy.a", 0.902, 0.0005),
    ("s43 signif: acuracia B", "significance_biobertpt_vs_bertimbau_seed43.json", "accuracy.b", 0.900, 0.0005),
    ("s43 signif: IC baixo", "significance_biobertpt_vs_bertimbau_seed43.json", "paired_bootstrap.ci95_low", -0.018, 0.0005),
    ("s43 signif: IC alto", "significance_biobertpt_vs_bertimbau_seed43.json", "paired_bootstrap.ci95_high", 0.057, 0.0005),
    ("s43 signif: bootstrap p", "significance_biobertpt_vs_bertimbau_seed43.json", "paired_bootstrap.p_value", 0.288, 0.0005),
    ("s43 signif: diferenca", "significance_biobertpt_vs_bertimbau_seed43.json", "target_f1.a_minus_b", 0.020, 0.0005),
    # --- Cap. Experimentos / Conclusao: robustez --------------------------
    ("s43 biobertpt: macro-F1", "summary_by_seed.json", "models.biobertpt.metrics.macro_f1.by_seed.43", 0.713, 0.0005),
    ("s43 biobertpt: F1 negation_of", "summary_by_seed.json", "models.biobertpt.metrics.f1_negation_of.by_seed.43", 0.739, 0.0005),
    ("s43 bertimbau: macro-F1", "summary_by_seed.json", "models.bertimbau.metrics.macro_f1.by_seed.43", 0.704, 0.0005),
    ("s43 bertimbau: F1 negation_of", "summary_by_seed.json", "models.bertimbau.metrics.f1_negation_of.by_seed.43", 0.718, 0.0005),
    # Medias entre as duas sementes, citadas em 6.6, na Discussao e na Conclusao.
    ("media biobertpt: macro-F1", "summary_by_seed.json", "models.biobertpt.metrics.macro_f1.mean", 0.703, 0.0005),
    ("media bertimbau: macro-F1", "summary_by_seed.json", "models.bertimbau.metrics.macro_f1.mean", 0.701, 0.0005),
    ("media biobertpt: F1 negation_of", "summary_by_seed.json", "models.biobertpt.metrics.f1_negation_of.mean", 0.717, 0.0005),
    ("media bertimbau: F1 negation_of", "summary_by_seed.json", "models.bertimbau.metrics.f1_negation_of.mean", 0.715, 0.0005),
    # --- Cap. Proposta / Experimentos: configuracao da fase 2 -------------
    # Citados em 5.6.4, 5.6.5, 6.7 e 7.2. A cobertura do lexico no dev e o F1
    # da regra no dev NAO estao neste JSON, que guarda so a configuracao
    # escolhida; os dois sao reproduzidos do corpus em `check_fase2_dev`.
    ("filtro: min_freq", "CALIBRACAO_filtro.json", "min_freq", 3, 0),
    ("filtro: formas do lexico", "CALIBRACAO_filtro.json", "lexicon_size", 11, 0),
    ("filtro: max_gap do espaco de candidatos", "CALIBRACAO_filtro.json", "combined_gap", 25, 0),
    ("regra pura: limiar de gap", "CALIBRACAO_filtro.json", "rule_gap", 1, 0),
    # --- Cap. Discussao: auditoria dos FP remanescentes -------------------
    # As fracoes (54,5%, 27,3%, 18,2%), os 24 casos e o teto hipotetico de
    # precisao e F1 sao derivados destas contagens e conferidos em
    # `check_auditoria_teto`.
    ("auditoria: FP remanescentes", "AUDITORIA_fp_negation_of.json", "n_fp", 33, 0),
    ("auditoria: erros de anotacao do gold", "AUDITORIA_fp_negation_of.json", "counts.provavel erro de anotacao do gold", 18, 0),
    ("auditoria: erros do modelo", "AUDITORIA_fp_negation_of.json", "counts.erro do modelo", 9, 0),
    ("auditoria: ambiguidades genuinas", "AUDITORIA_fp_negation_of.json", "counts.ambiguidade genuina", 6, 0),
]

# Amplitudes citadas em prosa ("0,043" no BioBERTpt, "0,007" no BERTimbau).
# Sao derivadas, nao um campo do JSON, entao vao num teste proprio. A razao
# entre as duas primeiras e o "cerca de seis vezes" de 6.6, da Discussao, da
# Conclusao e do resumo; a razao entre as duas ultimas e o fator citado para o
# macro-F1. Antes da rodada de 17-20/09 essas razoes eram "dez vezes".
AMPLITUDE_CLAIMS = [
    ("biobertpt", "f1_negation_of", 0.043),
    ("bertimbau", "f1_negation_of", 0.007),
    ("biobertpt", "macro_f1", 0.020),
    ("bertimbau", "macro_f1", 0.006),
]


# --------------------------------------------------------------------------- #
# Fase 2: filtro de pista lexical, regra pura e auditoria                      #
# --------------------------------------------------------------------------- #
# As Secoes 5.6, 6.7 e 7.2 citam numeros que nao existem em nenhum
# `baseline_*.json`. Eles vem de tres lugares, e cada um tem a sua lista:
#
#   1. das PREDICOES da fase 2 (`results/<sistema>.preds.json`), recalculadas
#      aqui -- FASE2_SYSTEM_CLAIMS e FASE2_RANGE_CLAIMS;
#   2. dos testes pareados (`results/significance_*_vs_*.json`)
#      -- FASE2_SIGNIFICANCE_CLAIMS;
#   3. do proprio corpus (`data/splits/`), reproduzidos do zero
#      -- DEV_COVERAGE_CLAIM, DEV_RULE_F1_CLAIM e OFFSET_EXACT_CLAIMS.
#
# Os numeros de configuracao e de auditoria (min_freq, formas do lexico, FP por
# categoria) sao campos de JSON e entram em CLAIMS, com o resto.

# Uma linha por sistema da Tabela do Cap. 6.7, na ordem em que a tabela as
# imprime: (chave do sidecar, precisao, recall, F1, macro-F1). Precisao, recall
# e F1 sao de `negation_of`. Os quatro valores sao recalculados de `y_pred`, e
# nao lidos de `FASE2_test_summary.json`: um resumo regerado sozinho ficaria
# conferindo consigo mesmo.
FASE2_SYSTEM_CLAIMS: list[tuple[str, float, float, float, float]] = [
    ("baseline_biobertpt_seed42", 0.5547, 0.9342, 0.6961, 0.6932),
    ("baseline_bertimbau_seed42", 0.5816, 0.9145, 0.7110, 0.6977),
    ("baseline_biobertpt_seed43", 0.6167, 0.9211, 0.7388, 0.7128),
    ("baseline_bertimbau_seed43", 0.5915, 0.9145, 0.7183, 0.7035),
    ("filtro_biobertpt_seed42", 0.7596, 0.9145, 0.8299, 0.7383),
    ("filtro_bertimbau_seed42", 0.7514, 0.8947, 0.8168, 0.7334),
    ("filtro_biobertpt_seed43", 0.8059, 0.9013, 0.8509, 0.7506),
    ("filtro_bertimbau_seed43", 0.7684, 0.8947, 0.8267, 0.7401),
    ("regra_pura", 0.6324, 0.7697, 0.6944, 0.5549),
]

# As faixas que 6.7, 7.2, 8.1 e o resumo citam ("de 0,5547--0,6167 para
# 0,7514--0,8059"). Conferir a faixa nao e o mesmo que conferir cada linha: ela
# afirma tambem QUEM e o extremo, e uma execucao nova que entrasse fora dela
# passaria pelas linhas e reprovaria aqui.
# (descricao, grupo, metrica, minimo, maximo)
FASE2_RANGE_CLAIMS: list[tuple[str, str, str, float, float]] = [
    ("baselines: precisao negation_of", "baseline", "precision", 0.5547, 0.6167),
    ("baselines: recall negation_of", "baseline", "recall", 0.9145, 0.9342),
    ("baselines: F1 negation_of", "baseline", "f1", 0.6961, 0.7388),
    ("baselines: macro-F1", "baseline", "macro_f1", 0.6932, 0.7128),
    ("RECLin-PT: precisao negation_of", "filtro", "precision", 0.7514, 0.8059),
    ("RECLin-PT: recall negation_of", "filtro", "recall", 0.8947, 0.9145),
    ("RECLin-PT: F1 negation_of", "filtro", "f1", 0.8168, 0.8509),
    ("RECLin-PT: macro-F1", "filtro", "macro_f1", 0.7334, 0.7506),
]

# As 16 comparacoes da tabela de significancia da fase 2 e as 4 da regra pura,
# nesta ordem: (sistema A, baseline B, diferenca, IC baixo, IC alto). O arquivo
# de origem e `significance_<A>_vs_<B>.json`.
#
# As 16 estao numa tabela GERADA, que por isso nao pode divergir do JSON. Elas
# entram aqui mesmo assim porque a prosa as RESUME ("as dezesseis comparacoes
# apresentam intervalo inteiramente acima de zero", "o caso mais desfavoravel
# ... da +0,0780"), e um resumo desses envelhece em silencio se uma reexecucao
# mudar a tabela por baixo dele. As 4 da regra pura sao citadas uma a uma na
# prosa de 6.7.3.
FASE2_SIGNIFICANCE_CLAIMS: list[tuple[str, str, float, float, float]] = [
    ("filtro_biobertpt_seed42", "baseline_biobertpt_seed42", +0.1338, +0.0998, +0.1700),
    ("filtro_biobertpt_seed42", "baseline_bertimbau_seed42", +0.1189, +0.0789, +0.1613),
    ("filtro_biobertpt_seed42", "baseline_biobertpt_seed43", +0.0911, +0.0473, +0.1356),
    ("filtro_biobertpt_seed42", "baseline_bertimbau_seed43", +0.1115, +0.0691, +0.1562),
    ("filtro_bertimbau_seed42", "baseline_biobertpt_seed42", +0.1207, +0.0780, +0.1647),
    ("filtro_bertimbau_seed42", "baseline_bertimbau_seed42", +0.1058, +0.0749, +0.1399),
    ("filtro_bertimbau_seed42", "baseline_biobertpt_seed43", +0.0780, +0.0340, +0.1225),
    ("filtro_bertimbau_seed42", "baseline_bertimbau_seed43", +0.0985, +0.0596, +0.1392),
    ("filtro_biobertpt_seed43", "baseline_biobertpt_seed42", +0.1549, +0.1110, +0.2009),
    ("filtro_biobertpt_seed43", "baseline_bertimbau_seed42", +0.1399, +0.0969, +0.1843),
    ("filtro_biobertpt_seed43", "baseline_biobertpt_seed43", +0.1121, +0.0788, +0.1476),
    ("filtro_biobertpt_seed43", "baseline_bertimbau_seed43", +0.1326, +0.0874, +0.1781),
    ("filtro_bertimbau_seed43", "baseline_biobertpt_seed42", +0.1307, +0.0859, +0.1762),
    ("filtro_bertimbau_seed43", "baseline_bertimbau_seed42", +0.1158, +0.0767, +0.1556),
    ("filtro_bertimbau_seed43", "baseline_biobertpt_seed43", +0.0880, +0.0430, +0.1348),
    ("filtro_bertimbau_seed43", "baseline_bertimbau_seed43", +0.1084, +0.0760, +0.1425),
    ("regra_pura", "baseline_biobertpt_seed42", -0.0017, -0.0588, +0.0540),
    ("regra_pura", "baseline_bertimbau_seed42", -0.0166, -0.0716, +0.0386),
    ("regra_pura", "baseline_biobertpt_seed43", -0.0444, -0.1022, +0.0145),
    ("regra_pura", "baseline_bertimbau_seed43", -0.0240, -0.0818, +0.0334),
]

# O par que a prosa de 6.7.2 destaca como "o caso mais desfavoravel possivel":
# a execucao filtrada mais fraca contra o baseline mais forte. Nao e uma cifra,
# e uma leitura -- conferida em `check_fase2_significance`, que recalcula quem
# sao esses dois extremos em vez de acreditar na frase.
FASE2_WORST_CASE = ("filtro_bertimbau_seed42", "baseline_biobertpt_seed43")

# Numeros do DEV citados em 5.6.4, 5.6.5 e 7.2. Nenhum deles esta em
# `results/`: `CALIBRACAO_filtro.json` guarda so a configuracao escolhida, e a
# varredura que a justifica ficou no `.md`. Sao portanto reproduzidos do zero a
# partir de `data/splits/`, induzindo o lexico do TRAIN com o `min_freq`
# congelado e aplicando a regra escolhida ao DEV. Conferir contra o corpus e
# mais forte que conferir contra um JSON que o mesmo pipeline escreveu.
DEV_COVERAGE_CLAIM = 0.9467   # fracao dos pares negation_of do dev com e1 no lexico
DEV_RULE_F1_CLAIM = 0.6935    # F1 de negation_of da regra R3 (gap <= 1) no dev

# Secao 7.4 (ameaca a validade dos offsets). Fracao das entidades cujo offset
# recorta exatamente o texto anotado da propria entidade, por particao.
OFFSET_EXACT_CLAIMS = {"train": 58.9, "dev": 65.8, "test": 64.4}
# "em cerca de 36% das entidades do conjunto de teste" -- complemento do valor
# acima, com a tolerancia frouxa que o "cerca de" do texto autoriza.
OFFSET_MISALIGNED_TEST_CLAIM = 36.0

# Teto hipotetico de precisao e F1 citado em 7.2, sempre com a ressalva de que
# nao e resultado do sistema. Nao e campo de JSON: sai dos TP/FN da execucao
# sorteada somados as contagens do relatorio de auditoria, e e recalculado em
# `check_auditoria_teto`.
AUDIT_CEILING_PRECISION = 0.9384
AUDIT_CEILING_F1 = 0.9195
AUDIT_NON_MODEL_FP = 24  # os 18 de anotacao mais as 6 ambiguidades
AUDIT_SHARE_PCT = {
    "provavel erro de anotacao do gold": 54.5,
    "erro do modelo": 27.3,
    "ambiguidade genuina": 18.2,
}


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


def check_significance_reading(results_dir: Path, verbose: bool) -> list[str]:
    """Confere o VEREDITO e a DIRECAO dos testes, e nao so os seus valores-p.

    O texto afirma, em 6.5, 6.6, 7.1, 7.3, 8.1 e no resumo, quatro coisas que
    CLAIMS sozinho nao consegue checar, porque sao relacoes entre campos e nao
    valores isolados:

    1. NENHUM dos dois testes rejeita H0 em qualquer das sementes. Esta e a
       afirmacao central do capitulo depois da rodada de 17-20/09, e a que mais
       custaria caro se uma reexecucao a invertesse em silencio.
    2. O IC95% do bootstrap inclui o zero nas duas sementes, que e o criterio de
       decisao declarado na metodologia (Secao 4.6).
    3. A contagem de discordancias do McNemar e a acuracia global apontam, nas
       duas sementes, para o BioBERTpt (modelo A, clinico), por margem
       compativel com o acaso.
    4. O sinal da metrica-alvo e negativo na semente 42 e positivo na 43, que e
       a inversao descrita em 6.6.

    Uma reexecucao dos experimentos que trocasse qualquer uma dessas quatro
    leituras reprovaria aqui mesmo que os literais de CLAIMS fossem atualizados
    junto. Foi assim que a versao anterior desta funcao pegou a inversao do
    McNemar: ela codificava `c > b`, e a rodada nova deu `b > c`.
    """
    failures = []
    alfa = 0.05
    sinal_esperado = {42: -1, 43: +1}  # sinal de a_minus_b por semente
    for seed in (42, 43):
        data = load_json(
            results_dir / f"significance_biobertpt_vs_bertimbau_seed{seed}.json"
        )
        b = data["mcnemar"]["b_only_a_correct"]   # so o BioBERTpt (A) acerta
        c = data["mcnemar"]["c_only_b_correct"]   # so o BERTimbau (B) acerta

        p_mcnemar = data["mcnemar"]["p_value"]
        if p_mcnemar <= alfa:
            failures.append(
                f"veredito do McNemar na semente {seed}: o texto diz que o teste "
                f"nao rejeita H0 em nenhuma das sementes, mas p={p_mcnemar:.4f} "
                f"<= {alfa}"
            )
        elif verbose:
            print(f"  ok  McNemar s{seed} nao rejeita H0: p={p_mcnemar:.4f}")

        p_boot = data["paired_bootstrap"]["p_value"]
        ci_low = data["paired_bootstrap"]["ci95_low"]
        ci_high = data["paired_bootstrap"]["ci95_high"]
        if p_boot <= alfa or not (ci_low < 0 < ci_high):
            failures.append(
                f"veredito do bootstrap na semente {seed}: o texto diz que o IC95% "
                f"inclui o zero e o teste nao rejeita H0 nas duas sementes, mas "
                f"p={p_boot:.4f} e IC=[{ci_low:+.4f}; {ci_high:+.4f}]"
            )
        elif verbose:
            print(
                f"  ok  bootstrap s{seed} nao rejeita H0: p={p_boot:.4f}, "
                f"IC=[{ci_low:+.4f}; {ci_high:+.4f}] inclui o zero"
            )

        if not b > c:
            failures.append(
                f"direcao do McNemar na semente {seed}: o texto diz que a contagem "
                f"de discordancias favorece o BioBERTpt (clinico) nas duas "
                f"sementes, mas b={b} nao supera c={c}"
            )
        elif verbose:
            print(f"  ok  McNemar s{seed} favorece o BioBERTpt: b={b} > c={c}")

        if data["accuracy"]["a"] <= data["accuracy"]["b"]:
            failures.append(
                f"direcao da acuracia na semente {seed}: o texto diz que a "
                f"acuracia acompanha a contagem do McNemar, mas o BioBERTpt nao "
                f"fica a frente"
            )
        elif verbose:
            print(f"  ok  acuracia s{seed} acompanha o McNemar (A > B)")

        diff = data["target_f1"]["a_minus_b"]
        esperado = sinal_esperado[seed]
        if (diff > 0) != (esperado > 0):
            failures.append(
                f"sinal da metrica-alvo na semente {seed}: o texto diz que a "
                f"diferenca e {'positiva' if esperado > 0 else 'negativa'} "
                f"(A - B), mas o JSON da {diff:+.4f}"
            )
        elif verbose:
            print(f"  ok  F1 negation_of s{seed}: sinal de A - B = {diff:+.4f}")
    return failures


def check_fase2_systems(systems: dict[str, dict], verbose: bool) -> list[str]:
    """Confere linha a linha a tabela da Secao 6.7 e as faixas citadas em prosa.

    As metricas sao recalculadas dos vetores de predicao (`load_fase2_systems`),
    de modo que a conferencia nao passa por nenhum resumo intermediario.

    Alem dos valores, confere tres leituras que a prosa faz da regra de
    rebaixamento e que nenhum numero isolado sustenta: que o filtro eleva a
    precisao, que ele nunca eleva o recall e que o F1 de `associated_with` fica
    intacto em todos os digitos. Se alguem trocar o destino do rebaixamento ou
    passar a promover predicoes, os numeros ate podem melhorar, mas a descricao
    do sistema na Secao 5.6.3 deixa de ser verdadeira -- e e isso que reprova
    aqui.
    """
    failures = []
    fields = (
        ("precisao", "precision"),
        ("recall", "recall"),
        ("F1", "f1"),
        ("macro-F1", "macro_f1"),
    )
    for claim in FASE2_SYSTEM_CLAIMS:
        key, expected_row = claim[0], claim[1:]
        for (name, field), expected in zip(fields, expected_row):
            actual = systems[key][field]
            if abs(actual - expected) > 0.00005:
                failures.append(
                    f"fase 2, {key}: texto/tabela diz {name} {expected}, "
                    f"{key}.preds.json da {actual:.4f}"
                )
            elif verbose:
                print(f"  ok  fase 2 {key}: {name} {expected} ({actual:.4f})")

    for description, group, field, low, high in FASE2_RANGE_CLAIMS:
        keys = FASE2_BASELINE_KEYS if group == "baseline" else FASE2_FILTER_KEYS
        values = {key: systems[key][field] for key in keys}
        for edge, expected, actual in (
            ("minimo", low, min(values.values())),
            ("maximo", high, max(values.values())),
        ):
            if abs(actual - expected) > 0.00005:
                failures.append(
                    f"faixa {description}: texto diz {edge} {expected}, os "
                    f"sidecars dao {actual:.4f}"
                )
            elif verbose:
                print(f"  ok  faixa {description}: {edge} {expected}")

    # As duas listas estao na mesma ordem (mesmo encoder, mesma semente), entao
    # `zip` pareia cada baseline com a sua propria versao filtrada.
    for base_key, filter_key in zip(FASE2_BASELINE_KEYS, FASE2_FILTER_KEYS):
        base, filtered = systems[base_key], systems[filter_key]
        base_assoc = base["f1_by_class"]["associated_with"]
        filtered_assoc = filtered["f1_by_class"]["associated_with"]
        if filtered_assoc != base_assoc:
            failures.append(
                f"regra de rebaixamento em {filter_key}: o texto (5.6.3 e 6.7.1) "
                f"diz que o F1 de associated_with fica inalterado em todos os "
                f"digitos, mas {base_assoc:.6f} virou {filtered_assoc:.6f}"
            )
        elif verbose:
            print(f"  ok  {filter_key}: F1 de associated_with intacto")

        if filtered["recall"] > base["recall"]:
            failures.append(
                f"regra de rebaixamento em {filter_key}: o texto diz que o "
                f"filtro nao pode elevar o recall, mas ele subiu de "
                f"{base['recall']:.4f} para {filtered['recall']:.4f}"
            )
        elif verbose:
            print(f"  ok  {filter_key}: recall nao subiu")

        if filtered["precision"] <= base["precision"]:
            failures.append(
                f"efeito do filtro em {filter_key}: o texto diz que o ganho e "
                f"de precisao, mas ela nao subiu ({base['precision']:.4f} -> "
                f"{filtered['precision']:.4f})"
            )
        elif verbose:
            print(f"  ok  {filter_key}: precisao subiu")
    return failures


def check_fase2_significance(
    results_dir: Path, systems: dict[str, dict], verbose: bool
) -> list[str]:
    """Confere as 20 comparacoes pareadas e o que a prosa LE nelas.

    Tres leituras, alem dos valores:

    1. As dezesseis comparacoes entre execucao filtrada e baseline tem IC95%
       inteiramente acima de zero. E a afirmacao central da Secao 6.7.2, do
       Capitulo 8 e do resumo.
    2. O par destacado como "caso mais desfavoravel possivel" continua sendo a
       execucao filtrada de menor F1 contra o baseline de maior F1, e continua
       sendo o de menor diferenca entre os dezesseis. A frase descreve uma
       posicao no conjunto, nao um par fixo.
    3. Os quatro intervalos da regra pura contem o zero, que e o que sustenta
       "empata estatisticamente com os quatro" em 6.7.3 e na Conclusao.
    """
    failures = []
    differences: dict[tuple[str, str], float] = {}
    for a_key, b_key, diff, ci_low, ci_high in FASE2_SIGNIFICANCE_CLAIMS:
        data = load_fase2_significance(results_dir, a_key, b_key)
        boot = data["paired_bootstrap"]
        name = f"significance_{a_key}_vs_{b_key}.json"
        for measure, expected, actual in (
            ("diferenca", diff, data["target_f1"]["a_minus_b"]),
            ("IC baixo", ci_low, boot["ci95_low"]),
            ("IC alto", ci_high, boot["ci95_high"]),
        ):
            if abs(actual - expected) > 0.00005:
                failures.append(
                    f"fase 2, {a_key} vs {b_key}: texto/tabela diz {measure} "
                    f"{expected}, {name} tem {actual:.4f}"
                )
            elif verbose:
                print(f"  ok  {a_key} vs {b_key}: {measure} {expected}")

        # Guarda-corpo: o teste pareado tem de ter medido os MESMOS vetores que
        # a tabela de desempenho imprime. Se um dos dois for regerado sozinho,
        # as duas tabelas passam a falar de execucoes diferentes.
        for role, key in (("a", a_key), ("b", b_key)):
            if abs(data["target_f1"][role] - systems[key]["f1"]) > 1e-9:
                failures.append(
                    f"{name}: F1({role})={data['target_f1'][role]:.6f} nao bate "
                    f"com {key}.preds.json ({systems[key]['f1']:.6f}) -- um dos "
                    f"dois foi regerado sem o outro"
                )

        differences[(a_key, b_key)] = data["target_f1"]["a_minus_b"]
        if a_key == FASE2_RULE_KEY:
            if not boot["ci95_low"] < 0 < boot["ci95_high"]:
                failures.append(
                    f"veredito da regra pura contra {b_key}: o texto diz que os "
                    f"quatro intervalos contem o zero, mas "
                    f"IC=[{boot['ci95_low']:+.4f}; {boot['ci95_high']:+.4f}]"
                )
            elif verbose:
                print(f"  ok  regra pura vs {b_key}: IC contem o zero")
        else:
            if not boot["ci95_low"] > 0:
                failures.append(
                    f"veredito de {a_key} contra {b_key}: o texto diz que as "
                    f"dezesseis comparacoes tem IC95% inteiramente acima de "
                    f"zero, mas IC=[{boot['ci95_low']:+.4f}; "
                    f"{boot['ci95_high']:+.4f}]"
                )
            elif verbose:
                print(f"  ok  {a_key} vs {b_key}: IC95% acima de zero")

    filtered_pairs = {
        pair: value
        for pair, value in differences.items()
        if pair[0] != FASE2_RULE_KEY
    }
    observed_worst = min(filtered_pairs, key=filtered_pairs.get)
    if observed_worst != FASE2_WORST_CASE:
        failures.append(
            f"caso mais desfavoravel: o texto destaca "
            f"{FASE2_WORST_CASE[0]} contra {FASE2_WORST_CASE[1]}, mas a menor "
            f"das dezesseis diferencas agora e {observed_worst[0]} contra "
            f"{observed_worst[1]} ({filtered_pairs[observed_worst]:+.4f})"
        )
    elif verbose:
        print(
            f"  ok  caso mais desfavoravel: {observed_worst[0]} vs "
            f"{observed_worst[1]}"
        )

    weakest = min(FASE2_FILTER_KEYS, key=lambda key: systems[key]["f1"])
    strongest = max(FASE2_BASELINE_KEYS, key=lambda key: systems[key]["f1"])
    if (weakest, strongest) != FASE2_WORST_CASE:
        failures.append(
            f"leitura do caso mais desfavoravel: o texto o descreve como a "
            f"execucao filtrada mais fraca contra o baseline mais forte, mas "
            f"esses sao {weakest} e {strongest}"
        )
    elif verbose:
        print("  ok  o caso destacado e o filtrado mais fraco vs o baseline mais forte")
    return failures


def check_fase2_dev(results_dir: Path, data_dir: Path, verbose: bool) -> list[str]:
    """Reproduz do corpus as duas afirmacoes do DEV (Secoes 5.6.4 e 5.6.5).

    `CALIBRACAO_filtro.json` guarda a configuracao escolhida, nao a varredura
    que a justifica, entao a cobertura do lexico e o F1 da regra no
    desenvolvimento nao existem como campo em `results/`. Em vez de copia-los do
    relatorio em Markdown, esta funcao induz o lexico do TRAIN com o `min_freq`
    congelado e mede no DEV. Conferir contra o corpus e mais forte do que
    conferir contra um arquivo que o mesmo pipeline escreveu.
    """
    # Import tardio, e o unico deste script que depende de `src/`: todo o resto
    # roda so com `results/` e `_artifacts.py`.
    for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from candidates import iter_candidate_pairs
        from make_rule_baseline import predict_rule
        from negation_lexicon import LABELS as LEXICON_LABELS
        from negation_lexicon import induce_lexicon, is_cue
    except ImportError as error:
        return [
            f"conferencia do DEV: nao foi possivel importar src/ ({error}). "
            f"Rode este script a partir do repositorio completo."
        ]
    # A inducao loga em INFO; aqui ela e meio, e nao resultado a reportar.
    logging.getLogger("negation_lexicon").setLevel(logging.WARNING)

    failures = []
    if LEXICON_LABELS != CLASS_ORDER:
        return [
            f"src/negation_lexicon.LABELS ({LEXICON_LABELS}) divergiu da ordem "
            f"canonica de classes ({CLASS_ORDER}); as metricas abaixo sairiam "
            f"trocadas"
        ]

    calib = load_json(results_dir / "CALIBRACAO_filtro.json")
    max_gap = calib["combined_gap"]
    lexicon = induce_lexicon(
        read_jsonl(data_dir / "splits" / "train.jsonl"), max_gap, calib["min_freq"]
    )
    if len(lexicon) != calib["lexicon_size"]:
        failures.append(
            f"lexico induzido do train com min_freq={calib['min_freq']}: "
            f"{len(lexicon)} formas, mas CALIBRACAO_filtro.json registra "
            f"{calib['lexicon_size']}"
        )
    elif verbose:
        print(f"  ok  lexico do train: {len(lexicon)} formas")

    candidates = [
        candidate
        for doc in read_jsonl(data_dir / "splits" / "dev.jsonl")
        for candidate in iter_candidate_pairs(doc, max_gap=max_gap)
    ]
    target_pairs = [c for c in candidates if c["label"] == FASE2_TARGET_CLASS]
    coverage = sum(
        1 for c in target_pairs if is_cue(c["e1"], lexicon)
    ) / len(target_pairs)
    if abs(coverage - DEV_COVERAGE_CLAIM) > 0.00005:
        failures.append(
            f"cobertura do lexico no dev: texto diz {DEV_COVERAGE_CLAIM}, o "
            f"corpus da {coverage:.4f}"
        )
    elif verbose:
        print(f"  ok  cobertura do lexico no dev: {coverage:.4f}")

    y_true = [CLASS_ORDER.index(c["label"]) for c in candidates]
    y_pred = predict_rule(candidates, lexicon, calib["rule"], calib["rule_gap"])
    f1 = class_metrics(y_true, y_pred, CLASS_ORDER.index(FASE2_TARGET_CLASS))["f1"]
    if abs(f1 - DEV_RULE_F1_CLAIM) > 0.00005:
        failures.append(
            f"F1 da regra {calib['rule']} (gap <= {calib['rule_gap']}) no dev: "
            f"texto diz {DEV_RULE_F1_CLAIM}, o corpus da {f1:.4f}"
        )
    elif verbose:
        print(f"  ok  F1 da regra pura no dev: {f1:.4f}")
    return failures


def check_offset_alignment(data_dir: Path, verbose: bool) -> list[str]:
    """Confere as fracoes de entidades bem ancoradas citadas na Secao 7.4.

    A ameaca a validade descrita la e uma propriedade do corpus, nao um
    resultado de experimento, entao a fonte e `data/splits/` e nao `results/`.
    """
    failures = []
    exact_pct = {}
    for split in SPLIT_ORDER:
        exact = total = 0
        for doc in read_jsonl(data_dir / "splits" / f"{split}.jsonl"):
            text = doc["text"]
            for entity in doc["entities"]:
                total += 1
                if text[entity["start"]:entity["end"]] == entity["text"]:
                    exact += 1
        exact_pct[split] = 100.0 * exact / total
        expected = OFFSET_EXACT_CLAIMS[split]
        if abs(exact_pct[split] - expected) > 0.05:
            failures.append(
                f"offsets exatos em {split}: texto diz {expected}%, o corpus da "
                f"{exact_pct[split]:.1f}%"
            )
        elif verbose:
            print(f"  ok  offsets exatos em {split}: {expected}%")

    # "em cerca de 36% das entidades do conjunto de teste": complemento do
    # valor acima, com a tolerancia frouxa que o "cerca de" autoriza.
    misaligned = 100.0 - exact_pct["test"]
    if abs(misaligned - OFFSET_MISALIGNED_TEST_CLAIM) > 1.0:
        failures.append(
            f"entidades deslocadas no teste: texto diz cerca de "
            f"{OFFSET_MISALIGNED_TEST_CLAIM:.0f}%, o corpus da {misaligned:.1f}%"
        )
    elif verbose:
        print(f"  ok  entidades deslocadas no teste: {misaligned:.1f}%")
    return failures


def check_auditoria_teto(
    results_dir: Path, systems: dict[str, dict], verbose: bool
) -> list[str]:
    """Confere o teto hipotetico de precisao e F1 citado na Secao 7.2.

    O teto nao e um campo de `AUDITORIA_fp_negation_of.json`: ele e derivado dos
    TP/FN da execucao sorteada com os FP que a auditoria nao atribui ao modelo.
    Recalcula-lo aqui e o que impede o texto de citar um teto que ja nao
    corresponde a contagem publicada -- foi justamente um deslocamento de uma
    casa nesse par de numeros que precisou ser corrigido antes de o texto
    entrar.
    """
    failures = []
    audit = load_json(results_dir / "AUDITORIA_fp_negation_of.json")
    drawn = audit["drawn_run"]
    if drawn not in systems:
        return [
            f"auditoria: a execucao sorteada ({drawn}) nao esta entre os "
            f"sistemas da fase 2 -- o sorteio ou os sidecars mudaram"
        ]
    metrics = systems[drawn]

    # Os FP contados um a um pelo relatorio tem de ser os FP do sidecar.
    if metrics["fp"] != audit["n_fp"]:
        failures.append(
            f"auditoria: o relatorio classifica {audit['n_fp']} falsos "
            f"positivos, mas {drawn}.preds.json tem {metrics['fp']}"
        )
    elif verbose:
        print(f"  ok  auditoria: {audit['n_fp']} FP, igual ao sidecar")

    counts = audit["counts"]
    for kind, expected in AUDIT_SHARE_PCT.items():
        share = 100.0 * counts[kind] / audit["n_fp"]
        if abs(share - expected) > 0.05:
            failures.append(
                f"auditoria, {kind}: texto diz {expected}% dos FP, as contagens "
                f"dao {share:.1f}%"
            )
        elif verbose:
            print(f"  ok  auditoria, {kind}: {expected}%")

    model_errors = counts["erro do modelo"]
    non_model = audit["n_fp"] - model_errors
    if non_model != AUDIT_NON_MODEL_FP:
        failures.append(
            f"auditoria: o texto fala em {AUDIT_NON_MODEL_FP} casos que nao "
            f"configuram erro de leitura clinica, as contagens dao {non_model}"
        )
    elif verbose:
        print(f"  ok  auditoria: {non_model} FP fora de erro do modelo")

    # Teto: os FP que nao sao erro do modelo passam a contar como acerto. Os FN
    # continuam contando, porque a auditoria nao os examinou.
    tp, fn = metrics["tp"], metrics["fn"]
    ceiling_precision = tp / (tp + model_errors)
    ceiling_f1 = 2 * tp / (2 * tp + model_errors + fn)
    for measure, expected, actual in (
        ("precisao", AUDIT_CEILING_PRECISION, ceiling_precision),
        ("F1", AUDIT_CEILING_F1, ceiling_f1),
    ):
        if abs(actual - expected) > 0.00005:
            failures.append(
                f"teto hipotetico da auditoria ({measure}): texto diz "
                f"{expected}, o par (sidecar, contagens) da {actual:.4f}"
            )
        elif verbose:
            print(f"  ok  teto hipotetico, {measure}: {expected} ({actual:.4f})")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confere os numeros do TCC contra results/ e data/.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "corpus e splits congelados; fonte das afirmacoes do dev e dos "
            "offsets (padrao: data/)."
        ),
    )
    parser.add_argument("--tcc-src", type=Path, default=TCC_SRC)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        ghost_failures, pending = check_ghost_paths(args.tcc_src, args.verbose)
        # Carregado uma vez e passado adiante: as tres conferencias da fase 2
        # falam dos mesmos nove vetores de predicao.
        systems = load_fase2_systems(args.results_dir)
        failures = (
            check_claims(args.results_dir, args.verbose)
            + check_significance_reading(args.results_dir, args.verbose)
            + check_fase2_systems(systems, args.verbose)
            + check_fase2_significance(args.results_dir, systems, args.verbose)
            + check_auditoria_teto(args.results_dir, systems, args.verbose)
            + check_fase2_dev(args.results_dir, args.data_dir, args.verbose)
            + check_offset_alignment(args.data_dir, args.verbose)
            + ghost_failures
            + check_stale_numbers(args.tcc_src, args.verbose)
            + check_inputs(args.tcc_src, args.verbose)
        )
    except (MissingResultError, KeyError, IndexError, ValueError) as error:
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

    total = (
        len(CLAIMS)
        + len(AMPLITUDE_CLAIMS)
        + 4 * len(FASE2_SYSTEM_CLAIMS)
        + 2 * len(FASE2_RANGE_CLAIMS)
        + 3 * len(FASE2_SIGNIFICANCE_CLAIMS)
        + len(AUDIT_SHARE_PCT)
        + 3  # os 24 casos e o par (precisão, F1) do teto hipotético
        + 2  # cobertura do léxico e F1 da regra pura, no dev
        + len(OFFSET_EXACT_CLAIMS)
        + 1  # as entidades deslocadas no teste
    )
    print(
        f"OK: {total} afirmações numéricas do texto conferem com results/ e "
        f"data/; nenhum dos dois testes rejeita H0, e a direção do McNemar e o "
        f"sinal da métrica-alvo batem com o texto nas duas sementes; as "
        f"dezesseis comparações da fase 2 têm IC95% acima de zero e os quatro "
        f"intervalos da regra pura contêm o zero; nenhum caminho fantasma novo; "
        f"nenhum dos {len(STALE_NUMBERS)} números obsoletos na prosa; todos os "
        f"\\input{{tabelas/...}} existem."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
