# OUTLINE — TCC RECLin-PT

Mapa dos capítulos. Mantenha esta página em sincronia com `src/textuais/`. Sempre que um capítulo mudar de status ou ganhar nova dependência de tabela/figura, edite aqui.

**Status legend:** `vazio` · `esboço` · `escrito` · `revisado` · `final`.

## Pré-textuais (`src/pre_textuais/`)

| Arquivo | Status | Notas |
|---|---|---|
| `dedicatoria.tex` | vazio | placeholder do template |
| `agradecimentos.tex` | vazio | placeholder a redigir antes da defesa |
| `epigrafe.tex` | vazio | placeholder a redigir antes da defesa |
| `resumo.tex` | escrito | PT + EN reescritos a partir dos resultados reais (BioBERTpt × BERTimbau, sementes 42/43) |
| `siglas.tex` | escrito | acrescentado `MCC`; `TF-IDF` continua em uso no referencial teórico |
| `simbolos.tex` | vazio | provavelmente vazio neste TCC |

## Textuais (`src/textuais/`)

| # | Arquivo | Status | Objetivo | Depende de |
|---|---|---|---|---|
| 1 | `introducao.tex` | escrito | Contexto clínico em PT, motivação, pergunta de projeto (*o pré-treinamento clínico importa?*), objetivos, organização. | — |
| 2 | `referencial_teorico.tex` | escrito | NER e RE em textos clínicos; regras (NegEx) vs. aprendizado; transformers (BERT, XLM-R, BioBERTpt). | bibliografia.bib |
| 3 | `trabalhos_relacionados.tex` | escrito | i2b2/n2c2 (EN); SemClinBr; BioBERTpt; posicionamento. | bibliografia.bib |
| 4 | `metodologia.tex` | escrito | Pipeline (6 etapas + determinismo); corpus SemClinBr (EDA, 5 figuras); splits 80/10/10 estratificados; pares candidatos (`max_gap=20`) + teto de recall por classe; protocolo de avaliação (McNemar + bootstrap pareado + robustez à semente); ambiente. | `tabelas/{distribuicao_relacoes,distancia_por_tipo,particao_relacoes,candidatos_por_particao,teto_recall}.tex`, `imagens/eda/*.png` |
| 5 | `proposta_prototipo.tex` | escrito | Os dois encoders comparados (BioBERTpt clínico × BERTimbau geral) e a paridade de implementação; tratamento do desbalanceamento; `sec:markers` (marcadores de posição, sem tipo semântico — descrição alinhada ao código). | bibliografia.bib |
| 6 | `experimentos_e_resultados.tex` | escrito | Configuração; curvas de treino por época; resultados no teste (semente 42); matrizes de confusão; significância; robustez à semente 43. | `tabelas/{dev_history,resultados,significancia,significancia_seed43,robustez_semente}.tex`, `imagens/resultados/*.png` |
| 7 | `discussao.tex` | escrito | Ausência de vantagem do pré-treinamento clínico; a classe `associated_with`; limitações; ameaças à validade. | — |
| 8 | `conclusao.tex` | escrito | Contribuições, resultados, trabalhos futuros. | — |

## Apêndices (`src/apendices/`) — incluídos em `main.tex` via `apendicesenv`

| Arquivo | Status | Conteúdo |
|---|---|---|
| `apendice_a_sha256sums.tex` | escrito | Hashes SHA-256 dos splits congelados. Gerados de `data/splits/MANIFEST.json` via `tabelas/sha256_splits.tex`. |
| `apendice_b_lexico_negacao.tex` | escrito | Marcadores de negação do português clínico — agora **caracterização linguística**, não parâmetro de modelo (nenhum modelo avaliado usa léxico). |
| `apendice_c_hiperparametros.tex` | escrito | Parâmetros gerais + ajuste fino, lidos do campo `config` de `results/baseline_*.json`. |

## Dependências código ↔ TCC

Todas as tabelas e figuras com dados são geradas por script, a partir de `data/` e `results/`. **Nenhuma é editada à mão.** Os fragmentos em `src/tabelas/` são flutuantes ABNT completos (com `\caption` e `\label`): o capítulo faz só `\input{tabelas/nome}`, sem envolver em `table`/`quadro`.

| Artefato | Gerado por | Fonte dos dados |
|---|---|---|
| `tabelas/distribuicao_relacoes.tex` | `scripts/make_tcc_eda.py` | `data/processed/dataset.jsonl` |
| `tabelas/distancia_por_tipo.tex` | `scripts/make_tcc_eda.py` | `data/processed/dataset.jsonl` |
| `tabelas/particao_relacoes.tex` | `scripts/make_tcc_eda.py` | `data/splits/*.jsonl` + `MANIFEST.json` |
| `tabelas/candidatos_por_particao.tex` | `scripts/make_tcc_eda.py` | `data/splits/*.jsonl`, `max_gap` de `results/` |
| `tabelas/teto_recall.tex` | `scripts/make_tcc_eda.py` | idem |
| `tabelas/sha256_splits.tex` | `scripts/make_tcc_eda.py` | `data/splits/MANIFEST.json` |
| `imagens/eda/0[1-5]_*.png` | `scripts/make_tcc_eda.py` | `data/` |
| `tabelas/resultados.tex` | `scripts/make_tcc_artifacts.py` | `results/baseline_*_seed42.json` |
| `tabelas/significancia.tex` | `scripts/make_tcc_artifacts.py` | `results/significance_*_seed42.json` |
| `tabelas/significancia_seed43.tex` | `scripts/make_tcc_artifacts.py` | `results/significance_*_seed43.json` |
| `tabelas/robustez_semente.tex` | `scripts/make_tcc_artifacts.py` | `results/summary_by_seed.json` |
| `imagens/resultados/{f1_por_classe,cm_*}.png` | `scripts/make_tcc_artifacts.py` | `results/baseline_*_seed42.json` |
| `tabelas/dev_history.tex` | `scripts/make_tcc_curves.py` | campo `dev_history` de `results/baseline_*.json` |
| `imagens/resultados/curvas_treino*.png` | `scripts/make_tcc_curves.py` | idem |
| `imagens/processo-semclinbr.png` | manual (figura conceitual) | — |

Regenerar tudo:

```bash
python scripts/make_tcc_eda.py --check-against-results
python scripts/make_tcc_artifacts.py
python scripts/make_tcc_curves.py
```

`--check-against-results` recomputa os pares candidatos dos splits em disco e compara com o `n_candidates` gravado pelos experimentos. Se falhar, os splits não são os que foram treinados — não publique nada antes de resolver.

## Divergências resolvidas

Registro do que já foi corrigido, para não ser reintroduzido:

- **Marcadores tipados / grupos UMLS.** `sec:markers` (Cap. 5) descrevia a entrada como marcadores **tipados** com agregação nos 15 grupos semânticos UMLS via `normalize_type` (`scripts/training/markers.py`, arquivo inexistente), e o cabeçote como concatenação de `h_E1`/`h_E2`. `build_marked_window` em `src/relation_extraction.py` insere apenas `[E1] … [/E1]` / `[E2] … [/E2]`, **sem tipo**, e a classificação usa `AutoModelForSequenceClassification` (representação agregada da sequência). A seção foi reescrita para descrever o código. A proposta de agregação UMLS e a citação que a sustentava (`mccray2001aggregating` aqui, `mccray:01` no artigo) foram **removidas por completo** — inclusive do `.bib` dos dois documentos e do `artigo.bbl` versionado. Os trabalhos futuros mantêm apenas o item genérico de enriquecer a entrada com o tipo semântico, sem prescrever esquema. Mesma correção aplicada a `artigo-sbc/artigo.tex`.
- **Tabelas órfãs de B0/B1/B2.** `comparacao_baselines.tex`, `confusao_{majority,rules,lr}.tex` e `mcnemar.tex` foram **apagadas** de `src/tabelas/` — resíduo de um pipeline (`scripts/baselines/`, `scripts/evaluation/`, `paper/tables/`) que nunca existiu neste repositório, sem `\input` apontando para elas.
- **Proporção de `negation_of` por partição.** Corrigida de "13,7%–15,0%" (parse antigo) para 14,09% / 15,70% / 12,16%, aqui e no artigo SBC, com a ressalva de que a estratificação garante presença da classe, não proporção igual.

## Diretrizes de escrita

- Citações no estilo `abntex2-alf` (autor-data): `\cite{chave}`, `\citeonline{chave}` para "Autor (ano)".
- Tabelas geradas pelo pipeline: **não editar à mão**, regenerar.
- Cada capítulo começa com `\chapter{Título}` (sem numeração manual — abnTeX2 cuida).
- Idioma principal `brazil`; trechos em inglês com `\foreignlanguage{english}{...}` se necessário.
- Marcadores `easyReview` (`\alert`, `\add`, `\info`) podem ser usados durante revisão.
- **Quadros vs. Tabelas:** dados numéricos → `table` (Tabela); conteúdo qualitativo/estruturado (hiperparâmetros, léxicos, hashes) → `quadro` (Quadro). O ambiente `quadro` foi habilitado em `src/macros.tex`.
- **Tabelas largas:** os geradores já aplicam `\resizebox{\textwidth}{!}{...}` quando necessário.
