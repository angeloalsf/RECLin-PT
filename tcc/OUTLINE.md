# OUTLINE — TCC RECLin-PT

Mapa dos capítulos. Mantenha esta página em sincronia com `src/textuais/`. Sempre que um capítulo mudar de status ou ganhar nova dependência de tabela/figura, edite aqui.

**Status legend:** `vazio` · `esboço` · `escrito` · `revisado` · `final`.

## Pré-textuais (`src/pre_textuais/`)

| Arquivo | Status | Notas |
|---|---|---|
| `dedicatoria.tex` | vazio | placeholder do template |
| `agradecimentos.tex` | vazio | placeholder a redigir antes da defesa |
| `epigrafe.tex` | vazio | placeholder a redigir antes da defesa |
| `resumo.tex` | escrito | PT + EN redigidos a partir do estado atual do projeto; revisar após experimentos |
| `siglas.tex` | escrito | lista ampliada (BERT, PLN, TF-IDF, XML, IC…) |
| `simbolos.tex` | vazio | provavelmente vazio neste TCC |

## Textuais (`src/textuais/`)

| # | Arquivo | Status | Objetivo | Depende de |
|---|---|---|---|---|
| 1 | `introducao.tex` | escrito | Contexto clínico em PT, motivação, problema, objetivos geral/específicos, organização do trabalho. | — |
| 2 | `referencial_teorico.tex` | escrito | NER e RE em textos clínicos; regras (NegEx) vs. aprendizado; transformers (BERT, XLM-R, BioBERTpt). | bibliografia.bib |
| 3 | `trabalhos_relacionados.tex` | escrito | i2b2/n2c2 (EN, TODO confirmar refs); SemClinBr; BioBERTpt; posicionamento. | bibliografia.bib |
| 4 | `metodologia.tex` | escrito | Visão geral do pipeline (`sec:pipeline`, 6 etapas + determinismo); Corpus SemClinBr (EDA c/ 5 figuras + categorias de entidade); parsing → splits 80/10/10 estratificados (tabela real + Sechidis + `fig:split_dist`); rótulos {negation_of, associated_with, no_relation}; pares candidatos (max_gap=200, 5 propriedades + teto de recall); protocolo de avaliação (bootstrap inst./doc. + McNemar); ambiente experimental. | `tabelas/distribuicao_relacoes.tex`, `tabelas/distancia_por_tipo.tex`, `tabelas/teto_recall.tex`, `imagens/eda/*.png` |
| 5 | `proposta_prototipo.tex` | escrito | Cadeia B0/B1/B2 + BioBERTpt, explicação didática; representação de entrada (`sec:markers`: marcadores tipados + agregação UMLS de McCray 2001); detalhes finais do transformer marcados como TODO. | bibliografia.bib (`mccray2001aggregating`) |
| 6 | `experimentos_e_resultados.tex` | escrito (parcial) | Baselines B0/B1/B2 reportados com números reais (B2 macro-F1 0,637); só BioBERTpt como TODO (a executar no Colab). Tabela curada com negrito no melhor + linha do modelo principal. | `tabelas/comparacao_baselines.tex` |
| 7 | `discussao.tex` | escrito | Análise dos baselines, limitações, ameaças à validade. | — |
| 8 | `conclusao.tex` | escrito | Contribuições, trabalhos futuros (TODO quantificar pós-experimentos). | — |

## Apêndices (`src/apendices/`) — criados e incluídos em `main.tex` via `apendicesenv`

| Arquivo | Status | Conteúdo |
|---|---|---|
| `apendice_a_sha256sums.tex` | escrito | Hashes SHA-256 dos splits congelados (`data/splits/SHA256SUMS.txt`). Crítico para reprodutibilidade. |
| `apendice_b_lexico_negacao.tex` | escrito | Marcadores NegEx-PT: "não", "sem", "nega", "ausência", "ausente", "afebril", "anictérico". |
| `apendice_c_hiperparametros.tex` | escrito (parcial) | Parâmetros gerais + B1 + B2 (reais) + BioBERTpt (planejados, de `docs/plano_notebook_biobertpt.md`); valores efetivos do transformer marcados como TODO. |

## Dependências código ↔ TCC

| Arquivo TCC | Gerado por | Quando regenerar |
|---|---|---|
| `src/tabelas/distribuicao_relacoes.tex` | `scripts/eda/eda_relations.py` (saída em `paper/tables/eda_reltype_distribution.tex`) | quando o corpus parseado mudar |
| `src/tabelas/distancia_por_tipo.tex` | `scripts/eda/eda_relations.py` (saída em `paper/tables/eda_distance_by_type.tex`) | idem |
| `src/tabelas/comparacao_baselines.tex` | `scripts/evaluation/compare_baselines.py` (saída em `paper/tables/baselines.tex`) — já inclui B0/B1/B2 | a cada novo baseline/modelo |
| `src/imagens/eda/*.png` | `scripts/eda/eda_relations.py` (cópia de `experiments/figures/eda/`) | quando o corpus parseado mudar |
| `src/imagens/eda/05_relations_by_split.png` (`fig:split_dist`) | `scripts/eda/plot_split_distribution.py` (lê `split_stats.json`; cópia de `experiments/figures/eda/`) | quando os splits mudarem |
| `src/imagens/processo-semclinbr.png` | manual (figura conceitual) | quando arquitetura mudar |

> **Nota:** a tabela curada de resultados no Cap. 6 (`tab:resultados`) é escrita à
> mão no `.tex` (com negrito no melhor valor e a linha do BioBERTpt a executar),
> pois acrescenta destaque e o modelo planejado além do artefato auto-gerado
> `tabelas/comparacao_baselines.tex`, que permanece como espelho fiel do pipeline.

> **ATENÇÃO (inconsistência a resolver):** o pipeline Python escreve as tabelas em
> `paper/tables/`, mas o TCC as inclui de `tcc/src/tabelas/`. Por ora os arquivos
> em `tcc/src/tabelas/` são cópias manuais sincronizadas. O ideal é apontar o
> `--tables-dir` dos scripts (`eda_relations.py`, `compare_baselines.py`) para
> `tcc/src/tabelas/` e regenerar, eliminando a cópia manual.

## Diretrizes de escrita

- Citações no estilo `abntex2-alf` (autor-data): `\cite{chave}`, `\citeonline{chave}` para "Autor (ano)".
- Tabelas geradas pelo pipeline: **não editar à mão**, regenerar.
- Cada capítulo começa com `\chapter{Título}` (sem numeração manual — abnTeX2 cuida).
- Idioma principal `brazil`; trechos em inglês com `\foreignlanguage{english}{...}` se necessário.
- Marcadores `easyReview` (`\alert`, `\add`, `\info`) podem ser usados durante revisão.
- **Quadros vs. Tabelas:** dados numéricos → `table` (Tabela); conteúdo qualitativo/estruturado (hiperparâmetros, léxicos, hashes) → `quadro` (Quadro). O ambiente `quadro` foi habilitado em `src/macros.tex` (o `iftex.cls` preparava a Lista de Quadros e o contador, mas deixava o `\newfloat` comentado; o override reusa o contador/`.loq` existentes sem recriá-los).
- **Tabelas largas:** envolver o `\input{tabelas/...}` em `\resizebox{\textwidth}{!}{...}` para não estourar a margem.
