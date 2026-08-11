# Makefile do RECLin-PT -- ALTERNATIVA OPCIONAL ao `run.sh`.
#
# `run.sh` continua valendo e continua sendo o caminho simples: roda o pipeline
# inteiro, na ordem, do inicio ao fim. Use-o se voce quer uma execucao unica e
# sequencial e nao quer instalar nada.
#
# Este Makefile existe para o outro caso: **rebuild incremental**. Cada baseline
# custa ~2h25 de GPU (os quatro, ~10 h). O `make` compara datas de arquivo e so
# refaz o alvo cuja entrada esta mais nova que a saida -- entao rodar `make` com
# tudo pronto nao retreina nada.
#
#   make -n all        # mostra o que SERIA executado, sem executar.
#                      #   Rode isto ANTES de comprometer horas de GPU.
#   make all           # pipeline completo, pulando o que ja esta atualizado
#   make status        # o que esta pronto e o que falta, sem tocar em nada
#   make aggregate     # so a agregacao multi-seed
#   make tables figures
#
# Os comandos de cada alvo sao **exatamente** os documentados no README ("Como
# rodar") e em `artigo-sbc/README.md` -- nenhuma flag nova. Se voce mudar um
# hiperparametro aqui, mude no README e no `run.sh` tambem: os tres tem de
# continuar concordando.
#
# ---------------------------------------------------------------------------
# POLITICA DE DEPENDENCIAS (a decisao de projeto deste arquivo)
#
#   data/splits/ e results/  = DADOS e EVIDENCIA experimental. Dependem so dos
#               DADOS de entrada (corpus, splits, preds), nunca dos .py que os
#               produziram.
#   artigo-sbc/tables|figs/, results/summary_by_seed.json
#             = SAIDA DERIVADA barata (segundos). Depende dos dados E do
#               script gerador.
#
# Por que os .py nao entram nas dependencias de `data/splits/` e `results/`:
# porque um `make` seria capaz de pedir 10 h de GPU porque alguem corrigiu um
# comentario em `relation_extraction.py` -- ou, pior, de reembaralhar os splits
# versionados porque `make_splits.py` foi tocado, invalidando em cascata os
# quatro resultados. O experimento e definido pelos dados e pelos
# hiperparametros; refatoracao que nao muda nenhum dos dois nao invalida um
# resultado. Quando o codigo mudar a ponto de invalidar de verdade, force
# explicitamente: `make -B baseline_biobertpt_seed42`.
# ---------------------------------------------------------------------------
#
# ATENCAO (Windows): `make` nao vem com o Windows. Use Git Bash com make, WSL
# ou Chocolatey -- ou simplesmente `bash run.sh`, que nao exige nada disso.

PYTHON ?= python

# Hiperparametros do artigo. NAO sao os defaults do argparse (32/75/192).
HP := --epochs 3 --batch-size 64 --max-gap 20 --max-length 128

XML_DIR    := SemClinBr-xml-public-v1
DATASET    := data/processed/dataset.jsonl
SPLITS_DIR := data/splits
TRAIN      := $(SPLITS_DIR)/train.jsonl
SPLITS     := $(TRAIN) $(SPLITS_DIR)/dev.jsonl $(SPLITS_DIR)/test.jsonl
MANIFEST   := $(SPLITS_DIR)/MANIFEST.json
XMLS       := $(wildcard $(XML_DIR)/*.xml)

MODELS := biobertpt bertimbau
SEEDS  := 42 43

BASELINES := $(foreach m,$(MODELS),$(foreach s,$(SEEDS),results/baseline_$(m)_seed$(s).json))
SIGNIF    := $(foreach s,$(SEEDS),results/significance_biobertpt_vs_bertimbau_seed$(s).json)
SUMMARY   := results/summary_by_seed.json

TABLES  := artigo-sbc/tables/tab_resultados.tex artigo-sbc/tables/tab_signif.tex
FIGURES := artigo-sbc/figs/f1_por_classe.pdf artigo-sbc/figs/cm_biobertpt.pdf \
           artigo-sbc/figs/cm_bertimbau.pdf

.PHONY: all help parse splits baselines significance aggregate tables figures \
        artigo status clean-derived \
        tcc tcc-eda tcc-artifacts tcc-curves tcc-check \
        baseline_biobertpt_seed42 baseline_biobertpt_seed43 \
        baseline_bertimbau_seed42 baseline_bertimbau_seed43 \
        significance_seed42 significance_seed43

all: splits baselines significance aggregate tables figures

help:
	@echo "Alvos principais: all parse splits baselines significance aggregate tables figures artigo"
	@echo "TCC:              tcc-eda tcc-artifacts tcc-curves tcc-check tcc"
	@echo "Alvos individuais: baseline_<modelo>_seed<N>  (modelo: $(MODELS) | seed: $(SEEDS))"
	@echo "                   significance_seed<N>"
	@echo "Inspecao: make status | make -n <alvo> (dry-run, nao executa)"

# ------------------------------------------------------------------ passo 1
# Requer o corpus restrito em $(XML_DIR)/. Sem ele $(XMLS) e vazio e o alvo
# nao dispara -- e o comportamento certo: quem so quer reproduzir os baselines
# usa os splits versionados e nunca precisa deste passo.
$(DATASET): $(XMLS)
	$(PYTHON) src/parse_semclinbr.py --xml-dir $(XML_DIR) --out $(DATASET)

parse: $(DATASET)

# ------------------------------------------------------------------ passo 2
# Uma execucao do script produz os tres .jsonl de uma vez. `train.jsonl` e o
# alvo canonico; `dev.jsonl` e `test.jsonl` saem junto e sao declarados com
# pre-requisito DE ORDEM (`|`) para o make nao rodar o script tres vezes nem
# marcar os dois como "atualizados" e disparar um retreino em cascata.
$(TRAIN): $(DATASET)
	$(PYTHON) src/make_splits.py --input $(DATASET) --out-dir $(SPLITS_DIR)

$(SPLITS_DIR)/dev.jsonl $(SPLITS_DIR)/test.jsonl: | $(TRAIN)

# O manifesto descreve os splits, entao depende deles (e nao o contrario).
# `--manifest-only` recalcula os hashes sem reembaralhar nada -- rodar este
# alvo nunca sobrescreve os splits versionados.
$(MANIFEST): $(SPLITS)
	$(PYTHON) src/make_splits.py --input $(DATASET) --out-dir $(SPLITS_DIR) --manifest-only

splits: $(SPLITS) $(MANIFEST)

# ------------------------------------------------------------------ passo 3
# Uma regra por (modelo, semente): ~2h25 de GPU cada. O `.preds.json` sai junto
# do `.json` na mesma execucao -- a regra vazia mais abaixo so registra o
# vinculo, para o passo 4 saber que retreinar um baseline invalida a
# significancia daquela semente.
define BASELINE_RULE
results/baseline_$(1)_seed$(2).json: $$(SPLITS)
	$$(PYTHON) src/baseline_$(1).py --splits-dir $$(SPLITS_DIR) \
	    $$(HP) --seed $(2) \
	    --ckpt-dir checkpoints/$(1)_seed$(2) \
	    --out results/baseline_$(1)_seed$(2).json

baseline_$(1)_seed$(2): results/baseline_$(1)_seed$(2).json
endef
$(foreach m,$(MODELS),$(foreach s,$(SEEDS),$(eval $(call BASELINE_RULE,$(m),$(s)))))

results/baseline_%.preds.json: results/baseline_%.json
	@:

baselines: $(BASELINES)

# ------------------------------------------------------------------ passo 4
# Depende dos dois `.preds.json` DA MESMA semente -- nunca 42 contra 43: o
# teste e pareado e `significance.py` aborta se os y_true divergirem.
define SIGNIF_RULE
results/significance_biobertpt_vs_bertimbau_seed$(1).json: \
		results/baseline_biobertpt_seed$(1).preds.json \
		results/baseline_bertimbau_seed$(1).preds.json
	$$(PYTHON) src/significance.py \
	    --a results/baseline_biobertpt_seed$(1).preds.json \
	    --b results/baseline_bertimbau_seed$(1).preds.json \
	    --target negation_of --n-boot 10000 --seed $(1) \
	    --out results/significance_biobertpt_vs_bertimbau_seed$(1).json

significance_seed$(1): results/significance_biobertpt_vs_bertimbau_seed$(1).json
endef
$(foreach s,$(SEEDS),$(eval $(call SIGNIF_RULE,$(s))))

significance: $(SIGNIF)

# ------------------------------------------------------------------ passo 5
# Agregacao multi-seed: le os quatro baseline_*.json e grava a media/desvio.
$(SUMMARY): $(BASELINES) scripts/aggregate_seeds.py scripts/_artifacts.py
	$(PYTHON) scripts/aggregate_seeds.py

aggregate: $(SUMMARY)

# ------------------------------------------------------------------ passo 6
# Tabelas e figuras da semente do artigo (42, o default dos scripts). Para
# outra semente use `--seed` direto nos scripts, como em artigo-sbc/README.md:
# nao ha alvo aqui porque o texto do artigo discute os numeros da 42 e teria de
# ser revisto a mao junto.
#
# Cada script grava varios arquivos numa unica execucao. Como no passo 2, um
# arquivo e o alvo canonico e os demais o acompanham por pre-requisito de
# ordem (`|`) -- senao o make chamaria `make_figures.py` tres vezes, uma por
# PDF.
TABLES_CANON  := artigo-sbc/tables/tab_resultados.tex
FIGURES_CANON := artigo-sbc/figs/f1_por_classe.pdf

$(TABLES_CANON): $(BASELINES) $(SIGNIF) scripts/make_tables.py scripts/_artifacts.py
	$(PYTHON) scripts/make_tables.py

$(filter-out $(TABLES_CANON),$(TABLES)): | $(TABLES_CANON)

$(FIGURES_CANON): $(BASELINES) scripts/make_figures.py scripts/_artifacts.py
	$(PYTHON) scripts/make_figures.py

$(filter-out $(FIGURES_CANON),$(FIGURES)): | $(FIGURES_CANON)

tables: $(TABLES)
figures: $(FIGURES)

# ------------------------------------------------------------------ artigo
# Roda de dentro de artigo-sbc/: os caminhos de artigo.tex sao relativos a la.
artigo: $(TABLES) $(FIGURES)
	cd artigo-sbc && latexmk -pdf artigo.tex

# ------------------------------------------------------------------- tcc
# Artefatos do TCC: mesma evidencia dos do artigo, sob convencoes ABNT
# (legenda acima, "Fonte:" abaixo, `quadro` para conteudo qualitativo), mais
# a EDA (que sai de data/, nao de results/) e as curvas por epoca.
#
# Nao ha alvo por arquivo como em `tables`/`figures`: cada script grava de 4 a
# 11 arquivos numa execucao que leva segundos, entao a granularidade fina so
# complicaria o Makefile sem economizar tempo.
TCC_EDA_CANON := tcc/src/tabelas/distribuicao_relacoes.tex

# --check-against-results recomputa os candidatos dos splits em disco e
# compara com o n_candidates gravado no treino. E o unico jeito de detectar
# que os splits mudaram depois dos experimentos sem retreinar -- por isso a
# flag esta no alvo, e nao apenas disponivel.
$(TCC_EDA_CANON): $(SPLITS) $(MANIFEST) $(BASELINES) \
                  scripts/make_tcc_eda.py scripts/_artifacts.py
	$(PYTHON) scripts/make_tcc_eda.py --check-against-results

tcc-eda: $(TCC_EDA_CANON)

tcc-artifacts: $(BASELINES) $(SIGNIF) $(SUMMARY)
	$(PYTHON) scripts/make_tcc_artifacts.py

tcc-curves: $(BASELINES)
	$(PYTHON) scripts/make_tcc_curves.py

# Confere as afirmacoes numericas do CORPO DO TEXTO (as tabelas ja vem dos
# scripts). Roda depois de gerar, porque le results/tcc_eda.json.
tcc-check: tcc-eda tcc-artifacts tcc-curves
	$(PYTHON) scripts/check_tcc_numbers.py

tcc: tcc-check
	cd tcc && docker compose run --rm build

status:
	@echo "== entradas =="
	@for f in $(DATASET) $(SPLITS) $(MANIFEST); do \
	    if [ -f $$f ]; then echo "  ok      $$f"; else echo "  FALTA   $$f"; fi; done
	@echo "== resultados (custam GPU) =="
	@for f in $(BASELINES) $(SIGNIF); do \
	    if [ -f $$f ]; then echo "  ok      $$f"; else echo "  FALTA   $$f"; fi; done
	@echo "== saida derivada (segundos) =="
	@for f in $(SUMMARY) $(TABLES) $(FIGURES); do \
	    if [ -f $$f ]; then echo "  ok      $$f"; else echo "  FALTA   $$f"; fi; done

# Apaga so o que e barato de refazer. NUNCA apaga results/baseline_* nem
# results/significance_* -- 10 h de GPU. Por isso o alvo nao se chama `clean`.
clean-derived:
	rm -f $(TABLES) $(FIGURES) $(SUMMARY)
