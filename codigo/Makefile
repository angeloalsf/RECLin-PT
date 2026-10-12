# RECLin-PT — pontos de entrada (etapa 7).
#
# Uso, a partir da raiz do repositório:   make -C codigo <alvo>
# ou dentro de codigo/:                    make <alvo>
# `make help` lista os alvos. Cada alvo só chama os scripts finos de
# codigo/scripts/ (os mesmos comandos estão em docs/reproducao.md, para quem não
# tem make, como no Windows).
#
# Os alvos de análise, relatório e conferência não treinam nem avaliam nada.
# Os alvos da fase científica (seção "Fase científica") rodam treinos ou
# avaliam o TEST: cada um exige AUTORIZADO=1, a autorização explícita de cada
# execução prevista no protocolo (docs/protocolo/fase-biobertpt-seed42.md,
# seção 6), e nenhum deles faz parte de outro alvo.

PY        ?= python
CODIGO    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SCRIPTS   := $(CODIGO)/scripts
EXECUCOES ?= $(CODIGO)/resultados/execucoes
NATUREZA  ?= cientifica
RELATORIOS ?= $(CODIGO)/resultados/relatorios/$(NATUREZA)
ANALISES  ?= $(CODIGO)/resultados/analises
XML       ?= $(CODIGO)/SemClinBr-xml-public-v1
TCC       ?= $(abspath $(CODIGO)/../tcc)
AFIRMACOES ?= $(TCC)/afirmacoes.json
# Fase científica: o commit exato do BioBERTpt (40 caracteres), definido no passo 0.
REVISAO   ?=
MODELO    := $(CODIGO)/modelos_pretreinados/biobertpt-all@$(REVISAO)
SUFIXO    := biobertpt_seed42

.PHONY: help testes testes-completos testes-sem-equivalencia conferir-dados conferir-dados-v1 \
        inventario-xml reconstruir-v2 conferir-v2 auditar-v2 comparar-v2 selecionar-min-freq lexico analises inventario \
        relatorios eda resultados curvas conferir-tcc \
        _autorizado _revisao pretreino verificar-inicializacao validacao-1v baseline calibrado-dev \
        restrito pair-aware estagio1 estagio2 compor-dev conferir-execucoes test-baseline \
        test-calibrado test-restrito test-pair-aware test-two-stage filtro-dev test-filtro

help:
	@echo "Testes e dados (sem treino, sem TEST científico):"
	@echo "  testes              suíte rápida (pytest -m 'not lento')"
	@echo "  testes-completos    suíte inteira, inclusive a reprodução técnica do legado (equivalencia/)"
	@echo "  testes-sem-equivalencia  unidade e integração, sem a reprodução técnica do legado (equivalencia/)"
	@echo "  conferir-dados      partições da fase (v2) x MANIFEST; conferir-dados-v1 para a v1"
	@echo "Dados v2 a partir dos XML (XML=$(XML); nada treina, TEST só estrutural):"
	@echo "  inventario-xml      inventário e proveniência dos XML"
	@echo "  reconstruir-v2      gera dados/particoes-v2 (DATA=AAAA-MM-DD; recusa sobrescrever)"
	@echo "  conferir-v2         regera a v2 numa pasta temporária e compara byte a byte"
	@echo "  auditar-v2 comparar-v2   entidades x anotações de origem; invariantes v1 x v2 -> ANALISES"
	@echo "  selecionar-min-freq refaz a seleção do min_freq da D4 (só TRAIN v2) -> ANALISES"
	@echo "Análises e relatórios (só leem partições e execuções verificadas):"
	@echo "  lexico              léxico congelado e cobertura"
	@echo "  analises            corpus, max_gap, alinhamento e marcadores -> ANALISES=$(ANALISES)"
	@echo "  inventario          execuções em EXECUCOES e a proveniência de cada uma"
	@echo "  relatorios          eda + resultados + curvas + relatorio.md -> RELATORIOS"
	@echo "  eda | resultados | curvas   cada parte em separado"
	@echo "  conferir-tcc        texto do TCC x valores gerados (pendente sem a pasta tcc/)"
	@echo "Fase científica (etapa 8; cada alvo exige AUTORIZADO=1 e, quando usa os pesos, REVISAO=<sha>):"
	@echo "  pretreino verificar-inicializacao validacao-1v baseline calibrado-dev restrito pair-aware"
	@echo "  estagio1 estagio2 compor-dev conferir-execucoes test-baseline test-calibrado test-restrito"
	@echo "  test-pair-aware test-two-stage filtro-dev test-filtro (léxico da D4: min_freq=10)"
	@echo "Variáveis: PY=$(PY) NATUREZA=$(NATUREZA) EXECUCOES=$(EXECUCOES)"

# --------------------------------------------------------------------------- #
# Testes e dados                                                              #
# --------------------------------------------------------------------------- #
testes:
	cd $(CODIGO) && $(PY) -m pytest -q -m "not lento"

testes-completos:
	cd $(CODIGO) && $(PY) -m pytest -q

testes-sem-equivalencia:
	cd $(CODIGO) && $(PY) -m pytest -q testes/unidade testes/integracao

conferir-dados:
	$(PY) $(SCRIPTS)/preparar_dados.py conferir

conferir-dados-v1:
	$(PY) $(SCRIPTS)/preparar_dados.py --pasta $(CODIGO)/dados/particoes conferir

# --------------------------------------------------------------------------- #
# Reconstrução dos dados a partir dos XML (v2-texto-fonte, regra C1 do D9)    #
# --------------------------------------------------------------------------- #
inventario-xml:
	$(PY) $(SCRIPTS)/reconstruir_dados.py --xml $(XML) inventario

reconstruir-v2:
	@if [ -z "$(DATA)" ]; then echo "Informe DATA=AAAA-MM-DD (data registrada no MANIFEST)."; exit 2; fi
	$(PY) $(SCRIPTS)/reconstruir_dados.py --xml $(XML) gerar --data $(DATA)

conferir-v2:
	$(PY) $(SCRIPTS)/reconstruir_dados.py --xml $(XML) conferir

auditar-v2:
	$(PY) $(SCRIPTS)/reconstruir_dados.py --xml $(XML) auditar --saida $(ANALISES)/auditoria_offsets.json

comparar-v2:
	$(PY) $(SCRIPTS)/reconstruir_dados.py --xml $(XML) comparar --saida $(ANALISES)/comparacao_v1_v2.json

# --------------------------------------------------------------------------- #
# Análises e relatórios                                                       #
# --------------------------------------------------------------------------- #
lexico:
	$(PY) $(SCRIPTS)/analisar.py lexico

analises:
	$(PY) $(SCRIPTS)/analisar.py corpus --saida $(ANALISES)/corpus.json
	$(PY) $(SCRIPTS)/analisar.py max-gap --saida $(ANALISES)/max_gap.json
	$(PY) $(SCRIPTS)/analisar.py alinhamento --saida $(ANALISES)/alinhamento.json
	$(PY) $(SCRIPTS)/analisar.py marcadores --saida $(ANALISES)/marcadores.json

inventario:
	$(PY) $(SCRIPTS)/gerar_tcc.py inventario --execucoes $(EXECUCOES)

relatorios:
	$(PY) $(SCRIPTS)/gerar_tcc.py tudo --execucoes $(EXECUCOES) --natureza $(NATUREZA) --destino $(RELATORIOS)

eda resultados curvas:
	$(PY) $(SCRIPTS)/gerar_tcc.py $@ --execucoes $(EXECUCOES) --natureza $(NATUREZA) --destino $(RELATORIOS)

conferir-tcc:
	$(PY) $(SCRIPTS)/conferir_tcc.py --tcc $(TCC) --relatorios $(RELATORIOS) --execucoes $(EXECUCOES) \
	    $(if $(wildcard $(AFIRMACOES)),--afirmacoes $(AFIRMACOES),)

# --------------------------------------------------------------------------- #
# Fase científica (etapa 8) — NÃO roda sem AUTORIZADO=1                       #
# --------------------------------------------------------------------------- #
_autorizado:
	@if [ "$(AUTORIZADO)" != "1" ]; then \
	    echo "Este alvo treina ou avalia o TEST: exige a autorização explícita da execução (AUTORIZADO=1)."; \
	    exit 2; fi

_revisao:
	@if [ -z "$(REVISAO)" ]; then echo "Informe REVISAO=<commit de 40 caracteres do BioBERTpt>."; exit 2; fi

pretreino: _autorizado _revisao
	$(PY) $(SCRIPTS)/pretreino.py baixar --revisao $(REVISAO)

verificar-inicializacao: _autorizado _revisao
	$(PY) $(SCRIPTS)/two_stage.py verificar-inicializacao --modelo $(MODELO)

validacao-1v: _autorizado _revisao
	$(PY) $(SCRIPTS)/treinar.py --estrategia baseline --modelo $(MODELO) --validacao-epocas 1 --saida $(EXECUCOES)
	$(PY) $(SCRIPTS)/two_stage.py treinar --estagio 1 --modelo $(MODELO) --validacao-epocas 1 --saida $(EXECUCOES)
	$(PY) $(SCRIPTS)/two_stage.py treinar --estagio 2 --modelo $(MODELO) --validacao-epocas 2 --saida $(EXECUCOES)
	$(PY) $(SCRIPTS)/two_stage.py compor-dev --estagio1 $(EXECUCOES)/twostage_e1_validacao_1ep \
	    --estagio2 $(EXECUCOES)/twostage_e2_validacao_2ep --saida $(EXECUCOES)
	$(PY) $(SCRIPTS)/baseline_calibrado.py calibrar-dev --base $(EXECUCOES)/baseline_validacao_1ep --saida $(EXECUCOES)

baseline: _autorizado _revisao
	$(PY) $(SCRIPTS)/treinar.py --estrategia baseline --modelo $(MODELO) --saida $(EXECUCOES) --checkpoint-a-cada 500

calibrado-dev: _autorizado
	$(PY) $(SCRIPTS)/baseline_calibrado.py calibrar-dev --base $(EXECUCOES)/baseline_$(SUFIXO) --saida $(EXECUCOES)

restrito: _autorizado _revisao
	$(PY) $(SCRIPTS)/treinar.py --estrategia restrito --modelo $(MODELO) --saida $(EXECUCOES)

pair-aware: _autorizado _revisao
	$(PY) $(SCRIPTS)/treinar.py --estrategia pair_aware --modelo $(MODELO) --saida $(EXECUCOES)

estagio1: _autorizado _revisao
	$(PY) $(SCRIPTS)/two_stage.py treinar --estagio 1 --modelo $(MODELO) --saida $(EXECUCOES) --checkpoint-a-cada 500

estagio2: _autorizado _revisao
	$(PY) $(SCRIPTS)/two_stage.py treinar --estagio 2 --modelo $(MODELO) --saida $(EXECUCOES)

compor-dev: _autorizado
	$(PY) $(SCRIPTS)/two_stage.py compor-dev --estagio1 $(EXECUCOES)/twostage_e1_$(SUFIXO) \
	    --estagio2 $(EXECUCOES)/twostage_e2_$(SUFIXO) --saida $(EXECUCOES)

conferir-execucoes:
	$(PY) $(SCRIPTS)/pretreino.py conferir --execucoes $(EXECUCOES)/baseline_$(SUFIXO) \
	    $(EXECUCOES)/baseline_calibrado_$(SUFIXO) $(EXECUCOES)/restrito_$(SUFIXO) \
	    $(EXECUCOES)/pairaware_$(SUFIXO) $(EXECUCOES)/twostage_e1_$(SUFIXO) \
	    $(EXECUCOES)/twostage_e2_$(SUFIXO) $(EXECUCOES)/twostage_$(SUFIXO) $(EXECUCOES)/filtro_$(SUFIXO)

test-baseline: _autorizado
	$(PY) $(SCRIPTS)/avaliar_test.py --execucao $(EXECUCOES)/baseline_$(SUFIXO)

test-calibrado: _autorizado
	$(PY) $(SCRIPTS)/baseline_calibrado.py aplicar-test --execucao $(EXECUCOES)/baseline_calibrado_$(SUFIXO)

test-restrito: _autorizado
	$(PY) $(SCRIPTS)/avaliar_test.py --execucao $(EXECUCOES)/restrito_$(SUFIXO)

test-pair-aware: _autorizado
	$(PY) $(SCRIPTS)/avaliar_test.py --execucao $(EXECUCOES)/pairaware_$(SUFIXO)

test-two-stage: _autorizado
	$(PY) $(SCRIPTS)/two_stage.py avaliar-test --execucao $(EXECUCOES)/twostage_$(SUFIXO)

# Filtro léxico (modo da fase): léxico congelado, sem arquivo de calibração. O
# min_freq=10 do léxico congelado foi escolhido na pendência D4, só com o TRAIN
# da v2 (codigo/docs/entregas/d4-min-freq.md).
selecionar-min-freq:
	$(PY) $(SCRIPTS)/selecionar_min_freq.py --saida $(ANALISES)/selecao_min_freq.json

filtro-dev: _autorizado
	$(PY) $(SCRIPTS)/filtro_pistas.py --base-execucao $(EXECUCOES)/baseline_$(SUFIXO) --particao dev --saida $(EXECUCOES)

test-filtro: _autorizado
	$(PY) $(SCRIPTS)/filtro_pistas.py --base-execucao $(EXECUCOES)/baseline_$(SUFIXO) --particao test --saida $(EXECUCOES)
