# Etapa 5 — execução das recomendações G1, G2, G3, K1, K2, M1, M3, M4 e M5

16/11/2026. Só TRAIN e DEV. **O TEST não foi lido, processado nem avaliado.**
Nenhum modelo real foi treinado; nenhum resultado foi simulado. Nenhum comando
Git foi executado na pasta de trabalho.

## 1. Estado de cada recomendação

| Rec. | O que era | Estado | Depende de |
| --- | --- | --- | --- |
| K1 | teste de unidade falhando (`referencia`) | **feito e verificado** | — |
| K2 | `import torch` adiado na Pair-Aware | **feito e verificado** | — |
| G2 | trava técnica do TEST | **feito e verificado** (unidade); integração ajustada, **não executada** | — |
| G3 | registrar a execução sem pesos; bootstrap por documento | **feito** | — |
| M4 | controles para BioBERTpt-clin e BERTimbau-base | **feito** (DEV) | — |
| M5 | `associated_with` sem direção | **feito** (DEV) | — |
| M1 | adjudicação cega dos FP | **pacote pronto**; a adjudicação é humana | autor |
| M3 | ablação do pooler | **código, testes e notebooks prontos**; execução em GPU pendente | autor (Colab) |
| G1 | repositórios privados | **README, `.gitignore` e guia alinhados**; visibilidade não alterada | autor |

## 2. O que foi implementado

**K1.** `test_entradas_e_legado.py` passou a procurar o componente de caminho
`/ "referencia"`, não a palavra: a chave `"referencia"` do JSON de
`sensibilidade_semente.py` é legítima e não foi renomeada (preserva o esquema
gravado).

**K2.** `pair_aware/__init__.py` e `pair_aware_mlp/__init__.py` importam a
cabeça só quando usada (`__getattr__` do módulo). `cabeca.py` intacto.

**G2.** `reclin/execucao/congelamento.py` e `scripts/congelar.py`; a trava é a
primeira instrução das três funções que avaliam o TEST; opções `--congelamento`
e `--liberacao` nos três scripts. O registro guarda os hashes de `config.json`,
`predicoes_dev.json`, `limiar.json` e dos pesos do melhor modelo (calculado da
pasta `checkpoints/melhor_modelo/`). Detalhes e procedimento em
`docs/protocolo/congelamento-e-test-final.md`.

**G3.** A auditoria separa a execução sem pesos (`chave_estrategia`), compara-a
com a ponderada, o baseline, o Two-Stage e o baseline calibrado, por candidato e
por documento, e mantém byte a byte os registros já gravados
(`preservar_registrados`). Novo diagnóstico de precisão e recall pareados
(`diagnosticos/comparar_precisao_recall.py`).

**M4.** Quatro execuções derivadas no DEV (baseline calibrado e filtro léxico
para os dois encoders), geradas com o código do `main` sem alterações.

**M5.** `reclin/analises/associated_sem_direcao.py` e
`diagnosticos/avaliar_assoc_sem_direcao.py`.

**M1.** A planilha cega existente já cobre os dois pontos de operação (os 34 FP
da execução sem pesos estão entre os 49 da ponderada). Acrescentados:
`adjudicacao_lotes.py` (ordem de trabalho em dois lotes que não revela o grupo
dos casos) e `adjudicacao_ponto_de_operacao.py` (projeção das decisões).

**M3.** `reclin/estrategias/ablacao_pooler/` (`cls_direto` e
`pair_aware_pooler`), registro nos scripts e na auditoria, dois notebooks em
`notebooks/ablacao/`. Ver `docs/estrategias/ablacao-pooler.md`.

**G1.** `README.md`, `.gitignore` (só comentários) e
`docs/repositorio-privado.md`.

## 3. Resultados (DEV, 19.064 candidatos, 98 documentos, semente 42)

### Pair-Aware Linear sem pesos (A) − comparada (B)

| B | Δ F1 neg | IC95% doc | Δ macro-F1 | IC95% doc |
| --- | --- | --- | --- | --- |
| Pair-Aware ponderada | +0,002 | [−0,025; +0,028] | +0,011 | [−0,005; +0,027] |
| Baseline | +0,030 | [−0,009; +0,070] | +0,033 | [+0,012; +0,054]* |
| Two-Stage | +0,001 | [−0,025; +0,024] | +0,027 | [+0,009; +0,044]* |
| Baseline calibrado | +0,022 | [−0,006; +0,052] | +0,031 | [+0,012; +0,049]* |

Sem pesos − ponderada, por documento: precisão de `negation_of` +0,052
[+0,026; +0,079]*; recall −0,067 [−0,113; −0,029]*. A retirada dos pesos troca
recall por precisão de forma detectável e não muda o F1.

### Controles nos três encoders

| Encoder | Sistema | P neg | R neg | F1 neg | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| BioBERTpt-all | Pair-Aware Linear | 0,742 | 0,940 | 0,829 | 0,741 |
| | Baseline calibrado (t=0,90) | 0,723 | 0,920 | 0,809 | 0,721 |
| | Filtro léxico | 0,750 | 0,900 | 0,818 | 0,724 |
| BioBERTpt-clin | Pair-Aware Linear | 0,732 | 0,947 | 0,826 | 0,746 |
| | Baseline calibrado (t=0,95) | 0,753 | 0,913 | 0,825 | 0,734 |
| | Filtro léxico | 0,753 | 0,893 | 0,817 | 0,732 |
| BERTimbau-base | Pair-Aware Linear | 0,713 | 0,960 | 0,818 | 0,736 |
| | Baseline calibrado (t=0,95) | 0,751 | 0,907 | 0,822 | 0,730 |
| | Filtro léxico | 0,763 | 0,900 | 0,826 | 0,731 |

No F1 de `negation_of`, a Pair-Aware Linear não se distingue de nenhum controle
em nenhum encoder. No macro-F1, a vantagem sobre os controles é detectada (por
documento) com os dois BioBERTpt e não com o BERTimbau-base. Os intervalos estão
na Tabela `dev_controles_encoders` do TCC e em `comparacoes_dev.json`.

### `associated_with` sem direção

O F1 sobe entre 0,09 e 0,16 em todos os sistemas (Pair-Aware Linear,
BioBERTpt-all: 0,436 → 0,592). O maior F1 sem direção é 0,608. A direção explica
parte do erro, não a maior parte.

## 4. Testes e verificações

| Verificação | Resultado |
| --- | --- |
| Suíte de unidade, na nuvem, com torch 2.11.0 e transformers 5.16.1, **sem** as partições v1, sem `test.jsonl` e sem os arquivos de referência do corpus | **733 passaram, 15 falharam, 18 erros, 6 ignorados**. Todas as falhas e erros são arquivos de dados ausentes na nuvem (partições v1, `test.jsonl`, FONTE/ANOMALIAS, referências). Antes da etapa, no mesmo ambiente: 699 passaram, com as mesmas 15 falhas e 18 erros mais a falha do K1 |
| Testes novos (congelamento, ablação, lotes, projeção, sem direção, precisão/recall, auditoria) | passaram |
| Treino de ponta a ponta das duas variantes da ablação com o modelo minúsculo, em subconjunto técnico de treino e DEV | terminou sem erro; saídas descartadas |
| Registros antigos da auditoria | idênticos aos originais (conferido entrada a entrada) |
| `congelar.py` | ensaiado com saída temporária, descartada; `resultados/congelamento/` não existe |
| Prévia do TCC | compila na nuvem sem erro e sem referência indefinida (105 páginas); o PDF oficial é o do Docker do autor |
| Suíte de integração e de equivalência | **não executadas** (a de equivalência processa o TEST v1; a de integração treina sobre as partições completas) |
| Suíte completa na máquina do autor | **pendente** |
| Notebooks da ablação | sintaxe e conteúdo conferidos por teste; **não executados** no Colab |

## 5. Arquivos

**Criados (código e testes):** `reclin/execucao/congelamento.py`,
`scripts/congelar.py`, `reclin/estrategias/ablacao_pooler/{__init__,cabeca}.py`,
`reclin/analises/{associated_sem_direcao,adjudicacao_lotes,adjudicacao_ponto_de_operacao}.py`,
`diagnosticos/{avaliar_assoc_sem_direcao,comparar_precisao_recall}.py`,
`notebooks/ablacao/reclin_{cls_direto,pair_aware_pooler}.ipynb`,
`testes/unidade/test_{congelamento,ablacao_pooler,assoc_sem_direcao,adjudicacao_lotes,adjudicacao_ponto_de_operacao,comparar_precisao_recall}.py`.

**Criados (resultados e análise):**
`resultados/diagnosticos/{associated_with_sem_direcao,precisao_recall_sem_pesos_x_ponderada}.json`,
`resultados/analises/adjudicacao_negation_of/pairaware_biobertpt_seed42/{cega/ordem_de_trabalho.csv,restrito/lotes.json}`,
`tcc/analise/execucoes_derivadas_dev/{baseline_calibrado,filtro}_{BioBERTpt-clin,BERTimbau}/`,
`tcc/analise/gerar_tabelas_etapa5.py`,
`tcc/tcc/src/tabelas/{dev_sem_pesos_comparacoes,dev_controles_encoders,dev_assoc_sem_direcao}.tex`,
`tcc/tcc/revisao/TCC_etapa5_{previa,diff}.pdf`.

**Criados (documentação):** este arquivo,
`docs/protocolo/{congelamento-e-test-final,registro-e4-experimentos-pos-fase}.md`,
`docs/estrategias/ablacao-pooler.md`, `docs/repositorio-privado.md`.

**Modificados:** `.gitignore`, `README.md`, `codigo/README.md`,
`diagnosticos/auditar_execucoes.py`, `reclin/analises/auditoria_execucoes.py`,
`reclin/estrategias/{__init__,baseline_calibrado}.py`,
`reclin/estrategias/pair_aware/__init__.py`,
`reclin/estrategias/pair_aware_mlp/__init__.py`,
`reclin/estrategias/two_stage/composicao.py`, `reclin/treino/classificador.py`,
`scripts/{avaliar_test,baseline_calibrado,treinar,two_stage}.py`,
`testes/integracao/test_estrategias_treinadas.py`,
`testes/unidade/test_{auditoria_execucoes,entradas_e_legado,pacote}.py`,
`resultados/auditoria_execucoes/*.json` (cinco arquivos; só acréscimos),
`docs/analises/adjudicacao_negation_of.md`, `docs/entregas/README.md`,
`docs/estrategias/README.md`,
`tcc/tcc/src/textuais/{experimentos_e_resultados,discussao,conclusao,metodologia}.tex`,
`tcc/tcc/src/tabelas/dev_sem_pesos.tex`.

**Removidos:** nenhum.

## 6. Critérios de aceitação

| Critério | Situação |
| --- | --- |
| TEST não lido nem avaliado | atendido |
| Resultados históricos preservados | atendido (registros antigos idênticos; nenhuma execução alterada) |
| Sem resultados simulados | atendido (a ablação não tem números) |
| Trava do TEST com teste automatizado | atendido na unidade; integração pendente de execução |
| TCC sincronizado com código e resultados | atendido para G3, M4, M5 e G2; M3 e M1 sem resultado a relatar |
| Configuração definitiva recomendada e congelamento preparado | recomendação **preliminar**; congelamento preparado e **não executado** |
| Suíte completa verde na máquina do autor | pendente |

## 7. Achados, limitações e riscos

1. **Caminhos que leem o TEST fora da trava.** `scripts/regra_pura.py` avalia DEV
   e TEST por padrão; a suíte de equivalência o roda sobre o TEST v1; há leituras
   estruturais do `test.jsonl` (hash, estatísticas, `test_dados_v2`). Lista em
   `congelamento-e-test-final.md`. A recomendação K4 da Etapa 4 ("rodar a suíte
   completa") implica essas leituras.
2. **Teste de integração alterado sem execução.** `test_comandos_de_ponta_a_ponta`
   avaliava o TEST científico inteiro com o modelo minúsculo; agora espera a
   recusa. `test_avaliacao_e_comparacao_das_etapas_anteriores` passou a comparar
   predições do DEV. Precisam ser rodados na máquina do autor.
3. **Gerador de tabelas da Etapa 3 divergente.** `gerar_tabelas_etapa3.py` não
   reproduz mais as tabelas de `tcc/src/tabelas/` (elas receberam ajustes de
   redação à mão, por exemplo "Pair-Aware Linear") e, rodado hoje, acrescentaria
   os controles novos a `dev_sistemas`. Não foi rodado; as tabelas novas vêm de
   `gerar_tabelas_etapa5.py`. A grafia "Linear"/"linear" está inconsistente
   entre tabelas antigas.
4. **Congelamento exige os pesos.** As pastas de resultados versionadas não têm
   `checkpoints/melhor_modelo/`; o congelamento e a avaliação precisam ser feitos
   onde os pesos estão (Drive/Colab ou pasta restaurada).
5. **Nova identidade de código.** O código desta etapa tem identidade diferente
   da `0526d9123ecc`. Execuções novas (ablação) registram a nova; derivar
   controles de bases antigas com o código novo é recusado pelos scripts (por
   isso os controles foram derivados com o `main` intacto).
6. **Exceção de arquitetura.** `ablacao_pooler` importa três funções de
   `pair_aware.cabeca`; a exceção foi registrada em `test_pacote.py`.
7. **Limiar na borda.** t = 0,95 nos baselines calibrados do BioBERTpt-clin e do
   BERTimbau-base é o extremo da grade.
8. **Uma semente.** Tudo é descritivo; a recomendação M2 (sementes) não fazia
   parte desta etapa.
9. **Repositórios ainda públicos**, com texto do corpus, inclusive o pacote de
   adjudicação.

## 8. Decisões que dependem do autor

1. **Emenda E3**: continua sendo proposta. Decidir E3-D3 e E3-D6 e aprová-la
   é pré-requisito do congelamento (sugestão em `congelamento-e-test-final.md`).
   Aprovar (ou não) a regra de decisão ligada à adjudicação e a recomendação
   preliminar.
2. Fazer commit e push do código antes de rodar os notebooks da ablação.
3. Adjudicar (lote 1, depois lote 2), congelar as decisões e rodar a projeção.
4. Tornar os repositórios privados.
5. Decidir sobre a trava de `regra_pura.py` no TEST e sobre a suíte de
   equivalência que processa o TEST v1.
6. Autorizar explicitamente o congelamento e, depois, a avaliação única do TEST.
