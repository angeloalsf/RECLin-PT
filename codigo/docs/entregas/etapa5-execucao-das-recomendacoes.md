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
| M3 | ablação do pooler | **executada (Colab) e analisada no DEV em 22/11** (ver adendo) | concluída |
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

## Adendo de 22/11/2026 — M3 executado e curva de limiar

**M3.** O autor executou os dois notebooks; a análise usou clone só leitura de
`resultados-colab@0c08b746385b` (apenas `dev.jsonl` e `MANIFEST.json` da partição; TEST não
lido), apagado ao final. `auditar_execucoes.py` rodou sobre os registros existentes: os 17
sistemas, 35 comparações, 18 registros de `associated_with`, 3 de τ e 23 execuções antigos
foram conferidos e mantidos sem alteração; acrescentados 2 sistemas e 4 comparações. Única
mudança em campo existente: `proveniencia.execucoes` passou a registrar a origem
(`resultados-colab@0c08b746385b`).

| Efeito (A − B), BioBERTpt-all | Δ F1 neg [IC doc] | Δ F1 assoc [IC doc] | Δ macro-F1 [IC doc] |
| --- | --- | --- | --- |
| ler marcadores, sem pooler (`pair_aware` − `cls_direto`) | +0,009 [−0,017; +0,036] | +0,021 [+0,006; +0,035]* | +0,012 [+0,001; +0,022]* |
| ler marcadores, com pooler (`pair_aware_pooler` − `baseline`) | +0,035 [+0,007; +0,064]* | +0,046 [+0,028; +0,063]* | +0,031 [+0,020; +0,043]* |
| retirar pooler, só [CLS] (`cls_direto` − `baseline`) | +0,018 [−0,012; +0,050] | +0,007 [−0,008; +0,022] | +0,010 [−0,0001; +0,021] (detectada só por candidato) |
| retirar pooler, com marcadores (`pair_aware` − `pair_aware_pooler`) | −0,007 [−0,026; +0,011] | −0,017 [−0,030; −0,005]* | −0,009 [−0,017; −0,001]* |

\* IC95% por documento exclui o zero. Métricas das quatro células (P/R/F1 neg, F1 assoc, macro-F1):
baseline 0,696/0,947/0,802, 0,408, 0,719; `cls_direto` 0,735/0,927/0,820, 0,415, 0,729;
`pair_aware_pooler` 0,754/0,940/0,837, 0,454, 0,750; Pair-Aware Linear 0,742/0,940/0,829, 0,436, 0,741.

Leitura: o ganho da Pair-Aware Linear sobre o baseline vem da leitura dos marcadores; retirar o
pooler não traz ganho detectado e, com marcadores, reduz macro-F1 e F1 de `associated_with`.
`pair_aware_pooler` é exploratória, não elegível ao TEST e não entra na seleção (plano fixado
antes, `docs/protocolo/registro-e4-experimentos-pos-fase.md`, §4). Uma semente, um encoder.

Não altera a recomendação preliminar: a ablação não participa da seleção. Se o autor quiser
adotar a cabeça com pooler, é decisão nova, que exige emenda antes do congelamento e é
pós-hoc (uma semente; diferença para a Pair-Aware Linear não detectada em F1 neg).

**Curva de limiar (17/11).** `diagnosticos/curva_limiar_negation_of.py` →
`resultados/diagnosticos/curva_limiar_negation_of.json`: P/R/F1 de `negation_of` por limiar
de P(neg) (0,50–0,99) para as execuções sem pesos e ponderada, com IC por documento das
diferenças ao argmax. Sem pesos, t = 0,80: P 0,845, R 0,800, F1 0,822 (ΔP +0,051
[+0,016; +0,094]; ΔR −0,073 [−0,117; −0,034]; ΔF1 −0,010 [−0,041; +0,020]). É um candidato a
regra de decisão; exige emenda com grade e critério declarados. Sem teste de unidade.

**TCC.** Nova seção "Resultado exploratório: ablação do pooler" (cap. 6) com as tabelas
`dev_ablacao_pooler_sistemas` e `dev_ablacao_pooler`; ajustes em Proposta (5.3), Discussão
(7.1 e Limitações) e Conclusão (principais resultados, limitações, trabalhos futuros).
Prévia e diff: `tcc/tcc/revisao/TCC_etapa5_ablacao_{previa,diff}.pdf`. As quatro tabelas da
Etapa 5 regeradas saíram idênticas.

**Arquivos (22/11).** Modificados: `resultados/auditoria_execucoes/*.json` (5, só
acréscimos), `tcc/analise/gerar_tabelas_etapa5.py`, TCC `textuais/{experimentos_e_resultados,
proposta_prototipo,discussao,conclusao}.tex`, `docs/estrategias/ablacao-pooler.md`,
`docs/protocolo/registro-e4-experimentos-pos-fase.md`, este relatório. Criados:
`tcc/tcc/src/tabelas/dev_ablacao_pooler{,_sistemas}.tex`, os dois PDFs. Nenhum removido.
Nenhum comando Git na pasta de trabalho.

## Adendo de 22/11/2026 (tarde) — configuração final decidida e preparada para o congelamento

**Decisão do autor:** configuração final = Pair-Aware Linear sem pesos de classe
(`pair_aware__a3ac671a77e2__t2`) com a regra P(`negation_of`) ≥ 0,80. Sem treino
novo. Pedido: fazer tudo, exceto a avaliação do TEST, que fica pausada.

**Registro.** `docs/protocolo/emenda-configuracao-final.md` (E3-D3, E3-D6 como
alternativa nova pós-DEV, evidência, riscos; SHA-256 do texto
`b866f6fb32b99625…` (versão de 22/11 com a §7 do experimento de 5 épocas; recalcule se editar), a conferir e registrar pelo autor na seção 8 do protocolo,
que não foi alterada). `emenda-e3-pair-aware.md` (tabela de decisões) e
`congelamento-e-test-final.md` (atualização) apontam para ela.

**Código.**
- `reclin/avaliacao/limiar.py`: critério opcional `PrecisaoComPisoDeRecall`
  (maior precisão de `negation_of` com recall ≥ piso; desempates F1, macro-F1,
  menor limiar). Sem o critério, configuração e hashes idênticos aos anteriores.
- `reclin/estrategias/pair_aware_limiar.py` + `scripts/pair_aware_limiar.py`:
  a regra da configuração final (mesma regra do baseline calibrado, base
  Pair-Aware, piso 4/5), `calibrar-dev` e `aplicar-test`.
- `reclin/treino/reempacotamento.py` + `scripts/reempacotar.py`: a base foi
  treinada com o código `0526d912…`; o código atual só a aceita depois de
  reproduzir o DEV (mesma classe nos 19.064 pares, probabilidades com diferença
  ≤ 0,001). Cria uma execução nova com cópia conferida dos pesos; nada é gravado
  se divergir; a base fica intacta.
- `reclin/execucao/congelamento.py`: configuração derivada congelada com as
  impressões da base (inclusive pesos); a trava aceita a base (primeiro) e a
  derivada (depois), uma vez cada, sob a mesma liberação; trilha com o papel de
  cada passo; do `config.json` entram a configuração e a identidade do código.
- `reclin/experimento/versionamento.py`: perfis `reempacotada` e `derivada`.
- `reclin/relatorios/tabelas.py`: lê o critério gravado no `limiar.json`.
- `notebooks/final/reclin_configuracao_final.ipynb`: Colab, só DEV — confere
  os SHA-256 dos pesos no Drive, reempacota, congela t (exige t = 0,80 com
  120/22/30), grava o registro de congelamento no Drive e, com confirmação,
  publica os arquivos do DEV. Não chama avaliação do TEST.
- `diagnosticos/avaliar_configuracao_final_dev.py` →
  `resultados/diagnosticos/configuracao_final_dev.json` (curva, métricas e
  comparações da configuração final no DEV; clone só leitura de
  `resultados-colab`, só DEV, apagado).

**Resultado no DEV (verificado).** t = 0,80; P 0,845, R 0,800, F1 0,822
(120/22/30), F1 assoc 0,448, macro-F1 0,748. Contra a referência anterior
(ponderada, argmax), por documento: precisão +0,103 [+0,059; +0,151], recall
−0,140 [−0,212; −0,076], F1 neg −0,007 [−0,053; +0,035], macro +0,008
[−0,013; +0,028].

**TCC.** Resumo/Abstract, Proposta (nova subseção 5.3.1 Regra de decisão),
Metodologia (perda, critério de t, trava em dois passos), Resultados (nova seção
6.9 com as tabelas `dev_limiar_negation_of` e `dev_configuracao_final`),
Discussão (justificativa e fragilidades), Conclusão. Prévia e diff:
`tcc/tcc/revisao/TCC_configuracao_final_{previa,diff}.pdf`. A afirmação antiga
de que, sem ponderação, o modelo preveria `no_relation` para quase tudo foi
corrigida (contradizia o experimento sem pesos). Slides não atualizados.

**Testes (nuvem, torch 2.11.0, transformers 5.16.1).** Suíte de unidade: 707
passaram, 48 falharam, 21 erros; as falhas e erros são os mesmos do `main`
antes das mudanças (arquivos de dados e da raiz ausentes na nuvem), mais o
`--help` dos dois scripts novos, que falha pela mesma razão de todos os
scripts (pacote não instalado) e passa com `PYTHONPATH` — com ele,
`test_pacote` só falha nos arquivos ausentes. Novos testes passaram:
`test_limiar` (+3), `test_congelamento` (+3), `test_configuracao_final` (5, de
ponta a ponta com o modelo minúsculo: treino sem pesos → reempacotamento →
limiar → congelamento → trava em dois passos com as partições sintéticas no
papel das oficiais). **Não executados:** integração, equivalência e a suíte na
máquina do autor; o notebook no Colab.

**Arquivos (22/11, tarde).** Criados: os 4 módulos/scripts acima, o notebook,
`testes/unidade/test_configuracao_final.py`, o diagnóstico e seu JSON,
`docs/protocolo/emenda-configuracao-final.md`, as tabelas
`dev_limiar_negation_of.tex` e `dev_configuracao_final.tex`, os dois PDFs.
Modificados: `limiar.py`, `congelamento.py`, `versionamento.py`, `tabelas.py`,
`estrategias/__init__.py`, `scripts/{avaliar_test,congelar}.py`,
`testes/unidade/test_{limiar,congelamento}.py`,
`tcc/analise/gerar_tabelas_etapa5.py`, 5 `.tex` textuais e o resumo,
`docs/protocolo/{emenda-e3-pair-aware,congelamento-e-test-final}.md`, este
relatório. Nenhum removido. Nenhum comando Git na pasta de trabalho.

**Pendências do autor.** Conferir a emenda e registrar o SHA; rodar a suíte;
commit e push; rodar o notebook da configuração final; guardar o
`configuracao_final.json` (Drive) em `codigo/resultados/congelamento/`; depois,
autorizar explicitamente a avaliação única do TEST.

**Experimento de 5 épocas (22/11, pré-registrado).** O autor aprovou, antes do
treino, repetir a execução sem pesos com `epochs = 5` e a regra: 5 épocas só
substitui 3 se a precisão de `negation_of` (cada modelo no seu t, pelo mesmo
critério) for maior com o IC95% por documento acima de zero. Registrado na §7
da emenda. Notebook: `notebooks/teste/reclin_pair_aware_sem_pesos_5epocas.ipynb`
(cópia do notebook sem pesos com `epochs` 5; a seção 8.3 aplica a regra e
imprime a decisão; não lê o TEST nem congela). Teste novo em
`test_configuracao_final.py` (conteúdo do notebook) conferido com as mesmas
asserções rodadas à parte; a suíte não foi rodada de novo. O congelamento fica
para depois da decisão.
