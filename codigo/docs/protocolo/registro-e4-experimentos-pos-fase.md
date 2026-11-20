# Registro E4 — experimentos e análises posteriores à fase (Etapa 5)

16/11/2026. Registro de natureza e rastreabilidade; **não altera** o protocolo
da fase, a emenda E3 nem os resultados gravados. Tudo aqui é do DEV, de uma
semente, e exploratório. Os itens marcados "proposta" dependem de aprovação do
autor.

## 1. Pair-Aware Linear sem pesos de classe (G3)

| Campo | Valor |
| --- | --- |
| Execução | `pair_aware__a3ac671a77e2__t2` (branch `resultados-colab`) |
| Diferença para o protocolo | `class_weight="none"`; o resto igual à `pairaware_biobertpt_seed42` |
| Natureza | exploratória, posterior ao protocolo; sem nome oficial; TEST não avaliado |
| Na auditoria | sistema `pair_aware_sem_pesos\|BioBERTpt-all` (`chave_estrategia` separa pela perda) |
| Arquivos | `resultados/auditoria_execucoes/{inventario,metricas_dev,comparacoes_dev,associated_with_dev}.json`; `resultados/diagnosticos/precisao_recall_sem_pesos_x_ponderada.json` |

Os registros já gravados na auditoria foram mantidos byte a byte
(`preservar_registrados` recalcula, confere com tolerância 1e-9 e só acrescenta
os novos).

## 2. Controles nos outros dois encoders (M4)

Baseline calibrado e filtro léxico derivados das predições já gravadas dos
baselines do BioBERTpt-clin e do BERTimbau-base, sem treino, com o código do
`main` sem alterações (identidade `0526d9123ecc`, a das bases). Pastas em
`tcc/analise/execucoes_derivadas_dev/`. O limiar do baseline calibrado ficou em
0,95 nos dois, o extremo da grade. Fora do protocolo (que previa os controles só
para o BioBERTpt-all).

## 3. `associated_with` sem direção (M5)

Análise secundária (`reclin/analises/associated_sem_direcao.py`,
`diagnosticos/avaliar_assoc_sem_direcao.py`,
`resultados/diagnosticos/associated_with_sem_direcao.json`). Não muda a tarefa
nem as métricas principais.

## 4. Ablação do pooler (M3) — plano de análise fixado antes da execução

Estratégias `cls_direto` e `pair_aware_pooler`
(`docs/estrategias/ablacao-pooler.md`), BioBERTpt-all, semente 42, mesmos
hiperparâmetros. **Ainda não executadas.** Análise prevista, no DEV, com
`diagnosticos/auditar_execucoes.py` (comparações já declaradas):

| Efeito | Comparação |
| --- | --- |
| marcadores, sem pooler | `pair_aware` − `cls_direto` |
| marcadores, com pooler | `pair_aware_pooler` − `baseline` |
| retirar o pooler, sem marcadores | `cls_direto` − `baseline` |
| retirar o pooler, com marcadores | `pair_aware` − `pair_aware_pooler` |

Métricas: F1 de `negation_of`, F1 de `associated_with`, macro-F1; bootstrap
pareado por candidato e por documento; "detectada" = IC95% exclui zero. Leitura
descritiva. As variantes não são elegíveis ao TEST e não entram na escolha da
configuração final.

## 5. Adjudicação cega (M1)

Planilha existente (107 casos: 49 FP, 49 controles VP, 9 FN), inalterada.
Acrescentados a ordem de trabalho em dois lotes e a projeção das decisões sobre
outros pontos de operação (`docs/analises/adjudicacao_negation_of.md`, seção
final). A regra de decisão ligada à adjudicação está em
`congelamento-e-test-final.md` (**proposta**).

## 6. Trava do TEST (G2)

`docs/protocolo/congelamento-e-test-final.md`.
