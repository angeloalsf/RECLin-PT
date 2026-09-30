# `results/finetuning_restrito/` · resultados da frente experimental

Único destino de gravação da frente `src/finetuning_restrito/`. Nada daqui entra no
Cap. 6, na fase 2 nem em `scripts/check_tcc_numbers.py`, e nenhum gerador de
artefato do TCC lê esta pasta.

## Estado em 26/09/2026

| tarefa | estado | arquivo |
|---|---|---|
| 1. espaço restrito + remapeamento | implementada e verificada em CPU (7 checagens) | `tarefa1_verificacao_cpu.json` |
| 2. sanidade (1 encoder × 42/43) | **pendente de GPU** (Colab T4, ~1 h) | `restrito_<encoder>_seed{42,43}.*`, `CRITERIO_PARADA_tarefa2_<encoder>.{json,md}` |
| 3. escala (2 × 10 sementes) | bloqueada até a Tarefa 2 passar | `restrito_<encoder>_seed<N>.*` |
| 4. avaliação final | bloqueada até a Tarefa 3 | `AVALIACAO_final.{json,md}`, `significancia/` |

Ainda não há nenhum número de modelo treinado nesta pasta.

## O que cada execução grava

| arquivo | conteúdo |
|---|---|
| `restrito_<enc>_seed<N>.json` | métricas no TEST remapeado, `dev_history`, bloco `restricted_space` (léxico, tamanhos por split, tetos de recall, pesos de classe efetivos), `instrumentation`, `environment` no formato dos baselines |
| `.preds.json` | TEST remapeado (19.210 pares), contrato de `src/significance.py`, com `restricted_indices` |
| `.dev_preds.json` | DEV da melhor época, remapeado (19.064 pares) |
| `.test_evals.jsonl` | trilha de multiplicidade |
| `.train_log.txt` | log completo, com os avisos do modo determinístico |

Só o F1 de `negation_of` é comparável aos baselines. Ver a seção "Decisões" de
`src/finetuning_restrito/README.md`.
