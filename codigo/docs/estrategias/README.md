# Estratégias

Resumo das estratégias implementadas no código novo, a partir dos módulos de
`reclin/estrategias/` (cada módulo tem a descrição completa no próprio
docstring). Os critérios de seleção e de parada da fase atual estão no
protocolo pré-registrado (`docs/protocolo/fase-biobertpt-seed42.md`, seção 3).

O plano de migração previa trazer para esta pasta os READMEs das frentes do
repositório original. Pela decisão da etapa 7 (o projeto novo não incorpora
artefatos do legado), eles não foram copiados; esta página descreve só o
código novo.

| Estratégia | Módulo | Comando | Execução (fase) | Dados de cada decisão | Saída |
| --- | --- | --- | --- | --- | --- |
| Baseline | `estrategias/baseline.py` | `treinar.py --estrategia baseline` | `baseline_biobertpt_seed42` | treino: TRAIN; época: maior macro-F1 do DEV (primeiro máximo); 3 épocas, sem parada antecipada | DEV; TEST em `avaliar_test.py` |
| Baseline calibrado | `estrategias/baseline_calibrado.py` | `baseline_calibrado.py calibrar-dev` / `aplicar-test` | `baseline_calibrado_biobertpt_seed42` | t na grade 0,05–0,95 pelo F1 de `negation_of` do DEV (desempates do protocolo), sobre as probabilidades do baseline | DEV; TEST uma vez, com `limiar.json` conferido |
| Filtro léxico | `estrategias/filtro_pistas.py` | `filtro_pistas.py --base-execucao PASTA --particao dev` e, no passo 7, `--particao test` | `filtro_biobertpt_seed42` | rebaixa a `no_relation` o `negation_of` cujo e1 não é pista do léxico congelado (induzido do TRAIN; `min_freq=10`, D4); sem arquivo de calibração | DEV; TEST uma vez, depois do TEST da base, com trilha |
| Regra pura | `estrategias/regra_pura.py` | `regra_pura.py` | `regra_pura` | regra R1–R4 com o léxico congelado | fora da fase atual |
| Restrito | `estrategias/restrito.py` | `treinar.py --estrategia restrito` | `restrito_biobertpt_seed42` | treino: pares do TRAIN com e1 no léxico; pesos `balanced` do TRAIN restrito; época: maior macro-F1 do DEV remapeado ao conjunto completo (fora do espaço → `no_relation`); 2 épocas (teto da fase atual, D5), sem parada antecipada | DEV e TEST no conjunto completo; só o F1 de `negation_of` é comparável |
| Pair-Aware | `estrategias/pair_aware/` | `treinar.py --estrategia pair_aware` | `pairaware_biobertpt_seed42` | todos os pares candidatos, sem filtro lexical nem léxico (D10), com a cabeça linear `[h_cls ; h_E1 ; h_E2] → Dropout → Linear` (sem MLP); época: maior macro-F1 do DEV completo (primeiro máximo); 3 épocas, sem parada antecipada | DEV; TEST em `avaliar_test.py` |
| Pair-Aware com MLP | `estrategias/pair_aware_mlp/` | `treinar.py --estrategia pair_aware_mlp` | `pairawaremlp_biobertpt_seed42` (sem nome oficial; **fora do protocolo**, exploratória) | todos os pares candidatos, sem filtro lexical nem léxico, cabeça `[h_cls ; h_E1 ; h_E2] → Linear → GELU → Dropout → Linear` (MLP); época: maior macro-F1 do DEV completo (primeiro máximo); 3 épocas, sem parada antecipada | DEV; TEST em `avaliar_test.py` |
| Two-Stage | `estrategias/two_stage/` | `two_stage.py treinar --estagio 1|2`, `compor-dev`, `avaliar-test` | `twostage_e1_…`, `twostage_e2_…`, `twostage_biobertpt_seed42` | estágio 1: maior AP de `relacao` no DEV (3 épocas); estágio 2: maior F1 de `negation_of` nos pares do DEV com relação anotada (10 épocas); τ pelo pipeline completo no DEV | DEV; TEST uma vez, com `limiar.json` conferido |

Nenhuma estratégia lê o TEST no treino nem na escolha de época ou de limiar; o
TEST de cada sistema é avaliado por um comando separado, uma vez, depois de
tudo congelado no DEV. As análises e os relatórios sobre as execuções estão em
[`../reproducao.md`](../reproducao.md).
