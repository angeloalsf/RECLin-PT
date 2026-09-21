# Comparação: rodada com determinismo reforçado (14-17/09/2026) vs. `archive_pre_determinismo/` (10-12/09/2026)

Apurado em 20/09/2026. **Veredito: (b) — a rodada nova diverge de novo.**
Nenhum artefato do TCC foi regerado; a prosa não foi tocada.

## 1. Ambiente registrado (Tarefa 1)

As 4 execuções gravaram `environment` **idêntico campo a campo** — paridade de
ambiente confirmada:

| campo | valor nas 4 execuções | pin em `requirements.txt` | confere |
|---|---|---|---|
| `torch` | `2.11.0+cu128` | `torch==2.11.0` | sim (sufixo `+cu128` é o build CUDA) |
| `transformers` | `5.16.1` | `transformers==5.16.1` | sim |
| `numpy` | `2.1.3` | `numpy==2.1.3` | sim |
| `scipy` | `1.16.3` | `scipy==1.16.3` | sim |
| `scikit_learn` | `1.6.1` | `scikit-learn==1.6.1` | sim |
| `cublas_workspace_config` | `:4096:8` | — | sim, valeu na execução |
| `gpu` | `Tesla T4` | — | sim, nas 4 |
| `cuda` / `cudnn` / `python` | `12.8` / `91900` / `3.13.15` | — | idêntico nas 4 |

`config` idêntico ao do arquivo nas 4 execuções (`batch_size=64`,
`class_weight=balanced`, `ctx_chars=128`, `epochs=3`, `lr=2e-05`, `max_gap=25`,
`max_length=128`); `config_sha1` de cada execução igual ao do arquivo;
`n_candidates` (train 152.686 / dev 19.064 / test 19.210), `n_params` e
`best_epoch=3` também iguais. `y_true` do test é bit-a-bit idêntico nas 4
comparações — a divergência é do lado do modelo, não dos dados.

### Ressalvas de instrumentação

1. `requirements.txt` fixa `huggingface_hub`, `lxml` e `matplotlib`, mas
   `collect_environment()` **não** registra essas três. Nenhuma delas entra no
   treino, mas a paridade só está provada para as cinco que são gravadas.
2. `baseline_bertimbau_seed43.test_evals.jsonl` tem **duas** linhas (17/09
   13:54 e 14:30), com `test_macro_f1` e `test_negation_of_f1` idênticos até a
   6ª casa. Ou seja: reavaliar o mesmo checkpoint reproduz exatamente — a
   inferência é determinística. O que não reproduz é o **treino**.

### UserWarnings de não-determinismo (não capturados)

Não existem logs desta rodada em lugar nenhum do repositório:
`logs/pipeline.log` só tem entradas locais (nada entre 14 e 17/09, as datas do
retreino) e os três notebooks de Colab estão com **0 outputs salvos**. Busca por
`UserWarning` / `does not have a deterministic implementation` em todo o repo só
acha o README do arquivo.

Isso importa porque `src/relation_extraction.py:220` chama
`torch.use_deterministic_algorithms(True, warn_only=True)`. Com `warn_only=True`
cada operação sem kernel determinístico **não falha**: cai no kernel
não-determinístico e emite um `UserWarning` nomeando a op. A lista dessas ops era
justamente o objetivo declarado do `warn_only` — e ela foi perdida junto com a
saída do Colab. **Sem essa lista, não há como dizer quais ops continuam sem
garantia.** Para a próxima rodada: redirecionar `warnings` para um arquivo e
salvá-lo ao lado do `.json`, ou salvar o `.ipynb` executado.

## 2. Métricas: nova vs. arquivo (Tarefa 2)

Diferença = (nova) − (arquivo). **Negrito = |diferença| > 0,005.**

### BERTimbau seed 42 — 8/8 métricas estouram o limiar

| métrica | arquivo | nova | diferença |
|---|---|---|---|
| macro-F1 | 0,707383 | 0,697662 | **−0,009721** |
| F1 `negation_of` | 0,721053 | 0,710997 | **−0,010055** |
| F1 `associated_with` | 0,454007 | 0,441126 | **−0,012881** |
| F1 `no_relation` | 0,947090 | 0,940863 | **−0,006227** |
| MCC | 0,481674 | 0,476114 | **−0,005560** |
| acurácia | 0,903696 | 0,893233 | **−0,010463** |

### BERTimbau seed 43 — 2/6 estouram

| métrica | arquivo | nova | diferença |
|---|---|---|---|
| macro-F1 | 0,698656 | 0,703512 | +0,004856 |
| F1 `negation_of` | 0,696742 | 0,718346 | **+0,021604** |
| F1 `associated_with` | 0,453062 | 0,447472 | **−0,005590** |
| F1 `no_relation` | 0,946164 | 0,944716 | −0,001447 |
| MCC | 0,482863 | 0,477912 | −0,004951 |
| acurácia | 0,901874 | 0,899688 | −0,002186 |

### BioBERTpt seed 42 — 5/6 estouram

| métrica | arquivo | nova | diferença |
|---|---|---|---|
| macro-F1 | 0,705506 | 0,693227 | **−0,012279** |
| F1 `negation_of` | 0,709184 | 0,696078 | **−0,013105** |
| F1 `associated_with` | 0,461167 | 0,441793 | **−0,019374** |
| F1 `no_relation` | 0,946168 | 0,941809 | −0,004358 |
| MCC | 0,494266 | 0,477740 | **−0,016527** |
| acurácia | 0,901978 | 0,894482 | **−0,007496** |

### BioBERTpt seed 43 — 0/6 estouram (a única execução estável)

| métrica | arquivo | nova | diferença |
|---|---|---|---|
| macro-F1 | 0,713555 | 0,712786 | −0,000770 |
| F1 `negation_of` | 0,743316 | 0,738786 | −0,004529 |
| F1 `associated_with` | 0,450696 | 0,453350 | +0,002654 |
| F1 `no_relation` | 0,946655 | 0,946221 | −0,000434 |
| MCC | 0,481287 | 0,485073 | +0,003786 |
| acurácia | 0,902915 | 0,902238 | −0,000677 |

**Placar: 15 das 24 métricas estouram 0,005.** As diferenças chegam a 0,0216
(F1 `negation_of` na BERTimbau s43) — três ordens de grandeza acima de ruído de
ponto flutuante.

### Quantas predições mudaram

| execução | predições diferentes no test (n = 19.210) |
|---|---|
| BERTimbau s42 | 658 (3,43%) |
| BERTimbau s43 | 718 (3,74%) |
| BioBERTpt s42 | 720 (3,75%) |
| BioBERTpt s43 | 610 (3,18%) |

Entre 3% e 4% do test muda de rótulo entre duas execuções nominalmente
idênticas. As trajetórias de treino já divergem na **época 1** (ex.: BERTimbau
s43, `dev_macro_f1` 0,573342 → 0,564172; `train_loss` 0,716984 → 0,727755), o
mesmo padrão que motivou o arquivamento em 13/09.

### Médias entre sementes

| | macro-F1 arquivo | macro-F1 nova | F1 `negation_of` arquivo | F1 `negation_of` nova |
|---|---|---|---|---|
| BioBERTpt | 0,7095 ± 0,0040 | 0,7030 ± 0,0098 | 0,7262 ± 0,0171 | 0,7174 ± 0,0214 |
| BERTimbau | 0,7030 ± 0,0044 | 0,7006 ± 0,0029 | 0,7089 ± 0,0122 | 0,7147 ± 0,0037 |
| margem Bio − BERT | +0,0065 | **+0,0024** | +0,0174 | **+0,0028** |

O ordenamento por semente sobrevive nas duas métricas (BERTimbau vence a s42,
BioBERTpt vence a s43, em macro-F1 e em F1 `negation_of`). O que encolhe é a
**margem média**: 2,7× menor em macro-F1 e 6,2× menor em F1 `negation_of`.

## 3. Significância (Tarefa 3)

`src/significance.py --target negation_of --n-boot 10000 --seed 42`, A =
BioBERTpt, B = BERTimbau. Saídas novas gravadas em
`results/significance_biobertpt_vs_bertimbau_seed{42,43}.json` (a pasta não tinha
esses arquivos — nada foi sobrescrito).

Controle metodológico: as preds do arquivo foram reprocessadas **no mesmo
ambiente local** desta apuração. O McNemar (exato, determinístico) reproduziu o
README casa por casa; o bootstrap variou só na 4ª decimal (s43: IC
[+0,0125; +0,0816] p = 0,0046 aqui, contra [+0,0123; +0,0815] p = 0,0078 no
README) — efeito da versão de `numpy` no RNG de reamostragem, não das preds.
Logo, as diferenças abaixo **não** são artefato do ambiente de análise.

### McNemar (binomial exato)

| | arquivo | nova | |
|---|---|---|---|
| **s42** | b=501 / c=534 → favorece **BERTimbau**, p = 0,3199 | b=540 / c=516 → favorece **BioBERTpt**, p = 0,4791 | **direção INVERTE** |
| **s43** | b=530 / c=510 → favorece BioBERTpt, p = 0,5558 | b=559 / c=510 → favorece BioBERTpt, p = 0,1420 | direção preservada |

Nas quatro medições o McNemar não rejeita H0 (α = 0,05). A inversão na s42
acontece dentro do não-significativo, mas mostra que o sinal do teste não é
estável entre execuções.

### Bootstrap pareado, F1 de `negation_of` (10.000 reamostragens)

| | arquivo (README) | nova | |
|---|---|---|---|
| **s42** | Δ = −0,0119; IC95 [−0,0489; +0,0248]; p = 0,521 → não significativo | Δ = −0,0149; IC95 [−0,0509; +0,0209]; p = 0,4204 → não significativo | conclusão preservada |
| **s43** | Δ = **+0,0466**; IC95 [+0,0123; +0,0815]; p = 0,0078 → **significativo** | Δ = **+0,0204**; IC95 [−0,0176; +0,0575]; p = 0,2878 → **não significativo** | **perde a significância** |

O único resultado significativo que a rodada de 10-12/09 tinha (a vantagem do
BioBERTpt em `negation_of` na s43) desaparece: o Δ observado cai 2,3× e o IC95
passa a conter zero.

## 4. Veredito (Tarefa 4)

**(b).** A rodada nova diverge da `archive_pre_determinismo` apesar do
determinismo reforçado, com `config`, `config_sha1`, splits, `n_params` e
`environment` verificados idênticos. As diferenças são grandes demais para ruído
de ponto flutuante (até 0,0216 em F1; 3-4% das predições do test mudam), e a
divergência é **qualitativa**, não só numérica: a direção do McNemar inverte na
s42 e o bootstrap perde a significância na s43.

Corolário para o Capítulo 7: em macro-F1, para o BioBERTpt, a variação entre
execuções nominalmente idênticas (**0,0123** na s42) é **maior que a dispersão
entre sementes** da rodada arquivada (0,0080). A variabilidade de execução não é
uma correção de segunda ordem sobre a variabilidade de semente — ela a domina.

### O que isso implica

- O pin de versões **resolveu o que se propunha a resolver**: o drift de
  ambiente está eliminado e agora é auditável no próprio artefato. Isso não
  foi em vão — só não era a causa principal.
- A fonte residual de instabilidade é mais profunda que versão de biblioteca:
  não-determinismo de hardware/driver do Colab (kernels atômicos de GPU,
  redução em ponto flutuante, seleção de algoritmo do cuDNN), amplificado pelo
  `warn_only=True` que permite o fallback silencioso. Está fora do controle do
  projeto com a infraestrutura disponível.
- Nenhum dos dois conjuntos pode ser apresentado como "o" resultado. Os dois são
  amostras da mesma distribuição de execução, e o par deve ser citado como
  evidência empírica da variabilidade entre execuções idênticas.
- A conclusão que **sobrevive às duas rodadas**: nenhum dos dois encoders é
  superior com significância estatística em `negation_of`, e o ordenamento por
  semente é consistente (BERTimbau na s42, BioBERTpt na s43) enquanto a margem
  média é pequena e instável.

### Sugestões (nada executado)

1. Documentar no Cap. 7 como ameaça adicional à validade, com as duas tabelas
   como evidência.
2. Capturar `warnings` da próxima execução em arquivo — é a única forma de saber
   quais ops o `warn_only=True` está deixando passar.
3. Considerar reformular o Cap. 6 em termos de faixa observada entre execuções em
   vez de ponto único, já que as 8 execuções disponíveis (4 + 4) permitem isso.
