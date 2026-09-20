# Arquivo: rodada de retreino de setembro/2026 (pré-determinismo)

Snapshot das 4 execuções de baseline retreinadas em **10-12/09/2026**, arquivado
em **13/09/2026** antes de reforçar o determinismo do ambiente e rodar de novo.

| execução | treino/avaliação | `config_sha1` |
|---|---|---|
| BERTimbau s42  | 2026-09-10 19:17 UTC | `ac01c52aa595` |
| BioBERTpt s42  | 2026-09-10 19:17 UTC | `ae69a0267d65` |
| BERTimbau s43  | 2026-09-12 20:09 UTC | `6c0e457e0091` |
| BioBERTpt s43  | 2026-09-12 20:36 UTC | `9758b7e0605b` |

## Por que foi arquivado

Os checkpoints da rodada de agosto (`archive_max_gap25/`, 19 e 21/08) foram
perdidos, e o retreino **não reproduziu** os números que o Capítulo 6 do TCC
publica — divergências de até 0,046 em F1, com `config` verificado campo a campo
como idêntico, mesmos splits (`n_candidates` iguais) e mesmo `n_params`. As
trajetórias de treino divergem já na época 1, então a causa é não-determinismo de
execução (GPU/ambiente), não mudança de configuração.

Duas lacunas conhecidas sustentam essa hipótese e serão corrigidas antes da
próxima rodada:

1. `requirements.txt` usa apenas `>=` — nenhuma versão fixada, e o ambiente das
   rodadas nunca foi gravado junto com o resultado.
2. Falta `torch.use_deterministic_algorithms` e `CUBLAS_WORKSPACE_CONFIG` no
   código de treino.

Esta pasta congela a rodada para que a próxima (com determinismo reforçado) possa
ser comparada contra ela sem reabrir os JSONs.

## Números-chave desta rodada (test, n = 19.210)

| execução | macro-F1 | F1 `negation_of` | F1 `associated_with` | F1 `no_relation` | MCC | acurácia |
|---|---|---|---|---|---|---|
| BioBERTpt s42 | 0,7055 | 0,7092 | 0,4612 | 0,9462 | 0,4943 | 0,9020 |
| BioBERTpt s43 | 0,7136 | 0,7433 | 0,4507 | 0,9467 | 0,4813 | 0,9029 |
| BERTimbau s42 | 0,7074 | 0,7211 | 0,4540 | 0,9471 | 0,4817 | 0,9037 |
| BERTimbau s43 | 0,6987 | 0,6967 | 0,4531 | 0,9462 | 0,4829 | 0,9019 |

Média entre sementes (n = 2, desvio populacional):

| modelo | macro-F1 | F1 `negation_of` |
|---|---|---|
| BioBERTpt | 0,7095 ± 0,0040 | 0,7262 ± 0,0171 |
| BERTimbau | 0,7030 ± 0,0044 | 0,7089 ± 0,0122 |

`best_epoch = 3` nas 4 execuções — que é a **última** época (`epochs = 3`). O
guarda-corpo "melhor ≠ última" está correto, mas não foi exercitado aqui.

### Delta contra a rodada de agosto (`archive_max_gap25/`, o que o Cap. 6 publica)

| execução | macro-F1 | F1 `negation_of` |
|---|---|---|
| BioBERTpt s42 | 0,675 → 0,7055 (**+0,031**) | 0,677 → 0,7092 (**+0,032**) |
| BERTimbau s42 | 0,686 → 0,7074 (**+0,021**) | 0,694 → 0,7211 (**+0,027**) |
| BioBERTpt s43 | 0,712 → 0,7136 (+0,002) | 0,754 → 0,7433 (−0,011) |
| BERTimbau s43 | 0,704 → 0,6987 (−0,005) | 0,702 → 0,6967 (−0,005) |

As 6 métricas da semente 42 estouram 0,005 nos dois encoders; a semente 43 mexeu
pouco.

## Significância recalculada sobre estas preds

BioBERTpt (`a`) vs. BERTimbau (`b`), classe-alvo `negation_of`:

| | McNemar (binomial exato) | bootstrap pareado (10.000 reamostragens) |
|---|---|---|
| **semente 42** | b=501 / c=534, **p = 0,320** | Δ F1 = −0,0119; IC95% [−0,0489; +0,0248]; **p = 0,521** |
| **semente 43** | b=530 / c=510, **p = 0,556** | Δ F1 = +0,0466; IC95% [+0,0123; +0,0815]; **p = 0,0078** |

O que muda em relação ao texto atual do Cap. 6:

- **O McNemar deixa de rejeitar H0 nas duas sementes** (era p = 1,15e-05 na s42 e
  p = 2,75e-08 na s43). A afirmação "o McNemar aponta para o BERTimbau nas duas
  sementes, com significância" não sobrevive a esta rodada.
- **O sinal inverte na semente 43**: a contagem discordante passa a favorecer o
  BioBERTpt (530 × 510), não o BERTimbau.
- Os dois testes **discordam na semente 43**: o bootstrap acusa diferença
  significativa em F1 de `negation_of` a favor do BioBERTpt (p = 0,0078) enquanto
  o McNemar, que mede acerto por instância em todas as classes, não acusa nada
  (p = 0,556). A distinção entre os dois testes já está estabelecida na
  Seção 6.3 do TCC.

## Conteúdo

**Movidos de `results/`** (19 arquivos):

- `baseline_{biobertpt,bertimbau}_seed{42,43}.json` — métricas, `config`,
  `dev_history`, matriz de confusão, `instrumentation`
- `.preds.json` (4) — predições no test
- `.dev_preds.json` (4) — predições no dev, com `probs` e `restored_best_state`
- `.test_evals.jsonl` (4) — sidecar de 1 linha por execução, com `config_sha1`
- `significance_biobertpt_vs_bertimbau_seed{42,43}.json` — regerados em 13/09 01:48
- `summary_by_seed.json` — regerado em 13/09 01:48

**Copiados de `tcc/src/`** (11 arquivos, em `tcc/`): as tabelas e figuras geradas
a partir dos JSONs acima. São **cópias**, não movimentações — os originais
continuam em `tcc/src/tabelas/` e `tcc/src/imagens/resultados/` porque são
entradas de build do LaTeX (`\input{}` e `\includegraphics{}` no Capítulo 6), e
removê-los quebraria a compilação do `main.tex`.

- `tcc/tabelas/`: `resultados.tex`, `robustez_semente.tex`, `significancia.tex`,
  `significancia_seed43.tex`, `dev_history.tex`
- `tcc/imagens_resultados/`: `f1_por_classe.png`, `cm_biobertpt.png`,
  `cm_bertimbau.png`, `curvas_treino.png`, `curvas_treino_biobertpt.png`,
  `curvas_treino_bertimbau.png`

**Não arquivados** (confirmados independentes desta rodada de treino):
`results/tcc_eda.json` e os artefatos de EDA em `tcc/src/tabelas/` e
`tcc/src/imagens/eda/`.

## Ordem para regerar depois da próxima rodada

`src/significance.py` + `scripts/aggregate_seeds.py` → `scripts/make_tcc_artifacts.py`
→ atualizar a prosa → `scripts/check_tcc_numbers.py`.

> Atenção: `check_mcnemar_direction` lê `results/significance_*.json`. Enquanto
> esses arquivos não forem regerados junto com as preds, o check dá **falso OK**
> na direção do McNemar.

---

# Adendo (20/09/2026): a investigação de determinismo está encerrada

`COMPARACAO_determinismo_20set.md`, nesta pasta, **é o documento que fecha a
investigação**. Ele compara execução a execução esta rodada arquivada (10-12/09)
com a de 17-20/09, feita já com `torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG` e as versões fixadas em `requirements.txt`.

**Veredito: as duas lacunas listadas acima foram fechadas e ainda assim a rodada
nova diverge.** O campo `environment` é idêntico nas 4 execuções e bate com os
pins; `config`, `config_sha1`, `n_candidates`, `n_params` e `best_epoch` batem com
esta pasta; o `y_true` do test é bit-a-bit idêntico. Mesmo assim 15 das 24 métricas
estouram 0,005 e 3% a 4% das predições do test mudam. Reavaliar o *mesmo*
checkpoint reproduz exatamente — **a inferência é determinística, o treino não**.
A causa mais provável é kernel de gradiente não-determinístico do CUDA
(`scatter_add` em `resize_token_embeddings`), limitação conhecida do PyTorch e
fora do escopo deste projeto.

Por isso esta pasta **não** será substituída por uma rodada "definitiva": ela é
metade da evidência. A rodada de **17-20/09 permanece em `results/` como os
números oficiais e finais do Capítulo 6**, e a instabilidade residual entre as
duas rodadas é documentada como ameaça à validade.

## Também arquivado aqui

- `COMPARACAO_determinismo_20set.md` — movido de `results/` em 20/09/2026.
- `significance_biobertpt_vs_bertimbau_seed43.OFFSPEC-bootstrap-seed42.json` —
  o arquivo que esteve em `results/` entre 20/09 16:30 e a regeneração da mesma
  data, gerado com `--seed 42` no bootstrap em vez de `--seed 43`. Difere do
  oficial só no *bootstrap* (IC95% [−0,016; +0,057], p = 0,2804, contra
  [−0,018; +0,057], p = 0,2878); McNemar, acurácia e F1 são idênticos, e o
  veredito ("Não") não muda. Preservado porque `COMPARACAO_determinismo_20set.md`
  ainda cita os valores dele.

> O aviso do `check_mcnemar_direction` acima está **resolvido**: os
> `results/significance_*.json` foram regerados junto com as preds em 20/09/2026,
> e o falso OK na direção do McNemar acabou — o check agora acusa a inversão nas
> duas sementes.
