# Pair-Aware linear com BERTimbau Base (variante de encoder)

Estado: **implementado e verificado só estaticamente**. Nada foi treinado, avaliado ou publicado; as métricas, quando existirem, serão do DEV e dependem da execução manual no Colab.

## O que é
A mesma estratégia `pair_aware` (D10: `[h_cls ; h_[E1] ; h_[E2]] → Dropout → Linear(3d, 3)`, sem MLP, sem filtro lexical, sem léxico, todos os pares candidatos) com **um único componente trocado: o encoder pré-treinado**. Notebook: `notebooks/reclin_pair_aware_bertimbau_base.ipynb`, derivado de `reclin_pair_aware.ipynb` (que não foi alterado). Não confundir com `pair_aware_mlp` nem com o Two-Stage.

## Encoder e revisão
| | Original | Variante |
|---|---|---|
| Chave em `ENCODERS` | `biobertpt` | `bertimbau` |
| Repositório | `pucpr/biobertpt-all` | `neuralmind/bert-base-portuguese-cased` |
| Revisão fixada | `36b11cfdfa49876b108fdbe88eaeeb9be3b03f1c` | `94d69c95f98f7d5b2a8700c420230ae10def0baa` (Hub, 14/06/2022; consultada só nos metadados) |
| Pasta dos pesos | `biobertpt-all@<sha>` | `bert-base-portuguese-cased@<sha>` |

## Mantido (idêntico ao `pairaware_biobertpt_seed42`)
Conferido no `config.json` da execução de referência: classe `ConfigPairAware`, seed 42, 3 épocas, batch 64, lr 2e-5, weight decay 0.01, warmup 0.1, max_grad_norm 1.0, class_weight balanced, max_gap 25, ctx_chars 128, max_length 128, `head_dropout=None`; `estrategia_config` `v2` (cabeça linear, sem MLP, sem filtro, sem léxico, fallback `h_cls`); seleção da época pelo macro-F1 do DEV completo; partições (hashes do TRAIN e do DEV) e `MANIFEST`; critérios de avaliação; TEST bloqueado. O código da estratégia (`reclin/estrategias/pair_aware/`) não foi tocado.

## Alterado
1. Encoder, revisão dos pesos e, por consequência, `config_id`/`run_id`, pasta dos pesos, nome do backup no HF (`reclin-pt-pairaware-bertimbau-seed42`).
2. Notebook: constantes `ENCODER`/`REPO_PESOS`/`NOME_PESOS`, textos, rótulo de natureza ("parâmetros do protocolo, encoder bertimbau (variante fora do pré-registro; DEV apenas)") e a lista da comparação no DEV (inclui `pairaware_biobertpt_seed42`).
3. Componentes compartilhados (extensões aditivas, sem efeito para o `biobertpt-all`): `Especificacao.encoder` (padrão = encoder da fase), `--encoder` repassado ao script só na variante, `pasta_dos_pesos(…, nome)`, guarda "repositório dos pesos = modelo do encoder" e filtro por encoder em `execucao_atual`. A variante não recebe nome oficial (`alias_oficial = None`).

## Isolamento
A chave de identidade da execução inclui encoder, modelo, SHA e hash do manifesto dos pesos, partições e código; a pasta é nomeada pelo `run_id`; a retomada só aceita execução compatível; as seções de resultados/publicação só selecionam execuções do mesmo encoder.

## Limitações / pendências
- O protocolo (`docs/protocolo/fase-biobertpt-seed42.md`) fixa `pucpr/biobertpt-all`; esta é uma variante fora do pré-registro (exige emenda para ser formalizada).
- O repositório do BERTimbau só tem `pytorch_model.bin` (além de flax/tf): o carregamento no `transformers` do Colab precisa ser confirmado na primeira execução.
- Nenhum teste foi executado; a comparação com o BioBERTpt-all (seção 8c) só funciona depois das duas execuções existirem e é recusada se os candidatos diferirem.
