# Pair-Aware linear com BioBERTpt-Clin (variante de encoder)

Estado: **implementado e verificado só estaticamente**. Nada foi treinado, avaliado, baixado ou publicado; não há resultado científico.

## O que é
A estratégia `pair_aware` (D10: `[h_cls ; h_[E1] ; h_[E2]] → Dropout → Linear(3d, 3)`, sem MLP, sem filtro lexical, sem léxico, todos os pares candidatos) com **um único componente trocado: o encoder**. Notebook: `notebooks/reclin_pair_aware_biobertpt_clin.ipynb`, derivado de `reclin_pair_aware.ipynb` (não alterado). Não é o `pair_aware_mlp` nem o Two-Stage.

## Encoder e revisão
| | Original | Variante |
|---|---|---|
| Chave em `ENCODERS` | `biobertpt` | `biobertpt_clin` |
| Repositório | `pucpr/biobertpt-all` | `pucpr/biobertpt-clin` |
| Revisão fixada | `36b11cfdfa49876b108fdbe88eaeeb9be3b03f1c` | `b59beec2e379e58a31b945039cc9ef163cfe22b7` (metadados do Hub; BERT-base cased, vocabulário de 119.547, só `pytorch_model.bin`) |
| Pasta dos pesos | `biobertpt-all@<sha>` | `biobertpt-clin@<sha>` |

Não há referência anterior ao `-clin` na documentação do projeto: a revisão veio do Hub e deve ser reconfirmada antes do primeiro treino. A revisão do `biobertpt-all` não foi reutilizada.

## Mantido (idêntico a `pairaware_biobertpt_seed42`)
Conferido no `config.json` da execução de referência: `ConfigPairAware`, seed 42, 3 épocas, batch 64, lr 2e-5, weight decay 0.01, warmup 0.1, max_grad_norm 1.0, class_weight balanced, max_gap 25, ctx_chars 128, max_length 128, `head_dropout=None`; `estrategia_config` v2; seleção da época pelo macro-F1 do DEV completo; partições e `MANIFEST`; TEST bloqueado (o notebook não o avalia). O código de `reclin/estrategias/pair_aware/` não foi tocado.

## Alterado
1. Notebook novo: `ENCODER`/`REPO_PESOS`/`NOME_PESOS`/`REVISAO_PESOS`, textos, backup HF `reclin-pt-pairaware-biobertpt-clin-seed42`, rótulo de natureza ("encoder biobertpt_clin, variante fora do pré-registro; DEV apenas") e lista da comparação no DEV (inclui `pairaware_biobertpt_seed42`).
2. Componentes compartilhados (aditivos; compartilhados com as demais variantes de encoder): `ENCODERS` (`reclin/config.py`), `Especificacao.encoder`, `--encoder` repassado só na variante, `pasta_dos_pesos(…, nome)`, guarda repositório×modelo e filtro por encoder em `execucao_atual` (`reclin/experimento/`).

## Cadeia de execução (conferida por leitura do código)
- Carregamento: `AutoTokenizer`/`AutoModel` sobre a pasta local dos pesos, com marcadores adicionados por `add_special_tokens` e `resize_token_embeddings` (genérico); o encoder é carregado sem pooler (`add_pooling_layer=False`).
- Pesos: `pretreino.baixar` é genérico (repositório + commit); o manifesto e o hash ficam na proveniência.
- Identidade/retomada: a chave inclui encoder, modelo, SHA e hash do manifesto dos pesos; pastas pelo `run_id` (sem nome oficial); retomada só de execução compatível; resultados e publicação só selecionam execuções do mesmo encoder.

## Impacto sobre a identidade do código
`reclin/config.py` participa da identidade global do código (`reclin/**/*.py`); `reclin/experimento/`, notebooks, testes e docs não participam. Logo, a entrada nova em `ENCODERS` muda a identidade global do código para execuções **novas**; execuções existentes não são alteradas, mas uma execução **interrompida** gravada com o código anterior não poderá ser retomada com este (divergência global bloqueia a retomada). Execuções concluídas e seus registros permanecem intactos.

## Pendências
- O protocolo fixa `pucpr/biobertpt-all`: esta é uma variante fora do pré-registro (formalizá-la exige emenda).
- Carregar o `pytorch_model.bin` com o `transformers` do Colab e o download extra de flax/tf precisam ser confirmados na primeira execução.
- Nenhum teste foi executado; a comparação no DEV (seção 8c) só funciona após as duas execuções existirem.
