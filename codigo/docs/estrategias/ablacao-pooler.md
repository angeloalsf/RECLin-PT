# Ablação do pooler (Etapa 5, M3)

Estado: **executada no Colab (T4) e analisada no DEV em 22/11/2026.** Execuções
`cls_direto__a59feee112b9__t1` e `pair_aware_pooler__2d00c2e15bfb__t3` (a `t2` falhou na
importação, por incompatibilidade torch/torchvision do ambiente, antes de treinar), branch
`resultados-colab@0c08b746385b`, código `24596718` (identidade `6d37bb304dfc`), torch 2.11.0,
transformers 5.16.1. `pooler_do_checkpoint: true` na variante com pooler.

## Por quê

O baseline e a Pair-Aware Linear diferem em duas coisas: a Pair-Aware lê os
marcadores `[E1]`/`[E2]` e dispensa o pooler (dense + tanh) do encoder. A
diferença observada entre as duas não separa os fatores.

| | com pooler | sem pooler |
| --- | --- | --- |
| só o `[CLS]` | `baseline` (treinado) | **`cls_direto`** |
| `[CLS]` + `[E1]` + `[E2]` | **`pair_aware_pooler`** | `pair_aware` (treinada) |

## O que foi implementado

- `reclin/estrategias/ablacao_pooler/` — `ConfigAblacao` (os campos e padrões da
  Pair-Aware Linear), as variantes `cls_direto` e `pair_aware_pooler`
  (`VARIANTES`) e a cabeça `CabecaAblacao` (`Dropout → Linear`, sobre `h_cls` com
  ou sem pooler, com ou sem os marcadores). `baseline` e `pair_aware` não foram
  alterados. O pacote não importa torch.
- Registro em `scripts/treinar.py` e `scripts/avaliar_test.py`; rótulos e as
  quatro comparações em `auditar_execucoes.py`.
- A variante com pooler registra em `treino.json`
  (`descricao.cabeca.pooler_do_checkpoint`) se os pesos do pooler vieram do
  checkpoint pré-treinado ou foram inicializados (como ocorre no baseline quando
  o checkpoint não os traz). Isso precisa ser lido antes de interpretar o efeito.
- Exceção estreita à regra "uma estratégia não importa outra"
  (`testes/unidade/test_pacote.py`): a ablação usa as três funções da
  Pair-Aware que leem os marcadores, para que a leitura seja a mesma.

## Verificações (software, modelo minúsculo)

`testes/unidade/test_ablacao_pooler.py`: com os mesmos pesos, a cabeça com
`(pooler, sem marcadores)` reproduz os logits do baseline e a com
`(sem pooler, marcadores)` reproduz os da Pair-Aware; salvar e recarregar
preserva os logits; os notebooks não avaliam nem carregam o TEST. Um treino de
ponta a ponta das duas variantes com o modelo minúsculo, num subconjunto técnico
de treino e DEV, terminou sem erro (números sem significado, descartados).

## Como executar (Colab, T4)

1. Fazer commit e push do código da Etapa 5 (as execuções registram a
   identidade do código; a ablação não existe no commit anterior).
2. Abrir `codigo/notebooks/ablacao/reclin_cls_direto.ipynb` e
   `reclin_pair_aware_pooler.ipynb`, com os secrets `GITHUB_PAT` e `HF_TOKEN`
   (o clone por token funciona com o repositório privado).
3. Rodar as seções 1 a 8 sem alterar `PARAMETROS` (o notebook confere que são
   os da Pair-Aware Linear). A seção 8.3 mostra as diferenças pareadas contra o
   baseline e a Pair-Aware.
4. Publicar os resultados na branch de resultados (seção 10) e avisar.

As pastas terão nomes `cls_direto__<config_id>__t1` e
`pair_aware_pooler__<config_id>__t1`. São exploratórias e não elegíveis ao TEST.

## Análise depois dos resultados

`diagnosticos/auditar_execucoes.py` sobre a branch de resultados atualizada; o
plano está em `docs/protocolo/registro-e4-experimentos-pos-fase.md`, seção 4.

## Resultados (DEV, semente 42)

Gerados por `diagnosticos/auditar_execucoes.py` (registros antigos conferidos e mantidos;
2 sistemas e 4 comparações acrescentados) e pelas tabelas `dev_ablacao_pooler*` do TCC
(`tcc/analise/gerar_tabelas_etapa5.py`).

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
