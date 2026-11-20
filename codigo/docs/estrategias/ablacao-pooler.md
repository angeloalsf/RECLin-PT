# Ablação do pooler (Etapa 5, M3)

Estado: **código, testes e notebooks prontos; nenhuma execução real feita.**

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
