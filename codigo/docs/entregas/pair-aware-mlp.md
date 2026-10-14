# Estratégia `pair_aware_mlp` (13/10/2026)

Nova estratégia **independente**, pedida pelo autor, ao lado da `pair_aware` (cabeça linear, D10), que **não foi alterada**.

| | `pair_aware` (D10) | `pair_aware_mlp` (nova) |
| --- | --- | --- |
| Módulo | `reclin/estrategias/pair_aware/` | `reclin/estrategias/pair_aware_mlp/` |
| Estratégia / execução | `pair_aware` / `pairaware_biobertpt_seed42` | `pair_aware_mlp` / `pairawaremlp_biobertpt_seed42` |
| Cabeça | `Dropout → Linear(3d, 3)` | `Linear(3d, m) → GELU → Dropout → Linear(m, 3)` |
| Arquivos do modelo | `pair_aware_config.json`, `pair_aware_head.safetensors` | `pair_aware_mlp_config.json`, `pair_aware_mlp_head.safetensors` |
| Config | `ConfigPairAware` (+ `head_dropout`) | `ConfigPairAwareMLP` (+ `mlp_hidden`, `head_dropout`) |
| Pares / léxico | todos / não usa | todos / não usa |
| Épocas / lote / seleção | 3 / 64 / macro-F1 do DEV completo | idem |
| Protocolo | consta (D10) | **fora do protocolo**: exploratória, sem nome oficial |

## Isolamento
Estratégia, nome de execução, `estrategia_config.versao`, arquivos do `melhor_modelo/` e classe do modelo são próprios; a
chave de configuração e os checkpoints diferem, então uma não retoma nem avalia a execução da outra (cada
`montagem_de_registro` recusa o registro da outra). A MLP reutiliza de `pair_aware/cabeca.py`, sem cópia, só o que não
depende da cabeça (posição dos marcadores, carga do encoder sem pooler, contagem de marcadores ausentes). Não foi alterado
nenhum arquivo de `pair_aware/`, de resultados históricos ou do protocolo. `scripts/pretreino.py` não a lista: a
conferência final da fase a recusa como «sem configuração de protocolo», por desenho.

## Executar
`python codigo/scripts/treinar.py --estrategia pair_aware_mlp --encoder biobertpt --seed 42 --modelo <pesos com PROVENIENCIA.json> --saida <execucoes>`
(ou `notebooks/reclin_pair_aware_mlp.ipynb` no Colab). Opções próprias: `--mlp-hidden`, `--head-dropout`.
O TEST não é avaliado no treino.

## Pendente
Nada foi treinado nem avaliado; os testes novos (`testes/unidade/test_pair_aware_mlp.py`) não foram executados.
