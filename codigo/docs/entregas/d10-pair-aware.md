# D10 — Substituição da Pair-Aware (13/10/2026)

Autorizada pelo autor em 13/10/2026, **antes de qualquer treino científico da Pair-Aware** (a execução anterior
foi apagada e nunca foi publicada). Registrada na seção 3 de `docs/protocolo/fase-biobertpt-seed42.md`.

| | Configuração anterior (histórica) | Configuração vigente (D10) |
| --- | --- | --- |
| Modelo / seed | `pucpr/biobertpt-all` / 42 | idem |
| Pares usados | só os com `e1` no léxico congelado (filtro lexical) | **todos** os candidatos elegíveis |
| Léxico | filtro e remapeamento do resto para `no_relation` | **não usado** (nem filtro, nem feature) |
| Cabeça | `[h_cls ; h_E1 ; h_E2] → Linear → GELU → Dropout → Linear` (MLP), formato 1 | `[h_cls ; h_E1 ; h_E2] → Dropout → Linear(3d, 3)`, **sem MLP**, formato 2 |
| Épocas / lote | 2 (D5) / 64 | **3 / 64** (os do baseline) |
| Seleção da época | macro-F1 do DEV remapeado | macro-F1 do DEV **completo** (primeiro máximo) |
| Métricas comparáveis com o baseline | só o F1 de `negation_of` | todas |

## Código
`reclin/estrategias/pair_aware/{__init__,cabeca}.py` (um só módulo, sem estratégia paralela); `scripts/treinar.py`
(a Pair-Aware segue o caminho do baseline, sem léxico); `reclin/relatorios/{fase,tabelas}.py` (a Pair-Aware deixa de
ser «só F1 de `negation_of`»); `notebooks/reclin_pair_aware.ipynb`; testes unitários, de integração e de
equivalência ajustados; `test_protocolo_fase.py` confere o bloco JSON do protocolo.

## Proveniência
`config.json -> config -> estrategia_config` leva `versao = "v2: cabeca linear sem MLP, sem filtro lexical, sem
lexico"`, `filtro_lexical = "desativado"`, `lexico = "nao utilizado …"` e `cabeca.mlp = false`. Como esse registro
entra na chave da execução, uma execução da configuração antiga **não é retomada nem avaliada** pelo código novo
(`montagem_de_registro` a recusa). O nome oficial `pairaware_biobertpt_seed42` é mantido.

## Histórico preservado
`testes/referencia/resultados_legado/pair_aware/` e `testes/referencia/treino/pairaware_minusculo_seed42.*` não foram
alterados. O teste que reproduzia a Pair-Aware antiga (`test_pair_aware_minusculo…`) ficou marcado como *skip*
com o motivo; as demais referências do legado (índices do espaço restrito, métricas) continuam conferidas.

## Pendente
Nada foi treinado nem avaliado. Os testes alterados/criados e `test_protocolo_fase` ainda não foram executados.
