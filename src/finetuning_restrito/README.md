# `src/finetuning_restrito/` · frente experimental isolada

Fine-tuning de um encoder **só no espaço restrito de candidatos** (pares cujo `e1`
é pista de negação segundo o léxico da fase 2), para testar se um modelo com pesos
próprios bate os 4 baselines do Cap. 6 da mesma forma que o filtro de pista bateu.
Se bater, ele é candidato a substituir o filtro como RECLin-PT entregue. Se não
bater na sanidade, a frente para ali.

## Isolamento (como é garantido, não só prometido)

| regra | mecanismo |
|---|---|
| nenhum arquivo existente é modificado | todo o código está nesta pasta; tudo o que ela grava vai para `results/finetuning_restrito/` |
| saída só em `results/finetuning_restrito/` | `_isolamento.exigir_saida_isolada` recusa `--out`, `--ckpt-dir` e relatórios em qualquer outro ponto de `results/` ou do repositório |
| reuso sem cópia | `build_marked_window`, `set_all_seeds`, `predict`, `evaluate`, `make_loader`, `collect_environment`, `save_best_model`, `save_last_checkpoint`, `load_last_checkpoint`, `load_best_state`, `best_epoch_from_history`, `build_preds_payload`, `build_arg_parser` são **importados** de `src/relation_extraction.py`; o léxico vem de `negation_lexicon.induce_lexicon`; a significância é `src/significance.py` executado sem alteração |
| `logs/pipeline.log` intocado | o logger do núcleo abre esse arquivo no import; `_isolamento` registra handlers antes, e `get_logger` (idempotente) deixa de abrir o arquivo. Os logs desta frente vão para `<out>.train_log.txt` |
| `src/__pycache__/` intocado | `sys.dont_write_bytecode = True` em todos os pontos de entrada |
| checkpoints e backups separados | `--ckpt-dir` com `best_model/` ou `last_checkpoint/` e sem `finetuning_restrito_guard.json` é recusado (seria a pasta de um baseline, e `save_best_model` apagaria o `best_model/` dele); `--hf-backup-repo` sem `restrito` no nome é recusado |

Por que uma **subpasta de `src/`** e não um diretório na raiz: os módulos de `src/`
são planos e se importam pelo nome (`from candidates import ...`). Uma subpasta
enxerga esse namespace com um único `sys.path.insert`, não acrescenta nenhum
módulo a ele e deixa a separação visível. O nome é o mesmo do diretório de
resultados, então um `grep finetuning_restrito` encontra o código e os números.
O notebook do Colab mora aqui, e não em `notebooks/`, para que desfazer a frente
seja apagar duas pastas.

## Arquivos

| arquivo | papel |
|---|---|
| `_isolamento.py` | loggers, bloqueio de bytecode, guarda de caminhos de saída |
| `restricted_space.py` | léxico (com guarda contra `results/CALIBRACAO_filtro.json`), espaço restrito, **remapeamento**, pesos `balanced` recalculados. Não importa torch |
| `train_restrito.py` | treino, seleção de época, sidecars remapeados, `environment`. Tarefas 2 e 3 |
| `test_remapeamento.py` | 7 checagens em CPU (Tarefa 1.4) |
| `criterio_parada.py` | regra de parada da Tarefa 2 (DEV, contra o baseline) |
| `avaliacao_final.py` | Tarefa 4: significância contra 4 baselines e 4 filtros, agregação por semente |
| `colab_finetuning_restrito.ipynb` | roteiro do Colab, com a Tarefa 3 bloqueada até o critério passar |

## Decisões que mudam a leitura dos números

**O espaço restrito real não é o de "~8.090 / teto 0,98".** Esses números eram do
filtro pelo tipo UMLS `Negation`. Com o léxico `min_freq=3` (11 formas) o espaço é:

| split | completo | restrito | fração | teto de recall `negation_of` | `no_relation` por `negation_of` |
|---|---|---|---|---|---|
| train | 152.686 | 6.805 | 4,46% | 0,9633 | 4,61 |
| dev | 19.064 | 774 | 4,06% | 0,9467 | 4,42 |
| test | 19.210 | 800 | 4,16% | 0,9737 | 4,39 |

**Pesos `balanced` recalculados no train restrito:** `negation_of` 1,876,
`associated_with` **133,43**, `no_relation` 0,4066 (no espaço completo eram
40,55 / 7,17 / 0,3526). O 133 vem de só **17** pares `associated_with` caberem no
espaço restrito. É a fórmula dos baselines aplicada ao novo treino, como pedido,
mas é um risco: um terço da perda recai sobre 17 exemplos. O critério de parada
registra quantos `negation_of` do dev acabam preditos como `associated_with`,
para que um fracasso por esse motivo seja visível.

**Remapeamento.** `y_true` nunca é remapeado (é o gold do espaço completo). Todo
par fora do espaço recebe `y_pred = no_relation` e `probs = [0, 0, 1]`. Por isso só
o **F1 de `negation_of`** é comparável aos baselines: `associated_with` fica quase
todo fora, o macro-F1 remapeado despenca, e o McNemar de `significance.py` (acerto
nas três classes) não é interpretável. No teste com predição fictícia sorteada, o
McNemar "favoreceu" a predição aleatória contra o baseline (b=1.775, c=1.192),
só porque `no_relation` fora do espaço acerta de graça.

**Seleção da melhor época:** macro-F1 do DEV **remapeado** (19.064 pares), o mesmo
critério e o mesmo conjunto dos baselines. Com `associated_with` praticamente
constante fora do espaço, ele se move quase só com o F1 de `negation_of`.

**Hiperparâmetros:** os dos baselines (`max_gap 25`, `ctx_chars 128`,
`max_length 128`, `batch 64`, `lr 2e-5`, `weight_decay 0,01`, `warmup 0,1`), exceto
`--epochs 10`. Mesmo com 10 épocas o restrito dá ~1.070 passos de otimização, contra
~7.160 dos baselines.

**Determinismo:** o mesmo dos baselines, por import (`CUBLAS_WORKSPACE_CONFIG`,
`PYTHONHASHSEED` exportado pelo notebook, `use_deterministic_algorithms(True,
warn_only=True)`, versões fixadas no `requirements.txt`). Novidade: os
`UserWarning` do modo determinístico são capturados e gravados no
`.train_log.txt` e em `instrumentation.avisos_capturados`.

## Estado

Tarefa 1 implementada e verificada em CPU
(`results/finetuning_restrito/tarefa1_verificacao_cpu.json`). Pipeline completo
exercitado de ponta a ponta com um BERT minúsculo aleatório (treino, seleção,
retomada com predições byte-idênticas, sidecars, critério de parada e avaliação
final), sem gravar nada desse ensaio no repositório. A Tarefa 2 precisa de GPU e
está pronta para rodar no notebook.
