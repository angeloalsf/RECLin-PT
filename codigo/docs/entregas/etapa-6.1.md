# Etapa 6.1 — Two-Stage, controle de calibração, proveniência e protocolo pré-registrado

| | |
| --- | --- |
| Etapa | 6.1 (antes da etapa 7) |
| Data | 11/10/2026 |
| Entrega | sem pacote ZIP: as alterações estão direto na pasta de trabalho, para revisão e versionamento pelo autor |
| Parte de | etapa 6, versão 1 |
| Escopo | implementação e **validação técnica em CPU** (modelo minúsculo, proveniência sintética). Nenhum treino em GPU, nenhum download do Hugging Face Hub, nenhuma avaliação do TEST real |

## O que foi implementado

**Two-Stage** (`reclin/estrategias/two_stage/`): dois classificadores binários
independentes, cada um com o seu fine-tuning a partir dos pesos **originais**
do BioBERTpt (numa revisão fixada).

| | Estágio 1 | Estágio 2 |
| --- | --- | --- |
| Pergunta | existe relação? `relacao` × `no_relation` | qual o tipo? `negation_of` × `associated_with` |
| TRAIN | todos os pares (8.351 / 144.335) | só os pares com relação anotada (1.255 / 7.096) |
| Checkpoint escolhido por (só DEV) | maior AP de `relacao` no DEV completo | maior F1 de `negation_of` nos 913 pares do DEV com relação anotada |
| Épocas | 3 (`ConfigEstagio1`) | 10 (`ConfigEstagio2`) |
| Pesos `balanced` | calculados no TRAIN do estágio | idem |

Inferência (`composicao`): P(relacao), arredondada a 4 casas, ≥ τ → estágio 2
(argmax); senão `no_relation`. τ é escolhido no DEV pelo pipeline completo,
pelo fluxo real (o estágio 2 só sobre os pares roteados por cada τ), e
congelado em `limiar.json`, vinculado aos hashes dos dois `melhor_modelo/`, às
configurações, à seed, à proveniência e à identidade do DEV. A saída tem os 3
rótulos sobre o conjunto completo, no formato padrão de sidecar.

**Baseline calibrado** (`reclin/estrategias/baseline_calibrado.py`): o controle
de calibração. P(negation_of) ≥ t → `negation_of`; senão o argmax entre
`associated_with` e `no_relation` (empate → `associated_with`). t é escolhido
no DEV sobre as probabilidades que o baseline já grava (4 casas) — sem GPU —
pela **mesma** busca do τ, e congelado do mesmo modo.

**Busca de limiar** (`reclin/avaliacao/limiar.py`, uma só para τ e t): grade
0,05–0,95 (19 valores); F1 de `negation_of` da saída final pelas contagens;
desempates: maior macro-F1, menor |limiar − 0,5|, maior limiar; comparação
exata por frações. O `limiar.json` tem `hash_registro` (limiar, configuração,
vínculos e tabela) e a escolha é refeita a partir das contagens da tabela na
conferência. Não há opção para informar um limiar.

**Proveniência dos pesos** (`reclin/pretreino.py`, `scripts/pretreino.py`),
versão enxuta: `baixar` aceita só SHA de 40 caracteres; `PROVENIENCIA.json` com
o commit, o SHA-256 e o tamanho de cada arquivo; conferência antes de cada
carga, com carga offline; commit e hash do manifesto no `config.json`;
`conferir --execucoes` antes da comparação final (código novo, sem validação,
configuração do protocolo, mesma semente, mesmo commit e manifesto).

**Inicialização independente** (`two_stage/inicializacao.py`), três
verificações separadas: encoder igual, tensor a tensor, ao arquivo de pesos do
commit; cabeça `classifier.*` nova, [2, hidden], inicializada depois de
`fixar_sementes(42)`; marcadores com os ids V a V+3 (os do baseline) e as linhas
novas de `resize_token_embeddings`. Recusa, antes de qualquer carga, o
`melhor_modelo/` do baseline, do outro estágio, do restrito ou da Pair-Aware,
qualquer caminho dentro de uma execução, tokenizer com os marcadores,
configuração com rótulos da tarefa ou cabeça de classificação, pasta sem
`PROVENIENCIA.json` e identificador do Hub.

**Validação técnica** (`reclin/execucao/validacao.py`): `--validacao-epocas N`
em `treinar.py` e `two_stage.py treinar` marca a execução no `config.json`
(nomes `*_validacao_<N>ep`; a composição e a calibração sobre elas também ficam
marcadas). Execuções de validação são recusadas por `avaliar_test.py`,
`two_stage.py avaliar-test`, `baseline_calibrado.py aplicar-test`,
`pretreino.py conferir --execucoes` e `comparar.py`. Na rodada 1V: baseline e
estágio 1 com 1 época, estágio 2 com **2 épocas** (exercita a seleção e o
desempate "primeira época").

**Protocolo pré-registrado**: `codigo/docs/protocolo/fase-biobertpt-seed42.md`
(escopo "validação inicial, seed 42", só BioBERTpt, nenhum resultado do
legado; revisão dos pesos; critérios de seleção e parada das 6 configurações;
limiares; rodada de validação; sequência; comparações; interpretação;
pendências C11 e C13), com um bloco JSON conferido contra o código.

## Arquivos

Criados (21):

```
codigo/reclin/pretreino.py
codigo/reclin/avaliacao/limiar.py
codigo/reclin/execucao/validacao.py
codigo/reclin/estrategias/baseline_calibrado.py
codigo/reclin/estrategias/two_stage/__init__.py
codigo/reclin/estrategias/two_stage/composicao.py
codigo/reclin/estrategias/two_stage/estagios.py
codigo/reclin/estrategias/two_stage/inicializacao.py
codigo/reclin/estrategias/two_stage/metricas.py
codigo/reclin/estrategias/two_stage/treino.py
codigo/scripts/pretreino.py
codigo/scripts/two_stage.py
codigo/scripts/baseline_calibrado.py
codigo/testes/unidade/test_pretreino.py
codigo/testes/unidade/test_limiar.py
codigo/testes/unidade/test_validacao.py
codigo/testes/unidade/test_two_stage.py
codigo/testes/unidade/test_protocolo_fase.py
codigo/testes/integracao/test_two_stage.py
codigo/docs/protocolo/fase-biobertpt-seed42.md
codigo/docs/entregas/etapa-6.1.md
```

Modificados (12):

| Arquivo | Mudança | Compartilhado |
| --- | --- | --- |
| `codigo/reclin/treino/laco.py` | `campo_selecao` opcional (padrão `dev_macro_f1`) | sim |
| `codigo/reclin/treino/classificador.py` | proveniência conferida e registrada só quando há `PROVENIENCIA.json`; marca de validação; `avaliar_test` recusa execuções de validação; `particoes_congeladas` reconhece uma cópia com o MANIFEST congelado idêntico | sim |
| `codigo/scripts/treinar.py` | carga offline quando o modelo tem manifesto; `--validacao-epocas` | sim |
| `codigo/scripts/comparar.py` | recusa sidecars de execuções de validação | sim |
| `codigo/reclin/__init__.py`, `codigo/reclin/avaliacao/__init__.py`, `codigo/reclin/execucao/__init__.py`, `codigo/reclin/estrategias/__init__.py` | só documentação | — |
| `.gitignore` | `codigo/modelos_pretreinados/` | — |
| `README.md`, `codigo/README.md`, `codigo/docs/entregas/README.md` | estado, comandos, fim dos ZIPs | — |

Removidos: nenhum. Nada em `codigo/testes/referencia/` (inclusive
`resultados_legado/`), nas partições, em `negacao/`, `execucao/` (fora o módulo
novo), `avaliacao/` (fora o módulo novo), nas outras estratégias nem no TCC foi
alterado.

## Testes e resultados

Ambiente: contêiner Linux na nuvem (2 núcleos, sem GPU), Python 3.13.16,
torch 2.11.0, transformers 5.16.1, tokenizers 0.23.2, numpy 2.1.3, scipy
1.16.3, huggingface_hub 1.29.0 (as versões fixadas de `requirements.txt`);
scikit-learn 1.6.1 instalado só para uma conferência do AP. A máquina Linux do
computador do autor não comportou o torch (índice de CPU do PyTorch bloqueado
pela rede; o pacote padrão não cabe no disco e na memória livres), então os
testes rodaram numa cópia de trabalho na nuvem, sincronizada com a pasta do
autor (hash agregado dos 185 arquivos conferido nos dois lados antes de
começar). Cada medição rodou numa cópia própria, isolada por `PYTHONPATH`.

| Rodada | Código | Testes | Resultado |
| --- | --- | --- | --- |
| Partida | intocado | suíte existente | 486 passaram, 1 pulado (46 min 24 s) — o mesmo número da etapa 6 |
| S1 | + `laco.py` | suíte existente, sem alteração | 486 passaram, 1 pulado (40 min 16 s) |
| S2 | + `classificador.py` (+ `pretreino.py`, `validacao.py`) | suíte existente, sem alteração | 488 passaram, 1 pulado (23 min 25 s); os 2 a mais são as verificações automáticas de `test_pacote` para os módulos novos `pretreino` e `validacao` |
| S3 | + `treinar.py` | suíte existente, sem alteração | 488 passaram, 1 pulado (23 min 14 s) |
| S4 | estado final (+ `comparar.py`, Two-Stage, calibrado, testes novos) | suíte completa | **574 passaram, 1 pulado** (26 min 34 s): os 487 existentes + 75 novos + 13 verificações automáticas de `test_pacote` (10 módulos e 3 scripts novos) |

O pulado é, em todas as rodadas, o treino completo do baseline minúsculo (só
com `RECLIN_TREINO_COMPLETO=1`), como nas etapas 5 e 6. A mudança em
`comparar.py` não teve rodada isolada: ela entrou na S4, com o restante.

Transparência sobre o TEST: a suíte **existente** (exigida por A1, sem nenhuma
alteração) já contém, desde as etapas 1 a 6, testes técnicos que leem a
partição TEST real — conferência de conjuntos, hashes de janelas e lotes e a
avaliação do TEST do **modelo minúsculo aleatório** pelos comandos. Eles
rodaram como sempre rodaram. Nenhum modelo real foi treinado ou avaliado, e
nenhum número de desempenho saiu desta etapa. Os testes **novos** não leem o
TEST real.

Testes novos: 75 (unidade: `test_pretreino` 19, `test_limiar` 12,
`test_validacao` 4, `test_two_stage` 12, `test_protocolo_fase` 5; integração:
`test_two_stage` 23). Os de integração não leem o TEST real: usam as 6
primeiras linhas do TRAIN, as 4 primeiras do DEV e um **TEST sintético** (linhas
5–8 do DEV), só para exercitar o caminho da avaliação; o treino e a escolha de
τ e de t rodam numa cópia sem o arquivo do TEST.

### Critérios de aceitação

| # | Critério | Situação | Evidência |
| --- | --- | --- | --- |
| A1 | Suíte existente passa sem nenhum teste alterado, antes e depois de cada mudança compartilhada | atendido | rodadas Partida, S1–S4 (tabela acima); nenhum teste existente foi modificado |
| A2 | Mesmos dados do baseline | atendido | `test_contagens_de_cada_estagio` (8.351/144.335, 1.255/7.096, 913/18.151, 150/763 nas partições congeladas); `test_janelas_sao_as_do_baseline` (SHA-256 das janelas do TRAIN e do DEV = `dados.json`; janelas do estágio 2 = as do baseline nos mesmos índices) |
| A3 | Três verificações de inicialização | atendido | `test_inicializacao_independente_dos_dois_estagios`, `test_verificar_inicializacao_pelo_comando`, `test_verificacao_da_carga_acusa_divergencias` |
| A4 | Pontos de partida ajustados recusados | atendido | `test_recusa_pontos_de_partida_ajustados` (7 casos; a execução não chega a ser criada) |
| A5 | Estágios independentes | atendido | `test_estagio2_nao_depende_do_estagio1` (igual bit a bit com e sem o estágio 1; retomada cruzada recusada) |
| A6 | Treino, gravação, recarga, retomada | atendido em CPU | `test_retomada_exata` (meio e fim de época, nos dois estágios: pesos, histórico, melhor época, hashes e predições iguais à contínua); `test_recarga_do_melhor_modelo_reproduz_as_predicoes` |
| A7 | Seleção de época | atendido | `test_selecao_pelo_campo_do_estagio`; `test_laco_com_outro_campo_de_selecao` (pesos guardados = os da época 2 no empate 2 × 3; padrão inalterado) |
| A8 | Busca de τ pelo fluxo real | atendido | `test_busca_de_tau_pelo_fluxo_real` (cada uma das 19 linhas = composição pelo fluxo real com aquele τ); `test_limiar.py` |
| A9 | Recusas de `avaliar-test` | atendido | `test_avaliacao_do_test_recusa_vinculos_incompativeis` (limiar editado, `melhor_modelo/` alterado, estágios trocados, outro DEV), `test_avaliacao_do_test_uma_vez` |
| A10 | Inferência conjunta | atendido | `test_busca_de_tau_pelo_fluxo_real` (não roteados = `no_relation`; 3 rótulos; sidecar válido) |
| A11 | TEST isolado | atendido | treino dos estágios, `compor-dev` e calibração numa pasta sem `test.jsonl`; `test_pasta_so_com_train_dev_e_o_manifest_congelado` |
| A12 | Avaliação existente aceita | atendido | `test_avaliacao_e_comparacao_existentes_aceitam_o_two_stage` (`avaliar.py`, `comparar.py par`) |
| A13 | Entradas do legado recusadas | atendido | `test_entradas_do_legado_sao_recusadas` |
| A14 | Escopo: BioBERTpt e seed 42 por padrão; sementes divergentes recusadas | atendido | `test_configuracoes_e_nomes`, `test_sementes_divergentes_sao_recusadas`, `test_conferencia_final_das_execucoes` |
| A15 | Importações e `--help` | atendido | `test_pacote.py` (descobre os módulos e scripts novos automaticamente) |
| A16 | Os testes acusam desvios | atendido (ver abaixo) | 12 desvios |
| A17 | Proveniência | atendido | `test_pretreino.py`; `test_proveniencia_registrada_e_conferida_no_treino`; `test_conferencia_final_das_execucoes` |
| A18 | Protocolo × código | atendido | `test_protocolo_fase.py` |
| A19 | Baseline calibrado | atendido | `test_regra_do_baseline_calibrado`, `test_baseline_calibrado_usa_a_mesma_busca`, `test_baseline_calibrado_de_ponta_a_ponta` |
| A20 | Execuções de validação recusadas no TEST, na checagem final e nas comparações | atendido | `test_rodada_de_validacao_tecnica` (rodada 1V completa pelos comandos: baseline 1 época, estágio 1 com 1, estágio 2 com 2, `compor-dev`, calibração; 8 recusas), `test_validacao.py` |
| A21 | O modo de validação não altera o protocolo | atendido | `test_sem_marca_nem_manifesto_o_config_nao_muda` (chaves do `config.json` iguais às de antes), `test_rodada_de_validacao_tecnica` (só épocas e marca mudam; `--epochs` com `--validacao-epocas` recusado), `test_conferencia_final_das_execucoes` (épocas fora do protocolo sem marca recusadas) |

### A16 — os testes acusam desvios

Cada desvio foi aplicado numa cópia do código, os testes indicados rodaram e a
cópia foi restaurada:

| Desvio | Testes | Resultado |
| --- | --- | --- |
| Estágio iniciado por checkpoint ajustado (guardas de origem desligadas) | `test_recusa_pontos_de_partida_ajustados` | falha |
| Estágio 1 escolhido pelo macro-F1 em vez do AP | `test_two_stage.py`, `test_protocolo_fase.py` | falha |
| Desempate do limiar invertido | `test_limiar.py` | falha |
| Desempate de épocas invertido (`>=` no laço) | `test_laco_com_outro_campo_de_selecao` | **não falhou na primeira versão do teste** (ele refazia a escolha pelo histórico); o teste foi corrigido para comparar os pesos guardados e passou a falhar |
| Estágio 2 treinado com todos os pares | `test_two_stage.py` | falha |
| τ aplicado ao TEST sem conferir os vínculos | `test_avaliacao_do_test_recusa_vinculos_incompativeis` | falha |
| Sidecar do legado aceito como estágio | `test_entradas_do_legado_sao_recusadas` | falha |
| Roteamento sem o arredondamento a 4 casas | `test_busca_de_tau_pelo_fluxo_real` | falha |
| Empate do argmax do calibrado a favor de `no_relation` | `test_regra_do_baseline_calibrado` | falha |
| Proveniência não conferida antes da carga | `test_proveniencia_registrada_e_conferida_no_treino` | falha |
| TEST do baseline aceito numa execução de validação | `test_rodada_de_validacao_tecnica` | falha |
| `comparar.py` aceita execução de validação | `test_rodada_de_validacao_tecnica` | falha (na recusa "comparação") |

## Item (b): o que exige `test.jsonl`

Verificado no código e nos testes: **nada** no treino (baseline, restrito,
Pair-Aware, estágios), em `compor-dev` nem na calibração do baseline abre ou
exige o arquivo do TEST; a conferência das partições é feita só para TRAIN e
DEV. O único ponto encontrado foi um enfraquecimento silencioso: numa pasta
que não fosse a congelada, a conferência do tamanho dos conjuntos de
referência era desligada. Corrigido sem enfraquecer o isolamento:
`classificador.particoes_congeladas` reconhece uma pasta com o
`MANIFEST.json` congelado idêntico byte a byte (por exemplo, só
`train.jsonl`, `dev.jsonl` e o MANIFEST), e nela o tamanho continua conferido.
`preparar_dados.py conferir` (que confere as três partições) não faz parte da
rodada de validação.

## Desvios do plano aprovado

- **Onde ficou a busca de τ:** o plano previa `two_stage/limiar.py`; a busca e
  o congelamento ficaram em `reclin/avaliacao/limiar.py` (aprovado como módulo
  novo, compartilhado pelo τ e pelo t) e o vínculo do τ em
  `two_stage/composicao.py`.
- **Nome da composição de validação:** `twostage_validacao` (sem sufixo de
  épocas, porque os estágios têm 1 e 2 épocas); o calibrado de validação é
  `baseline_calibrado_validacao_1ep`.
- **`particoes_congeladas`** em `classificador.py`: não estava na lista de
  mudanças; decorre do item (b) e só fortalece a conferência.
- **Ambiente dos testes:** nuvem, não a máquina do computador do autor (ver
  "Testes e resultados"); Windows não foi executado.
- **A16:** a primeira versão de um teste não acusou o desvio do desempate de
  épocas; corrigida (tabela acima).

## Limitações e pendências

- **Nada em GPU:** os passos 0, 1, 1V e seguintes dependem de autorização
  específica. A verificação da inicialização com os pesos reais do BioBERTpt
  (pooler, nomes de tensores de um checkpoint antigo, `LayerNorm.gamma/beta`)
  só acontece no passo 1; o código trata esses casos, mas eles não foram
  exercitados com o modelo real.
- **Retomada e determinismo em GPU:** seguem não verificados (pendência da
  etapa 5); o teste opcional de retomada do estágio 2 está aprovado para o
  passo 1V.
- **Windows:** não executado.
- **Custo da suíte:** a suíte completa levou 26 min 34 s (S4, sozinha, 2
  núcleos); os testes novos acrescentam poucos minutos. Rodando em paralelo com
  outras medições, os de integração chegaram a 16 min; os dois de retomada do
  estágio 1 estão marcados `lento` por isso (sozinhos, cerca de 11 s cada).
- **Durações:** as rodadas Partida e S1 foram mais lentas porque dividiram a
  CPU com o desenvolvimento e os testes de mutação; os números de testes, não
  os tempos, são o que se compara.
- **C11 (revisão acadêmica, etapa 9):** sem o BERTimbau, a comparação entre
  encoders, pergunta central do texto atual do TCC, não terá resultado novo
  nesta fase. O TCC não foi alterado.
- **C13 (etapa 10):** a regra de escolha do modelo final será definida numa
  etapa posterior.
- Seguem em aberto, de etapas anteriores: `corpus.py` e a pasta `tcc/`.

## Próximo passo

Aguardar a revisão desta etapa. A etapa 7 e os passos da fase científica (0,
1, 1V, …) só começam com autorização explícita.
