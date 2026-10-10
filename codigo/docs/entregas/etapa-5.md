# Entrega da etapa 5 — treino (entrada, modelos, treino/)

| | |
| --- | --- |
| Etapa | 5 |
| Versão | 1 |
| Data | 09/10/2026 |
| Pacote | `RECLin-PT-etapa-5-v1.zip` |
| Parte de | etapa 4, versão 1 (`RECLin-PT-etapa-4-v1.zip`) |

O pacote é o **projeto completo** até a etapa 5: extraído numa pasta limpa,
tem tudo o que é preciso para instalar, executar e testar o que já foi
implementado.

## Resultado dos critérios de aceitação

| Critério | Resultado | Evidência |
| --- | --- | --- |
| A. Janelas idênticas às do legado | **aprovado** | Janelas marcadas das 3 partições completas (190.960 textos) com o mesmo SHA-256; `y_true` e quantidades iguais; tokenizer com os marcadores nos mesmos ids (118 a 121); **cada lote** (`input_ids`, `attention_mask`, rótulos) igual ao do DataLoader do legado, na mesma ordem: TRAIN embaralhado nas épocas 1 e 2 (2 × 2.386 lotes), DEV (298) e TEST (301). Limite: só com o tokenizer minúsculo (os tokenizers reais não puderam ser baixados neste ambiente) |
| B. Reprodução do treino pequeno | **aprovado** | Baseline do legado sobre um subconjunto, 3 épocas: mesmos parâmetros iniciais; mesma loss em **cada um dos 249 passos** e em cada um dos 129 lotes do DEV; mesmos parâmetros, estado do otimizador e lr ao fim de **cada época**; mesmos parâmetros da melhor época; histórico, predições do DEV e do TEST (com `probs`) e métricas iguais. O baseline minúsculo sobre as partições **completas** (2 épocas, predições variadas, melhor época 2) também saiu igual, pelos scripts (teste opcional, `RECLIN_TREINO_COMPLETO=1`) |
| C. Retomada exata | **aprovado em CPU** | Interrompida no meio de uma época, no fim de uma época, no primeiro passo e três vezes seguidas: pesos, otimizador, agendador, geradores (globais e do DataLoader), histórico, melhor modelo, predições do DEV e do TEST **idênticos** aos da execução contínua. Em GPU não verificado (ver "Limitações") |
| D. Separação entre treino e TEST | **aprovado** | O treino roda com o arquivo do TEST ausente e não lê a partição (verificado); não grava predições, métricas nem trilha do TEST; a avaliação do TEST é outro comando, recusa treino incompleto, não altera nenhum arquivo do treino, e uma segunda avaliação exige `--reavaliar` e fica registrada na trilha |

Testes pedidos (seção 5 das instruções):

| # | Teste | Onde | Situação |
| --- | --- | --- | --- |
| 1 | Construção e alinhamento das entradas | `unidade/test_entrada.py` | aprovado |
| 2 | Equivalência das janelas | `equivalencia/test_treino_legado.py` | aprovado |
| 3 | Configuração e inicialização do modelo | `unidade/test_treino.py`, `equivalencia/test_treino_legado.py::test_parametros_iniciais` | aprovado |
| 4 | Ciclo de treino mínimo | `integracao/test_treino_retomada.py`, treino pequeno da equivalência | aprovado |
| 5 | Salvamento e carregamento de checkpoints | `unidade/test_treino.py` (checkpoint), integração | aprovado |
| 6 | Retomada exata × execução contínua | `integracao/test_treino_retomada.py` | aprovado (CPU) |
| 7 | Reprodutibilidade com a mesma configuração e semente | `integracao/test_treino_retomada.py` | aprovado (CPU, mesmo ambiente) |
| 8 | Separação entre treino, DEV e TEST | `integracao/test_treino_retomada.py` | aprovado |
| 9 | Compatibilidade com as etapas 1 a 4 | integração (`avaliar.py`, `comparar.py` sobre as execuções novas); toda a suíte anterior | aprovado |
| 10 | Comandos e mensagens diante de configurações inválidas | `integracao/test_treino_retomada.py` | aprovado |

## Diagnóstico (resumo)

### Como o legado treinava

`src/relation_extraction.py` (o restrito e a Pair-Aware copiam o laço):

- **Entrada:** janela de `±ctx_chars` (128) em torno do par, marcadores
  `[E1] [/E1] [E2] [/E2]` (espaço entre marcador e entidade; na mesma posição,
  fechamentos antes de aberturas), um exemplo por candidato na ordem de
  `iter_candidate_pairs`. Tokenização em cada lote (`padding=True`,
  `truncation`, `max_length` 128).
- **Modelo:** `AutoTokenizer` + `add_special_tokens` com os 4 marcadores;
  `AutoModelForSequenceClassification` (3 rótulos) + `resize_token_embeddings`.
- **Otimização:** CrossEntropy com pesos `balanced` do TRAIN; AdamW (`lr`,
  `weight_decay` 0,01); agendador linear com aquecimento de 10% dos passos;
  recorte do gradiente em 1,0; lotes de 64; 3 épocas.
- **Loader:** DataLoader com `torch.Generator` próprio semeado com a semente;
  embaralhado só no TRAIN.
- **Seleção:** a cada época, loss e macro-F1 no DEV; melhor época pelo
  macro-F1 estritamente maior (empate: a primeira). No fim, os pesos da melhor
  época são restaurados, o DEV é previsto (`.dev_preds.json`) e o TEST também,
  **no mesmo comando**.
- **Reprodutibilidade:** `set_all_seeds` (python, numpy, torch, CUDA, cuDNN
  determinístico, `use_deterministic_algorithms(warn_only=True)`), variáveis
  `CUBLAS_WORKSPACE_CONFIG` e `PYTHONHASHSEED`. A ordem sementes → dados →
  tokenizer → modelo → loaders decide o consumo dos geradores.
- **Checkpoint:** `last_checkpoint/training_state.pt` só no fim de cada
  época, com pesos, otimizador, agendador e geradores globais; `best_model/`.

### Lacunas do legado na retomada

1. O estado do gerador do DataLoader não era salvo: no baseline e no restrito,
   a primeira época retomada repetiria a ordem da época 1 (só a Pair-Aware
   avançava o gerador, por épocas inteiras).
2. Só no fim de uma época: uma queda no meio de uma época de ~48 min (T4)
   perdia a época.
3. Configuração divergente → o checkpoint era ignorado em silêncio e o treino
   recomeçava do zero.

As referências do legado são todas de execuções contínuas; a equivalência com
o legado é conferida sem retomada, e a retomada exata é uma propriedade
testada internamente (como previsto no plano).

### O que as evidências não decidem

- Se as execuções oficiais do legado (GPU) foram retomadas em algum momento:
  não há logs no repositório. Se foram, os números delas correspondem à
  retomada do legado (com a lacuna 1), não à execução contínua.
- A equivalência das janelas tokenizadas com os tokenizers reais (BioBERTpt,
  BERTimbau): o huggingface.co está bloqueado neste ambiente. O código é o
  mesmo para qualquer tokenizer, e as janelas (texto) são idênticas.
- O determinismo bit a bit em GPU e entre sistemas (Windows × Linux).

## O que foi implementado

### Módulos

| Módulo | Responsabilidade | Origem no legado |
| --- | --- | --- |
| `reclin/entrada.py` | `janela_marcada`, `MARCADORES`, `exemplos` (textos e rótulos alinhados com os candidatos), `criar_loader` (tokeniza por lote, gerador próprio) | `build_marked_window`, `build_dataset`, `make_loader` |
| `reclin/modelos.py` | `carregar_tokenizer` (marcadores), `carregar_classificador` (3 rótulos + resize), `salvar` (atômico), `recarregar` (`type(modelo).from_pretrained`), `identidade` (vocabulário, configuração, parâmetros) | carga em `run()`, `save_best_model`, `load_best_state` |
| `reclin/treino/reprodutibilidade.py` | `configurar_ambiente` (só scripts), `fixar_sementes`, `estados_rng`/`restaurar_rng`, `ambiente`, `coletar_avisos` | `set_all_seeds`, `collect_environment`, `environment_extra`, captura de avisos |
| `reclin/treino/checkpoint.py` | Estado completo de retomada (`ultimo.pt`, atômico), resumo (`ultimo.json`), `melhor_modelo/`, conferência de identidade | `save_last_checkpoint`, `load_last_checkpoint`, `_config_guard`, `checar_ckpt_dir` |
| `reclin/treino/laco.py` | `treinar` (épocas, DEV, seleção, checkpoints por época e a cada N passos, retomada exata), `prever`, `avaliar_loss`, `melhor_epoca`, `pesos_balanced` | laço de `run()`, `predict`, `evaluate`, `best_epoch_from_history` |
| `reclin/treino/classificador.py` | `treinar_execucao` (TRAIN e DEV, grava a execução) e `avaliar_test` (operação separada) | `run()`, dividido em dois |

O laço não conhece estratégias: o que é de uma estratégia (exemplos, como
pontuar o DEV, cabeça) entra como parâmetro. `classificador` é o classificador
da tarefa — todos os candidatos, cabeça [CLS], `Config` base — que a etapa 6
usará como baseline e do qual o restrito e a Pair-Aware partem; nenhuma
estratégia foi criada nesta etapa.

Acréscimos compatíveis no núcleo: `tarefa.conjunto_referencia(...,
conferir_tamanho=True)` (desligável para partições que não são as congeladas,
como os subconjuntos dos testes); dependências `torch` e `transformers` em
`pyproject.toml`; `tokenizers` e `safetensors` fixados em `requirements.txt`.

### Comandos

| Comando | O que faz |
| --- | --- |
| `scripts/treinar.py --nome N [--encoder E] [--seed S] [hiperparâmetros] [--modelo PASTA] [--checkpoint-a-cada K] [--retomar] [--parar-apos-passo P]` | Treina ou retoma a execução `N`; lê só TRAIN e DEV. Saída 0 concluído, 3 interrompido (retomável), 1 erro |
| `scripts/avaliar_test.py --execucao PASTA [--reavaliar]` | Avalia o TEST com o `melhor_modelo/` de uma execução concluída: `predicoes_test.json`, métricas e uma linha na trilha |

### O diretório de uma execução treinada

```
<saida>/<nome>/
    config.json            nome, estratégia ("classificador"), configuração, modelo,
                           SHA-256 das partições usadas, identidade dos conjuntos
    treino.json            dev_history, melhor época, n_params, pesos, passos,
                           identidade do modelo, sessões (nova/retomada, de onde a
                           onde, ambiente, avisos), concluido, test_avaliado
    checkpoints/ultimo.pt  estado completo de retomada
    checkpoints/ultimo.json
    checkpoints/melhor_modelo/
    predicoes_dev.json     DEV na melhor época — mesmo formato e extras do legado
    metricas.json          DEV; TEST depois de avaliar_test
    predicoes_test.json    só depois de avaliar_test — mesmo formato do legado
    avaliacoes_test.jsonl  trilha (eval_index, eval_index_for_config)
```

### Referências novas (pelo gerador, a partir do legado)

`gerar_referencias.py --partes etapa5`, num checkout separado do commit
`a5f055c`, no ambiente do legado (Python 3.13.16, torch 2.11.0, transformers
5.16.1, tokenizers 0.23.2, uma thread). Os "drivers" importam o código do
legado e só observam (não mudam nenhum cálculo):

| Arquivo | Conteúdo |
| --- | --- |
| `testes/referencia/lotes.json` | SHA-256 de cada lote do `make_loader` do legado nas partições completas: TRAIN épocas 1 e 2, DEV, TEST |
| `testes/referencia/treino/baseline_pequeno_seed42.{json,preds.json,dev_preds.json,registro.json}` | Baseline do legado sobre as primeiras 30/10/10 linhas das partições, 3 épocas, `lr` 1e-3, rodado duas vezes com resultados idênticos; o `.registro.json` tem a loss de cada passo e de cada lote do DEV e os hashes de parâmetros e otimizador |

As referências anteriores não mudaram (`referencias.json` só ganhou as
entradas novas).

## Decisões e divergências em relação ao legado

| # | Legado | Agora | Por quê |
| --- | --- | --- | --- |
| 1 | O TEST era previsto e registrado no fim do comando de treino | `avaliar_test.py`, separado, depois do treino concluído | Critério D; o treino não abre o arquivo do TEST |
| 2 | Checkpoint só no fim da época, sem o gerador do DataLoader | Fim de época e, opcionalmente, a cada N passos; gerador do início da época e atual; soma da loss da época | Retomada exata em qualquer ponto (critério C) |
| 3 | Configuração divergente: checkpoint ignorado em silêncio | Erro, com as chaves divergentes; retomar exige `--retomar` e criar exige nome novo | Uma execução não muda de identidade sem aviso |
| 4 | `best_model/`, `last_checkpoint/training_state.pt` | `checkpoints/melhor_modelo/`, `checkpoints/ultimo.pt` + `ultimo.json` | Sem compatibilidade com as pastas do Drive do legado (plano) |
| 5 | `<out>.json` com DEV e TEST | `treino.json` (treino e DEV) + `metricas.json` (DEV e TEST) | Consequência de 1; os sidecars mantêm o formato do legado |
| 6 | Métricas pelo sklearn | `avaliacao.metricas` | Iguais ao sklearn (etapa 2); o treino não depende do sklearn |
| 7 | Estado da CUDA restaurado se possível, com aviso | Retomar um checkpoint de GPU sem GPU (ou o contrário) é recusado | A retomada não seria a mesma execução |
| 8 | Backup no Hugging Face Hub a cada melhor época | Não implementado | Depende de rede e não é necessário aos critérios; previsto em `treino/backup_hf.py` para quando houver treino em GPU (etapa 6/8) |

## Evidências

### Reprodução do legado

| Verificação | Resultado |
| --- | --- |
| Janelas das 3 partições (SHA-256) | iguais |
| Lotes: TRAIN época 1 e 2, DEV, TEST | 2.386 + 2.386 + 298 + 301 lotes, todos iguais (hash de cada lote, 3 primeiros, tokens reais, maior comprimento) |
| Pesos `balanced` do TRAIN | iguais (`dados.json`) |
| Parâmetros iniciais (semente 42, tokenizer, classificador) | iguais |
| Treino pequeno: loss de 249 passos, 129 lotes do DEV | iguais (floats exatos) |
| Treino pequeno: parâmetros, otimizador e lr no fim de cada época; parâmetros da melhor época | iguais |
| Treino pequeno: histórico, predições DEV/TEST (com `probs`), métricas do TEST (inclusive o relatório por classe e a matriz) | iguais |
| Treino completo do modelo minúsculo (partições inteiras, 2 épocas) | histórico das 2 épocas, predições do DEV e do TEST (com `probs`) e métricas iguais; rodado com `RECLIN_TREINO_COMPLETO=1` (cerca de 9 min em 2 núcleos) |

Observação: no treino pequeno o modelo minúsculo prevê só `no_relation` no
DEV (as três épocas empatam no macro-F1, e a escolha é a primeira — o que
exercita o desempate); o treino completo é o que exercita predições variadas e
a escolha de uma época posterior (época 2).

### Os testes acusam desvios na retomada

Cada alteração foi aplicada ao laço, os testes de retomada rodaram e o código
foi restaurado:

| Alteração | Resultado |
| --- | --- |
| Não restaurar o gerador do DataLoader na retomada no meio da época | falha |
| Não restaurar o gerador na retomada no fim da época (o que o legado fazia no baseline e no restrito) | falha |
| Não restaurar os geradores globais (dropout) | falha |
| Não restaurar a soma da loss da época | falha |
| Não restaurar o agendador | falha |
| Não restaurar o otimizador | falha |
| Parar no último lote sem esgotar o iterador do DataLoader | os testes de retomada passaram (a contínua e a retomada fazem o mesmo); foi acrescentado `test_gerador_avanca_como_uma_passada_completa`, que compara com uma passada completa pelo loader (como a do legado) e falha com lotes todos completos |

## Testes

| Tipo | Onde | Quantidade | Depende de |
| --- | --- | --- | --- |
| Unidade | `testes/unidade/` | 185 (novos: 8 de `test_entrada.py`, 18 de `test_treino.py`, 7 da verificação de níveis para os módulos novos, 2 do `--help` dos scripts novos) | nada externo |
| Integração | `testes/integracao/` (nova) | 21 | modelo minúsculo, partições pequenas |
| Equivalência com as referências | `testes/equivalencia/` | 205 (novos: 18 de `test_treino_legado.py`) | só os arquivos versionados em `testes/referencia/` |
| Execuções com recurso externo | `testes/referencia/gerar_referencias.py` | — | um checkout do legado; fora da suíte |

Total: 411 testes, 27 marcados `lento` (as 26 comparações e os lotes do
TRAIN). Pulados conforme o ambiente: o treino completo (sem
`RECLIN_TREINO_COMPLETO=1`) e, fora do ambiente das referências, os 7 de
reprodução numérica do legado. Nenhum acessa a rede, uma GPU ou o legado.

### Resultados obtidos

Ambiente: Linux (contêiner de nuvem, 2 núcleos, sem GPU). O pacote foi extraído
numa pasta vazia, comparado byte a byte com o projeto e instalado em ambientes
novos (torch 2.11.0 do PyPI, que no Linux traz as bibliotecas de CUDA, sem uso
de GPU):

| Python | numpy | scipy | torch / transformers / tokenizers | Versões | Suíte completa |
| --- | --- | --- | --- | --- | --- |
| 3.10.20 | 2.2.6 | 1.15.3 | 2.11.0 / 5.16.1 / 0.23.2 | fixadas, exceto numpy e scipy (inexistentes para 3.10) | 410 passaram, 1 pulado |
| 3.12.3 | 2.1.3 | 1.16.3 | 2.11.0 / 5.16.1 / 0.23.2 | fixadas (`-c codigo/requirements.txt`) | 410 passaram, 1 pulado |
| 3.13.16 | 2.1.3 | 1.16.3 | 2.11.0 / 5.16.1 / 0.23.2 | fixadas | 410 passaram, 1 pulado |

O pulado nos três é o treino completo do modelo minúsculo (opcional, sem
`RECLIN_TREINO_COMPLETO=1`). Ele foi rodado à parte, com
`RECLIN_TREINO_COMPLETO=1`, sobre uma extração do pacote, com Python 3.13:
passou, em 9 min 16 s (o código é o do pacote final; depois dessa rodada só a
documentação mudou). Os testes de reprodução numérica do legado
rodaram nos três ambientes (todos Linux x86_64 com torch 2.11.0).

Em cada ambiente também: importação de todos os 28 módulos, `--help` dos 9
scripts e `preparar_dados.py conferir` (3 partições conferem). A suíte completa
levou de 4 min 16 s a 4 min 35 s por ambiente.

**Não executado:** nada foi rodado no **Windows** nem em **GPU**.

## Garantias de reprodutibilidade

- **Garantido e verificado (CPU):** no mesmo ambiente, mesma configuração e
  semente → mesmo resultado bit a bit; interrompida e retomada (em qualquer
  passo) → mesmo resultado bit a bit que a contínua; checkpoints periódicos não
  mudam o resultado.
- **Verificado contra o legado:** no ambiente das referências (Linux x86_64,
  torch 2.11.0, transformers 5.16.1, uma thread), o código novo reproduz o
  treino do legado bit a bit.
- **Não garantido:** igualdade bit a bit entre máquinas, sistemas, versões,
  número de threads, ou entre CPU e GPU. Em GPU, o determinismo depende de
  `CUBLAS_WORKSPACE_CONFIG` (definido pelos scripts) e de cuDNN; não foi
  verificado aqui.

## Como aplicar

Numa pasta limpa, no PowerShell:

```powershell
$zip     = "$env:USERPROFILE\Downloads\RECLin-PT-etapa-5-v1.zip"   # ajuste se salvou em outro lugar
$destino = "C:\Users\angeloals\Documents\RECLin-PT"

if (Test-Path $destino) { Rename-Item $destino "RECLin-PT-antes-etapa-5" }   # nada é apagado
Expand-Archive -Path $zip -DestinationPath $destino

cd $destino
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\codigo[testes]" -c codigo\requirements.txt
.\.venv\Scripts\python.exe codigo\scripts\preparar_dados.py conferir
.\.venv\Scripts\python.exe -m pytest codigo -rs
```

O pip instala agora também torch 2.11.0 (no Windows, a versão de CPU) e
transformers 5.16.1. Com Python 3.10, use o comando de `codigo/README.md`, que
fixa só as versões do treino (verificado aqui em Linux). No Windows, os testes de reprodução numérica do legado
são **pulados** com o motivo (as referências foram geradas em Linux); os de
retomada, reprodutibilidade, janelas e lotes rodam. Com `-rs` o pytest lista os
pulados e o motivo. Para rodá-los assim mesmo:
`$env:RECLIN_FORCAR_EQUIVALENCIA = "1"` antes do pytest (podem falhar nos
últimos bits; isso não indica erro de implementação).

Demonstração de treino, interrupção, retomada e avaliação do TEST em CPU, com o
modelo minúsculo e as partições completas (alguns minutos):

```powershell
$m = "codigo\testes\referencia\modelo_minusculo"
.\.venv\Scripts\python.exe codigo\scripts\treinar.py --nome demo --saida reproducao --modelo $m --epochs 1 --lr 1e-3 --parar-apos-passo 200
# saída 3: interrompido no passo 200, com checkpoint
.\.venv\Scripts\python.exe codigo\scripts\treinar.py --nome demo --saida reproducao --modelo $m --epochs 1 --lr 1e-3 --retomar
.\.venv\Scripts\python.exe codigo\scripts\avaliar_test.py --execucao reproducao\demo
Get-Content reproducao\demo\treino.json | Select-String '"tipo"'
```

O esperado (verificado aqui com os mesmos comandos, em Linux; cerca de 4 min):
o primeiro termina com código 3 e "Interrompido em {'epoca': 1,
'passo_na_epoca': 200, ...}"; o segundo, com "Retomando: época 1, passo 200 de
2386" e "Época 1/1: início (retomada no passo 201)", termina com código 0 e
"Melhor época: 1 (dev_macro_f1=0.3533)"; o terceiro grava o TEST
("macro-F1=0.3497 | F1 negation_of=0.1013"); e `treino.json` tem duas sessões,
`"tipo": "nova"` e `"tipo": "retomada"`. No Windows os números podem diferir
nos últimos dígitos. A pasta `reproducao\` pode ser apagada depois.

### Registrar no git só as alterações da etapa 5

A partir de um repositório que já tem o commit da etapa 4, com a cópia de
trabalho limpa. A etapa 5 não remove nenhum arquivo, então extrair por cima
equivale a extrair numa pasta limpa. Esperado: **10 modificados, 19
adicionados, 0 removidos**. Os arquivos são preparados pelo nome (sem
`git add -A`), e a conferência antes do commit mostra que não sobrou nada fora
da lista:

```powershell
cd C:\Users\angeloals\Documents\RECLin-PT
git status --short                          # não deve listar nada

Expand-Archive -Path "$env:USERPROFILE\Downloads\RECLin-PT-etapa-5-v1.zip" -DestinationPath . -Force

git status --short --untracked-files=all    # 10 linhas " M" e 19 linhas "??"

$modificados = @(
  "README.md",
  "codigo/README.md",
  "codigo/docs/entregas/README.md",
  "codigo/pyproject.toml",
  "codigo/requirements.txt",
  "codigo/reclin/tarefa.py",
  "codigo/testes/conftest.py",
  "codigo/testes/referencia/README.md",
  "codigo/testes/referencia/gerar_referencias.py",
  "codigo/testes/referencia/referencias.json"
)
$criados = @(
  "codigo/reclin/entrada.py",
  "codigo/reclin/modelos.py",
  "codigo/reclin/treino/__init__.py",
  "codigo/reclin/treino/reprodutibilidade.py",
  "codigo/reclin/treino/checkpoint.py",
  "codigo/reclin/treino/laco.py",
  "codigo/reclin/treino/classificador.py",
  "codigo/scripts/treinar.py",
  "codigo/scripts/avaliar_test.py",
  "codigo/testes/unidade/test_entrada.py",
  "codigo/testes/unidade/test_treino.py",
  "codigo/testes/integracao/test_treino_retomada.py",
  "codigo/testes/equivalencia/test_treino_legado.py",
  "codigo/testes/referencia/lotes.json",
  "codigo/testes/referencia/treino/baseline_pequeno_seed42.json",
  "codigo/testes/referencia/treino/baseline_pequeno_seed42.preds.json",
  "codigo/testes/referencia/treino/baseline_pequeno_seed42.dev_preds.json",
  "codigo/testes/referencia/treino/baseline_pequeno_seed42.registro.json",
  "codigo/docs/entregas/etapa-5.md"
)
git add -- $modificados $criados

git status --short --untracked-files=all    # 10 "M " e 19 "A ", e nenhuma outra linha
git diff --cached --stat                    # 29 files changed
git commit -m "Etapa 5 (v1): infraestrutura de treino (entrada, modelos, treino/)" `
  -m "entrada (janelas marcadas e loader), modelos (tokenizer com marcadores, classificador, gravação e recarga) e treino/ (reprodutibilidade, checkpoint, laço com retomada exata, classificador da tarefa). treinar.py não lê o TEST; avaliar_test.py avalia o TEST em separado." `
  -m "Equivalência com o legado: janelas e lotes idênticos; treino pequeno e completo do modelo minúsculo iguais bit a bit. Retomada exata (meio e fim de época) igual à execução contínua. Referências novas: lotes.json e treino/baseline_pequeno_seed42.*."

git log --oneline -n 3
git show --stat --format="%h %s" HEAD       # 29 files changed
git rev-parse "HEAD^{tree}"
git status --short                          # não deve listar nada
```

O hash do commit no seu computador será diferente do da origem (autor, data e
mensagem entram no hash). O que precisa coincidir é o **hash da árvore**
(`HEAD^{tree}`), que depende só do conteúdo: o valor esperado está na
mensagem de entrega (não pode constar neste arquivo, que faz parte da árvore).

## Arquivos criados e modificados nesta etapa

Criados (19):

```
codigo/reclin/entrada.py
codigo/reclin/modelos.py
codigo/reclin/treino/__init__.py
codigo/reclin/treino/reprodutibilidade.py
codigo/reclin/treino/checkpoint.py
codigo/reclin/treino/laco.py
codigo/reclin/treino/classificador.py
codigo/scripts/treinar.py
codigo/scripts/avaliar_test.py
codigo/testes/unidade/test_entrada.py
codigo/testes/unidade/test_treino.py
codigo/testes/integracao/test_treino_retomada.py
codigo/testes/equivalencia/test_treino_legado.py
codigo/testes/referencia/lotes.json
codigo/testes/referencia/treino/baseline_pequeno_seed42.json
codigo/testes/referencia/treino/baseline_pequeno_seed42.preds.json
codigo/testes/referencia/treino/baseline_pequeno_seed42.dev_preds.json
codigo/testes/referencia/treino/baseline_pequeno_seed42.registro.json
codigo/docs/entregas/etapa-5.md
```

Modificados (10):

| Arquivo | Mudança |
| --- | --- |
| `README.md` | Estado (etapas 1 a 5), dependências e os comandos de treino |
| `codigo/README.md` | Instalação com torch, treino, checkpoints, retomada, avaliação do TEST, reprodutibilidade, testes |
| `codigo/docs/entregas/README.md` | Linha da etapa 5; pesos e dependências na lista do que não vai no pacote |
| `codigo/pyproject.toml` | Dependências torch e transformers; descrição do marcador `lento` |
| `codigo/requirements.txt` | tokenizers 0.23.2 e safetensors 0.8.0 fixados; instrução de instalação |
| `codigo/reclin/tarefa.py` | `conjunto_referencia(..., conferir_tamanho=True)` |
| `codigo/testes/conftest.py` | Fixture `subconjunto` (partições pequenas com MANIFEST); pasta `integracao/` |
| `codigo/testes/referencia/README.md` | `lotes.json`, o treino pequeno e as receitas dos hashes |
| `codigo/testes/referencia/gerar_referencias.py` | Parte F (`--partes etapa5`): lotes e treino pequeno instrumentado |
| `codigo/testes/referencia/referencias.json` | SHA-256 dos 5 arquivos novos e o registro da parte F (as entradas anteriores não mudaram) |

Removidos: nenhum. As partições, os rótulos, o léxico congelado, as referências anteriores e as funções das etapas 1 a 4 não mudaram de comportamento.

## Limitações e pendências

- **Windows**: confirmar a instalação (com torch) e os testes no seu computador;
  os de reprodução numérica do legado serão pulados ali.
- **GPU**: a retomada e o determinismo em GPU não foram verificados (o
  caminho do estado da CUDA não é exercitado em CPU). A primeira execução em
  GPU (etapa 8) deve conferir a retomada também lá.
- **Tokenizers reais**: a equivalência dos lotes foi verificada com o
  tokenizer minúsculo; com BioBERTpt e BERTimbau, a tokenização é a da
  biblioteca e as janelas são idênticas, mas os tensores não foram comparados.
- **Backup no Hugging Face Hub** (`treino/backup_hf.py`): não implementado
  nesta etapa.
- **Etapa 6**: `estrategias/baseline.py`, `restrito.py` e `pair_aware/` (com o
  espaço restrito, o remapeamento e a cabeça), que vão usar este laço.
- Seguem em aberto, de etapas anteriores: `corpus.py` e a pasta `tcc/`.
