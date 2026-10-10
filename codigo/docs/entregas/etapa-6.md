# Entrega da etapa 6 — estratégias treinadas (baseline, restrito, Pair-Aware)

| | |
| --- | --- |
| Etapa | 6 |
| Versão | 1 |
| Data | 10/10/2026 |
| Pacote | `RECLin-PT-etapa-6-v1.zip` |
| Parte de | etapa 5, versão 1 (`RECLin-PT-etapa-5-v1.zip`) |

O pacote é o **projeto completo** até a etapa 6: extraído numa pasta limpa,
tem tudo o que é preciso para instalar, executar e testar o que já foi
implementado.

## Resultado dos critérios de aceitação

| Estratégia | Critério | Resultado | Evidência |
| --- | --- | --- | --- |
| `baseline` | A execução pequena reproduz as referências | **aprovado** | Pela estratégia `baseline`, o baseline pequeno do legado (subconjunto 30/10/10, 3 épocas): histórico, parâmetros, melhor época, métricas do TEST e os **dois sidecars byte a byte**. A estratégia é o classificador da etapa 5 com outro nome (mesmos bytes, testado), que já reproduzia a loss de cada passo, os pesos e o treino completo |
| `restrito` | Índices idênticos aos de referência | **aprovado** | Os índices do espaço restrito das 3 partições completas têm o SHA-256 de `dados.json` (6.805 / 774 / 800) e são **iguais, elemento a elemento**, aos `restricted_indices` das 5 execuções oficiais em GPU (DEV e TEST) e da Pair-Aware oficial (DEV); janelas restritas e resumo por partição iguais |
| `restrito` | A execução pequena corresponde às referências | **aprovado** | Restrito minúsculo, 10 épocas: histórico (com `dev_restrito_*`), parâmetros, melhor época (3), pesos `balanced` do TRAIN restrito, léxico, configuração, resumo do espaço, métricas do TEST no conjunto completo e no subconjunto e os **dois sidecars byte a byte** (com `restricted_indices` e `probs` remapeadas) |
| `pair_aware` | A execução pequena reproduz as referências | **aprovado** | Pair-Aware minúscula, 3 épocas: tudo o que vale para o restrito, mais a configuração da cabeça, o número de parâmetros da cabeça, os marcadores ausentes por partição (TRAIN, DEV e TEST) e as métricas do DEV da melhor época (completo e restrito); sidecars byte a byte |
| todas | Treino, checkpoints e retomada pela infraestrutura da etapa 5 | **aprovado em CPU** | Restrito e Pair-Aware interrompidos (meio e fim de época) e retomados: estado, melhor modelo (inclusive a cabeça) e predições iguais aos da execução contínua. Em GPU não verificado (como na etapa 5) |
| todas | Separação entre estratégias | **aprovado** | Retomar a execução de uma estratégia com outra, com outro léxico ou outra semente é recusado; a avaliação do TEST com a montagem de outra estratégia é recusada; uma configuração de outra estratégia é recusada |
| todas | TEST fora do treino e da seleção | **aprovado** | Restrito e Pair-Aware treinam com o arquivo do TEST ausente e leem só TRAIN e DEV; a contagem de marcadores do TEST da Pair-Aware passou para a avaliação do TEST |

Testes pedidos (seção 7 das instruções):

| # | Teste | Onde | Situação |
| --- | --- | --- | --- |
| 1 | Construção das entradas e compatibilidade com a etapa 5 | `unidade/test_estrategias_treinadas.py`, `integracao/test_estrategias_treinadas.py::test_baseline_e_o_classificador_da_etapa_5`, os testes das etapas 1 a 5 | aprovado |
| 2 | Configuração e execução mínima do baseline | `unidade/test_estrategias_treinadas.py`, `equivalencia/...::test_baseline_pequeno_pela_estrategia` | aprovado |
| 3 | Seleção e identidade dos índices do restrito | `equivalencia/test_estrategias_treinadas_legado.py` (17 testes), unidade | aprovado |
| 4 | Construção e comportamento dos pares da Pair-Aware | `unidade/test_estrategias_treinadas.py` (posições, leitura de `h_cls`/`h_E1`/`h_E2`, fallback, arquitetura, inicialização, gravação, marcadores ausentes) | aprovado |
| 5 | Reprodução das referências das três estratégias | `equivalencia/test_estrategias_treinadas_legado.py` | aprovado |
| 6 | Separação de configurações, sementes e checkpoints | `integracao/test_estrategias_treinadas.py` | aprovado |
| 7 | Retomada com a infraestrutura existente | `integracao/test_estrategias_treinadas.py` | aprovado (CPU) |
| 8 | Compatibilidade com a avaliação da etapa 2 | integração (`avaliar.py`, `comparar.py` sobre execuções do restrito e da Pair-Aware) | aprovado |
| 9 | Sem acesso ao TEST no treino e na seleção | integração (`test_treino_nao_le_o_test`) | aprovado |
| 10 | Comandos e configurações inválidas | integração (comandos de ponta a ponta, recusas) | aprovado |

## Diagnóstico (resumo)

- **Infraestrutura da etapa 5:** `entrada`, `modelos`, `treino/laco` (genérico:
  recebe modelo, loaders e `pontuar_dev`), `treino/checkpoint` e o orquestrador
  `treino/classificador`, que só montava o espaço completo com a cabeça [CLS].
- **Contratos:** o diretório e os sidecars da etapa 2; a época pelo
  `dev_macro_f1`; a ordem de consumo de RNG (sementes → dados → tokenizer →
  modelo → loaders); o léxico congelado; uma estratégia não importa outra; o
  TEST só em `avaliar_test`.
- **Comportamento do legado:**
  - `restrito` (`train_restrito.py`, `restricted_space.py`): pares com e1 no
    léxico (`min_freq=3`), na ordem do conjunto completo; pesos `balanced` do
    TRAIN restrito; 10 épocas; época pelo macro-F1 do DEV remapeado (fora do
    espaço: `no_relation`, `[0, 0, 1]`); histórico com `dev_restrito_*`;
    sidecars com `espaco`, `encoder`, `n_restrito`, `restricted_indices`.
  - `pair_aware` (`train_pair_aware.py`, `model.py`): o restrito com
    `AutoModel` sem pooler e a cabeça `[h_cls ; h_E1 ; h_E2] → Linear(3d,m) →
    GELU → Dropout → Linear(m,3)`; marcador = primeira ocorrência com máscara
    1, ausente → `h_cls`; cabeça inicializada com normal(0,
    `initializer_range`) depois do resize; marcadores ausentes contados antes
    do treino; sidecars com `cabeca: "pair-aware"`.
- **Não incluído (sem referências versionadas; é aplicação de protocolo, etapa
  8):** os critérios de parada e de decisão (`criterio_parada.py`,
  `avaliacao_final.py`, `criterio_parada_pair_aware.py`).

## Como cada estratégia funciona

| | `baseline` | `restrito` | `pair_aware` |
| --- | --- | --- | --- |
| Exemplos de TRAIN/DEV/TEST | todos os candidatos | só os pares cujo e1 é pista do léxico congelado (11 formas), na ordem do conjunto | os do restrito |
| Modelo | encoder + cabeça [CLS] | o do baseline | encoder sem pooler + cabeça Pair-Aware |
| Pesos de classe | `balanced` do TRAIN | `balanced` do TRAIN restrito | os do restrito |
| Épocas (padrão) | 3 (`Config`) | 10 (`ConfigRestrito`) | 10 (`ConfigPairAware`; difere do restrito só em `mlp_hidden` e `head_dropout`) |
| Seleção da época | macro-F1 do DEV | macro-F1 do DEV remapeado ao conjunto completo | como o restrito |
| Predições gravadas | conjunto completo | conjunto completo (fora do espaço: `no_relation`, `[0, 0, 1]`) + `restricted_indices` | como o restrito + `cabeca` |
| Nome da execução | `baseline_<enc>_seed<N>` | `restrito_<enc>_seed<N>` | `pairaware_<enc>_seed<N>` |

O que é comum fica no núcleo: a `treino.montagem.Montagem` descreve o que
muda (seleção, construtor do modelo, classe para recarregar, registro da
estratégia, extras dos sidecars, descrição medida antes do treino), e o
orquestrador `treino.classificador` faz o resto — com o mesmo laço, os mesmos
checkpoints, a mesma retomada e a mesma avaliação do TEST. O remapeamento ao
conjunto completo está em `execucao.subconjunto`. A seleção dos pares é
repetida em `restrito` e `pair_aware` (duas linhas), para que uma estratégia
não importe a outra.

## O que foi implementado

| Arquivo | O que é | Origem no legado |
| --- | --- | --- |
| `reclin/estrategias/baseline.py` | `montagem`, `nome_execucao` | `baseline_*.py`, `run()` |
| `reclin/estrategias/restrito.py` | `ConfigRestrito`, `selecionar`, `montagem`, `montagem_de_registro` | `train_restrito.py`, `restricted_space.construir_espaco` |
| `reclin/estrategias/pair_aware/__init__.py` | `ConfigPairAware`, `selecionar`, `criar_modelo`, `montagem` (com `descrever`: cabeça e marcadores ausentes) | `train_pair_aware.py` |
| `reclin/estrategias/pair_aware/cabeca.py` | `PairAwareClassifier`, `posicoes_dos_marcadores`, `contar_marcadores_ausentes` | `model.py` (sem mudança de comportamento) |
| `reclin/execucao/subconjunto.py` | `remapear`, `PROBS_FORA`, `resumo`, `conferir_indices` | `restricted_space.remapear`, `EspacoRestrito.resumo` |
| `reclin/treino/montagem.py` | `Montagem` e `CLASSIFICADOR` | — |

Mudanças em módulos anteriores (compatíveis; os testes das etapas 1 a 5
continuam passando sem alteração, exceto o de níveis, ajustado para pacotes):

| Arquivo | Mudança | Por quê |
| --- | --- | --- |
| `reclin/treino/classificador.py` | `treinar_execucao` e `avaliar_test` recebem `montagem` (padrão: o classificador da etapa 5); seleção, remapeamento, métricas do subconjunto e extras dos sidecars; `config.json` grava a estratégia e o registro dela | Uma só implementação do treino para as três estratégias |
| `reclin/entrada.py` | `Exemplos.subconjunto(indices)` | Exemplos do espaço restrito sem montar janelas de novo |
| `scripts/treinar.py` | `--estrategia` (padrão `baseline`), opções da configuração da estratégia escolhida, nome padrão da execução | Um só comando de treino |
| `scripts/avaliar_test.py` | Reconstrói a montagem a partir de `config.json` | O TEST com a estratégia que treinou a execução |
| `testes/unidade/test_pacote.py` | A verificação de níveis aceita estratégias em pacote (`estrategias/pair_aware/`) | Previsto no plano |

## Evidências

### Comparações com as referências

| Referência | Comparação | Resultado |
| --- | --- | --- |
| `dados.json`, espaço restrito (3 partições) | SHA-256 dos índices e das janelas, tamanho, resumo | iguais |
| `dados.json`, pesos `balanced` do restrito | os 3 pesos | iguais |
| 5 restritos oficiais (GPU): DEV e TEST; Pair-Aware oficial: DEV | `restricted_indices`, `n_restrito`, `y_true` | iguais, elemento a elemento |
| `treino/restrito_minusculo_seed42.*` | histórico, parâmetros, melhor época, pesos, léxico, configuração, resumo do espaço, métricas do TEST (completo e restrito), sidecars | iguais; sidecars byte a byte |
| `treino/pairaware_minusculo_seed42.*` | o mesmo + cabeça, parâmetros da cabeça, marcadores ausentes, métricas do DEV da melhor época (completo e restrito) | iguais; sidecars byte a byte |
| `treino/baseline_pequeno_seed42.*` (pela estratégia `baseline`) | histórico, parâmetros, melhor época, métricas, sidecars | iguais; sidecars byte a byte |

Diferenças encontradas: nenhuma nos números e nos sidecars. Diferenças de
formato, intencionais: o `<out>.json` do legado (que juntava treino e TEST)
virou `treino.json` + `metricas.json` (etapa 5), e a contagem de marcadores do
TEST da Pair-Aware é feita na avaliação do TEST (no legado, no treino).

### Os testes acusam desvios

Cada alteração foi aplicada ao código, os testes rodaram e o código foi
restaurado:

| Alteração | Resultado |
| --- | --- |
| Restrito seleciona pelo e2 em vez do e1 | falha (índices) |
| Probabilidades fora do espaço `[0, 0.1, 0.9]` em vez de `[0, 0, 1]` | falha (sidecars byte a byte) |
| Pesos `balanced` do espaço completo no restrito | falha (reprodução) |
| Cabeça Pair-Aware inicializada antes do resize dos embeddings | falha (reprodução) |
| Extras do restrito em outra ordem | falha (formato do sidecar) |
| Época escolhida pelo macro-F1 do DEV restrito, não do remapeado | falha (reprodução) |

## Testes

| Tipo | Onde | Quantidade | Depende de |
| --- | --- | --- | --- |
| Unidade | `testes/unidade/` | 218 (novos: 8 de `test_subconjunto.py`, 19 de `test_estrategias_treinadas.py`, 6 da verificação de níveis para os módulos novos) | nada externo |
| Integração | `testes/integracao/` | 43 (novos: 22 de `test_estrategias_treinadas.py`) | modelo minúsculo, partições pequenas e completas |
| Equivalência com as referências | `testes/equivalencia/` | 226 (novos: 21 de `test_estrategias_treinadas_legado.py`) | só os arquivos versionados em `testes/referencia/` |

Total: 487 testes, 28 marcados `lento` (um novo: o restrito minúsculo de 10
épocas). Pulado conforme o ambiente: o treino completo do baseline minúsculo
(sem `RECLIN_TREINO_COMPLETO=1`) e, fora do ambiente das referências, os de
reprodução numérica do legado (agora 10: os 7 da etapa 5 e os 3 das
estratégias). Nenhum acessa a rede, uma GPU ou o legado.

Nenhuma referência foi criada ou alterada nesta etapa: os testes usam as que
já estavam versionadas desde a etapa 0 (`dados.json`, as predições oficiais em
GPU do restrito e da Pair-Aware, `treino/restrito_minusculo_seed42.*`,
`treino/pairaware_minusculo_seed42.*`) e a da etapa 5
(`treino/baseline_pequeno_seed42.*`); `referencias.json` confere os hashes de
todas.

### Resultados obtidos

Ambiente: Linux (contêiner de nuvem, 2 núcleos, sem GPU). O pacote foi extraído
numa pasta vazia, comparado byte a byte com o projeto (185 arquivos, todos
iguais) e instalado em ambientes novos (torch 2.11.0 do PyPI, sem uso de GPU):

| Python | numpy | scipy | torch / transformers / tokenizers | Versões | Suíte completa |
| --- | --- | --- | --- | --- | --- |
| 3.10.20 | 2.2.6 | 1.15.3 | 2.11.0 / 5.16.1 / 0.23.2 | fixadas, exceto numpy e scipy (inexistentes para 3.10) | 486 passaram, 1 pulado |
| 3.12.3 | 2.1.3 | 1.16.3 | 2.11.0 / 5.16.1 / 0.23.2 | fixadas (`-c codigo/requirements.txt`) | 486 passaram, 1 pulado |
| 3.13.16 | 2.1.3 | 1.16.3 | 2.11.0 / 5.16.1 / 0.23.2 | fixadas | 486 passaram, 1 pulado |

O pulado nos três é o treino completo do baseline minúsculo (opcional, sem
`RECLIN_TREINO_COMPLETO=1`). Como o orquestrador do treino mudou nesta etapa,
ele foi rodado à parte, com `RECLIN_TREINO_COMPLETO=1`, sobre o projeto com o
código desta entrega (Python 3.13): passou, em 10 min 54 s. Os testes de
reprodução numérica do legado (os 10) rodaram nos três ambientes (todos Linux
x86_64 com torch 2.11.0).

Em cada ambiente também: importação de todos os 34 módulos (6 novos), `--help`
dos 9 scripts e `preparar_dados.py conferir` (3 partições conferem). A suíte
completa levou de 12 min 24 s a 12 min 59 s por ambiente.

**Não executado:** nada foi rodado no **Windows** nem em **GPU**; nenhum
encoder real foi baixado.

## Como aplicar

Numa pasta limpa, no PowerShell:

```powershell
$zip     = "$env:USERPROFILE\Downloads\RECLin-PT-etapa-6-v1.zip"   # ajuste se salvou em outro lugar
$destino = "C:\Users\angeloals\Documents\RECLin-PT"

if (Test-Path $destino) { Rename-Item $destino "RECLin-PT-antes-etapa-6" }   # nada é apagado
Expand-Archive -Path $zip -DestinationPath $destino

cd $destino
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\codigo[testes]" -c codigo\requirements.txt
.\.venv\Scripts\python.exe codigo\scripts\preparar_dados.py conferir
.\.venv\Scripts\python.exe -m pytest codigo -rs
```

No Windows, os testes de reprodução numérica do legado são pulados com o
motivo (as referências foram geradas em Linux); os de índices, janelas,
retomada, separação e comandos rodam.

Demonstração das três estratégias em CPU com o modelo minúsculo (partições
completas, 1 época cada; cerca de 3 min o restrito e a Pair-Aware, mais de 10 min
o baseline):

```powershell
$m = "codigo\testes\referencia\modelo_minusculo"
foreach ($e in "baseline", "restrito", "pair_aware") {
  .\.venv\Scripts\python.exe codigo\scripts\treinar.py --estrategia $e --saida reproducao --modelo $m --epochs 1 --lr 1e-3
}
Get-ChildItem reproducao    # baseline_biobertpt_seed42, restrito_biobertpt_seed42, pairaware_biobertpt_seed42
.\.venv\Scripts\python.exe codigo\scripts\avaliar_test.py --execucao reproducao\pairaware_biobertpt_seed42
```

O esperado (verificado aqui, em Linux, com as partições completas): o
restrito e a Pair-Aware levam cerca de 3 min cada (107 passos); o baseline,
2.386 passos, leva mais de 10 min. Cada um termina com código 0 e "Melhor
época: 1 (dev_macro_f1=...)"; no restrito e na Pair-Aware, `metricas.json`
tem `dev` e `dev_restrito` e, depois de `avaliar_test.py`, `test` e
`test_restrito`; o `treino.json` da Pair-Aware tem, em `descricao`, a cabeça
(835 parâmetros no modelo minúsculo) e os marcadores ausentes do TRAIN (1.264
de 6.805) e do DEV (141 de 774), e em `descricao_test` os do TEST (149 de
800). O modelo minúsculo é aleatório: com 1 época, as três estratégias preveem
só `no_relation` (F1 de `negation_of` 0); os números mostram que o fluxo
funciona, não o desempenho. A pasta `reproducao\` pode ser apagada depois.

### Registrar no git só as alterações da etapa 6

A partir de um repositório que já tem o commit da etapa 5, com a cópia de
trabalho limpa. A etapa 6 não remove nenhum arquivo, então extrair por cima
equivale a extrair numa pasta limpa. Esperado: **10 modificados, 11
adicionados, 0 removidos**. Os arquivos são preparados pelo nome (sem
`git add -A`), e a conferência antes do commit mostra que não sobrou nada fora
da lista:

```powershell
cd C:\Users\angeloals\Documents\RECLin-PT
git status --short                          # não deve listar nada
git log --oneline -n 1                      # o commit da etapa 5

Expand-Archive -Path "$env:USERPROFILE\Downloads\RECLin-PT-etapa-6-v1.zip" -DestinationPath . -Force

git status --short --untracked-files=all    # 10 linhas " M" e 11 linhas "??"
git diff --stat                             # só os 10 modificados

$modificados = @(
  "README.md",
  "codigo/README.md",
  "codigo/docs/entregas/README.md",
  "codigo/reclin/entrada.py",
  "codigo/reclin/estrategias/__init__.py",
  "codigo/reclin/treino/__init__.py",
  "codigo/reclin/treino/classificador.py",
  "codigo/scripts/avaliar_test.py",
  "codigo/scripts/treinar.py",
  "codigo/testes/unidade/test_pacote.py"
)
$criados = @(
  "codigo/reclin/execucao/subconjunto.py",
  "codigo/reclin/treino/montagem.py",
  "codigo/reclin/estrategias/baseline.py",
  "codigo/reclin/estrategias/restrito.py",
  "codigo/reclin/estrategias/pair_aware/__init__.py",
  "codigo/reclin/estrategias/pair_aware/cabeca.py",
  "codigo/testes/unidade/test_subconjunto.py",
  "codigo/testes/unidade/test_estrategias_treinadas.py",
  "codigo/testes/integracao/test_estrategias_treinadas.py",
  "codigo/testes/equivalencia/test_estrategias_treinadas_legado.py",
  "codigo/docs/entregas/etapa-6.md"
)
git add -- $modificados $criados

git status --short --untracked-files=all    # 10 "M " e 11 "A ", e nenhuma outra linha
git diff --cached --stat                    # 21 files changed
git commit -m "Etapa 6 (v1): estratégias treinadas (baseline, restrito, Pair-Aware)" `
  -m "estrategias/baseline, estrategias/restrito e estrategias/pair_aware sobre a infraestrutura de treino da etapa 5 (treino/montagem, execucao/subconjunto). treinar.py --estrategia; avaliar_test.py reconstrói a estratégia pelo config.json. Retomada e avaliação do TEST recusam a execução de outra estratégia." `
  -m "Equivalência com o legado: índices do espaço restrito idênticos aos das execuções oficiais; restrito e Pair-Aware minúsculos e baseline pequeno com sidecars iguais byte a byte. Referências não alteradas."

git log --oneline -n 3
git show --stat --format="%h %s" HEAD       # 21 files changed
git rev-parse "HEAD^{tree}"
git status --short                          # não deve listar nada
```

O hash do commit no seu computador será diferente do da origem (autor, data e
mensagem entram no hash). O que precisa coincidir é o **hash da árvore**
(`HEAD^{tree}`), que depende só do conteúdo: o valor esperado está na
mensagem de entrega (não pode constar neste arquivo, que faz parte da árvore).

## Arquivos criados e modificados nesta etapa

Criados (11):

```
codigo/reclin/execucao/subconjunto.py
codigo/reclin/treino/montagem.py
codigo/reclin/estrategias/baseline.py
codigo/reclin/estrategias/restrito.py
codigo/reclin/estrategias/pair_aware/__init__.py
codigo/reclin/estrategias/pair_aware/cabeca.py
codigo/testes/unidade/test_subconjunto.py
codigo/testes/unidade/test_estrategias_treinadas.py
codigo/testes/integracao/test_estrategias_treinadas.py
codigo/testes/equivalencia/test_estrategias_treinadas_legado.py
codigo/docs/entregas/etapa-6.md
```

Modificados (10):

| Arquivo | Mudança |
| --- | --- |
| `README.md` | Estado (etapas 1 a 6) e os comandos das três estratégias |
| `codigo/README.md` | Seção das estratégias treinadas, recusa de retomada, tabela de testes, o que já existe |
| `codigo/docs/entregas/README.md` | Etapa 5 marcada como substituída; linha da etapa 6 |
| `codigo/reclin/entrada.py` | `Exemplos.subconjunto(indices)` |
| `codigo/reclin/estrategias/__init__.py` | Descrição do pacote com as estratégias treinadas |
| `codigo/reclin/treino/__init__.py` | Descrição com `montagem` |
| `codigo/reclin/treino/classificador.py` | `treinar_execucao` e `avaliar_test` recebem a `Montagem` (padrão: o classificador da etapa 5) |
| `codigo/scripts/avaliar_test.py` | Montagem reconstruída a partir de `config.json` |
| `codigo/scripts/treinar.py` | `--estrategia`, opções da configuração da estratégia, nome padrão |
| `codigo/testes/unidade/test_pacote.py` | Verificação de níveis aceita estratégias em pacote |

Removidos: nenhum. As partições, os rótulos, o léxico congelado, as regras, as
referências e o comportamento das etapas 1 a 5 não mudaram (o classificador
com a montagem padrão grava os mesmos sidecars da etapa 5, e os testes dela
passam sem alteração).

## Limitações e pendências

- **Windows e GPU**: não executados. A retomada e o determinismo em GPU seguem
  não verificados (pendência da etapa 5, não alterada aqui); os encoders reais
  (BioBERTpt, BERTimbau) não foram baixados.
- **Critérios de parada e de decisão** do restrito e da Pair-Aware
  (`criterio_parada.py`, `avaliacao_final.py`, `criterio_parada_pair_aware.py`
  e o `avaliacao/decisao` do plano): não implementados — não há referências
  versionadas dos relatórios deles, e aplicá-los é o protocolo da etapa 8.
- **Backup no Hugging Face Hub**: segue fora (etapa 5).
- **Comparabilidade**: no restrito e na Pair-Aware, só o F1 de `negation_of` é
  comparável com o baseline (macro-F1, MCC e McNemar não), como no legado.
- Seguem em aberto, de etapas anteriores: `corpus.py` e a pasta `tcc/`.
