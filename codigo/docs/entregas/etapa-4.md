# Entrega da etapa 4 — estratégias sem GPU (filtro de pistas e regra pura)

| | |
| --- | --- |
| Etapa | 4 |
| Versão | 1 |
| Data | 09/10/2026 |
| Pacote | `RECLin-PT-etapa-4-v1.zip` |
| Parte de | etapa 3, versão 1 (`RECLin-PT-etapa-3-v1.zip`) |

O pacote é o **projeto completo** até a etapa 4: extraído numa pasta limpa,
tem tudo o que é preciso para instalar, executar e testar o que já foi
implementado.

## Resultado dos critérios de aceitação

| # | Critério | Obtido | Situação |
| --- | --- | --- | --- |
| — | `filtro_*.preds.json` bit a bit | 4 de 4 idênticos | atendido |
| — | `regra_pura.preds.json` bit a bit | idêntico | atendido |
| — | `CALIBRACAO_filtro.json` bit a bit | idêntico | atendido |
| 1 | Referências, entradas, parâmetros e regras identificados | ver "Como o legado produziu cada artefato" | atendido |
| 2 | Regras reproduzidas do comportamento histórico, sem ajuste | as regras e os desempates do legado, sem mudança. A única escolha nova, a média com `math.fsum`, não muda nenhuma decisão (ver "Decisões") | atendido |
| 3 | Ordenação, tipos, serialização e precisão preservados | ordem das chaves, ids inteiros, `seed: null`, JSON compacto sem quebra final (sidecars), `indent=2` + chaves ordenadas sem quebra final (calibração) | atendido |
| 4 | Testes que comparam bytes, além dos testes unitários das regras | 19 de equivalência (bytes dos 6 arquivos pelos scripts e pelos módulos; varredura contra o relatório do legado) e 18 de unidade | atendido |
| 5 | Parâmetros de calibração, dados usados e diferenças entre as estratégias documentados | seções abaixo | atendido |
| — | TEST não usado para calibrar | a calibração roda com uma pasta de partições **sem** `test.jsonl` e reproduz o arquivo | atendido |
| — | Partições, rótulos, léxico da etapa 3 e critérios de avaliação inalterados | MANIFEST conferido; léxico congelado (`70c93fa807de`) usado e conferido; nenhum teste anterior mudou | atendido |

Os seis arquivos reproduzidos (SHA-256 iguais aos da cópia do legado em
`codigo/testes/referencia/resultados_legado/`):

| Arquivo | SHA-256 |
| --- | --- |
| `CALIBRACAO_filtro.json` | `c790f8ddd2289b623a46a97782b4c5350b46c9c38c82e8734c3f38cbc4616fb1` |
| `filtro_biobertpt_seed42.preds.json` | `8c51da839870cf41b3ff33a0b882a3940580f2d6eddd9ed0435f31bbee656b08` |
| `filtro_bertimbau_seed42.preds.json` | `b60ab84b50a647ffc040707ba0101506d5c2419b289e65ff704fd6189991e906` |
| `filtro_biobertpt_seed43.preds.json` | `9dda88c9924c39a7418161f01db3c70e9730508090259a9b7591a6cb76574ed6` |
| `filtro_bertimbau_seed43.preds.json` | `60fc884d76bfa067ffbc348264cf09a1e3af9f6f8f0e4fd05f904493c26b7786` |
| `regra_pura.preds.json` | `2721f9a5d05c33e18dbe34a23bb1fee5137002619bf4bc17ec22744709cbf499` |

## Como o legado produziu cada artefato (commit `a5f055c`)

| Artefato | Script (data) | Entradas | Regra e parâmetros |
| --- | --- | --- | --- |
| `CALIBRACAO_filtro.json` | `scripts/calibrate_cue_filter.py` (20/09/2026) | `data/splits/dev.jsonl`; as predições do DEV dos 4 baselines (`baseline_{biobertpt,bertimbau}_seed{42,43}.dev_preds.json`, melhor época); `train.jsonl` só para o léxico. **Não lê o TEST** | Três varreduras, abaixo. Gravado com `json.dumps(..., ensure_ascii=False, indent=2, sort_keys=True)`, sem quebra de linha final |
| `filtro_<enc>_seed<N>.preds.json` (4) | `scripts/run_fase2_test.py` (`apply_cue_filter` de `src/negation_lexicon.py`) | `baseline_<enc>_seed<N>.preds.json` (TEST), `test.jsonl`, léxico do TRAIN com o `min_freq` da calibração | Rebaixa `negation_of` → `no_relation` quando o e1 não é pista. Chaves: `model` = `filtro(<checkpoint> s<semente>)`, `seed`, `labels`, `y_true`, `y_pred`, `base_preds`, `postproc` (`kind`, `min_freq`, `lexicon_size`, `max_gap`, `demote_to`, `calibrated_on`). JSON compacto, sem quebra final |
| `regra_pura.preds.json` | `scripts/run_fase2_test.py` (`predict_rule` de `scripts/make_rule_baseline.py`) | `test.jsonl`, léxico do TRAIN com o `min_freq` da calibração, `rule` e `rule_gap` da calibração | R3 com `gap <= 1`. Chaves: `model` = `regra R3 (gap<=1, min_freq=3)`, `seed: null`, `labels`, `y_true`, `y_pred`, `postproc` (`kind`, `rule`, `max_target_gap`, `min_freq`, `lexicon_size`, `max_gap`, `calibrated_on`). JSON compacto, sem quebra final |

O mesmo `run_fase2_test.py` também acrescentou uma linha de avaliação do TEST
às trilhas (`regra_pura.test_evals.jsonl` e as dos baselines) e gravou
`FASE2_test_summary.json`. Isso é avaliação do TEST, não geração de
predições, e fica fora desta etapa (ver "Limitações").

### O protocolo da calibração (o critério, fixado no legado antes de rodar)

| # | O que se escolhe | Valores varridos | Critério | Desempate | Escolhido |
| --- | --- | --- | --- | --- | --- |
| 1 | `min_freq` do léxico | 1, 2, 3, 5, 10 | maior **média** do F1 de `negation_of` do filtro sobre as 4 execuções de base, no DEV | menor `min_freq` | **3** (11 formas) |
| 2 | Regra pura e limiar | R1 e R2 (com a janela, 25); R3 e R4 com gap 0, 1, 2, 3, 5, 10, 25 | maior F1 de `negation_of` no DEV, com o léxico de (1) | regra de menor índice, depois menor gap | **R3, gap ≤ 1** |
| 3 | Porta de gap depois do filtro | 0, 1, 2, 3, 5, 10, 25 | maior média do F1 sobre as 4 execuções, com o léxico de (1) | menor gap | **25** (porta desligada) |

O F1 do critério é o de `make_rule_baseline.score`: 2·P·R/(P+R). Dados usados:
19.064 candidatos do DEV (150 `negation_of`), `max_gap = 25`; o léxico vem dos
800 documentos do TRAIN. As margens entre o escolhido e o segundo colocado
são de 0,0039 (`min_freq` 3 contra 10), 0,0735 (R3 com gap 1 contra R4 com
gap 1) e 0,0106 (porta 25 contra 10): nenhuma escolha está perto de um empate.

O resultado é o mesmo léxico congelado na etapa 3: `min_freq=3`, 11 formas,
`lexico_sha1=70c93fa807de`.

## Diferenças entre as duas estratégias

| | Filtro de pistas (`filtro_pistas`) | Regra pura (`regra_pura`) |
| --- | --- | --- |
| Precisa de modelo | Sim: pós-processa as predições de uma execução de base (os baselines) | Não: só o léxico e a distância |
| O que prevê | As predições da base, com os `negation_of` sem pista rebaixados para `no_relation`; `associated_with` fica como a base previu | `negation_of` quando o e1 é pista e o par passa na regra; o resto é `no_relation`. Nunca prevê `associated_with` |
| Usa a distância | Não (a porta de gap foi testada e não adotada) | Sim: R3 com `entity_gap <= 1` |
| Semente | A da base | Nenhuma (`seed: null`) |
| Parâmetros calibrados no DEV | `min_freq` (e a porta de gap, desligada) | Regra e limiar, com o léxico de `min_freq` calibrado |
| Execuções | Uma por base: `filtro_<encoder>_seed<N>` | Uma: `regra_pura` |
| Teto do recall de `negation_of` | A cobertura do léxico e o recall da base | A cobertura do léxico |

As duas usam o léxico congelado (especialização, etapa 3) e nenhuma importa a
outra. O vínculo entre elas é histórico e de parâmetro, não de código: o
`min_freq` da regra pura é o que a calibração do filtro escolheu, e o léxico
resultante é o congelado. Só o script `calibrar.py`, que reproduz o
procedimento único do legado, chama as duas.

TEST, para contexto (as métricas não são critério desta etapa; os números são
os do legado, recalculados pelo `avaliar.py`): a regra pura tem F1 de
`negation_of` 0,6944 (TP 117, FP 68, FN 35); os quatro filtros ficam entre
0,8168 e 0,8509, contra 0,6961 a 0,7388 dos baselines que eles filtram
(`FASE2_test_summary.json`).

## O que foi implementado

### `reclin/estrategias/`

| Arquivo | Função ou constante | O que faz | Origem no legado |
| --- | --- | --- | --- |
| `__init__.py` | — | Documenta o nível: estratégias combinam núcleo e especialização e nunca importam umas às outras | — |
| `filtro_pistas.py` | `filtrar` | O filtro | `apply_cue_filter` |
| | `porta_gap` | A variante testada na calibração | `gap_gate` |
| | `montar_predicoes` | Sidecar do filtro (formato e chaves do legado) | `run_fase2_test.py` |
| | `nome_execucao` | `baseline_x_seedN` → `filtro_x_seedN` | nomes de `results/` |
| | `calibrar_min_freq`, `calibrar_porta_gap` | Varreduras 1 e 3, com a varredura completa no resultado | `calibrate_cue_filter.py` |
| | `MIN_FREQS`, `GAPS_PORTA` | Os valores varridos | idem |
| `regra_pura.py` | `prever` | R1 a R4, com o desempate de "mais próximo" (gap, `e2.start`, `int(e2.id)`) | `predict_rule` |
| | `montar_predicoes`, `nome_modelo` | Sidecar da regra | `run_fase2_test.py` |
| | `calibrar` | Varredura 2 | `calibrate_cue_filter.py` |
| | `REGRAS`, `GAPS` | As variantes e os limiares | `RULES`, `GAPS` |

As estratégias não leem arquivos: recebem candidatos, predições e o léxico.
Quem lê as partições, e portanto decide que a calibração só vê o DEV, são os
scripts.

### Núcleo (acréscimos compatíveis)

| Arquivo | Mudança |
| --- | --- |
| `reclin/avaliacao/metricas.py` | `f1_pr` e `resumo_alvo_pr`: o `score` do legado (F1 = 2·P·R/(P+R)), usado só no critério das calibrações. As funções existentes não mudaram |
| `reclin/particoes.py` | `conferir_particoes(pasta, nomes=PARTICOES)`: o argumento novo permite conferir só as partições que serão lidas; sem ele, o comportamento é o anterior |

### Scripts

| Script | O que faz |
| --- | --- |
| `scripts/calibrar.py` | Calibração no DEV: grava `CALIBRACAO_filtro.json` (formato do legado), mostra as três varreduras e, com `--detalhes`, grava a varredura completa em JSON. Confere e lê só `train` e `dev`. `--conferir ARQUIVO` compara bytes; `--sobrescrever` para regravar |
| `scripts/filtro_pistas.py` | Cria as execuções `filtro_<encoder>_seed<N>` com as predições filtradas do DEV e do TEST (`--particao` escolhe). As predições da base vêm de `--resultados` (formato novo ou do legado). `--conferir PASTA` compara bytes com a mesma execução em PASTA |
| `scripts/regra_pura.py` | Cria a execução `regra_pura` com as predições do DEV e do TEST. `--conferir PASTA` idem |

Os dois scripts das estratégias não escolhem nada: leem a calibração, usam o
léxico congelado e **falham** se a calibração tiver outro `min_freq` ou outro
tamanho de léxico (uma recalibração precisa atualizar o registro
`CONGELADO`). Gerar as predições do TEST não é avaliá-lo: a avaliação é o
`avaliar.py`, e o registro na trilha fica para a etapa 5.

### Formato das execuções

O de `reclin.execucao` (etapa 2), sem mudança:

```
<saida>/filtro_biobertpt_seed42/
    config.json           estratégia "filtro_pistas"; base, min_freq, max_gap, n_formas,
                          lexico_sha1, a calibração; identidade dos conjuntos do DEV e do TEST
    predicoes_dev.json    sidecar do DEV (base_preds: "baseline_biobertpt_seed42.dev_preds.json")
    predicoes_test.json   sidecar do TEST — idêntico a filtro_biobertpt_seed42.preds.json
<saida>/regra_pura/
    config.json           estratégia "regra_pura"; regra, max_target_gap, min_freq, ...
    predicoes_dev.json
    predicoes_test.json   idêntico a regra_pura.preds.json
```

`base_preds` é o caminho das predições da base relativo a `--resultados`:
para o legado, o nome do arquivo (como no legado); para uma execução nova,
`<base>/predicoes_<partição>.json`.

### Referências

Acrescentado `codigo/testes/referencia/resultados_legado/CALIBRACAO_filtro.md`,
copiado sem alteração pelo gerador (`--partes resultados`), a partir de um
checkout separado do commit `a5f055c`. É o único registro do legado com a
varredura completa da calibração (com 4 casas decimais). Os 73 arquivos de
referência anteriores e os seus SHA-256 não mudaram; `referencias.json` só
ganhou a linha do arquivo novo e a contagem (74).

## Decisões e divergências

- **Fórmula do F1 no critério.** A calibração do legado usava 2·P·R/(P+R),
  que difere de 2·TP/(2·TP+FP+FN) na última casa decimal em 12 dos 20 F1 da
  varredura de `min_freq`. As escolhas seriam as mesmas com qualquer das duas,
  mas o relatório do legado confirma a fórmula: para BERTimbau s42 com
  `min_freq=5`, 2·P·R/(P+R) = 0.7687499999999999 (relatório: 0,7687), e por
  contagens daria 0.76875 (0,7688). A fórmula do legado foi mantida em
  `metricas.resumo_alvo_pr`, usada só aqui; as métricas de avaliação
  continuam as da etapa 2.
- **Média entre as execuções.** O legado fazia `sum(...) / 4`, executado no
  Python 3.13, cujo `sum()` de floats é compensado (no 3.10 não é). O código
  novo usa `math.fsum`, que dá o mesmo resultado em qualquer versão. As médias
  de 4 casas do relatório conferem, e as margens acima (≥ 0,0039) estão
  muito acima de qualquer diferença de arredondamento. Um teste confirma as
  margens.
- **TEST.** O protocolo documentado (`CALIBRACAO_filtro.md`) é só DEV. Não há
  divergência a sinalizar. O relatório registra que uma análise exploratória
  anterior, feita no TEST em 09/09, preferia `min_freq=10`. Essa análise não
  entrou na escolha, e o código novo não a reproduz.
- **Execuções do DEV do filtro e da regra.** São geradas (e avaliáveis), mas o
  legado não versionou as equivalentes (`results/filtro_dev/`), então não há
  referência byte a byte para elas. Os F1 do filtro no DEV conferem, a 4
  casas, com a coluna `min_freq = 3` do relatório da calibração; os da regra
  (R3, gap ≤ 1: TP 112, FP 61, FN 38) conferem com a linha da regra.
- **Sem variantes por linha de comando.** Os scripts aplicam só a configuração
  calibrada. Outras regras e limiares estão nas funções (`regra_pura.prever`)
  e na varredura de `calibrar.py --detalhes`.

## Evidências

### Comparação bit a bit

| Verificação | Como | Resultado |
| --- | --- | --- |
| `CALIBRACAO_filtro.json` | `calibrar.py` com uma pasta de partições sem `test.jsonl`, `--conferir` | idêntico |
| 4 `filtro_*.preds.json` | `filtro_pistas.py --conferir` (o `predicoes_test.json` de cada execução) e `filtro_pistas.montar_predicoes` + `predicoes.gravar` | idênticos, pelos dois caminhos |
| `regra_pura.preds.json` | `regra_pura.py --conferir` e `regra_pura.montar_predicoes` + `predicoes.gravar` | idêntico, pelos dois caminhos |
| Varredura da calibração | as três tabelas e a lista de formas de `CALIBRACAO_filtro.md`, célula a célula (5 + 16 + 7 linhas) | iguais |

### Os testes acusam desvios

Cada alteração foi aplicada ao código, a suíte da etapa rodou e o código foi
restaurado:

| Alteração no código | Resultado |
| --- | --- |
| Filtro olha o e2 em vez do e1 | unidade e equivalência falham |
| Limiar da regra estrito (`>=` em vez de `>`) | unidade e equivalência falham |
| "Mais próximo" pelo maior `e2.start` | unidade e equivalência (tabela da regra) falham |
| "Mais próximo" com `e2.id` como texto | só a unidade falha (no SemClinBr nenhum empate chega ao id) |
| R1 e R2 varridos em todos os limiares | unidade e equivalência falham |
| Desempate de `min_freq`, da regra ou da porta invertido | só a unidade falha (no DEV não há empate no topo) |
| F1 por contagens no critério de `min_freq` | equivalência falha (a tabela do relatório, ver acima) |
| Chaves de `postproc` em outra ordem | equivalência falha (bytes) |
| Quebra de linha no fim da calibração | equivalência falha (bytes) |
| Média somada em outra ordem | nenhum teste falha: não muda nenhuma escolha (ver "Decisões") |

### Arquitetura

`test_pacote.py` ganhou uma verificação automática dos níveis, sobre todos os
módulos do pacote: o núcleo não importa `negacao` nem `estrategias`;
`negacao` não importa `estrategias`; uma estratégia não importa outra.

## Testes

| Tipo | Onde | Quantidade | Depende de |
| --- | --- | --- | --- |
| Unidade | `testes/unidade/` | 150 (novos: 18 de `test_estrategias.py`, 3 do `--help` dos scripts novos, 22 da verificação de níveis) | nada externo |
| Equivalência com as referências | `testes/equivalencia/` | 187 (novos: 19 de `test_estrategias_legado.py`; 26 marcados `lento`, da etapa 2) | só os arquivos versionados em `testes/referencia/` |
| Execuções com recurso externo | `testes/referencia/gerar_referencias.py` | — | um checkout do legado; fora da suíte |

Total: 337 testes. Nenhum acessa o legado, a rede ou uma GPU.

### Resultados obtidos

Ambiente: Linux (contêiner de nuvem). O pacote foi extraído numa pasta vazia,
comparado byte a byte com o projeto e instalado em ambientes novos:

| Python | numpy | scipy | Versões | Suíte completa |
| --- | --- | --- | --- | --- |
| 3.10.20 | 2.2.6 | 1.15.3 | escolhidas pelo pip (as fixadas não existem para 3.10) | 337 passaram |
| 3.12.3 | 2.1.3 | 1.16.3 | fixadas (`-c codigo/requirements.txt`) | 337 passaram |
| 3.13.16 | 2.1.3 | 1.16.3 | fixadas | 337 passaram |

Em cada ambiente também: importação de todos os módulos, `--help` dos 7
scripts e `preparar_dados.py conferir` (3 partições conferem). A integridade
das 74 referências do legado é conferida pelo `test_pacote.py`. Os três
comandos de reprodução de "Como aplicar" foram rodados, por fora da suíte,
numa extração limpa do pacote, com Python 3.10 e 3.13: os seis arquivos
saíram idênticos nos dois.

**Não executado:** nada foi rodado no **Windows**.

## Como aplicar

Numa pasta limpa, no PowerShell:

```powershell
$zip     = "$env:USERPROFILE\Downloads\RECLin-PT-etapa-4-v1.zip"   # ajuste se salvou em outro lugar
$destino = "C:\Users\angeloals\Documents\RECLin-PT"

if (Test-Path $destino) { Rename-Item $destino "RECLin-PT-antes-etapa-4" }   # nada é apagado
Expand-Archive -Path $zip -DestinationPath $destino

cd $destino
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\codigo[testes]" -c codigo\requirements.txt
.\.venv\Scripts\python.exe codigo\scripts\preparar_dados.py conferir
.\.venv\Scripts\python.exe -m pytest codigo
```

Reprodução bit a bit dos três grupos de artefatos (cada comando termina com
"idêntico"/"idêntica" e código de saída 0; com diferença, sai com 1):

```powershell
$ref = "codigo\testes\referencia\resultados_legado"
.\.venv\Scripts\python.exe codigo\scripts\calibrar.py --resultados $ref `
    --saida reproducao\CALIBRACAO_filtro.json --detalhes reproducao\calibracao_detalhes.json `
    --conferir $ref\CALIBRACAO_filtro.json
.\.venv\Scripts\python.exe codigo\scripts\filtro_pistas.py --resultados $ref `
    --calibracao $ref\CALIBRACAO_filtro.json --saida reproducao\execucoes --conferir $ref
.\.venv\Scripts\python.exe codigo\scripts\regra_pura.py `
    --calibracao $ref\CALIBRACAO_filtro.json --saida reproducao\execucoes --conferir $ref
.\.venv\Scripts\python.exe codigo\scripts\avaliar.py --execucao reproducao\execucoes\regra_pura
```

O esperado: `calibrar.py` com "Escolhido min_freq=3 ... é o léxico
congelado", "Escolhida a regra R3 com gap<=1", "Escolhida a porta gap<=25" e
"A calibração gravada é idêntica, byte a byte"; `filtro_pistas.py` com
"Conferência: 4 sidecars comparados, 4 idênticos" (o DEV aparece como "sem
referência", porque o legado não versionou o DEV do filtro); `regra_pura.py`
com "Conferência: 1 sidecars comparados, 1 idênticos"; o pytest com
`337 passed` (cerca de 2 minutos). A pasta `reproducao\` pode ser apagada
depois. Ela não faz parte do projeto (`git status` a mostra como não
rastreada). Com Python 3.10, rode o `pip install` sem o
`-c codigo\requirements.txt`.

### Registrar no git só as alterações da etapa 4

A partir de um repositório que já tem o commit da etapa 3, com a cópia de
trabalho limpa. A etapa 4 não remove nenhum arquivo, então extrair por cima
equivale a extrair numa pasta limpa:

```powershell
cd C:\Users\angeloals\Documents\RECLin-PT
git status --short          # não deve listar nada

Expand-Archive -Path "$env:USERPROFILE\Downloads\RECLin-PT-etapa-4-v1.zip" -DestinationPath . -Force

git status --short          # 9 modificados (M) e 8 novos (??; a pasta estrategias/ aparece como uma linha)
git add -A
git status --short          # os 19 arquivos de "Arquivos" abaixo, preparados (M e A)
git commit -m "Etapa 4 (v1): estratégias sem GPU (filtro de pistas e regra pura)" `
  -m "estrategias/filtro_pistas e estrategias/regra_pura, com a calibração no DEV (min_freq=3, R3 com gap<=1, porta de gap 25). Scripts calibrar.py, filtro_pistas.py e regra_pura.py. Reproduzem byte a byte CALIBRACAO_filtro.json, os 4 filtro_*.preds.json e regra_pura.preds.json do legado; a calibração não lê o TEST." `
  -m "Referência acrescentada: resultados_legado/CALIBRACAO_filtro.md (varredura completa da calibração)."

git log --oneline -n 3
git show --stat --format="%h %s" HEAD
git rev-parse "HEAD^{tree}"
```

Os valores esperados de `git show --stat` e do hash da árvore estão na
mensagem de entrega (o hash não pode constar neste arquivo, que faz parte da
árvore).

## Arquivos criados e modificados nesta etapa

Criados:

```
codigo/reclin/estrategias/__init__.py
codigo/reclin/estrategias/filtro_pistas.py
codigo/reclin/estrategias/regra_pura.py
codigo/scripts/calibrar.py
codigo/scripts/filtro_pistas.py
codigo/scripts/regra_pura.py
codigo/testes/unidade/test_estrategias.py
codigo/testes/equivalencia/test_estrategias_legado.py
codigo/testes/referencia/resultados_legado/CALIBRACAO_filtro.md
codigo/docs/entregas/etapa-4.md
```

Modificados:

| Arquivo | Mudança |
| --- | --- |
| `README.md` | Estado do repositório (etapas 1 a 4) e os comandos das estratégias |
| `codigo/README.md` | Uso dos três scripts, módulos da etapa 4, número de testes |
| `codigo/docs/entregas/README.md` | Linha da etapa 4 |
| `codigo/reclin/avaliacao/metricas.py` | `f1_pr` e `resumo_alvo_pr` (acréscimo) |
| `codigo/reclin/particoes.py` | Argumento opcional `nomes` em `conferir_particoes` |
| `codigo/testes/unidade/test_pacote.py` | Verificação dos níveis de importação |
| `codigo/testes/referencia/README.md` | `CALIBRACAO_filtro.md` em `resultados_legado/` (74 arquivos) |
| `codigo/testes/referencia/gerar_referencias.py` | Copia também `CALIBRACAO_filtro.md` |
| `codigo/testes/referencia/referencias.json` | SHA-256 do arquivo novo e a contagem (os anteriores não mudaram) |

As partições, os rótulos, o léxico congelado e as referências históricas não
mudaram. Nenhuma função existente mudou de comportamento.

## Limitações e pendências

- **Windows**: confirmar a instalação, os 337 testes e os três comandos de
  reprodução no seu computador.
- **Avaliação e registro do TEST** (as linhas de trilha `regra_pura.test_evals.jsonl`
  e as do filtro nas trilhas dos baselines, com `config_sha1`
  `3c4b401a2653` para a regra): ficam para o comando de avaliação do TEST
  (etapa 5). As linhas têm data e hora, então a reprodução será dos campos,
  não dos bytes.
- **`FASE2_test_summary.json`** (o quadro da fase 2) e o relatório
  `CALIBRACAO_filtro.md`: são geradores de relatório (`relatorios/`, etapa 7).
  O relatório tem a data da execução no cabeçalho.
- **Recalibração** sobre os baselines novos (etapa 8): `calibrar.py` já roda
  sobre execuções no formato novo. Os scripts das estratégias recusam uma
  calibração diferente até o registro `CONGELADO` ser atualizado.
- Seguem em aberto, de etapas anteriores: `corpus.py` e a pasta `tcc/`.
