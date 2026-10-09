# Entrega da etapa 3 — especialização (léxico de pistas de negação)

| | |
| --- | --- |
| Etapa | 3 |
| Versão | 1 |
| Data | 09/10/2026 |
| Pacote | `RECLin-PT-etapa-3-v1.zip` |
| Parte de | etapa 2, versão 1 (`RECLin-PT-etapa-2-v1.zip`) |

O pacote é o **projeto completo** até a etapa 3: extraído numa pasta limpa,
tem tudo o que é preciso para instalar, executar e testar o que já foi
implementado.

## Resultado dos critérios de aceitação

| # | Critério | Obtido | Referência | Situação |
| --- | --- | --- | --- | --- |
| 1 | Léxico com `min_freq=3`, só do TRAIN | induzido dos 800 documentos do TRAIN, 1.255 pares candidatos `negation_of` | — | atendido |
| 2 | Exatamente 11 formas | 11 formas, as mesmas, com as mesmas frequências e na mesma ordem | `dados.json` e as 6 execuções do legado que gravaram o léxico | atendido |
| 3 | Mesmo `lexico_sha1` | `70c93fa807de` | `70c93fa807de` (`dados.json`, 5 restritos, Pair-Aware) | atendido |
| 4 | Cobertura de 0,9467 no DEV | 142 / 150 = 0.9466666666666667, que arredonda para 0,9467 | 0.9466666666666667 (`dados.json`, restritos) e 0,9467 (TCC) | atendido, com igualdade exata |
| 5 | Testes de construção, frequência mínima, determinismo, hash e cobertura | 20 de unidade e 26 de equivalência novos | — | atendido |
| 6 | O léxico não usa DEV nem TEST | ver "Só o TRAIN" | — | atendido |
| 7 | Compatível com as etapas 1 e 2 | usa `tarefa.iter_candidate_pairs` e `tarefa.MAX_GAP`; nada das etapas anteriores mudou de comportamento; os 228 testes anteriores continuam passando | — | atendido |

### As 11 formas

| Forma normalizada | Frequência no TRAIN |
| --- | --- |
| sem | 662 |
| nega | 353 |
| nao | 128 |
| ausente | 12 |
| ausencia | 11 |
| s | 11 |
| evacuacao | 10 |
| s/ | 10 |
| ausentes | 5 |
| eliminacao fecal | 4 |
| indolor | 3 |

Somam 1.209 das 1.255 ocorrências de `negation_of` no TRAIN. "evacuacao",
"eliminacao fecal", "s" e "s/" estão no léxico porque o SemClinBr anota essas
formas como primeiro argumento de `negation_of` no TRAIN pelo menos 3 vezes.
O léxico é o que os dados induzem; nada foi acrescentado, retirado ou
ajustado.

## Como o legado definia o léxico (confirmado no código e nos resultados)

Fontes: `src/negation_lexicon.py` (`normalize_surface`, `count_cue_forms`,
`induce_lexicon`, `is_cue`), `src/finetuning_restrito/restricted_space.py`
(`lexico_sha1` e a guarda `carregar_lexico`), `scripts/check_tcc_numbers.py`
(conferência da cobertura citada no TCC) e `results/CALIBRACAO_filtro.json`.

- **Normalização:** NFKD, remoção dos diacríticos (caracteres combinantes),
  colapso de espaços em branco, `strip` e `casefold`.
- **Contagem:** para cada documento do TRAIN, cada par candidato de
  `iter_candidate_pairs` com `max_gap=25` cujo rótulo é `negation_of` conta
  uma ocorrência da forma normalizada do texto de `e1`. Conta pares
  candidatos, não relações anotadas: são 1.255 pares contra 1.299 relações,
  porque 44 ficam fora da janela.
- **Limiar:** ficam as formas com frequência ≥ `min_freq`.
- **Ordem:** frequência decrescente e, no empate, ordem alfabética.
- **Pista:** a forma normalizada do texto de `e1` pertence ao léxico.
- **`lexico_sha1`:** `sha1(json.dumps(sorted(léxico.items()), ensure_ascii=False))`,
  12 primeiros caracteres hexadecimais. Cobre formas e frequências.
- **Congelamento:** `min_freq=3` foi escolhido pela calibração do filtro no DEV
  (`CALIBRACAO_filtro.json`: `min_freq` 3, `lexicon_size` 11, `combined_gap`
  25). A guarda do legado conferia `min_freq` e o número de formas contra esse
  arquivo.
- **Cobertura:** entre os pares candidatos de uma partição com rótulo
  `negation_of`, a fração cujo `e1` é pista (definição do TCC e de
  `check_tcc_numbers.py`). Como o filtro e o fine-tuning restrito só predizem
  `negation_of` onde `e1` é pista, ela é o teto de recall dessas estratégias
  na classe-alvo, somado ao teto da janela `max_gap`.

## O que foi implementado

`reclin/negacao/lexico.py` (módulo único, como no plano: `negacao/lexico.py`):

| Função ou constante | O que faz | Origem no legado |
| --- | --- | --- |
| `normalizar(texto)` | Forma de comparação | `normalize_surface` |
| `contar_formas(documentos, max_gap)` | Frequência das formas de `e1` nos candidatos `negation_of` | `count_cue_forms` |
| `induzir(documentos_train, min_freq, max_gap)` | O léxico, ordenado | `induce_lexicon` |
| `e_pista(entidade, lexico)` | O predicado de pista | `is_cue` |
| `lexico_sha1(lexico)` | Identidade do léxico | `restricted_space.lexico_sha1` |
| `cobertura(documentos, lexico, max_gap)` | Pares `negation_of`, cobertos e a fração | `teto_recall` e `check_tcc_numbers` |
| `CONGELADO` | Registro do léxico congelado: `min_freq` 3, `max_gap` 25, 11 formas, `70c93fa807de`, origem | `CALIBRACAO_filtro.json` |
| `carregar_congelado(documentos_train)` | Induz o léxico e falha se não for exatamente o registrado | guarda de `carregar_lexico` |

`reclin/negacao/__init__.py` documenta o nível de especialização: só
conhecimento, nenhuma estratégia. O filtro de pistas, a regra pura e a
calibração ficam para a etapa 4.

`scripts/analisar.py lexico` mostra o léxico (o congelado, ou outro com
`--min-freq`), o hash e a cobertura nas três partições, depois de conferir as
partições contra o MANIFEST. Com `--saida`, grava um JSON.

### Esquema do léxico

Em memória, um `dict[str, int]`: forma normalizada → frequência no TRAIN,
ordenado por frequência decrescente e forma. Serve como conjunto
(`forma in lexico`). O JSON gravado por `analisar.py lexico --saida`:

```json
{
  "origem": "congelado",            // ou "induzido" (com --min-freq)
  "min_freq": 3,
  "max_gap": 25,
  "n_formas": 11,
  "lexico_sha1": "70c93fa807de",
  "formas": {"sem": 662, "nega": 353, "...": 0},
  "cobertura": {"train": {"negation_of": 1255, "cobertos": 1209, "cobertura": 0.96...},
                "dev":   {"negation_of": 150,  "cobertos": 142,  "cobertura": 0.94...},
                "test":  {"negation_of": 152,  "cobertos": 148,  "cobertura": 0.97...}}
}
```

### Referências

Acrescentado `codigo/testes/referencia/resultados_legado/CALIBRACAO_filtro.json`
(5 linhas), copiado sem alteração do legado pelo gerador (`--partes
resultados`), a partir de um checkout separado do commit `a5f055c`. É a fonte
da guarda do léxico congelado. Os 72 arquivos de referência anteriores e os
seus SHA-256 não mudaram; `referencias.json` só ganhou a linha do arquivo
novo.

## Evidências

### Construção, frequência mínima, hash e cobertura

`testes/equivalencia/test_lexico_legado.py`, 26 testes, igualdade exata (sem
tolerância) contra `dados.json` (gerado pelo código do legado), os resultados
do legado e o número do TCC:

| Verificação | Resultado |
| --- | --- |
| Formas e frequências no TRAIN com `min_freq=1` | 51 formas, 1.255 ocorrências, iguais a `contagem_formas_train` |
| Léxico com `min_freq` 1, 2, 3, 5, 10 | 51 / 17 / 11 / 9 / 8 formas, as mesmas listas de `por_min_freq` |
| Léxico congelado | as 11 formas, as frequências, a ordem, 1.209 ocorrências e `70c93fa807de` |
| Registro `CONGELADO` × `CALIBRACAO_filtro.json` | `min_freq` 3, 11 formas, `max_gap` 25 |
| Léxico gravado pelas 6 execuções do legado (5 restritos, Pair-Aware) | mesmas formas, frequências, sha1, `min_freq` e `max_gap`; `config.lexicon_sha1` igual |
| Cobertura por partição | TRAIN 1209/1255 = 0.9633466135458167; DEV 142/150 = 0.9466666666666667; TEST 148/152 = 0.9736842105263158 — iguais a `dados.json` e às execuções do restrito |
| Cobertura citada no TCC | arredondada a 4 casas, 0,9467 |
| `e_pista` sobre todos os 190.960 candidatos | escolhe os mesmos 6.805 / 774 / 800 pares que o legado, na mesma ordem (hash dos índices), com as mesmas contagens por rótulo |

### Só o TRAIN

| Evidência | Resultado |
| --- | --- |
| O módulo não lê arquivos: recebe os documentos de quem chama (`test_modulo_nao_le_arquivos`) | confirmado |
| Esvaziar DEV e TEST não muda o léxico | confirmado |
| Acrescentar DEV, TEST ou os dois à indução muda o léxico | TRAIN+DEV: 12 formas, `42a6f77190ff`; TRAIN+TEST: 12, `0ed86508faa2`; os três: 13, `3d005144bc3d`. A guarda `carregar_congelado` recusa os três |

Como só a indução apenas com o TRAIN produz 11 formas e `70c93fa807de`,
reproduzir esses dois valores é evidência de que nem o DEV nem o TEST foram
usados.

### Determinismo

O mesmo léxico, na mesma ordem e com o mesmo hash, com os documentos em outra
ordem (unidade e equivalência, nesta com o TRAIN embaralhado) e em chamadas
repetidas.

### Os testes acusam desvios

| Alteração no código | Resultado |
| --- | --- |
| NFC em vez de NFKD (acentos não são removidos) | testes de equivalência falham |
| Contar `e2` em vez de `e1` | falham |
| Limiar estrito (`>` em vez de `>=`) | falham |
| Contar sem janela (como relações anotadas) | falham |
| Sem colapsar espaços | só o teste de unidade falha |
| `lower()` em vez de `casefold()` | só o teste de unidade falha |

As duas últimas não mudam nada no SemClinBr (nenhuma forma de `e1` tem
espaços repetidos nem caracteres em que `casefold` e `lower` diferem). Foram
mantidas por fidelidade ao legado e ficam protegidas pelos testes de unidade.

### Divergências

Nenhuma. Todos os valores de referência foram reproduzidos com igualdade
exata. A cobertura de 0,9467 citada no TCC é o arredondamento a 4 casas de
142/150 = 0.9466666666666667, e o valor exato confere com o gravado pelo
legado.

## Testes

| Tipo | Onde | Quantidade | Depende de |
| --- | --- | --- | --- |
| Unidade | `testes/unidade/` | 107 (20 de `test_negacao.py` e 1 do `--help` de `analisar.py`, novos) | nada externo |
| Equivalência com as referências | `testes/equivalencia/` | 168 (26 de `test_lexico_legado.py`, novos; 26 marcados `lento`, da etapa 2) | só os arquivos versionados em `testes/referencia/` |
| Execuções com recurso externo | `testes/referencia/gerar_referencias.py` | — | um checkout do legado; fora da suíte |

Total: 275 testes. Nenhum acessa o legado, a rede ou uma GPU.

### Resultados obtidos

Ambiente: Linux (contêiner de nuvem). O pacote foi extraído numa pasta vazia,
comparado byte a byte com o projeto e instalado em ambientes novos:

| Python | numpy | scipy | Versões | Suíte completa |
| --- | --- | --- | --- | --- |
| 3.10.20 | 2.2.6 | 1.15.3 | escolhidas pelo pip (as fixadas não existem para 3.10) | 275 passaram |
| 3.12.3 | 2.1.3 | 1.16.3 | fixadas (`-c codigo/requirements.txt`) | 275 passaram |
| 3.13.16 | 2.1.3 | 1.16.3 | fixadas | 275 passaram |

Em cada ambiente também: importação de todos os módulos, `--help` dos 4
scripts e `preparar_dados.py conferir` (3 partições conferem). A integridade
das 73 referências do legado é conferida pelo `test_pacote.py`.

**Não executado:** nada foi rodado no **Windows**.

## Como aplicar

Numa pasta limpa, no PowerShell:

```powershell
$zip     = "$env:USERPROFILE\Downloads\RECLin-PT-etapa-3-v1.zip"   # ajuste se salvou em outro lugar
$destino = "C:\Users\angeloals\Documents\RECLin-PT"

if (Test-Path $destino) { Rename-Item $destino "RECLin-PT-antes-etapa-3" }   # nada é apagado
Expand-Archive -Path $zip -DestinationPath $destino

cd $destino
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\codigo[testes]" -c codigo\requirements.txt
.\.venv\Scripts\python.exe codigo\scripts\preparar_dados.py conferir
.\.venv\Scripts\python.exe codigo\scripts\analisar.py lexico
.\.venv\Scripts\python.exe -m pytest codigo
```

O esperado: `conferir` com as três partições "confere"; `analisar.py lexico`
com 11 formas, `lexico_sha1=70c93fa807de` e "Cobertura dev: 142 de 150 ... =
0.9467"; o pytest com `275 passed` (cerca de 2 minutos). Com Python 3.10, rode
o `pip install` sem o `-c codigo\requirements.txt`.

### Registrar no git só as alterações da etapa 3

A partir de um repositório que já tem o commit da etapa 2, com a cópia de
trabalho limpa. A etapa 3 não remove nenhum arquivo, então extrair por cima
equivale a extrair numa pasta limpa:

```powershell
cd C:\Users\angeloals\Documents\RECLin-PT
git status --short          # não deve listar nada

Expand-Archive -Path "$env:USERPROFILE\Downloads\RECLin-PT-etapa-3-v1.zip" -DestinationPath . -Force

git status --short          # 6 modificados (M) e 6 novos (??), os de "Arquivos" abaixo
git add -A
git status --short          # os mesmos 13 arquivos, agora preparados (M e A)
git commit -m "Etapa 3 (v1): léxico de pistas de negação (especialização)" `
  -m "negacao/lexico: normalização, indução no TRAIN, e_pista, lexico_sha1, cobertura e o léxico congelado com guarda (min_freq=3, 11 formas, 70c93fa807de; cobertura no DEV 142/150 = 0,9467), validados contra o legado. Script analisar.py lexico." `
  -m "Referência acrescentada: resultados_legado/CALIBRACAO_filtro.json (fonte da guarda do léxico congelado)."

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
codigo/reclin/negacao/__init__.py
codigo/reclin/negacao/lexico.py
codigo/scripts/analisar.py
codigo/testes/unidade/test_negacao.py
codigo/testes/equivalencia/test_lexico_legado.py
codigo/testes/referencia/resultados_legado/CALIBRACAO_filtro.json
codigo/docs/entregas/etapa-3.md
```

Modificados:

| Arquivo | Mudança |
| --- | --- |
| `README.md` | Estado do repositório (etapas 1 a 3) e o comando do léxico |
| `codigo/README.md` | Uso de `analisar.py`, módulos da etapa 3, número de testes |
| `codigo/docs/entregas/README.md` | Linha da etapa 3 |
| `codigo/testes/referencia/README.md` | `CALIBRACAO_filtro.json` em `resultados_legado/` (73 arquivos) |
| `codigo/testes/referencia/gerar_referencias.py` | Copia também `CALIBRACAO_filtro.json` |
| `codigo/testes/referencia/referencias.json` | SHA-256 do arquivo novo (os anteriores não mudaram) |

Nenhum módulo das etapas 1 e 2 (`tarefa`, `particoes`, `config`, `util`,
`execucao`, `avaliacao`) foi alterado. As partições, os rótulos e as
referências históricas também não.

## Limitações e pendências

- **Windows**: confirmar a instalação e os 275 testes no seu computador.
- O registro `CONGELADO` reflete a calibração do legado. Quando o filtro for
  recalibrado sobre os baselines novos (etapa 8), o registro muda, com a nova
  origem.
- **Etapa 4** (filtro de pistas e regra pura, com calibração): não iniciada.
- Seguem em aberto, de etapas anteriores: `corpus.py` e a pasta `tcc/`.
