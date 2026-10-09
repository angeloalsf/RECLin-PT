# Entrega da etapa 1 — esqueleto do núcleo

| | |
| --- | --- |
| Etapa | 1 (inclui a etapa 0, adaptada) |
| Versão | 2 |
| Data | 09/10/2026 |
| Pacote | `RECLin-PT-etapa-1-v2.zip` |
| Substitui | `RECLin-PT-etapa-1.zip` (versão 1) |

O pacote é o **projeto completo** ao fim da etapa 1: extraído numa pasta limpa,
tem tudo o que é preciso para instalar, executar e testar o que já foi
implementado. Não é preciso combinar com a versão 1.

## O que mudou em relação à versão 1

- O pacote passa a trazer `CITATION.cff` e `LICENSE` (sem alteração em
  relação ao repositório original): sem eles, a versão 1 não era
  autossuficiente.
- Novos testes de estrutura e de importação (`testes/unidade/test_pacote.py`):
  arquivos obrigatórios, integridade das referências, importação de **todos**
  os módulos de `reclin` (descobertos automaticamente) sem efeitos colaterais e
  `--help` de todos os scripts. O teste de importação que estava em
  `test_util.py` foi absorvido por ele.
- Índice das entregas em `codigo/docs/entregas/README.md`, com o que nunca vai
  no pacote e como cada pacote é verificado.
- O código da etapa (núcleo, script, dados e referências) não mudou.

## O que esta etapa entrega

A primeira parte do núcleo do RECLin-PT, escrita do zero a partir da lógica do
legado e verificada contra ele:

- **Tarefa** (`reclin/tarefa.py`): os três rótulos, a enumeração dos pares
  candidatos e a identidade do conjunto de referência em que toda estratégia é
  avaliada (152.686 / 19.064 / 19.210 pares em train / dev / test).
- **Partições** (`reclin/particoes.py`): a divisão 80/10/10 por documento,
  estratificada pela presença de `negation_of`, e o `MANIFEST.json` com o
  SHA-256 de cada partição.
- **Configuração base** (`reclin/config.py`): os hiperparâmetros comuns às
  estratégias treinadas, com os valores usados nos experimentos. Cada
  estratégia declarará a sua configuração herdando desta.
- **Utilitários** (`reclin/util/`): leitura e gravação de JSON/JSONL com
  quebra de linha LF, SHA-256, logging configurado só pelos scripts e caminhos
  padrão.
- **Script** `scripts/preparar_dados.py`: `conferir`, `manifesto` e
  `particionar`.
- **Dados**: as partições congeladas, copiadas do legado sem alteração, e o
  MANIFEST regerado pelo código novo.
- **Referências do legado** (`testes/referencia/`) e **72 testes**.

### Relação com as etapas 0 e 1 do plano

- **Etapa 0, adaptada.** O plano previa mover o legado para uma pasta
  `legado/` dentro do repositório. Com o repositório novo começando sem o
  legado, ele fica no repositório original (commit `a5f055c`), e da etapa 0
  sobra o que o repositório novo precisa: as referências de validação em
  `codigo/testes/referencia/`. O gerador recebe o caminho do legado em
  `--legado` e foi rodado de novo a partir de um checkout separado do original
  (ver Verificação).
- **Etapa 1, como planejada**: `pyproject`, `util`, `config`, `tarefa`,
  partições copiadas para `codigo/dados/particoes/` e MANIFEST regerado.

## Conteúdo do pacote (49 arquivos, 1,4 MB)

```
.gitattributes · .gitignore · README.md · LICENSE · CITATION.cff
codigo/
├── README.md                     instalação, uso e testes
├── pyproject.toml                pacote reclin (pip install -e codigo/)
├── requirements.txt              versões fixadas dos experimentos (as do legado)
├── reclin/
│   ├── __init__.py
│   ├── tarefa.py · particoes.py · config.py
│   └── util/  __init__.py · io.py · log.py · caminhos.py
├── scripts/
│   └── preparar_dados.py
├── dados/particoes/
│   ├── train.jsonl · dev.jsonl · test.jsonl     copiados do legado (mesmos SHA-256)
│   └── MANIFEST.json                             regerado
├── docs/entregas/
│   ├── README.md                 índice das entregas
│   └── etapa-1.md                este arquivo
└── testes/
    ├── conftest.py
    ├── unidade/      test_tarefa.py · test_particoes.py · test_config.py · test_util.py · test_pacote.py
    ├── equivalencia/ test_conjunto_referencia.py · test_particoes.py · test_config.py
    └── referencia/   README.md · gerar_referencias.py · dados.json · referencias.json
                      modelo_minusculo/ (config, pesos ~60 KB, tokenizer)
                      treino/ (sidecars do legado: baseline, restrito, Pair-Aware)
```

### Arquivos da raiz em relação ao repositório original

| Arquivo | Situação | O que mudou | O que foi preservado |
| --- | --- | --- | --- |
| `CITATION.cff`, `LICENSE` | Iguais ao original | — | Tudo |
| `.gitattributes` | Modificado | Acrescenta LF obrigatório para `*.jsonl`, `*.toml`, `*.txt` e `*.ipynb`. O primeiro é necessário: as partições são conferidas pelo SHA-256, e um checkout com CRLF no Windows (`core.autocrlf`) mudaria o hash | Todas as regras anteriores |
| `.gitignore` | Modificado | Ignora `codigo/dados/brutos/` (XML do SemClinBr) e `codigo/dados/processados/`; abre exceção para os pesos do modelo minúsculo das referências; as duas notas passam a citar os caminhos novos (`codigo/dados/particoes/`, `codigo/resultados/`) | Todas as regras anteriores, inclusive `SemClinBr-xml-public-v1/` e `data/processed/` |
| `README.md` | Modificado | Ganha "Estado do repositório", "Arquitetura", "Como usar" e "Projeto legado"; os caminhos passam aos do layout novo. Saem a árvore e os comandos do código antigo (`src/`, `scripts/`, `run.sh`, `Makefile`, notebooks), que não existem neste repositório, e a seção do artigo SBC, que fica no original | Autoria e orientação, a pergunta de pesquisa, o acesso ao corpus, hardware e tempo de treino, métricas, decisões principais, limitações e licença |

### O que não vem no pacote

O projeto legado, o XML do SemClinBr e o `dataset.jsonl`, ambientes virtuais,
caches, pesos de modelos e as dependências externas — ver a tabela em
[README.md](README.md) desta pasta, com o motivo e onde obter cada um. A pasta
`tcc/` ainda não faz parte do projeto (ver Pendências).

## Verificação

### Pacote

O pacote foi verificado pela rotina descrita no [índice](README.md):

- montado a partir dos arquivos versionados do projeto, sem nenhum item
  proibido e com todos os obrigatórios (a rotina foi testada recusando um
  pacote com uma pasta `legado/` e sem `LICENSE`);
- extraído numa pasta vazia: os 49 arquivos são idênticos, byte a byte, aos do
  projeto, e não há nenhum arquivo a mais;
- nessa pasta, em ambientes virtuais novos, uma vez com cada Python:

| Python | Instalação | Módulos importados | `--help` dos scripts | `conferir` | pytest |
| --- | --- | --- | --- | --- | --- |
| 3.10.20 | ok | 7 de 7 | ok | 3 partições conferem | 72 passaram |
| 3.12.3 | ok | 7 de 7 | ok | 3 partições conferem | 72 passaram |
| 3.13.16 | ok | 7 de 7 | ok | 3 partições conferem | 72 passaram |

Ambiente: Linux (contêiner de nuvem). São 54 testes de unidade e 18 de
equivalência.

**Não executado:** nada foi rodado no **Windows**. O código não usa nada
específico de sistema (todos os arquivos são abertos com UTF-8 e gravados com
LF), mas a confirmação no seu computador é o último passo de "Como aplicar".

### O que os testes de equivalência conferem (portão da etapa)

| Verificação | Resultado |
| --- | --- |
| Candidatos por partição com `max_gap=25` | 152.686 / 19.064 / 19.210, iguais ao legado |
| Sequência de candidatos (documento, e1, e2, rótulo) e de `y_true` | Idênticas, elemento a elemento, às do legado; o `y_true` do DEV e do TEST tinha sido conferido contra os sidecars do baseline BioBERTpt semente 42 |
| Contagem por rótulo | Igual nas três partições |
| Arquivos das partições | Mesmo SHA-256 do MANIFEST do legado |
| MANIFEST regerado | Hashes, contagens e semente iguais ao antigo; mudam só o gerador e o caminho do dataset |
| Algoritmo de partição | `particionar` reproduz as três partições byte a byte a partir dos 1.000 documentos, também pelo script |
| Configuração | Os valores padrão de `Config` são os do `config` gravado nos quatro baselines, e recalcular o hash de configuração do legado confere `weight_decay` e `warmup_ratio` |

Os testes de unidade cobrem a enumeração de candidatos (direção, janela, pares
excluídos, desempate de entidades com o mesmo trecho), a partição, o MANIFEST e
sua conferência, a validação da configuração, os utilitários e o pacote
(estrutura, referências, importações e scripts).

Para confirmar que os testes acusam desvios de verdade, o código foi alterado de
propósito em seis pontos (janela `>` trocada por `>=`, desempate por id
removido, exclusão de pares com o mesmo trecho removida, ordem do sorteio
invertida, `weight_decay` e `warmup_ratio` alterados): em todos, algum teste
falhou. A alteração do desempate só foi pega depois de acrescentado um teste de
unidade, porque no SemClinBr as 40 entidades com trecho repetido já vêm em
ordem de id.

### Referências regeradas pelo legado

O gerador de referências foi rodado a partir de um checkout separado do
repositório original no commit `a5f055c` (`--legado`), nas partes `dados` e
`testes`:

- `dados.json` saiu idêntico ao anterior, exceto o campo de procedência
  `y_true_confere_com`, que agora é relativo à raiz do legado
  (`results/baseline_biobertpt_seed42.preds.json`, antes
  `legado/results/...`);
- os dois testes de CPU do legado passaram (7 de 7 e 10 de 10).

As partes `modelo` e `treino` não foram regeradas: não mudaram desde a etapa 0,
quando cada treino foi rodado duas vezes com resultado idêntico.

### Quebras de linha com o git no Windows

O pacote foi extraído numa pasta limpa e versionado com git usando
`core.autocrlf=true` (a configuração comum no Windows): o `git add` gravou
todos os arquivos de código e de dados com LF. O git só avisou que os quatro
arquivos da raiz sem regra própria (`.gitattributes`, `.gitignore`,
`CITATION.cff` e `LICENSE`) ficariam com CRLF na cópia de trabalho, o que não
afeta nada.

## Como aplicar

O pacote foi feito para ser extraído numa **pasta limpa**. No PowerShell:

```powershell
$zip     = "$env:USERPROFILE\Downloads\RECLin-PT-etapa-1-v2.zip"   # ajuste se salvou em outro lugar
$destino = "C:\Users\angeloals\Documents\RECLin-PT"

# 1. Se a pasta já existe, guarde-a com outro nome (nada é apagado).
#    Feche antes editores e terminais abertos nela.
if (Test-Path $destino) { Rename-Item $destino "RECLin-PT-antes-etapa-1-v2" }

# 2. Extraia numa pasta limpa
Expand-Archive -Path $zip -DestinationPath $destino

# 3. Instale e confira
cd $destino
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\codigo[testes]"
.\.venv\Scripts\python.exe codigo\scripts\preparar_dados.py conferir
.\.venv\Scripts\python.exe -m pytest codigo
```

O esperado é `conferir` listar as três partições como "confere" e o pytest
terminar com `72 passed`.

A pasta antiga fica em `C:\Users\angeloals\Documents\RECLin-PT-antes-etapa-1-v2`.
Se você tinha alterado algum arquivo lá, compare com a versão nova antes de
apagá-la. Se ela era um repositório git, leve o histórico para a pasta nova:

```powershell
Move-Item "C:\Users\angeloals\Documents\RECLin-PT-antes-etapa-1-v2\.git" "$destino\.git"
git -C $destino status
```

O `git status` passa a mostrar exatamente o que a entrega mudou, inclusive
arquivos removidos.

## Pendências

- **Windows**: confirmar a instalação e os 72 testes no seu computador (passo 3
  acima).
- **`corpus.py`** (XML do SemClinBr → `dataset.jsonl`) não está em nenhuma
  etapa do plano. Só pode ser validado com o XML, conferindo o SHA-256 do
  dataset registrado no MANIFEST (`397ef42…`). Proposta: incluí-lo na etapa 2,
  com um teste que só roda quando o XML estiver em `codigo/dados/brutos/`, para
  ser executado no seu computador.
- **`tcc/`**: precisa entrar no projeto antes da etapa que gera tabelas e
  figuras (7). É preciso decidir se ela vem do repositório original como está.
- **Próxima etapa (2)**: `execucao` e `avaliacao/` — formato das execuções,
  métricas e significância —, validada pelas métricas dos sidecars do legado e
  pelas 26 comparações já gravadas.

## Decisões para revisar

- Os nomes dos hiperparâmetros continuam em inglês (`max_gap`, `ctx_chars`,
  `epochs`, `class_weight`…), porque o TCC os cita e as execuções os gravam
  assim. O resto do código está em português.
- O valor padrão de `max_gap` passou de 75 (que nenhum experimento usava) para
  25, o valor de todos os experimentos.
- Os nomes `iter_candidate_pairs` e `entity_gap` foram mantidos porque o texto
  do TCC os cita.
