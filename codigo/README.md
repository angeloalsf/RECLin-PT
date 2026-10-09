# RECLin-PT — código

Implementação nova do RECLin-PT, construída por etapas e validada contra o
projeto anterior (o "legado": repositório original, commit `a5f055c`). O código
novo não importa nada do legado: o que é preciso dele para validar está
registrado em `testes/referencia/`, e os testes rodam sem ele.

Quando um módulo diz "Origem no legado", os caminhos citados (`src/...`,
`scripts/...`) são os do repositório original.

## Instalação

Linux/macOS, a partir da raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "codigo/[testes]" -c codigo/requirements.txt
```

Windows (PowerShell), a partir da raiz do repositório:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\codigo[testes]" -c codigo\requirements.txt
```

(No Windows, os comandos abaixo podem ser rodados com `.\.venv\Scripts\python.exe`
no lugar de `python`, sem ativar o ambiente.)

A instalação editável (`-e`) é a forma suportada: os caminhos padrão
(`reclin/util/caminhos.py`) são calculados a partir da pasta `codigo/`.

Até a etapa 2 o pacote depende só de **numpy** e **scipy**. O `-c
codigo/requirements.txt` fixa as versões usadas nos experimentos (numpy 2.1.3
e scipy 1.16.3, registradas nas execuções do legado) sem instalar o resto da
lista. Essas versões existem para Python 3.11 a 3.13; em Python 3.10, rode o
mesmo comando sem o `-c` (o pip escolhe versões compatíveis; ver a nota da
etapa 2 sobre o que foi verificado com elas). As demais dependências de
`requirements.txt` (torch, transformers...) passam a ser necessárias quando
entrar o treino:

```bash
pip install -r codigo/requirements.txt
```

## Uso

Dados:

```bash
python codigo/scripts/preparar_dados.py conferir      # partições x MANIFEST (saída 1 se divergirem)
python codigo/scripts/preparar_dados.py manifesto     # recalcula o MANIFEST sem tocar nas partições
python codigo/scripts/preparar_dados.py particionar   # regera as partições a partir do dataset (exige --sobrescrever se já existirem)
```

`particionar` lê `codigo/dados/processados/dataset.jsonl`, que só existe com o
corpus SemClinBr (a conversão do XML entra numa etapa seguinte). Os dois
primeiros comandos funcionam sem o corpus.

Léxico de pistas de negação (especialização):

```bash
python codigo/scripts/analisar.py lexico                 # léxico congelado: 11 formas, lexico_sha1 e cobertura por partição
python codigo/scripts/analisar.py lexico --min-freq 2    # outro limiar, sem a guarda do congelado
```

Avaliação:

```bash
# métricas de sidecars de predições (ou de uma execução, com --execucao PASTA)
python codigo/scripts/avaliar.py codigo/testes/referencia/resultados_legado/baseline_biobertpt_seed42.preds.json

# uma comparação de significância entre dois sistemas
python codigo/scripts/comparar.py par --a A.preds.json --b B.preds.json --saida comparacao.json

# as 26 comparações do TCC, conferidas byte a byte contra as do legado (~2 min)
python codigo/scripts/comparar.py protocolo \
    --resultados codigo/testes/referencia/resultados_legado \
    --saida comparacoes --conferir codigo/testes/referencia/resultados_legado
```

## Testes

```bash
python -m pytest codigo                  # tudo (~1,5 min, dominado pelas 26 comparações)
python -m pytest codigo -m "not lento"   # sem as 26 comparações (~25 s)
```

| Pasta | O que verifica |
| --- | --- |
| `testes/unidade/` | O comportamento de cada módulo sobre exemplos pequenos, os scripts e a estrutura do pacote |
| `testes/equivalencia/` | O código novo contra as referências do legado: são os portões das etapas |
| `testes/referencia/` | As referências e o script que as gera (ver o README da pasta) |

Nenhum teste depende de recurso externo: todos rodam com os arquivos
versionados, sem rede, sem GPU e sem o legado. O único script que lê o legado,
`testes/referencia/gerar_referencias.py`, não faz parte da suíte.

## Organização

O pacote `reclin` tem três níveis, e um nível só importa os de cima:

- **Núcleo** — a tarefa e a infraestrutura para resolvê-la e avaliá-la:
  `tarefa`, `corpus`, `particoes`, `entrada`, `modelos`, `treino/`,
  `execucao/`, `avaliacao/`, `config`, `util/`.
- **Especialização** — o conhecimento sobre como a negação aparece no corpus:
  `negacao/`.
- **Estratégias experimentais** — formas de combinar núcleo e especialização
  para prever, comparadas pelas execuções que gravam: `estrategias/`. Só a
  vencedora compõe o modelo entregue, e só `modelo_final.py` a nomeia.

Os módulos de `reclin/` não configuram logging, não alteram `os.environ` e não
mexem em `sys.path` ao serem importados; isso fica com os scripts.

### O que já existe (etapas 1 a 3)

| Caminho | Responsabilidade |
| --- | --- |
| `reclin/tarefa.py` | Rótulos, enumeração dos pares candidatos (`iter_candidate_pairs`) e identidade do conjunto de referência (152.686 / 19.064 / 19.210 pares) |
| `reclin/particoes.py` | Partição 80/10/10 por documento e o MANIFEST que a congela |
| `reclin/config.py` | `Config`, a configuração base das estratégias treinadas, e `ENCODERS` |
| `reclin/execucao/` | O formato em disco das execuções: sidecar de predições (`predicoes`), diretório da execução (`diretorio`) e trilha das avaliações do TEST (`trilha`) |
| `reclin/avaliacao/` | Métricas (`metricas`), McNemar e bootstrap pareado (`significancia`), as 26 comparações do TCC (`protocolo`) e a agregação entre sementes (`agregacao`) |
| `reclin/negacao/lexico.py` | Especialização: normalização, indução do léxico de pistas no TRAIN, `e_pista`, `lexico_sha1`, cobertura e o léxico congelado com a guarda (`min_freq=3`, 11 formas, `70c93fa807de`) |
| `reclin/util/` | JSON/JSONL e SHA-256 (`io`), logging configurado só por scripts (`log`), caminhos padrão (`caminhos`) |
| `scripts/preparar_dados.py` | `conferir`, `manifesto` e `particionar` |
| `scripts/avaliar.py` | Métricas de sidecars ou de uma execução |
| `scripts/comparar.py` | Uma comparação (`par`) ou as 26 do TCC (`protocolo`) |
| `scripts/analisar.py` | `lexico`: o léxico de pistas, seu hash e a cobertura por partição |
| `dados/particoes/` | `train/dev/test.jsonl` congelados e `MANIFEST.json` (versionados) |
| `testes/` | 275 testes (unidade, equivalência com o legado e estrutura do pacote) e as referências do legado |
| `docs/entregas/` | Índice das entregas e, por etapa, o que foi entregue, como foi verificado e o que ficou pendente |

`dados/brutos/` (XML do SemClinBr) e `dados/processados/` (`dataset.jsonl`)
ficam fora do git: o corpus tem licença restrita. As partições versionadas
contêm o texto das notas, como no repositório original.

### Formato de uma execução

```
<raiz>/<nome>/                 ex.: resultados/execucoes/baseline_biobertpt_seed42/
    config.json                nome, estratégia, configuração e identidade dos conjuntos de referência
    predicoes_dev.json         sidecar do DEV  (model, seed, labels, y_true, y_pred[, probs, extras])
    predicoes_test.json        sidecar do TEST
    metricas.json              métricas por partição
    avaliacoes_test.jsonl      trilha das avaliações do TEST (eval_index, eval_index_for_config)
```

O sidecar é o do legado, sem mudança, e `execucao.diretorio.caminho_predicoes`
também encontra as predições no formato antigo (`<nome>.preds.json`,
`<nome>.dev_preds.json`). Os relatórios de comparação mantêm o nome e o formato
do legado (`significance_<A>_vs_<B>.json`).
