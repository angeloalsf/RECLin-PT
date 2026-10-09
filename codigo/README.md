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
pip install -e "codigo/[testes]"
```

Windows (PowerShell), a partir da raiz do repositório:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\codigo[testes]"
```

(No Windows, os comandos abaixo podem ser rodados com `.\.venv\Scripts\python.exe`
no lugar de `python`, sem ativar o ambiente.)

A instalação editável (`-e`) é a forma suportada: os caminhos padrão
(`reclin/util/caminhos.py`) são calculados a partir da pasta `codigo/`. Até a
etapa 1 o pacote só usa a biblioteca padrão do Python (3.10 ou mais novo); as
versões fixadas dos experimentos, em `requirements.txt`, passam a ser
necessárias quando entrarem o treino e a avaliação:

```bash
pip install -r codigo/requirements.txt
```

## Uso

```bash
python codigo/scripts/preparar_dados.py conferir      # partições x MANIFEST (saída 1 se divergirem)
python codigo/scripts/preparar_dados.py manifesto     # recalcula o MANIFEST sem tocar nas partições
python codigo/scripts/preparar_dados.py particionar   # regera as partições a partir do dataset (exige --sobrescrever se já existirem)
```

`particionar` lê `codigo/dados/processados/dataset.jsonl`, que só existe com o
corpus SemClinBr (a conversão do XML entra numa etapa seguinte). Os dois
primeiros comandos funcionam sem o corpus.

## Testes

```bash
python -m pytest codigo          # da raiz do repositório
```

| Pasta | O que verifica |
| --- | --- |
| `testes/unidade/` | O comportamento de cada módulo sobre exemplos pequenos |
| `testes/equivalencia/` | O código novo contra as referências do legado: são os portões das etapas |
| `testes/referencia/` | As referências e o script que as gera (ver o README da pasta) |

## Organização

O pacote `reclin` tem três níveis, e um nível só importa os de cima:

- **Núcleo** — a tarefa e a infraestrutura para resolvê-la e avaliá-la:
  `tarefa`, `corpus`, `particoes`, `entrada`, `modelos`, `treino/`,
  `execucao`, `avaliacao/`, `config`, `util/`.
- **Especialização** — o conhecimento sobre como a negação aparece no corpus:
  `negacao/`.
- **Estratégias experimentais** — formas de combinar núcleo e especialização
  para prever, comparadas pelas execuções que gravam: `estrategias/`. Só a
  vencedora compõe o modelo entregue, e só `modelo_final.py` a nomeia.

Os módulos de `reclin/` não configuram logging, não alteram `os.environ` e não
mexem em `sys.path` ao serem importados; isso fica com os scripts.

### O que já existe (etapa 1)

| Caminho | Responsabilidade |
| --- | --- |
| `reclin/tarefa.py` | Rótulos, enumeração dos pares candidatos (`iter_candidate_pairs`) e identidade do conjunto de referência (152.686 / 19.064 / 19.210 pares) |
| `reclin/particoes.py` | Partição 80/10/10 por documento e o MANIFEST que a congela |
| `reclin/config.py` | `Config`, a configuração base das estratégias treinadas, e `ENCODERS` |
| `reclin/util/` | JSON/JSONL e SHA-256 (`io`), logging configurado só por scripts (`log`), caminhos padrão (`caminhos`) |
| `scripts/preparar_dados.py` | `conferir`, `manifesto` e `particionar` |
| `dados/particoes/` | `train/dev/test.jsonl` congelados e `MANIFEST.json` (versionados) |
| `testes/` | 72 testes (unidade, equivalência com o legado e estrutura do pacote) e as referências do legado |
| `docs/entregas/` | Índice das entregas e, por etapa, o que foi entregue, como foi verificado e o que ficou pendente |

`dados/brutos/` (XML do SemClinBr) e `dados/processados/` (`dataset.jsonl`)
ficam fora do git: o corpus tem licença restrita. As partições versionadas
contêm o texto das notas, como no repositório original.
