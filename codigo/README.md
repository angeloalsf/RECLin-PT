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

O pacote depende de **numpy** e **scipy** (avaliação) e, desde a etapa 5, de
**torch** e **transformers** (treino). O `-c codigo/requirements.txt` fixa as
versões dos experimentos sem instalar o resto da lista: numpy 2.1.3, scipy
1.16.3, torch 2.11.0, transformers 5.16.1 e tokenizers 0.23.2 — as registradas
nas execuções do legado e no ambiente que gerou as referências. numpy 2.1.3 e
scipy 1.16.3 existem para Python 3.11 a 3.13; em Python 3.10, fixe só as do
treino (numpy e scipy ficam com as que o pip escolher; ver as notas das etapas
2 e 5 sobre o que foi verificado assim):

```bash
pip install -e "codigo/[testes]" torch==2.11.0 transformers==5.16.1 tokenizers==0.23.2 safetensors==0.8.0 huggingface_hub==1.29.0
```

O torch do PyPI para Windows e macOS é o de CPU (algumas centenas de MB). No
Linux, o do PyPI vem com as bibliotecas de CUDA (alguns GB); para uma
instalação só de CPU, instale antes o torch do índice do PyTorch
(`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`;
não verificado aqui, porque este ambiente não acessa esse índice).
O treino em GPU usa o mesmo comando, numa máquina com CUDA.

As demais dependências de `requirements.txt` (lxml, matplotlib,
scikit-learn) entram com as análises e os relatórios das etapas seguintes.

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

Estratégias sem GPU (calibração no DEV, filtro de pistas e regra pura):

```bash
REF=codigo/testes/referencia/resultados_legado

# calibração no DEV (lê só train e dev): CALIBRACAO_filtro.json, conferido byte a byte
python codigo/scripts/calibrar.py --resultados $REF --saida reproducao/CALIBRACAO_filtro.json \
    --detalhes reproducao/calibracao_detalhes.json --conferir $REF/CALIBRACAO_filtro.json

# filtro de pistas sobre os quatro baselines: execuções filtro_<encoder>_seed<N> (DEV e TEST)
python codigo/scripts/filtro_pistas.py --resultados $REF --calibracao $REF/CALIBRACAO_filtro.json \
    --saida reproducao/execucoes --conferir $REF

# regra pura: execução regra_pura (DEV e TEST)
python codigo/scripts/regra_pura.py --calibracao $REF/CALIBRACAO_filtro.json \
    --saida reproducao/execucoes --conferir $REF
```

`--resultados` é onde estão as predições das execuções de base (formato novo
ou do legado); sem `--saida`, as execuções vão para `codigo/resultados/execucoes/`.
`--conferir` compara os bytes com os arquivos de referência e sai com 1 se
algum diferir. Os scripts das estratégias usam o léxico congelado e recusam
uma calibração que não corresponda a ele.

Treino (o classificador da tarefa; o TEST fica para um comando separado):

```bash
# treino novo: cria codigo/resultados/execucoes/<nome>/ (lê só TRAIN e DEV)
python codigo/scripts/treinar.py --nome classificador_biobertpt_seed42 --encoder biobertpt --seed 42 \
    --checkpoint-a-cada 500

# se for interrompido: o MESMO comando com --retomar continua do último checkpoint
python codigo/scripts/treinar.py --nome classificador_biobertpt_seed42 --encoder biobertpt --seed 42 \
    --checkpoint-a-cada 500 --retomar

# depois de concluído o treino: avaliação do TEST com o melhor modelo (uma vez)
python codigo/scripts/avaliar_test.py --execucao codigo/resultados/execucoes/classificador_biobertpt_seed42
```

Configuração: os hiperparâmetros têm como padrão os de `reclin.config.Config`
(os dos experimentos: `max_gap` 25, `ctx_chars` 128, `max_length` 128, 3
épocas, lotes de 64, `lr` 2e-5, `weight_decay` 0,01, aquecimento de 10%,
recorte do gradiente em 1,0, pesos `balanced`); cada um muda com a opção de
mesmo nome (`--epochs`, `--batch-size`, `--lr`...). `--modelo PASTA` usa um
checkpoint local no lugar do de `--encoder`. Para experimentar em CPU, o modelo
minúsculo das referências serve:

```bash
python codigo/scripts/treinar.py --nome teste --saida reproducao --modelo codigo/testes/referencia/modelo_minusculo \
    --epochs 1 --lr 1e-3 --parar-apos-passo 200     # sai com código 3: interrompido, com checkpoint
python codigo/scripts/treinar.py --nome teste --saida reproducao --modelo codigo/testes/referencia/modelo_minusculo \
    --epochs 1 --lr 1e-3 --retomar                  # continua do passo 201
```

Checkpoints, em `<execução>/checkpoints/` (fora do git):

| Arquivo | Conteúdo |
| --- | --- |
| `ultimo.pt` | Estado completo de retomada: pesos, otimizador, agendador, geradores aleatórios (python, numpy, torch, CUDA) e do DataLoader, posição (época, passo na época, passo global), soma da loss da época, histórico do DEV, melhor macro-F1, identidade da execução e ambiente |
| `ultimo.json` | Resumo legível do mesmo checkpoint (posição e data) |
| `melhor_modelo/` | Pesos e tokenizer da melhor época pelo macro-F1 do DEV (`save_pretrained`): é o que `avaliar_test.py` usa |

Um checkpoint é gravado no fim de cada época e, com `--checkpoint-a-cada N`,
a cada N passos; a gravação é atômica (arquivo temporário renomeado). A
retomada recusa uma configuração, um modelo ou partições diferentes dos
gravados, e `treino.json` registra cada sessão (nova ou retomada, de onde
partiu, até onde foi). Saídas de `treinar.py`: 0 concluído, 3 interrompido
(retomável), 1 erro.

Reprodutibilidade: no mesmo ambiente (mesmas versões, mesmo dispositivo, mesmo
número de threads), a mesma configuração e a mesma semente dão o mesmo
resultado, bit a bit, e uma execução interrompida e retomada dá o mesmo
resultado que a contínua — conferido em CPU pelos testes. Entre máquinas,
sistemas, versões ou entre CPU e GPU, não há garantia de igualdade bit a bit.
Em GPU, o determinismo depende do cuBLAS/cuDNN (`CUBLAS_WORKSPACE_CONFIG` é
definido pelos scripts) e não foi verificado nesta etapa.

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
python -m pytest codigo                  # tudo (~4,5 min numa CPU de 2 núcleos)
python -m pytest codigo -m "not lento"   # sem as 26 comparações e os lotes do TRAIN (~2,5 min)
```

| Pasta | O que verifica |
| --- | --- |
| `testes/unidade/` | O comportamento de cada módulo sobre exemplos pequenos, os scripts e a estrutura do pacote |
| `testes/integracao/` | O treino de ponta a ponta em CPU com o modelo minúsculo: retomada exata, reprodutibilidade, separação entre treino e TEST, scripts |
| `testes/equivalencia/` | O código novo contra as referências do legado: são os portões das etapas |
| `testes/referencia/` | As referências e o script que as gera (ver o README da pasta) |

Nenhum teste depende de recurso externo: todos rodam com os arquivos
versionados, sem rede, sem GPU e sem o legado. O único script que lê o legado,
`testes/referencia/gerar_referencias.py`, não faz parte da suíte.

Dois controles por variável de ambiente:

- a reprodução numérica do treino do legado (loss de cada passo, pesos,
  predições) só é igual bit a bit no ambiente em que as referências foram
  geradas (Linux x86_64, torch 2.11.0, transformers 5.16.1); em outro, esses
  testes são pulados com o motivo. `RECLIN_FORCAR_EQUIVALENCIA=1` roda assim
  mesmo;
- o treino completo do modelo minúsculo (as partições inteiras, cerca de 11
  min) só roda com `RECLIN_TREINO_COMPLETO=1`.

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

### O que já existe (etapas 1 a 5)

| Caminho | Responsabilidade |
| --- | --- |
| `reclin/tarefa.py` | Rótulos, enumeração dos pares candidatos (`iter_candidate_pairs`) e identidade do conjunto de referência (152.686 / 19.064 / 19.210 pares) |
| `reclin/particoes.py` | Partição 80/10/10 por documento e o MANIFEST que a congela |
| `reclin/config.py` | `Config`, a configuração base das estratégias treinadas, e `ENCODERS` |
| `reclin/entrada.py` | Janela marcada de cada candidato (`[E1]`/`[/E1]`/`[E2]`/`[/E2]`, `ctx_chars` de contexto), exemplos alinhados com os candidatos e o DataLoader que tokeniza cada lote com gerador próprio |
| `reclin/modelos.py` | Tokenizer com os marcadores, classificador de sequência de 3 rótulos, gravação atômica, recarga genérica e identidade (vocabulário, configuração) |
| `reclin/treino/` | Reprodutibilidade (ambiente, sementes, geradores), checkpoints (estado de retomada e melhor modelo), o laço de treino com retomada exata e o classificador da tarefa (`treinar_execucao` e `avaliar_test`, separados) |
| `reclin/execucao/` | O formato em disco das execuções: sidecar de predições (`predicoes`), diretório da execução (`diretorio`) e trilha das avaliações do TEST (`trilha`) |
| `reclin/avaliacao/` | Métricas (`metricas`), McNemar e bootstrap pareado (`significancia`), as 26 comparações do TCC (`protocolo`) e a agregação entre sementes (`agregacao`) |
| `reclin/negacao/lexico.py` | Especialização: normalização, indução do léxico de pistas no TRAIN, `e_pista`, `lexico_sha1`, cobertura e o léxico congelado com a guarda (`min_freq=3`, 11 formas, `70c93fa807de`) |
| `reclin/estrategias/filtro_pistas.py` | Estratégia: filtro de pistas sobre as predições de outra execução, com a calibração de `min_freq` e da porta de gap no DEV |
| `reclin/estrategias/regra_pura.py` | Estratégia: regra pura R1 a R4 (pista + distância, sem modelo), com a calibração da regra no DEV |
| `reclin/util/` | JSON/JSONL e SHA-256 (`io`), logging configurado só por scripts (`log`), caminhos padrão (`caminhos`) |
| `scripts/preparar_dados.py` | `conferir`, `manifesto` e `particionar` |
| `scripts/avaliar.py` | Métricas de sidecars ou de uma execução |
| `scripts/comparar.py` | Uma comparação (`par`) ou as 26 do TCC (`protocolo`) |
| `scripts/analisar.py` | `lexico`: o léxico de pistas, seu hash e a cobertura por partição |
| `scripts/calibrar.py` | Calibração no DEV do filtro e da regra (`CALIBRACAO_filtro.json`) |
| `scripts/filtro_pistas.py` | Execuções do filtro de pistas sobre execuções de base |
| `scripts/regra_pura.py` | Execução da regra pura |
| `scripts/treinar.py` | Treino (e retomada) do classificador da tarefa; não lê o TEST |
| `scripts/avaliar_test.py` | Avaliação do TEST de uma execução de treino concluída |
| `dados/particoes/` | `train/dev/test.jsonl` congelados e `MANIFEST.json` (versionados) |
| `testes/` | Testes de unidade, de integração e de equivalência com o legado, e as referências do legado |
| `docs/entregas/` | Índice das entregas e, por etapa, o que foi entregue, como foi verificado e o que ficou pendente |

`dados/brutos/` (XML do SemClinBr) e `dados/processados/` (`dataset.jsonl`)
ficam fora do git: o corpus tem licença restrita. As partições versionadas
contêm o texto das notas, como no repositório original.

### Formato de uma execução

```
<raiz>/<nome>/                 ex.: resultados/execucoes/baseline_biobertpt_seed42/
    config.json                nome, estratégia, configuração e identidade dos conjuntos de referência
    treino.json                (execuções treinadas) histórico do DEV, melhor época, sessões, ambiente
    checkpoints/               (execuções treinadas) ultimo.pt, ultimo.json, melhor_modelo/
    predicoes_dev.json         sidecar do DEV  (model, seed, labels, y_true, y_pred[, probs, extras])
    predicoes_test.json        sidecar do TEST
    metricas.json              métricas por partição
    avaliacoes_test.jsonl      trilha das avaliações do TEST (eval_index, eval_index_for_config)
```

O sidecar é o do legado, sem mudança, e `execucao.diretorio.caminho_predicoes`
também encontra as predições no formato antigo (`<nome>.preds.json`,
`<nome>.dev_preds.json`). Os relatórios de comparação mantêm o nome e o formato
do legado (`significance_<A>_vs_<B>.json`).
