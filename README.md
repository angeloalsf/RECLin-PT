# RECLin-PT — Extração de Relações em Notas Clínicas em Português

Trabalho de Conclusão de Curso (TCC) de **Angelo Antonio Lima Silveira Filho**,
Bacharelado em Sistemas de Informação — **IFES, Campus Cachoeiro de
Itapemirim**. Orientação: **Prof. Cristiano Colombo**.

## O que este trabalho investiga

A pergunta é simples de enunciar e difícil de responder sem um experimento
controlado:

> **Pré-treinamento em domínio clínico dá vantagem real sobre um encoder de
> domínio geral na extração de relações em notas clínicas em português?**

A intuição diz que sim: um modelo que já leu texto médico deveria entender
melhor um prontuário. Mas encoders clínicos em português são treinados com
muito menos dados que os de domínio geral, e essa troca — vocabulário
especializado *versus* volume de pré-treinamento — não é obviamente favorável a
nenhum dos lados.

Para responder, este repositório monta um **estudo controlado**: dois encoders
em português são treinados na **mesma tarefa**, com os **mesmos dados**, os
**mesmos hiperparâmetros** e a **mesma implementação**, mudando **somente o
checkpoint de pré-treino**:

- **BioBERTpt** (`pucpr/biobertpt-all`) — encoder **clínico**.
- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) — encoder de **domínio
  geral** (pré-treinado no brWaC).

A tarefa é classificação de relações entre pares de entidades em notas clínicas
do corpus **SemClinBr**, num espaço de **3 rótulos**: `negation_of`,
`associated_with` e `no_relation`.

A métrica que responde à pergunta central — *o modelo detecta bem negação?* — é
o **F1 da classe `negation_of`** no conjunto de teste. Também são reportados
Micro-F1, Macro-F1, Weighted-F1, MCC, F1 por classe, *classification report* e
matriz de confusão, mais um **teste de significância pareado** entre os dois
modelos. O artigo em `artigo-sbc/` discute os resultados; este README documenta
como reproduzi-los.

## Por que dois entry-points

Existem dois scripts de baseline — `src/baseline_biobertpt.py` e
`src/baseline_bertimbau.py` — mas eles **não são duas implementações**. Ambos são
entry-points finos sobre um **único núcleo compartilhado**,
`src/relation_extraction.py`. Isso é a garantia de **paridade**: os dois
baselines são idênticos por construção em tudo, exceto o modelo. Uma diferença
de desempenho não pode vir de uma diferença de implementação, porque não existe
uma.

O que o núcleo compartilhado fixa para os dois:

- **Representação de entrada:** marcadores de entidade tipados
  `[E1] ... [/E1] [E2] ... [/E2]` (tokens especiais) + janela de contexto.
- **Loss:** CrossEntropy com `class_weight=balanced`.
- **Otimização:** scheduler linear com warmup; seleção da melhor época pelo
  macro-F1 no **dev** (early stopping).
- **Checkpoints:** `best_model/` (pesos da melhor época, formato HF) e
  `last_checkpoint/` (estado completo de retomada), gravados de forma atômica.
- **Métricas:** Micro-F1, Macro-F1, Weighted-F1, MCC, F1 por classe,
  *classification report*, matriz de confusão e curvas de treino/validação
  (`train_loss` e `dev_loss` por época).
- **Reprodutibilidade:** seed fixo (42 por padrão) e logging estruturado
  centralizado.

Trocar BioBERTpt por BERTimbau é mudar `--model`, `--ckpt-dir` e `--out`. Nada
mais.

## Estrutura do repositório

```
SemClinBr-xml-public-v1/   # XMLs do corpus (você coloca aqui; licença restrita)
src/
  parse_semclinbr.py       # XML -> data/processed/dataset.jsonl
  candidates.py            # gera pares-candidatos (inclui os negativos no_relation)
  make_splits.py           # splits 80/10/10 doc-level, seed 42 + MANIFEST.json
  relation_extraction.py   # NÚCLEO compartilhado: treino, métricas, checkpoints, logging
  baseline_biobertpt.py    # entry-point fino: BioBERTpt (clínico)
  baseline_bertimbau.py    # entry-point fino: BERTimbau (geral)
  significance.py          # McNemar + bootstrap pareado no F1 de negation_of
  utils/logger.py          # logging central -> terminal + logs/pipeline.log
scripts/
  make_tables.py           # results/*.json -> artigo-sbc/tables/*.tex
  make_figures.py          # results/*.json -> artigo-sbc/figs/*.pdf
  aggregate_seeds.py       # results/*.json -> results/summary_by_seed.json (multi-seed)
  _artifacts.py            # leitura/validação compartilhada de results/
data/
  processed/dataset.jsonl  # gerado
  splits/{train,dev,test}.jsonl
  splits/MANIFEST.json     # SHA-256 + contagens + seed dos splits (versionado)
notebooks/
  01_baseline_biobertpt_colab.ipynb  # Colab T4: treina BioBERTpt com retomada
  02_baseline_bertimbau_colab.ipynb  # Colab T4: treina BERTimbau com retomada
  03_significance_colab.ipynb        # Colab CPU: McNemar + bootstrap (após treinar os dois)
results/
  baseline_<modelo>_seed<N>.json        # métricas (modelo: biobertpt | bertimbau)
  baseline_<modelo>_seed<N>.preds.json  # predições do test (para significância)
  significance_biobertpt_vs_bertimbau_seed<N>.json  # relatório do teste pareado
  summary_by_seed.json                  # média/desvio entre as seeds (gerado)
artigo-sbc/                # artigo SBC: .tex/.bib/.bbl, figs/, tables/, entregas/
  README.md                # como regerar tabelas/figuras e compilar o artigo
logs/pipeline.log          # log da execução local (gerado)
run.sh                     # pipeline fim a fim (parse -> splits -> baselines -> significância)
Makefile                   # ALTERNATIVA opcional ao run.sh: rebuild incremental
requirements.txt           # dependências (Python 3.10)
LICENSE                    # MIT (código; os dados têm licença própria)
CITATION.cff               # metadados de citação
.gitattributes             # normalização de fim de linha dos .json
```

**Convenção de nomes em `results/`.** Todo artefato experimental carrega o
modelo e a semente no nome: `baseline_<modelo>_seed<N>.json`, e as predições do
teste no arquivo irmão `.preds.json`. O relatório de significância nomeia o par
e a semente: `significance_biobertpt_vs_bertimbau_seed<N>.json`. O teste é
**pareado dentro da mesma semente** — nunca 42 contra 43.

## Pré-requisitos

**Python 3.10.** Foi a versão usada em todas as execuções deste repositório.
Versões mais novas não foram testadas — atenção se o `python` do seu PATH for
3.12+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Corpus SemClinBr.** Não é distribuído aqui: é de **acesso restrito** e precisa
ser solicitado aos autores do corpus (Oliveira et al.). Coloque os 1.000
arquivos `.xml` em `SemClinBr-xml-public-v1/` na raiz do projeto. A pasta está no
`.gitignore` — dados clínicos nunca devem ser versionados.

Você **não precisa do corpus** para reproduzir os baselines: os splits derivados
(`data/splits/*.jsonl`) são versionados, junto de `data/splits/MANIFEST.json` —
o SHA-256 de cada partição, para conferir que são os mesmos arquivos que
produziram os resultados. O corpus só é necessário para reexecutar os passos 1 e
2 (parse e splits).

**Hardware e tempo.** Treino validado em **GPU NVIDIA T4** (Google Colab,
~16 GB VRAM). Cada época leva ≈48 min; cada baseline (3 épocas) leva **≈2h25**.
Reproduzir os quatro experimentos do artigo (2 modelos × 2 sementes) custa
**≈10 h de GPU**. Em CPU o treino roda, mas é impraticável para o tamanho do
dataset (128.380 candidatos de treino — contagem de `max_gap=20`, a ser
recontada após o retreino com `max_gap=25`).

Use `--ckpt-dir` apontando para o Google Drive: o treino grava
`last_checkpoint/` ao fim de cada época e **retoma automaticamente** se o runtime
do Colab cair. Siga a convenção `checkpoints/<modelo>_seed<N>` para as execuções
não colidirem entre si.

## Como rodar — local

O pipeline completo, passo a passo. Estes são os comandos canônicos: `run.sh` e
o `Makefile` executam **exatamente** estes, sem nenhuma flag adicional.

```bash
pip install -r requirements.txt

# 1) parse
python src/parse_semclinbr.py --xml-dir SemClinBr-xml-public-v1 \
    --out data/processed/dataset.jsonl

# 2) splits 80/10/10 (doc-level, seed 42) + data/splits/MANIFEST.json
#    ATENÇÃO: sobrescreve os splits versionados. Confira o MANIFEST no git diff.
#    Para só recalcular os hashes, sem reembaralhar: --manifest-only
python src/make_splits.py --input data/processed/dataset.jsonl \
    --out-dir data/splits

# 3a) baseline CLÍNICO (BioBERTpt) -- hiperparâmetros do artigo
python src/baseline_biobertpt.py --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 25 --max-length 128 --seed 42 \
    --ckpt-dir checkpoints/biobertpt_seed42 \
    --out results/baseline_biobertpt_seed42.json

# 3b) baseline GERAL (BERTimbau) -- MESMOS hiperparâmetros, só muda o modelo
python src/baseline_bertimbau.py --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 25 --max-length 128 --seed 42 \
    --ckpt-dir checkpoints/bertimbau_seed42 \
    --out results/baseline_bertimbau_seed42.json

# 4) significância (usa os *.preds.json gerados em 3a/3b)
python src/significance.py \
    --a results/baseline_biobertpt_seed42.preds.json \
    --b results/baseline_bertimbau_seed42.preds.json \
    --target negation_of --n-boot 10000 --seed 42 \
    --out results/significance_biobertpt_vs_bertimbau_seed42.json

# 5) artefatos do artigo: tabelas (.tex) e figuras (.pdf) derivadas de results/
python scripts/make_tables.py
python scripts/make_figures.py

# 6) agregação multi-seed -> results/summary_by_seed.json (precisa das duas seeds)
python scripts/aggregate_seeds.py
```

> **Os hiperparâmetros do pipeline não são os defaults do `argparse`.**
> `--batch-size 64 --max-gap 25 --max-length 128` são os valores ativos; os
> defaults do `argparse` são **32 / 75 / 192**. Passe-os explicitamente.
> `max_gap=25` é a decisão de maior impacto: controla quantos pares negativos
> entram no dataset e, portanto, o desbalanceamento de classes.
>
> ⚠️ **`max_gap` mudou de 20 para 25 e os experimentos ainda não foram
> refeitos.** A janela de 20 caracteres descartava 1.324 relações anotadas
> (1.272 delas `associated_with`) antes de virarem candidato — ver
> `analysis/max_gap/`. Tudo o que está em `results/`, e todo número derivado
> dele neste README, no `artigo-sbc/` e no `tcc/`, ainda vem de `max_gap=20`.
> Os quatro experimentos (2 modelos × 2 sementes) precisam ser reexecutados
> antes de qualquer número ser atualizado.

### Duas formas de executar o pipeline

**`run.sh` — o caminho simples.** Roda os passos 1 a 6 na ordem, do início ao
fim, sem exigir nada além do Python e das dependências:

```bash
bash run.sh
```

**`make` — alternativa para rebuild incremental.** O `Makefile` existe para o
caso oposto: **não refazer o que já está pronto e atualizado**. Compara datas de
arquivo e só reexecuta o alvo cuja entrada está mais nova que a saída.

```bash
make -n all     # mostra o que SERIA executado, sem executar (rode ANTES de gastar GPU)
make all        # roda só o que está desatualizado
make status     # o que já existe e o que falta, sem tocar em nada
make help       # lista os alvos
make splits | baselines | significance | aggregate | tables | figures | artigo
make baseline_biobertpt_seed42   # alvos individuais por modelo e semente
make significance_seed42
```

Com os quatro `results/baseline_*.json` no lugar, `make all` responde "Nothing to
be done" em vez de gastar ~10 h de GPU retreinando.

Duas decisões de projeto do `Makefile` que vale conhecer antes de usar:

- **`results/` e `data/splits/` não dependem dos `.py`** — só dos dados de
  entrada. Assim um comentário corrigido em `relation_extraction.py` não pede
  10 h de GPU, e tocar em `make_splits.py` não reembaralha os splits
  versionados. Para forçar mesmo assim: `make -B <alvo>`.
- **`make clean-derived` não apaga `results/`** — só tabelas, figuras e o
  `summary_by_seed.json`, que se refazem em segundos. É por isso que o alvo não
  se chama `clean`.

**No Windows o `make` não vem instalado** (precisa de Git Bash com make, WSL ou
Chocolatey). Se isso for um estorvo, ignore o `Makefile`: `bash run.sh` faz o
mesmo pipeline sem dependência nova.

Se você mudar um hiperparâmetro, mude nos três lugares — README, `run.sh` e
`Makefile`. Eles precisam continuar concordando.

## Como rodar — Colab

Para quem não tem GPU local, `notebooks/` traz o pipeline pronto para o Colab.
Rode na ordem do prefixo:

1. **`01_baseline_biobertpt_colab.ipynb`** (GPU T4)
2. **`02_baseline_bertimbau_colab.ipynb`** (GPU T4)

Cada um clona o repositório, monta o Google Drive, treina com **retomada
automática** por época, plota as curvas de treino/validação, publica
`results/*.json` e `results/*.preds.json` e, opcionalmente, envia o modelo final
ao Hugging Face Hub.

3. **`03_significance_colab.ipynb`** (CPU — não precisa de GPU nem do Drive),
   depois que os dois baselines terminaram: roda o teste pareado entre eles.

## Artigo (`artigo-sbc/`)

Como **regerar as tabelas e figuras** a partir de `results/*.json`
(`scripts/make_tables.py` e `scripts/make_figures.py`) e como **compilar o
artigo** com `latexmk` está documentado em
[`artigo-sbc/README.md`](artigo-sbc/README.md), junto do próprio artigo.

Dois pontos que valem o aviso aqui na entrada:

- `artigo-sbc/tables/` e `artigo-sbc/figs/` são **saída derivada** de
  `results/*.json`. Não edite os `.tex` e `.pdf` gerados à mão — a próxima
  execução dos scripts descarta a edição.
- O `latexmk` **não sabe** que esses artefatos derivam de `results/`. Se você
  acabou de mexer nos números, regenere antes de compilar.

## Métricas e como interpretar

- Compare o **F1 de `negation_of`** e o **Macro-F1** dos dois `results/*.json`.
  Micro-F1, Weighted-F1 e accuracy são dominados por `no_relation` (~99% dos
  pares) e servem só de contexto — não são a manchete.
- **MCC** (`test_mcc`) é um número único robusto a desbalanceamento.
- **Curvas**: `dev_history` traz `train_loss` e `dev_loss` por época (para ver
  overfitting) e `dev_macro_f1` / `dev_negation_of_f1` (seleção de época).
- **Significância** (`src/significance.py`): o **McNemar** diz se os padrões de
  erro dos dois modelos diferem; o **bootstrap pareado** dá o intervalo de 95% e
  o p-valor da diferença no F1 de `negation_of`. Se o IC95 não cruza zero, a
  vantagem é significativa. O teste aborta se os `y_true` dos dois arquivos
  divergirem — é a checagem de que o pareamento é legítimo (16.074 exemplos de
  teste nos quatro `.preds.json` — contagem de `max_gap=20`, a ser recontada
  após o retreino com `max_gap=25`).

### Múltiplas sementes

Rode uma semente por execução, mudando `--seed`, `--out` e `--ckpt-dir` (para
não colidir):

```bash
python src/baseline_bertimbau.py --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 25 --max-length 128 --seed 43 \
    --ckpt-dir checkpoints/bertimbau_seed43 \
    --out results/baseline_bertimbau_seed43.json
```

Depois agregue com `scripts/aggregate_seeds.py` — média e desvio-padrão
populacional do Macro-F1 e do F1 por classe, para os dois modelos:

```bash
python scripts/aggregate_seeds.py            # sementes 42 e 43
python scripts/aggregate_seeds.py --check    # só imprime, não grava
```

Grava `results/summary_by_seed.json` com média, desvio e os valores por semente
que entraram na conta. Aceita `--seeds`, `--results-dir` e `--out`. Não treina
nada: só lê `results/baseline_*.json`.

> Com duas sementes, o desvio é **dispersão observada**, não intervalo de
> confiança — ver "Limitações" abaixo.

## Decisões principais

- **Split em nível de documento** (não de relação): evita vazamento de
  vocabulário do mesmo prontuário entre train/test. Estratificado pela presença
  de `negation_of`. Seed fixo 42, e o resultado registrado em
  `data/splits/MANIFEST.json`.
- **Candidatos negativos gerados, não anotados**: o SemClinBr só anota relações
  positivas; os pares `no_relation` são gerados como pares **ordenados** dentro
  de uma janela `max_gap`. A direção importa para `negation_of`.
- **Baseline único e compartilhado**: marcadores de entidade tipados + janela de
  contexto, cabeça de classificação em 3 classes, CrossEntropy com
  `class_weight=balanced`. Melhor época escolhida pelo macro-F1 no **dev**;
  teste reportado uma única vez.
- **Artefatos do artigo derivados por script**, nunca transcritos à mão: uma
  fonte única (`results/*.json`) para tabelas, figuras e números do texto.

## Limitações conhecidas e próximos passos

- **Duas sementes por modelo.** O teste pareado foi refeito dentro de cada
  semente (42×42 e 43×43), mas duas execuções não bastam para estimar a
  variância de inicialização com intervalo de confiança. Mais sementes é o
  próximo passo mais barato em valor por hora de GPU.
- **Uma única arquitetura de cabeça de classificação, sem busca de
  hiperparâmetros.** Isso é por design: uma busca por modelo quebraria a
  paridade que sustenta a comparação. O custo é que nenhum dos dois baselines
  está necessariamente no seu melhor ponto de operação.
- **`max_gap=25` limita os pares candidatos a entidades próximas.** Relações de
  longa distância estão fora do espaço de avaliação — o resultado vale para o
  regime de vizinhança curta. O valor era 20 e foi elevado para 25 após a
  análise de sensibilidade (`analysis/max_gap/`), que mostrou que 20 já cortava
  11,6% das relações anotadas; o teto de recall com 25 é maior, mas continua
  abaixo de 100%.
- **Três classes.** O espaço de rótulos é um recorte do SemClinBr; ampliá-lo
  muda a dificuldade da tarefa e pediria reexecutar tudo.

## Licença

O **código** e a **documentação** deste repositório são MIT — veja
[`LICENSE`](LICENSE).

> **Os dados têm licença separada.** O corpus **SemClinBr não é distribuído
> aqui**, possui licença própria e restrita, e deve ser obtido separadamente
> junto aos autores do corpus. A licença MIT deste repositório **não se estende a
> ele**.

Para citar este trabalho, use os metadados de [`CITATION.cff`](CITATION.cff).
