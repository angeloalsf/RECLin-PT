# RECLin-PT (minimo)

Versao enxuta para responder uma pergunta de pesquisa:

> **"O pre-treinamento clinico importa para extracao de relacoes em textos
> medicos em portugues?"**

Comparamos dois encoders em portugues na MESMA tarefa de extracao de relacao em
notas clinicas do **SemClinBr**, mudando **somente o checkpoint de pre-treino**:

- **BioBERTpt** (`pucpr/biobertpt-all`) — encoder **clinico**.
- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) — encoder **geral**
  (pre-treinado no brWaC).

Espaco de rotulos (3 classes): `negation_of`, `associated_with`, `no_relation`.
A metrica que responde "detecta bem negacao?" e o **F1 de `negation_of`** no
teste; reportamos tambem Micro-F1, Macro-F1, Weighted-F1, MCC, F1 por classe,
classification report e matriz de confusao -- e um **teste de significancia**
entre os dois modelos.

## Paridade entre os baselines

Os dois baselines compartilham **um unico nucleo** (`src/relation_extraction.py`),
entao sao identicos por construcao em tudo, exceto o modelo:

- Entity markers / representacao de entrada: `[E1] ... [/E1] [E2] ... [/E2]`
  (tokens especiais) + janela de contexto.
- Loss: CrossEntropy com `class_weight=balanced`.
- Scheduler: linear com warmup; early stopping pela melhor epoca no **dev**.
- Salvamento de checkpoints: `best_model/` (pesos do melhor epoch, formato HF) e
  `last_checkpoint/` (estado completo de retomada), gravados de forma atomica.
- Metricas: Micro-F1, Macro-F1, Weighted-F1, MCC, F1 por classe, classification
  report, matriz de confusao; curvas de treino/validacao (`train_loss` e
  `dev_loss` por epoca).
- Seed de reprodutibilidade (42) e logging estruturado centralizado.

Trocar BioBERTpt por BERTimbau e so mudar `--model`, `--ckpt-dir` e `--out`.

## Estrutura

```
SemClinBr-xml-public-v1/   # XMLs do corpus (voce coloca aqui; licenca restrita)
src/
  parse_semclinbr.py       # XML -> data/processed/dataset.jsonl
  candidates.py            # gera pares-candidatos (inclui os negativos no_relation)
  make_splits.py           # splits 80/10/10 doc-level, seed 42
  relation_extraction.py   # NUCLEO compartilhado: treino, metricas, checkpoints, logging
  baseline_biobertpt.py    # entry-point fino: BioBERTpt (clinico)
  baseline_bertimbau.py    # entry-point fino: BERTimbau (geral)
  significance.py          # McNemar + bootstrap pareado no F1 de negation_of
  utils/logger.py          # logging central -> terminal + logs/pipeline.log
scripts/
  make_tables.py           # results/*.json -> artigo-sbc/tables/*.tex
  make_figures.py          # results/*.json -> artigo-sbc/figs/*.pdf
  aggregate_seeds.py       # results/*.json -> results/summary_by_seed.json (multi-seed)
  _artifacts.py            # leitura/validacao compartilhada de results/
data/
  processed/dataset.jsonl  # gerado
  splits/{train,dev,test}.jsonl
  splits/MANIFEST.json     # SHA-256 + contagens + seed dos splits (versionado)
notebooks/
  01_baseline_biobertpt_colab.ipynb  # Colab T4: treina BioBERTpt com retomada
  02_baseline_bertimbau_colab.ipynb  # Colab T4: treina BERTimbau com retomada
  03_significance_colab.ipynb        # Colab CPU: McNemar + bootstrap (apos treinar os dois)
results/
  baseline_<modelo>_seed<N>.json        # metricas (modelo: biobertpt | bertimbau)
  baseline_<modelo>_seed<N>.preds.json  # predicoes do test (para significancia)
  significance_biobertpt_vs_bertimbau_seed<N>.json  # relatorio do teste pareado
  summary_by_seed.json                  # media/desvio entre as seeds (gerado)
artigo-sbc/                # artigo SBC: .tex/.bib/.bbl, figs/, tables/, entregas/
  README.md                # como regerar tabelas/figuras e compilar o artigo
logs/pipeline.log          # log da execucao local (gerado)
run.sh                     # pipeline fim a fim (parse -> splits -> baselines -> significancia)
Makefile                   # ALTERNATIVA opcional ao run.sh: rebuild incremental
requirements.txt           # dependencias (Python 3.10)
LICENSE                    # MIT (codigo; os dados tem licenca propria)
CITATION.cff               # metadados de citacao
RELATORIO_ORGANIZACAO.md   # auditoria inicial do repositorio (registro historico)
RELATORIO_LIMPEZA.md       # relatorio vigente de limpeza/consolidacao
.gitattributes             # normalizacao de fim de linha dos .json
```

## Pré-requisitos

**Python 3.10.** Foi a versão usada em todas as execuções deste repositório.
Versões mais novas não foram testadas — atenção se o `python` do seu PATH
for 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Corpus SemClinBr.** Não é distribuído aqui: é de **acesso restrito** e
precisa ser solicitado aos autores do corpus (Oliveira et al.). Coloque os
1.000 arquivos `.xml` em `SemClinBr-xml-public-v1/` na raiz do projeto.
A pasta está no `.gitignore` — os dados clínicos nunca devem ser versionados.

Você **não precisa do corpus** para reproduzir os baselines: os splits
derivados (`data/splits/*.jsonl`) são versionados no repositório, junto de
`data/splits/MANIFEST.json` — o SHA-256 de cada partição, para conferir que
são os mesmos arquivos que produziram os resultados. O corpus só é necessário
para reexecutar os passos 1 e 2 (parse e splits).

> **Licenças são separadas.** O **código** deste repositório é MIT (veja
> `LICENSE`). Os **dados** não são: o SemClinBr tem licença própria e restrita,
> não é distribuído aqui e precisa ser obtido separadamente pelo usuário junto
> aos autores do corpus. A licença MIT não se estende a eles.

**Hardware e tempo.** Treino validado em **GPU NVIDIA T4** (Google Colab,
~16 GB VRAM). Cada época leva ≈48 min; cada baseline (3 épocas) leva
**≈2h25**. Reproduzir os quatro experimentos do artigo (2 modelos × 2 seeds)
custa **≈10 h de GPU**. Em CPU o treino roda, mas é impraticável para o
tamanho do dataset (128.380 candidatos de treino).

Use `--ckpt-dir` apontando para o Google Drive: o treino grava
`last_checkpoint/` ao fim de cada época e **retoma automaticamente** se o
runtime do Colab cair. Siga a convenção `checkpoints/<modelo>_seed<N>` para as
execuções não colidirem entre si.

## Como rodar (local)

```bash
pip install -r requirements.txt

# 1) parse
python src/parse_semclinbr.py --xml-dir SemClinBr-xml-public-v1 \
    --out data/processed/dataset.jsonl

# 2) splits 80/10/10 (doc-level, seed 42) + data/splits/MANIFEST.json
#    ATENCAO: sobrescreve os splits versionados. Confira o MANIFEST no git diff.
python src/make_splits.py

# 3a) baseline CLINICO (BioBERTpt) -- hiperparametros do artigo
python src/baseline_biobertpt.py --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
    --ckpt-dir checkpoints/biobertpt_seed42 \
    --out results/baseline_biobertpt_seed42.json

# 3b) baseline GERAL (BERTimbau) -- MESMOS hiperparametros, so muda o modelo
python src/baseline_bertimbau.py --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
    --ckpt-dir checkpoints/bertimbau_seed42 \
    --out results/baseline_bertimbau_seed42.json

# 4) significancia (usa os *.preds.json gerados em 3a/3b)
python src/significance.py \
    --a results/baseline_biobertpt_seed42.preds.json \
    --b results/baseline_bertimbau_seed42.preds.json \
    --target negation_of --n-boot 10000 --seed 42 \
    --out results/significance_biobertpt_vs_bertimbau_seed42.json
```

> Os valores acima (`--batch-size 64 --max-gap 20 --max-length 128`) são os que
> produziram os números do artigo, e **não** coincidem com os defaults do
> `argparse` (32 / 75 / 192). Passe-os explicitamente. `max_gap=20` é a
> decisão de maior impacto: controla quantos pares negativos entram no
> dataset e, portanto, o desbalanceamento de classes.

ou simplesmente `bash run.sh` (passos 1, 2, 3a, 3b e 4 — e ainda o passo 5, que
regenera as tabelas e figuras do artigo). No Colab (GPU T4), use os notebooks
em `notebooks/`, na ordem do prefixo: rode primeiro os dois de treino
(`01_baseline_biobertpt_colab.ipynb` e `02_baseline_bertimbau_colab.ipynb`) —
cada um clona o repo, monta o Drive, treina com **retomada automatica** por
epoca, plota as curvas, publica `results/*.preds.json` e (opcional) envia o
modelo final ao Hugging Face Hub. Depois rode `03_significance_colab.ipynb`
(CPU, sem GPU/Drive) para o teste pareado entre os dois.

### `make` — alternativa opcional para rebuild incremental

`run.sh` continua sendo o caminho simples e nao vai a lugar nenhum: roda tudo,
na ordem, sem instalar nada. O `Makefile` e uma **alternativa** para quem quer
o contrario disso — nao refazer o que ja esta pronto e atualizado:

```bash
make -n all     # mostra o que SERIA executado, sem executar
make all        # roda so o que esta desatualizado
make status     # o que ja existe e o que falta, sem tocar em nada
make aggregate  # ou: make splits | baselines | significance | tables | figures
make baseline_biobertpt_seed42   # alvos individuais por modelo e semente
```

Cada alvo usa **exatamente** os comandos da secao acima e de
[`artigo-sbc/README.md`](artigo-sbc/README.md) — nenhuma flag nova. A diferenca
esta nas dependencias de arquivo: com os quatro `results/baseline_*.json` no
lugar, `make all` responde "Nothing to be done" em vez de gastar ~10 h de GPU
retreinando.

Duas escolhas de projeto que valem saber antes de usar:

- **`results/` e `data/splits/` nao dependem dos `.py`.** So dos dados de
  entrada. Assim um comentario corrigido em `relation_extraction.py` nao pede
  10 h de GPU, e tocar em `make_splits.py` nao reembaralha os splits
  versionados. Quando quiser forcar mesmo assim: `make -B <alvo>`.
- **`make clean-derived` nao apaga `results/`** — so tabelas, figuras e o
  `summary_by_seed.json`, que se refazem em segundos. Por isso nao se chama
  `clean`.

**No Windows `make` nao vem instalado** (precisa de Git Bash com make, WSL ou
Chocolatey). Se isso for um estorvo, ignore o `Makefile`: `bash run.sh` faz o
mesmo pipeline sem dependencia nova.

## Artigo (`artigo-sbc/`)

Como **regerar as tabelas e figuras** a partir de `results/*.json`
(`scripts/make_tables.py` e `scripts/make_figures.py`) e como **compilar o
artigo** com `latexmk` está documentado em [`artigo-sbc/README.md`](artigo-sbc/README.md),
junto do próprio artigo.

## Metricas e como interpretar

- Compare o **F1 de `negation_of`** e o **Macro-F1** dos dois `results/*.json`.
  Micro-F1, Weighted-F1 e accuracy sao dominados por `no_relation` (~99% dos
  pares) e servem so de contexto -- nao sao a manchete.
- **MCC** (`test_mcc`) e um numero unico robusto a desbalanceamento.
- **Curvas**: `dev_history` traz `train_loss` e `dev_loss` por epoca (overfitting)
  e `dev_macro_f1`/`dev_negation_of_f1` (selecao de epoca).
- **Significancia** (`src/significance.py`): McNemar diz se os padroes de erro
  diferem; o bootstrap pareado da o intervalo de 95% e o p-valor da diferenca no
  F1 de `negation_of`. Se o IC95 nao cruza zero, a vantagem e significativa.

### Multiplas seeds

Rode uma seed por execucao mudando `--seed`, `--out` e `--ckpt-dir` (para nao
colidir):

```bash
python src/baseline_bertimbau.py --seed 43 \
    --ckpt-dir .../checkpoints_bertimbau_seed43 \
    --out results/baseline_bertimbau_seed43.json
```

Depois agregue com `scripts/aggregate_seeds.py` — media e desvio-padrao
populacional do Macro-F1 e do F1 por classe, para os dois modelos:

```bash
python scripts/aggregate_seeds.py            # sementes 42 e 43
python scripts/aggregate_seeds.py --check    # so imprime, nao grava
```

Grava `results/summary_by_seed.json` com media, desvio e os valores por
semente que entraram na conta. Aceita `--seeds`, `--results-dir` e `--out`.
Nao treina nada: so le `results/baseline_*.json`.

> Com duas sementes, o desvio e **dispersao observada**, nao intervalo de
> confianca — ver "Limitacoes" abaixo.

## Decisoes principais

- **Split em nivel de documento** (nao de relacao): evita vazamento de
  vocabulario do mesmo prontuario entre train/test. Estratificado pela presenca
  de `negation_of`. Seed fixo 42.
- **Candidatos negativos**: o SemClinBr so anota relacoes positivas; geramos os
  pares `no_relation` (pares ordenados, janela `max_gap`). Direcao importa para
  `negation_of`.
- **Baseline**: marcadores de entidade tipados + janela de contexto, cabeca de
  classificacao em 3 classes, CrossEntropy com `class_weight=balanced`. Melhor
  epoca pelo macro-F1 no dev; teste reportado uma unica vez.

## Limitacoes conhecidas e proximos passos

**Ja resolvido (era escopo pendente):**

- **Hashes SHA dos splits.** `make_splits.py` grava
  `data/splits/MANIFEST.json` com SHA-256, contagem de documentos e contagem
  de relacoes por particao. O manifesto e versionado, entao regerar os splits
  por engano muda o hash e aparece no `git diff`. Antes a evidencia era so
  indireta (os quatro `.preds.json` tem exatamente 16.074 exemplos e
  `significance.py` aborta se os `y_true` divergirem).
- **Agregacao multi-seed automatizada.** `scripts/aggregate_seeds.py`
  substitui o calculo manual e grava `results/summary_by_seed.json`.

**Limitacoes de metodo (discutidas no artigo):**

- Duas sementes por modelo. O teste pareado foi refeito dentro de cada semente
  (42x42 e 43x43), mas duas execucoes nao bastam para estimar a variancia de
  inicializacao com intervalo de confianca.
- Uma unica arquitetura de cabeca de classificacao; sem busca de
  hiperparametros (por design — a busca quebraria a paridade).
- `max_gap=20` limita os pares candidatos a entidades proximas; relacoes de
  longa distancia estao fora do espaco de avaliacao.
