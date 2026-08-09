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
data/
  processed/dataset.jsonl  # gerado
  splits/{train,dev,test}.jsonl
notebooks/
  01_baseline_biobertpt_colab.ipynb  # Colab T4: treina BioBERTpt com retomada
  02_baseline_bertimbau_colab.ipynb  # Colab T4: treina BERTimbau com retomada
  03_significance_colab.ipynb        # Colab CPU: McNemar + bootstrap (apos treinar os dois)
results/
  baseline_{biobertpt,bertimbau}.json          # metricas
  baseline_{biobertpt,bertimbau}.preds.json    # predicoes do test (para significancia)
  significance_*.json                          # relatorio do teste pareado
run.sh                     # pipeline fim a fim (parse -> splits -> os dois baselines)
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
derivados (`data/splits/*.jsonl`) são versionados no repositório. O corpus só
é necessário para reexecutar os passos 1 e 2 (parse e splits).

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

# 2) splits 80/10/10 (doc-level, seed 42)
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

ou simplesmente `bash run.sh` (passos 1-3). No Colab (GPU T4), use os notebooks
em `notebooks/`, na ordem do prefixo: rode primeiro os dois de treino
(`01_baseline_biobertpt_colab.ipynb` e `02_baseline_bertimbau_colab.ipynb`) —
cada um clona o repo, monta o Drive, treina com **retomada automatica** por
epoca, plota as curvas, publica `results/*.preds.json` e (opcional) envia o
modelo final ao Hugging Face Hub. Depois rode `03_significance_colab.ipynb`
(CPU, sem GPU/Drive) para o teste pareado entre os dois.

## Regerar as tabelas e figuras do artigo

Dois scripts derivam os artefatos de `artigo-sbc/` diretamente de
`results/*.json`, em vez de transcrever os números à mão. **São eles que
produzem as tabelas e figuras que o artigo compila** — não há mais versão
manual paralela:

```bash
python scripts/make_tables.py      # -> artigo-sbc/tables/
python scripts/make_figures.py     # -> artigo-sbc/figs/
```

**`make_tables.py`** grava um fragmento `.tex` por tabela: `tab_resultados.tex`
(métricas dos dois baselines no teste, com o melhor valor de cada coluna em
negrito) e `tab_signif.tex` (comparação pareada — McNemar e bootstrap no F1 de
`negation_of`). Cada fragmento é um ambiente `table` completo, com `\caption` e
`\label`; `artigo.tex` os inclui por `\input{tables/tab_resultados.tex}` e
`\input{tables/tab_signif.tex}`.

**`make_figures.py`** grava os três PDFs do artigo: `f1_por_classe.pdf` (barras
agrupadas, F1 por classe nos dois baselines) e `cm_biobertpt.pdf` /
`cm_bertimbau.pdf` (matrizes de confusão normalizadas por linha — a diagonal é
o recall por classe). São exatamente os arquivos que `artigo.tex` referencia
via `\includegraphics{figs/...}`.

Os dois aceitam `--seed` (padrão 42, a semente do artigo), `--results-dir`
(padrão `results/`) e `--out-dir`. Além disso, `make_tables.py` aceita
`--check`, que imprime as tabelas no terminal sem gravar nada, e
`make_figures.py` aceita `--format {pdf,png,svg}` e `--dpi`. Nenhum dos dois
treina nada: só leem `results/` e escrevem nas pastas de saída.

> **Rodar sem argumentos sobrescreve os artefatos reais do artigo.** É o
> comportamento pretendido: os scripts são a fonte única, e `artigo-sbc/figs/`
> e `artigo-sbc/tables/` são saída derivada — não edite os `.tex` de
> `tables/` à mão, a próxima execução descarta a edição. Para só conferir sem
> tocar no artigo, gere em outro lugar com `--out-dir` (ex.:
> `--out-dir /tmp/conferencia`) e compare.

Trocar a semente do artigo é, portanto, um comando: `make_tables.py --seed 43 &&
make_figures.py --seed 43` regenera os cinco artefatos coerentes entre si. Os
`\label` (`tab:resultados`, `tab:signif`) e os nomes de arquivo não mudam com a
semente, então as remissões `\ref{}` do texto continuam válidas — mas o texto
corrido em volta discute os números da semente 42 e teria de ser revisto à mão.

## Compilar o artigo

**Pré-requisito:** uma distribuição TeX Live com `latexmk` (a compilação
publicada foi feita com TeX Live 2022/dev, pdfTeX 3.141592653-2.6-1.40.22 e
latexmk 4.76). O `sbc-template.sty` e o `sbc.bst` já estão no repositório, e o
`artigo.bbl` é versionado — não é preciso instalar nada da SBC à parte.

```bash
cd artigo-sbc
latexmk -pdf artigo.tex     # -> artigo-sbc/artigo.pdf
latexmk -c                  # apaga .aux/.log/.fls/... e mantém o .pdf
```

O `latexmk` resolve sozinho as passadas de `pdflatex` e `bibtex` até as
referências cruzadas estabilizarem. Rode-o **de dentro de `artigo-sbc/`**: os
caminhos em `artigo.tex` (`figs/...`, `tables/...`) são relativos a esse
diretório.

Se tiver acabado de mexer nos números, regenere os artefatos antes de compilar
(seção anterior) — o `latexmk` não sabe que eles derivam de `results/*.json` e
não os reconstrói sozinho.

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

### Multiplas seeds (manual, no Colab)

Rode uma seed por execucao mudando `--seed`, `--out` e `--ckpt-dir` (para nao
colidir), e agregue offline:

```bash
python src/baseline_bertimbau.py --seed 43 \
    --ckpt-dir .../checkpoints_bertimbau_seed43 \
    --out results/baseline_bertimbau_seed43.json
```

```python
import json, glob, statistics as st
v = [json.load(open(f))["test_f1_per_class"]["negation_of"]
     for f in glob.glob("results/baseline_bertimbau_seed*.json")]
print(f"negation_of F1: {st.mean(v):.4f} ± {st.pstdev(v):.4f} (n={len(v)})")
```

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

**Ainda nao feito, por escopo:**

- **Hashes SHA dos splits.** `make_splits.py` nao grava manifesto com o SHA de
  cada particao. Sem isso, nao ha prova criptografica de que o test usado nas
  quatro execucoes foi o mesmo arquivo (a evidencia hoje e indireta: os quatro
  `.preds.json` tem exatamente 16.074 exemplos e `significance.py` aborta se
  os `y_true` divergirem).
- **Agregacao multi-seed automatizada.** As seeds 42 e 43 ja foram rodadas para
  os dois modelos, mas media e desvio sao calculados a mao (snippet acima).
- ~~**Adocao das tabelas e figuras geradas.**~~ **Feito.** O caminho de
  `results/*.json` ate o `.tex`/`.pdf` esta fechado: `scripts/make_tables.py` e
  `scripts/make_figures.py` sao a fonte unica dos cinco artefatos. Antes da
  troca a saida foi conferida contra o que estava publicado — as duas tabelas
  batem celula a celula (inclusive quais valores estao em negrito) e as tres
  figuras batem por extracao de conteudo do PDF, com **zero divergencias**.
  `artigo.tex` agora puxa as tabelas por `\input{tables/...}` e as figuras
  geradas ocupam `artigo-sbc/figs/`.

**Limitacoes de metodo (discutidas no artigo):**

- Duas sementes por modelo. O teste pareado foi refeito dentro de cada semente
  (42x42 e 43x43), mas duas execucoes nao bastam para estimar a variancia de
  inicializacao com intervalo de confianca.
- Uma unica arquitetura de cabeca de classificacao; sem busca de
  hiperparametros (por design — a busca quebraria a paridade).
- `max_gap=20` limita os pares candidatos a entidades proximas; relacoes de
  longa distancia estao fora do espaco de avaliacao.
