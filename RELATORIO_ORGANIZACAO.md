# Relatório de Organização — RECLin-PT

**Data da análise:** 7 de agosto de 2026
**Escopo:** diagnóstico do repositório `D:\angeloalsf\tcc\RECLin-PT` (remoto: `https://github.com/angeloalsf/RECLin-PT.git`)
**Natureza deste documento:** somente leitura e proposta. Nenhum arquivo do projeto foi movido, renomeado, editado ou removido. Este relatório é o único arquivo criado.

**Estado do git no momento da análise:** 1 único commit (`f26c40c "first commit"`, 07/08/2026, 31 arquivos, 5.651 linhas), branch `main`, sincronizado com `origin/main`. Único arquivo modificado no working tree: `gcm-diagnose.log`. Repositório com 23 MB no disco, dos quais 1,9 MB em `.git`.

---

## Sumário executivo — os 6 achados que mais importam

| # | Achado | Gravidade |
|---|--------|-----------|
| 1 | `gcm-diagnose.log` está **versionado** e contém fingerprint da máquina (usuário Windows, nome do computador, PATH completo, caminhos locais, e-mail do git). Sem tokens em claro, mas não deveria estar no repositório. | Alta |
| 2 | Os **notebooks apontam para outro repositório**: `REPO_NAME = 'RECLin-PT-Min'` → `github.com/angeloalsf/RECLin-PT-Min`. O remoto real é `RECLin-PT`. Rodar os notebooks como estão clona o projeto errado (ou falha). | Alta |
| 3 | **Três valores diferentes** para os mesmos hiperparâmetros: README diz `--batch-size 16`; os defaults do CLI dizem `32 / max_gap 75 / max_length 192`; notebooks e **os resultados reais** dizem `64 / 20 / 128`. O README não descreve o experimento que existe. | Alta |
| 4 | **Não existe script que gere as figuras** `artigo-sbc/figs/*.pdf` nem as tabelas do artigo. A ponte entre `results/*.json` e o artigo é manual e não reproduzível. | Alta |
| 5 | `results/` está **inteiramente no `.gitignore`** (404 KB, 9 arquivos) — a evidência empírica do TCC não está versionada, e nenhum arquivo de `results/` entrou no commit atual. Existe só no seu disco. | Alta |
| 6 | O artigo afirma que a robustez por semente foi feita **só para o BioBERTpt**, mas `results/baseline_bertimbau_seed43.json` existe (16/07, posterior ao artigo de 23/06). O artigo está atrás dos resultados. | Média |

---

## 1. Mapa completo do projeto (árvore real)

Árvore obtida por inspeção direta do disco em 07/08/2026 — **não** a do README.

```
RECLin-PT/                                    23 MB total
├── .git/                                     1,9 MB — histórico; 1 commit, remoto angeloalsf/RECLin-PT
├── .gitignore                                75 B — 6 regras (ver seção 7)
├── README.md                                 6,4 KB / 146 linhas — doc principal (ver seções 2 e 3)
├── requirements.txt                          82 B — 6 dependências, sem pin de versão de Python
├── run.sh                                    992 B / 25 linhas — pipeline fim a fim, DESATUALIZADO (ver seção 5)
├── gcm-diagnose.log                          6,7 KB / 178 linhas — LIXO ACIDENTAL (ver seção 6)
│
├── SemClinBr-xml-public-v1/                  8,3 MB — 1.000 arquivos .xml, corpus bruto
│   └── 8906.xml … 9905.xml                   dados de origem; licença restrita; gitignored (correto)
│
├── src/                                      132 KB — 1.202 linhas de Python
│   ├── parse_semclinbr.py                    143 linhas — XML → dataset.jsonl. Colapsa quebras de linha
│   │                                         preservando offsets; conta (não aborta) offsets divergentes.
│   ├── candidates.py                         45 linhas — MÓDULO, não script. Gera pares ordenados de
│   │                                         entidades e rotula `no_relation` os não anotados. Importado
│   │                                         por relation_extraction.py; nunca chamado direto pelo pipeline.
│   ├── make_splits.py                        130 linhas — splits 80/10/10 em nível de DOCUMENTO,
│   │                                         estratificados pela presença de `negation_of`, seed 42.
│   ├── relation_extraction.py                552 linhas — NÚCLEO. Concentra representação de entrada
│   │                                         ([E1]/[E2] + janela), loss ponderada, scheduler, seleção de
│   │                                         época por dev macro-F1, checkpoints atômicos, métricas,
│   │                                         sidecar .preds.json e o argparse compartilhado.
│   ├── baseline_biobertpt.py                 44 linhas — entry-point fino; só fixa `pucpr/biobertpt-all`.
│   ├── baseline_bertimbau.py                 47 linhas — entry-point fino; só fixa BERTimbau.
│   ├── significance.py                       167 linhas — McNemar exato (binomial) + bootstrap pareado
│   │                                         no F1 da classe-alvo. Roda em CPU, segundos.
│   ├── utils/
│   │   ├── __init__.py                       0 B — só marca o pacote.
│   │   └── logger.py                         74 linhas — get_logger(): console + logs/pipeline.log,
│   │                                         idempotente, caminho ancorado na raiz do repo (parents[2]).
│   │   └── __pycache__/                      2 .pyc — NÃO versionado (coberto pelo .gitignore).
│   └── __pycache__/                          7 .pyc, compilados com CPython 3.10 — ver nota abaixo.
│
├── data/
│   ├── processed/
│   │   └── dataset.jsonl                     5,7 MB — 1.000 docs, 11.458 relações. GERADO, gitignored.
│   └── splits/                               VERSIONADO no git — é o que garante que os dois baselines
│       │                                     vejam exatamente o mesmo teste.
│       ├── train.jsonl                       4,5 MB / 800 docs (533 com negação; 1.299 negation_of)
│       ├── dev.jsonl                         558 KB / 100 docs (67 com negação; 155 negation_of)
│       └── test.jsonl                        573 KB / 100 docs (67 com negação; 152 negation_of)
│
├── results/                                  404 KB — 9 arquivos. GITIGNORED e ausente do commit atual.
│   ├── baseline_biobertpt_seed42.json        2,7 KB — macro-F1 0,7065 | negation_of 0,7239 | MCC 0,4875
│   ├── baseline_biobertpt_seed43.json        2,7 KB — macro-F1 0,6646 | negation_of 0,6535
│   ├── baseline_bertimbau_seed42.json        2,7 KB — macro-F1 0,7043 | negation_of 0,7342 | MCC 0,4713
│   ├── baseline_bertimbau_seed43.json        2,7 KB — macro-F1 0,7062 | negation_of 0,7263
│   ├── baseline_*_seed4*.preds.json          4 × ~96 KB — y_true/y_pred do test (16.074 exemplos),
│   │                                         sidecar consumido por significance.py. Os 4 são distintos
│   │                                         (md5 conferido) — são 4 execuções reais, não cópias.
│   └── significance_biobertpt_vs_bertimbau.json  712 B — McNemar p=0,1825; bootstrap IC95
│                                             [-0,0469, +0,0262]; conclusão: EMPATE. Compara seed 42 × 42.
│
├── logs/
│   └── pipeline.log                          3,3 KB — trilha acumulada. GITIGNORED. Contém o parse
│                                             (17/06), os splits (17/06) e DUAS execuções de significância:
│                                             uma de 22/06 (smoke test, n=2000, rótulos "A-clinico"/
│                                             "B-geral", saída em results/_sig.json que NÃO EXISTE mais)
│                                             e a definitiva de 23/06 (n=16.074). Ver seção 2, item 17.
│
├── notebooks/                                32 KB — 3 notebooks, todos SEM outputs salvos (bom).
│   ├── baseline_biobertpt_colab.ipynb        21 células — clona repo, monta Drive, treina com retomada,
│   │                                         plota curvas, dá `git add -f` nos resultados, push, e
│   │                                         (opcional) envia best_model/ ao HF Hub.
│   ├── baseline_bertimbau_colab.ipynb        21 células — espelho exato do anterior, só troca o modelo.
│   └── significance_colab.ipynb              13 células — CPU, sem Drive. Clona, confere os .preds.json
│                                             e roda significance.py.
│
└── artigo-sbc/                               524 KB — NÃO MENCIONADO NO README (ver seção 2).
    ├── artigo.tex                            34 KB / 562 linhas — artigo completo formato SBC.
    ├── resumo-expandido.tex                  9,1 KB / 174 linhas — versão curta do mesmo trabalho.
    ├── artigo.bib                            5,7 KB — bibliografia BibTeX.
    ├── artigo.bbl                            4,7 KB — bibliografia COMPILADA (derivado do .bib).
    ├── artigo.docx                           121 KB — conversão Word; derivado, não fonte.
    │                                         [movido em 09/08/2026 → entregas/2026-06-23_artigo.docx]
    ├── reclin-pt_extracao_relacoes_semclinbr.pdf  262 KB — PDF compilado; derivado.
    │                                         [movido em 09/08/2026 → entregas/2026-06-24_artigo.pdf]
    ├── sbc-template.sty                      6,1 KB — template oficial SBC (arquivo de terceiros).
    ├── sbc.bst                               22 KB — estilo BibTeX oficial SBC (arquivo de terceiros).
    └── figs/                                 3 PDFs, ~16 KB cada — cm_biobertpt.pdf, cm_bertimbau.pdf,
                                              f1_por_classe.pdf. SEM SCRIPT GERADOR NO REPO.
```

### Notas de leitura da árvore

**Não existem no repositório** (bom saber, para não procurar): ambiente virtual (`venv/`, `.venv/`), `.ipynb_checkpoints/`, `Makefile`, `docs/`, `scripts/`, `tests/`, `.gitattributes`, `CITATION.cff`, `LICENSE`, `CLAUDE.md`, arquivos de configuração de editor.

**Nota sobre a versão do Python.** Os `.pyc` em `src/__pycache__/` são `cpython-310`, ou seja, o pipeline foi executado localmente com **Python 3.10**. Já o `PATH` registrado em `gcm-diagnose.log` mostra `C:\Python314` como Python do sistema. Se você rodar `python src/...` hoje na máquina de origem, provavelmente vai pegar o 3.14, não o 3.10 que produziu estes artefatos. Isso justifica documentar a versão (seção 3).

**Nota sobre `data/splits/` versionado.** É uma decisão acertada e vale explicitar no README: os splits ocupam 5,6 MB, ficam abaixo do limite do GitHub e são exatamente o que garante que BioBERTpt e BERTimbau sejam avaliados no mesmo conjunto congelado — inclusive no Colab, onde o corpus bruto não está disponível. Sem eles versionados, os notebooks não teriam como reproduzir a partição.

**Nota sobre custo computacional.** `dev_history` registra `duration_s` por época: ~2.870 s (≈48 min). Com 3 épocas, cada baseline leva **≈2h25 em GPU T4**. Quatro execuções (2 modelos × 2 seeds) ≈ **10 horas de GPU**. Esse número não está em lugar nenhum da documentação e é a informação mais importante para alguém que vá reexecutar.

---

## 2. Inconsistências entre o README e o repositório real

### 2.1 Arquivos citados no README que não existem com esse nome

| README diz | Realidade no disco |
|---|---|
| `results/baseline_{biobertpt,bertimbau}.json` | Só existem com sufixo de seed: `baseline_biobertpt_seed42.json`, `_seed43`, idem BERTimbau |
| `results/baseline_{biobertpt,bertimbau}.preds.json` | Idem: só `..._seed42.preds.json` / `..._seed43.preds.json` |
| `python src/significance.py --a results/baseline_biobertpt.preds.json --b results/baseline_bertimbau.preds.json` | **Esse comando falha hoje** — nenhum dos dois caminhos existe |

Consequência prática: os comandos das linhas 86–91 do README não rodam num clone atual do repositório.

### 2.2 Presentes no disco, ausentes do README

1. **`artigo-sbc/` inteiro** — 524 KB, o artigo do TCC, com `artigo.tex`, `resumo-expandido.tex`, bibliografia, template SBC e 3 figuras. É a **saída final do trabalho** e o README não o menciona uma única vez.
2. **`logs/`** — o README fala em "logging estruturado centralizado" (linha 35) e cita `logs/pipeline.log` na árvore da seção Estrutura via `utils/logger.py`, mas a pasta `logs/` não aparece como item da árvore.
3. **`requirements.txt`** — usado no comando `pip install -r requirements.txt`, mas ausente da árvore da seção Estrutura.
4. **`gcm-diagnose.log`** — arquivo na raiz, versionado, não documentado (é lixo; ver seção 6).
5. **`data/processed/` e `data/splits/` — política de versionamento** — o README lista os dois na árvore sem dizer que `processed/` é gitignored e `splits/` é versionado. Essa é uma distinção importante para quem clona.
6. **`src/utils/__init__.py`** — trivial, mas a árvore do README só cita `utils/logger.py`.

### 2.3 Divergências de hiperparâmetros — a inconsistência mais séria

Três fontes discordam sobre os mesmos parâmetros:

| Parâmetro | README / `run.sh` | Default do CLI (`relation_extraction.py:336-341`) | Notebooks Colab | **`config` gravado nos results (o que realmente rodou)** |
|---|---|---|---|---|
| `--batch-size` | **16** | 32 | **64** | **64** |
| `--max-gap` | (não passa) | 75 | **20** | **20** |
| `--max-length` | (não passa) | 192 | **128** | **128** |
| `--ctx-chars` | (não passa) | 128 | (não passa) | 128 |
| `--epochs` | 3 | 3 | 3 | 3 |
| `--lr` | (não passa) | 2e-5 | (não passa) | 2e-5 |

Ou seja: **quem seguir o README treina um modelo diferente do que gerou os números do artigo.** E o `max_gap` é especialmente relevante — 20 vs. 75 muda drasticamente quantos pares negativos entram no dataset, portanto muda a distribuição de classes e as métricas.

Observação adicional: o docstring de `candidates.py` documenta `max_gap` default 75 e o de `baseline_biobertpt.py` sugere `--batch-size 32` no exemplo de uso. Nenhum dos dois bate com a execução real.

### 2.4 Divergências nos notebooks

7. **Repositório errado.** Os três notebooks definem `REPO_NAME = 'RECLin-PT-Min'` e clonam `https://{TOKEN}@github.com/angeloalsf/RECLin-PT-Min`. O remoto deste repositório é `angeloalsf/RECLin-PT`. Se `RECLin-PT-Min` ainda existir no GitHub, os notebooks operam num projeto paralelo; se não existir, falham no clone.
8. **Nomes de saída sem seed.** As células de treino gravam `results/baseline_bertimbau.json` / `baseline_biobertpt.json`; a célula de push dá `git add -f` nesses mesmos nomes; e `significance_colab.ipynb` procura `results/baseline_biobertpt.preds.json`. **Nenhum desses arquivos existe no disco** — os arquivos reais têm `_seed42`/`_seed43`. O renomeio foi manual, fora do fluxo documentado.
9. **`--ckpt-dir` aponta para `MyDrive/RECLin-PT-Min/checkpoints_*`** — coerente com o nome errado do repo, e não documentado no README (que nunca menciona `--ckpt-dir` na seção "Como rodar", só na de multi-seed).
10. **Dependências não declaradas.** Os notebooks usam `matplotlib` (células de curvas) e `huggingface_hub` (upload opcional). Nenhum dos dois está em `requirements.txt`.
11. **E-mail e usuário embutidos.** `GIT_USER_NAME = 'angeloalsf'` e `GIT_USER_EMAIL = 'angeloalsf@gmail.com'` estão hard-coded nas três primeiras células. O token vem de `userdata.get('GITHUB_PAT')` (correto, não hard-coded) — mas o `git remote set-url origin` grava a URL **com o PAT embutido** no `.git/config` do clone do Colab. É efêmero, mas vale um comentário no notebook alertando disso.

### 2.5 Pastas vazias e arquivos órfãos

12. **Nenhuma pasta vazia** foi encontrada — todas contêm ao menos um arquivo.
13. **`results/_sig.json` não existe.** O `logs/pipeline.log` registra, em 22/06/2026, uma execução de significância que gravou `results/_sig.json` com `A=A-clinico`, `B=B-geral`, `n_test=2000`, McNemar `p=3.772e-06` e conclusão **"É significativa"** — resultado **oposto** ao definitivo. É claramente um smoke test com dados sintéticos, mas está no log logo acima do resultado real e **pode ser lido por engano como o resultado do trabalho**. Vale uma nota no relatório de execução ou a limpeza dessa entrada.

### 2.6 Desatualização entre o artigo e os resultados

14. `artigo.tex` (linhas 510-513) afirma: *"a análise de robustez por semente foi conduzida apenas para o BioBERTpt"* e trata isso como limitação declarada. Mas `results/baseline_bertimbau_seed43.json` existe, datado de **16/07/2026** — três semanas depois do artigo (23/06). Os números: BERTimbau seed 43 → macro-F1 0,7062, negation_of 0,7263.

    Isso **fortalece** a tese do artigo: enquanto o BioBERTpt oscila de 0,724 → 0,653 entre seeds (Δ=0,071), o BERTimbau oscila de 0,734 → 0,726 (Δ=0,008). O encoder geral é não só equivalente como **mais estável**. Essa é uma conclusão nova que o artigo ainda não incorpora — e reforça diretamente a frase "a variação entre sementes supera a diferença entre eles".

15. `results/significance_biobertpt_vs_bertimbau.json` compara **apenas seed 42 × seed 42**. Não há teste de significância envolvendo a seed 43. Se o artigo for atualizado com o BERTimbau seed 43, provavelmente convém rodar também `significance` nessa combinação.

### 2.7 A seção "O que foi deixado de fora (de propósito)"

O README (linhas 143-146) lista três itens. Situação real:

| Item | Ainda é verdade? |
|---|---|
| Hashes SHA dos splits | ✅ **Sim.** `make_splits.py` inclusive documenta a ausência no docstring ("Sem hashes SHA por opção de escopo"). |
| Intervalos de confiança multi-seed automatizados | ⚠️ **Parcialmente falso.** A seed 43 **já foi rodada para os dois modelos** (4 arquivos de resultado no disco). O que falta é a *automação* e a *agregação*; a execução manual já aconteceu. A frase precisa ser reescrita. |
| Geração de tabelas LaTeX | ⚠️ **Parcialmente falso.** O artigo **já tem** as tabelas e as 3 figuras (`figs/*.pdf`). O que falta é o *script* que as gera a partir de `results/*.json` — hoje o caminho de `results/` até o artigo é 100% manual, e **não há registro de como as figuras foram produzidas**. Isso é pior do que "deixado de fora": é uma dependência não reproduzível. |

---

## 3. Sugestões de atualização do `README.md` (propostas — não aplicadas)

O README atual tem 146 linhas e uma estrutura boa. Os problemas são de **atualidade** e de **completude para reexecução**, não de organização.

### 3.1 Ordem sugerida das seções

Atual: Título → Paridade → Estrutura → Como rodar → Métricas → Multi-seed → Decisões → Deixado de fora.

Proposta:

1. Título e pergunta de pesquisa *(manter)*
2. **Resultado principal** — 3 linhas com o achado (empate estatístico) e ponteiro para o artigo. **Seção nova.** Quem abre o repositório quer saber a conclusão antes do método.
3. **Pré-requisitos** — Python, hardware, tempo, corpus, licença. **Seção nova** (ver 3.4).
4. Como rodar *(corrigir hiperparâmetros)*
5. Estrutura *(atualizar; incluir `artigo-sbc/`, `logs/`, `requirements.txt`, política de gitignore)*
6. Paridade entre os baselines *(manter quase intacta — é a melhor parte do README)*
7. Métricas e como interpretar *(manter)*
8. Múltiplas seeds *(atualizar: já foram rodadas)*
9. Decisões principais *(manter; acrescentar a decisão de `max_gap=20`)*
10. Limitações e próximos passos *(renomear "O que foi deixado de fora")*

### 3.2 Antes/depois — trechos prontos para colar

---

**(A) Seção nova, logo após a pergunta de pesquisa — linha 20**

> **DEPOIS** *(inserir)*
>
> ```markdown
> ## Resultado principal
>
> No conjunto de teste congelado (16.074 pares candidatos, 100 documentos), os
> dois encoders **empatam estatisticamente** no F1 de `negation_of`:
> BioBERTpt 0,724 vs. BERTimbau 0,734 (McNemar p=0,18; bootstrap pareado
> IC95 [-0,047; +0,026]). A variação entre sementes de um mesmo modelo é maior
> que a diferença entre os modelos — no BioBERTpt o F1 de negação cai de 0,724
> (seed 42) para 0,653 (seed 43).
>
> Ou seja: **o pré-treinamento clínico não trouxe ganho mensurável nesta tarefa**,
> e o encoder geral, mais barato e mais disponível, é a escolha razoável para a
> continuação do RECLin-PT. Detalhamento completo em [`artigo-sbc/artigo.tex`](artigo-sbc/artigo.tex).
> ```

---

**(B) Seção nova de pré-requisitos — antes de "Como rodar"**

> **DEPOIS** *(inserir)*
>
> ```markdown
> ## Pré-requisitos
>
> **Python 3.10.** Foi a versão usada em todas as execuções deste repositório.
> Versões mais novas não foram testadas — atenção se o `python` do seu PATH
> for 3.12+.
>
> ```bash
> python -m venv .venv
> source .venv/bin/activate      # Windows: .venv\Scripts\activate
> pip install -r requirements.txt
> ```
>
> **Corpus SemClinBr.** Não é distribuído aqui: é de **acesso restrito** e
> precisa ser solicitado aos autores do corpus (Oliveira et al.). Coloque os
> 1.000 arquivos `.xml` em `SemClinBr-xml-public-v1/` na raiz do projeto.
> A pasta está no `.gitignore` — os dados clínicos nunca devem ser versionados.
>
> Você **não precisa do corpus** para reproduzir os baselines: os splits
> derivados (`data/splits/*.jsonl`) são versionados no repositório. O corpus só
> é necessário para reexecutar os passos 1 e 2 (parse e splits).
>
> **Hardware e tempo.** Treino validado em **GPU NVIDIA T4** (Google Colab,
> ~16 GB VRAM). Cada época leva ≈48 min; cada baseline (3 épocas) leva
> **≈2h25**. Reproduzir os quatro experimentos do artigo (2 modelos × 2 seeds)
> custa **≈10 h de GPU**. Em CPU o treino roda, mas é impraticável para o
> tamanho do dataset (128.380 candidatos de treino).
>
> Use `--ckpt-dir` apontando para o Google Drive: o treino grava
> `last_checkpoint/` ao fim de cada época e **retoma automaticamente** se o
> runtime do Colab cair.
> ```

---

**(C) Comandos de "Como rodar" — hiperparâmetros e nomes de arquivo**

> **ANTES** *(linhas 78-91)*
>
> ```bash
> # 3a) baseline CLINICO (BioBERTpt)
> python src/baseline_biobertpt.py --epochs 3 --batch-size 16 \
>     --out results/baseline_biobertpt.json
>
> # 3b) baseline GERAL (BERTimbau) -- MESMOS hiperparametros
> python src/baseline_bertimbau.py --epochs 3 --batch-size 16 \
>     --out results/baseline_bertimbau.json
>
> # 4) significancia (usa os *.preds.json gerados em 3a/3b)
> python src/significance.py \
>     --a results/baseline_biobertpt.preds.json \
>     --b results/baseline_bertimbau.preds.json \
>     --target negation_of \
>     --out results/significance_biobertpt_vs_bertimbau.json
> ```
>
> **DEPOIS**
>
> ```bash
> # 3a) baseline CLINICO (BioBERTpt) -- hiperparametros do artigo
> python src/baseline_biobertpt.py --splits-dir data/splits \
>     --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
>     --ckpt-dir checkpoints/biobertpt_seed42 \
>     --out results/baseline_biobertpt_seed42.json
>
> # 3b) baseline GERAL (BERTimbau) -- MESMOS hiperparametros, so muda o modelo
> python src/baseline_bertimbau.py --splits-dir data/splits \
>     --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
>     --ckpt-dir checkpoints/bertimbau_seed42 \
>     --out results/baseline_bertimbau_seed42.json
>
> # 4) significancia (usa os *.preds.json gerados em 3a/3b)
> python src/significance.py \
>     --a results/baseline_biobertpt_seed42.preds.json \
>     --b results/baseline_bertimbau_seed42.preds.json \
>     --target negation_of --n-boot 10000 --seed 42 \
>     --out results/significance_biobertpt_vs_bertimbau.json
> ```
>
> **Acrescentar logo abaixo:**
>
> ```markdown
> > Os valores acima (`--batch-size 64 --max-gap 20 --max-length 128`) são os que
> > produziram os números do artigo, e **não** coincidem com os defaults do
> > `argparse` (32 / 75 / 192). Passe-os explicitamente. `max_gap=20` é a
> > decisão de maior impacto: controla quantos pares negativos entram no
> > dataset e, portanto, o desbalanceamento de classes.
> ```

---

**(D) Bloco "Estrutura" — versão atualizada**

> **ANTES** *(linhas 41-64)* — não menciona `artigo-sbc/`, `logs/`, `requirements.txt`; usa nomes de `results/` que não existem.
>
> **DEPOIS**
>
> ```
> SemClinBr-xml-public-v1/   # XMLs do corpus (voce coloca aqui; licenca restrita; gitignored)
> src/
>   parse_semclinbr.py       # XML -> data/processed/dataset.jsonl
>   candidates.py            # MODULO: gera pares-candidatos (inclui os negativos no_relation)
>   make_splits.py           # splits 80/10/10 doc-level, seed 42
>   relation_extraction.py   # NUCLEO compartilhado: treino, metricas, checkpoints, logging
>   baseline_biobertpt.py    # entry-point fino: BioBERTpt (clinico)
>   baseline_bertimbau.py    # entry-point fino: BERTimbau (geral)
>   significance.py          # McNemar + bootstrap pareado no F1 de negation_of
>   utils/logger.py          # logging central -> terminal + logs/pipeline.log
> data/
>   processed/dataset.jsonl  # GERADO a partir dos XMLs; gitignored
>   splits/{train,dev,test}.jsonl   # VERSIONADO: e o que congela o teste entre os dois modelos
> notebooks/
>   baseline_biobertpt_colab.ipynb   # Colab T4: treina BioBERTpt com retomada (~2h25)
>   baseline_bertimbau_colab.ipynb   # Colab T4: treina BERTimbau com retomada (~2h25)
>   significance_colab.ipynb         # Colab CPU: McNemar + bootstrap (apos treinar os dois)
> results/                   # metricas, predicoes e teste de significancia (ver convencao de nomes)
>   baseline_<modelo>_seed<N>.json         # metricas do test + dev_history + config usada
>   baseline_<modelo>_seed<N>.preds.json   # y_true/y_pred do test, para significance.py
>   significance_<A>_vs_<B>.json           # relatorio do teste pareado
> logs/pipeline.log          # trilha acumulada de todas as execucoes (append)
> artigo-sbc/                # artigo do TCC (formato SBC) + resumo expandido + figuras
>   artigo.tex               # fonte principal; artigo.bbl/.docx/.pdf sao DERIVADOS
>   resumo-expandido.tex     # versao curta do mesmo trabalho
>   figs/*.pdf               # matrizes de confusao e F1 por classe
> requirements.txt           # dependencias Python (Python 3.10)
> run.sh                     # pipeline fim a fim (parse -> splits -> os dois baselines)
> ```

---

**(E) Seção "Múltiplas seeds" — deixou de ser hipotética**

> **ANTES** *(linhas 113-116)*
>
> ```markdown
> ### Multiplas seeds (manual, no Colab)
>
> Rode uma seed por execucao mudando `--seed`, `--out` e `--ckpt-dir` (para nao
> colidir), e agregue offline:
> ```
>
> **DEPOIS**
>
> ```markdown
> ### Multiplas seeds
>
> Ja foram executadas as seeds **42 e 43 para os dois modelos** (quatro arquivos
> em `results/`). A agregacao ainda e manual: rode uma seed por execucao mudando
> `--seed`, `--out` e `--ckpt-dir` (para nao colidir), e agregue com o snippet
> abaixo.
>
> | Modelo | seed 42 | seed 43 | amplitude |
> |---|---|---|---|
> | BioBERTpt (clinico) | 0,724 | 0,653 | **0,071** |
> | BERTimbau (geral)   | 0,734 | 0,726 | **0,008** |
>
> *(F1 de `negation_of` no test.)* A oscilacao do BioBERTpt entre sementes e
> ~7x maior que a diferenca entre os dois modelos na seed 42 (0,010) — e o
> BERTimbau e visivelmente mais estavel. E o principal argumento para a leitura
> de empate.
> ```

---

**(F) Seção final — renomear e atualizar**

> **ANTES** *(linhas 143-146)*
>
> ```markdown
> ## O que foi deixado de fora (de proposito)
>
> Hashes SHA dos splits, intervalos de confianca multi-seed automatizados e
> geracao de tabelas LaTeX. Sao adicoes diretas depois.
> ```
>
> **DEPOIS**
>
> ```markdown
> ## Limitacoes conhecidas e proximos passos
>
> **Ainda nao feito, por escopo:**
> - **Hashes SHA dos splits.** `make_splits.py` nao grava manifesto com o SHA de
>   cada particao. Sem isso, nao ha prova criptografica de que o test usado nas
>   quatro execucoes foi o mesmo arquivo (a evidencia hoje e indireta: os quatro
>   `.preds.json` tem exatamente 16.074 exemplos e `significance.py` aborta se
>   os `y_true` divergirem).
> - **Agregacao multi-seed automatizada.** As seeds 42 e 43 ja foram rodadas para
>   os dois modelos, mas media e desvio sao calculados a mao (snippet acima).
> - **Geracao automatica de tabelas e figuras LaTeX.** As tabelas e as tres
>   figuras de `artigo-sbc/figs/` foram produzidas fora deste repositorio; nao ha
>   script que va de `results/*.json` ate o `.tex`/`.pdf`. **Esta e a maior lacuna
>   de reprodutibilidade do projeto hoje.**
>
> **Limitacoes de metodo (discutidas no artigo):**
> - Teste de significancia so na combinacao seed 42 x seed 42.
> - Uma unica arquitetura de cabeca de classificacao; sem busca de
>   hiperparametros (por design — a busca quebraria a paridade).
> - `max_gap=20` limita os pares candidatos a entidades proximas; relacoes de
>   longa distancia estao fora do espaco de avaliacao.
> ```

---

### 3.3 Duplicação de informação a resolver

- A **lista de itens mantidos idênticos entre os baselines** aparece três vezes, quase palavra por palavra: README linhas 26-35, docstring de `relation_extraction.py` linhas 20-40, e os markdowns de abertura dos dois notebooks. Sugestão: manter a versão canônica no docstring de `relation_extraction.py` (fica junto do código que a garante) e, no README e nos notebooks, deixar um resumo de 3 linhas + link.
- A **seção "Métricas e como interpretar"** (README linhas 101-111) repete a explicação de McNemar e bootstrap que já está, com mais detalhe e melhor escrita, no docstring de `significance.py`. Cortar o README para o essencial ("IC95 que não cruza zero = significativo") e apontar para o docstring.

### 3.4 Informação que falta para reexecução do zero

Consolidada em **(B)** acima. Recapitulando o que hoje está ausente e é bloqueante: versão do Python (3.10), como obter o SemClinBr e sob qual licença, o fato de que os splits versionados dispensam o corpus, hardware (T4), tempo (~2h25/baseline, ~10 h no total), e o mecanismo de retomada via `--ckpt-dir`. Também vale acrescentar `matplotlib` e `huggingface_hub` a `requirements.txt` (ou criar um `requirements-notebooks.txt`).

---

## 4. Padronização de nomes e estrutura de pastas (proposta)

### 4.1 Convenções de nomenclatura

**Resultados** — a convenção *de facto* já em uso no disco funciona bem; falta só formalizá-la e alinhar o código a ela:

```
results/baseline_<modelo>_seed<N>.json          # metricas
results/baseline_<modelo>_seed<N>.preds.json    # predicoes do test (sidecar automatico)
results/significance_<A>_vs_<B>[_seed<N>].json  # teste pareado
```

- `<modelo>` ∈ `{biobertpt, bertimbau}` — minúsculo, sem `-`, curto.
- `<N>` sempre explícito, **inclusive para a seed 42**. Nada de "sem sufixo = seed padrão"; foi exatamente isso que gerou a divergência entre notebooks e disco.
- O `.preds.json` é gerado automaticamente por `relation_extraction.py` via `Path(args.out).with_suffix(".preds.json")` — basta acertar `--out` e o sidecar sai com o nome certo. **Nenhuma mudança de código é necessária.**
- Ganho colateral: a convenção deixa o glob `results/baseline_bertimbau_seed*.json` do snippet de agregação do README funcionar corretamente.

**Checkpoints:**

```
checkpoints/<modelo>_seed<N>/best_model/        # pesos do melhor epoch (save_pretrained)
checkpoints/<modelo>_seed<N>/last_checkpoint/   # estado completo de retomada
```

Hoje os notebooks usam `MyDrive/RECLin-PT-Min/checkpoints_biobertpt` (sem seed) — duas seeds do mesmo modelo colidem no mesmo `last_checkpoint/`. Incluir a seed no caminho resolve sem tocar no código. `checkpoints/` inteiro vai para o `.gitignore`.

> **Correção (versão anterior deste relatório).** Estava escrito aqui que `load_last_checkpoint` "compara a config, mas **não** a seed" e por isso "pode retomar da seed errada silenciosamente". Isso está errado: `_config_guard` já incluía `"seed": args.seed` desde o início, e o `mismatch` itera todas as chaves do guard, seed inclusive. O comportamento real com seed divergente era `log.warning("config divergente em ['seed']")` seguido de **descarte do checkpoint e treino do zero** — nunca uma retomada com a seed errada.
>
> O risco real, portanto, não era inconsistência numérica: era **perder progresso de treino sem perceber**. Horas de GPU descartadas e o `last_checkpoint` sobrescrito, sinalizados apenas por um WARNING no meio do log — fácil de não ver numa saída longa de Colab.
>
> **Corrigido nesta sessão.** `save_last_checkpoint` passou a gravar `"seed"` também no topo do payload, e `load_last_checkpoint` ganhou uma guarda de seed **antes** da comparação de config: seed divergente agora levanta `RuntimeError` explícito (com o `--ckpt-dir` sugerido já com o sufixo da seed correta) em vez de warning + descarte silencioso. Checkpoints antigos sem seed registrada apenas emitem aviso e seguem, para não invalidar o que já está em disco/no Drive. A comparação de config existente ficou intocada.

**Notebooks:**

```
notebooks/01_baseline_biobertpt_colab.ipynb
notebooks/02_baseline_bertimbau_colab.ipynb
notebooks/03_significance_colab.ipynb
```

O prefixo numérico codifica a ordem de execução — que hoje só está escrita em prosa no README e no markdown do notebook 3. Regra fixa: **notebooks sempre commitados sem outputs** (já é o caso; formalizar).

**Logs:**

```
logs/pipeline.log                       # trilha acumulada local (gitignored, como hoje)
experiments/<data>_<modelo>_seed<N>/train.log   # log da rodada especifica, versionado
```

### 4.2 Onde separar melhor

| Proposta | Motivo |
|---|---|
| **`docs/`** | Mover para lá o conteúdo pesado do README: interpretação de métricas, decisões de método, glossário de hiperparâmetros. `docs/metricas.md`, `docs/decisoes.md`. O README fica com ~60 linhas: pergunta → resultado → pré-requisitos → como rodar → mapa. |
| **`scripts/`** | Separar de `src/` o que é **ferramental de apoio** e não faz parte do pipeline científico: `scripts/make_figures.py`, `scripts/make_tables.py`, `scripts/aggregate_seeds.py`. `src/` fica com o que responde à pergunta de pesquisa. Isso preserva a legibilidade de `src/` — hoje uma das qualidades reais do projeto (7 arquivos, papéis nítidos). |
| **`experiments/`** | Uma pasta por rodada, para não perder histórico (detalhado em 4.3). |
| **`checkpoints/`** | Raiz do projeto, gitignored. Hoje os checkpoints só existem no Drive e nada no repositório documenta isso. |
| **Manter `artigo-sbc/` como está** | Está bem organizado. Só vale um `artigo-sbc/README.md` de 5 linhas dizendo qual é a fonte (`artigo.tex`) e quais são derivados (`.bbl`, `.docx`, `.pdf`), e um `.gitignore` local para os artefatos de compilação LaTeX. |

### 4.3 Estrutura para acomodar as extensões já sinalizadas

As três extensões do README ("hashes SHA", "multi-seed automatizado", "tabelas LaTeX") têm uma coisa em comum: todas produzem **artefatos derivados de `results/`**. Se cada uma escrever na raiz ou em `results/`, a pasta vira um depósito indistinguível em poucas semanas. A proposta é separar por **camada de derivação**:

```
data/splits/
  train.jsonl  dev.jsonl  test.jsonl
  MANIFEST.json            # <- EXTENSAO 1: sha256 + n_docs + n_relacoes por particao,
                           #    seed e versao do parser. Versionado. Gerado por make_splits.py.
                           #    Fica JUNTO dos dados que descreve, nao em results/.

experiments/               # <- EXTENSAO 2: uma pasta por rodada, imutavel apos gravada
  2026-06-23_biobertpt_seed42/
    result.json            # o baseline_*.json de hoje
    preds.json             # o *.preds.json de hoje
    train.log              # recorte do pipeline.log so desta rodada
    command.txt            # linha de comando exata + commit sha + versao do torch/transformers
  2026-06-23_bertimbau_seed42/
  2026-07-16_bertimbau_seed43/
  2026-06-23_biobertpt_seed43/

results/                   # <- somente o CONSOLIDADO "atual", regeravel a partir de experiments/
  summary_by_seed.json     # media +- desvio por modelo/metrica (agregador le experiments/)
  significance_biobertpt_vs_bertimbau.json
  tables/                  # <- EXTENSAO 3: saida do gerador, consumida pelo artigo
    tab_baselines.tex      #    \input{} direto no artigo.tex
    tab_significancia.tex
    tab_seeds.tex
  figures/
    cm_biobertpt.pdf  cm_bertimbau.pdf  f1_por_classe.pdf

scripts/
  aggregate_seeds.py       # experiments/*/result.json -> results/summary_by_seed.json
  make_tables.py           # results/ -> results/tables/*.tex
  make_figures.py          # experiments/*/result.json -> results/figures/*.pdf
```

A regra que mantém isso limpo é uma só: **`experiments/` só cresce, `results/` só é regenerado.** Toda rodada nova vira uma pasta nova em `experiments/` (nada é sobrescrito, o histórico é imutável); `results/` é inteiramente descartável e reconstruível rodando os três scripts. E `artigo-sbc/artigo.tex` passa a fazer `\input{../results/tables/tab_baselines.tex}` e `\includegraphics{../results/figures/...}` em vez de conter números digitados à mão — que é exatamente o que hoje faz o artigo ficar defasado em relação a `results/` (achado 2.6).

**Migração:** as 4 rodadas atuais viram 4 pastas em `experiments/` com as datas de modificação dos arquivos (23/06 para três, 16/07 para o `bertimbau_seed43`). O `command.txt` de cada uma pode ser reconstruído a partir do campo `config` já gravado dentro de cada `result.json` — a informação não se perdeu.

---

## 5. Comandos do projeto — consolidados

Ordem correta, com pré-requisito e produto de cada passo. **Todos os caminhos são relativos à raiz do repositório.**

### Passo 0 — Ambiente *(uma vez)*

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **Requer:** Python 3.10.
- **Produz:** ambiente com `lxml`, `numpy`, `scikit-learn`, `scipy`, `torch`, `transformers`.
- **Falta em `requirements.txt`:** `matplotlib` (curvas nos notebooks) e `huggingface_hub` (upload opcional).

### Passo 1 — Parse do corpus

```bash
python src/parse_semclinbr.py \
    --xml-dir SemClinBr-xml-public-v1 \
    --out data/processed/dataset.jsonl
```

- **Requer:** 1.000 `.xml` em `SemClinBr-xml-public-v1/` (licença restrita, obtidos com os autores do corpus).
- **Produz:** `data/processed/dataset.jsonl` (5,7 MB, 1.000 linhas). Log: 11.458 relações (9.852 `associated_with`, 1.606 `negation_of`) e 18.079 offsets divergentes — erros conhecidos do SemClinBr, contados e tolerados por design.
- **Pode pular se:** você só quer reproduzir os baselines — os splits já estão versionados.

### Passo 2 — Splits

```bash
python src/make_splits.py \
    --input data/processed/dataset.jsonl \
    --out-dir data/splits \
    --seed 42
```

- **Requer:** `data/processed/dataset.jsonl`.
- **Produz:** `data/splits/{train,dev,test}.jsonl` → 800/100/100 docs.
- ⚠️ **Sobrescreve os splits versionados.** Se rodar por engano com outra seed, `git checkout data/splits/` para restaurar. Este é o argumento mais forte para o `MANIFEST.json` com SHA.

### Passo 3a — Baseline clínico (BioBERTpt)

```bash
python src/baseline_biobertpt.py \
    --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
    --ckpt-dir checkpoints/biobertpt_seed42 \
    --out results/baseline_biobertpt_seed42.json
```

- **Requer:** `data/splits/*.jsonl`; GPU (T4 ou melhor); download de `pucpr/biobertpt-all` (~440 MB).
- **Produz:** `results/baseline_biobertpt_seed42.json` **e**, automaticamente, `results/baseline_biobertpt_seed42.preds.json`. Grava `checkpoints/.../best_model/` e `.../last_checkpoint/` a cada época.
- **Custo:** ≈2h25 em T4. **Retomável:** se cair, rode de novo o mesmo comando — pula as épocas concluídas.

### Passo 3b — Baseline geral (BERTimbau)

```bash
python src/baseline_bertimbau.py \
    --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
    --ckpt-dir checkpoints/bertimbau_seed42 \
    --out results/baseline_bertimbau_seed42.json
```

Idêntico ao 3a exceto o modelo (`neuralmind/bert-base-portuguese-cased`) e os caminhos. **É essa identidade que sustenta a validade da comparação** — se você mudar um hiperparâmetro aqui, mude no 3a também.

### Passo 3c/3d — Seeds adicionais *(robustez)*

```bash
python src/baseline_biobertpt.py --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 43 \
    --ckpt-dir checkpoints/biobertpt_seed43 \
    --out results/baseline_biobertpt_seed43.json
# idem para bertimbau com --seed 43
```

⚠️ **`--ckpt-dir` precisa mudar junto com a seed.** Apontar para o diretório da seed 42 rodando com `--seed 43` agora aborta com `RuntimeError` (guarda de seed em `load_last_checkpoint`). Antes da correção não havia retomada errada — o checkpoint era descartado e o treino recomeçava do zero, avisado só por um WARNING no log; o custo era perder horas de GPU sem perceber. Ver a nota em "Checkpoints" (Seção 4.1).

### Passo 4 — Significância

```bash
python src/significance.py \
    --a results/baseline_biobertpt_seed42.preds.json \
    --b results/baseline_bertimbau_seed42.preds.json \
    --target negation_of --n-boot 10000 --seed 42 \
    --out results/significance_biobertpt_vs_bertimbau.json
```

- **Requer:** os dois `.preds.json` dos passos 3a e 3b. **Só CPU**, segundos.
- **Produz:** `results/significance_biobertpt_vs_bertimbau.json`.
- **Guarda-corpo já implementado:** aborta com código 2 se os `labels` ou os `y_true` de A e B divergirem — ou seja, é impossível comparar acidentalmente predições de conjuntos de teste diferentes. Bom design; vale mencionar no README.

### Passo 5 — Agregação multi-seed *(hoje manual)*

```python
import json, glob, statistics as st
v = [json.load(open(f))["test_f1_per_class"]["negation_of"]
     for f in glob.glob("results/baseline_bertimbau_seed*.json")]
print(f"negation_of F1: {st.mean(v):.4f} ± {st.pstdev(v):.4f} (n={len(v)})")
```

Candidato natural a virar `scripts/aggregate_seeds.py` (seção 4.3).

### Passo 6 — Figuras e tabelas do artigo

**Não existe comando.** As figuras `artigo-sbc/figs/*.pdf` e as tabelas de `artigo.tex` foram produzidas fora do repositório e não há registro de como. Maior lacuna de reprodutibilidade do projeto.

### Caminho alternativo — Colab

```
1. notebooks/baseline_biobertpt_colab.ipynb   (GPU T4, ~2h25, com retomada)
2. notebooks/baseline_bertimbau_colab.ipynb   (GPU T4, ~2h25, com retomada)
3. notebooks/significance_colab.ipynb         (CPU, segundos)
```

Pré-requisitos: secret `GITHUB_PAT` no Colab; Google Drive montado para os checkpoints; opcionalmente `HF_TOKEN` para publicar o `best_model/`.

⚠️ **Os três notebooks apontam para `angeloalsf/RECLin-PT-Min`, não para este repositório**, e gravam/procuram `results/baseline_*.json` sem sufixo de seed. Precisam ser corrigidos antes de rodarem contra `RECLin-PT` (ver 2.4).

### 5.1 Diagnóstico do `run.sh`

`run.sh` está **desatualizado em cinco pontos**:

| Linha | Problema |
|---|---|
| Passo 3a/3b | `--batch-size 16` — não é o que rodou (foi 64) |
| Passo 3a/3b | Não passa `--max-gap 20` nem `--max-length 128`, então usa os defaults 75/192 — **experimento diferente do do artigo** |
| Passo 3a/3b | `--out results/baseline_biobertpt.json`, sem seed — colide com a convenção real |
| Passo 3a/3b | Não passa `--ckpt-dir`, logo **não há retomada**: se cair na hora 2 de 2h25, perde tudo |
| Ausente | Não tem o passo 4 (significância), embora o README o documente |

Além disso, `set -e` faz o script abortar no primeiro erro — o que significa que uma falha no BioBERTpt (passo 3a) impede o BERTimbau de rodar, mesmo sendo independente.

### 5.2 `Makefile` vs. `run.sh` — recomendação

**Recomendação: `Makefile`, mantendo `run.sh` como atalho de uma linha que chama `make all`.**

O argumento decisivo é a **idempotência por dependência de arquivo**. Cada passo aqui é caro (2h25) e o `make` só reexecuta o que tem entrada mais nova que a saída:

```make
data/processed/dataset.jsonl: $(wildcard SemClinBr-xml-public-v1/*.xml)
results/baseline_biobertpt_seed42.json: data/splits/train.jsonl
```

Com isso, `make significance` percorre a cadeia inteira e não retreina nada se os resultados já existirem. Com `run.sh` + `set -e`, rodar de novo significa **10 horas de GPU desperdiçadas** ou editar o script à mão para comentar linhas.

Vantagens adicionais: alvos nomeados (`make parse`, `make splits`, `make biobertpt-seed42`, `make significance`, `make figures`, `make artigo`) documentam o pipeline melhor que prosa; e `make -n` mostra o que seria executado sem executar — útil antes de comprometer horas de GPU.

Ressalvas honestas: `make` não vem por padrão no Windows (você está em Windows — precisaria do Git Bash com make, WSL ou Chocolatey), a sintaxe de tabs é chata, e nenhum dos passos pesados roda localmente mesmo (rodam no Colab, onde `make` não ajuda). Se isso pesar, a alternativa razoável é **manter `run.sh` mas corrigi-lo**: acertar os hiperparâmetros, adicionar `--ckpt-dir`, adicionar o passo 4 e proteger cada passo com um `if [ ! -f saida ]; then ... fi`. É 80% do benefício com zero dependência nova.

De qualquer forma, os comandos canônicos precisam estar **em um só lugar**. Hoje estão em três (README, `run.sh`, notebooks) e os três discordam.

---

## 6. Triagem de arquivos: manter, histórico, ignorar, excluir

### 6.1 Manter versionado no git

| Arquivo/pasta | Justificativa |
|---|---|
| `src/**/*.py` (1.202 linhas) | Código-fonte. É o produto intelectual do trabalho. |
| `README.md` | Porta de entrada. |
| `requirements.txt` | Reprodutibilidade do ambiente. |
| `run.sh` (ou `Makefile`) | Definição executável do pipeline. |
| `.gitignore` | Política de versionamento é parte do projeto. |
| `notebooks/*.ipynb` **sem outputs** | São o caminho de execução real (Colab). Hoje já estão limpos — manter assim. |
| `data/splits/*.jsonl` (5,6 MB) | **Caso especial justificado.** São derivados, mas é o que congela o conjunto de teste entre os dois modelos e permite reproduzir no Colab sem o corpus restrito. Vale os 5,6 MB. |
| `artigo-sbc/artigo.tex`, `resumo-expandido.tex`, `artigo.bib` | Fontes do artigo. |
| `artigo-sbc/sbc-template.sty`, `sbc.bst` | Terceiros, mas necessários para compilar e não instaláveis via gerenciador. Manter. |
| `artigo-sbc/figs/*.pdf` | Enquanto não houver `make_figures.py`, **são insubstituíveis** — não há como regerá-las. Depois que o script existir, reavaliar. |

**Zona cinzenta — `artigo.bbl`, `artigo.docx`, `reclin-pt_*.pdf` (390 KB, derivados).** O `.bbl` justifica-se (garante compilação idêntica sem rodar BibTeX). O `.docx` e o `.pdf` são conversões: manter só se forem os arquivos que você efetivamente submete/entrega. Se sim, mover para `artigo-sbc/entregas/` com a data no nome (`2026-06-23_artigo.pdf`) — assim o histórico de versões submetidas fica legível. **[Executado em 09/08/2026: ambos foram confirmados como submetidos e movidos para `artigo-sbc/entregas/`.]**

### 6.2 Guardar como histórico

**Todos os arquivos de `results/` deveriam estar versionados. Hoje nenhum está** — `results/` inteiro está no `.gitignore` e o commit atual não contém nada de lá.

O raciocínio: são **404 KB no total** (irrelevante para o git), custaram **~10 horas de GPU**, e são **a evidência empírica das afirmações do artigo**. Se o seu disco falhar hoje, o TCC perde os dados que sustentam a conclusão. Os notebooks já reconhecem isso implicitamente — usam `git add -f` para furar o `.gitignore` (célula 18). Melhor tirar `results/` do `.gitignore` do que contorná-lo.

| Categoria | Tamanho | Recomendação |
|---|---|---|
| `results/baseline_*.json` (4) | 11 KB | Versionar. Contêm o `config` usado — é o registro do que efetivamente rodou. |
| `results/*.preds.json` (4) | 386 KB | Versionar. Permitem refazer **qualquer** teste de significância sem retreinar. Alto valor por byte. |
| `results/significance_*.json` | 712 B | Versionar. |
| `logs/pipeline.log` | 3,3 KB | Manter gitignored (é uma trilha local acumulada, com histórico misturado). Mas extrair o recorte de cada rodada para `experiments/<rodada>/train.log` e versionar **esse**. |
| Curvas de treino/validação | — | Já embutidas em `dev_history` dentro dos `baseline_*.json`. Não precisam de arquivo próprio. |

**Convenção para não perder rodadas antigas.** Hoje o risco é concreto e não hipotético: o `run.sh` grava em `results/baseline_biobertpt.json` (nome fixo) e os notebooks fazem o mesmo. Duas execuções seguidas com o mesmo `--out` **sobrescrevem silenciosamente** — sem aviso, sem backup, e o `.preds.json` sidecar junto.

Por que ainda não deu problema: você adotou, na mão, o sufixo `_seed42`/`_seed43`, o que fez cada rodada cair em arquivo próprio. Mas isso é disciplina manual contradita pela ferramenta — o `run.sh` e os notebooks continuam apontando para os nomes sem sufixo. Basta rodar `bash run.sh` uma vez para sobrescrever os resultados da seed 42.

Três opções, da mais leve à mais robusta:

1. **Mínima (imediata):** alinhar `run.sh` e os notebooks à convenção `_seed<N>` que você já usa. Resolve a colisão entre seeds, mas não entre reexecuções da mesma seed.
2. **Intermediária (recomendada):** `experiments/<AAAA-MM-DD>_<modelo>_seed<N>/` (seção 4.3). Cada rodada é imutável; `results/` vira consolidado regenerável. Custa reorganizar 8 arquivos.
3. **Máxima:** guarda no `relation_extraction.py` — se `--out` já existir, aborta com mensagem pedindo `--force` ou um caminho novo. Torna a sobrescrita impossível por acidente, mas exige mexer no código (fora do escopo agora).

### 6.3 Ignorar via `.gitignore`, manter localmente

| Item | Estado atual | Justificativa |
|---|---|---|
| `SemClinBr-xml-public-v1/` (8,3 MB, 1.000 XMLs) | ✅ já ignorado | **Dados clínicos com licença restrita.** Nunca devem ser versionados — não é questão de tamanho, é de licença e de privacidade de dados de saúde. Correto como está. |
| `data/processed/dataset.jsonl` (5,7 MB) | ✅ já ignorado | Derivado do corpus restrito; herda a restrição. Regerável em ~40 s. |
| `__pycache__/`, `*.pyc` | ✅ já ignorado | Bytecode. As 9 pastas/arquivos presentes localmente não estão no commit — a regra funciona. |
| `logs/` | ✅ já ignorado | Trilha local. Ver 6.2 para a alternativa por rodada. |
| Checkpoints (`best_model/`, `last_checkpoint/`, `*.pt`, `*.safetensors`) | ❌ **não ignorado** | ~440 MB por modelo. Acima do limite de 100 MB/arquivo do GitHub. Hoje só existem no Drive, mas nada impede que apareçam localmente. **Falta a regra.** |
| `.ipynb_checkpoints/` | ❌ **não ignorado** | Aparece assim que você abrir um notebook em Jupyter local. |
| Ambientes virtuais (`.venv/`, `venv/`, `env/`) | ❌ **não ignorado** | Não existem hoje, mas o README instruirá a criar um. |
| Artefatos LaTeX (`*.aux`, `*.log`, `*.out`, `*.blg`, `*.toc`, `*.synctex.gz`, `*.fdb_latexmk`, `*.fls`) | ❌ **não ignorado** | Não estão presentes (você deve compilar em Overleaf), mas qualquer compilação local os cria em `artigo-sbc/`. |
| `.DS_Store`, `Thumbs.db`, `.vscode/`, `.idea/` | ❌ **não ignorado** | Higiene padrão. |
| Cache do Hugging Face (`~/.cache/huggingface`) | fora do repo | Não é problema — fica no home do usuário. |

### 6.4 Candidato a exclusão definitiva — `gcm-diagnose.log`

**Diagnóstico:** confirmado. 6.681 bytes, 178 linhas, UTF-8 com BOM, terminadores CRLF/LF mistos. É a saída de `git credential-manager diagnose` (GCM 2.7.3), gerada no diretório de trabalho — o próprio log confirma: `GetCurrentDirectory(): D:\angeloalsf\tcc\RECLin-PT`. Não tem relação alguma com o pipeline. Foi commitado por acidente no "first commit".

**Contém informação sensível?** Nada crítico, mas bastante coisa que não deveria ser pública:

| Categoria | O que aparece | Risco |
|---|---|---|
| Identidade da máquina | `COMPUTERNAME=DESKTOP-9C3263H`, `LOGONSERVER=\\DESKTOP-9C3263H`, `USERDOMAIN=DESKTOP-9C3263H` | Baixo/Médio — fingerprint da máquina |
| Usuário do SO | `USERNAME=angeloalsf`, `USERPROFILE=C:\Users\angeloalsf`, `HOMEPATH=\Users\angeloalsf` | Baixo |
| Caminhos locais completos | `C:\Users\angeloalsf\OneDrive`, `D:\angeloalsf\tcc\RECLin-PT`, `C:\Users\angeloalsf\.gcm`, `C:\Users\angeloalsf\AppData\Local\.IdentityService\msal.cache` | Médio — mapeia a estrutura do disco pessoal |
| `PATH` completo | Revela Python 3.14, Node, Chocolatey, VS Code, GitHub CLI, npm global | Médio — **inventário de software**, útil para quem procura versões vulneráveis |
| E-mail | `user.email=angeloalsf@gmail.com` (do `.gitconfig`) | Baixo — já está público no `artigo.tex` |
| Hardware | `PROCESSOR_IDENTIFIER=Intel64 Family 6 Model 69`, `NUMBER_OF_PROCESSORS=4`, `OSVersion: 10.0 (build 22631)` | Baixo |
| Remoto | `remote.origin.url=https://github.com/angeloalsf/RECLin-PT.git` | Nenhum |

**Segredos:** ✅ **nenhum token, senha ou chave em claro.** A seção "Credential storage" apenas grava/lê/apaga uma credencial de teste e reporta `OK` — não imprime valores. A seção MSAL informa o *caminho* do cache (`msal.cache`), não o conteúdo. A seção GitHub API só reporta que o endpoint `/meta` respondeu.

**Veredito:** remover. Não é vazamento grave, mas é **ruído puro** — zero valor para o projeto, e expõe um inventário desnecessário do ambiente pessoal num repositório de TCC que provavelmente será público (ou pelo menos visto pela banca).

**Detalhe adicional:** o arquivo aparece como *modificado* no working tree (156 inserções / 156 deleções). Ele foi **regerado em 07/08/2026 02:56 UTC**, depois do commit — ou seja, você rodou `git credential-manager diagnose` de novo recentemente e ele sobrescreveu o arquivo no mesmo lugar. Enquanto estiver rastreado, cada diagnóstico futuro vai poluir o `git status`.

#### Comandos para remoção — **NÃO EXECUTADOS**

**Etapa 1 — tirar do working tree e do índice** *(resolve daqui pra frente)*

```bash
cd D:/angeloalsf/tcc/RECLin-PT
git rm --cached gcm-diagnose.log     # tira do git, MANTEM o arquivo no disco
# ou, para apagar tambem do disco (recomendado, e lixo):
git rm gcm-diagnose.log

echo "gcm-diagnose.log" >> .gitignore
echo "*.log" >> .gitignore            # cobre diagnosticos futuros; ver ressalva na secao 7

git commit -m "chore: remove log de diagnostico do GCM commitado por acidente"
git push
```

**Etapa 2 — tirar do histórico** *(o arquivo continua recuperável em `f26c40c` sem isto)*

Como o repositório tem **um único commit**, este é o momento mais barato possível para fazer isso — e as duas primeiras opções nem precisam de ferramenta externa.

*Opção A — amend (a mais simples, dado 1 commit só):*

```bash
git rm --cached gcm-diagnose.log
echo "gcm-diagnose.log" >> .gitignore
git add .gitignore
git commit --amend --no-edit          # reescreve o unico commit sem o arquivo
git push --force-with-lease origin main
```

*Opção B — recomeçar o histórico (também limpa qualquer outra coisa indesejada):*

```bash
git checkout --orphan limpo
git rm --cached gcm-diagnose.log
git add -A
git commit -m "first commit"
git branch -D main && git branch -m main
git push --force-with-lease origin main
```

*Opção C — `git filter-repo` (a ferramenta correta se o histórico crescer):*

```bash
pip install git-filter-repo
git filter-repo --invert-paths --path gcm-diagnose.log --force
git remote add origin https://github.com/angeloalsf/RECLin-PT.git
git push --force origin main
```

*Opção D — BFG Repo-Cleaner (alternativa ao filter-repo, exige Java):*

```bash
java -jar bfg.jar --delete-files gcm-diagnose.log .
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force origin main
```

**Avisos antes de reescrever o histórico:**

- Todas as opções **reescrevem SHAs** e exigem `push --force`. Como o repositório é seu, tem um commit e (aparentemente) nenhum colaborador, o risco é mínimo — mas faça uma cópia da pasta antes.
- Se alguém já clonou, precisará reclonar.
- O GitHub mantém objetos órfãos acessíveis por SHA direto por algum tempo mesmo após o force-push. Como não há segredo real aqui, isso não é preocupante. **Se houvesse token,** a única resposta correta seria revogá-lo, não apagá-lo do histórico.
- `--force-with-lease` é preferível a `--force`: aborta se o remoto tiver algo que você não viu.

**Recomendação:** **Opção A**. Um commit só, um `--amend`, resolvido. Não vale instalar `filter-repo` nem Java para isso.

---

## 7. Checagem do `.gitignore` atual

### 7.1 Conteúdo atual (6 regras, 75 bytes)

```gitignore
__pycache__/
*.pyc
data/processed/
SemClinBr-xml-public-v1/
results/
logs/
```

**Avaliação:** as quatro primeiras regras estão corretas e funcionam (nenhum `.pyc` nem XML entrou no commit). As duas últimas são **discutíveis** e uma delas está sendo ativamente contornada.

### 7.2 O que falta

| Falta | Por quê |
|---|---|
| `.ipynb_checkpoints/` | Jupyter cria ao abrir qualquer notebook local. |
| `.venv/` `venv/` `env/` `ENV/` | O README instruirá a criar um venv. |
| `checkpoints/` `*.pt` `*.pth` `*.safetensors` `*.bin` `best_model/` `last_checkpoint/` | ~440 MB por modelo, acima do limite de 100 MB/arquivo do GitHub. Um `git add .` distraído com checkpoints locais produz um push travado ou um repositório inchado permanentemente. **É a omissão mais perigosa da lista.** |
| `gcm-diagnose.log` | Ver seção 6.4. É regerado a cada `git credential-manager diagnose`. |
| Artefatos LaTeX: `*.aux` `*.bbl.bak` `*.blg` `*.out` `*.toc` `*.lof` `*.lot` `*.fls` `*.fdb_latexmk` `*.synctex.gz` | Qualquer compilação local suja `artigo-sbc/`. ⚠️ **Não** ignorar `*.bbl` — `artigo.bbl` é versionado de propósito. |
| `.DS_Store` `Thumbs.db` `desktop.ini` | Ruído de SO. |
| `.vscode/` `.idea/` `*.swp` | Config de editor. |
| `*.egg-info/` `build/` `dist/` `.pytest_cache/` `.mypy_cache/` `.ruff_cache/` | Caso o projeto vire pacote ou ganhe testes/linters. |

### 7.3 Regras a **reconsiderar**

**`results/` — recomendo remover do `.gitignore`.**

Argumentos:
- São **404 KB** — irrelevante para o git.
- Custaram **~10 h de GPU**. Não são regeráveis de graça.
- São a **evidência empírica das afirmações do artigo**. Sem elas versionadas, o TCC não é auditável.
- **Os notebooks já a contornam** com `git add -f` (célula 18 dos dois notebooks de treino e célula 12 do de significância) — sinal claro de que a regra está errada. Uma regra que você precisa furar rotineiramente deveria ser reescrita, não contornada.

Se houver receio de que checkpoints acabem lá, a regra precisa é *seletiva*, não total:

```gitignore
results/**/*.pt
results/**/*.safetensors
results/**/*.bin
```

**`logs/` — manter ignorado, com ressalva.** `pipeline.log` é uma trilha acumulada e local; versioná-la geraria conflitos a cada execução. Mas o log **de cada rodada** tem valor de auditoria — a proposta da seção 4.3 é extrair o recorte para `experiments/<rodada>/train.log` e versionar esse.

**Cuidado com `*.log` genérico.** Se você adicionar `*.log` para cobrir o `gcm-diagnose.log`, ele também esconde artefatos de compilação LaTeX (bom) **e** qualquer log de rodada que você queira versionar (ruim). Se for usar `*.log`, acrescente a exceção:

```gitignore
*.log
!experiments/**/*.log
```

### 7.4 `.gitignore` proposto (completo, para revisão)

```gitignore
# --- Python ---
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# --- Ambientes virtuais ---
.venv/
venv/
env/
ENV/

# --- Dados: corpus restrito e derivados diretos ---
# SemClinBr tem licenca restrita: NUNCA versionar.
SemClinBr-xml-public-v1/
data/processed/
# NOTA: data/splits/ E versionado de proposito (congela o test entre os modelos).

# --- Modelos e checkpoints (centenas de MB) ---
checkpoints/
best_model/
last_checkpoint/
*.pt
*.pth
*.bin
*.safetensors

# --- Logs ---
logs/
gcm-diagnose.log
*.log
!experiments/**/*.log

# --- Notebooks ---
.ipynb_checkpoints/

# --- LaTeX (artefatos de compilacao; artigo.bbl E versionado) ---
*.aux
*.blg
*.out
*.toc
*.lof
*.lot
*.fls
*.fdb_latexmk
*.synctex.gz

# --- SO e editores ---
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/
*.swp

# NOTA: results/ NAO esta ignorado. Sao ~400 KB que custaram ~10 h de GPU e
# sustentam os numeros do artigo. Apenas pesos eventualmente salvos la sao
# excluidos pelas regras de *.pt / *.safetensors acima.
```

⚠️ **Atenção ao aplicar:** `results/` já está ignorado hoje, então remover a regra **não versiona os arquivos automaticamente** — é preciso `git add results/` explicitamente depois.

---

## 8. Rascunho do futuro `PROJECT_MAP.md`

Proposta de conteúdo. **Não criado como arquivo** — fica para depois da sua revisão.

<details>
<summary><strong>Conteúdo proposto (clique para expandir)</strong></summary>

````markdown
# PROJECT_MAP — RECLin-PT

Referência rápida: o que é cada coisa e onde fica.
Para *rodar* o projeto, veja o README. Este arquivo é para *se localizar*.

## Em uma frase

Comparamos dois encoders BERT em português — um clínico (BioBERTpt) e um geral
(BERTimbau) — na mesma tarefa de extração de relações sobre o SemClinBr,
mudando apenas o checkpoint de pré-treino. **Resultado: empate estatístico.**

## Os 4 números que resumem tudo

| | BioBERTpt (clínico) | BERTimbau (geral) |
|---|---|---|
| F1 `negation_of` (seed 42) | 0,724 | **0,734** |
| Macro-F1 (seed 42) | **0,707** | 0,704 |
| F1 `negation_of` (seed 43) | 0,653 | 0,726 |
| Variação entre seeds | **0,071** | 0,008 |

McNemar p=0,18 · bootstrap IC95 [-0,047; +0,026] → **diferença não significativa**.
A oscilação entre sementes de um mesmo modelo é maior que a diferença entre os
modelos. O encoder geral também é mais estável.

## Onde está cada coisa

### Se você quer entender o método
| Pergunta | Arquivo |
|---|---|
| Como o XML vira dados? | `src/parse_semclinbr.py` (docstring explica as decisões) |
| De onde vêm os negativos `no_relation`? | `src/candidates.py` — pares ordenados, `max_gap` |
| Por que split por documento? | `src/make_splits.py` (docstring) e README "Decisões" |
| Como o modelo é treinado? | `src/relation_extraction.py` — **leia o docstring do topo primeiro** |
| Como sei que os dois baselines são comparáveis? | `src/relation_extraction.py`, seção "POR QUE UM NÚCLEO ÚNICO" |
| Como funciona o teste estatístico? | `src/significance.py` (docstring é a melhor explicação do repo) |

### Se você quer rodar
| O quê | Onde |
|---|---|
| Pipeline completo, local | `run.sh` · README "Como rodar" |
| Treino com GPU | `notebooks/*_colab.ipynb` (Colab T4, ~2h25 por modelo) |
| Só o teste estatístico | `notebooks/significance_colab.ipynb` (CPU, segundos) |
| Dependências | `requirements.txt` (Python 3.10) |

### Se você quer os resultados
| O quê | Onde |
|---|---|
| Métricas de cada rodada | `results/baseline_<modelo>_seed<N>.json` |
| Predições brutas do test | `results/baseline_<modelo>_seed<N>.preds.json` (16.074 exemplos) |
| Teste de significância | `results/significance_biobertpt_vs_bertimbau.json` |
| Curvas de treino/validação | dentro de cada `baseline_*.json`, campo `dev_history` |
| Config exata que rodou | dentro de cada `baseline_*.json`, campo `config` |

### Se você quer escrever/compilar o artigo
| O quê | Onde |
|---|---|
| Artigo completo (SBC) | `artigo-sbc/artigo.tex` |
| Resumo expandido | `artigo-sbc/resumo-expandido.tex` |
| Bibliografia | `artigo-sbc/artigo.bib` |
| Figuras | `artigo-sbc/figs/*.pdf` ⚠️ sem script gerador |
| Template SBC | `artigo-sbc/sbc-template.sty`, `sbc.bst` |

## Dados: o que existe e o que é versionado

| Caminho | Tamanho | No git? | O que é |
|---|---|---|---|
| `SemClinBr-xml-public-v1/` | 8,3 MB | ❌ | 1.000 XMLs. **Licença restrita.** Você fornece. |
| `data/processed/dataset.jsonl` | 5,7 MB | ❌ | 1.000 docs, 11.458 relações. Gerado em ~40 s. |
| `data/splits/train.jsonl` | 4,5 MB | ✅ | 800 docs · 1.299 `negation_of` |
| `data/splits/dev.jsonl` | 558 KB | ✅ | 100 docs · 155 `negation_of` |
| `data/splits/test.jsonl` | 573 KB | ✅ | 100 docs · 152 `negation_of` · **congelado** |

Os splits são versionados de propósito: garantem que os dois modelos sejam
avaliados exatamente no mesmo teste, inclusive no Colab, sem precisar do corpus.

## Números de referência

- **Corpus:** 1.000 documentos · 11.458 relações (9.852 `associated_with`, 1.606 `negation_of`)
- **Candidatos gerados** (`max_gap=20`): 128.380 treino · 15.994 dev · 16.074 test
- **Classes:** `negation_of`, `associated_with`, `no_relation` (~99% dos pares)
- **Parâmetros do modelo:** ~177,9 M
- **Hiperparâmetros do artigo:** 3 épocas · batch 64 · lr 2e-5 · max_gap 20 · max_length 128 · ctx_chars 128 · class_weight balanced
- **Seeds executadas:** 42 e 43, para os dois modelos
- **Custo:** ~48 min/época · ~2h25/baseline · ~10 h para reproduzir tudo (T4)

## Pegadinhas conhecidas

1. **Hiperparâmetros divergem entre fontes.** Os defaults do `argparse` (batch 32,
   max_gap 75, max_length 192) **não** são os do artigo (64/20/128). Passe sempre
   explicitamente. Confira o campo `config` dos `results/*.json` em caso de dúvida.
2. **`--ckpt-dir` precisa incluir a seed.** Reusar o diretório de outra seed agora
   levanta `RuntimeError` na retomada. Antes da correção o checkpoint era
   descartado em silêncio (só um WARNING): a seed 43 nunca retomava da 42, mas
   você perdia o progresso já treinado sem notar.
3. **`make_splits.py` sobrescreve `data/splits/`.** Se rodar por engano,
   `git checkout data/splits/`.
4. **`logs/pipeline.log` contém um smoke test antigo** (22/06, n=2000, "A-clinico"
   vs "B-geral", p=3,8e-06) com resultado **oposto** ao real. O resultado válido é
   o de 23/06, n=16.074.
5. **Os notebooks apontam para `RECLin-PT-Min`**, não para este repositório.

## Glossário rápido

- **`negation_of`** — a relação-alvo: uma entidade nega outra. Métrica principal.
- **Entity markers** — `[E1] ... [/E1] [E2] ... [/E2]` inseridos no texto (Soares et al., 2019).
- **`max_gap`** — distância máxima em caracteres entre dois spans para virarem par candidato.
- **McNemar** — testa se os *padrões de erro* de dois modelos diferem.
- **Bootstrap pareado** — reamostra o test 10.000× para obter IC95 da diferença de F1.
- **Split doc-level** — partição por documento, não por relação: evita vazamento de vocabulário do mesmo prontuário.
````

</details>

**Nota de manutenção:** metade do valor deste arquivo está nos números, e números envelhecem. Se a proposta da seção 4.3 for adotada, a tabela "Os 4 números" e a seção "Números de referência" podem ser geradas por `scripts/make_tables.py` a partir de `results/`, mantendo o resto escrito à mão.

---

## 9. Checklist priorizado de próximos passos

Ordenado por **risco evitado ÷ esforço**. Nada abaixo foi executado.

### 🔴 Prioridade 1 — Fazer agora (minutos, risco alto)

- [ ] **1. Remover `gcm-diagnose.log` do repositório e do histórico.** Opção A da seção 6.4 (`git rm` + `--amend`). Com um único commit, é literalmente agora ou nunca — depois fica mais caro. *(~5 min)*

- [ ] **2. Versionar `results/`.** Tirar a regra do `.gitignore` e commitar os 9 arquivos (404 KB). Hoje **~10 h de GPU e toda a evidência do TCC existem só no seu disco**. Um HD que falha custa o TCC. *(~5 min)*

- [ ] **3. `.gitignore` atualizado.** Aplicar a versão da seção 7.4. O item crítico é `checkpoints/` e `*.safetensors` — um `git add .` com checkpoints locais inviabiliza o push. *(~5 min)*

### 🟠 Prioridade 2 — Esta semana (risco de invalidar o trabalho)

- [ ] **4. Corrigir os hiperparâmetros no README e no `run.sh`.** Hoje o README documenta um experimento (`batch 16`, defaults de `max_gap`/`max_length`) que **não é o que gerou os números do artigo**. Se a banca tentar reproduzir, obtém números diferentes. Aplicar o antes/depois **(C)** da seção 3.2. *(~15 min)*

- [ ] **5. Corrigir os notebooks: `RECLin-PT-Min` → `RECLin-PT`.** Três notebooks, uma variável cada (`REPO_NAME`), mais os caminhos de `--out` e `--ckpt-dir` para a convenção `_seed<N>`. Como estão, não rodam contra este repositório. *(~20 min)*

- [ ] **6. Atualizar o artigo com o BERTimbau seed 43.** O resultado (0,726 vs. 0,734, amplitude 0,008 contra 0,071 do BioBERTpt) **fortalece** a tese: o encoder geral não só empata como é mais estável. Requer editar a seção "Robustez à semente" e as limitações declaradas (linhas ~461-513 de `artigo.tex`). *(~1 h)*

### 🟡 Prioridade 3 — Antes de entregar (reprodutibilidade)

- [ ] **7. Escrever `scripts/make_figures.py` e `scripts/make_tables.py`.** Hoje não há como regerar as 3 figuras nem as tabelas do artigo — é a maior lacuna de reprodutibilidade do projeto, e a que uma banca técnica mais provavelmente vai apontar. *(~3 h)*

- [ ] **8. Adicionar a seção de pré-requisitos ao README.** Python 3.10, como obter o SemClinBr e sob qual licença, hardware (T4), tempo (~2h25/baseline, ~10 h total), e o fato de que os splits versionados dispensam o corpus. Bloco **(B)** da seção 3.2. *(~30 min)*

- [ ] **9. Acrescentar `matplotlib` e `huggingface_hub` ao `requirements.txt`.** Usados nos notebooks, não declarados. *(~2 min)*

- [ ] **10. Reescrever a seção "O que foi deixado de fora".** Dois dos três itens já não são verdade como escritos. Bloco **(F)** da seção 3.2. *(~15 min)*

### 🟢 Prioridade 4 — Quando houver fôlego (organização)

- [ ] **11. Migrar para `experiments/<data>_<modelo>_seed<N>/`.** Seção 4.3. Torna o histórico de rodadas imutável e prepara o terreno para multi-seed. Só vale a pena se você pretende rodar mais seeds. *(~1 h)*

- [ ] **12. Criar `PROJECT_MAP.md` definitivo.** Rascunho na seção 8, para revisão. *(~30 min)*

- [ ] **13. Decidir `Makefile` vs. `run.sh` corrigido.** Seção 5.2. O `Makefile` evita retreinos acidentais de 10 h; o `run.sh` corrigido entrega 80% disso sem dependência nova no Windows. *(~1 h)*

- [ ] **14. `scripts/aggregate_seeds.py`.** Substitui o snippet manual do README. *(~30 min)*

- [ ] **15. Manifesto SHA dos splits.** `data/splits/MANIFEST.json` com sha256, contagens e seed. Fecha o último item genuinamente pendente da lista "deixado de fora" e protege contra a sobrescrita acidental do passo 2. *(~30 min)*

- [ ] **16. Adicionar `LICENSE` e `CITATION.cff`.** Se o repositório for público. Deixa claro que o **código** tem uma licença e os **dados** têm outra (restrita) — distinção importante num projeto com dados clínicos. *(~15 min)*

- [ ] **17. Mover conteúdo pesado do README para `docs/`.** Seção 4.2. Cosmético; só depois que 1-10 estiverem feitos. *(~1 h)*

---

## Nota final

O projeto está em condição melhor do que o volume deste relatório sugere. `src/` tem 1.202 linhas com separação de responsabilidades nítida e docstrings que explicam **decisões**, não só mecânica — o docstring de `significance.py` e a seção "POR QUE UM NÚCLEO ÚNICO" de `relation_extraction.py` são melhores do que se vê na maioria dos repositórios de pesquisa. A decisão do núcleo compartilhado é metodologicamente sólida e resolve o problema de paridade por construção. O guarda-corpo em `significance.py`, que aborta se os `y_true` divergirem, mostra cuidado com validade.

Os problemas encontrados são quase todos de **sincronização** — documentação, notebooks e código apontando para estados diferentes do projeto, resultado natural de um trabalho que evoluiu em paralelo ao artigo. As correções de prioridade 1 e 2 somam menos de duas horas e eliminam os riscos reais: perda de dados, história poluída e um README que descreve um experimento diferente do que foi feito.

---

*Relatório gerado por inspeção somente-leitura em 07/08/2026. Nenhum arquivo do projeto foi modificado.*
