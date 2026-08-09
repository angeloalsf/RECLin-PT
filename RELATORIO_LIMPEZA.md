# Relatório de Limpeza — RECLin-PT

**Data:** 09/08/2026 · **Tipo:** auditoria somente-leitura · **Escopo:** o que
excluir e o que consolidar no estado **atual** do projeto.

> **Nenhum arquivo do projeto foi modificado, movido, renomeado ou apagado.**
> Nenhum comando `git` que altera estado foi executado. A única saída desta
> sessão é este arquivo.

Este relatório é **complementar** ao `RELATORIO_ORGANIZACAO.md` (07/08/2026),
que permanece como registro histórico das decisões já tomadas. Onde os dois
divergirem, **este vale** — o projeto mudou bastante desde então.

---

## Sumário executivo

| # | Achado | Gravidade |
|---|---|---|
| 1 | `gcm-diagnose.log` **ainda está no histórico do git** (blob `7238ca6`, 6.525 B, em `f26c40c`). O item 1 do checklist original foi feito pela metade: saiu do índice, não saiu da história. Com 6 commits em cima, `--amend` já não resolve. | 🔴 Alta |
| 2 | Dois derivados obsoletos do artigo estão versionados e **contradizem** o artigo atual: `reclin-pt_extracao_relacoes_semclinbr.pdf` (262 KB, 24/06) e `artigo.docx` (121 KB, 23/06, com título e e-mails diferentes). São 383 KB — 18% de todo o `.git`. | 🔴 Alta |
| 3 | A migração para "fonte única" cobriu `artigo.tex` mas **esqueceu `resumo-expandido.tex`**, que ainda tem uma tabela digitada à mão e o texto da robustez à semente na versão pré-BERTimbau-seed43. | 🟠 Média |
| 4 | `figs_generated/` e `tables_generated/` já sumiram do disco mas **continuam no índice do git** — 5 arquivos aguardando um commit. | 🟠 Média |
| 5 | A seção "Estrutura" do README lista 6 dos 14 itens da raiz. `scripts/`, `artigo-sbc/`, `LICENSE`, `CITATION.cff` e `requirements.txt` não aparecem. O README cresceu 98 linhas no último commit e a árvore não acompanhou. | 🟠 Média |
| 6 | `README.md` (284 linhas) tem ~66 linhas só sobre o artigo (regerar + compilar) que pertencem a `artigo-sbc/`, não à porta de entrada do repositório. O item 17 finalmente tem massa crítica. | 🟡 Baixa |

**Leitura de uma frase:** o projeto está limpo em código e sujo em **derivados
do artigo** — o `src/` e o `scripts/` estão em bom estado, e quase toda a
gordura restante está em `artigo-sbc/` e no histórico do git.

---

## 1. Candidatos a exclusão

### 1.1 Exclusão de alta prioridade

#### (a) `gcm-diagnose.log` — remanescente no histórico do git

**Estado:** o arquivo não está mais no working tree nem no índice (removido no
commit `855e6c8`). Mas o blob continua **alcançável**:

```
$ git rev-list --all --objects | grep gcm-diagnose
7238ca6b1847fae70ab91b711e6ef639226f4b9d gcm-diagnose.log

$ git show f26c40c:gcm-diagnose.log | head -1
Diagnose log at 2026-08-07T02:56:32Z
```

**Por que ainda importa.** Qualquer pessoa com acesso ao repositório recupera o
conteúdo com um `git show`. O que ele expõe está catalogado na seção 6.4 do
relatório anterior: `COMPUTERNAME`, `USERNAME`, `USERPROFILE`, o `PATH`
completo (inventário de software instalado), caminhos de `msal.cache`. Nenhum
segredo em claro, mas é ruído que não deveria estar num repositório público.

**Por que ficou pela metade.** O checklist original recomendava `git rm` **+
`--amend`**, e avisava: *"com um único commit, é literalmente agora ou nunca —
depois fica mais caro"*. Hoje há 7 commits. O `--amend` não alcança mais o
`f26c40c`.

**O que sobrou como opção** (nenhuma executada aqui):

- `git filter-repo --path gcm-diagnose.log --invert-paths` — reescreve todos os
  hashes. Se o repositório já foi clonado/forkado por alguém, exige coordenação.
- Aceitar e seguir — defensável, dado que o conteúdo é de risco baixo. Mas então
  vale **fechar o item 1 explicitamente** no checklist, em vez de deixá-lo
  parecendo pendente.

Decisão sua. O relatório só registra que o item não está fechado.

#### (b) `artigo-sbc/reclin-pt_extracao_relacoes_semclinbr.pdf` — 262 KB, versionado

**O que é:** a compilação de `artigo.tex` em **24/06/2026 02:01 UTC**
(`/CreationDate (D:20260624020159Z)`, pdfTeX 1.40.22).

**Por que deixou de ser necessário:** foi substituído por
`artigo-sbc/artigo.pdf`, compilado em **09/08/2026 14:17** — a versão que
incorpora a seed 43 do BERTimbau. Não é uma cópia velha inofensiva: os dois
**dizem coisas diferentes** sobre o resultado central.

| | PDF antigo (§4.4) | `artigo.pdf` atual (§4.4) |
|---|---|---|
| Sementes | "treinamos o **BioBERTpt** com uma segunda semente (43)" | "treinamos **ambos** com uma segunda semente (43)" |
| BERTimbau seed 43 | ausente | macro-F1 0,704→0,706; F1 neg. 0,734→0,726 |
| Veredito de significância | só empate (semente 42) | empate na 42, **diferença significativa** na 43 |
| Abstract | "the two baselines are statistically tied" | "...**but this verdict does not hold across the two seeds**" |

Manter um PDF versionado que afirma o oposto do artigo vigente é um risco real
numa entrega de TCC — é o arquivo que alguém abre por engano.

**Se for a versão submetida a algum lugar**, a saída não é manter na raiz de
`artigo-sbc/`: é `artigo-sbc/entregas/2026-06-24_artigo.pdf`, com a data no
nome (proposta que já estava na seção 6.1 do relatório anterior e continua
válida). Caso contrário: excluir.

#### (c) `artigo-sbc/artigo.docx` — 121 KB, versionado

**O que é:** um Word de **23/06/2026 19:06**, com 30.474 caracteres de texto e
imagens embutidas (`word/media/rId38.png` etc.).

**Por que deixou de ser necessário:** não é um derivado de `artigo.tex` — é um
**fork** dele, que já divergiu em três pontos verificáveis:

| Campo | `artigo.docx` | `artigo.tex` (atual) |
|---|---|---|
| Título | "RECLin-PT: o pré-treinamento clínico importa para a extração de relações em notas clínicas em português? Um estudo controlado sobre o SemClinBr" | "RECLin-PT: Extração de Relações em Notas Clínicas em Português sobre o Corpus SemClinBr" |
| E-mails | `angelo.silveira@aluno.ifes.edu.br`, `cristiano.colombo@ifes.edu.br` | `angeloalsf@gmail.com`, `cristiano.colombo@gmail.com` |
| Abstract | "both models are statistically tied" (sem a ressalva de semente) | "...but this verdict does not hold across the two seeds" |

Além disso: **nada no repositório gera esse `.docx`.** Não há passo de conversão
no `run.sh`, nos `scripts/`, nos notebooks nem na seção "Compilar o artigo" do
README. Ele é um artefato manual órfão, e as imagens embutidas nele são as
figuras **pré-`make_figures.py`**.

> **Nota lateral, fora do escopo de limpeza:** a divergência de e-mails entre
> `.docx` (institucional) e `.tex`/`resumo-expandido.tex` (gmail) sugere que em
> algum momento houve uma decisão sobre qual usar na submissão. Vale conferir
> qual é a correta antes de excluir o `.docx` — pode ser a única cópia dessa
> informação.

#### (d) `artigo-sbc/figs_generated/` e `artigo-sbc/tables_generated/` — 5 arquivos no índice

**Estado:** já não existem no disco. Continuam rastreados:

```
$ git status --short
 D artigo-sbc/figs_generated/cm_bertimbau.pdf
 D artigo-sbc/figs_generated/cm_biobertpt.pdf
 D artigo-sbc/figs_generated/f1_por_classe.pdf
 D artigo-sbc/tables_generated/tab_resultados.tex
 D artigo-sbc/tables_generated/tab_signif.tex
```

**Por que deixaram de ser necessários:** foram criados no commit `cb75f8c` como
uma **área de conferência** — a saída dos scripts novos vivia em `*_generated/`
enquanto as figuras publicadas continuavam em `figs/`, para permitir a
comparação byte a byte descrita no README. A conferência terminou com zero
divergências e o conteúdo gerado foi promovido para `figs/` e `tables/`. Prova
de que a promoção aconteceu de fato:

| Figura | `figs/` no `HEAD` (manual) | `figs_generated/` no `HEAD` | `figs/` no disco (hoje) |
|---|---|---|---|
| `cm_bertimbau.pdf` | 17.147 B | 15.957 B | **15.957 B** |
| `cm_biobertpt.pdf` | 15.510 B | 15.330 B | **15.330 B** |
| `f1_por_classe.pdf` | 16.227 B | 13.229 B | **13.229 B** |

O disco já tem o conteúdo gerado. As pastas `*_generated/` são andaime da
migração. **Não requer decisão — requer um commit.**

### 1.2 Exclusão de trechos (não de arquivos)

| Local | Trecho | Por que sai |
|---|---|---|
| `.gitignore:32` | `gcm-diagnose.log` | Redundante com `*.log` na linha seguinte. Só faz sentido enquanto o arquivo for uma ameaça recorrente — e ele não é mais gerado no diretório. |
| `.gitignore:34` | `!experiments/**/*.log` | Aponta para `experiments/`, a estrutura proposta na seção 4.3 do relatório anterior que **nunca foi adotada**. Regra sem alvo. |
| `.gitignore:31,33` | `logs/` + `*.log` | Sobreposição parcial: `logs/` cobre o caso real; `*.log` cobre o resto. Manter as duas é defensável, mas com a linha 34 fora, `*.log` sozinho já basta. |
| `scripts/_artifacts.py:110-125` | `load_significance()` com fallback de nome | 16 linhas que existem só porque `significance_biobertpt_vs_bertimbau.json` (semente 42) não tem sufixo, enquanto o da 43 tem. Ver §2.4(a) — renomear o arquivo apaga a função inteira e deixa um one-liner. |
| `src/relation_extraction.py:351,354` | parâmetro `default_out` de `build_arg_parser` | Morto na prática: **os dois** call-sites passam `default_out=None` e resolvem o caminho depois do `parse_args()`. O parâmetro existe para um caso que a convenção `_seed<N>` tornou impossível. |
| `README.md:267-274` | Item riscado `~~Adoção das tabelas e figuras geradas.~~ **Feito.**` | 8 linhas de *changelog* dentro de uma seção chamada "Limitações conhecidas e próximos passos". A informação é boa (o relato da conferência sem divergências), mas o lugar dela é o histórico, não a lista de pendências. |
| `README.md:141` | "`bash run.sh` (passos 1-3)" | O `run.sh` executa os passos 1, 2, 3a, 3b **e 4** (significância). A parentética está errada. |

### 1.3 Não excluir — falsos positivos verificados

Itens que *parecem* candidatos e não são:

- **`results/*.preds.json` (4 × ~96 KB).** Não são duplicata dos
  `baseline_*.json`: os `.json` têm métricas agregadas (`test_f1_per_class`,
  `confusion_matrix`, `dev_history`), os `.preds.json` têm os vetores
  `y_true`/`y_pred` do teste. Sem eles, refazer qualquer teste pareado exige
  retreinar. **Alto valor por byte — manter.**
- **`artigo-sbc/artigo.bbl`.** Derivado do `.bib`, mas versionado de propósito
  (o README explica: garante compilação sem rodar BibTeX). Coerente com o
  `.gitignore`, que ignora `*.blg` mas não `.bbl`.
- **`artigo-sbc/sbc-template.sty` e `sbc.bst`.** Código de terceiros, não
  instalável por gerenciador. Manter.
- **`data/splits/*.jsonl` (5,6 MB).** Derivados, mas são o que congela o teste
  entre os quatro experimentos e o que permite reproduzir no Colab sem o corpus
  restrito. A decisão já foi tomada e está documentada no `.gitignore`.
- **`src/utils/__init__.py` (0 bytes).** Necessário para o pacote. Não é lixo.
- **`src/candidates.py` (45 linhas).** Importado por
  `relation_extraction.py:64`. Vivo.
- **`logs/pipeline.log` e `src/__pycache__/`.** Existem no disco, corretamente
  ignorados, não rastreados. Apagar é higiene local sem efeito no repositório.

---

## 2. Candidatos a consolidação

### 2.1 O `README.md` está grande demais? — Sim, e de forma desequilibrada

284 linhas, distribuídas assim:

| Seção | Linhas | Volume | Pertence ao README? |
|---|---|---|---|
| Intro + pergunta de pesquisa | 1-20 | 20 | ✅ sim |
| Paridade entre os baselines | 21-38 | 18 | ✅ sim |
| **Estrutura** | 39-65 | 27 | ⚠️ sim, mas **está errada** (§2.2) |
| Pré-requisitos | 66-102 | 37 | ✅ sim |
| Como rodar (local) | 103-147 | 45 | ✅ sim |
| **Regerar tabelas e figuras** | 148-191 | **44** | ❌ → `artigo-sbc/` |
| **Compilar o artigo** | 192-213 | **22** | ❌ → `artigo-sbc/` |
| Métricas e como interpretar | 214-243 | 30 | ⚠️ → `docs/` |
| Decisões principais | 244-255 | 12 | ⚠️ duplica o artigo (§2.3) |
| Limitações e próximos passos | 256-284 | 29 | ⚠️ duplica o artigo (§2.3) |

**O desequilíbrio:** 66 linhas (23% do README) explicam como operar
`artigo-sbc/` — mais do que as 45 linhas que explicam como rodar o experimento,
que é o objeto do repositório. E `artigo-sbc/` **não tem README próprio**.

**Proposta mínima (a que eu faria primeiro):** criar
`artigo-sbc/README.md` com as duas seções (regerar + compilar) e deixar no
README raiz um ponteiro de 3 linhas. Ganho: −60 linhas no README, e a
documentação do artigo passa a viver ao lado do artigo — que é onde alguém a
procura. Isso já estava na seção 4.2 do relatório anterior ("vale um
`artigo-sbc/README.md` de 5 linhas"); a diferença é que agora há 66 linhas
prontas para preenchê-lo.

**Proposta maior (item 17 do checklist):** `docs/metricas.md` (a seção
"Métricas e como interpretar") e `docs/decisoes.md` ("Decisões principais" +
"Limitações"). Isso deixaria o README em ~140 linhas. **Minha leitura:** o item
17 vale a pena **agora** para o bloco do artigo (é o maior e o mais claramente
deslocado), e ainda **não** para o resto — "Decisões principais" e
"Limitações" são justamente o que um leitor quer ver sem clicar duas vezes, e
mover 12 linhas para um arquivo próprio troca gordura por fragmentação.

**Reordenação sugerida** (sem mover nada para fora): trazer "Estrutura" para
depois de "Como rodar". Hoje o leitor recebe a árvore de diretórios antes de
saber o que o projeto faz na prática.

### 2.2 A árvore da seção "Estrutura" ficou defasada

Ela lista 6 itens da raiz. Existem 14:

| Na raiz (disco) | Aparece na árvore do README? |
|---|---|
| `SemClinBr-xml-public-v1/`, `src/`, `data/`, `notebooks/`, `results/`, `run.sh` | ✅ |
| `scripts/` | ❌ — **a pasta criada nesta rodada de trabalho** |
| `artigo-sbc/` | ❌ — a saída final do TCC |
| `LICENSE`, `CITATION.cff` | ❌ — criados nesta rodada |
| `requirements.txt` | ❌ |
| `RELATORIO_ORGANIZACAO.md`, `logs/`, `.gitattributes` | ❌ |

Há também um erro de convenção dentro da árvore. O README escreve:

```
results/
  baseline_{biobertpt,bertimbau}.json          # metricas
  baseline_{biobertpt,bertimbau}.preds.json    # predicoes do test
```

Não existe nenhum arquivo com esses nomes. Todos os 8 seguem
`baseline_<modelo>_seed<N>[.preds].json` — a convenção que o próprio README
documenta 170 linhas abaixo, que os entry-points impõem via `OUT_TEMPLATE`, e
que `_artifacts.py` valida. **A árvore contradiz o resto do arquivo.**

Isso conecta diretamente ao item 12 (`PROJECT_MAP.md`): ver §4.

### 2.3 Duplicação entre `README.md`, `RELATORIO_ORGANIZACAO.md` e `artigo.tex`

Três eixos, com naturezas bem diferentes:

**(a) README ↔ `artigo.tex` — duplicação real, mas legítima.**
"Decisões principais" (README:244-255) e "Limitações conhecidas"
(README:256-284) cobrem o mesmo terreno que `artigo.tex` §3.2, §3.3 e
§4.6 (Limitações e ameaças à validade). Mas cobrem **para públicos
diferentes**: o README diz *"split em nível de documento: evita vazamento"* em
3 linhas; o artigo dedica um parágrafo com citação a `sechidis:11` e as
proporções por partição. Consolidar isso empobreceria os dois.

**O que vale mesmo consolidar** é o subconjunto que é *fato numérico*
duplicado — `max_gap=20`, 128.380 candidatos de treino, 16.074 de teste,
16.074/14.959/964/151 — que hoje aparece digitado no README, no `artigo.tex`,
nos `results/*.json` e nos docstrings. Cada um é uma cópia que pode divergir.
A solução limpa não é apagar: é fazer `make_tables.py` emitir também um
`tables/numeros.tex` com `\newcommand{\nTeste}{16.074}` etc., e o artigo usar
as macros. Isso é **extensão**, não limpeza — registro aqui só para não se
perder.

**(b) README ↔ `RELATORIO_ORGANIZACAO.md` — o relatório tem seções mortas.**
O relatório anterior (1.169 linhas) foi escrito como *proposta*. Boa parte já
virou realidade e agora existe em duas cópias:

| Seção do relatório | Linhas | Estado hoje |
|---|---|---|
| §7.4 `.gitignore` proposto | 919-985 | **É literalmente o `.gitignore` atual**, comentários inclusive. Duplicata exata. |
| §3.2 Antes/depois do README | 215-446 | **232 linhas.** Blocos (B), (C), (F) foram aplicados. O "depois" agora está no README. |
| §5 Comandos consolidados | 560-712 | Sobrepõe-se à seção "Como rodar" do README, que foi corrigida. |
| §2 Inconsistências | 123-193 | Quase todas corrigidas (hiperparâmetros, notebooks, artigo/seed 43). |
| §6.4 `gcm-diagnose.log` | 772-863 | Feito pela metade (§1.1a). |

**Recomendação: não editar o relatório.** Ele é registro histórico, e reescrevê-lo
destrói a informação de *o que estava errado antes*. O que resolve a confusão é
uma **caixa de status de ~10 linhas no topo**, dizendo quais seções já foram
aplicadas e apontando para este documento — custa nada e evita que alguém (ou
você, em três meses) aplique de novo uma proposta já implementada.

**(c) `artigo.tex` ↔ `resumo-expandido.tex` — esta é a duplicação que dói.**
Ver §2.4(c). É o único caso onde a duplicação está **numericamente ativa**.

### 2.4 Fragmentação e sobreposição — casos concretos

#### (a) `results/` — 9 arquivos, 1 nome fora do padrão

Os 8 arquivos de baseline são regulares e sem sobreposição:
`baseline_{biobertpt,bertimbau}_seed{42,43}[.preds].json`. Nada a consolidar —
cada um carrega informação que os outros não têm (verificado: os `.json` têm
`config`/`confusion_matrix`/`dev_history`; os `.preds.json` têm
`y_true`/`y_pred`).

O nono quebra o padrão:

```
significance_biobertpt_vs_bertimbau.json          ← semente 42, SEM sufixo
significance_biobertpt_vs_bertimbau_seed43.json   ← semente 43, COM sufixo
```

O custo dessa exceção está em `scripts/_artifacts.py:110-125` — uma função de
16 linhas cuja única razão de existir é adivinhar qual dos dois nomes procurar,
com um `if seed == 42: candidates.reverse()` no meio. E em `run.sh:35`, que
grava no nome legado.

**Consolidação:** renomear para `..._seed42.json`, ajustar `run.sh` e o
notebook 3, e `load_significance()` vira:

```python
def load_significance(results_dir: Path, seed: int) -> dict:
    return load_json(results_dir / f"significance_biobertpt_vs_bertimbau_seed{seed}.json")
```

Três arquivos tocados, 15 linhas a menos, e a convenção `_seed<N>` passa a valer
sem exceção — que era exatamente o que a seção 4.1 do relatório anterior pedia
("`<N>` sempre explícito, **inclusive para a seed 42**"). É a consolidação de
melhor relação ganho/esforço da lista.

#### (b) Os notebooks — 13 de 21 células são idênticas

Diff célula a célula entre `baseline_biobertpt_colab.ipynb` e
`baseline_bertimbau_colab.ipynb`:

```
células idênticas: 13/21
```

As 8 que diferem mudam **só** o nome do modelo, o caminho do checkpoint e o
nome do arquivo de saída:

| Célula | Diferença |
|---|---|
| 0 | título e prosa: "BioBERTpt" ↔ "BERTimbau" |
| 5, 6 | `checkpoints_biobertpt_seed42` ↔ `checkpoints_bertimbau_seed42` |
| 11, 12 | `src/baseline_biobertpt.py --model pucpr/biobertpt-all` ↔ `src/baseline_bertimbau.py --model neuralmind/...` |
| 14, 18 | caminhos de `results/baseline_<modelo>_seed42.json` |
| 20 | `HF_REPO = 'seu-usuario/reclin-pt-<modelo>'` |

Idênticas: clonar repo, `nvidia-smi`, montar Drive, `pip install`, conferir
splits, plotar curvas, `git add`/`commit`.

**Vale unificar?** Tecnicamente sim — um notebook com `MODEL = 'biobertpt'` na
primeira célula elimina 13 células duplicadas. **Mas eu não faria**, por três
razões que valem mais que a duplicação:

1. O Colab é um ambiente de *uma execução por sessão*. Um notebook parametrizado
   convida a rodar o segundo modelo por cima do primeiro, com o Drive já montado
   no `ckpt_dir` errado — exatamente a colisão que a convenção `_seed<N>` foi
   criada para evitar.
2. Cada notebook carrega o **registro de execução** de um baseline específico.
   Fundir os dois apaga essa correspondência 1:1.
3. Os 8 pontos de diferença são todos *literais de configuração*, não lógica.
   Duplicação de configuração é barata; duplicação de lógica é cara. Aqui é a
   primeira.

**O que eu faria em vez disso:** as células 8 (`pip install`) e 18 (`git add` +
`commit`) são copiadas nos **três** notebooks, e a célula 8 instala uma lista de
pacotes **diferente** do `requirements.txt`
(`transformers>=4.40 scikit-learn>=1.4 lxml`, sem `torch`, `numpy`, `scipy`,
`matplotlib`, `huggingface_hub`). Trocar essas linhas por
`!pip install -q -r requirements.txt` remove a terceira cópia da lista de
dependências e elimina uma fonte de divergência real.

**Prefixo numérico** (`01_`, `02_`, `03_`) — a proposta da seção 4.1 do
relatório anterior continua boa e barata. A ordem de execução hoje só existe em
prosa.

#### (c) `resumo-expandido.tex` — a migração para fonte única não chegou aqui

Este é o achado mais consequente da seção 2.

`scripts/make_tables.py` e `make_figures.py` foram adotados como **fonte única**
dos artefatos do artigo, e `artigo.tex` foi convertido para `\input{}`. Mas
`resumo-expandido.tex` (174 linhas, 24/06) ficou de fora:

```latex
% resumo-expandido.tex:114-128 — DIGITADO A MAO
\begin{table}[ht]
\caption{Resultados dos \emph{baselines} no teste (semente 42)...}
\label{tab:res}
\begin{tabular}{lcccc}
BioBERTpt (clínico) & \textbf{0,707} & 0,724 & \textbf{0,487} & \textbf{0,885} \\
BERTimbau (geral)   & 0,704 & \textbf{0,734} & 0,471 & 0,882 \\
```

Compare com `artigo-sbc/tables/tab_resultados.tex`, que abre com
`%% GERADO AUTOMATICAMENTE por scripts/make_tables.py -- nao edite a mao.` e traz
as mesmas células (mais duas colunas, `F1 assoc.` e `F1 no_rel.`) e o
`\label{tab:resultados}`.

Os números **hoje conferem**. O problema é estrutural: são duas cópias, uma
gerada e uma manual, e a próxima mudança em `results/` só atualiza uma delas.
É precisamente o modo de falha que os scripts foram escritos para eliminar.

Há um segundo problema, este já **ativo**. O texto de robustez à semente do
resumo (linha 147) diz:

> "Treinando **o BioBERTpt** com uma segunda semente (43), o F1 de
> `negation_of` cai de 0,724 para 0,653..."

É a redação anterior à seed 43 do BERTimbau — a mesma que foi corrigida em
`artigo.tex` e que motivou a recompilação do PDF. O resumo expandido ficou na
versão antiga.

E um terceiro: o `resumo-expandido.tex` **não é mencionado no README** — nem na
seção "Compilar o artigo" (que só documenta `latexmk -pdf artigo.tex`), nem em
lugar algum. Sua única menção no projeto está no `RELATORIO_ORGANIZACAO.md`.

**Três saídas, em ordem de preferência:**

1. **Excluir**, se o resumo expandido já foi submetido e não será mais usado. É
   um documento datado de 24/06 com números que o artigo já superou.
2. **Consolidar**: trocar a tabela manual por `\input{tables/tab_resultados.tex}`
   (as colunas extras cabem — o resumo usa 4 das 6) e atualizar o parágrafo de
   robustez para bater com o artigo. Custa ~20 minutos e fecha o último furo da
   fonte única.
3. Deixar como está — mas então **documentar no README** que o resumo é um
   congelado de 24/06 e não acompanha `results/`, para ninguém o citar por
   engano.

O que eu **não** recomendo é o estado atual: um `.tex` vivo, compilável, com
números manuais e uma conclusão desatualizada, e sem nenhuma indicação disso.

#### (d) Os entry-points — não, não vale unificar mais

`baseline_biobertpt.py` (49 linhas) e `baseline_bertimbau.py` (52 linhas) são
quase idênticos. Mas quase tudo é docstring. O código executável de cada um:

```python
DEFAULT_MODEL  = "pucpr/biobertpt-all"                      # difere
OUT_TEMPLATE   = "results/baseline_biobertpt_seed{seed}.json"  # difere
log = get_logger("baseline_biobertpt")                      # difere

def main() -> int:                                          # idêntico (6 linhas)
    ap = build_arg_parser(default_model=DEFAULT_MODEL, default_out=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = OUT_TEMPLATE.format(seed=args.seed)
    return run(args, log)
```

**A duplicação real é de 6 linhas.** As outras ~40 de cada arquivo são
documentação que explica *por que* o projeto tem dois entry-points em vez de um
`--model` — e essa explicação é metodologicamente relevante: os dois arquivos
**são** a materialização da paridade.

Unificar num `baseline.py --model X` custaria as 6 linhas duplicadas e traria de
volta o risco que a separação previne: uma flag esquecida troca o experimento
inteiro sem deixar rastro no nome do arquivo. **Recomendação: manter.**

O único ajuste que faria: eliminar o parâmetro `default_out` de
`build_arg_parser` (§1.2), já que os dois call-sites passam `None` e a
convenção `_seed<N>` tornou impossível haver um default estático.

#### (e) `scripts/` — a divisão está boa

`_artifacts.py` (170 linhas) centraliza o que `make_tables.py` (266) e
`make_figures.py` (233) compartilham: `CLASS_ORDER`, `MODEL_LABELS`,
`MODEL_SLUGS`, `BASELINE_ORDER`, carregamento com validação de seed,
`row_normalized`, `ptbr`. A separação está correta e sem sobreposição residual.
Nada a consolidar aqui além do `load_significance` de §2.4(a).

---

## 3. Estrutura de pastas — reavaliação da seção 4 do relatório anterior

A seção 4 do `RELATORIO_ORGANIZACAO.md` era proposta. Passados dois dias de
trabalho, o terreno mudou. Item a item:

| Proposta (§4 do relatório anterior) | Veredito hoje | Por quê |
|---|---|---|
| **`scripts/` separado de `src/`** | ✅ **Implementado** | Existe, com `_artifacts.py`, `make_tables.py`, `make_figures.py`. A justificativa original ("preservar a legibilidade de `src/`") se confirmou: `src/` continua com 7 arquivos e papéis nítidos. |
| **`results/tables/` e `results/figures/`** | ❌ **Obsoleto — a decisão tomada foi melhor** | A proposta era o artigo fazer `\input{../results/tables/...}`. O que se fez foi o inverso: os scripts escrevem em `artigo-sbc/tables/` e `artigo-sbc/figs/`, e o artigo usa caminhos relativos limpos (`\input{tables/...}`). Isso mantém `latexmk` rodando de dentro de `artigo-sbc/` sem `../` atravessando fronteira de projeto. **Não migrar.** |
| **`data/splits/MANIFEST.json`** | ⏳ **Válido, ainda pendente** | É o item 15. Continua sendo o lugar certo — junto dos dados que descreve, não em `results/`. Nada mudou que o invalide. |
| **`experiments/<data>_<modelo>_seed<N>/`** | ⚠️ **Enfraqueceu bastante** | A justificativa era proteger contra sobrescrita silenciosa e organizar rodadas futuras. Duas coisas mudaram: (1) `results/` agora está **versionado**, então o git já é o histórico imutável; (2) a convenção `_seed<N>` está implementada nos entry-points via `OUT_TEMPLATE`, então rodadas de seeds diferentes não colidem mais. O que sobra é a colisão entre reexecuções da *mesma* seed — risco baixo, e o git avisa (`git status` mostra o arquivo modificado). **Só vale se você for rodar 5+ seeds.** Enquanto forem 2, migrar 8 arquivos e reescrever os scripts custa mais do que resolve. |
| **`docs/`** | ✅ **Agora faz sentido — mas com escopo menor** | Ver §2.1. O bloco do artigo (66 linhas) é o candidato claro; o resto do README não. |
| **`artigo-sbc/README.md`** | ✅ **Agora tem conteúdo pronto** | Era "5 linhas" na proposta original. Hoje há 66 linhas no README raiz esperando por ele. |
| **`.gitignore` local em `artigo-sbc/`** | ❌ **Desnecessário** | O `.gitignore` raiz já cobre `*.aux`, `*.blg`, `*.out`, `*.toc`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`. Verificado: não há artefato de compilação solto em `artigo-sbc/` mesmo depois do `latexmk` de 09/08. |
| **Prefixo numérico nos notebooks** | ⏳ **Válido, ainda pendente** | Barato e resolve a ordem de execução implícita. |
| **`checkpoints/` na raiz, gitignored** | ✅ **Implementado** | `.gitignore:22-28` cobre `checkpoints/`, `best_model/`, `last_checkpoint/`, `*.pt`, `*.safetensors`. |

**Sugestão nova, não prevista na §4 original:** `artigo-sbc/entregas/`, com PDFs
datados das versões efetivamente submetidas
(`2026-06-24_artigo.pdf`). É onde o `reclin-pt_extracao_relacoes_semclinbr.pdf`
deveria estar, se for para ficar (§1.1b). Resolve o problema de fundo — que não
é o arquivo, é o fato de um derivado obsoleto estar ao lado do vigente sem nada
distinguindo os dois.

---

## 4. Itens pendentes do checklist original — relação com exclusão/consolidação

| # | Item | Relação com §1-3 | Leitura |
|---|---|---|---|
| **12** | `PROJECT_MAP.md` | 🔗 **Fortemente acoplado a §2.1 e §2.2** | Não é independente: a árvore da seção "Estrutura" do README está errada (§2.2) e o `PROJECT_MAP.md` seria uma **segunda** árvore. Criar um sem corrigir/absorver o outro produz duas fontes divergentes — o problema que a §2.3 descreve. **Decisão prévia necessária:** ou o `PROJECT_MAP.md` substitui a seção "Estrutura" (que vira um link), ou não se cria o `PROJECT_MAP.md` e se corrige a árvore no lugar. Para um projeto de 14 itens na raiz e 7 arquivos em `src/`, **a segunda opção é a certa** — o rascunho da §8 tem 117 linhas para descrever um repositório que cabe numa tela. |
| **13** | `Makefile` vs. `run.sh` | 🔗 **Fracamente acoplado** | O `run.sh` hoje tem um problema de consolidação concreto: **para no passo 4** e não regenera tabelas/figuras, apesar de o README documentar `make_tables.py`/`make_figures.py` como parte do fluxo. Então o pipeline vive em dois lugares (o `.sh` até a significância, o README dali em diante). Isso se resolve **adicionando 2 linhas ao `run.sh`** — não precisa de Makefile. O argumento pró-Makefile (evitar retreinos de 10 h por dependência de timestamp) continua válido, mas é independente da limpeza. **Baixa prioridade.** |
| **14** | `scripts/aggregate_seeds.py` | 🔗 **Acoplado a §1.2 por um trecho** | Substituiria o snippet Python de `README.md:237-242`. É **adição**, não limpeza — mas remove 6 linhas de código embutido em documentação, que é uma forma de código morto (ninguém testa um snippet de README). Também é o encaixe natural para o item 15. **Independente da decisão de excluir/consolidar.** |
| **15** | Manifesto SHA dos splits | ⚪ **Independente** | Puramente aditivo: `data/splits/MANIFEST.json`. Não remove nem funde nada. A única conexão é de local (§3 confirma que `data/splits/` é o lugar certo). É o **último item genuinamente pendente** da lista "deixado de fora" do README — depois dele, aquela seção fica só com limitações de método. |
| **17** | Mover README para `docs/` | 🔗 **É o item 2.1 deste relatório** | Deixou de ser cosmético. O README ganhou 98 linhas no commit `cb75f8c` e 66 delas são sobre operar `artigo-sbc/`, que não tem README. **Recomendação: fazer a metade barata agora** (`artigo-sbc/README.md`) e adiar `docs/metricas.md`/`docs/decisoes.md`. |

**Resumindo:** 12 e 17 são **inseparáveis** da limpeza — fazê-los sem as decisões
de §2.1/§2.2 cria duplicação nova. 13 tem um pedaço de limpeza embutido (o
`run.sh` incompleto) que é barato e vale destacar. 14 e 15 são **independentes**:
podem ser feitos antes, depois ou nunca, sem afetar nada aqui.

---

## 5. Checklist priorizado

Ordenado por **risco evitado ÷ esforço**. **Nada abaixo foi executado.**

### 🔴 Fazer primeiro — o repositório está inconsistente até isso ser feito

- [ ] **L1. Commitar as remoções pendentes de `figs_generated/` e
      `tables_generated/`** (5 arquivos, §1.1d), junto com os untracked que estão
      soltos: `LICENSE`, `CITATION.cff`, `artigo-sbc/tables/`,
      `artigo-sbc/artigo.pdf`. Hoje o `git status` tem 14 entradas e o repositório
      não reflete o estado real do trabalho. *(~10 min)*

- [x] **L2. Decidir sobre `reclin-pt_extracao_relacoes_semclinbr.pdf` e
      `artigo.docx`** (§1.1b, §1.1c) — ✅ **Resolvido em 09/08/2026.** Confirmado
      que ambos foram efetivamente submetidos, portanto não foram excluídos:
      arquivados em `artigo-sbc/entregas/` com a data no nome
      (`2026-06-24_artigo.pdf` e `2026-06-23_artigo.docx`), com um `README.md` na
      pasta avisando que não são a fonte vigente. A contradição com o artigo
      atual deixa de ser risco: nada mais fica ao lado do `artigo.pdf` sem
      distinção. A questão dos e-mails institucionais vs. gmail continua aberta,
      mas o `.docx` preserva a informação.

### 🟠 Esta semana — duplicação numericamente ativa

- [ ] **L3. Resolver `resumo-expandido.tex`** (§2.4c). É o único furo restante da
      fonte única: tabela manual + parágrafo de robustez na versão pré-seed-43 do
      BERTimbau. Excluir, ou converter para `\input{tables/tab_resultados.tex}` e
      atualizar o texto. *(~20 min)*

- [ ] **L4. Corrigir a árvore da seção "Estrutura" do README** (§2.2). Faltam
      `scripts/`, `artigo-sbc/`, `LICENSE`, `CITATION.cff`, `requirements.txt`; e
      os nomes de `results/` estão na convenção antiga, sem `_seed<N>` — o README
      se contradiz. *(~15 min)*

- [ ] **L5. Padronizar `significance_*.json` para `_seed42`** (§2.4a). Renomear 1
      arquivo, ajustar `run.sh` e o notebook 3, e apagar as 16 linhas de
      `load_significance()` em `_artifacts.py`. Melhor ganho/esforço da lista.
      *(~20 min)*

### 🟡 Antes de entregar

- [ ] **L6. Criar `artigo-sbc/README.md`** movendo para lá as seções "Regerar as
      tabelas e figuras" e "Compilar o artigo" (§2.1). README raiz cai de 284 para
      ~220 linhas e a documentação do artigo passa a viver junto do artigo.
      Fecha metade do item 17. *(~30 min)*

- [ ] **L7. Caixa de status no topo do `RELATORIO_ORGANIZACAO.md`** (§2.3b): ~10
      linhas listando quais seções já foram aplicadas (§7.4, §3.2 B/C/F, §2) e
      apontando para este documento. **Não reescrever o corpo** — é registro
      histórico. *(~10 min)*

- [ ] **L8. Faxina de trechos** (§1.2): `.gitignore` linhas 32 e 34; parâmetro
      `default_out` de `build_arg_parser`; item riscado em `README.md:267-274`;
      "(passos 1-3)" em `README.md:141`. Todos independentes entre si. *(~20 min)*

- [ ] **L9. Fechar o item 1 do checklist original** (§1.1a) — decidir
      explicitamente entre `git filter-repo` e aceitar o blob no histórico, e
      registrar a decisão. Hoje o item parece feito e não está. *(~15 min de
      decisão; ~30 min se optar por reescrever)*

### 🟢 Quando houver fôlego

- [ ] **L10. `!pip install -q -r requirements.txt` nos três notebooks** (§2.4b) —
      remove a terceira cópia da lista de dependências. *(~10 min)*

- [ ] **L11. Prefixo numérico nos notebooks** (`01_`/`02_`/`03_`, §3) — a ordem de
      execução hoje só existe em prosa. *(~10 min)*

- [ ] **L12. Duas linhas no `run.sh`** chamando `make_tables.py` e
      `make_figures.py` (§4, item 13) — fecha o pipeline no `.sh` em vez de
      metade nele e metade no README. *(~5 min)*

### ⚪ Independentes desta limpeza

Sem relação com exclusão/consolidação; podem entrar quando fizer sentido:
**item 15** (manifesto SHA dos splits), **item 14** (`aggregate_seeds.py`),
**item 13** na versão Makefile.

### ❌ Recomendo não fazer

- **Item 12 (`PROJECT_MAP.md`) como arquivo novo.** Criaria uma segunda árvore
  divergindo da do README. Fazer L4 no lugar (§4).
- **Migrar para `experiments/`** (§3). A justificativa caiu depois que `results/`
  passou a ser versionado e a convenção `_seed<N>` entrou nos entry-points.
  Reavaliar só se for rodar 5+ sementes.
- **Unificar `baseline_biobertpt.py` e `baseline_bertimbau.py`** (§2.4d). A
  duplicação real é de 6 linhas; a separação é o que materializa a paridade.
- **Unificar os dois notebooks de treino** (§2.4b). 13 células duplicadas, mas
  todas de configuração — e um notebook parametrizado convida à colisão de
  `ckpt_dir` no Drive.
- **Mover "Decisões principais" e "Limitações" para `docs/`.** São 41 linhas que
  o leitor quer ver sem clicar duas vezes. Fragmentação sem ganho.

---

## Nota final

O contraste com o relatório de 07/08 é grande. Lá, os problemas eram de
**sincronização** — README, notebooks e código descrevendo estados diferentes
do mesmo projeto. Isso foi corrigido: os hiperparâmetros batem, a seed 43 está
no artigo, e o caminho de `results/*.json` até o PDF está fechado por
`make_tables.py`/`make_figures.py`.

O que sobrou é de outra natureza: **resíduo de migração**. `figs_generated/`
esperando um commit, um PDF de junho ao lado do de agosto, um `.docx` que virou
fork, um `resumo-expandido.tex` que ficou para trás quando o `artigo.tex`
avançou. São os detritos normais de uma refatoração que funcionou — e a maior
parte se resolve em decisões de dois minutos, não em trabalho.

A exceção é o `gcm-diagnose.log` no histórico: o único item aqui que ficou
**mais caro** desde o último relatório, exatamente como aquele relatório avisou
que ficaria.

`src/` (1.235 linhas) e `scripts/` (669 linhas) não têm nada a excluir e quase
nada a consolidar. A gordura toda está em `artigo-sbc/` e no `.git`.

---

*Auditoria por inspeção somente-leitura em 09/08/2026. Nenhum arquivo do projeto
foi modificado, movido ou removido; nenhum comando de escrita do git foi
executado. A única escrita desta sessão é este arquivo.*
