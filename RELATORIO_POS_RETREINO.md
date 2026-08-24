# Relatório pós-retreino — auditoria do RECLin-PT

Data: 23/08/2026 · Escopo: **somente auditoria**. Nenhum arquivo do projeto foi
movido, renomeado, apagado, editado ou criado além deste relatório. Nenhum
comando `git add`/`commit`/`push` foi executado.

Contexto da auditoria: `max_gap` foi de 20 para 25, os quatro experimentos
(2 modelos × 2 sementes) foram retreinados, e o achado central se inverteu na
semente 43. Os relatórios anteriores (`RELATORIO_ORGANIZACAO.md`,
`RELATORIO_LIMPEZA.md`, `AVALIACAO_DUVIDAS.md`) permanecem intocados — ver,
porém, o item **H14**: dois deles não estão no working tree.

## O que foi verificado

| Verificação | Resultado |
|---|---|
| `python3 scripts/check_tcc_numbers.py` | **OK** — 93 afirmações, 0 caminhos fantasma, 0 números obsoletos, todos os `\input{tabelas/...}` existem |
| Compilação limpa de `tcc/src/main.tex` (TeX Live 2025, stub `breakcites`) | **OK** — 64 páginas, **0** referências/citações indefinidas, 1 `Overfull \hbox` de 5,0 pt |
| `bibliografia.bib` × citações | **OK** — 18 entradas, 18 citadas, 0 órfãs, 0 citações sem entrada |
| `git status --porcelain -uall` | **limpo** (com a ressalva do mount do bridge) |
| Leitura integral de `tcc/src/` | 8 capítulos + 3 apêndices + pré-textuais + 11 tabelas + `main.tex` + `macros.tex` |
| Recontagem independente do McNemar a partir dos `.preds.json` | ver **N1** |
| Varredura do PDF compilado (`pdfplumber`) | placeholders e tamanhos de fonte, ver §3 |

O documento está numericamente sólido. Os problemas encontrados são de
**direção de afirmação**, **resíduo de enquadramento** e **pré-textuais nunca
tocados** — exatamente as categorias que `check_tcc_numbers.py` não alcança.

---

# 1. Higiene do projeto / repositório

## 1.1 O que pode ser excluído ou consolidado

### H1 🔴 `_to_delete/` foi **commitado** — 13 MB de tarballs de transporte

Os cinco arquivos abaixo estão **rastreados no git** (entraram em `adc21a3` e
`6f29d2b`), não apenas soltos no disco:

```
_to_delete/_reclin_out.tar.gz          2,1 MB
_to_delete/_reclin_stage.tar.gz        7,0 MB
_to_delete/reclin_build.tar.gz.bak     4,2 MB
_to_delete/tcc_eda_archive_...json     3,7 KB
_to_delete/git-index.lock.stale        0 B
```

São artefatos de transporte entre a máquina e o container — nenhum é entrada de
nada. `.git/` está hoje em **26 MB**, e apagar os arquivos do working tree **não**
os tira do histórico: seria preciso `git rm` + reescrita (`filter-repo`) para
recuperar o espaço. Como o repositório é pequeno e o custo de uma reescrita é
reescrever o histórico compartilhado, a recomendação é `git rm -r _to_delete/` e
**aceitar** os 13 MB no histórico, acrescentando `_to_delete/` ao `.gitignore`
para que o padrão de trabalho (mover para lá em vez de apagar) não volte a
commitá-los.

### H2 🔴 `results/archive_max_gap25/` é duplicata byte a byte — e também foi commitada

Comparei os 8 arquivos um a um: **todos idênticos** aos de `results/` (raiz).

```
IDENTICO  baseline_bertimbau_seed42.json         IDENTICO  ...seed42.preds.json
IDENTICO  baseline_bertimbau_seed43.json         IDENTICO  ...seed43.preds.json
IDENTICO  baseline_biobertpt_seed42.json         IDENTICO  ...seed42.preds.json
IDENTICO  baseline_biobertpt_seed43.json         IDENTICO  ...seed43.preds.json
```

Nota: em 21/08 essa pasta era **não rastreada**; agora está commitada (`adc21a3`).
Um `git add -A` a varreu para dentro. `results/archive_max_gap20/` é diferente —
é a comparação histórica intencional e **deve ficar**.

### H3 🔴 `artigo-sbc/` inteiro ficou na rodada de `max_gap=20`, sem nenhuma marca

O artigo SBC é hoje o artefato mais desatualizado do repositório, e nada avisa
isso ao leitor:

- `artigo.tex:278-279` — 128.380 / 15.994 / **16.074** candidatos (números de 20).
- `artigo.tex:371` — teste "composto por 16.074 pares candidatos (14.959 …)".
- `artigo.tex:384-387` — macro-F1 0,707 × 0,704; negação 0,724 × 0,734.
- `artigo.tex:452-462` — semente 43 com os valores antigos (BioBERTpt **cai** de
  0,724 para 0,653).
- `artigo.tex:562` — **"Como o BERTimbau geral não fica atrás em nenhuma das
  sementes…"** — é literalmente a frase do enquadramento antigo que o TCC
  abandonou. Não está no TCC; está aqui.
- `artigo.tex:270` — **o único `TODO(max_gap 20->25)` que sobrou no repositório**
  (a nota de memória que dizia não haver mais nenhum estava errada; ela cobria
  só `tcc/`).
- `tables/tab_signif.tex:6` — `$n=16.074$`.

Agrava: `artigo-sbc/README.md` ainda diz *"Quem quiser ler o artigo atual deve
usar `artigo-sbc/artigo.pdf`"*. Ironicamente, `artigo-sbc/entregas/README.md`
documenta com todo cuidado que `entregas/` está defasado **em relação ao
`artigo.tex`** — ninguém acrescentou a nota equivalente um nível acima.

### H4 🔴 Armadilha no `Makefile`: `make all` produz um artigo internamente contraditório

```make
all: splits baselines significance aggregate tables figures
artigo: $(TABLES) $(FIGURES)
	cd artigo-sbc && latexmk -pdf artigo.tex
```

`make_tables.py` e `make_figures.py` leem `results/` **sem `--results-dir`**, ou
seja, a rodada de 25. Rodar `make all` ou `make artigo` hoje regenera
`artigo-sbc/tables/*.tex` e `figs/*.pdf` com os números novos **enquanto a prosa
do `artigo.tex` continua nos antigos**. O PDF resultante teria tabela dizendo
`n=19.210` ao lado de texto dizendo 16.074, e uma conclusão ("não fica atrás em
nenhuma das sementes") que a própria tabela regenerada contradiz.

É o item de maior risco desta seção porque **falha em silêncio**: compila sem
erro. Três saídas possíveis: congelar o artigo (tirar `tables`/`figures` do
`all` e apontar o alvo `artigo` para `--results-dir results/archive_max_gap20/`),
reancorar o artigo como se fez com o TCC, ou mover `artigo-sbc/` para um
`arquivo/` declaradamente histórico.

### H5 🟠 Notebooks 01 e 02 ainda mandam refazer o retreino que já foi feito

`notebooks/01_...ipynb:168` e `02_...ipynb:168` (idênticos):

> ⚠️ **`max_gap` = 25 (era 20).** … **Os quatro experimentos (2 modelos × 2
> sementes) precisam ser refeitos do zero com 25** — apague/renomeie o
> `--ckpt-dir` antigo antes de rodar…

Os quatro **já foram** refeitos em 21/08. O aviso agora induz a repetir ~4 h de
GPU. O `--max-gap 25` da célula de comando (linha 183) está correto; é só a
prosa do aviso.

### H6 🟠 `tcc/src/main.loq` está rastreado

`tcc/.gitignore` cobre `*.lol` mas **não** `*.loq` (a extensão da Lista de
Quadros que `macros.tex` define via `\def\ext@quadro{loq}`). É o único auxiliar
de LaTeX que escapou: `.aux`, `.bbl`, `.brf`, `.idx`, `.ilg`, `.ind`, `.log`,
`.toc`, `.fls`, `.fdb_latexmk` estão todos corretamente ignorados. Acrescentar
`*.loq` e `git rm --cached tcc/src/main.loq`.

### H7 🟠 `tcc/src/TCC_Angelo_Antonio.pdf` — PDF antigo redundante, rastreado

1,37 MB, de maio/2026, ao lado do `main.pdf` vigente (1,34 MB). Dois PDFs
compilados versionados na mesma pasta, sem nada dizendo qual é qual.

### H8 🟡 `tcc/src/bibliografia.bib.examples` — sobra do template

2 KB de exemplos do template IFES. Inofensivo, mas é ruído numa pasta que a
banca pode abrir.

### H9 🟡 CRLF em `results/archive_max_gap20/*.json`

Todo `git diff` no repositório imprime quatro avisos:

```
warning: CRLF will be replaced by LF in results/archive_max_gap20/baseline_bertimbau_seed42.json.
```

`.gitattributes` cobre `*.json text eol=lf`, mas esses quatro arquivos ficaram
com CRLF no working tree. Um `git add --renormalize results/archive_max_gap20/`
resolve (não muda conteúdo, só a linha).

### H10 🟡 `.gitattributes` não cobre extensões que já existem no repo

Cobre `.json .tex .bib .cls .sty .bst .md .py`. Ficaram de fora, e já há
arquivos de cada: **`.sh`** (`run.sh`), **`.yml`** (`docker-compose.yml`),
**`.cff`** (`CITATION.cff`), **`.txt`** (`requirements.txt`), **`.jsonl`**
(`data/splits/*`), **`.ipynb`** (`notebooks/*`), **`.bbl`**
(`artigo-sbc/artigo.bbl`, que é versionado de propósito).

### H11 🟡 `tcc/Dockerfile`: o comentário promete o que o código não faz

```dockerfile
# Por que pinar a imagem: garante que a compilação do TCC em 2027/2028
# produza o MESMO PDF que produz hoje. Se quiser atualizar, basta trocar
# o digest abaixo.
FROM texlive/texlive:latest-full
```

`latest-full` é tag móvel, não digest — a imagem muda sob os pés. Não é teórico:
a divergência de versão do TeX Live **já quebrou** este projeto uma vez (os
auxiliares gerados em outra versão faziam o `latexmk` falhar com `Undefined
control sequence <argument> r@cap:`). Pinar com `@sha256:...` ou assumir o
comentário como aspiracional.

## 1.2 O que está consistente (verificado, nada a fazer)

`max_gap=25` está coerente em **todos** os pontos de configuração:
`Makefile:54` (`HP := --epochs 3 --batch-size 64 --max-gap 25 --max-length 128`),
`run.sh:22,28`, notebooks 01/02 (célula de comando), `src/candidates.py:13-14`
(documenta a mudança e a razão), `README.md:205-210,300,359`,
`scripts/check_tcc_numbers.py` (com a lista `STALE_NUMBERS` de 9 cifras da
rodada antiga, que passa limpa).

`git status --porcelain -uall` vem **vazio** — nada solto além do que está acima.

`README.md` **não cita resultados numéricos**, então a inversão não o afetou.
E o item 1 do `AVALIACAO_DUVIDAS.md` (README descrevendo marcadores como
"tipados") **já foi corrigido**: não há mais nenhuma ocorrência de
"tipado"/"typed" no README.

### H12 ℹ️ Arquivo que esta auditoria deixou para trás

Para trazer o repositório ao container precisei escrever **um** tarball dentro
de uma pasta montada:

```
D:\angeloalsf\tcc\RECLin-PT\_to_delete\_audit_stage.tar.gz    (6,7 MB)
```

Não consegui apagá-lo daqui (`device_bash` não tem permissão de exclusão e a
sessão não expôs a ferramenta de pedido de permissão). Ele é descartável e some
junto com o `_to_delete/` do item **H1**. É o único arquivo que esta auditoria
criou fora deste relatório.

### H13 🔴 `RELATORIO_ORGANIZACAO.md` e `RELATORIO_LIMPEZA.md` **não estão no working tree**

Você pediu para não sobrescrevê-los "porque são registro histórico" — mas eles
não existem mais no disco. Foram apagados no commit:

```
5bd2b44  delete: remoção de arquivos .md
```

Só `tcc/AVALIACAO_DUVIDAS.md` sobreviveu. Os outros dois existem apenas no
histórico do git (`git show 5bd2b44^:RELATORIO_LIMPEZA.md` os recupera). Se a
intenção é mantê-los como registro, precisam ser restaurados; se a exclusão foi
deliberada, vale registrar isso em algum lugar para não se perder a decisão.

---

# 2. Coerência narrativa do `tcc/`

Esta é a seção principal. Os itens estão ordenados por gravidade.

## N1 🔴🔴 §6.6 atribui o McNemar da semente 43 ao *encoder* clínico — e ele aponta para o geral

**O texto atual** (`experimentos_e_resultados.tex:193-199`):

> Na semente 43, a ordenação entre os dois *baselines* no F1 de negação **se
> inverte**: o BioBERTpt clínico passa à frente, com margem de 0,053, e o mesmo
> protocolo acusa a diferença como **significativa** — **agora em favor do
> *encoder* clínico** (McNemar *p* ≈ 2,7×10⁻⁸; *bootstrap* pareado IC95%
> [+0,016; +0,090], *p*=0,0054)…

**O que os dados dizem.** Recalculei a tabela de discordâncias diretamente dos
`.preds.json`, sem passar pelo JSON de significância:

| | só BioBERTpt acerta (*b*) | só BERTimbau acerta (*c*) | McNemar favorece | acurácia BioBERTpt | acurácia BERTimbau |
|---|---|---|---|---|---|
| semente 42 | 540 | **695** | **BERTimbau** | 0,8796 | **0,8877** |
| semente 43 | 442 | **624** | **BERTimbau** | 0,8984 | **0,9079** |

O McNemar aponta para o **BERTimbau nas duas sementes**. O que inverteu foi
**somente o *bootstrap* pareado na classe-alvo**. A frase acima empacota os dois
testes sob "em favor do *encoder* clínico" e atribui ao McNemar uma direção que
ele nunca teve.

Os números conferem com a `Tabela 11` do próprio TCC, que reporta
`b/c = 442 / 624` — a tabela está certa; é o parágrafo que a lê errado.

**Por que o `check` não pegou.** `check_tcc_numbers.py:219` confere
`mcnemar.p_value` da semente 43 contra `2.7e-8` — e o valor está certo. O que
falha é a **direção**, que não é um literal numérico. Para as sementes 42 há
entradas de `b` e `c` (linhas 213-214); para a 43 não há, porque o texto nunca
as cita.

**A boa notícia:** a leitura correta é *mais forte* do que a que está escrita. O
McNemar é **consistente nas duas sementes**; a instabilidade é **específica da
métrica-alvo**. Isso confirma e reforça, em vez de complicar, a tese de §6.5 de
que *"os dois testes não respondem à mesma pergunta"*.

**Trechos afetados** (todos precisam do mesmo ajuste de escopo):

| Local | Trecho | Problema |
|---|---|---|
| `experimentos_e_resultados.tex:195-199` | "o mesmo protocolo … agora em favor do *encoder* clínico (McNemar …)" | atribui o McNemar ao clínico |
| `experimentos_e_resultados.tex:209` | "O que atravessa as duas sementes **não é**, portanto, **uma direção**." | falso: o McNemar e a acurácia atravessam com direção estável |
| `discussao.tex:15-16` | "na semente 43, em que a comparação pareada acusa diferença significativa — e em favor do *encoder* clínico" | "a comparação pareada" é vago demais; só o *bootstrap* |
| `conclusao.tex:33-35` | "na semente 43, a mesma comparação pareada inverte o sinal" | idem; escapa por citar só o IC |
| `resumo.tex:24-26` e `:60-63` | "a diferença *inverte de sinal* e passa a favorecer o *encoder* clínico de forma significativa (IC95% …)" | tecnicamente escopado ao IC, mas o leitor generaliza |

## N2 🔴 §8.2 se contradiz internamente: "sem custo esperado na métrica-alvo"

`conclusao.tex:55-56`:

> …o que libera a construção do RECLin-PT sobre um modelo sensivelmente menor
> **sem custo esperado na métrica-alvo**.

A métrica-alvo é o F1 de `negation_of`. Média das duas sementes:

- BioBERTpt: (0,677 + 0,754) / 2 = **0,715**
- BERTimbau: (0,694 + 0,702) / 2 = **0,698**

Há, sim, custo esperado: **0,017** na métrica-alvo. E §6.6 diz isso com todas as
letras (`experimentos_e_resultados.tex:212-213`): *"Tomada a média das duas
sementes, o BioBERTpt fica marginalmente à frente na métrica-alvo (0,715 contra
0,698)"*.

Pior: **onze linhas abaixo**, a mesma §8.2 acerta (`conclusao.tex:65-67`):

> …ressalvado que a semente 43 mostra o clínico à frente na negação, de modo que
> a escolha se apoia **em custo e estabilidade, e não em superioridade média
> demonstrada**.

As duas frases estão na mesma seção e dizem coisas opostas. A de cima é resíduo
do enquadramento antigo; a de baixo é a reancorada. **Respondendo diretamente à
sua pergunta sobre §8.2:** o argumento de "custo e estabilidade, não
superioridade média" está consistente em `discussao.tex:58-61`, em
`conclusao.tex:65-67` e no resumo — e esta é a **única** frase do documento que o
quebra. Onde a comparação é em **macro-F1** o texto está certo em toda parte
(0,695 × 0,694, "acompanha"); o deslize é usar a mesma linguagem para a
métrica-alvo, onde a média favorece o clínico.

## N3 🔴 Apêndice B carrega a frase antiga inteira — e nenhuma rodada passou por lá

`apendices/apendice_b_lexico_negacao.tex:5-8`:

> …é uma das explicações discutidas no Capítulo~\ref{cap:discussao} para o fato
> de **um *encoder* de domínio geral capturar a negação tão bem quanto um
> clínico**.

Afirmado como **fato**, sem ressalva de semente. Na semente 43 o clínico está
0,052 à frente exatamente nessa classe, com o *bootstrap* rejeitando a nula.
O apêndice não foi tocado pela Fase 3 — é o exemplo mais puro do que você
suspeitava: uma afirmação qualitativa que sobreviveu por não ser um número.

## N4 🟠 §6.4 e §7.1: "aprendem essencialmente a mesma função / erram nos mesmos lugares"

`experimentos_e_resultados.tex:110-113`:

> A semelhança qualitativa entre as matrizes reforça a leitura de que os dois
> *encoders* **aprendem essencialmente a mesma função** para esta tarefa: não
> apenas chegam a números próximos, mas **erram nos mesmos lugares**.

Repetido em `discussao.tex:45-48`. Duas tensões:

1. **Contradiz o próprio §6.5.** O McNemar rejeita a nula justamente sobre o
   *padrão global de erro*: 1.235 discordâncias em 19.210 instâncias, "assimetria
   grande demais para o acaso". Se os modelos errassem nos mesmos lugares, o
   McNemar não teria o que rejeitar.
2. **É uma generalização a partir de uma semente só.** As duas matrizes de
   confusão são da semente 42 (`fig:cm_biobertpt`, `fig:cm_bertimbau`, ambas
   geradas de `*_seed42.json`). Na 43 os modelos diferem em 0,052 na negação.

O que a evidência sustenta é mais estreito e continua útil: os dois **concentram
o erro na mesma fronteira** (`associated_with` × `no_relation`) — que é
exatamente o que §7.1 precisa para a explicação estrutural. "Aprendem a mesma
função" é forte demais.

## N5 🟠 §7.1: as duas "explicações plausíveis" ainda explicam o achado **antigo**

`discussao.tex:27-48` oferece duas explicações:

1. **Linguística** — a negação em português é lexicalizada, então o *encoder*
   geral a captura "tão bem quanto um clínico".
2. **Estrutural** — a tarefa depende da estrutura local do par, fornecida
   igualmente aos dois pela representação compartilhada.

Ambas explicam **um empate**. Nenhuma explica:

- por que o sinal **inverte** entre sementes; nem
- por que o *encoder* clínico é **dez vezes mais disperso** (amplitude 0,078 ×
  0,008) — que é o que o próprio capítulo, três parágrafos antes e depois,
  declara ser o achado que efetivamente atravessa as duas sementes.

Esta é a **lacuna estrutural mais relevante do documento**: a seção de análise
continua analisando o resultado que foi substituído. Não é uma frase a corrigir,
é um parágrafo a acrescentar — o que na literatura explicaria maior variância de
um *checkpoint* adaptado sob ajuste fino em conjunto pequeno (deslocamento de
domínio entre pré-treino e a tarefa, vocabulário especializado com *embeddings*
mal povoados, sensibilidade à inicialização do cabeçote). Sem isso, o leitor
recebe o achado central sem nenhuma hipótese sobre sua causa.

## N6 🟠 Resumo/abstract, lido por quem só lê essa página

`resumo.tex:21-23`:

> No conjunto de teste congelado, os dois **empatam estatisticamente** no F1 de
> `negation_of` na semente de referência (0,677 contra 0,694; *bootstrap*
> pareado IC95% [−0,053; +0,018], *p*=0,345)…

Lendo isolado, o leitor sai com: *"empate na 42, inversão na 43"*. Duas coisas
ele não recebe:

1. Que **já na semente 42** um dos dois testes era significativo. §6.5 dedica
   quatro parágrafos a dizer que os testes divergem nessa mesma semente e que o
   McNemar rejeita a nula; o resumo apresenta a 42 como empate limpo. O "empatam
   estatisticamente" é verdadeiro para o *bootstrap* e falso para o McNemar.
2. Que — corrigido o **N1** — o fato mais estável de todo o trabalho é que **o
   McNemar aponta para o *encoder* geral nas duas sementes**. Isso não aparece no
   resumo em nenhuma forma.

Uma oração resolveria as duas: qualificar "empatam" como *na classe-alvo* e
registrar que no padrão global de acertos o geral leva vantagem consistente.

O restante do resumo está correto: "cerca de 60% dos parâmetros" (108,9 × 177,9 M
= 61,2%), "dez vezes mais estável" (0,078 / 0,008 = 9,75), "acompanha o clínico
em macro-F1 médio" (0,695 × 0,694). A versão EN espelha fielmente a PT.

## N7 🟡 §7.1: "Esse achado **contrasta** com a expectativa difundida"

`discussao.tex:22-25`. Com resultado misto, "contrasta" é forte demais: na
semente 43 o achado **concorda** com a expectativa (o clínico ganha, com
significância). "Não confirma" ou "não sustenta" seria exato. É uma palavra.

## N8 🟡 §7.1 abertura: "nem mesmo na classe `negation_of`"

`discussao.tex:9-13`:

> …o BioBERTpt não produz ganho **estável** sobre o BERTimbau, **nem mesmo na
> classe `negation_of`**, em que a especialização de domínio seria mais
> plausível.

A palavra "estável" segura a frase tecnicamente, mas a construção "nem mesmo"
ecoa o enquadramento antigo (*não ganha em lugar nenhum*) — e é justamente em
`negation_of`, na semente 43, que ele ganha de forma significativa. Inverter a
ênfase ("ganha nessa classe em uma das duas sementes, e é essa alternância que
constitui a ausência de ganho *confiável*") diz a mesma coisa sem induzir.

## N9 🟡 O enquadramento do Capítulo 1 sobrevive — mas subdeclara o achado

Você perguntou especificamente se o enquadramento de "metodologia como
contribuição" (`introducao.tex:41-47`) ainda faz sentido. **Faz — e faz melhor
do que antes.** Quanto mais instável a resposta empírica, mais a infraestrutura
reprodutível é o entregável defensável. Nada a corrigir ali.

O que falta é o **outro lado**. O Cap. 1 anuncia a cadeia experimental como
contribuição, mas não anuncia o achado empírico mais forte que o trabalho
produziu: **o efeito da semente de treino supera o do *checkpoint* de
pré-treino** nesta tarefa. Isso é um resultado com valor próprio, não um
subproduto — e um leitor que chegue a §6.6 vindo do Cap. 1 não foi preparado
para ele.

Mesmo problema na lista de contribuições de §8.1 (`conclusao.tex:16-27`): os
seis itens (i)–(vi) são todos de infraestrutura e comparação; **a análise de
robustez à semente não aparece como contribuição**, embora o Cap. 4 a declare
"parte do protocolo, e não um complemento opcional" (`metodologia.tex:392-397`).
Sugiro um item (vii).

## N10 🟡 §1.2 apoia-se numa referência que aponta para metade do achado

`introducao.tex:57-60` cita `lin_2020_negacao` (BERT geral > variantes
adaptadas, em inglês) para sustentar que o ganho clínico "não é automático". Com
o resultado hoje misto, essa é uma escolha seletiva: metade dos seus dados
concorda com `lin_2020`, metade com `schneider_2020_biobertpt`. Não é erro —
mas sinalizar que a literatura é dividida deixa a motivação mais honesta e
antecipa a pergunta da banca.

## N11 🟡 Objetivos: "identificar a melhor base" pressupõe veredito de desempenho

`introducao.tex:89-91` e `:116-117` — "de modo a identificar **a melhor base**
sobre a qual construir o RECLin-PT". Com o resultado atual a base é escolhida
por **custo e previsibilidade**, não por ser "a melhor". Trocar por "a base mais
adequada" alinha o objetivo ao que §8.2 efetivamente entrega.

## N12 🟡 O título nomeia o modelo que §8.2 declara ser trabalho futuro

`macros.tex:3`: *"RECLin-PT: extração de relações em notas clínicas em português
sobre o corpus SemClinBr"*. §8.2 argumenta com cuidado que entregar *baselines* e
deixar a especialização para depois é **sequenciamento, não lacuna** — mas o
título anuncia o modelo. É a primeira coisa que a banca lê, e é exatamente a
pergunta que `revisao_tcc.md` já levantou ("o TCC é para entregar o fine
tuning"). Não é um defeito; é uma decisão que vale tomar conscientemente, e não
por inércia.

## O que **não** encontrei (verificado, está limpo)

- **Nenhuma ocorrência** de "em nenhuma das sementes", "não fica à frente" ou eco
  equivalente dentro de `tcc/` — essa frase migrou para `artigo-sbc/artigo.tex:562`
  e ficou lá (**H3**).
- §6.3, §6.5, §7.2, §7.3 (limitações), §7.4 (ameaças), Cap. 2, 3, 4, 5, apêndices
  A e C: **coerentes** com o resultado misto, com as ressalvas de semente no
  lugar certo. §6.5 em particular está muito bem construída — é ela que dá o
  vocabulário para corrigir **N1**.
- Todos os números de prosa conferem (93 afirmações). As médias, amplitudes,
  razão de parâmetros e o "dez vezes mais estável" foram recalculados à mão e
  batem.

---

# 3. Estado geral do documento

## 3.1 Pré-textuais — nunca tocados por nenhuma rodada, e é a lacuna mais séria para a entrega

Todos os itens abaixo **aparecem no PDF compilado hoje**:

| Onde | O que aparece | Página do PDF |
|---|---|---|
| Folha de aprovação | **`Profa. Dra. Fulana de Tal`** e **`Prof. Dr. Cicrano de Tal`** como examinadores | 4 |
| Folha de rosto + aprovação | **`Profa. Dra. Beltrana de Tal`** como coorientadora | 2 e 4 |
| Ficha catalográfica | `Elaborada por XXXXXXXXXXXXXXXXXXXXX – CRB-X/ES - XXX`; cutter `X999y`; CDD `000.00` | 3 |
| Capa, folha de rosto, ficha | `Cachoeiro de **itapemirim**` — minúscula | 1, 2, 3 |
| Folha de aprovação | `\approvaldate{01}{Agosto}{2026}` — data **já passada** | 4 |

Fonte: `macros.tex:14-18, 20, 34, 6`.

**Agradecimentos e epígrafe existem como arquivo, mas não são impressos.**
`macros.tex` registra os dois (`\editaragradecimentos`, `\editarepigrafe`), mas
`main.tex:36` chama apenas `\imprimirdedicatoria` — nunca
`\imprimiragradecimentos` nem `\imprimirepigrafe`. Confirmei no PDF: a
dedicatória (pág. 5) é seguida direto pelo resumo (pág. 6). Como os dois
arquivos são placeholders com `TODO: redigir antes da defesa`, nada está
quebrado hoje — mas a decisão precisa ser tomada: redigir e imprimir, ou remover
as macros para que não fiquem como pendência invisível.

**`siglas.tex` (14 entradas) está incompleta.** Faltam duas siglas que o corpo
do texto **introduz com expansão**, o que é o critério ABNT para entrar na lista:

- **EDA** — `metodologia.tex:106` ("análise exploratória (*Exploratory Data
  Analysis*, EDA)")
- **UMLS** — `proposta_prototipo.tex:169` ("tipos semânticos da UMLS")

Também ausentes, com uso menor: RECLin-PT, XLM-R, GPU, CPU, SHA-256, AdamW, brWaC.

**`simbolos.tex` está inteiramente comentado** — verifiquei no PDF e o abntex2
não imprime lista vazia, então não há página órfã. Nada a fazer.

## 3.2 Discussão fora dos trechos reancorados

`discussao.tex` é o capítulo com maior dívida: §7.1 concentra **N4, N5, N7 e N8**.
§7.2 (`associated_with`), §7.3 (limitações) e §7.4 (ameaças à validade) estão
**bem reancorados** — §7.3 em particular abre reconhecendo que "a limitação mais
séria recai sobre a própria caracterização de empate", que é exatamente a
postura certa.

## 3.3 Tipografia e copidesque

**Separador de milhar inconsistente entre prosa e tabelas.** A prosa usa espaço
fino (`19\,210` → "19 210"); **todas** as tabelas geradas usam ponto
(`19.210`). Contagem: 24 ocorrências de `\,` na prosa (16 só em
`metodologia.tex`) contra 28 de ponto nas tabelas. O caso mais visível é dentro
do **mesmo arquivo**: `metodologia.tex:384` escreve "$10\,000$ vezes" no texto
corrido e o Quadro 2 (`:429`), duas páginas adiante, escreve "10.000
reamostragens". Escolher uma convenção — e, se for o ponto, isso é mudança nos
geradores (`scripts/_artifacts.py`), não nos `.tex`.

**Tabela 6 (`dev_history`) renderiza a 7,91 pt** — 66% do corpo do texto
(11,96 pt), a menor fonte do documento fora do quadro de hashes SHA-256. Ela
dobrou de 6 para 12 linhas ao ganhar a segunda semente, e o `\resizebox` a
espremeu para caber na largura. Verifiquei que **nenhuma tabela está sendo
ampliada** (o guarda `\ifdim\width>\textwidth` funciona) — o problema é só esta.
Opções: página em paisagem, dividir por modelo, ou abreviar os cabeçalhos
("Perda (treino)" → "L_tr" etc.). Nota: o item *"Tabela 8 e Tabela 10 estão
muito grandes (a fonte)"* de `revisao_tcc.md` **foi resolvido** — não há mais
nenhuma tabela acima de 11,96 pt.

**Quebra feia no abstract:** `negation_` / `of` é partido entre linhas na pág. 7,
efeito do `\allowbreak` após o sublinhado que `macros.tex:59-63` instala de
propósito. Cosmético.

## 3.4 Documentos de apoio desatualizados

**`tcc/OUTLINE.md`** — quatro divergências:

- "Pipeline (**6 etapas** + determinismo)" — são **7** desde a inserção do
  "Ajuste fino" entre *Marcação* e *Classificação*.
- `dedicatoria.tex` marcada `vazio | placeholder do template` — está **escrita**,
  e é pessoal.
- Discussão descrita como "**Ausência de vantagem** do pré-treinamento clínico" —
  falta o "**consistente**" que o capítulo agora usa.
- **Nenhum** capítulo saiu do status `escrito` para `revisado`, apesar de três
  blocos de revisão + a reancoragem da Fase 3. A legenda de status existe e não
  está sendo usada.

**`tcc/revisao_tcc.md`** — é uma lista de pendências **já resolvidas** que
continua lendo como aberta. Vários itens citam frases que não existem mais
("Essa expectativa, contudo, é raramente testada de forma controlada para o
português", "costuma ser assumida pela literatura em português, e não
verificada"). Vale marcar como fechada ou anotar item a item.

**`tcc/AVALIACAO_DUVIDAS.md`** — registro histórico de 19/08, corretamente
preservado. Duas notas de rodapé: o item 1 (README "tipados") **já foi
corrigido**, e o item 3 fala em "**9 referências** no `.bib`" quando hoje são
**18** com zero órfãs. O item 2 (`max_gap`) já foi atualizado com "RESOLVIDO".

---

# 4. Checklist priorizado

Nada abaixo foi implementado. Ordem sugerida de aprovação.

### P0 — Factual. Corrigir antes de qualquer outra coisa

| # | Item | Onde |
|---|---|---|
| 1 | **N1** — reescrever a atribuição do McNemar na semente 43; separar "*bootstrap* na classe-alvo inverte" de "McNemar aponta para o geral nas duas". Ajustar as 5 passagens da tabela em N1. | §6.6, §7.1, §8.1, resumo PT/EN |
| 2 | **N1b** — acrescentar `mcnemar.b/c` da semente 43 ao `CLAIMS` de `check_tcc_numbers.py`, para que a direção passe a ser conferível | `scripts/check_tcc_numbers.py` |
| 3 | **N2** — remover/reescrever "sem custo esperado na métrica-alvo" | `conclusao.tex:55-56` |
| 4 | **N3** — reancorar a frase do Apêndice B | `apendice_b_lexico_negacao.tex:5-8` |

### P1 — Entrega. A banca vê

| # | Item |
|---|---|
| 5 | Pré-textuais: examinadores reais, coorientadora (ou remover), ficha catalográfica, `Itapemirim` maiúsculo, data de aprovação |
| 6 | Decidir agradecimentos e epígrafe: redigir + acrescentar `\imprimir…` em `main.tex`, ou remover as macros |
| 7 | **H3/H4** — congelar `artigo-sbc/`: nota de obsolescência no README **e** tirar `tables`/`figures` do `make all` (senão `make artigo` gera um PDF que se contradiz) |
| 8 | `siglas.tex`: acrescentar **EDA** e **UMLS** no mínimo |

### P2 — Coerência narrativa

| # | Item |
|---|---|
| 9 | **N5** — parágrafo novo em §7.1 explicando a **instabilidade**, não o empate (é a maior lacuna conceitual) |
| 10 | **N4** — trocar "aprendem a mesma função / erram nos mesmos lugares" por "concentram o erro na mesma fronteira", com ressalva de semente |
| 11 | **N6** — resumo: qualificar "empatam estatisticamente" como *na classe-alvo* e registrar a consistência do McNemar |
| 12 | **N9** — acrescentar a robustez à semente como contribuição (vii) em §8.1 e antecipá-la no Cap. 1 |
| 13 | **N7, N8, N11** — três ajustes de uma frase cada |

### P3 — Higiene do repositório

| # | Item |
|---|---|
| 14 | **H1** — `git rm -r _to_delete/` + `.gitignore` (inclui o `_audit_stage.tar.gz` desta auditoria, **H12**) |
| 15 | **H2** — `git rm -r results/archive_max_gap25/` (duplicata byte a byte; `archive_max_gap20/` **fica**) |
| 16 | **H5** — corrigir o aviso "precisam ser refeitos" nos notebooks 01 e 02 |
| 17 | **H6** — `*.loq` no `tcc/.gitignore` + `git rm --cached tcc/src/main.loq` |
| 18 | **H7, H8** — decidir sobre `TCC_Angelo_Antonio.pdf` e `bibliografia.bib.examples` |
| 19 | **H9** — `git add --renormalize results/archive_max_gap20/` (mata os 4 avisos de CRLF) |
| 20 | **H10, H11** — extensões faltantes no `.gitattributes`; digest no `Dockerfile` |
| 21 | **H13** — decidir sobre `RELATORIO_ORGANIZACAO.md` / `RELATORIO_LIMPEZA.md` (só no histórico desde `5bd2b44`) |

### P4 — Acabamento e decisões de autor

| # | Item |
|---|---|
| 22 | Separador de milhar: escolher `\,` ou ponto e uniformizar (mexe nos geradores) |
| 23 | Tabela 6 a 7,91 pt — paisagem, divisão ou cabeçalhos abreviados |
| 24 | Atualizar `OUTLINE.md` (7 etapas, status `revisado`, dedicatória, "consistente") |
| 25 | Fechar `revisao_tcc.md`; nota de rodapé em `AVALIACAO_DUVIDAS.md` |
| 26 | **N12** — decidir sobre o título; **N10** — equilibrar a referência de §1.2 |

---

## Resumo em uma linha

O documento está numericamente correto e compila limpo; o que sobrou da
reancoragem às pressas são **uma atribuição errada de direção do McNemar
(N1)**, **uma contradição interna em §8.2 (N2)**, **uma frase antiga intacta no
Apêndice B (N3)**, **uma seção de análise que ainda explica o resultado
substituído (N5)** — e uns pré-textuais que nenhuma das rodadas olhou.
