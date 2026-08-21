# Avaliação das dúvidas da revisão manual — TCC RECLin-PT

Data: 19/08/2026 · Escopo: **somente avaliação**. Nenhum arquivo de `tcc/`,
`artigo-sbc/` ou `src/` foi alterado. Nenhum comando `git` foi executado.

Documento de referência de contexto: `tcc/revisao_tcc.md` (as anotações originais).
Este arquivo é a resposta a elas, item por item, na mesma ordem.

**Legenda dos vereditos**

- 🔴 **Válido, corrigir** — a preocupação procede e há defeito no texto.
- 🟠 **Sem base suficiente** — a afirmação existe, mas não se sustenta sem referência ou reformulação.
- 🟡 **Parcialmente válido** — a informação existe, mas a estrutura/ênfase induz ao erro.
- 🟢 **Leitura equivocada** — já está coberto; indico onde.

---

## 1.2 Motivação — o texto está focado demais em negação?

**O que o texto realmente faz.** A Seção 1.2 é **um único parágrafo**, e ele é
integralmente sobre negação: abre com "A negação clínica é um fenômeno linguístico
de alto impacto", segue para a lexicalização em português, e fecha na expectativa
sobre pré-treinamento clínico. Não há uma linha sobre `associated_with` nem sobre
extração de relações em sentido amplo.

**Mas o projeto é, de fato, centrado em negação.** Isso não é impressão — está no
protocolo:

- `metodologia.tex` §4.7 nomeia o F1 de `negation_of` como **"métrica-alvo"**, com essas palavras.
- Os **dois** testes de significância (McNemar e *bootstrap* pareado) são calculados
  sobre o F1 de `negation_of`, e só sobre ele.
- A pergunta de pesquisa (§1.1) é resolvida por essa métrica.
- Os trabalhos futuros (§8.2) miram `negation_of`.

Ou seja: a Motivação **não está descalibrada em relação ao trabalho**. Ela está
calibrada em relação ao que o trabalho de fato mede.

**O problema real é outro, e é o inverso do que você anotou.** O espaço de rótulos
prometido em §1.1 e §1.3.1 tem **três** classes, e a introdução inteira nunca
justifica por que `associated_with` está lá. A pergunta que a banca faz não é
"por que tanta negação?", é:

> *"Se sua métrica-alvo é só `negation_of` e seus dois testes estatísticos são só
> sobre `negation_of`, por que o objetivo geral promete um classificador de três
> classes? O que `associated_with` está fazendo no trabalho?"*

Há resposta técnica (a classe `no_relation` exige gerar negativos, e `associated_with`
é a outra classe positiva anotada no corpus — está em §2.2 e §4.6), mas ela está no
Cap. 2 e no Cap. 4, não na motivação. Quem lê o Cap. 1 de ponta a ponta não recebe
essa justificativa.

**Veredito: 🟡 parcialmente válido — mas por motivo oposto ao anotado.** O foco em
negação está correto e é coerente com o protocolo. Falta é **uma ou duas frases em
1.2 motivando a formulação de três classes**, para que 1.2 e 1.3.1 não pareçam falar
de trabalhos diferentes. Não é reescrever a motivação; é acrescentar a ponte.

---

## 1.2 Motivação — a frase "raramente testada de forma controlada para o português"

**Verificação.** `introducao.tex` tem **1 citação no arquivo inteiro**
(`oliveira_2022_semclinbr`, no segundo parágrafo, sobre o SemClinBr). A frase em
questão está na linha 48-51 e **não tem nenhuma citação**, nem antes nem depois.

A frase faz **três** afirmações, todas negativas (afirmações de ausência — o tipo
mais difícil de defender):

1. a expectativa "é raramente testada de forma controlada para o português";
2. "trabalhos costumam adotar um modelo clínico por hipótese";
3. "sem isolar o efeito do pré-treinamento de domínio sob protocolo estatístico rigoroso".

Nenhuma delas tem apoio no texto ou nas referências. Para sustentar (1) e (2) seria
preciso ter examinado um conjunto de trabalhos em português e reportado o que se
encontrou. A bibliografia do TCC tem **9 entradas no total**, das quais apenas **3**
são específicas do português (SemClinBr, BioBERTpt, BERTimbau). Não dá para afirmar
o que "a literatura em português costuma fazer" citando três trabalhos dela.

**Se a banca perguntar "de onde vem isso?", hoje não há resposta no documento.**

**Agravante que não estava na sua lista:** o **mesmo parágrafo** fecha com uma segunda
afirmação igualmente sem apoio — *"a falta de reprodutibilidade é uma das principais
críticas dirigidas à literatura de PLN biomédico"*. Essa é bem mais fácil de referenciar
(é literatura consolidada), mas hoje está tão descoberta quanto a outra.

**Veredito: 🟠 sem base suficiente, precisa de referência ou reformulação.** Duas saídas,
em ordem de preferência:

- **(a)** Acrescentar um levantamento curto — 3 a 5 trabalhos de PLN clínico em português —
  e citá-los. Isso resolve esta frase **e** as duas irmãs dela em 3.2 e 3.3 (ver abaixo).
- **(b)** Reformular como afirmação sobre a própria busca, que é defensável sem
  levantamento exaustivo: *"não localizamos, para o português, avaliação controlada
  que isole o efeito do pré-treinamento de domínio na extração de relações"*. É uma
  afirmação sobre o que **você** procurou, não sobre o que a literatura **é**.

---

## 1.3.1 Objetivo geral — foca no pipeline e não no modelo RECLin-PT?

**Como "RECLin-PT" é definido no TCC.** Rastreei todas as 14 ocorrências no texto.
A definição está em `introducao.tex:30-31`, na primeira vez que o nome aparece:

> "Este trabalho **integra o desenvolvimento** do **RECLin-PT**, um modelo aprimorado
> de extração de relações em texto clínico em português sobre o corpus SemClinBr.
> **Como etapa inicial dessa construção**, investiga-se a seguinte questão de projeto..."

Isso é repetido, com a mesma estrutura, em `proposta_prototipo.tex:5-11` e em
`conclusao.tex:6-7`. E `conclusao.tex:65-66` fecha: *"os baselines aqui reportados
estabelecem o ponto de partida do RECLin-PT, e não um encerramento."*

Portanto, **no TCC**, RECLin-PT = um modelo **ainda a construir**, do qual este
trabalho é a etapa preliminar. Sob essa definição, o objetivo geral está correto e
inclusive **nomeia o RECLin-PT explicitamente na sua última linha**: *"...de modo a
identificar a melhor base sobre a qual construir o RECLin-PT."* Ele não "deixa de
falar do modelo"; ele diz que o produto desta etapa é a **decisão de qual encoder usar**.

**Veredito: 🟢 leitura equivocada — está coberto em `introducao.tex:30-35` e na última
oração do próprio §1.3.1.**

**Porém — e isto é um achado real que a sua dúvida encontrou por tabela:** o nome
"RECLin-PT" significa **três coisas diferentes em três artefatos do mesmo projeto**:

| Artefato | O que "RECLin-PT" designa |
|---|---|
| `README.md` (título) | O **projeto/repositório inteiro**, isto é, este TCC |
| `artigo-sbc/artigo.tex:154` | O **trabalho atual**, em presente: *"o RECLin-PT aborda a etapa subsequente de RE"* |
| `tcc/src/textuais/*.tex` | Um **modelo futuro**, ainda não treinado |

Um membro da banca que abra o repositório, leia o README ("RECLin-PT — Extração de
Relações em Notas Clínicas em Português"), depois leia no TCC que o RECLin-PT é
trabalho futuro, **vai** apontar isso. Ver também o item de 8.2 abaixo — é o mesmo
problema, e é a raiz das suas duas dúvidas sobre nomenclatura.

---

## 1.3.2 Objetivos específicos — focados demais no pipeline?

**Contagem.** São 8 itens:

| # | Item | Natureza |
|---|---|---|
| 1 | *parser* fiel ao XML | infraestrutura |
| 2 | caracterização/EDA do corpus | infraestrutura |
| 3 | partição estratificada e congelada | infraestrutura |
| 4 | gerador determinístico de candidatos | infraestrutura |
| 5 | **"Ajustar dois *encoders* para o português..."** | **ajuste fino** |
| 6 | comparação sob teste pareado | avaliação |
| 7 | robustez à semente | avaliação |
| 8 | **"Identificar... a melhor base sobre a qual desenvolver o RECLin-PT"** | **RECLin-PT** |

O ajuste fino **está** (item 5) e o RECLin-PT **está** (item 8). A percepção de
ausência tem uma causa identificável: o item 5 é redigido pela ótica da *paridade*
("...sob um único núcleo de implementação, de modo que difiram apenas no *checkpoint*
de pré-treino"), o que o faz **ler como garantia metodológica, não como "treinar o
modelo"**. O verbo "ajustar" está lá, mas soterrado pela cláusula que vem depois.

**Veredito: 🟢 leitura equivocada — está coberto nos itens 5 e 8.** Ajuste opcional e
de baixo custo: mover a cláusula de paridade do item 5 para uma oração subordinada,
ou separar em dois itens ("ajustar os dois encoders" / "garantir paridade total de
implementação"), de modo que o ato de treinar apareça como objetivo próprio.

---

## 2.5 Modelos baseados em *transformers* — o XLM-R é usado?

**Não. Em lugar nenhum.** Busca em todo o repositório (`.py`, `.json`, `.md`, `.tex`,
`.bib`, `Makefile`, `.sh`, `.yml`), excluindo `.git`:

- `src/` — **0 ocorrências**
- `results/` — **0 ocorrências**
- `Makefile`, `run.sh`, `notebooks/` — **0 ocorrências**
- Únicas ocorrências: `referencial_teorico.tex:70`, `trabalhos_relacionados.tex:30`,
  `conclusao.tex:61` (trabalhos futuros) e a entrada `conneau_2020_xlmr` no `.bib`.

**A frase de 2.5 é, como está escrita, factualmente falsa.** Linhas 70-75:

> "Para o português, modelos multilíngues como o **XLM-R** e, sobretudo, modelos
> especializados no domínio clínico como o **BioBERTpt**... constituem candidatos
> naturais para a extração de relações clínicas, **e é sobre eles que se apoia a
> proposta deste trabalho**."

O "eles" abrange XLM-R, que não é usado. E há um problema **maior**, que passou
despercebido na sua leitura: **o BERTimbau não é sequer mencionado na Seção 2.5**.
Ou seja, a seção que declara a base teórica do trabalho **nomeia um modelo que não é
usado e omite metade do experimento**.

**Correção de premissa:** você anotou "Bert, Bertimbau, XLM-R e BioBert". O texto de
2.5 cita, na verdade, **BERT, XLM-R e BioBERT*pt***. Não cita BERTimbau, e não cita
BioBERT (o inglês, de Lee et al.) — esse não está nem no `.bib` do TCC (mas está no
do artigo; ver item 3.1).

As outras duas ocorrências do XLM-R estão **corretas** e não precisam de nada:
`trabalhos_relacionados.tex:30-32` o apresenta como *"alternativa de base ampla, útil
como ponto de comparação futuro"*, e `conclusao.tex:61` o lista em trabalhos futuros.
O defeito é exclusivo da oração final de 2.5.

**Veredito: 🔴 válido, corrigir — e o problema é maior do que a dúvida sugeria.**
A frase precisa (i) parar de dizer que o trabalho se apoia no XLM-R e (ii) incluir o
BERTimbau, que é um dos dois modelos efetivamente comparados.

---

## 3.1 Recursos e desafios em inglês — qualidade da seção

Sua leitura está correta em todos os pontos. Os fatos:

- **Zero citações.** O arquivo `trabalhos_relacionados.tex` tem 4 citações no total,
  **todas em 3.2 e 3.3**. A Seção 3.1 não tem nenhuma.
- **Nomeia "i2b2" e "n2c2" sem explicar o que são** e sem expandir as siglas. Nenhuma
  das duas consta de `pre_textuais/siglas.tex`.
- **É a seção mais curta do documento**: um parágrafo, ~7 linhas, para uma seção
  numerada inteira.
- Não conecta a nada: menciona "preocupações transpostas para este trabalho" sem
  dizer onde, no TCC, essa transposição acontece.

**E aqui está o achado que muda o custo da correção.** Comparei com o parágrafo
equivalente em `artigo-sbc/artigo.tex:134-145`. O artigo tem **a mesma redação**,
mas **com a citação** — e com uma frase a mais que o TCC perdeu:

> "...campanhas de avaliação como as séries i2b2 e n2c2**~\cite{uzuner:11}**, que
> disponibilizaram corpora anotados... No plano dos modelos, o
> **BioBERT~\cite{lee:20}** demonstrou ganhos da adaptação de domínio biomédico em
> inglês, e técnicas de representação de pares por **marcadores de
> entidade~\cite{soares:19}** tornaram-se padrão para RE com modelos baseados em BERT."

As três entradas — `uzuner:11` (i2b2/VA 2010), `lee:20` (BioBERT) e `soares:19`
(*Matching the Blanks*) — **existem em `artigo-sbc/artigo.bib` e estão ausentes de
`tcc/src/bibliografia.bib`**. Isto é uma **regressão** introduzida na reescrita do TCC,
não uma lacuna original. A correção é praticamente mecânica: copiar as três entradas.

**Bônus — a mesma referência resolve outro buraco.** `proposta_prototipo.tex:92-93`
afirma *"A técnica é padrão em extração de relações com modelos baseados em BERT"*
**sem citação**. É exatamente a afirmação que `soares:19` sustenta, e o artigo já a
cita para isso (`artigo.tex:294`). Trazer a entrada resolve os dois pontos de uma vez.

**Veredito: 🔴 válido, corrigir — e a correção já está pronta no documento irmão.**
Além das citações, a seção precisa expandir i2b2/n2c2 na primeira menção (e entrar
na lista de siglas) e ganhar uma frase ligando explicitamente as "preocupações
transpostas" às seções do TCC onde elas aparecem (§4.5 vazamento, §4.6 desbalanceamento).

---

## 3.2 Recursos para o Português — "costuma ser assumida pela literatura... e não verificada"

O parágrafo (linhas 34-37) tem **duas** afirmações, e elas têm status muito diferente.
Vale separá-las, porque só uma é problema:

**(1) *"os ganhos relatados para o BioBERTpt foram medidos em NER, e não em extração
de relações"* — 🟢 sustentada.** É verificável na própria referência já citada:
`schneider_2020_biobertpt` se intitula *"BioBERTpt — A Portuguese Neural Language
Model for **Clinical Named Entity Recognition**"*. Se a banca perguntar, a resposta
é o título do artigo. Sem problema.

**(2) *"A transferência dessa vantagem para a etapa subsequente costuma ser assumida
pela literatura em português, e não verificada"* — 🟠 não sustentada.** É a mesma
afirmação de ausência de 1.2, aplicada a "a literatura em português", sem citação e
sem levantamento. Vale aqui o mesmo argumento: com 3 trabalhos em português na
bibliografia, não há base para descrever o hábito de uma literatura.

**Veredito: 🟠 sem base suficiente na segunda metade do parágrafo, precisa de
referência ou reformulação.** A primeira metade pode ficar como está. Recomendação:
manter o contraste, mas ancorá-lo no que é verificável —

> *"Os ganhos relatados para o BioBERTpt foram medidos em NER (SCHNEIDER et al., 2020).
> Não localizamos, para o português, trabalho que verifique se essa vantagem se
> transfere para a extração de relações, lacuna que este trabalho endereça."*

Isso mantém o posicionamento intacto e troca uma afirmação sobre a literatura (que
exige levantamento) por uma afirmação sobre a própria busca (que não exige).

---

## 3.3 — "boa parte da literatura em português concentra-se em NER e adota um modelo clínico por hipótese"

**É a terceira ocorrência da mesma afirmação sem apoio.** Sem citação, novamente.

Vale ver as três lado a lado, porque isso muda a estratégia de correção:

| Local | Afirmação |
|---|---|
| `introducao.tex:48-51` (§1.2) | "raramente testada... trabalhos costumam adotar um modelo clínico por hipótese" |
| `trabalhos_relacionados.tex:35-37` (§3.2) | "costuma ser assumida pela literatura em português, e não verificada" |
| `trabalhos_relacionados.tex:41-42` (§3.3) | "boa parte da literatura em português concentra-se em NER e adota um modelo clínico por hipótese" |

É **a mesma tese**, repetida três vezes, **sem sustentação em nenhuma delas** — e ela
é o pilar do posicionamento do trabalho inteiro. Se a banca derrubar essa tese, o
"gap" que justifica o TCC vai junto. É o ponto mais frágil de toda a lista.

A boa notícia: **um único conserto resolve as três.** Um parágrafo de levantamento em
§3.2 ou §3.3 — 3 a 5 trabalhos de PLN clínico em português, dizendo o que cada um
faz e apontando que nenhum isola o efeito do pré-treinamento em RE — sustenta a
afirmação, e os outros dois pontos passam a poder referenciá-la.

**Veredito: 🟠 é suposição do autor, precisa de referência ou reformulação — e é a
correção de maior alavancagem da lista inteira.**

---

## 3.3 — "estabelecer... uma cadeia completa" em vez do fine-tuning do RECLin-PT

**O enquadramento é intencional e está declarado — em três lugares diferentes, com
todas as letras:**

- `trabalhos_relacionados.tex:46-47` (a própria §3.3, frase seguinte à que te incomodou):
  *"**A contribuição não está em propor uma arquitetura inédita**, mas em estabelecer,
  de maneira reprodutível e estatisticamente fundamentada, uma cadeia completa..."*
- `proposta_prototipo.tex:23-25`: *"A premissa metodológica central é a **paridade**:
  a infraestrutura descrita aqui **não constitui um modelo com nome próprio**, é apenas
  a cadeia que viabiliza a comparação de forma justa."*
- `conclusao.tex:16-27`: as seis contribuições (i)–(vi) são, todas, metodológicas.

Portanto não é o texto "insistindo" em falar de outra coisa — é uma escolha explícita
de posicionamento: **metodologia como contribuição**, com o modelo como etapa seguinte.

**Veredito: 🟢 leitura equivocada — está coberto em `trabalhos_relacionados.tex:46-47`
e `proposta_prototipo.tex:23-25`.**

**Ressalva honesta, porém.** O enquadramento é legítimo, mas é uma aposta: ele só se
sustenta se a banca aceitar "metodologia como contribuição". Como o título e o objetivo
prometem "extração de relações" e o entregável é a comparação de dois modelos
prontos-de-prateleira, a pergunta *"onde está a sua contribuição técnica?"* é
plenamente previsível. O texto **tem** a resposta — mas ela está na última frase da
§3.3 e no meio da §5.1, ou seja, nos dois lugares em que menos gente procura.
Recomendo **antecipar essa declaração para o Cap. 1** (§1.1 ou §1.3), para que ela
seja lida como escolha deliberada logo de saída, e não como desculpa dada no meio do
documento.

---

## 4 Metodologia — o fine-tuning está ausente?

**Não está ausente, mas a sua leitura tem fundamento estrutural real.** Os fatos:

**Onde o ajuste fino de fato aparece:**

- §4.8 (`Ambiente experimental`) — o **Quadro 2** traz taxa de aprendizado, *batch*,
  épocas, perda, escalonador, critério de seleção de modelo e hardware.
- Cap. 5 inteiro — os dois *encoders*, a perda, a representação de entrada, a paridade.
- §6.1 (`Configuração experimental`) — a configuração de treino repetida.
- Apêndice C — o **Quadro 7** com os hiperparâmetros de ajuste fino completos, lidos do
  campo `config` de `results/baseline_*.json`.

**Por que ainda assim parece ausente — e isto é defeito real, não impressão:**

1. **A enumeração de 6 etapas do *pipeline* (§4.1) não tem etapa de treino.** As etapas
   são: *parsing* → particionamento → geração de candidatos → marcação → **classificação**
   → avaliação. A etapa 5 ("Classificação: texto marcado → predições") é **inferência**.
   O ajuste fino, que é o que produz o modelo que faz essa inferência, **não é uma etapa
   do pipeline descrito**. Um leitor atento pergunta: *"cadê o treino no seu pipeline?"*
2. **A expressão "ajuste fino" aparece exatamente UMA vez no capítulo inteiro** — linha
   398, de passagem, numa frase sobre hardware ("o ajuste fino dos *encoders* é executado
   em ambiente com GPU").
3. **A frase de abertura do capítulo exclui o treino da lista**: *"Este capítulo descreve
   o corpus, o pipeline de processamento, a estratégia de particionamento, a geração de
   pares candidatos e o protocolo de avaliação."* Foi exatamente essa frase que você leu
   e anotou — ela **anuncia** que o fine-tuning não está ali. E o Cap. 1 (§1.4, Organização
   do trabalho) repete a mesma lista incompleta.
4. Nenhuma seção do Cap. 4 se chama "Ajuste fino" ou "Treinamento".

**Veredito: 🟡 parcialmente válido — a informação existe (§4.8 + Cap. 5 + §6.1 +
Apêndice C), mas a ausência do treino na enumeração do pipeline é um buraco estrutural
real.** É, das coisas desta lista, a mais fácil de a banca notar, porque um *pipeline*
apresentado como "cadeia completa, do texto bruto às métricas finais" que pula o treino
é uma contradição visível a olho nu. Correção barata: inserir uma etapa entre 4 e 5
("Ajuste fino: texto marcado → modelo treinado") e ajustar as duas frases de abertura
(§4.1 do Cap. 4 e §1.4) para mencioná-la.

---

## 8.2 Trabalhos futuros — por que "o desenvolvimento do RECLin-PT" está aqui?

**Internamente, o TCC é consistente.** Ele avisa desde a primeira menção do nome que
este trabalho é a *etapa inicial* de uma construção maior, e fecha reafirmando isso:

- `introducao.tex:30-32` — "integra o desenvolvimento do RECLin-PT... Como etapa inicial dessa construção"
- `proposta_prototipo.tex:5-11` — idem
- `conclusao.tex:65-66` — "O trabalho encontra-se, portanto, **em andamento**: os
  *baselines* aqui reportados estabelecem o ponto de partida do RECLin-PT, e não um encerramento."

Quem lê o documento inteiro, na ordem, não se confunde.

**Veredito sobre a consistência interna: 🟢 leitura equivocada — coberto em
`introducao.tex:30-32` e `conclusao.tex:65-66`.**

**Mas a sua desconfiança acertou um problema real, que é externo ao TCC.** Como
mostrei no item 1.3.1, "RECLin-PT" designa coisas diferentes no README (o projeto/
este TCC), no artigo SBC (o trabalho atual, em presente) e no TCC (um modelo futuro).
Um avaliador que chegue ao TCC pelo repositório ou pelo artigo **chega com a definição
errada na cabeça** e lê 8.2 exatamente como você leu.

**E há uma segunda questão, que o texto não antecipa em lugar nenhum:** *"um TCC pode
entregar um estudo de baselines e deixar como trabalho futuro justamente o modelo que
dá nome ao trabalho?"* Isso é uma pergunta de banca legítima e provável. Tem defesa
razoável (a decisão de encoder é pré-requisito, e o resultado — o encoder clínico não
ganha — é em si um achado publicável, não um resultado nulo vazio), mas **essa defesa
não está escrita em lugar nenhum do documento**. Vale escrevê-la, e vale alinhar o nome
nos três artefatos antes da entrega.

---

## 8.2 — por que o foco em `negation_of` e não na extração como um todo?

**É consistente com o resto do texto.** O trabalho inteiro é organizado em torno dessa
classe: §1.2 é integralmente sobre negação; §4.7 declara o F1 de `negation_of` como
"métrica-alvo" com essas palavras; **os dois** testes de significância são calculados
só sobre ela; e a justificativa dada em 8.2 ("a mais crítica para a correta interpretação
clínica") é eco direto da frase de abertura de §1.2 ("tratar um achado negado como
presente pode inverter completamente a interpretação de um prontuário"). Não há
contradição com os capítulos anteriores.

**Veredito: 🟢 consistente — não contradiz o enquadramento anterior.**

**Ressalva, e essa é uma pergunta que a banca pode muito bem fazer.** O próprio TCC
diz, em §7.2, que `associated_with` é **a classe mais difícil** (F1 ≈ 0,45, contra
0,73 de `negation_of`) e aponta duas causas para isso. Ou seja: o documento identifica
`associated_with` como o problema e, três páginas depois, propõe melhorar a classe que
**já é a mais forte**. A pergunta é direta:

> *"Por que investir em `negation_of`, que está em 0,73, e não em `associated_with`,
> que o seu próprio Capítulo 7 diz que está em 0,45?"*

A resposta existe e é boa — criticidade clínica: um achado negado lido como presente
inverte a interpretação, um `associated_with` perdido apenas empobrece a extração —
mas **o texto nunca coloca os dois fatos lado a lado**, então nunca dá essa resposta.
Basta uma frase em 8.2 reconhecendo o trade-off explicitamente. Sem ela, a escolha
parece feita por conveniência (mexer no que já vai bem) e não por critério.

---

## Tabelas 8 e 10 — a fonte está grande demais?

**Confirmado, e não é impressão de leitura.** Medi diretamente no PDF compilado
(`tcc/src/main.pdf`, 59 páginas), extraindo o tamanho de fonte real de cada caractere
do corpo de cada tabela (excluídos legenda e linha "Fonte:").

**Corpo de texto do documento: 11,96 pt** (verificado numa página de prosa corrida).

| Tabela | Conteúdo | Fonte medida | vs. corpo | Colunas |
|---|---|---:|---:|:--:|
| **8** | Comparação pareada (semente 42) | **16,07 pt** | **+34%** | 2 |
| **10** | Comparação pareada (semente 43) | **16,07 pt** | **+34%** | 2 |
| 2 | Distância por tipo | 15,49 pt | +30% | 8 |
| 7 | Resultados no teste | 12,84 pt | +7% | 7 |
| 1 | Distribuição das relações | 11,96 pt | 0% | 3 |
| 4 | Candidatos por partição | 11,00 pt | −8% | 6 |
| 9 | Robustez à semente | 10,85 pt | −9% | 6 |
| 3 | Relações por partição | 10,72 pt | −10% | 6 |
| 5 | Teto de *recall* | 8,59 pt | −28% | 6 |
| 6 | Evolução por época | 7,91 pt | −34% | 8 |

**As Tabelas 8 e 10 são, empatadas, as maiores do documento** — 34% acima do corpo de
texto e **2,03× maiores que a menor tabela** (Tabela 6, a 7,91 pt). A sua percepção
está correta.

**Causa raiz identificada.** Todos os fragmentos gerados em `src/tabelas/` são
embrulhados em `\resizebox{\textwidth}{!}{...}`, que força o `tabular` a ocupar
**exatamente** a largura do texto — **inclusive esticando-o quando ele é naturalmente
mais estreito**. O fator que decide se a tabela encolhe ou cresce é a **largura natural
do conteúdo**, não o número de colunas. As Tabelas 8 e 10 têm só duas colunas
(`\begin{tabular}{lc}`) e portanto largura natural pequena: o `resizebox` as **amplia**
em 34%. A Tabela 2 tem 8 colunas, mas o conteúdo é curtíssimo (`3,5`, `1`, `11`), de
modo que a largura natural também é pequena e ela é ampliada quase tanto. Já as Tabelas
5 e 6, com 6 e 8 colunas de rótulos longos, têm largura natural maior que a página e
são **reduzidas** a 8,59 e 7,91 pt. A única tabela **sem** `resizebox` é a Tabela 1
(`distribuicao_relacoes.tex`) — e é justamente a única que cai exatamente nos 11,96 pt
corretos, o que confirma o diagnóstico.

**Dois achados adicionais que não estavam na sua lista:**

1. **A Tabela 2 tem o mesmo defeito** (15,49 pt, +30%) e você não a marcou. Mesma causa,
   mesmo conserto.
2. **A dispersão em si é um problema de padronização.** A variação de 7,91 pt a 16,07 pt
   entre tabelas do mesmo documento é o tipo de inconsistência que um avaliador criterioso
   comenta, mesmo sem medir.

**Veredito: 🔴 válido, confirmado por medição — e o escopo é maior (inclui a Tabela 2).**

**Onde consertar (não apliquei nada).** As tabelas são geradas por script e o próprio
`OUTLINE.md` avisa: *"Tabelas geradas pelo pipeline: **não editar à mão**, regenerar."*
Então o conserto é nos geradores (`scripts/make_tcc_artifacts.py` e
`scripts/make_tcc_eda.py`), não nos `.tex`. O padrão que **limita sem nunca ampliar** é:

```latex
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
```

Isso reduz a tabela quando ela estoura a página e a deixa no tamanho natural (11,96 pt)
quando não estoura. Alternativa, se preferir controle explícito: retirar o `resizebox`
das tabelas estreitas e usar `\small`/`\footnotesize` nas largas — mas aí o tamanho vira
decisão manual por tabela, o que atrita com a regra de geração automática.

---

## Observações fora da sua lista

Coisas que apareceram na verificação e que valem registro, ainda que você não tenha
perguntado:

1. **`README.md` descreve a representação de entrada como "marcadores de entidade
   **tipados**"** — em **duas** passagens (linhas 53 e 337). O código (`build_marked_window` em
   `src/relation_extraction.py`) insere `[E1]…[/E1]`/`[E2]…[/E2]` **sem tipo**, e o
   `OUTLINE.md` registra que essa divergência já foi corrigida no TCC e no artigo.
   **O README ficou para trás.** Se a banca abrir o repositório, vê a versão errada.

2. **`max_gap` 20 vs. 25** — ~~pendente~~ **RESOLVIDO em 21/08/2026.** Os quatro
   experimentos foram reexecutados com `--max-gap 25`, a significância foi refeita
   sobre os novos `.preds.json`, e texto, tabelas e figuras foram reancorados
   (`check_tcc_numbers.py` passa limpo com 93 afirmações). Os resultados de
   `max_gap=20` estão preservados em `results/archive_max_gap20/`. Atenção: a
   mudança **inverteu o sinal** da diferença entre os encoders na semente 43 — ver
   §6.6 e o Capítulo 7, que foram reescritos por causa disso.

3. **Densidade bibliográfica geral.** 9 referências no `.bib`, ~20 citações no texto
   inteiro, e dois capítulos com **1 citação cada** (`introducao.tex`, `discussao.tex`)
   e um com **zero** (`experimentos_e_resultados.tex` — aceitável, é capítulo de
   resultados). Trazer `uzuner:11`, `lee:20` e `soares:19` do artigo já sobe para 12 e
   tapa os buracos mais visíveis, mas o levantamento sugerido no item 3.3 é o que
   realmente resolve.
