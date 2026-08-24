# Rotas para `negation_of`

*Sete caminhos para especializar o modelo — e um diagnóstico que inverte a premissa de três deles.*

**RECLin-PT · Fase 2 · Levantamento**

`24 ago 2026` · `max_gap = 25` · `4 execuções · 2 sementes` · **nada implementado · nada treinado**

---

## Ponto de partida: O gargalo não é o que a pergunta supõe

A hipótese natural para uma classe minoritária é que ela sofre de *recall*: poucos exemplos, o modelo aprende a ignorá-la. Nas quatro execuções da fase 1, o oposto acontece.

| Métrica | Valor |
|---|---|
| **Recall** de `negation_of` nas quatro execuções | 0,88–0,90 — das 152 negações do teste, o modelo encontra 134 a 137 |
| **Precisão** nas mesmas execuções | 0,54–0,66 — é aqui que o F1 se perde, e **~96% dos falsos positivos vêm de `no_relation`**, não de confusão com `associated_with` |
| Peso que `class_weight=balanced` já atribui a `negation_of` | **40,55** — **115×** o peso de `no_relation` |

A leitura é direta: a classe já está fortemente sobre-corrigida, e o preço aparece todo em precisão. **Qualquer alternativa que dê ainda mais peso a `negation_of`** — peso de classe maior, *oversampling*, *focal loss* com α na minoritária — **empurra o eixo errado** e tende a piorar o F1. Isso atinge as alternativas 1 e 2 da lista, e é por isso que elas aparecem abaixo com veredito diferente do esperado.

O segundo ajuste de premissa é sobre o desbalanceamento. Os 1.606 × 9.852 do corpus não são o que o modelo enxerga: no espaço de *candidatos*, `negation_of` é **0,82% do treino** (1.255 de 152.686) — razão de 121:1, não de 1:6. E esse espaço é quase todo descartável.

---

## O achado: Um filtro de 23 palavras vale mais que qualquer ajuste de treino

Toda negação anotada no SemClinBr tem uma **pista lexical** como primeiro argumento — `SEM`, `NEGA`, `NÃO`, `AUSÊNCIA`, e também negações morfológicas como `AFEBRIL`, `INDOLOR`, `ASSINTOMÁTICO`, `ACIANÓTICO`. Induzi um léxico **apenas do `train`** e o apliquei ao `test` como filtro sobre as predições *já salvas*: se o modelo prediz `negation_of` num par cujo e1 não é pista, a predição cai.

F1 de `negation_of` no teste, antes e depois do filtro léxico. Nenhum treino envolvido — o filtro roda sobre `results/*.preds.json`.

| Execução | F1 atual | F1 c/ filtro | Δ |
|---|---|---|---|
| BioBERTpt · s42 | 0,6765 | 0,8232 | +0,147 |
| BERTimbau · s42 | 0,6937 | 0,8232 | +0,130 |
| BioBERTpt · s43 | 0,7542 | 0,8636 | +0,109 |
| BERTimbau · s43 | 0,7016 | 0,8173 | +0,116 |

A precisão sobe de 0,54–0,66 para **0,77–0,85**; o recall cai de 0,90 para 0,88. O ganho é consistente nas quatro execuções e nas duas sementes.

Três propriedades importam mais que o número em si:

- **Não depende da anotação UMLS.** O léxico é superfície pura, induzido do treino. Não reabre a decisão de `sec:markers_limitacao` — aquilo era sobre as 828 combinações multirrótulo; isto é um bit, não um vocabulário.
- **Não custa GPU.** Roda em segundos na CPU, como `src/significance.py`.
- **Cobre 147 das 152 negações do teste** com 23 formas (frequência ≥ 3 no treino). Com 83 formas, cobre 149.

> **Ressalva que não pode sumir do texto.** O número acima é o F1 *de `negation_of`*. O efeito no macro-F1 exige saber para qual classe cada predição filtrada deveria migrar, e os `.preds.json` guardam só o *argmax* — sem probabilidades. Medir o macro-F1 exige reexecutar a inferência com o `best_model/` (que está no Drive, não no repositório).

---

## Alternativas: Sete caminhos, com o que cada um custa

### 01 — Função de perda e limiar de decisão
**Veredito: eixo invertido**

- **Em que consiste:** Trocar a *cross-entropy* ponderada por *focal loss*, ajustar os pesos de classe, ou calibrar o limiar de decisão em vez de usar *argmax*.
- **Por que ajudaria:** Provavelmente não ajuda na direção assumida. O peso já está em 40,55 (115× `no_relation`) e o recall já é 0,90. Mais peso compra recall que não existe para comprar, pagando em precisão. Se há ganho aqui, o sinal do ajuste é **reduzir** o peso ou calibrar o limiar — o oposto do que "tratar o desbalanceamento" sugere.
- **Custo:** Uma execução completa ≈ 3 h de GPU (3 épocas × ~57 min). Uma varredura mínima de 3 pesos × 2 sementes ≈ 18 h. Implementação trivial: o `--class-weight` já existe em `relation_extraction.py`.
- **Como avaliar:** Seleção no `dev`, teste uma vez; *bootstrap* pareado sobre o F1 de `negation_of` contra a fase 1 via `src/significance.py`. O espaço de candidatos não muda, então as predições são pareáveis.
- **Risco principal:** Gastar a maior parte do orçamento de GPU no eixo errado e concluir "não houve ganho" — um resultado nulo caro e pouco informativo para a banca.

### 02 — Estratégias em nível de dado
**Veredito: parcialmente redundante**

- **Em que consiste:** Três coisas distintas, com vereditos diferentes:
  - **a) Oversampling da minoritária.** Matematicamente quase equivalente ao peso de classe que já está aplicado — mesmo eixo, mesmo problema da alternativa 01.
  - **b) Aumento sintético / paráfrase.** Gerar variantes de trechos negados.
  - **c) Recursos de negação em português.** Ver a alternativa 03 para o léxico induzido do próprio corpus, que é o caminho barato. Externamente existe o corpus de [Dalloux et al. (2020)](https://www.cambridge.org/core/journals/natural-language-engineering/article/supervised-learning-for-the-detection-of-negation-and-of-its-scope-in-french-and-brazilian-portuguese-biomedical-corpora/5E5DB27872B07185DB58A1507DFA05D8), com pista *e escopo* anotados em narrativas clínicas brasileiras (~24 mil *tokens*, 1.400 sentenças, 18% com negação; F1 de pista 94,68, de escopo 77,87).
- **Por que ajudaria:** Só (c) ataca a precisão. O corpus de Dalloux traz o que o SemClinBr não tem: **escopo anotado** — exatamente o modo de erro residual descrito na alternativa 06.
- **Custo:** (a) trivial e provavelmente inútil. (b) alto e arriscado. (c) **custo desconhecido e fora do seu controle**: o material veio de três hospitais brasileiros e o artigo não declara disponibilidade pública — presumivelmente restrito, como o próprio SemClinBr. Exigiria contato com os autores, com prazo incerto para um TCC.
- **Como avaliar:** Para (c), o uso seria como pré-treino intermediário ou fonte de supervisão de escopo, avaliado no mesmo teste do RECLin-PT — o corpus externo nunca entra na avaliação.
- **Risco principal:** Em (b), a semântica da negação é frágil a paráfrase: trocar "nega dor" por "não relata dor" é seguro, mas geradores introduzem inversões de polaridade silenciosas, e você estaria injetando ruído justamente na classe que quer proteger. Em (c), o risco é de cronograma, não técnico.

### 03 — Híbrido: pista lexical + classificador neural
**Veredito: maior razão ganho/custo**

- **Em que consiste:** Duas variantes, do mais barato ao mais integrado:
  - **a) Filtro a posteriori.** A predição `negation_of` só vale se e1 for pista de negação. Zero treino — é a tabela da seção anterior.
  - **b) Pista como *feature* de entrada.** Um marcador adicional (por exemplo `[NEG]` junto de `[E1]`) informando que aquele argumento é pista. O modelo passa a poder condicionar em vez de reinferir da superfície.
- **Por que ajudaria:** Ataca exatamente o gargalo medido. Os falsos positivos são pares onde o modelo vê contexto de negação na janela e atribui a relação ao par errado; a pista restringe o espaço onde a classe pode ser predita. Em (b), o modelo ainda decide o alvo — o que preserva a capacidade de tratar escopo, que uma regra rígida não teria.
- **Custo:** (a) **zero GPU**, segundos de CPU, algumas dezenas de linhas. (b) uma execução completa por semente (~3 h cada); a mudança em `build_marked_window` é pequena.
- **Como avaliar:** (a) é comparável à fase 1 sem ressalvas — mesmo `y_true`, predições pareáveis, *bootstrap* direto. Reserve o dev para escolher a frequência mínima do léxico (1, 2 ou 3) e não toque no teste até fechar. Para (b), comparar contra (a), não contra a fase 1: a pergunta é se aprender a pista supera aplicá-la.
- **Risco principal:** Duplo. **Científico:** o Apêndice B argumenta que abordagens léxicas não tratam escopo amplo nem conjunção, e que é isso que motiva modelos contextuais — introduzir a pista precisa ser enquadrado como *restrição do espaço de decisão*, não como retorno ao NegEx, ou o texto entra em contradição consigo mesmo. **Metodológico:** o léxico deve ser induzido só do `train`; induzi-lo do corpus inteiro é vazamento.

### 04 — Escolha do *encoder*-base
**Veredito: decidir depois, não antes**

- **Em que consiste:** Quatro opções reais:
  - **a) Manter BERTimbau** pelo argumento de custo e estabilidade já escrito no §8.
  - **b) Trocar de variante do BioBERTpt.** O projeto usa `biobertpt-all`. No mesmo SemClinBr, `biobertpt-clin` supera `all` em NER (0,6981 × 0,6830). Você pode estar comparando o BERTimbau com a variante clínica mais fraca.
  - **c) mmBERT-base** ([jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)), o melhor no mesmo corpus em NER (micro-F1 0,7646), sem qualquer pré-treino biomédico em português.
  - **d) Nenhum dos anteriores por ora** — gastar o mesmo orçamento em **mais sementes** dos dois candidatos atuais.
- **Por que ajudaria:** Um *encoder* melhor eleva o piso de todas as outras alternativas. Mas note que a inversão de sinal na semente 43 é o argumento mais forte para (d): com duas sementes e amplitude de 0,078 no BioBERTpt, **o resultado misto não é evidência de empate, é evidência de que duas sementes não decidem**. Trocar de *encoder* agora troca uma incerteza mal medida por outra.
- **Custo:** (b) ≈ 3 h por semente, nenhuma mudança de código além do identificador. (c) idem, mais o risco de *tokenizer* e de compatibilidade da arquitetura ModernBERT com o laço de treino atual. (d) ≈ 3 h por semente por modelo; 4 sementes adicionais em ambos ≈ 24 h.
- **Como avaliar:** Com 5+ sementes, agregação formal com IC da diferença — que é a primeira continuidade já prometida no §8. Cuidado: os resultados de NER citados **não** transferem para RE; servem para escolher o que testar, nunca para afirmar o resultado.
- **Risco principal:** Trocar a base antes de estabilizar a medição invalida a comparação da fase 1 como ponto de partida, e o TCC perde a linha que liga as duas fases. Já (c) tem risco adicional de escopo: um *encoder* multilíngue moderno muda a narrativa de "clínico × geral em português" para outra coisa.

### 05 — Modelo especializado em espaço restrito
**Veredito: torna a fase 2 barata**

- **Em que consiste:** Duas etapas: a pista lexical restringe os candidatos, e um classificador dedicado decide o alvo dentro desse espaço. O modelo de três classes da fase 1 continua respondendo pelo resto.
- **Por que ajudaria:** Muda a economia do problema. Restringindo a candidatos cujo e1 é pista de negação:

  | Split | Candidatos | Restrito | Positivos | neg:pos | Recall preservado |
  |---|---|---|---|---|---|
  | train | 152.686 | 8.090 (5,3%) | 0,82% → **15,0%** | 121:1 → **5,7:1** | 0,967 |
  | test | 19.210 | 975 (5,1%) | 0,79% → **15,3%** | 126:1 → **5,5:1** | 0,980 |

  O desbalanceamento cai de duas ordens de grandeza para uma, e o treino fica **~19× mais barato**: minutos por época em vez de ~57. Isso é o que viabiliza varrer hiperparâmetros de verdade dentro de um TCC.
- **Custo:** ≈ 10 min por execução em vez de 3 h. Implementação moderada: um segundo laço de treino e uma regra de composição das duas saídas.
- **Como avaliar:** **Aqui mora a maior armadilha do levantamento.** O modelo restrito produz um `y_true` de tamanho diferente, e `src/significance.py` aborta ao parear predições de espaços distintos — a mesma armadilha já registrada na migração de `max_gap`. As predições precisam ser **remapeadas para o espaço completo de 19.210** (tudo que o filtro descartou entra como não-`negation_of`) antes de qualquer teste.
- **Risco principal:** O teto de recall passa a ser o do filtro: 0,967 no treino, 0,980 no teste. As 3 negações do teste cujo e1 não é pista tornam-se irrecuperáveis por construção, e isso precisa ser declarado como limitação, do mesmo modo que o teto de `max_gap` já é.

### 06 — Escopo da negação sobre listas coordenadas
**Veredito: o erro real, o mais difícil**

- **Em que consiste:** Tratar explicitamente o escopo, em vez de classificar cada par isoladamente: decidir de uma vez sobre quais alvos uma pista se estende. É a continuidade "tratamento explícito do escopo da negação" já prevista no §8.
- **Por que ajudaria:** É o que sobra depois do filtro. Nos ~40 falsos positivos remanescentes, o padrão dominante é a lista coordenada — `NEGA DM, DLPD, TABAGISMO, ALERGIA MEDICAMENTOSA`, em que o modelo liga a pista a todos os itens. Os falsos negativos são o espelho: em `NEGA HF DE GLAUCOMA E CEGUEIRA`, o gold liga aos dois e o modelo prediz `associated_with`.
- **Custo:** Alto. Exige mudar a formulação de classificação de pares para algo com decisão conjunta, e é a única alternativa da lista que não cabe no laço de treino atual.
- **Como avaliar:** Métrica por pista, não por par: dada uma pista, o conjunto de alvos previsto bate com o anotado? Isso mede o fenômeno; o F1 por par continua sendo o número comparável com a fase 1.
- **Risco principal:** **O gold pode não sustentar o ganho.** A convenção de anotação do SemClinBr é de uma pista para um alvo em **91,9% do treino e 95,2% do teste** (apenas 94 de 1.167 pistas têm mais de um alvo). Um modelo que aprendesse escopo corretamente seria penalizado pela métrica sempre que o anotador tivesse parado no primeiro item. Ver a alternativa 07 antes de investir aqui.

### 07 — Auditoria da anotação residual
**Veredito: faça primeiro**

- **Em que consiste:** Revisar manualmente uma amostra dos falsos positivos e falsos negativos de `negation_of` e classificar cada um: erro do modelo, lacuna de anotação, ou ambiguidade genuína. São ~100 casos por execução — uma tarde de trabalho.
- **Por que ajudaria:** Define o teto real antes de você gastar GPU perseguindo-o. Já há indício forte: **13% a 17% dos falsos positivos são pistas que negam outra coisa no mesmo documento** — ou seja, pistas de negação legítimas, com alvo anotado em outro par. E casos como `SEM PRESENÇA DE SINTOMAS` ou `sem sinais de isquemia miocárdica`, contados hoje como erro, são negações que qualquer clínico leria como tal.
- **Custo:** O mais baixo da lista: zero GPU, zero código novo de treino, algumas horas de leitura.
- **Como avaliar:** O produto é um número — a fração dos "erros" que é lacuna de anotação — e ele reancora a leitura de todos os resultados seguintes. Vira material de apêndice, no mesmo espírito do Apêndice B.
- **Risco principal:** Ser lido como desculpa para desempenho ruim. O enquadramento tem de ser prospectivo (estabelecer o teto antes de otimizar), e a amostra tem de ser definida por critério fixo, não escolhida a dedo depois de ver os resultados.

---

## Comparação: As sete lado a lado

Custo de GPU estimado a partir de ~57 min por época com max_gap=25, 3 épocas por execução. Ganho esperado refere-se ao F1 de negation_of.

| # | Alternativa | Custo GPU | Complexidade | Ganho esperado | Risco principal |
|---|---|---|---|---|---|
| 07 | Auditoria da anotação | nenhum | baixa | nenhum direto — define o teto | ler como desculpa |
| 03a | Filtro léxico a posteriori | nenhum | baixa | +0,109 a +0,147 (medido) | contradizer o Apêndice B |
| 05 | Espaço restrito, modelo dedicado | ~10 min/exec. | média | alto, ainda não medido | predições não pareáveis |
| 03b | Pista como *feature* | ~3 h/semente | baixa | médio, sobre 03a | pode não superar 03a |
| 04b | Trocar para biobertpt-clin | ~3 h/semente | trivial | incerto (evidência é de NER) | refazer a comparação da fase 1 |
| 04d | Mais sementes dos dois atuais | ~24 h | trivial | nenhum em F1 — reduz o IC | consumir o orçamento sem ganho |
| 01 | Perda / limiar | ~18 h (varredura) | baixa | baixo; eixo provavelmente invertido | resultado nulo caro |
| 02a | *Oversampling* | ~3 h/exec. | baixa | redundante com o peso atual | piorar a precisão |
| 02b | Aumento sintético | ~3 h/exec. + geração | alta | incerto | inverter polaridade em silêncio |
| 02c | Corpus de Dalloux (escopo) | indefinido | alta | alto, se houver acesso | disponibilidade não confirmada |
| 04c | mmBERT-base | ~3 h/semente | média | incerto (evidência é de NER) | muda a narrativa do trabalho |
| 06 | Escopo e coordenação | alto | alta | alto se o gold sustentar | gold com 92–95% de fan-out 1 |

---

## Sugestão — a decisão é sua: Se fosse eu, nesta ordem

O que segue é recomendação, não conclusão. A lógica é simples: estabelecer o teto antes de persegui-lo, colher o ganho que não custa GPU, e só então gastar GPU — em execuções baratas, não caras.

1. **Auditar os erros residuais (alt. 07)** — Algumas horas, sem GPU. Sem esse número você não sabe se está perseguindo 0,86 ou 0,95, e a alternativa 06 pode estar morta antes de começar.
2. **Fechar o filtro léxico como resultado (alt. 03a)** — Já está medido: +0,11 a +0,15 nas quatro execuções, sem treino. Escolha a frequência mínima do léxico no `dev` e reexecute a inferência para poder reportar também o macro-F1.
3. **Construir o modelo especializado no espaço restrito (alt. 05)** — É a "fase 2 de verdade" que você descreveu, e é o único caminho em que o custo por experimento cai o suficiente para você iterar. Remapeie as predições para o espaço completo desde o primeiro dia.
4. **Só então decidir o *encoder* (alt. 04)** — Com execuções de ~10 minutos, 5 ou 10 sementes deixam de ser proibitivas e a pergunta "clínico ou geral" finalmente ganha uma medida com IC estreito — que é justamente o que a fase 1 não conseguiu entregar. Decidir agora, com duas sementes, é decidir no ruído.

> **Sobre o argumento do BERTimbau.** Ele sobrevive, mas por um motivo que vale explicitar: não porque o resultado misto o favoreça, e sim porque nenhuma das alternativas acima depende do *encoder*. O ganho de 03a e 05 aparece nas quatro execuções, com os dois modelos. Isso torna a escolha do *encoder* uma decisão de custo — que é exatamente como o §8 já a enquadra — e não um pré-requisito da fase 2.

---

## Bloqueios: O que depende de você antes de qualquer código

**A pista lexical entra ou não entra?**
É a decisão de fundo, e é conceitual antes de ser técnica. O Apêndice B hoje argumenta que abordagens léxicas *não bastam* — o que continua verdadeiro —, mas o texto precisa distinguir "regra que decide" de "regra que restringe o espaço onde o modelo decide". Sem esse enquadramento, as alternativas 03 e 05 entram em contradição com o que já está escrito.

**A fase 2 assume entidades *gold*?**
Todos os números aqui assumem spans anotados — que já é a premissa da fase 1, já que os candidatos saem de `doc["entities"]`. Mas fim a fim, com NER automática, o teto passa a ser o F1 da NER no SemClinBr (0,65–0,76). Se a fase 2 pretende ser fim a fim, isso muda o alvo; se não, precisa ficar declarado.

**Vale pedir acesso ao corpus de Dalloux?**
É o único recurso externo com escopo de negação anotado em português clínico brasileiro, e a disponibilidade não está declarada no artigo. Se você quiser essa porta aberta, o pedido tem de sair agora — o prazo de resposta é o risco, não a técnica.

**Qual o orçamento real de GPU até a entrega?**
A ordenação acima muda bastante se houver 100 h disponíveis ou 20 h. Com folga, a alternativa 04d (mais sementes) sobe de posição, porque ela conserta a fraqueza metodológica que a banca mais provavelmente vai apontar.

**O `artigo-sbc/` acompanha ou congela?**
Pendência herdada da fase 1, e ela bloqueia qualquer `make`: a prosa do artigo ainda está em `max_gap=20` enquanto as tabelas se regeraram em 25. Qualquer alvo do Makefile rodado durante a fase 2 produz um PDF internamente contraditório que compila sem erro.

---

**Como os números foram obtidos.** Tudo em cima de `results/*.preds.json`, `results/tcc_eda.json` e `data/splits/*.jsonl`, com a geração de candidatos reimplementada de forma idêntica a `src/candidates.py` (`max_gap=25`). O alinhamento com os sidecars foi verificado: zero divergências entre o `y_true` reconstruído e o salvo, nas quatro execuções. Nada foi escrito no repositório e nenhum treino foi executado.

**Fontes externas:** [Dalloux et al., *Natural Language Engineering* (2020)](https://www.cambridge.org/core/journals/natural-language-engineering/article/supervised-learning-for-the-detection-of-negation-and-of-its-scope-in-french-and-brazilian-portuguese-biomedical-corpora/5E5DB27872B07185DB58A1507DFA05D8) · [Almeida et al., arXiv:2603.26510](https://arxiv.org/html/2603.26510v1) · [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) · [mmBERT, arXiv:2509.06888](https://arxiv.org/html/2509.06888v1)
