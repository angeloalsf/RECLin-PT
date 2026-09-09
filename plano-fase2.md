# Plano da Fase 2 — RECLin-PT especializado em `negation_of`

*Diagnóstico sobre o código e os `preds.json` reais, e a sequência mínima de treinos até uma conclusão defensável.*

`09 set 2026` · `max_gap = 25` · `4 execuções · 2 sementes` · **nada da fase 2 implementado** · continuação de `rotas-negation-of.md` (24/08)

---

## Sumário executivo

1. **Três premissas do enunciado estão erradas ou desatualizadas.** Os falsos positivos de `negation_of` não vêm de confusão com `associated_with`: **95% a 97% vêm de `no_relation`**. O gargalo é precisão, e o peso de classe já está em 40,55.
2. **A meta da fase 2 já está batida, com zero GPU.** O filtro léxico *a posteriori* sozinho supera **os dois** baselines, **nas duas sementes**, de forma significativa sob o protocolo do próprio projeto (bootstrap pareado, IC95 exclui zero, pior caso `p = 0,0148`). Com a regra de um-alvo-por-pista, o pior caso vai a `p < 0,0001`.
3. **O macro-F1 deixa de ser incógnita.** A ressalva de `rotas-negation-of.md` (“exige reexecutar a inferência”) só valia enquanto o destino da predição rebaixada estivesse em aberto. Fixando “rebaixa para `no_relation`”, o macro-F1 é determinado: sobe **+0,034 a +0,047**, e o F1 de `associated_with` fica **inalterado**.
4. **Existe um número que a banca vai pedir e que ainda não foi medido: a regra pura.** Uma regra sem rede neural (`pista + gap ≤ 1`) atinge **F1 = 0,6841** no teste, ou seja, **empata ou supera dois dos quatro baselines**. Sem esse baseline no texto, o Cap. 6 fica exposto; com ele, o argumento fica mais forte, porque a pilha completa o supera por **+0,15 a +0,18** com IC95 folgado.
5. **O risco agora não é o ganho, é a validade.** Todos os números acima foram calculados **no teste**. O bloqueio técnico real não é o remapeamento de espaço citado no cronograma, é mais simples: **`relation_extraction.py` não salva as predições do `dev`**, então nenhuma dessas regras pode ser calibrada honestamente hoje.

---

## 1. Três correções de premissa

### 1.1 Os falsos positivos não vêm de `associated_with`

Recontado agora sobre `results/*.preds.json`, com os candidatos reconstruídos de `src/candidates.py` (`max_gap=25`, alinhamento com `y_true` verificado, zero divergências):

| Execução | TP | FP | FN | P | R | F1 | FP de `no_relation` | FP de `associated_with` |
|---|---|---|---|---|---|---|---|---|
| BioBERTpt · s42 | 137 | 116 | 15 | 0,542 | 0,901 | 0,6765 | **111 (95,7%)** | 5 |
| BERTimbau · s42 | 137 | 106 | 15 | 0,564 | 0,901 | 0,6937 | **101 (95,3%)** | 5 |
| BioBERTpt · s43 | 135 | 71 | 17 | 0,655 | 0,888 | 0,7542 | **68 (95,8%)** | 3 |
| BERTimbau · s43 | 134 | 96 | 18 | 0,583 | 0,882 | 0,7016 | **88 (91,7%)** | 8 |

A leitura correta: o modelo **acha o contexto de negação e o atribui ao par errado**, não confunde negação com associação. Isso muda o alvo da intervenção. Restringir *onde* a classe pode ser predita ataca o erro real; mexer na fronteira entre `negation_of` e `associated_with` não ataca quase nada, porque essa fronteira responde por 3 a 8 casos em 152.

### 1.2 O peso de classe já está no extremo oposto

Com `class_weight=balanced` sobre o treino real (1.255 / 7.096 / 144.335 em 152.686 candidatos), os pesos são **40,55** para `negation_of`, 7,17 para `associated_with` e **0,3526** para `no_relation` — razão de **115×**. O recall já está em 0,88 a 0,90.

Consequência direta, e ela vale como veto: **toda técnica cujo efeito é comprar recall para a minoritária empurra o eixo errado**. Isso inclui aumentar o peso, *oversampling*, e *focal loss* com α na minoritária. Se há ajuste útil nesse eixo, o sinal é **reduzir** o peso, não aumentar.

### 1.3 O filtro por tipo de entidade estava definido errado

`e["type"]` no SemClinBr é multirrótulo separado por `|` (`Negation`, `Negation|Negation`, `Abbreviation|Negation`). Isso muda tudo na alternativa 05:

| Definição do filtro | Candidatos no teste | Teto de recall |
|---|---|---|
| `e1["type"] == "Negation"` | 674 (3,5%) | **0,8355** |
| `"Negation" in e1["type"].split("\|")` | 975 (5,1%) | **0,9803** |

A versão com `==` perde 25 das 152 negações do teste por construção e inviabilizaria a rota. A versão com `in split("\|")` reproduz os 8.090 / 975 candidatos e o teto 0,967 / 0,980 já registrados em memória. **Qualquer implementação do espaço restrito precisa usar a segunda.**

---

## 2. O que já está ganho, sem GPU

Medido agora com a mesma mecânica de `src/significance.py` (bootstrap pareado, 10.000 reamostragens, IC95, α = 0,05). Léxico induzido **só do `train`** (formas de superfície de `e1` nos pares gold `negation_of`, frequência ≥ 2, 17 formas). Regra: se o modelo prediz `negation_of` num par cujo `e1` não é pista, a predição vira `no_relation`.

### 2.1 O filtro contra o próprio baseline

| Execução | F1 base | F1 c/ filtro | Δ | IC95 | p |
|---|---|---|---|---|---|
| BioBERTpt · s42 | 0,6765 | 0,8157 | +0,1392 | [+0,1052; +0,1744] | < 0,0001 |
| BERTimbau · s42 | 0,6937 | 0,8207 | +0,1270 | [+0,0944; +0,1617] | < 0,0001 |
| BioBERTpt · s43 | 0,7542 | 0,8553 | +0,1011 | [+0,0692; +0,1348] | < 0,0001 |
| BERTimbau · s43 | 0,7016 | 0,8073 | +0,1057 | [+0,0759; +0,1385] | < 0,0001 |

A precisão sobe de 0,54–0,66 para **0,75–0,84**; o recall cai de 0,90 para 0,87–0,89.

### 2.2 O teste que interessa: contra o **melhor** baseline, não contra o próprio

A alegação da fase 2 é “supera os DOIS baselines”. O pior caso é a execução filtrada mais fraca contra o baseline mais forte (BioBERTpt · s43 = 0,7542):

| Sistema | F1 | Δ vs melhor baseline | IC95 | p | Significativo? |
|---|---|---|---|---|---|
| filtro(BioBERTpt · s42) | 0,8157 | +0,0615 | [+0,0177; +0,1060] | 0,0048 | **sim** |
| filtro(BERTimbau · s42) | 0,8207 | +0,0665 | [+0,0241; +0,1096] | 0,0018 | **sim** |
| filtro(BioBERTpt · s43) | 0,8553 | +0,1011 | [+0,0692; +0,1348] | < 0,0001 | **sim** |
| filtro(BERTimbau · s43) | 0,8073 | +0,0531 | [+0,0107; +0,0964] | 0,0148 | **sim** |

**As quatro passam, inclusive a mais fraca contra o mais forte.** O IC95 mais apertado ainda exclui zero por +0,0107.

### 2.3 O macro-F1 sai de graça, e não há dano colateral

| Execução | macro base | macro c/ filtro | Δ | `associated_with` | `no_relation` |
|---|---|---|---|---|---|
| BioBERTpt · s42 | 0,6749 | 0,7219 | **+0,0470** | 0,4154 → 0,4154 | 0,9327 → 0,9346 |
| BERTimbau · s42 | 0,6856 | 0,7285 | **+0,0429** | 0,4256 → 0,4256 | 0,9376 → 0,9393 |
| BioBERTpt · s43 | 0,7124 | 0,7465 | **+0,0341** | 0,4390 → 0,4390 | 0,9440 → 0,9451 |
| BERTimbau · s43 | 0,7038 | 0,7394 | **+0,0356** | 0,4601 → 0,4601 | 0,9496 → 0,9508 |

O F1 de `associated_with` não muda **nenhum dígito**, porque a regra só toca predições `negation_of` e o destino é `no_relation`. E o F1 de `no_relation` sobe, porque ~96% do que foi rebaixado era `no_relation` de verdade. Isso **encerra a ressalva** de `rotas-negation-of.md`: ela pressupunha uma migração indeterminada, mas a regra fixa o destino, e o macro-F1 passa a ser exato.

### 2.4 Um-alvo-por-pista: mais um degrau sem GPU

Regra: entre as predições `negation_of` que compartilham a mesma pista `e1` no mesmo documento, manter só a de menor `entity_gap` (desempate pelo `e2.start` menor). Motivada pelo fan-out do gold — **91,9% no treino e 95,2% no teste** têm uma pista para um alvo só.

| Execução | base | + filtro | + um-alvo | Δ do último passo | IC95 | p |
|---|---|---|---|---|---|---|
| BioBERTpt · s42 | 0,6765 | 0,8157 | 0,8387 | +0,0230 | [−0,0028; +0,0498] | 0,085 (ns) |
| BERTimbau · s42 | 0,6937 | 0,8207 | 0,8544 | +0,0337 | [+0,0093; +0,0588] | 0,0074 |
| BioBERTpt · s43 | 0,7542 | 0,8553 | 0,8618 | +0,0065 | [−0,0093; +0,0230] | 0,421 (ns) |
| BERTimbau · s43 | 0,7016 | 0,8073 | 0,8452 | +0,0378 | [+0,0176; +0,0602] | 0,0002 |

O degrau é significativo em 2 de 4, mas nunca prejudica. Com a pilha completa, o pior caso contra o melhor baseline vira **+0,0845, IC95 [+0,0411; +0,1282], p < 0,0001**.

### 2.5 Tetos

| Teto | Valor | O que ele significa |
|---|---|---|
| Precisão-oráculo no recall atual | 0,937 – 0,948 | se todo FP sumisse, sem ganhar recall |
| Cobertura do léxico no teste | 149/152 = 0,9803 | F1 máximo alcançável = **0,9900** |
| Estado atual da pilha | 0,839 – 0,862 | sobram ~0,08 até o oráculo de precisão |

Ou seja: **há folga real acima de 0,86**, e ela está toda em precisão. Não em recall.

---

## 3. O número que falta: a regra pura

Nada disso foi comparado contra o baseline mais óbvio de todos, que é **não usar rede neural nenhuma**. Medido agora:

| Regra (sem modelo) | TP | FP | FN | F1 |
|---|---|---|---|---|
| R1 — todo par com `e1` pista é `negation_of` | 149 | 711 | 3 | 0,2945 |
| R2 — pista + alvo mais próximo (1 por pista) | 90 | 108 | 62 | 0,5143 |
| **R3 — pista + `gap ≤ 1`** | 118 | 75 | 34 | **0,6841** |
| R4 — pista + mais próximo + `gap ≤ 1` | 82 | 69 | 70 | 0,5413 |

**R3 = 0,6841 supera o BioBERTpt · s42 (0,6765) e fica a 0,01 do BERTimbau · s42 (0,6937).** Uma regra de duas linhas empata com dois dos quatro baselines de 11,4 h de GPU.

Isso é uma ameaça e uma oportunidade ao mesmo tempo:

- **Ameaça** se ficar de fora. É a primeira coisa que um examinador atento calcula, e descobri-la depois da defesa é pior do que apresentá-la.
- **Oportunidade** se entrar como terceiro baseline, porque a pilha completa a supera com folga:

| Sistema | F1 | Δ vs R3 | IC95 | p |
|---|---|---|---|---|
| pilha(BioBERTpt · s42) | 0,8387 | +0,1547 | [+0,1072; +0,2047] | < 0,0001 |
| pilha(BERTimbau · s42) | 0,8544 | +0,1703 | [+0,1253; +0,2193] | < 0,0001 |
| pilha(BioBERTpt · s43) | 0,8618 | +0,1778 | [+0,1336; +0,2262] | < 0,0001 |
| pilha(BERTimbau · s43) | 0,8452 | +0,1611 | [+0,1156; +0,2108] | < 0,0001 |

Com esse baseline no texto, a tese fica limpa e defensável: *a pista lexical restringe o espaço; o modelo contextual é o que decide o alvo dentro dele, e essa decisão vale +0,16 de F1 sobre a melhor regra fixa.* É exatamente o enquadramento que o Apêndice B precisa para não se contradizer.

---

## 4. A ameaça real é validade, não ganho

Cinco problemas metodológicos, em ordem de gravidade.

**(a) Tudo na seção 2 e 3 foi calculado no TESTE.** Nenhuma dessas regras foi escolhida no `dev`. A sensibilidade mostra por que isso importa:

| `min_freq` do léxico | \|léxico\| | BioB·42 | BERT·42 | BioB·43 | BERT·43 |
|---|---|---|---|---|---|
| 1 | 51 | 0,8036 | 0,7941 | 0,8471 | 0,7952 |
| 2 | 17 | 0,8157 | 0,8207 | 0,8553 | 0,8073 |
| 3 | 11 | 0,8257 | 0,8221 | 0,8571 | 0,8086 |
| 5 | 9 | 0,8308 | 0,8221 | 0,8571 | 0,8086 |
| 10 | 8 | **0,8333** | **0,8246** | **0,8599** | **0,8111** |

No teste, quanto mais restritivo, melhor — monotonicamente. Mas no `dev` a cobertura cai de 0,9667 (`min_freq` 2) para 0,9133 (`min_freq` 5), então uma escolha honesta no `dev` provavelmente pegaria **um valor pior no teste**. A diferença entre a versão honesta e a otimista é da ordem de 0,02 a 0,03. Escolher no teste queima o argumento inteiro por um ganho que não precisa.

**(b) Não existem predições de `dev` salvas.** `relation_extraction.run` grava só `<out>.preds.json` do teste; do `dev` sobram apenas os agregados em `dev_history`. **Sem isso, nenhuma regra pós-hoc pode ser calibrada.** Este é o bloqueio real da fase 2, e é mais barato de resolver do que o remapeamento de espaço apontado no cronograma.

**(c) O McNemar entre variantes aninhadas é degenerado.** Filtro contra baseline dá `b = 42–67` e `c = 2` sempre, `p` entre 1e−10 e 1e−18. É significativo **por construção**: a regra só muda predições numa direção. Reportar isso como evidência independente do bootstrap seria inflar o resultado. O McNemar continua válido **entre encoders**; entre um sistema e sua própria versão filtrada, ele não informa nada.

**(d) O p-valor do bootstrap em `significance.py` é derivado de percentil, não de permutação.** `p = 2·P(diff ≤ 0)` satura em 0 quando nenhuma reamostra troca de sinal, e não é um teste de hipótese nula próprio. Nos números acima o IC95 exclui zero com folga, então não muda a conclusão. Mas se algum experimento futuro der marginal, o `p` reportado não deve ser o critério — o IC95 deve.

**(e) Multiplicidade.** Cada variante avaliada no teste infla a taxa de falso positivo. Com 4 execuções × N variantes, “achar uma que dá significativo” fica fácil. A disciplina precisa ser declarada antes: **selecionar tudo no `dev`, tocar o teste uma vez por sistema candidato final**, e registrar quantas vezes o teste foi tocado.

---

## 5. Mudanças, da mais barata e menos arriscada à mais cara

| # | Mudança | Custo GPU | Onde entra | Ganho esperado em F1(`negation_of`) |
|---|---|---|---|---|
| N0 | Salvar predições de `dev` | 0 (ou ~5 min de inferência) | `relation_extraction.run` | habilitante, nenhum direto |
| N1 | Filtro léxico pós-hoc | 0 | módulo novo + `significance.py` | **+0,10 a +0,14 (medido)** |
| N2 | Um-alvo-por-pista | 0 | mesmo módulo | +0,01 a +0,04 (medido) |
| N3 | Seleção de época por F1(`negation_of`) no `dev` | 0 | `run`, `build_arg_parser` | 0 a +0,02 |
| N4 | Baseline de regra pura como 3º sistema | 0 | script novo | nenhum — protege o Cap. 6 |
| N5 | Auditoria dos FP residuais | 0 | nenhum | nenhum — define o teto |
| N6 | Salvar probabilidades + calibrar limiar no `dev` | ~5 min/exec. | `predict`, `run` | +0,01 a +0,03 |
| N7 | Pista como *feature* de entrada (`[NEG]`) | ~3 h/semente | `build_marked_window` | incerto, médio |
| N8 | Espaço restrito + modelo dedicado | ~10 min/exec. | `build_dataset` + remapeamento | alto, ainda não medido |
| N9 | Reduzir peso de classe / focal loss | ~3 h/exec. | `run` (loss) | baixo; eixo suspeito |
| N10 | Mineração de negativos difíceis | ~3 h + 1 passada | `build_dataset` | incerto |
| N11 | *Ensemble* entre sementes | ~3 h/semente extra | script novo | +0,00 a +0,02 |
| N12 | Reduzir `max_gap` só para negação | ~3 h/exec. | `--max-gap` (já existe) | provavelmente ≤ 0 |
| N13 | Trocar encoder | ~3 h/semente | identificador | incerto |
| N14 | Decisão conjunta de escopo | alto | reformulação | alto se o gold sustentar |

### N0 — Salvar predições de `dev` *(habilitante, faça primeiro)*

**Onde:** `relation_extraction.run`, no bloco que hoje grava só o sidecar do teste. Também é preciso guardar o `dev` do **melhor epoch**, não do último, o que significa reexecutar `predict` no `dv_loader` depois de `model.load_state_dict(best_state)`.

**Flags novas:** nenhuma obrigatória. Opcionalmente `--save-split-preds {test,dev+test}` (default `dev+test`).

**Como pode piorar:** não piora resultado, mas **muda o hash dos artefatos** e exige reexecutar inferência nos 4 `best_model/` que estão no Drive. Se algum `best_model/` tiver sumido do Drive, esse caminho fecha e a única saída é retreinar. **Confirme que os quatro `best_model/` existem antes de planejar em cima disso.**

**Risco de vazamento:** nenhum, desde que o `dev` continue fora de qualquer decisão de treino além da seleção de época, que já é o caso.

### N1 — Filtro léxico *a posteriori*

**O que é:** léxico de formas de superfície de `e1` induzido **só do `train`**, aplicado sobre predições salvas. Já medido: +0,10 a +0,14, significativo nas quatro execuções, inclusive no pior caso contra o melhor baseline.

**Onde:** um módulo novo, por exemplo `src/negation_lexicon.py`, com `induce_lexicon(train_docs, max_gap, min_freq)` e `apply_cue_filter(cands, y_pred, lexicon)`. **Não entra em `relation_extraction.py`**: é pós-processamento, não treino, e mantê-lo fora preserva a paridade entre encoders por construção. Consumido por um script `scripts/run_reclin.py` que gera um `preds.json` no **mesmo espaço de 19.210**, diretamente pareável pelo `significance.py` atual.

**Flags novas:** `--lexicon-min-freq` (int, default a definir **no dev**), `--cue-source {lexicon,entity_type,both}`, `--demote-to {no_relation,associated_with}`.

**Como pode piorar ou viciar:**
- **Vazamento clássico:** induzir o léxico do corpus inteiro, ou escolher `min_freq` olhando o teste. A tabela de sensibilidade mostra que isso vale 0,02–0,03 de F1 inflado. `min_freq` tem de sair do `dev` (o que depende de N0).
- **Teto de recall duro:** as 3 negações do teste cujo `e1` não é pista viram irrecuperáveis. Precisa ser declarado como limitação, no mesmo lugar onde o teto de `max_gap` já é.
- **Contradição com o Apêndice B**, que argumenta que abordagens léxicas não bastam. O enquadramento tem de ser *restrição do espaço de decisão*, não *regra que decide*. A seção 3 deste documento é a prova empírica disso: a regra sozinha vale 0,684, a pilha vale 0,85.
- **Fragilidade a variação de superfície:** o léxico é casamento exato normalizado. Uma grafia nova no teste (`s\`, `s.`, `nao ha`) derruba silenciosamente para `no_relation`. Vale medir quantos FN o filtro cria por forma não vista antes de fechar.
- **Ruído no léxico induzido:** com `min_freq = 1` entram formas como `osteomielite` e `atb`, que não são pistas e sim erro de anotação de `e1`. Isso não é bug do método, é o que `min_freq` existe para conter.

### N2 — Um-alvo-por-pista

**O que é:** entre predições `negation_of` que compartilham `(doc_id, e1_id)`, manter só a de menor `entity_gap`.

**Onde:** mesmo módulo de N1, aplicado depois do filtro.

**Flags novas:** `--fanout-policy {none,nearest,all}`, `--fanout-max` (int, default 1).

**Como pode piorar:** é a mudança com o pior perfil de risco desta lista apesar de barata. **Ela codifica uma convenção de anotação, não o fenômeno linguístico.** Em `NEGA HF DE GLAUCOMA E CEGUEIRA` o gold liga aos dois; a regra mata um. Os 7 casos de fan-out 2 do teste são perdidos por construção, e o número sobe se o `test` for reamostrado. Além disso ela **degrada o recall** (de 0,88 para 0,86 em algumas execuções) num sistema cujo problema é precisão. Em duas das quatro execuções o ganho é não significativo. Se o objetivo for uma tese sobre negação, e não uma métrica, N2 é a candidata mais forte a ser deixada de fora com justificativa explícita.

### N3 — Seleção de época pelo F1 de `negation_of` no `dev`

**O que é:** `run` hoje escolhe o melhor epoch por `dev_macro_f1` (`if macro > best_f1`). A métrica-alvo é outra.

**Onde:** `relation_extraction.run`, na comparação de melhor epoch, e no `_config_guard` (senão a retomada de checkpoint aceita um critério e treina outro).

**Flags novas:** `--select-metric {macro_f1,negation_f1}`, default `macro_f1` para não quebrar reprodutibilidade dos baselines.

**Como pode piorar:** nas 4 execuções atuais **não mudaria nada** — o `dev_history` mostra que os dois critérios escolhem o mesmo epoch. O ganho só aparece com mais épocas. E há um custo real: selecionar pela classe-alvo com 150 exemplos de `dev` é seleção sobre uma estatística ruidosa, e tende a **piorar o macro-F1** relatado. Se aplicado, tem de ser aplicado aos dois encoders e os baselines precisam ser reportados sob os dois critérios, senão a comparação com a fase 1 deixa de valer.

**Nota separada:** o `dev_macro_f1` ainda subia no epoch 3 em 3 das 4 execuções, com `train_loss` caindo. **Os baselines estão sub-treinados.** Isso é irrelevante enquanto cada época custar 57 min, e vira barato dentro de N8.

### N4 — Baseline de regra pura

**O que é:** R1–R4 da seção 3, como terceiro sistema no Cap. 6.

**Onde:** `scripts/make_rule_baseline.py`, gerando um `preds.json` no espaço completo.

**Flags novas:** `--rule {cue_only,cue_gap,cue_nearest}`, `--rule-max-gap`.

**Como pode piorar:** não piora o resultado; piora a **narrativa** se o limiar `gap ≤ 1` for escolhido no teste. Escolha no `dev`. E há um efeito colateral desejável mas incômodo: a regra pura empata com dois baselines, o que enfraquece a leitura “o encoder clínico não ajuda” ao mostrar que **nenhum dos dois encoders ajuda muito acima de uma regra**. Essa é uma conclusão honesta e publicável, mas o Cap. 7 precisa acomodá-la.

### N5 — Auditoria dos FP residuais

**O que é:** ler os ~40 FP que sobram depois do filtro e classificar cada um como erro do modelo, lacuna de anotação, ou ambiguidade.

**Por que agora:** na execução BERTimbau · s43, dos 43 FP residuais, **16 (37,2%) têm um `e1` que nega outra coisa no gold do mesmo documento**. E os exemplos são eloquentes:

```
e1='SEM'       e2='PRESENÇA DE GRUMOS'      gap=1   gold=no_relation
e1='S/'        e2='RA'                      gap=1   gold=no_relation
e1='evacuação' e2='ausente'                 gap=1   gold=no_relation
e1='AUSENTES'  e2='ELIMINAÇÕES INTESTINAIS' gap=1   gold=no_relation
```

Nenhum clínico leria esses quatro como não-negação. Se essa proporção se confirmar numa amostra com critério fixo, **o teto real não é 0,99, é algo perto de 0,90**, e boa parte do que N7–N14 comprariam já é ruído do gold. Custa uma tarde e reancora tudo o que vem depois.

**Como pode dar errado:** ser lido como desculpa. Enquadre prospectivamente e defina a amostra por critério fixo **antes** de olhar (por exemplo, todos os FP de uma execução escolhida por sorteio), nunca a dedo depois.

### N6 — Salvar probabilidades e calibrar o limiar no `dev`

**O que é:** hoje `predict` devolve `argmax`. Salvar `softmax` permite calibrar `P(negation_of) > τ` com τ escolhido no `dev`, e também combinar modelos por média de probabilidade em vez de voto.

**Onde:** `relation_extraction.predict` (devolver logits), e o payload de `preds.json`. Cuidado: 19.210 × 3 floats por execução é aceitável (~500 KB), mas some no `git` com quatro execuções.

**Flags novas:** `--save-probs`, `--decision {argmax,threshold}`, `--threshold` (ou `--threshold-source {fixed,dev}`).

**Como pode piorar:**
- **É o eixo que já está saturado.** Recall 0,90, precisão 0,55. Subir τ compra precisão às custas de recall, que é a direção certa — mas o filtro de N1 já compra precisão sem custar recall quase nenhum. **Aplicados juntos, os dois competem pelo mesmo ganho**, e o incremento de N6 sobre N1 tende a ser pequeno.
- **τ escolhido no teste é vazamento puro**, e é o erro mais fácil de cometer aqui porque a curva P×R é tentadora de olhar.
- Com 150 positivos no `dev`, o τ ótimo é instável entre sementes. Fixe **um τ por configuração**, não um τ por execução, senão o sistema final não é reprodutível.

### N7 — Pista como *feature* de entrada

**O que é:** marcar no texto de entrada que `e1` é pista, por exemplo `[NEG] [E1] SEM [/E1]`, em vez de deixar o modelo reinferir da superfície.

**Onde:** `build_marked_window` (assinatura ganha o léxico ou um booleano) e `build_dataset`, que precisa passar o flag adiante. `MARKER_TOKENS` ganha `[NEG]`, e `resize_token_embeddings` já cuida do resto. Como está em `relation_extraction.py`, entra **idêntico nos dois encoders** por construção, que é o que a restrição do enunciado exige.

**Flags novas:** `--cue-feature {none,marker,type}`, `--cue-lexicon-path`.

**Como pode piorar:**
- **Pode não superar N1.** O modelo pode aprender a ignorar o marcador, ou aprender a confiar demais nele e predizer `negation_of` para toda pista, que é exatamente a regra R1 (F1 = 0,2945). O experimento tem de ser avaliado contra N1, não contra o baseline da fase 1 — senão você paga 6 h de GPU para reproduzir um ganho que já tinha de graça.
- **Vazamento se o léxico for induzido do `train+dev+test`.** Aqui é mais perigoso que em N1, porque o léxico entra no *treino*: um léxico contaminado ensina o modelo a reconhecer pistas que ele não deveria conhecer.
- **Quebra a comparabilidade com a fase 1**, porque a representação de entrada muda. Os baselines da fase 1 continuam válidos como referência, mas a ablação “com e sem `[NEG]`” precisa das duas pernas treinadas sob o mesmo protocolo.

### N8 — Espaço restrito + modelo dedicado

**O que é:** treinar só nos candidatos cujo `e1` é pista (5,1% do espaço), e recompor com o modelo de 3 classes da fase 1 para o resto.

| Split | Candidatos | Restrito | Positivos | neg:pos | Teto de recall |
|---|---|---|---|---|---|
| train | 152.686 | 8.090 (5,3%) | 0,82% → 15,0% | 121:1 → **5,7:1** | 0,9673 |
| dev | 19.064 | 956 (5,0%) | 0,79% → 15,2% | 126:1 → 5,6:1 | 0,9667 |
| test | 19.210 | 975 (5,1%) | 0,79% → 15,3% | 126:1 → **5,5:1** | 0,9803 |

**Por que é a rota estratégica:** o treino cai de ~3 h para ~10 min. Isso é o que transforma “2 sementes” em “10 sementes”, que é a fraqueza metodológica que o orientador já apontou. E permite finalmente treinar até convergir em vez de parar em 3 épocas.

**Onde:** `build_dataset` ganha um predicado de filtragem; um entry-point novo `src/reclin_negation.py` reutilizando `run`. O remapeamento para o espaço de 19.210 (tudo que o filtro descartou entra como não-`negation_of`) tem de estar no **primeiro commit**, não depois — `significance.py` aborta ao parear espaços distintos, e essa armadilha já cobrou seu preço na migração de `max_gap`.

**Flags novas:** `--restrict-to-cues`, `--cue-source`, `--remap-to-full-space` (default ligado), `--full-space-splits-dir`.

**Como pode piorar:**
- **O modelo restrito vê menos contexto negativo** e pode ficar pior em precisão dentro do próprio espaço restrito do que o modelo completo + filtro. É plausível: com 5,7:1 em vez de 121:1, o `class_weight=balanced` cai de 40,55 para ~2, e o comportamento muda bastante. **N8 pode perder para N1, que é grátis.**
- **`class_weight=balanced` recalculado no espaço restrito é uma mudança implícita de configuração** que não aparece em nenhuma flag. Registre os pesos efetivos no JSON de saída.
- **Vazamento pelo predicado**: se o predicado usar o tipo de entidade `Negation`, ele usa anotação do SemClinBr que estaria disponível no teste, o que é defensável (as entidades já são gold em todo o trabalho), mas tem de ser dito. Se usar léxico, vale a regra de indução só do `train`.
- **O teto de 0,9803 vira teto do sistema**, e três negações do teste ficam irrecuperáveis.

### N9 — Reponderação de perda e *focal loss*

**O que é:** mexer em `loss_fn`.

**Onde:** `run`, no bloco de `class_weight`. Uma `FocalLoss` precisaria de uma classe nova no módulo.

**Flags novas:** `--loss {ce,focal}`, `--focal-gamma`, `--focal-alpha`, `--class-weight {balanced,none,custom}`, `--class-weight-values`.

**Como pode piorar:** **na direção convencional, quase certamente piora.** O peso já está em 115× e o recall em 0,90. Focal loss com γ > 0 amplifica exemplos difíceis, e os difíceis aqui são os `no_relation` que parecem negação — ou seja, γ alto pode **aumentar** os FP. Se este eixo for testado, teste `--class-weight` **reduzido** (por exemplo `sqrt` do balanced) e γ baixo, e trate um resultado nulo como resultado. Uma varredura de 3 pesos × 2 sementes custa ~18 h de T4 para provavelmente confirmar o que a tabela da seção 1.2 já indica. **Não gaste orçamento aqui antes de N8.**

### N10 — Mineração de negativos difíceis

**O que é:** treinar uma segunda vez enfatizando os `no_relation` que o modelo da primeira passada classificou como `negation_of`.

**Onde:** `build_dataset` (peso por exemplo ou reamostragem), mais um sampler no `make_loader`.

**Flags novas:** `--hard-negatives-from` (caminho de um `preds.json` do `dev`), `--hard-negative-weight`.

**Como pode piorar ou vazar:**
- **Vazamento severo se a mineração usar o `test`.** Os negativos difíceis têm de vir do `train` (via *cross-validation* dentro do treino) ou do `dev` — e se vierem do `dev`, o `dev` deixa de ser limpo para seleção de época, o que exige separar um `dev` de mineração.
- **É o mesmo alvo do filtro.** Os negativos difíceis são majoritariamente pares sem pista, que N1 elimina de graça. Depois de N1, sobra pouco para minerar.
- Custo: pelo menos duas passadas de treino, ~6 h por semente.

### N11 — *Ensemble* entre sementes

Medido agora, por voto de maioria sobre o `argmax`:

| Combinação | F1 sem filtro | F1 com filtro |
|---|---|---|
| 2 encoders · s42 (k=2) | 0,7514 | 0,8344 |
| 2 encoders · s43 (k=2) | 0,7784 | 0,8393 |
| BERTimbau · 2 sementes (k=2) | 0,7479 | 0,8202 |
| 4 execuções (k=3) | 0,7727 | **0,8454** |
| 4 execuções (k=4) | 0,7740 | 0,8227 |

**Como pode piorar:** o melhor *ensemble* filtrado (0,8454) **não supera a melhor execução única filtrada com um-alvo** (0,8618). Ou seja, para esta métrica o *ensemble* não paga. Além disso ele **quebra o enquadramento do trabalho**: um sistema que mistura BioBERTpt e BERTimbau não responde mais à pergunta “o pré-treino clínico importa?”. Se usar *ensemble*, use só entre sementes do mesmo encoder — e aí o ganho é ainda menor. Com probabilidades (N6) a média provavelmente bate o voto, mas continua no mesmo patamar.

### N12 — Reduzir `max_gap` só para negação

**Como pode piorar:** `negation_of` tem p95 de gap = 15 e máximo 54; com 25 o teste já preserva **100%** das negações. Reduzir não ganha recall e corta associações. **E o pior:** muda o espaço de candidatos, portanto muda `y_true`, portanto **as predições deixam de ser pareáveis com os baselines** e `significance.py` aborta. Todo o Cap. 6 teria de ser refeito. Recomendação: **não mexer.** Se for mexer, é dentro de N8, onde o espaço já é outro e o remapeamento já existe.

### N13 — Troca de encoder

`biobertpt-clin` em vez de `biobertpt-all`, ou mmBERT-base. **Decidir depois de N8**, não antes: com execuções de 10 min, 10 sementes deixam de ser proibitivas, e a pergunta “clínico ou geral” finalmente ganha IC estreito. Decidir agora, com 2 sementes e amplitude de 0,078, é decidir no ruído. A evidência citada é de NER e **não transfere para RE**.

### N14 — Decisão conjunta de escopo

É o erro real que sobra, e é o único item que não cabe no laço de treino atual. **Antes de investir, N5 tem de responder se o gold sustenta o ganho**: com fan-out 1 em 91,9% do treino e 95,2% do teste, um modelo que aprendesse escopo corretamente seria penalizado toda vez que o anotador parou no primeiro item. Provável trabalho futuro, não fase 2.

---

## 6. Sequência recomendada

O critério que organiza tudo: **estabelecer a validade antes de gastar GPU, e gastar GPU só onde uma execução custa 10 minutos.**

### Etapa 0 — desbloquear (hoje, 0 GPU, ~1 h de código)

Implementar N0. Confirmar antes que os quatro `best_model/` ainda existem no Drive; se não existirem, esta etapa vira retreino e o cronograma muda. Produto: `results/baseline_*.dev_preds.json`.

### Etapa 1 — fechar o resultado que já existe (0 GPU, ~1 dia)

1. Implementar N1 + N4 como módulo e script.
2. **Escolher `min_freq` e `--rule-max-gap` no `dev`**, registrar a escolha e não voltar atrás.
3. Rodar N5 (auditoria) sobre uma execução escolhida por sorteio, com critério fixo.
4. Tocar o teste **uma vez**, gerando `results/reclin_lexfilter_<encoder>_seed{42,43}.json` + sidecar, e rodar `significance.py` contra os quatro baselines.

**Critério de parada:** se o IC95 excluir zero contra os quatro baselines (o que a seção 2.2 diz que vai acontecer), **a meta da fase 2 está formalmente atingida** e o Cap. 8 pode ser reescrito. A partir daqui, tudo o mais é melhoria opcional, não risco de prazo.

Decida N2 aqui, com a auditoria na mão. Se a auditoria mostrar que boa parte dos FP de fan-out é lacuna de anotação, N2 é um truque de métrica e é mais defensável deixá-lo fora, documentando a decisão.

### Etapa 2 — o RECLin-PT de verdade (≈ 4 h de GPU total, ~1 semana)

Implementar N8 com remapeamento desde o primeiro commit. Depois:

1. **Sanidade (2 execuções, ~20 min):** 1 encoder × 2 sementes no espaço restrito, com `--epochs` maior (6 a 10, já que cada época custa minutos). Comparar contra N1 **no `dev`**.
   **Parada:** se o restrito não superar N1 no `dev`, **pare aqui** e reporte N1 como o RECLin-PT, com o restrito como ablação negativa. Isso é um resultado publicável e custa 20 minutos descobrir.
2. **Escala (20 execuções, ~3,5 h):** 2 encoders × 10 sementes. Seleção de época e hiperparâmetros no `dev`.
3. **Teste, uma vez:** as 20 predições remapeadas, `significance.py` contra os quatro baselines e contra N1, mais a agregação por semente com IC da diferença — que é a continuidade já prometida no §8 e a resposta direta à pergunta do orientador sobre duas sementes.

**Critério de parada:** IC95 da diferença média entre sementes excluindo zero contra os dois baselines. Com 10 sementes por lado, isso é uma afirmação muito mais forte do que qualquer coisa na fase 1.

### Etapa 3 — só se sobrar tempo e o `dev` indicar (≈ 6 h)

Nesta ordem, e cada uma só entra se a anterior **não** tiver batido no teto da auditoria: N6 (limiar) → N7 (`[NEG]` como *feature*, avaliado contra N1, não contra a fase 1) → N13 (encoder, agora com 10 sementes por lado).

**Não entram** nesta fase: N9, N10, N11, N12, N14. Os quatro primeiros por eixo errado ou redundância medida; N14 por depender de um gold que provavelmente não o sustenta.

### Orçamento total

| Etapa | GPU | Calendário |
|---|---|---|
| 0 — desbloquear | 0 (ou ~20 min de inferência) | 1 dia |
| 1 — filtro + auditoria + regra pura | **0** | 2 a 3 dias |
| 2 — espaço restrito, 2 × 10 sementes | ~4 h | 1 semana |
| 3 — opcional | ~6 h | conforme sobra |

Contra as 11,4 h que a fase 1 consumiu para produzir um resultado ambíguo, a fase 2 fecha com **~4 h e 10 sementes por lado** — e a etapa 1, que sozinha já bate a meta declarada, não usa GPU nenhuma.

---

## 7. Decisões que dependem de você

1. **N2 entra?** É a única mudança da lista que otimiza contra a convenção de anotação e não contra o fenômeno. Vale +0,01 a +0,04 e custa uma limitação a mais para defender.
2. **A regra pura (N4) entra no Cap. 6?** Recomendação: sim, e explicitamente. Ela empata com dois baselines, o que é desconfortável, mas é o que dá sentido à contribuição do RECLin-PT.
3. **Os quatro `best_model/` ainda estão no Drive?** Toda a etapa 0 depende disso.
4. **O `artigo-sbc/` acompanha ou congela?** Pendência herdada da fase 1: a prosa ainda está em `max_gap=20` e qualquer alvo do Makefile produz um PDF internamente contraditório que compila sem erro.

---

**Como os números foram obtidos.** Sobre `results/*.preds.json` e `data/splits/*.jsonl`, com os candidatos gerados por `src/candidates.iter_candidate_pairs` (`max_gap=25`) importado do próprio repositório. O alinhamento entre o `y_true` reconstruído e o salvo nos sidecars foi verificado: zero divergências nas quatro execuções (19.210 exemplos; 152 `negation_of`, 996 `associated_with`, 18.062 `no_relation`). Bootstrap pareado com 10.000 reamostragens, semente 42, replicando `src/significance.py`. **Nada foi escrito no repositório e nenhum treino foi executado.**
