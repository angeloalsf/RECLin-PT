# A janela de candidatos `max_gap = 25`

Data: 01/11/2026.

**O valor fica como está:** `MAX_GAP = 25` em `reclin/tarefa.py`, o mesmo das partições e dos conjuntos de referência congelados. Este documento corrige e completa a descrição da janela. Não muda a tarefa.

**Números**
- Comando: `python codigo/diagnosticos/diagnosticar.py max-gap`.
- Saída: `codigo/resultados/diagnosticos/sensibilidade_max_gap.json`.
- Partições v2: TRAIN e DEV, exceto no bloco que reproduz o "11,6%".

## 1. Definição: distância simétrica entre os spans

`entity_gap(e1, e2)` (`reclin/tarefa.py`) é o número de caracteres **entre o fim do primeiro span no texto e o início do segundo**, qualquer que seja o e1. Spans que se tocam ou se sobrepõem têm distância 0.

A distância é simétrica: o par (e1, e2) e o par (e2, e1) têm a mesma distância e ambos viram candidatos, com rótulos próprios. O par inverso de uma `negation_of` anotada é `no_relation`.

Um par é candidato quando:
- `entity_gap <= 25`;
- os spans não são idênticos;
- as entidades são diferentes.

**Correção.** A descrição "do fim de e1 ao início de e2" é imprecisa: vale só quando e1 vem antes de e2 no texto.

## 2. Distância nas relações anotadas

| Tipo | Partição | n | mediana | p90 | p95 | p99 | máx | ≤ 1 | > 25 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `negation_of` | TRAIN | 1.299 | 1 | 12 | 15 | 32 | 54 | 1.010 | 27 |
| | DEV | 155 | 1 | 11 | 21 | 31 | 31 | 117 | 3 |
| `associated_with` | TRAIN | 7.922 | 4 | 26 | 47 | 165 | 541 | 2.740 | 806 |
| | DEV | 832 | 4 | 22 | 35 | 74 | 334 | 326 | 68 |

Na negação, 78% (TRAIN) e 75% (DEV) das relações têm distância ≤ 1: a pista vem colada ao conceito.

## 3. Cobertura de cada classe e custo da ampliação

Cobertura = candidatos com o rótulo da classe ÷ relações anotadas da classe. As relações repetidas no mesmo par contam uma vez como candidato, então o teto da negação é menor que 1: 153 pares distintos para 155 relações no DEV.

| `max_gap` | Pares TRAIN | Pares DEV | × (25 = 1) | Cob. `negation_of` TRAIN | Cob. `negation_of` DEV | Cob. `associated_with` TRAIN | Cob. `associated_with` DEV | `no_relation` por relação (DEV) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 77.784 | 9.634 | 0,51 | 0,872 | 0,852 | 0,747 | 0,781 | 11,3 |
| 20 | 128.380 | 15.994 | 0,84 | 0,954 | 0,942 | 0,865 | 0,887 | 17,1 |
| **25** | **152.686** | **19.064** | **1,00** | **0,966** | **0,968** | **0,896** | **0,917** | **19,9** |
| 30 | 176.808 | 22.148 | 1,16 | 0,975 | 0,974 | 0,916 | 0,939 | 22,8 |
| 40 | 224.042 | 27.926 | 1,47 | 0,985 | 0,987 | 0,941 | 0,963 | 28,3 |
| 50 | 269.338 | 33.506 | 1,76 | 0,986 | 0,987 | 0,953 | 0,975 | 33,8 |
| 100 | 482.456 | 59.456 | 3,16 | 0,987 | 0,987 | 0,979 | 0,992 | 59,8 |

**Leitura**

| Classe | Efeito de ampliar a janela |
| --- | --- |
| `negation_of` | Com 25, a janela cobre 96,6% (TRAIN) e 96,8% (DEV). Com 40, já cobre os 153 pares distintos do DEV. Com 50, ganha +3 negações no DEV (+0,019 de recall no máximo) ao custo de ×1,76 pares |
| `associated_with` | Com 25, perde cerca de 10% das relações. Com 50, perde cerca de 3–5%. É a classe que paga o custo da janela |

O custo é linear no número de pares: treino e inferência proporcionais, e mais `no_relation` por relação (de 19,9 para 33,8 no DEV com 50). Nada disto foi treinado: os números descrevem só o espaço de candidatos.

**Cobertura não é recall.** O recall reportado das execuções é calculado sobre os candidatos. Sobre as relações anotadas, ele é multiplicado pela cobertura. No DEV, com a Pair-Aware linear e o BioBERTpt-all (auditoria v2):

| Medida | Valor |
| --- | --- |
| Recall sobre os candidatos | 141/150 = 0,940 |
| Recall sobre as relações anotadas | 141/155 = 0,910 |

## 4. Origem do "11,6%" citado no código

O comentário de `reclin/tarefa.py` (l. 53–55) cita 11,6% das relações anotadas a mais de 20 caracteres, "quase todas `associated_with`".

**Reproduzido** (`origem_11_6` na saída; teste `test_origem_do_11_6_por_cento`):

| Medida | Valor |
| --- | --- |
| Relações anotadas no corpus inteiro (1.000 documentos: TRAIN, DEV **e** TEST) | 11.458 |
| Relações com distância simétrica acima de 20 | **1.324** = 11,56% ≈ **11,6%** |
| … `associated_with` | 1.272 (96%) |
| … `negation_of` | 52 |

As mesmas contagens fora do corpus inteiro **não** dão 11,6%:

| Conjunto | Fração acima de 20 |
| --- | ---: |
| TRAIN | 11,82% |
| TRAIN + DEV | 11,67% |
| DEV | 10,23% |
| TEST | 10,64% |

Também testei outras definições de distância (direcional, entre os inícios) e de contagem (pares distintos, até 75 caracteres). Só a distância simétrica sobre o corpus inteiro, por relação ou por par distinto (11,56% e 11,59%), arredonda para 11,6%.

**Conjunto de dados usado:** o número citado no código é uma contagem de cobertura das anotações feita com o **corpus completo, TEST incluído**, e é registrado como tal (protocolo, emenda E2). É uma contagem das anotações, não um resultado de modelo. O script que produziu o número citado não faz parte do projeto atual; o cálculo acima o reproduz e está versionado (`max_gap.distancias_anotadas`, bloco `origem_11_6`).

Com 25, no corpus inteiro ficam fora 1.004 relações (8,8%), 974 delas `associated_with`.

O comentário de `reclin/tarefa.py` **não foi editado**: o arquivo entra na identidade do código das execuções (`0526d9…`). A correção fica neste documento e no `README.md`. Mudar o comentário depende da sua decisão.
