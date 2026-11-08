# Integridade das partições: quase duplicatas, semente e consistência das anotações

Data: 01/11/2026.

**Nada muda:** a divisão continua 800/100/100, por documento, estratificada pela presença de `negation_of`, com a semente 42 (`reclin/particoes.py`). As partições, as anotações e as predições não foram alteradas.

**Números**
- Comando: `python codigo/diagnosticos/diagnosticar.py duplicatas | semente`.
- Saídas: `codigo/resultados/diagnosticos/{quase_duplicatas,sensibilidade_semente}.json`.
- Testes: `testes/unidade/test_diagnosticos.py`.

## 1. Documentos quase duplicados

**Detector** (`reclin/analises/quase_duplicatas.py`):
- Normaliza o texto: minúsculas, sem nenhum espaço em branco.
- Compara os textos pelo Jaccard dos *shingles* de 8 caracteres, com limiar 0,8, nas três partições.
- **Só registra:** não remove nem move documentos e não grava texto.

**Resultado:** 7 pares. Todos ficam **idênticos sem os espaços**, e nenhum é idêntico byte a byte.

| Par | Partições | Diferença no texto original | Entidades (a / b / em comum) | Relações (a / b / em comum) | `negation_of` (a / b / em comum) |
| --- | --- | --- | --- | --- | --- |
| 9280 / 9271 | TRAIN / DEV | 4 espaços | 1 / 1 / 1 | 0 / 0 / 0 | — |
| 9531 / 9632 | TRAIN / TRAIN | 13 espaços | 40 / 43 / 21 | 13 / 17 / 8 | 1 / 1 / 1 |
| 9532 / 9633 | TRAIN / TRAIN | 4 espaços | 34 / 33 / 20 | 8 / 7 / 6 | — |
| 9533 / 9634 | TRAIN / TRAIN | 4 espaços | 49 / 55 / 23 | 18 / 26 / 10 | 1 / 1 / 1 |
| **9635 / 9534** | **TRAIN / TEST** | 3 espaços | 59 / 58 / 34 | 20 / 19 / 11 | 2 / 2 / 2 |
| **9636 / 9535** | **TRAIN / TEST** | 8 espaços | 58 / 60 / 39 | 24 / 27 / 16 | 2 / 2 / 2 |
| **9637 / 9536** | **TRAIN / TEST** | 4 espaços | 46 / 47 / 30 | 22 / 22 / 16 | 1 / 1 / 1 |

"Em comum" compara texto da entidade (sem caixa) e tipo, e, nas relações, os textos dos dois argumentos e o tipo. A comparação é sem offsets, porque os espaços deslocam as posições.

**Leitura**
- Seis notas do corpus aparecem duas vezes, com ids separados por 101 (9531–9536 e 9632–9637), **anotadas de forma diferente**.
- Concordância entre as duas anotações, somando os seis pares:

| | Concordância (F1) |
| --- | --- |
| Entidades | 0,57 |
| `associated_with` | 0,57 |
| `negation_of` | 1,00 (7 de 7) |

- O par 9280/9271 tem 19–23 caracteres, uma entidade e nenhuma relação ou candidato.
- **No TEST**, as notas 9534, 9535 e 9536 têm cópia no TRAIN. Elas somam 5 `negation_of`, 63 `associated_with` e 1.118 dos 19.210 pares candidatos. As 5 negações são iguais às das cópias do TRAIN.
- **No DEV**, o único par é o trivial 9271/9280.

## 2. Análise de sensibilidade preparada (sem 9534–9536)

**Preparada, não executada no TEST.**
- `reclin/analises/exclusao_documentos.py` recalcula as métricas de uma execução **a partir das predições já gravadas**, sem os candidatos dos documentos excluídos (por padrão, `DOCUMENTOS_TEST_COM_COPIA_NO_TREINO = ("9534", "9535", "9536")`).
- Não roda modelo, não muda partição nem predições.
- O comando `diagnosticar.py excluir-documentos --particao test` recusa o TEST sem `--test-autorizado`. A função recusa sem `permitir_test=True`.

**Uso pré-registrado:** emenda E2 do protocolo (`docs/protocolo/fase-biobertpt-seed42.md`, seção 8).
- A análise principal continua sendo a do TEST completo.
- A análise sem os três documentos é **secundária**, relatada ao lado da principal, depois da avaliação autorizada do TEST.

**Limite do efeito sobre a negação:** 5 de 152 relações, ou seja, no máximo 3,3 pontos de recall. Em `associated_with`, o efeito pode ir nos dois sentidos: das 63 relações das três notas, só 38 coincidem com as das cópias do TRAIN.

## 3. Sensibilidade à semente

O algoritmo de `particoes.particionar`, com as sementes 0 a 199 (a 42 incluída). A semente 42 reproduz as partições congeladas documento a documento.

| Contagem (relações) | Partição | mín | p5 | mediana | p95 | máx | semente 42 | percentil da 42 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `negation_of` | DEV | 127 | 137 | 157 | 186 | 201 | 155 | 47% |
| `negation_of` | TEST | 118 | 138 | 159 | 191 | 202 | 152 | 28% |
| `associated_with` | DEV | 765 | 841 | 979 | 1.117 | 1.211 | **832** | **4,5%** |
| `associated_with` | TEST | 812 | 875 | 984 | 1.103 | 1.153 | **1.098** | **94%** |
| `negation_of` | TRAIN | 1.236 | 1.254 | 1.286 | 1.315 | 1.341 | 1.299 | 78% |

**Leitura**
- A estratificação binária fixa os documentos com negação (67 no DEV e 67 no TEST, em qualquer semente), mas não o número de relações.
- Para `negation_of`, a semente 42 é típica.
- Para `associated_with`, ela deixou o DEV pobre e o TEST rico em associações. Por isso, o macro-F1 e o F1 de `associated_with` do DEV não antecipam bem os do TEST.
- **80/10/10:** não há justificativa registrada no código nem nos documentos do projeto. É uma convenção (constantes `PROPORCAO_DEV` e `PROPORCAO_TEST`). Mantida.

## 4. Consistência das anotações (TRAIN e DEV)

Contagens recalculadas das partições v2. Descrevem a anotação; não dizem se o modelo ou o anotador errou.

| | TRAIN | DEV |
| --- | ---: | ---: |
| `negation_of` com o e1 sem o tipo `Negation` e o e2 com (direção invertida) | 38 | 5 |
| `negation_of` sem nenhuma entidade `Negation` / com as duas | 4 / 1 | 0 / 0 |
| Mesmo par com `negation_of` nas duas direções | 2 | 0 |
| Mesmo par com `negation_of` numa direção e `associated_with` na outra | 1 | 0 |
| Mesmo par com `associated_with` nas duas direções | 47 | 2 |
| Relações repetidas (mesmo par, mesmo tipo; colapsam num candidato) | 37 | 3 |
| Pares com rótulos conflitantes na mesma direção | 0 | 0 |
| Entidades `Negation` que não são argumento de nenhuma `negation_of` | 684 / 1.841 | 89 / 230 |
| Relações cujos argumentos têm spans sobrepostos ou idênticos | 0 | 0 |

**Consequências para `no_relation`**
- Os pares não anotados viram `no_relation`.
- Quando uma pista sem relação anotada está colada a uma entidade (distância ≤ 1), o par pode ser uma omissão da anotação ou um `no_relation` correto. Os dados sozinhos não decidem.
- Há 162 pares assim no TRAIN e 19 no DEV, com a entidade logo depois da pista.
- A adjudicação cega já preparada (`docs/analises/adjudicacao_negation_of.md`) é o instrumento para medir isso.
