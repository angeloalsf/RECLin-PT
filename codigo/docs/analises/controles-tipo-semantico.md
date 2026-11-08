# Controles informados pelas anotações: tipo semântico `Negation`

Data: 01/11/2026. Natureza: **controles diagnósticos informados pelas anotações**. Não são configurações do RECLin-PT nem entram nas comparações do protocolo.

As quatro configurações não recebem o tipo semântico:

| Configuração | Papel |
| --- | --- |
| Baseline | referência experimental |
| Pair-Aware linear | arquitetura oficial |
| Two-Stage | estratégia experimental |
| Pair-Aware MLP | variante experimental |

Os marcadores `[E1]`/`[E2]` são sem tipo (`reclin/entrada.py`), e nenhum módulo de treino lê `entity["type"]`.

**Tipos semânticos nas partições.** São preservados como estão: o campo `type` de cada entidade é a cópia do atributo `tag` do XML do SemClinBr. Esse atributo traz tipos UMLS e os rótulos do corpus (`Negation`, `Abbreviation`), unidos por `|`. Nada foi alterado.

**Código e comando**
- Código: `reclin/analises/controles_anotacao.py`.
- Comando: `python codigo/diagnosticos/diagnosticar.py tipos`.
- Saída: `codigo/resultados/diagnosticos/controles_tipo_semantico.json`.
- Testes: `testes/unidade/test_diagnosticos.py` (exemplos sintéticos e os números abaixo com os dados reais).

**O que os controles usam**
- Dados reais da fase: `dados/particoes-v2/`, conferidos contra o MANIFEST `18264860…`.
- Os tipos originais.
- Os candidatos da tarefa (`max_gap = 25`).
- Só TRAIN e DEV. O TEST é recusado pelo código.

**Correspondência com os controles pedidos para a Etapa 2**

| Controle | Seção |
| --- | --- |
| Tipo `Negation` → identificação da pista de negação | 1a |
| Tipo `Negation` + posição → entidade seguinte como possível alvo | 2 |
| Análise de relações usando os tipos anotados (pares candidatos previstos como `negation_of` pelo tipo dos argumentos) | 1b; a consistência entre tipos e relações anotadas está em `docs/analises/integridade-particoes.md`, seção 4 |

## Definições

| Controle | Previsão | Verdade |
| --- | --- | --- |
| **1a. Tipo → pista** (nível da entidade) | a entidade é pista se o seu tipo inclui `Negation` | entidades que são e1 de ao menos uma `negation_of` anotada (convenção da tarefa: e1 nega e2) |
| **1b. Tipo → `negation_of`** (nível do par) | `negation_of` para todo candidato com e1 `Negation` e e2 sem `Negation` | rótulo anotado do candidato |
| **2. Tipo + posição** | para cada entidade `Negation`, o alvo é a primeira entidade, na ordem (start, end, id), que começa no fim da pista ou depois e não tem `Negation`. Prevê `negation_of` para (pista, alvo) se o par for candidato (distância ≤ 25); o resto é `no_relation` | rótulo anotado do candidato |

**Fórmulas:** as de `avaliacao.metricas.resumo_alvo`, F1 = 2·TP / (2·TP + FP + FN). No controle 2, o recall também é dado sobre as relações anotadas: as que a janela não alcança contam como falso negativo.

## Resultados

| Controle | Partição | TP | FP | FN | Precisão | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a. Tipo → pista (entidades) | TRAIN (36.455 entidades) | 1.125 | 716 | 42 | 0,611 | 0,964 | 0,748 |
| | DEV (4.513 entidades) | 136 | 94 | 5 | 0,591 | 0,965 | 0,733 |
| 1b. Tipo → `negation_of` (pares) | TRAIN (152.686 pares) | 1.213 | 6.191 | 42 | 0,164 | 0,967 | 0,280 |
| | DEV (19.064 pares) | 145 | 751 | 5 | 0,162 | 0,967 | 0,277 |
| **2. Tipo + posição** | TRAIN | 1.089 | 684 | 166 | 0,614 | 0,868 | **0,719** |
| | DEV | 130 | 92 | 20 | 0,586 | 0,867 | **0,699** |

**Contagens da regra 2**

| | Pistas `Negation` | Pares previstos | Alvo fora da janela | Recall sobre as `negation_of` anotadas |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 1.841 | 1.773 | 55 (13 pistas sem entidade seguinte) | 1.089 / 1.299 = 0,838 |
| DEV | 230 | 222 | 8 | 130 / 155 = 0,839 |

O controle 2 no DEV reproduz exatamente o valor da auditoria v2: F1 0,699, P 0,586, R 0,867.

## Leitura

1. **O tipo `Negation` quase revela as pistas.** Cobre 96% das pistas anotadas. As 42 (TRAIN) e 5 (DEV) pistas sem esse tipo correspondem às relações cujo e1 não tem o tipo `Negation`:
     - no TRAIN, 38 invertidas (o e1 anotado é o conceito) e 4 sem nenhuma entidade `Negation`;
     - no DEV, 5 invertidas.
     
     Ver `docs/analises/integridade-particoes.md`.
   - A precisão é de cerca de 0,6 porque cerca de 37% das entidades `Negation` não são e1 de nenhuma `negation_of` anotada.
   - Isso pode ser omissão da anotação, negação de um conceito que não virou entidade ou uso sem valor de negação. Os controles não distinguem esses casos.
2. **No nível do par, o tipo sozinho dá recall de 0,967 e precisão de 0,16.** Saber qual é a pista não resolve a tarefa: a dificuldade está em **escolher o alvo**.
3. **Tipo + posição chega a F1 de cerca de 0,70–0,72 sem nenhum aprendizado**, usando um atributo que vem da própria anotação. Se o tipo entrasse como atributo de entrada, um modelo receberia de graça a identificação da pista. Esse é o motivo para **não** usá-lo na Pair-Aware linear oficial e nas demais configurações.
4. Os números **não** são comparáveis a resultados de modelos como se fossem um sistema concorrente. Servem só como diagnóstico do atalho.

## Sugestões para trabalhos futuros (não implementadas)

Registradas só como possibilidades. Nada disto foi implementado, e nenhuma delas muda a Pair-Aware linear oficial.

- **Ablação com tipos mascarados.** Uma variante experimental separada com marcadores tipados por grupos amplos, com o tipo `Negation` mascarado. Mediria se o tipo do alvo ajuda sem entregar a pista. Exige protocolo próprio, treino e avaliação só no DEV.
- **Distribuição dos tipos dos argumentos** de `negation_of` e `associated_with` no TRAIN, como análise descritiva do corpus.
- **Tipo + posição com coordenação.** Estender o controle 2 para enumerações, como na forma "sem A, B ou C", para medir quanto do erro restante é estrutural.

## Limitações

- A verdade da pista segue a convenção da tarefa (e1). As negações invertidas contam como pistas não encontradas, e a entidade `Negation` correspondente conta como falso positivo.
- "Entidade seguinte" é uma heurística posicional simples. Ela não trata coordenação ("sem A, B ou C") nem pistas depois do conceito.
- Nenhum resultado no TEST. Se um dia for necessário, exige autorização explícita e é descritivo, nunca critério de escolha.
