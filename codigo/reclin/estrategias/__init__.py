"""Estratégias experimentais para detectar `negation_of`.

Cada estratégia combina o núcleo (`tarefa`, `execucao`, `avaliacao`) e a
especialização (`negacao`) para prever sobre o conjunto completo de candidatos
de uma partição, e grava execuções no formato de `reclin.execucao`, que a
avaliação compara entre si. Uma estratégia nunca importa outra: quando um
procedimento histórico envolve duas (a calibração do filtro e da regra pura,
feita de uma vez no DEV), quem as combina é o script.

Implementadas:

* `filtro_pistas` — pós-processamento das predições de outra execução:
  rebaixa para `no_relation` todo `negation_of` cujo e1 não é pista do léxico.
  Inclui a calibração de `min_freq` e da porta de gap no DEV.
* `regra_pura` — sem modelo: marca `negation_of` pelas pistas do léxico e pela
  distância até o alvo (regras R1 a R4). Inclui a calibração da regra no DEV.

Treinadas (etapa 6), sobre a infraestrutura de `reclin.treino`, cada uma
descrita por uma `treino.montagem.Montagem`:

* `baseline` — o classificador da tarefa: todos os candidatos, cabeça [CLS];
* `restrito` — só os pares cujo e1 é pista do léxico congelado; época
  escolhida no DEV remapeado ao conjunto completo;
* `pair_aware` (pacote, com `cabeca`) — o restrito com a cabeça
  `[h_cls ; h_[E1] ; h_[E2]] → MLP`.

Prevista para o futuro: `two_stage`. Uma estratégia é um módulo ou um pacote
e só importa o núcleo e `negacao/`.
"""
