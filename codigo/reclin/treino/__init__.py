"""Infraestrutura de treino: comum a todas as estratégias treinadas.

* `reprodutibilidade` — variáveis de ambiente, sementes, estados dos geradores
  aleatórios, registro do ambiente e dos avisos;
* `checkpoint` — estado completo de retomada e o melhor modelo, gravados de
  forma atômica, com a identidade da execução que os produziu;
* `laco` — o laço de treino com avaliação no DEV a cada época, seleção da
  melhor época, checkpoints (fim de época e, opcionalmente, a cada N passos) e
  retomada exata;
* `classificador` — o treino do classificador da tarefa (todos os candidatos,
  cabeça [CLS]) gravado no diretório da execução, e a avaliação do TEST como
  operação separada.

O que é de uma estratégia (quais exemplos, como pontuar o DEV, qual cabeça)
entra no laço como parâmetro; o laço não conhece estratégias.

Depende do núcleo (`tarefa`, `entrada`, `modelos`, `execucao`, `avaliacao`,
`config`) e de torch e transformers.
"""
