"""Infraestrutura de treino: comum a todas as estratégias treinadas.

* `reprodutibilidade` — variáveis de ambiente, sementes, estados dos geradores
  aleatórios, registro do ambiente e dos avisos;
* `checkpoint` — estado completo de retomada e o melhor modelo, gravados de
  forma atômica, com a identidade da execução que os produziu;
* `laco` — o laço de treino com avaliação no DEV a cada época, seleção da
  melhor época, checkpoints (fim de época e, opcionalmente, a cada N passos) e
  retomada exata;
* `montagem` — o que uma estratégia treinada informa ao orquestrador (quais
  candidatos, qual modelo, o que vai para os sidecars);
* `classificador` — o orquestrador: o treino de uma montagem (por padrão, o
  classificador da tarefa: todos os candidatos, cabeça [CLS]) gravado no
  diretório da execução, e a avaliação do TEST como operação separada.

O que é de uma estratégia entra como parâmetro (`Montagem`, `pontuar_dev`);
o laço e o orquestrador não importam `reclin.estrategias`.

Depende do núcleo (`tarefa`, `entrada`, `modelos`, `execucao`, `avaliacao`,
`config`) e de torch e transformers.
"""
