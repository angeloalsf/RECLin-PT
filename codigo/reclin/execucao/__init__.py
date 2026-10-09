"""Execução: o formato em disco que liga as estratégias à avaliação.

Uma execução é o resultado de uma estratégia com uma configuração e uma
semente: as predições do DEV e do TEST sobre o conjunto de referência, a
configuração usada, as métricas e a trilha das avaliações do TEST. É o único
formato que toda etapa produz ou consome: as estratégias gravam execuções; a
avaliação, as comparações e os relatórios só leem execuções.

* `predicoes` — o sidecar de predições (o formato do legado: `model`, `seed`,
  `labels`, `y_true`, `y_pred` e, opcionalmente, `probs` e campos extras):
  montar, ler, validar, gravar e conferir contra o conjunto de referência.
* `diretorio` — o diretório de uma execução: registrar, ler e localizar
  predições, inclusive nos resultados do legado.
* `trilha` — a trilha das avaliações do TEST (`test_evals`), que conta
  quantas vezes cada configuração teve o TEST medido.

Não calcula métricas (isso é de `avaliacao`) e não treina.
"""
