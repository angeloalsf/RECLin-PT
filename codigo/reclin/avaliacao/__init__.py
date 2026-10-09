"""Avaliação: métricas, significância e agregação entre sementes.

Lê predições (sidecars) e execuções; nunca produz predições nem importa
estratégias.

* `metricas` — P/R/F1 por classe, macro, micro e weighted-F1, MCC, relatório
  por classe e matriz de confusão: a única implementação do pacote.
* `significancia` — McNemar exato e bootstrap pareado no F1 da classe-alvo,
  com o relatório no formato das comparações do legado.
* `protocolo` — as 26 comparações do TCC e a regra de semente do bootstrap.
* `agregacao` — média e desvio-padrão populacional entre sementes.
"""
