"""RECLin-PT: extração de relações em notas clínicas em português.

O pacote tem três níveis, e um nível só importa os de cima:

* núcleo — a tarefa e a infraestrutura para resolvê-la e avaliá-la
  (`tarefa`, `corpus`, `particoes`, `entrada`, `modelos`, `treino/`,
  `execucao`, `avaliacao/`, `config`, `util/`);
* especialização — o conhecimento sobre como a negação aparece no corpus
  (`negacao/`);
* estratégias — formas experimentais de combinar os dois para prever
  (`estrategias/`), comparadas pelas execuções que gravam. Só uma compõe o
  modelo entregue, e só `modelo_final.py` a nomeia.

Nenhum módulo do pacote configura logging, altera `os.environ` ou mexe em
`sys.path` ao ser importado. Isso é feito só pelos scripts em `codigo/scripts/`.
"""

__version__ = "0.1.0"
