"""Apresentação dos resultados: tabelas, curvas, figuras, relatório Markdown,
registro de valores e conferência do texto do TCC.

Nível do núcleo (não importa `negacao` nem `estrategias`). Os relatórios só
leem execuções verificadas pelo inventário (`execucao.inventario`) e as
partições congeladas; cada número publicado leva a proveniência das execuções
de onde veio. Uma tabela cujos sistemas ainda não foram executados é gerada
com as células marcadas como pendentes — nunca com valores de outra origem.

* `formatacao` — LaTeX (ABNT: vírgula decimal), Markdown e CSV;
* `fase` — os sistemas e as comparações da fase atual (protocolo
  pré-registrado, `docs/protocolo/fase-biobertpt-seed42.md`);
* `tabelas` — o modelo de tabela, as tabelas de resultados, de comparações,
  de limiares, de proveniência e das partições, e o registro de valores;
* `curvas` — o histórico por época das execuções treinadas;
* `eda` — as tabelas das análises do corpus;
* `figuras` — figuras (matplotlib, importado só quando usado);
* `markdown` — o relatório da fase em Markdown;
* `conferencia` — a conferência do texto do TCC: caminhos antigos e
  afirmações numéricas contra o registro de valores.
"""
