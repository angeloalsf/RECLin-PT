"""Corpus SemClinBr: dos XML originais ao dataset versionado.

A conversão é feita em três passos, cada um num módulo, para que cada
transformação possa ser conferida em separado:

1. `fonte` — inventário e proveniência da pasta de XML: lista dos arquivos,
   identificadores, SHA-256 de cada arquivo e do conjunto, codificação,
   quebras de linha, erros de leitura e anomalias estruturais. Só lê.
2. `semclinbr` — leitura de um XML numa **representação intermediária**
   (`DocumentoFonte`): o `<TEXT>` exatamente como está nos bytes (com os
   `\\r\\n`), o texto como um parser XML o entrega, e as anotações e relações
   com todos os atributos e a ordem da fonte. Nenhum offset é alterado.
3. `conversao` — da representação intermediária ao registro do dataset
   (`tarefa.Documento`), com a regra de texto da versão de dados (C1, ver o
   módulo) e o registro, por documento, do que foi transformado e por quê, e
   das anotações defeituosas da própria fonte (mantidas como estão).

`auditoria` compara cada entidade de um dataset (de qualquer versão) com a
anotação de origem correspondente, pela identidade (documento e `id`), nunca
por busca de texto; `comparacao` confere os invariantes entre duas versões dos
dados (documentos por partição, entidades, relações, candidatos, `y_true`).

Os XML são entrada somente leitura: nenhum módulo daqui grava na pasta deles.
Os arquivos com o texto do corpus (dataset e representação intermediária)
ficam em `dados/processados/`, fora do controle de versão (licença).
"""
