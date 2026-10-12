"""Infraestrutura compartilhada dos notebooks de experimento (Google Colab).

Orquestra o que os notebooks de estratégia têm em comum, sem reimplementar
nenhum algoritmo científico (treino, avaliação e composição continuam nos
módulos e scripts do projeto):

* `codigo_colab` — obtém o código por tarball de uma revisão exata (SHA de 40
  caracteres) do GitHub, sem Git; só biblioteca padrão, para poder ser baixado
  e usado antes de o resto do pacote existir;
* `raiz` — localiza a pasta `codigo/` e a ativa para o processo e os
  subprocessos;
* `colab` — o que depende do Colab (detecção, Drive, Secrets);
* `segredos` — credenciais pelos Secrets, separadas por finalidade, que nunca
  aparecem em repr, logs ou registros;
* `ambiente` — versões, GPU, plano de instalação e compatibilidade para
  retomar;
* `ids` — identificadores de configuração e de execução;
* `proveniencia` — o registro `experimento.json` ao lado de cada execução;
* `persistencia` — sincronização com o Google Drive (arquivo temporário,
  renomeação, verificação) e restauração;
* `publicacao_hf` — plano e publicação de artefatos em repositório privado do
  Hugging Face, sempre por ação explícita.

Nada aqui executa treinamento, avaliação, download de pesos ou de dados, nem
publica algo ao ser importado. Os módulos só importam a biblioteca padrão e
`reclin.util`; `huggingface_hub`, `torch` e `google.colab` são importados
dentro das funções que os usam.
"""
