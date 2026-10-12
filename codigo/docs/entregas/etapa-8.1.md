# Etapa 8.1 — decisões de arquitetura e rascunho da emenda de protocolo

Data: 12/10/2026. Etapa **documental**. Alterações feitas direto na pasta de
trabalho. Nada foi executado: nenhum teste, treino, calibração, download, acesso
à rede, avaliação do TEST nem comando de controle de versão. Código, dados,
MANIFEST, referências, protocolo vigente e critérios de aceitação não foram
alterados. A etapa 8.2 (implementação) **não** foi iniciada.

## 1. O que esta etapa entrega

1. O registro das decisões de arquitetura do autor (seção 2).
2. O levantamento preliminar dos arquivos que afetam a identidade de cada
   componente (seção 4), feito por análise estática de importações.
3. O rascunho da emenda de protocolo para a evolução do Two-Stage:
   [`../protocolo/rascunho-emenda-two-stage-desenvolvimento.md`](../protocolo/rascunho-emenda-two-stage-desenvolvimento.md).
   **Não está em vigor** e não é conferida por nenhum teste.

## 2. Decisões registradas (autor, 12/10/2026)

| # | Decisão | Consequência para a implementação |
| --- | --- | --- |
| 1 | **Identidade por componente** (baseline, estágio 1, estágio 2, composição e demais). Mudanças não relacionadas não invalidam checkpoints compatíveis. Documentar arquivos e dependências de cada identidade | Novo algoritmo de identidade por componente; retomada, reaproveitamento e inventário passam a usar a identidade do componente. Seção 4 é o levantamento preliminar |
| 2 | **Natureza `desenvolvimento`** e emenda em rascunho, sem ativação automática. Nenhuma configuração é científica só por ser BioBERTpt e seed 42. Emenda e procedimento de promoção revisados antes de qualquer experimento dessa fase | O inventário passa a conferir a configuração contra o protocolo; marca de desenvolvimento verificada no código; `promocao.json` |
| 3 | **Código no Colab:** tarball de revisão exata do GitHub, SHA completo, explícita e verificável. **Nenhum comando Git no notebook.** Tokens só pelos Secrets do Colab, sem aparecer em arquivos, logs ou saídas | Bootstrap com download por SHA e verificação; redação de segredos em logs e saídas; verificação de que nenhum notebook salvo contém segredo |
| 4 | **Persistência:** treino no disco local temporário; sincronização de checkpoints e resultados com o Drive por arquivo temporário, renomeação segura e verificação de integridade. Retomada detecta execuções interrompidas e confere a compatibilidade | Sincronizador com verificação; estado derivado de `config.json` e `checkpoints/ultimo.json` (o `treino.json` só existe ao fim da sessão) |
| 5 | **Hugging Face:** repositório privado para checkpoints e artefatos grandes; manifesto de integridade e proveniência por execução; **não publicar `ultimo.pt` por padrão**; estrutura que não sobrescreva execuções anteriores | Layout por `run_id`, imutável; manifesto de SHA-256 |
| 6 | **Identidade dos pesos** por revisão e manifesto, não por caminho absoluto | Mudança no núcleo: o caminho do modelo deixa de entrar na comparação de retomada e na identidade do checkpoint, e a composição deixa de depender de caminhos absolutos dos estágios |
| 7 | **TEST em `reclin_teste.ipynb`**, separado, com autorização explícita, conferência de proveniência e das condições de congelamento | Os notebooks de estratégia não têm células de TEST |
| 8 | **`reclin.ipynb` mantido**, sem ser obrigatório no novo fluxo; finalidade preservada; sem migração destrutiva | Nenhuma alteração nele |
| 9 | **Nomes oficiais preservados** e associados aos novos identificadores por registro. Novas configurações exploratórias com identificadores próprios, sem sobrescrever os oficiais | Registro de aliases; `relatorios/fase.py` e `Makefile` continuam valendo |
| 10 | **Estágio 2 com hiperparâmetros próprios** quando tecnicamente válido; **não remover** `CAMPOS_BASE` sem substituí-lo por validação explícita dos parâmetros compartilhados, dos específicos e das condições reais de compatibilidade (composição e retomada) | Substituição de `CAMPOS_BASE` em `composicao.conferir_par`, com testes |
| 11 | **GPU:** T4 de referência; L4/A100 permitidas se compatíveis, com hardware registrado e aviso de variação numérica; **nunca trocar a GPU em silêncio na retomada** quando isso violar a reprodução | Registro e comparação do hardware por sessão; recusa ou confirmação explícita na retomada |

**Ordem de implementação** (autor): (1) infraestrutura compartilhada;
(2) `reclin_baseline.ipynb`; (3) `reclin_two_stage.ipynb`; (4)
`reclin_resultados.ipynb`; (5) `reclin_teste.ipynb`; (6) demais estratégias;
(7) documentação e conferência final.

**Operações explícitas exigidas nos notebooks:** conferir o ambiente; listar
execuções; planejar em dry-run; iniciar novo treinamento; retomar; consultar o
DEV; preparar a publicação de artefatos. Os notebooks chamam os scripts reais e
não duplicam algoritmos.

## 3. Regras de processo vigentes para a etapa 8

- As falhas anteriores da suíte não são investigadas nem corrigidas nesta
  etapa; dados, MANIFEST, referências, protocolo e critérios não são alterados
  para fazê-las passar.
- **A suíte não é executada automaticamente.** Testes focados nos novos
  componentes podem ser criados ou atualizados quando a arquitetura exigir, mas
  não são executados sem autorização; preferem-se verificações estáticas e
  revisão de código.
- Nenhum treino, calibração, download, experimento científico ou avaliação do
  TEST durante a implementação. A fase continua limitada a BioBERTpt e seed 42.
- Nenhuma operação científica é ativada ao abrir ou executar um notebook.
- Nenhum comando Git por mim; o controle de versão é seu.
- Após cada etapa, relatório e **parada** para a sua revisão.

## 4. Identidade por componente: levantamento preliminar

**Método:** fecho transitivo das importações (`import` e `from … import`,
inclusive as feitas dentro de funções, e os `__init__.py` dos pacotes) a partir
dos módulos de cada componente, por análise sintática (AST) sobre `reclin/`.
Nenhum código do projeto foi importado ou executado. **É um levantamento, não a
especificação final** (que será feita e testada na etapa 8.2, junto com o
algoritmo).

**Núcleo comum a todos os componentes (8 arquivos):**

`reclin/__init__.py`, `reclin/avaliacao/__init__.py`, `reclin/avaliacao/metricas.py`, `reclin/execucao/__init__.py`, `reclin/execucao/predicoes.py`, `reclin/tarefa.py`, `reclin/util/__init__.py`, `reclin/util/io.py`

**Arquivos específicos de cada componente** (fecho de importações menos o núcleo comum):

| Componente | Específicos | Arquivos |
| --- | --- | --- |
| baseline | 20 | `config.py`, `entrada.py`, `estrategias/__init__.py`, `estrategias/baseline.py`, `execucao/diretorio.py`, `execucao/subconjunto.py`, `execucao/trilha.py`, `execucao/validacao.py`, `modelos.py`, `particoes.py`, `pretreino.py`, `treino/__init__.py`, `treino/checkpoint.py`, `treino/classificador.py`, `treino/laco.py`, `treino/montagem.py`, `treino/reprodutibilidade.py`, `util/caminhos.py`, `util/identidade_codigo.py`, `versao_dados.py` |
| restrito | 22 | `config.py`, `entrada.py`, `estrategias/__init__.py`, `estrategias/restrito.py`, `execucao/diretorio.py`, `execucao/subconjunto.py`, `execucao/trilha.py`, `execucao/validacao.py`, `modelos.py`, `negacao/__init__.py`, `negacao/lexico.py`, `particoes.py`, `pretreino.py`, `treino/__init__.py`, `treino/checkpoint.py`, `treino/classificador.py`, `treino/laco.py`, `treino/montagem.py`, `treino/reprodutibilidade.py`, `util/caminhos.py`, `util/identidade_codigo.py`, `versao_dados.py` |
| pair_aware | 23 | `config.py`, `entrada.py`, `estrategias/__init__.py`, `estrategias/pair_aware/__init__.py`, `estrategias/pair_aware/cabeca.py`, `execucao/diretorio.py`, `execucao/subconjunto.py`, `execucao/trilha.py`, `execucao/validacao.py`, `modelos.py`, `negacao/__init__.py`, `negacao/lexico.py`, `particoes.py`, `pretreino.py`, `treino/__init__.py`, `treino/checkpoint.py`, `treino/classificador.py`, `treino/laco.py`, `treino/montagem.py`, `treino/reprodutibilidade.py`, `util/caminhos.py`, `util/identidade_codigo.py`, `versao_dados.py` |
| estagio (1 e 2) | 24 | `config.py`, `entrada.py`, `estrategias/__init__.py`, `estrategias/two_stage/__init__.py`, `estrategias/two_stage/estagios.py`, `estrategias/two_stage/inicializacao.py`, `estrategias/two_stage/metricas.py`, `estrategias/two_stage/treino.py`, `execucao/diretorio.py`, `execucao/subconjunto.py`, `execucao/trilha.py`, `execucao/validacao.py`, `modelos.py`, `particoes.py`, `pretreino.py`, `treino/__init__.py`, `treino/checkpoint.py`, `treino/classificador.py`, `treino/laco.py`, `treino/montagem.py`, `treino/reprodutibilidade.py`, `util/caminhos.py`, `util/identidade_codigo.py`, `versao_dados.py` |
| composicao | 22 | `avaliacao/limiar.py`, `config.py`, `entrada.py`, `estrategias/__init__.py`, `estrategias/two_stage/__init__.py`, `estrategias/two_stage/composicao.py`, `execucao/diretorio.py`, `execucao/subconjunto.py`, `execucao/trilha.py`, `execucao/validacao.py`, `modelos.py`, `particoes.py`, `pretreino.py`, `treino/__init__.py`, `treino/checkpoint.py`, `treino/classificador.py`, `treino/laco.py`, `treino/montagem.py`, `treino/reprodutibilidade.py`, `util/caminhos.py`, `util/identidade_codigo.py`, `versao_dados.py` |
| baseline_calibrado | 9 | `avaliacao/limiar.py`, `estrategias/__init__.py`, `estrategias/baseline_calibrado.py`, `execucao/diretorio.py`, `execucao/trilha.py`, `execucao/validacao.py`, `util/caminhos.py`, `util/identidade_codigo.py`, `versao_dados.py` |
| filtro_pistas | 4 | `estrategias/__init__.py`, `estrategias/filtro_pistas.py`, `negacao/__init__.py`, `negacao/lexico.py` |


**Raízes usadas:** baseline = `estrategias/baseline` + `treino/classificador`;
restrito e pair_aware = o módulo da estratégia + `treino/classificador`;
estágio = `estrategias/two_stage/treino`; composição =
`estrategias/two_stage/composicao`; baseline_calibrado e filtro_pistas = o
módulo da estratégia.

**Pontos que o levantamento mostra e que a etapa 8.2 precisa decidir:**

1. **`util/identidade_codigo.py` está no fecho de quase todos os componentes**
   (tudo o que grava `config.json`). Mudar o próprio algoritmo de identidade
   invalida tudo, o que é o esperado; mudar o resto do pacote não.
2. **A composição não inclui os arquivos do treino dos estágios**, mas lê os
   `melhor_modelo/` deles: a dependência de dados (hash do `melhor_modelo/`, já
   conferido) cobre isso. A compatibilidade entre estágios precisa ser
   verificada por campos, não pela identidade de código do outro componente.
3. **Scripts.** `scripts/treinar.py` serve baseline, restrito e pair_aware;
   `scripts/two_stage.py` serve estágios e composição. Incluir os bytes do
   script na identidade de todos os componentes que ele serve faz uma mudança
   de argumentos de uma estratégia tocar as outras. Decisão: incluir só a parte
   relevante, separar a lógica em módulos por componente, ou aceitar o efeito.
   Para `filtro_pistas` e `baseline_calibrado`, parte da lógica está nos
   scripts (`filtro_pistas.py`, `baseline_calibrado.py`): devem entrar.
4. **Docstrings e comentários.** A identidade atual hasheia os bytes; editar um
   docstring invalida. Proposta (a confirmar na 8.2): hashear a AST normalizada
   (sem docstrings/comentários), com novo algoritmo e versão.
5. **`pyproject.toml`** entra hoje inteiro (inclui marcadores do pytest). Para a
   identidade de treino, só a seção de dependências interessa.
6. **Dependências externas** (torch, transformers…) não entram no hash de
   código: continuam registradas em `ambiente` por sessão. A política de
   compatibilidade entre versões na retomada precisa ser explícita.
7. **Limite do método:** arquivos lidos em tempo de execução (partições,
   pesos) não aparecem no fecho de importações; eles já têm identidade própria
   (SHA-256 das partições, manifesto dos pesos).

## 5. Rascunho da emenda de protocolo

Resumo do que o rascunho propõe, sem ativá-lo: natureza `desenvolvimento` com
TEST recusado no código; espaço de configuração e orçamento **a definir por
você**; critério de comparação só no DEV (F1 de `negation_of` do pipeline
completo, com τ pela busca pré-registrada); registro de todas as tentativas e
declaração de otimismo do DEV; reaproveitamento de estágios por chave de
compatibilidade; promoção por `promocao.json` com autorização explícita; o
protocolo vigente permanece intacto. Pontos em aberto na seção 9 do rascunho.

## 6. Arquivos

Criados (2):

| Arquivo | Finalidade |
| --- | --- |
| `codigo/docs/entregas/etapa-8.1.md` | esta nota |
| `codigo/docs/protocolo/rascunho-emenda-two-stage-desenvolvimento.md` | rascunho da emenda E1, não em vigor |

Modificados (1): `codigo/docs/entregas/README.md` (linha da etapa 8.1).
Removidos: nenhum. Código, testes, dados, MANIFEST, referências e o protocolo
vigente: **não alterados**.

## 7. Verificações

| Verificação | Resultado |
| --- | --- |
| Análise estática de importações (seção 4), só leitura, sem importar o projeto | executada |
| Busca nos testes por varredura de `docs/protocolo/` | nenhuma encontrada; os testes de protocolo leem arquivos específicos |
| Testes, treino, download, rede, Git | **não executados** (conforme instruções) |
| Efeito dos novos arquivos sobre a identidade do código | nenhum (documentos fora de `reclin/`, `scripts/` e `pyproject.toml`); **inferência** pelas regras de `identidade_codigo.py`, não executada |

## 8. Pendências e o que depende de você

1. Revisar a nota e o rascunho da emenda; nenhuma implementação começa antes da
   sua autorização explícita para a etapa 8.2.
2. Preencher no rascunho: espaço de configuração, N, campos compartilhados e
   específicos, efeito da promoção nas comparações.
3. Decisões técnicas da seção 4 (scripts, docstrings, `pyproject.toml`) — podem
   ser tomadas na própria 8.2.
4. Referências históricas: `src/hf_backup.py` (`HFBackup`, `resolve_token`) e a
   célula 11 dos notebooks 01 e 02 do legado, commit `a5f055c`. Não foram lidas.
   Preciso saber se você autoriza consultá-las (só comportamento, sem copiar)
   ao iniciar a infraestrutura de publicação.
5. D5 (redação do teto de 2 épocas) segue por confirmar; não bloqueia.
