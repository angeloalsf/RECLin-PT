# Especificação — identidade por componente (etapa 8.2)

> **Estado: ESPECIFICAÇÃO FECHADA na etapa 8.2 (12/10/2026), com as decisões
> finais do autor. Nada aqui está implementado.** Nenhum código, teste, dado,
> MANIFEST ou protocolo foi alterado. O que vale hoje é a identidade global D6
> (`reclin/util/identidade_codigo.py`, algoritmo `sha256-manifesto-v1`),
> descrita na seção 1; esta especificação orienta a implementação da etapa 8.3.
>
> **Decisões finais aplicadas (autor, 12/10/2026):** (1) identidade separada
> por estágio e por composição, com identidade comum para o que é
> compartilhado; (2) scripts compartilhados mapeados aos componentes que
> afetam; (3) **hash integral dos arquivos, incluindo docstrings; sem
> normalização de AST**; (4) `pyproject.toml` fora do hash, com versões e
> ambiente registrados e conferidos antes de retomar; (5) `__version__`
> permanece no hash; (6) uma mudança que atinge só um componente não invalida
> os independentes, mas invalida todos os efetivamente afetados por módulos ou
> scripts compartilhados. As alternativas O1/O2/O3 estão **encerradas**.

Base: levantamento estático da etapa 8.1 (fecho transitivo das importações por
AST, sem importar o projeto), complementado nesta etapa por leitura de
`composicao.py`, `treino.py` do Two-Stage, `config.py`, `modelos.py`,
`pyproject.toml` e por uma verificação AST dos `__init__.py`. Onde a conclusão
é inferência e não leitura, está dito.

## 1. Situação atual (fatos verificados)

- Um único SHA-256 sobre `reclin/**/*.py`, `scripts/*.py` e `pyproject.toml`
  (CRLF→LF), exceto `reclin/analises/`, `reclin/relatorios/`,
  `execucao/inventario.py`, `util/formatacao.py` e os scripts `analisar`,
  `gerar_tcc`, `conferir_tcc`. Gravado em `config.json["codigo"]`, nas sessões
  de treino e na trilha do TEST.
- Divergência global **bloqueia**: retomada (`diretorio.conferir_codigo`),
  uso de estágio (`composicao.ler_estagio`), TEST, execuções derivadas e a
  classificação do inventário.
- Consequência: editar `scripts/comparar.py`, um docstring ou uma marca do
  pytest em `pyproject.toml` invalida todos os treinos e checkpoints.
- Fatos novos desta etapa:
  1. `__doc__` não é usado em `reclin/`; só os scripts o usam (texto de ajuda
     do `argparse`). Verificado por busca textual.
  2. Os `__init__.py` que contêm **apenas docstring** (verificado por AST):
     `avaliacao`, `corpus`, `estrategias`, `execucao`, `negacao`, `treino`,
     `util`. `reclin/__init__.py` **não** é só docstring: define
     `__version__ = "0.1.0"`. `estrategias/two_stage/__init__.py` e
     `estrategias/pair_aware/__init__.py` têm código.
  3. `two_stage/treino.py` importa `ErroExecucao` de `treino/classificador.py`
     (o treino do baseline/restrito/pair_aware) **apenas para a exceção**.
     Por isso o fecho do estágio inclui `classificador.py` inteiro.
  4. `pyproject.toml` entra inteiro, inclusive `[tool.pytest.ini_options]`
     (marcadores e `addopts`) e `[project.optional-dependencies]`.
  5. O registro de execução guarda `modelo` como **string de caminho**
     (`str(modelo)`), e a retomada compara o `registro` completo: a identidade
     de configuração hoje depende do caminho local dos pesos.

## 2. Princípios (regras que sustentam toda a especificação)

- **P1 — assimetria de segurança.** Invalidar sem necessidade custa
  recomputação; reaproveitar sem validade custa a validade científica. Em caso
  de dúvida, invalida-se. Nenhuma regra abaixo pode causar reaproveitamento
  indevido; só pode reduzir invalidação indevida, e só com evidência.
- **P2 — nada é "não funcional" por presunção.** Uma alteração só deixa de
  invalidar um componente se o arquivo está fora do conjunto declarado dele
  **e** fora do fecho de importações. Não há isenção para docstrings,
  comentários ou formatação (seção 6).
- **P3 — o que não é código tem identidade própria.** Dados (SHA-256 das
  partições, versão `v2-texto-fonte`, MANIFEST), léxico (`lexico_sha1`),
  pesos (revisão de 40 hex + manifesto `PROVENIENCIA.json`), tokenizer
  (hash do vocabulário e ids dos marcadores) e configuração efetiva entram na
  chave de cada componente; não são capturados pelo hash de código.
- **P4 — identidade por arquivo, não só agregada.** O manifesto grava o hash
  de cada arquivo do conjunto; o agregado é derivado. Assim a comparação diz
  **quais arquivos mudaram** e **quais componentes isso atinge**.

## 3. Os quatro níveis e seus conjuntos de arquivos

Conjuntos obtidos do fecho de importações (8.1). Contagens conferem:
estágio = 8 + 19 + 5; composição = 8 + 19 + 3.

### 3.1 Núcleo comum (N0) — 8 arquivos; afeta **todos** os componentes

`reclin/__init__.py`, `avaliacao/__init__.py`, `avaliacao/metricas.py`,
`execucao/__init__.py`, `execucao/predicoes.py`, `tarefa.py`,
`util/__init__.py`, `util/io.py`.

Observações: `reclin/__init__.py` tem `__version__` (código): mudar a versão do
pacote invalida tudo — **decisão final: `__version__` permanece no hash**. `avaliacao/metricas.py` define as métricas de
seleção no DEV; `predicoes.py` o arredondamento de probabilidades usado no
roteamento: ambos são numéricos, não "formais".

### 3.2 Núcleo de treino (N1) — 19 arquivos; afeta baseline, restrito, pair_aware, estágios e composição

Dividido por função, para a política de duas classes da seção 6.3:

| Grupo | Arquivos | Papel | Classe (descritiva) |
| --- | --- | --- | --- |
| N1-numérico | `config.py`, `entrada.py`, `modelos.py`, `particoes.py`, `pretreino.py`, `versao_dados.py`, `treino/__init__.py`, `treino/checkpoint.py`, `treino/classificador.py`, `treino/laco.py`, `treino/montagem.py`, `treino/reprodutibilidade.py`, `execucao/subconjunto.py` | definem dados, entrada, rede, laço, sementes, estado de retomada | **S** (semântica) |
| N1-controle | `execucao/diretorio.py`, `execucao/trilha.py`, `execucao/validacao.py`, `util/caminhos.py`, `estrategias/__init__.py` (só docstring) | gravam e conferem registros, trilhas, caminhos | **R** (registro) |
| N1-identidade | `util/identidade_codigo.py` | define a própria identidade | **S** (mudar o algoritmo invalida tudo, como esperado) |

A coluna "classe" é **descritiva** (papel do arquivo). Não há tratamento
diferente por classe: todo arquivo do conjunto invalida o componente (seção 10).

### 3.3 Identidade da estratégia (N2)

| Estratégia | Arquivos próprios (além de N0+N1) | Observação |
| --- | --- | --- |
| baseline | `estrategias/baseline.py` | |
| restrito | `estrategias/restrito.py`, `negacao/__init__.py` (só docstring), `negacao/lexico.py` | + identidade do léxico congelado (`lexico_sha1`) na chave |
| pair_aware | `estrategias/pair_aware/__init__.py`, `estrategias/pair_aware/cabeca.py` | desde a D10 não usa o léxico (sem `negacao/*` e sem `lexico_sha1` na chave); a chave leva `versao` da configuração |
| baseline_calibrado | `estrategias/baseline_calibrado.py`, `avaliacao/limiar.py` + subconjunto de N1-controle (`diretorio`, `trilha`, `validacao`, `caminhos`, `identidade_codigo`, `versao_dados`, `estrategias/__init__`) | **não** depende de N1-numérico: não importa `entrada`/`laco`/`modelos`; depende de uma execução do baseline por dados (hash das probabilidades), não por código |
| filtro_pistas | `estrategias/filtro_pistas.py`, `estrategias/__init__.py`, `negacao/*` | só N0 + isso |
| regra_pura | **a levantar na 8.3** (não foi incluída no levantamento da 8.1) | pendência |

### 3.4 Identidade do estágio (E1, E2) — 5 arquivos próprios

`estrategias/two_stage/__init__.py`, `estagios.py`, `inicializacao.py`,
`metricas.py`, `treino.py`. Os dois estágios compartilham **o mesmo conjunto de
arquivos** porque o código é parametrizado por `estagio`. A identidade de cada
estágio é então:

```
identidade_estagio(n) = { conjunto_de_codigo (N0 + N1 + 5 próprios, hashes por arquivo),
                          estagio: n,
                          estrategia_config: registro_estagio(n),   # rótulos, dados de treino, campo de seleção
                          config_efetiva: ConfigEstagio<n> completa,
                          dados, pesos, tokenizer }          # P3
```

Problema (fato): constantes dos dois estágios vivem no mesmo arquivo
(`two_stage/__init__.py`: `ROTULOS`, `CAMPO_SELECAO`, `DADOS_TREINO`,
`CONFIGS`, `rotulo_estagio1/2`). Alterar só o critério do estágio 2 muda os
bytes do arquivo e invalida o estágio 1, embora o `registro_estagio(1)` não
mude. **Requisito da 8.3 (decorrente da decisão de não invalidar componentes independentes):** separar a parte de cada estágio em
módulos próprios (por exemplo `two_stage/estagio1.py` e `estagio2.py`) e deixar
em `__init__.py` só o que é realmente comum, de modo que o conjunto declarado
do estágio 1 não contenha o código exclusivo do estágio 2 e vice-versa. Se a separação não for viável dentro do escopo, vale a invalidação
conservadora dos dois (seção 10, item 1).

Segundo problema (fato): a importação de `ErroExecucao` arrasta
`treino/classificador.py` para o estágio. **Requisito da 8.3 (decorrente da decisão de não invalidar componentes independentes):**
mover a exceção para um módulo neutro de N0 (ou de N1-controle) e remover a
dependência do estágio sobre o treino do baseline. Efeito pretendido: mudar a
lógica de treino específica do baseline/restrito/pair_aware deixa de invalidar
estágios treinados — **somente se** o estágio não usar nenhuma outra função de
`classificador.py` (a conferir na 8.3 por leitura; o levantamento por
importações mostra o módulo, não as funções usadas).

### 3.5 Identidade da composição — 3 arquivos próprios

`estrategias/two_stage/composicao.py`, `estrategias/two_stage/__init__.py`,
`avaliacao/limiar.py` (+ N0 + N1).

```
identidade_composicao = { conjunto_de_codigo (N0 + N1 + 3 próprios),
                          vinculo_estagio1: chave_de_reaproveitamento do estágio 1 + sha256 do melhor_modelo/,
                          vinculo_estagio2: idem,
                          parametros: regra de roteamento (P(relacao) arredondada a 4 casas >= τ),
                                      grade de τ (0,05–0,95, passo 0,05) e desempates (protocolo),
                                      conjunto DEV avaliado (sha256 da partição) }
```

A composição **não** inclui os 4 arquivos de treino do estágio
(`estagios`, `inicializacao`, `metricas`, `treino`): a dependência é de **dados**
(hash do `melhor_modelo/`, já conferido por `ler_estagio`) e de campos
(`compatibilidade-two-stage.md`), não de código do outro componente. A
composição é inferência no DEV: invalidá-la é barato; por isso ela usa o
fecho completo, sem refinamento.

## 4. Scripts (decisão: mapear cada script aos componentes que ele afeta)

Fatos: `scripts/treinar.py` serve baseline, restrito e pair_aware (e carrega o
léxico congelado para os dois últimos); `scripts/two_stage.py` serve os
subcomandos `verificar-inicializacao`, `treinar --estagio`, `compor-dev` e
`avaliar-test`; `scripts/baseline_calibrado.py`, `filtro_pistas.py` e
`regra_pura.py` contêm parte da lógica da estratégia. Hoje todos entram no hash
global.

Regra adotada: o hash integral (bytes) de cada script entra **somente** na
identidade dos componentes que ele serve:

| Script | Entra na identidade de |
| --- | --- |
| `treinar.py` | baseline, restrito, pair_aware |
| `two_stage.py` | estágio 1, estágio 2, composição (e registrado na trilha do TEST) |
| `baseline_calibrado.py` | baseline_calibrado |
| `filtro_pistas.py` | filtro_pistas |
| `regra_pura.py` | regra_pura |
| `pretreino.py` | nenhum componente de treino: a identidade dos pesos é o manifesto (P3); entra no registro de proveniência como ferramenta que produziu o manifesto |
| `avaliar.py`, `avaliar_test.py`, `comparar.py`, `calibrar.py` | identidade de **avaliação** (trilha do TEST e relatórios), não a de treino |
| `preparar_dados.py`, `reconstruir_dados.py`, `selecionar_min_freq.py` | identidade dos **dados**, não a de treino |
| `analisar.py`, `gerar_tcc.py`, `conferir_tcc.py` | fora (como hoje) |

Consequência aceita (P1): mudar `treinar.py` invalida baseline, restrito e
pair_aware — componentes que ele efetivamente afeta — e **não** o Two-Stage;
mudar `two_stage.py` invalida estágios e composição e **não** o baseline. A
configuração efetiva completa (todos os campos de `Config` e da estratégia)
continua sempre gravada, de modo que um valor *default* alterado aparece como
configuração diferente.

## 5. `pyproject.toml` e ambiente (decisão: fora do hash; ambiente registrado)

Fatos: o arquivo inteiro entra no hash hoje; o pytest marca testes lentos ali;
dependências com limite inferior (`numpy>=1.24`, `scipy>=1.11`, `torch>=2.11`,
`transformers>=5.16`).

Regra adotada:

1. `pyproject.toml` **sai** do hash de código dos componentes.
2. Cada sessão de treino grava `ambiente`: versão do Python, `numpy`, `scipy`,
   `torch`, `transformers`, `tokenizers`, CUDA/cuDNN, modelo da GPU, e o hash
   normalizado de `[project].dependencies` e `requires-python`.
3. **Antes de retomar uma execução**, o ambiente atual é comparado com o
   gravado. Divergência de Python (minor), `torch`, `transformers`,
   `tokenizers`, `numpy` ou do tipo/modelo de GPU → a retomada é **bloqueada**
   (o estado inclui RNG e otimizador; `restaurar_rng` já recusa CPU×GPU).
   Divergência em dependências sem efeito numérico → aviso registrado.
4. No reaproveitamento de estágio concluído (só inferência), divergência de
   versões ou de GPU gera aviso registrado em `experimento.json`; divergência
   de código do componente continua bloqueando.
5. Risco aceito: uma mudança só em empacotamento (`[tool.setuptools]`) deixa de
   ser detectada pelo hash; não afeta o código importado em execução.

## 6. Docstrings, comentários e alterações "não funcionais" (decisão final)

**Regra:** o hash cobre o **arquivo inteiro, byte a byte** (com a normalização
CRLF→LF já usada), **incluindo docstrings, comentários e espaço em branco**.
Não há normalização de AST nem isenção para alteração textual.

Regra de sustentação (por que nada é presumido irrelevante): hoje nenhum
módulo de `reclin/` lê `__doc__` (os scripts o usam no texto de ajuda do
`argparse`), mas isso não garante que continuará assim, e um docstring pode ser
inspecionado por bibliotecas. Em vez de manter uma lista de exceções que
precisaria ser verificada a cada edição, qualquer alteração em arquivo do
conjunto de um componente o invalida (P1/P2).

Consequências documentadas:

- Editar um docstring ou comentário num arquivo de um componente invalida
  **apenas** os componentes cujo conjunto contém esse arquivo (seção 8). A
  economia vem da granularidade por componente, não de isentar texto.
- Os 7 `__init__.py` que contêm só docstring (`avaliacao`, `corpus`,
  `estrategias`, `execucao`, `negacao`, `treino`, `util`) entram nos conjuntos
  em que aparecem no fecho; `estrategias/__init__.py` aparece em quase todos.
  Quem alterar esses docstrings deve contar com a invalidação dos componentes
  listados na seção 8.
- Como o hash é por arquivo (P4), o relatório de invalidação mostra quais
  arquivos mudaram, o que permite ao autor julgar se a mudança foi só textual
  — mas essa leitura não altera a invalidação.

As opções O1, O2 e O3 anteriormente discutidas estão **encerradas**; a regra é
a O1 (hash integral).

## 7. Algoritmo e versionamento

- Novo algoritmo `sha256-manifesto-v2` (nome proposto), coexistindo com `v1`:
  mesma regra de bytes (CRLF→LF), mas calculado **por componente**, sem
  `pyproject.toml` e com os scripts mapeados da seção 4. Uma execução `v1`
  continua válida e legível e **nunca é reescrita**; a comparação `v1` × `v2`
  é "não comparável → trata-se como diferente" (P1).
- O manifesto de cada componente grava: algoritmo, lista ordenada de
  `{caminho, sha256}` (bytes), agregado, nome do componente e versão desta
  especificação.
- Conjunto declarado × fecho: um teste estático (a escrever na 8.3, sem
  execução automática) compara, para cada componente, o conjunto declarado com
  o fecho de importações por AST (usada só para listar arquivos, nunca para
  calcular o hash) e **falha** se o fecho tiver arquivo fora do conjunto.
  Conjunto declarado maior que o fecho é permitido, com justificativa (P1).
- `experimento.json` (ao lado de cada execução, nunca dentro de `config.json`)
  guarda os manifestos dos componentes envolvidos, a chave de reaproveitamento
  (ver `compatibilidade-two-stage.md`) e o `ambiente`.

## 8. Matriz de invalidação

Linhas: onde ocorre a mudança. ● = invalida o componente; – = não invalida.
Derivada dos conjuntos acima; **inferência estática**, a ser confirmada por
teste de conjunto×fecho na 8.3.

| Mudança em… | baseline | restrito | pair_aware | estágio 1 | estágio 2 | composição | baseline_calibrado | filtro_pistas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0 (8 arquivos) | ● | ● | ● | ● | ● | ● | ● | ● |
| N1-numérico (13) | ● | ● | ● | ● | ● | ● | – | – |
| N1-controle (5) e `util/identidade_codigo.py` | ● | ● | ● | ● | ● | ● | ● (exceto `estrategias/__init__` e subconj.) | só `estrategias/__init__` |
| `baseline.py` | ● | – | – | – | – | – | – | – |
| `restrito.py` | – | ● | – | – | – | – | – | – |
| `pair_aware/*` | – | – | ● | – | – | – | – | – |
| `negacao/*` (léxico) | – | ● | ● | – | – | – | – | ● |
| `two_stage/__init__.py` (hoje) | – | – | – | ● | ● | ● | – | – |
| `two_stage/{estagios,inicializacao,metricas,treino}.py` | – | – | – | ● | ● | – | – | – |
| `two_stage/composicao.py`, `avaliacao/limiar.py` | – | – | – | – | – | ● | ● (só `limiar.py`) | – |
| `scripts/treinar.py` (S1) | ● | ● | ● | – | – | – | – | – |
| `scripts/two_stage.py` (S1) | – | – | – | ● | ● | ● | – | – |
| docstring/comentário/espaço em arquivo do conjunto | invalida os componentes que contêm o arquivo (mesma regra da linha do arquivo) | | | | | | | |
| `pyproject.toml` (fora do hash; ambiente conferido antes de retomar, seção 5) | – | – | – | – | – | – | – | – |

Linha adicional, **depois** da separação por estágio (requisito da 3.4, decorrente da decisão de não invalidar componentes independentes): mudar
só o código do estágio 2 deixa de invalidar o estágio 1, e vice-versa; mudar
`classificador.py` deixa de invalidar os estágios (se a conferência da 3.4
confirmar).

Mudanças fora de qualquer conjunto (docs, testes, notebooks, `analises/`,
`relatorios/`, `inventario.py`, `formatacao.py`) **não invalidam nada**, com a
guarda de que notebooks não contêm algoritmo (teste estático da 8.3: notebooks
só chamam scripts).

## 9. Como a especificação cumpre as decisões

- Núcleo comum, estratégia, estágio e composição estão separados com listas de
  arquivos (seções 3.1–3.5).
- `treinar.py`, `two_stage.py`, `pyproject.toml` e os módulos de identidade
  têm regra explícita (seções 4, 5, 7).
- Docstrings: hash integral, sem isenção (seção 6).
- A invalidação é localizada: hash por arquivo (P4), matriz da seção 8 e a
  separação do código por estágio.

## 10. Registro de escolhas aplicadas sem nova consulta e itens da 8.3

Escolhas feitas para completar detalhes não especificados, sem ampliar escopo:

1. A separação das constantes por estágio e a remoção da dependência do estágio
   sobre `treino/classificador.py` (3.4) são a implementação mínima da decisão
   "não invalidar componentes independentes". Se a 8.3 constatar que cumpri-las
   exige alterar além do necessário, vale a invalidação conservadora (P1) e a
   constatação é registrada no relatório da 8.3.
2. As classes S/R de arquivo (tabela 3.2) **não** são adotadas: todo arquivo
   do conjunto invalida o componente. A coluna "papel" da tabela é apenas
   descritiva.
3. `regra_pura` e os scripts de avaliação/dados serão levantados na 8.3 por
   leitura, aplicando a mesma regra de mapeamento.
4. Quais funções de `classificador.py`, `checkpoint.py` e `diretorio.py` são
   usadas por estágio e composição será conferido na 8.3 por leitura.
5. `__version__` permanece no hash (3.1).

## 11. Infraestrutura compartilhada de execução (aplicado em 12/10/2026)

Aplicação mínima desta especificação à infraestrutura do Colab
(`reclin/experimento/**/*.py` e `scripts/experimento.py`), sem implementar a
identidade `v2` por componente (que continua sendo a etapa 8.3):

- A infraestrutura **não pertence a nenhum componente de treino** (não está no
  fecho de importações de baseline, restrito, pair_aware, estágios nem
  composição, e nada fora dela a importa — conferido por teste). Pela regra da
  seção 8 ("mudanças fora de qualquer conjunto não invalidam nada"), editá-la
  não pode invalidar treinos nem checkpoints.
- Por isso saiu da identidade global v1 (`util/identidade_codigo.py`:
  `reclin/experimento/` em `EXCLUIDOS_PREFIXOS`, `scripts/experimento.py` em
  `EXCLUIDOS`). O algoritmo `sha256-manifesto-v1` e a regra de bytes não
  mudaram; só dois caminhos passaram a ser excluídos, como já eram
  `inventario.py`, `formatacao.py` e os relatórios. Efeito: o hash global do
  código dos treinos existentes é o mesmo de antes da infraestrutura existir.
- A infraestrutura tem **identidade própria** (`reclin/experimento/identidade.py`,
  componente `infraestrutura`, mesma regra de bytes, docstrings incluídas),
  gravada em `experimento.json` (`codigo.infraestrutura`) como proveniência.
  É informativa: **nunca bloqueia retomada nem reaproveitamento**.
- `pyproject.toml` continua dentro da identidade v1 até a 8.3 (decisão 4 só
  se aplica com o algoritmo `v2`); isto não foi alterado aqui.

