# Etapa 8.3.2 — notebook do baseline simplificado (relatório)

`notebooks/reclin_baseline.ipynb` foi reescrito para ficar parecido com o notebook legado
(`notebooks/01_baseline_biobertpt_colab.ipynb`, de RECLin-PT-original): seções curtas, um markdown e uma célula
de código por finalidade, **11 células de código** (eram 22). Só a interface mudou: algoritmos, protocolo, treino,
avaliação, dados, MANIFEST, `scripts/`, `operacoes.py` e `reclin.ipynb` não foram tocados.

**Nada foi treinado, baixado do Hub nem avaliado.** Nenhum comando Git foi executado (regras do projeto).

## Estrutura

| Célula | Conteúdo |
| --- | --- |
| 1 | Configuração única, tudo preenchido (`REPOSITORIO`, `BRANCH_CODIGO`, `PASTA_NO_DRIVE`, `REVISAO_PESOS`, `BRANCH_RESULTADOS`, `PARAMETROS` do protocolo); só `GIT_NOME`, `GIT_EMAIL` e `REPOSITORIO_HF` ficam como `<PREENCHER>` |
| 2 | Clona a branch `BRANCH_CODIGO` (`codigo_colab.clonar_codigo`) e imprime o commit (HEAD) |
| 3 | Verifica Drive, GitHub e Hugging Face (`apresentacao.verificar_conexoes`) e define os contextos |
| 4 | Instala dependências se faltarem, sem flag; avisa se precisa reiniciar a sessão |
| 5 | GPU, pesos (avisa se o Hub avançou além de `REVISAO_PESOS`) e partições |
| 6a / 6b | Menu `OPERACAO_TREINO` (4/5/6), `MOTIVO` opcional, recomendação e quadro / treino |
| 7 | Métricas e curvas do DEV da execução mais recente (`apresentacao.execucao_atual`) |
| 8a / 8b / 8c | Salvar no Drive / GitHub (branch `BRANCH_RESULTADOS`) / Hugging Face |

## Removido do notebook

`SHA`, `EXECUCAO`, `AUTORIZACAO_TEST`, `CONFIRMACAO_GITHUB`, a frase de confirmação do Hugging Face,
`CONFIRMAR_INSTALACAO`, a preparação do TEST (antiga 5.1), e as operações 1, 3, 8, 13 e 14 como células próprias
(o ambiente e a instalação vão para as células 3 e 4; listar/estado/restaurar viram automáticos).

## Mantido (sem perguntar nada)

TEST nunca avaliado nem lido no notebook; tokens só por Secrets (nunca em célula, URL, log ou `.git/config`;
`GITHUB_TOKEN_ESCRITA` só na 8b, `HF_TOKEN` só na 8c, por `segredo_no_ambiente`); push só para
`BRANCH_RESULTADOS` (`main`/`master` recusadas; o merge é manual, fora do notebook); `ultimo.pt`, checkpoints,
pesos, logs e artefatos do TEST nunca vão ao GitHub (só `config.json`, `treino.json`, `experimento.json`,
`predicoes_dev.json`, `metricas.json`); retomada só de execução compatível, com a diferença mostrada.

## Decisões técnicas

- **Operação 4 e parâmetros:** `PARAMETROS` vem com o protocolo. A 4 só vale se forem iguais; se algo mudar, o
  quadro dá impedimento, a recomendação passa a ser a 6 e o aviso diz que a execução é **exploratória** (fora da fase
  científica, não elegível ao TEST). `MOTIVO` vazio grava `"sem motivo informado"`.
- **Código por branch:** a branch se move; por isso o commit clonado é registrado e a retomada continua conferindo a
  identidade do código (`sha256-manifesto-v1`), recusando execução incompatível. `clonar_codigo` reaproveita o clone
  da sessão sem atualizar.
- **Commit clonado em `experimento.json`, não em `config.json`:** `config.json` é escrito por
  `scripts/treinar.py` (científico, intocado). O HEAD vai para `experimento.json` (`codigo.revisao`, junto de
  repositório e branch); a identidade global do código já é gravada em `config.json` por `vincular_config`.
  **Desvio do pedido literal — precisa da sua decisão** se quiser o HEAD também em `config.json`.
- **Publicação no Hugging Face sem frase:** a 8c calcula a frase com `operacoes.frase_de_publicacao` e a passa
  sozinha; `operacoes.py` ficou intacto. A 8b não pede frase (`versionamento.salvar(confirmacao=None)`); se uma
  frase for passada e estiver errada, ainda recusa.
- **Aviso de revisão dos pesos:** só consulta metadados do Hub; o download continua conferindo hashes e congelando o
  commit baixado.
- **Mensagens:** referências de numeração antigas (2.1, 2.2, Fase 0…) em `apresentacao.py` e `versionamento.py`
  foram trocadas pelas células novas (6a, 6b, 1, 3, 8b).

## Arquivos

Criados: `docs/entregas/etapa-8.3.2-notebook-simplificado.md`.
Alterados: `notebooks/reclin_baseline.ipynb`, `reclin/experimento/codigo_colab.py` (`clonar_codigo`, `askpass`,
`validar_branch`), `reclin/experimento/proveniencia.py` (`origem: "git"`), `reclin/experimento/apresentacao.py`
(`verificar_conexoes`, `instalar_se_faltar`, `avisar_revisao_pesos`, `execucao_atual`, `recomendar_ramo` por
`PARAMETROS`, textos), `reclin/experimento/versionamento.py` (confirmação opcional, `askpass` compartilhado),
`reclin/experimento/__init__.py` (docstring), `testes/unidade/test_operacoes_baseline.py`,
`docs/arquitetura/infraestrutura-colab.md`, `docs/entregas/etapa-8.3.1-notebook-em-fases.md` (aviso),
`docs/entregas/README.md`.
Removidos: nenhum. `guia_de_secrets` saiu de `apresentacao.py` (função sem uso).
Não alterados: `scripts/`, `operacoes.py`, `segredos.py` (nesta etapa), `reclin.ipynb`, dados, MANIFEST, protocolo.

## Limitações e pendências

- O notebook não foi executado num Colab real (clone, instalação, streaming do treino, push e publicação só foram
  exercitados com Git e rede simulados).
- `GIT_NOME`, `GIT_EMAIL` e `REPOSITORIO_HF` precisam ser preenchidos por você; sem eles, só as células 8b/8c recusam.
- O repositório precisa ter `dados/particoes-v2/` na branch `BRANCH_CODIGO` (a célula 5 diz se faltar).
- Reprodutibilidade por branch móvel: vale a identidade do código registrada, não o nome da branch.

## Atualização — notebook alinhado ao legado (12/10/2026, mais tarde)

A pedido do autor, `reclin_baseline.ipynb` foi refeito para seguir a estrutura, a sequência e o código das células do
`01_baseline_biobertpt_colab.ipynb` do legado (11 seções), alterando só o necessário para a arquitetura atual. Esta
atualização **substitui a tabela de estrutura acima** (11 células de código com 6a/6b/8a-c); o restante da nota
(infraestrutura, `clonar_codigo`, decisões de segurança) continua valendo.

| Seção | Legado | Adaptação |
| --- | --- | --- |
| 1 | clone/pull com `GITHUB_PAT`, `git config`, `remote set-url` | igual, mas o token vai por `GIT_ASKPASS` (não entra na URL nem em `.git/config`); depois importa `codigo/` e registra o commit (HEAD) com `clonar_codigo` |
| 2 | GPU | igual |
| 3 | Drive e `CKPT_DIR` | monta o Drive; execuções espelhadas em `MyDrive/RECLin-PT/execucoes/`; informa se há execução retomável |
| 4 | `pip install -r requirements.txt` | `codigo/requirements.txt` |
| 5 | confere/gera splits | só confere `dados/particoes-v2/` (arquivos + MANIFEST); não baixa nem gera; confere e baixa os pesos na revisão fixada |
| 6 | `HF_TOKEN`, `HF_USER='angeloalsf'`, repo por seed | `REPOSITORIO_HF` removido; repo `angeloalsf/reclin-pt-biobertpt-seed42` (seed da constante do protocolo); só verifica o token, sem exportá-lo |
| 7 | `baseline_biobertpt.py ...` | `operacoes` escolhe sozinha treino novo/retomada (`recomendar_ramo`); `PARAMETROS` do protocolo; `NOVA_EXECUCAO` para exploratória |
| 8 | métricas do TEST | só DEV (`OP_DEV`) |
| 9 | curvas | código do legado, lendo `treino.json` |
| 10 | `git add -f`, commit, pull --rebase, push | `versionamento` (mesmo fluxo, 5 arquivos da execução, branch `resultados-colab`); só envia com `PUBLICAR_NO_GITHUB = True` |
| 11 | reenvio manual do `best_model` | publicação privada (`OP_PUBLICACAO`/`OP_PUBLICAR`); só envia com `PUBLICAR_NO_HF = True` |

Diferenças que dependem de decisão ou de teste real: o `treinar.py` atual não tem backup do HF por época (o legado
tinha `--hf-backup-repo`); o backup é após o treino, na seção 11. O secret de leitura/escrita do GitHub é o
`GITHUB_PAT` (o catálogo de `segredos.py` ainda lista `GITHUB_TOKEN`/`GITHUB_TOKEN_ESCRITA`; o notebook não usa esses
nomes). `test_operacoes_baseline.py` ainda descreve o notebook anterior (11 células com `#@title`) e precisa ser
adaptado; não foi executado nem alterado nesta atualização, por instrução.

### Revisão final pontual (12/10/2026)

- **Retomada × nova tentativa:** a seção 7 imprime a operação e a natureza (protocolo ou exploratória); `NOVA_EXECUCAO = True`
  com uma execução interrompida compatível exige `MOTIVO` (senão para e manda retomar). A retomada continua validada por
  `operacoes.retomar_treino` (configuração, dados, pesos e identidade do código; sem execução compatível, nada é reutilizado).
- **Execução selecionada:** a seção 8 grava `EXECUCAO_SELECIONADA` e a natureza; as seções 10 e 11 usam só ela (param se a 8 não foi
  rodada) e imprimem o nome e a natureza antes de enviar.
- **Teste estrutural:** a parte do notebook de `test_operacoes_baseline.py` foi reescrita para a estrutura atual; **não executada**.
- **Execução controlada:** `pip install` só na seção 4 e download de pesos (revisão fixada) só na segunda célula da seção 5.
  A seção 5 também consulta o Hub só para metadados (`avisar_revisao_pesos`), sem baixar.
