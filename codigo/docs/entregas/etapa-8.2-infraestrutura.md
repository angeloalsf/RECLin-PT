# Etapa 8.2 — infraestrutura compartilhada para Colab (relatório)

Nenhum comando Git foi usado. Nenhum treino, calibração, download de pesos ou
de dados, avaliação do TEST ou experimento foi executado. A suíte completa de
testes **não** foi executada. Dados, MANIFEST, referências, protocolo, critérios
e `reclin.ipynb` não foram alterados; a emenda E1 continua inativa.

## Arquivos

Criados: `reclin/experimento/{__init__,codigo_colab,raiz,colab,segredos,ambiente,ids,proveniencia,persistencia,publicacao_hf,identidade,plano}.py`,
`scripts/experimento.py`, `testes/unidade/test_experimento.py`,
`testes/unidade/test_experimento_plano.py`, `docs/arquitetura/infraestrutura-colab.md`, este relatório.

Alterados: `reclin/util/identidade_codigo.py` (a infraestrutura sai da
identidade v1: `reclin/experimento/` e `scripts/experimento.py` excluídos),
`testes/unidade/test_identidade_codigo.py` (casos novos de exclusão),
`docs/arquitetura/identidade-por-componente.md` (seção 11),
`docs/entregas/README.md` (linha desta etapa).

Residual: `reclin/experimento/__pycache__/codigo_colab.cpython-310.pyc`
(inofensivo, fora das identidades; pode ser apagado manualmente).

## Ajustes desta rodada

1. **Identidade.** A infraestrutura não pertence a nenhum componente de treino
   (nada fora dela a importa; teste estático). Por isso saiu do hash global v1,
   que não muda de algoritmo nem de regra de bytes. Tem identidade própria
   (`identidade.py`), gravada em `experimento.json`, só informativa. A
   identidade `v2` por componente continua na 8.3.
2. **Configuração e execuções** (`plano.py`): chave determinística a partir da
   configuração efetiva, pesos (`PROVENIENCIA.json`), versão e SHA-256 das
   partições (`train`, `dev`; TEST fora) e identidade do código; `run_id` por
   tentativa; `NOMES_OFICIAIS` (os 4 do protocolo, conferidos por teste contra o
   texto do protocolo); `planejar` (dry-run: retomar, iniciar planejada, já
   concluída, nova tentativa) e `preparar`; comando `experimento.py planejar`.
   A comparação usa a chave gravada, nunca caminho de sessão nem nome de pasta.
3. **Hugging Face.** O repositório de artefatos é argumento explícito
   (`--repositorio`), exigido só para publicar; é criado privado apenas se não
   existir.

## Validação executada (resultados)

Ambiente da máquina: Python 3.10.12, numpy 2.2.6; **sem** torch, scipy,
transformers nem huggingface_hub. `pytest 9.1.1` foi instalado do PyPI num
ambiente virtual fora da pasta do projeto (`--system-site-packages`); nenhum
arquivo foi criado na pasta (sem cache, `PYTHONDONTWRITEBYTECODE=1`).

| Verificação | Resultado |
| --- | --- |
| Importar os 11 módulos de `reclin.experimento` em interpretador novo e pasta vazia | OK; nenhum arquivo criado |
| `scripts/experimento.py --help` e `--help` dos 9 subcomandos | todos saída 0 |
| `test_experimento.py` + `test_experimento_plano.py` (infraestrutura, sem rede; Hugging Face simulado) | **51 passaram** |
| Esses dois + `test_identidade_codigo.py` + `test_pacote.py` (sem os testes que leem dados/referências: arquivos obrigatórios, partições protegidas, referências) | 179 passaram, 4 falharam, 24 não selecionados |

As 4 falhas são `ModuleNotFoundError: torch` neste ambiente, em módulos que não
alterei: `test_todos_os_modulos_importam_sem_efeito_colateral` e
`test_scripts_respondem_a_ajuda` de `avaliar_test.py`, `pretreino.py` e
`treinar.py` (importam `pair_aware`, que usa torch). Não foram investigadas
nem corrigidas. Passaram: `--help` de `experimento.py` e o teste "um nível só
importa os de cima" para todos os módulos de `reclin/experimento/`. (Uma
primeira execução sem `PYTHONPATH` falhou em todos os scripts porque o
pacote não está instalado no ambiente virtual; foi repetida com
`PYTHONPATH=codigo`.)

## Não verificado

- Colab real (`google.colab`, Secrets, montagem do Drive), Hugging Face real e
  GitHub real: os testes usam diretórios temporários e um objeto Hugging Face
  simulado.
- Suíte completa; `test_entradas_e_legado.py`; qualquer teste com torch.
- Equivalência do hash v1 com o das execuções antigas: raciocínio (só dois
  caminhos novos excluídos), não comparação com um `config.json` real.

## Critérios

Atendidos (verificados pelos testes acima): identidade fora do hash global,
`config_id`/`run_id`, nomes oficiais, detecção de execução compatível sem
caminhos de sessão, plano sem escrita, importação e `--help`. Pendentes: tudo
que exige Colab, GPU, torch ou rede.
