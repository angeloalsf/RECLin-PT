# Etapa 8.3.1 — notebook do baseline em fases (relatório)

> **Substituída em parte pela etapa 8.3.2** ([etapa-8.3.2-notebook-simplificado.md](etapa-8.3.2-notebook-simplificado.md)): o notebook foi simplificado (11 células de código, sem SHA/EXECUCAO/autorizações/frases). A estrutura em fases abaixo é histórica; `apresentacao.py`, `versionamento.py` e `segredos.py` seguem em uso.

Reorganiza `notebooks/reclin_baseline.ipynb` (etapa 8.3) em **fases**: a ordem obrigatória fica em células
sequenciais e o menu só aparece onde há decisão (treino novo, retomada ou nova tentativa). Nenhuma lógica
científica foi duplicada: o notebook orquestra `reclin/experimento/operacoes.py` e os scripts reais.

**Nada foi treinado, calibrado, baixado ou avaliado. Os testes desta etapa foram escritos e não foram
executados.** Também não foi usado Git (regras do projeto).

## Perguntas anteriores à escrita

1. **As operações validam os pré-requisitos sozinhas?** Sim, nos pontos que causam dano: pesos ausentes (4, 5, 6),
   versão das partições, Drive, GPU/ambiente, `MOTIVO`/`PARAMETROS`, execução duplicada ou concluída, compatibilidade
   da retomada. Não validam: dependências ausentes (aparecem tarde, como erro de import), existência das partições
   com contagem por split e a recomendação de ramo. As células da Fase 1 cobrem esses três pontos.
2. **Arquivos gerados por uma execução** (`/content/reclin_resultados/execucoes/<nome>/`): `config.json`,
   `treino.json`, `experimento.json`, `predicoes_dev.json`, `metricas.json` (`dev`; `test` só após o TEST),
   `checkpoints/ultimo.pt` e `ultimo.json`, `checkpoints/melhor_modelo/`, `logs/treino-*.log`; no Drive, `SYNC.json`;
   do TEST (nunca aqui): `predicoes_test.json` e `avaliacoes_test.jsonl`.
3. **Streaming do treino:** já existia (`raiz.rodar_script` com `Popen`, leitura linha a linha, `PYTHONUNBUFFERED=1`,
   log em arquivo, espelhamento periódico com `[espelho] …`). Nenhum script científico foi alterado. Não foi
   verificado em um Colab real.

## Estrutura do notebook

| Fase | Células |
| --- | --- |
| 0. Configuração e código | 0.1 parâmetros (inclui `GIT_NOME`, `GIT_EMAIL`, `BRANCH_RESULTADOS`); 0.2 código por SHA + Drive; 0.3 semente da fase (constante do protocolo, `assert == 42`) e guia de Secrets (não lê token) |
| 1. Preparação | 1.1 ambiente (op 1); 1.2 instalar (op 14, só com confirmação); 1.3 GPU (op 2); 1.4 pesos (op 11); 1.5 partições; 1.6 execuções anteriores (op 3) + recomendação 4/5/6 |
| 2. Treino | 2.1 menu 4/5/6, `PARAMETROS`, `MOTIVO`, quadro pré-treino; 2.2 treinar (recusa se o quadro não foi visto, mudou ou tem impedimentos) |
| 3. Resultados (só DEV) | 3.0 execução; 3.1 métricas (op 7); 3.2 estado e logs (op 8); 3.3 curvas |
| 4. Preservação | 4.1 sincronizar (op 12); 4.2 restaurar (op 13) |
| 5. Fechamento | 5.1 preparar TEST (op 9, não avalia); 5.2 preparar publicação (op 10); 5.3 publicar (op 15, `HF_TOKEN` só aqui) |
| 6. Salvar no GitHub | 6.1 conferir; 6.2 commit + push |

## Decisões técnicas

- **Partições (1.5):** são versionadas e vêm no tarball do SHA; a célula confere (`preparar_dados.py conferir`) e
  imprime os documentos por split (do MANIFEST). Se faltarem, diz que o **SHA informado não contém as partições**.
  Não há geração a partir de XML (o SemClinBr não é distribuído e `parse_semclinbr.py`/`make_splits.py` não existem
  no projeto atual).
- **Semente (0.3):** vem da constante do protocolo (`SEMENTE_DA_FASE`), conferida contra o sufixo `seedNN` da pasta
  oficial; não é lida de `PASTA_NO_DRIVE`.
- **Quadro e recomendação:** em `apresentacao.py`, sem alterar `operacoes.py`. O quadro é só leitura
  (`plano.planejar`); a 2.2 compara a assinatura das escolhas com a do quadro.
- **Segredos:** `HF_TOKEN` é lido e exportado só na 5.3 (`apresentacao.segredo_no_ambiente`, `os.environ.pop` ao sair,
  também em erro). `segredos.obter` passou a aceitar a variável de ambiente quando o `userdata` não pode ser lido
  (subprocesso). Novo Secret `GITHUB_TOKEN_ESCRITA`, separado do `GITHUB_TOKEN` de leitura.
- **Fase 6 (`versionamento.py`):** clone de trabalho **raso e separado** (`/content/reclin_versionamento`), porque o
  código vem de tarball sem Git e não pode ser misturado com o clone. Os resultados vão para uma **branch própria**
  (`BRANCH_RESULTADOS`, criada a partir do SHA do código se não existir; `main`/`master` recusadas). O token vai por
  `GIT_ASKPASS` (script temporário, 700, apagado em `finally`) e existe só no ambiente do subprocesso do Git: não vai
  para a URL do remote, `.git/config`, argv nem `os.environ`. A identidade do commit vai por `-c`, sem gravar config.
- **Destino no repositório:** `codigo/resultados/execucoes/<nome>/` (convenção de `docs/reproducao.md`).
- **Arquivos versionados:** `config.json`, `treino.json`, `experimento.json`, `predicoes_dev.json`, `metricas.json`,
  por lista explícita com `git add -f`. Nunca `ultimo.pt`/checkpoints, pesos, logs, `SYNC.json` nem artefatos do TEST.
  Recusas: execução não concluída, `metricas.json` com `test`, `predicoes_dev.json` com texto, segredo em arquivo,
  arquivo diferente do que já está no clone. Reexecutar sem mudança não cria commit.
- **`predicoes_dev.json`:** o sidecar guarda só `y_true`, `y_pred`, `probs`, `labels` e metadados numéricos; não há
  texto clínico (conferido no código de `predicoes.montar`/`_predicoes_dev` e em um sidecar de referência do legado).
  `versionamento.sem_texto_clinico` recusa o arquivo se isso mudar.
- **Confirmação extra (não pedida):** a 6.2 exige a frase `SALVAR <nome> EM <repo>@<branch>`, como a 5.3, porque o
  push é visível fora da sessão.

## Arquivos

Criados: `reclin/experimento/apresentacao.py`, `reclin/experimento/versionamento.py`,
`docs/entregas/etapa-8.3.1-notebook-em-fases.md`.
Alterados: `notebooks/reclin_baseline.ipynb` (reescrito; 31 células, sem outputs), `reclin/experimento/segredos.py`,
`testes/unidade/test_operacoes_baseline.py`, `docs/arquitetura/infraestrutura-colab.md`, `docs/entregas/README.md`.
Não alterados: `scripts/`, `reclin/experimento/operacoes.py`, `notebooks/reclin.ipynb`, dados, MANIFEST, protocolo.

## Verificações

Executadas: `ast.parse` de todas as células do notebook e dos módulos e testes alterados; verificação de imports sem uso;
importação de `apresentacao`, `versionamento` e `segredos` (smoke) e chamada de `semente_da_fase()` e
`frase_de_salvamento()`. **Não executadas:** `test_operacoes_baseline.py` e o restante da suíte; o notebook num Colab
real; qualquer clone, commit ou push (o Git é simulado nos testes escritos).
