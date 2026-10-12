# Etapa 8.3 — notebook independente do baseline (relatório)

Nenhum comando Git foi usado. Nenhum treino, calibração, download de pesos ou
de dados, avaliação do TEST ou experimento foi executado, e **nenhum teste foi
executado** (nem a suíte, nem os testes novos). Dados, MANIFEST, referências,
protocolo, critérios e `reclin.ipynb` não foram alterados; a emenda E1
continua inativa.

## Diagnóstico do estado antes da etapa

- Já existia e foi reutilizado sem mudança de algoritmo: `scripts/treinar.py`
  (baseline, `--retomar`, `--checkpoint-a-cada`, uma opção por campo de
  `Config`), `scripts/pretreino.py` (`baixar`, `conferir --execucoes`),
  `scripts/avaliar_test.py` (não chamado pelo notebook) e a infraestrutura da
  8.2.
- Referências históricas (repositório legado): não foram necessárias.
- Divergência resolvida sem ampliar o escopo: a decisão de arquitetura 5 (TEST
  só em `reclin_teste.ipynb`) e o item 9 do pedido ("preparar avaliação final")
  são cumpridos assim: a operação 9 do baseline **só confere condições** e
  nunca chama `avaliar_test.py`; a avaliação ficará no notebook do TEST.
- Não exposto de propósito: a rodada de validação técnica 1V
  (`--validacao-epocas`), porque o protocolo a trata como passo próprio, com
  autorização específica e nomes próprios.

## Arquivos

Criados: `notebooks/reclin_baseline.ipynb`, `reclin/experimento/operacoes.py`,
`testes/unidade/test_operacoes_baseline.py` (não executado),
`docs/entregas/etapa-8.3.md`.

Alterados (infraestrutura, fora da identidade de código v1):
`reclin/experimento/raiz.py` (`rodar_script` ganhou `log` e `periodico`; a
saída aparece na célula, é gravada em arquivo sem segredos e chama o espelho
no Drive durante o treino), `reclin/experimento/persistencia.py`
(`sincronizar_e_registrar`, a lógica que era do comando `sincronizar`),
`reclin/experimento/plano.py` (`execucoes`), `scripts/experimento.py`
(`sincronizar` passou a usar a função acima), `docs/arquitetura/infraestrutura-colab.md`,
`docs/entregas/README.md`.

## Operações do notebook

Escolhidas numa lista (seção 4) e executadas na seção 5; as seções 1–3 não
rodam nada científico.

| # | Operação | Chama |
| --- | --- | --- |
| 1 | Conferir ambiente e dependências | `ambiente.verificar` (não instala) |
| 2 | Conferir GPU | `ambiente.coletar` / `classificar_gpu` |
| 3 | Identificar execuções anteriores | `persistencia.listar`, `plano.compativeis` |
| 4 | Novo treinamento (protocolo) | `scripts/treinar.py` |
| 5 | Retomar execução interrompida | `scripts/treinar.py --retomar` |
| 6 | Nova execução (outra tentativa/parâmetros) | `scripts/treinar.py` |
| 7 | Métricas do DEV | lê `treino.json` e `metricas.json["dev"]` |
| 8 | Estado e logs | `experimento.json`, `logs/` |
| 9 | Preparar avaliação do TEST | só confere; `pretreino.py conferir --execucoes` |
| 10 | Preparar publicação | `publicacao_hf.planejar` (não envia) |
| 11 | Obter pesos | local → cópia no Drive → `pretreino.py baixar` |
| 12 | Sincronizar com o Drive | `persistencia.sincronizar_e_registrar` |
| 13 | Restaurar do Drive | `persistencia.restaurar` |
| 14 | Instalar dependências | `ambiente.instalar` |
| 15 | Publicar no Hugging Face | `scripts/experimento.py publicar --executar` |

## Parâmetros configuráveis

- Fixos pela fase: `encoder` (`biobertpt`) e `seed` (42); não são parâmetros.
- `PARAMETROS` (dicionário): vazio = configuração do protocolo. Permitidos
  (os campos de `Config` que `treinar.py` expõe): `max_gap, ctx_chars,
  max_length, epochs, batch_size, lr, weight_decay, warmup_ratio,
  max_grad_norm, class_weight`. Valores fora desta lista são recusados.
  Não há busca: cada execução da operação 6 é uma configuração explícita.
- Operação 4 exige `PARAMETROS` vazio; operação 6 exige `MOTIVO` e é sempre
  uma nova tentativa. Execuções com parâmetros diferentes do protocolo são
  exploratórias: sem nome oficial e não elegíveis ao TEST.
- Operacionais (não mudam o resultado): `CHECKPOINT_A_CADA` (padrão 500, o do
  exemplo do script), `ESPELHAR_A_CADA_MIN` (padrão 10), `LINHAS_DO_LOG`.
- Obrigatórios onde usados, sem valor padrão inventado: `SHA` (commit do
  código), `REVISAO_PESOS` (commit congelado pelo protocolo),
  `REPOSITORIO_HF` (só para publicar).

## Retomada

A operação 5 recalcula a chave de configuração (estratégia, parâmetros,
revisão dos pesos, SHA-256 das partições, identidade do código) e procura uma
execução interrompida com a mesma chave no disco e no Drive. Sem
correspondência, para e lista, por execução interrompida, os campos que
diferem; nada é retomado. Com correspondência, confere ambiente (Python,
torch, transformers, tokenizers, numpy, GPU) e identidade do código, restaura
do Drive se o estado mais avançado está lá e só então chama
`treinar.py --retomar`, que ainda recusa configuração ou código diferentes.

## Preservação

Pasta própria por execução (`run_id`; nome oficial só se livre e se a
configuração for a do protocolo), `experimento.json`, espelho no Drive durante
e ao fim do treino (inclusive interrompido ou falho; arquivo temporário,
SHA-256, renomeação; nunca regride), `logs/` por sessão copiado junto,
pesos com cópia verificada no Drive, publicação privada no Hugging Face só
com a frase de confirmação exata, sem `ultimo.pt` e sem artefatos do TEST.

## Verificações realizadas (todas estáticas ou funções puras)

- JSON do notebook válido (nbformat 4.5; 12 células; ids únicos), as 5
  células de código compilam, sem saídas gravadas.
- Sem o token `git` e sem padrão de token no notebook; sem `import torch`,
  sem chamada a `avaliar_test`.
- Lista suspensa do notebook idêntica a `operacoes.OPERACOES`; campos
  permitidos documentados idênticos aos de `Config`.
- Todas as flags passadas a `treinar.py`, `pretreino.py` e `experimento.py`
  existem nesses scripts (leitura do código-fonte).
- Chamadas das funções puras `config_da_operacao` (recusa `seed`, `encoder`,
  campo desconhecido e valor inválido) e `argumentos_de_treino`.
- Análise por AST do módulo novo: sem nomes indefinidos nem imports sem uso.

**Não verificado:** o fluxo de treino, retomada, espelhamento, log, pesos e
publicação (dependem de Colab, GPU, torch, rede e Hugging Face); o teste
escrito (`test_operacoes_baseline.py`) e a suíte.

## Riscos e pontos a saber

- A identidade de código ainda é a global v1: qualquer edição em
  `reclin/**/*.py`, `scripts/*.py` (exceto os excluídos) ou `pyproject.toml`
  muda o `config_id`, e uma execução interrompida deixa de ser retomável com o
  código novo (a operação 5 mostra `codigo.identidade` como diferença). Isso é
  a regra P1 das decisões da 8.2; a identidade por componente é a etapa de
  identidade.
- `RAIZ_LOCAL` é `/content/reclin_resultados/execucoes`, não a pasta padrão de
  `codigo/resultados/execucoes`; os notebooks de resultados e do TEST devem
  usar a mesma raiz (ou o Drive).
- O tarball do código inclui `test.jsonl` (as partições são versionadas); o
  treino não o lê.

## Falta para iniciar o treino real (nada de código)

1. Commit/versionamento manual destas alterações para obter o `SHA`.
2. `REVISAO_PESOS`: o commit de `pucpr/biobertpt-all` que o protocolo manda
   comunicar antes do primeiro treino.
3. Autorização específica do passo da sequência (protocolo, seção 6).
