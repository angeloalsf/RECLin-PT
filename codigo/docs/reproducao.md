# Reprodução das análises e dos relatórios (etapa 7)

Como gerar, a partir do código novo, as análises do corpus, as tabelas, as
curvas, os valores citados no TCC e o relatório da fase, e como conferir o
texto do TCC contra eles. Os mesmos comandos estão no `codigo/Makefile`
(`make -C codigo help`) e no notebook único (`codigo/notebooks/reclin.ipynb`).

Nada aqui treina modelos nem avalia o TEST pela primeira vez. Os experimentos
da fase científica (etapa 8) têm comandos próprios, cada um com autorização
explícita (protocolo `docs/protocolo/fase-biobertpt-seed42.md`, seção 6).

## Origem dos dados e dos resultados

| Entrada | De onde vem | Como é conferida |
| --- | --- | --- |
| Partições | `codigo/dados/particoes-v2/` (versão `v2-texto-fonte`, a da fase; congeladas). A v1 (`codigo/dados/particoes/`) fica para a regressão técnica | SHA-256 de cada arquivo contra o `MANIFEST.json` antes de qualquer conta; a versão é reconhecida pelo SHA-256 (`reclin.versao_dados`, seção abaixo) |
| Execuções | `codigo/resultados/execucoes/<nome>/`, gravadas pelo código novo | `reclin.execucao.inventario`: `config.json` obrigatório; cada sidecar contra o conjunto de referência registrado (número de candidatos, `y_true` pelo hash, `max_gap`) e a partição pelo SHA-256 do MANIFEST; `metricas.json` igual ao recalculado; TEST só com a trilha de avaliação; treino concluído; dependências presentes e válidas, com a mesma identidade do código; filtro com o SHA-256 das predições da base em cada sidecar; nenhuma configuração com arquivo de calibração externo |
| Código | identidade do código gravada em cada execução (`config.json` → `codigo`; também nas sessões do treino e na trilha do TEST) | `reclin/util/identidade_codigo.py` (abaixo) |
| Pesos | `PROVENIENCIA.json` gravado por `scripts/pretreino.py baixar` | o commit e o hash do manifesto, registrados em cada execução, entram na tabela de proveniência |

O inventário **recusa**, sem opção de aceitar: qualquer caminho dentro de
`codigo/testes/referencia/` (as referências do legado, só para os testes de
equivalência), arquivos com os nomes ou o formato dos resultados do legado
(`*.preds.json`, `*.dev_preds.json`, `*.test_evals.jsonl`,
`significance_*.json`, `CALIBRACAO_filtro.*`, `FASE2_test_summary.json`,
`summary_by_seed.json`), arquivos soltos e pastas sem `config.json`.

Cada execução recebe uma **natureza**, e naturezas nunca se misturam numa
tabela nem num destino:

| Natureza | Quando | Uso |
| --- | --- | --- |
| `cientifica` | seed 42, BioBERTpt, proveniência dos pesos registrada, partições congeladas da versão dos dados da fase (v2), identidade do código igual à de referência (por padrão, a do código atual; `--codigo-referencia` em `gerar_tcc.py`), sem marca de validação | resultados da fase ("validação inicial, seed 42") |
| `validacao` | execução da rodada de validação técnica (passo 1V), própria ou herdada | só verifica o pipeline; não avalia desempenho |
| `fora_da_fase` | válida, mas fora dessas condições (modelo minúsculo, outra seed, outras partições, outra versão do código, sem a identidade do código) | testes técnicos; nunca é resultado do TCC |

## Versões dos dados (D9)

| Versão | Pasta | Origem | Uso |
| --- | --- | --- | --- |
| `v1-conversao-legado` | `dados/particoes/` | conversão do legado (texto normalizado, offsets da fonte: desalinhamento do D9) | regressão técnica (`testes/equivalencia/`) e histórico; nunca entrada dos experimentos novos |
| `v2-texto-fonte` | `dados/particoes-v2/` | `reclin/corpus` a partir dos XML (regra C1) | a fase científica (`caminhos.PARTICOES`, padrão de todos os scripts) |

- **Geração e conferência** (sem treino; TEST só estrutural):
  `python codigo/scripts/reconstruir_dados.py inventario | gerar --data AAAA-MM-DD | conferir | auditar | comparar`
  (`make inventario-xml`, `reconstruir-v2 DATA=…`, `conferir-v2`, `auditar-v2`,
  `comparar-v2`). Os XML (`codigo/SemClinBr-xml-public-v1/`, ou `--xml`) são
  só lidos; o dataset e a representação intermediária vão para
  `dados/processados/` (fora do git).
- **Identificação:** cada execução registra o SHA-256 das partições
  (`config.json`: `config.particoes` do treino e `conjuntos.*.particao_sha256`)
  e, legível, `versoes_dados`; `reclin.versao_dados.REGISTRO` congela os
  SHA-256 das duas versões.
- **Guardas:** `diretorio.gravar_predicoes` recusa predições de outra versão
  registrada que a do treino ou dos conjuntos já gravados; a avaliação do TEST
  (classificador e Two-Stage) e o filtro recusam partições de outra versão; o
  inventário marca como problema a execução que mistura versões ou a derivada
  de uma base de outra versão, classifica como fora da fase o que não é da v2
  e não seleciona versões diferentes numa tabela.

## Identidade do código (D6)

`reclin/util/identidade_codigo.py` calcula, sem Git, um SHA-256 determinístico
do código que pode mudar o que uma execução grava:

- **Incluídos:** `reclin/**/*.py`, `scripts/*.py` e `pyproject.toml`
  (relativos a `codigo/`).
- **Excluídos:** o que só lê execuções — `reclin/analises/`,
  `reclin/relatorios/`, `reclin/execucao/inventario.py`,
  `reclin/util/formatacao.py`, `scripts/analisar.py`, `scripts/gerar_tcc.py`,
  `scripts/conferir_tcc.py` —, e nunca entram testes, documentação, notebooks,
  `Makefile`, partições, resultados, relatórios gerados, checkpoints, pesos,
  caches e temporários.
- **Algoritmo `sha256-manifesto-v1`:** SHA-256 de cada arquivo, com `\r\n`
  trocado por `\n` (a mesma identidade em Windows e Linux); manifesto
  `{caminho: sha256}` em ordem; identidade = SHA-256 do JSON do manifesto
  (chaves ordenadas, separadores `(",", ":")`).
- **Registro:** `diretorio.criar` grava `{algoritmo, identidade, n_arquivos,
  arquivos}` no `config.json`; cada sessão do treino e cada linha da trilha do
  TEST gravam a identidade.
- **Verificação:** retomar o treino, avaliar o TEST, aplicar um limiar ao TEST,
  compor o Two-Stage, calibrar o baseline ou aplicar o filtro sobre uma
  execução de outra versão do código (ou sem a identidade) é recusado. O
  inventário confere a consistência do registro (identidade recalculada do
  manifesto gravado), das sessões e da trilha, das dependências, e classifica
  como fora da fase o que não tem a identidade de referência. Para ver a
  identidade atual: `python -c "from reclin.util import identidade_codigo as i;
  print(i.atual()['identidade'])"`; o inventário (`gerar_tcc.py inventario`)
  mostra a de cada execução.
- **Execuções antigas, sem a identidade:** não são completadas depois nem
  aceitas como equivalentes às novas; ficam fora da fase e não recebem
  artefatos novos (refazê-las com o código atual é o caminho).

## Comandos

```bash
# análises do corpus (TRAIN e DEV; o TEST só com --incluir-test, como descrição)
python codigo/scripts/analisar.py corpus --saida codigo/resultados/analises/corpus.json
python codigo/scripts/analisar.py max-gap --saida codigo/resultados/analises/max_gap.json
python codigo/scripts/analisar.py alinhamento --saida codigo/resultados/analises/alinhamento.json
python codigo/scripts/analisar.py marcadores [--modelo PASTA_DO_PRETREINO] --saida codigo/resultados/analises/marcadores.json

# filtro léxico da fase (léxico congelado; o TEST só depois do TEST da base, uma vez, com trilha)
python codigo/scripts/filtro_pistas.py --base-execucao codigo/resultados/execucoes/baseline_biobertpt_seed42 --particao dev
python codigo/scripts/filtro_pistas.py --base-execucao codigo/resultados/execucoes/baseline_biobertpt_seed42 --particao test

# falsos positivos de negation_of de uma execução verificada (DEV; TEST só se já avaliado)
python codigo/scripts/analisar.py falsos-positivos --execucao baseline_biobertpt_seed42 \
    --saida codigo/resultados/analises/fp_baseline_dev.json

# inventário: o que existe, a natureza e a proveniência de cada execução
python codigo/scripts/gerar_tcc.py inventario

# tudo: EDA, resultados, comparações, limiares, proveniência, curvas e relatório
python codigo/scripts/gerar_tcc.py tudo                         # natureza cientifica
python codigo/scripts/gerar_tcc.py tudo --natureza validacao    # rodada de validação técnica

# conferência do texto do TCC (pendente enquanto a pasta tcc/ não existir)
python codigo/scripts/conferir_tcc.py --afirmacoes tcc/afirmacoes.json
```

## O que é gravado

Em `codigo/resultados/relatorios/<natureza>/` (ou `--destino`):

| Caminho | Conteúdo |
| --- | --- |
| `inventario.json` | execuções, natureza, partições utilizáveis, problemas, avisos e itens recusados |
| `tabelas/<nome>.tex` | `tabular` com `booktabs` e vírgula decimal (ABNT), com cabeçalho de comentário: aviso de arquivo gerado, natureza, status e origem |
| `tabelas/<nome>.csv` / `.md` | os mesmos valores, completos (CSV, ponto decimal) e para leitura (Markdown) |
| `valores/<grupo>.json` / `.tex` | registro de valores (chave → valor, texto ABNT, tabela, natureza, execuções de origem); no LaTeX, `\rv{test.two_stage.f1_negation_of}` |
| `indice/<grupo>.json` | status (`completa`, `parcial`, `pendente`), pendências, notas e proveniência de cada tabela |
| `comparacoes/<A>__vs__<B>.json` | relatório completo de cada comparação do protocolo (bootstrap pareado e McNemar) |
| `analises/` | estatísticas do corpus e sensibilidade ao `max_gap` usadas na EDA |
| `figuras/` | PDF e PNG (só com o matplotlib instalado; sem ele, as figuras ficam pendentes no índice) |
| `relatorio.md` | relatório da fase: escopo, inventário, tabelas, pendências |

Tabelas geradas: `resultados_test`, `diagnostico_dev`, `comparacoes_test`,
`limiares_dev`, `proveniencia`, `particoes`, `curva_<execução>` e as da EDA
(`eda_corpus`, `eda_cobertura`, `eda_distancias_train`, `eda_entidades_train`,
`eda_proporcao_rotulos`, `eda_max_gap_train_dev`).

## Regras que as rotinas aplicam

- Os números são recalculados das predições (`avaliacao.metricas`,
  `avaliacao.significancia`), não lidos de resumos.
- O TEST só entra quando a execução tem a trilha de avaliação
  (`avaliacoes_test.jsonl`): as tabelas nunca avaliam um TEST pela primeira
  vez. Reavaliações registradas na trilha aparecem como aviso.
- Um sistema sem execução, sem a partição avaliada ou só com execução de outra
  natureza aparece como `pendente`, com o motivo; uma execução com problemas
  de proveniência interrompe a geração.
- No restrito e na Pair-Aware, macro-F1, MCC e McNemar aparecem como `n/c`
  (não comparáveis com o baseline); só o F1 de `negation_of` se compara.
- Os números do DEV do baseline calibrado e do Two-Stage são marcados como
  otimistas (o limiar foi escolhido nesse DEV).
- `conferir_tcc.py` também confere que as execuções de onde vieram os valores
  não mudaram depois da geração (SHA-256 dos sidecars e do `config.json`,
  número de avaliações do TEST e natureza).
