# Entrega da etapa 2 — execução e avaliação

| | |
| --- | --- |
| Etapa | 2 |
| Versão | 1 |
| Data | 09/10/2026 |
| Pacote | `RECLin-PT-etapa-2-v1.zip` |
| Parte de | etapa 1, versão 2 (`RECLin-PT-etapa-1-v2.zip`) |

O pacote é o **projeto completo** até a etapa 2: extraído numa pasta limpa,
tem tudo o que é preciso para instalar, executar e testar o que já foi
implementado.

## Resultado

| Critério de aceitação | Resultado |
| --- | --- |
| Métricas dos sidecars do legado iguais às registradas | Atendido. Todas as métricas registradas de todos os 24 sidecars versionados foram recalculadas e conferem com **igualdade exata**. A única exceção, explicada abaixo, é o F1 de 3 dos 9 sistemas do `FASE2_test_summary.json`, gravado pelo legado com outra fórmula; ali se confere que a fórmula do legado, aplicada às mesmas contagens, dá exatamente o valor gravado |
| 26 comparações reproduzidas | Atendido. **26 de 26** relatórios recalculados saem **idênticos byte a byte** aos gravados pelo legado, em Python 3.10, 3.12 e 3.13 e também pelo script `comparar.py protocolo` |

## Como o legado executava e avaliava

Confirmado nos arquivos do commit `a5f055c`:

- **Treinos** (`relation_extraction.run`, `train_restrito.run`,
  `train_pair_aware.run`) gravavam, para cada execução, `<nome>.json` (métricas
  do TEST calculadas com o scikit-learn, `dev_history` por época, configuração e
  ambiente), `<nome>.preds.json` (sidecar do TEST), `<nome>.dev_preds.json`
  (sidecar do DEV na melhor época) e `<nome>.test_evals.jsonl` (trilha das
  avaliações do TEST).
- **Fase 2** (`scripts/run_fase2_test.py`): aplicava o filtro e a regra pura
  aos sidecars, gravava `filtro_*.preds.json` e `regra_pura.preds.json`, o
  `FASE2_test_summary.json` (com fórmulas manuais) e uma linha em cada trilha.
- **Comparações** (`src/significance.py`): chamadas pelo `Makefile` (fase 1)
  e por `scripts/run_fase2_significance.py` (fase 2, por subprocesso),
  gravando `significance_*.json`.
- **Agregação** (`scripts/aggregate_seeds.py`): `summary_by_seed.json`, com
  média e desvio-padrão populacional entre as sementes 42 e 43.
- **Ambiente**: todas as execuções registradas em `results/` usaram Python
  3.13.15 no Colab, numpy 2.1.3, scipy 1.16.3 e scikit-learn 1.6.1 (campo
  `environment`). O README antigo dizia Python 3.10; o README da raiz foi
  corrigido.

## O que foi implementado

### `reclin/execucao/` — o formato das execuções

| Módulo | Responsabilidade |
| --- | --- |
| `predicoes.py` | O sidecar do legado, sem mudança de formato: `montar` (mesma ordem de chaves e arredondamento de `probs` a 4 casas), `ler`, `validar`, `gravar` e `conferir_conjunto` (o `y_true` tem de ser o do conjunto de referência da etapa 1) |
| `diretorio.py` | O diretório de uma execução (`config.json`, `predicoes_dev.json`, `predicoes_test.json`, `metricas.json`, `avaliacoes_test.jsonl`). As predições de uma partição são conferidas contra o conjunto de referência antes de gravadas e não são regravadas sem `sobrescrever=True`. `caminho_predicoes` encontra as predições também no formato do legado |
| `trilha.py` | A trilha das avaliações do TEST, com `eval_index`, `eval_index_for_config` e o `config_sha1` na receita do legado. Lê as linhas das três implementações antigas |

Não calcula métricas nem treina. Os nomes das execuções são os do legado
(`baseline_biobertpt_seed42`, `regra_pura`...).

### `reclin/avaliacao/` — métricas, significância e agregação

| Módulo | Responsabilidade |
| --- | --- |
| `metricas.py` | A única implementação de P/R/F1 por classe, macro, micro e weighted-F1, MCC, relatório por classe (formato do `classification_report` do sklearn) e matriz de confusão, a partir das contagens e com as mesmas operações em ponto flutuante do scikit-learn 1.6.1. Não depende do sklearn |
| `significancia.py` | McNemar exato (`scipy.stats.binomtest`) e bootstrap pareado no F1 de `negation_of`, com a mesma sequência de sorteios do legado (`default_rng(seed)`, uma chamada `integers(0, n, n)` por reamostra) e o relatório no formato do legado |
| `protocolo.py` | As 26 comparações do TCC, na ordem e com os nomes do legado, e a regra de semente do bootstrap (a de A; a de B quando A é a regra pura) |
| `agregacao.py` | Média e desvio-padrão populacional entre sementes (`statistics`, como no legado) |

### Scripts

| Script | Uso |
| --- | --- |
| `scripts/avaliar.py` | Métricas de sidecars avulsos ou das partições de uma execução (grava `metricas.json`) |
| `scripts/comparar.py` | `par`: uma comparação entre dois sidecars. `protocolo`: as 26 comparações sobre uma pasta de resultados (formato novo ou do legado), com `--conferir` byte a byte contra uma pasta de referência |

### Referências do legado

`codigo/testes/referencia/resultados_legado/`: 72 arquivos copiados sem
alteração de `results/` do legado (10,5 MB; cerca de 1 MB no ZIP), na mesma
estrutura de pastas, com o SHA-256 de cada um em `referencias.json`:

- 4 baselines: `.json`, `.preds.json`, `.dev_preds.json`, `.test_evals.jsonl`;
- 4 `filtro_*.preds.json`, `regra_pura.preds.json` e sua trilha,
  `FASE2_test_summary.json`;
- os 26 `significance_*.json` e o `summary_by_seed.json`;
- fine-tuning restrito, sementes 42 a 46: `.json`, `.preds.json`,
  `.dev_preds.json`, `.test_evals.jsonl`;
- Pair-Aware semente 42: `.json` e `.dev_preds.json` (o TEST não foi avaliado).

Foram copiados por uma nova parte do gerador (`gerar_referencias.py --partes
resultados`), rodada a partir de um checkout separado do repositório original no
commit `a5f055c`; a cópia confere o SHA-256 de cada arquivo com o original. Ficam
de fora os `archive_*` (rodadas substituídas), as 7 comparações de DEV dos
critérios de parada e os `filtro_dev/`, que pertencem às etapas das estratégias.

### Fica para as etapas que os usam

| Item do plano | Etapa | Motivo |
| --- | --- | --- |
| Subconjunto e remapeamento genéricos em `execucao` | 6 | Só o restrito e a Pair-Aware os usam, e é lá que podem ser validados contra as referências de treino |
| `avaliacao/decisao` (critérios de decisão) | 4 e 6 | Só os critérios das estratégias os usam |
| Comparações de DEV dos critérios de parada | 4 e 6 | Idem |
| Avaliação do TEST como comando separado (com registro na trilha) | 5 | Depende do treino |
| Rótulos de apresentação ("BioBERTpt (clínico)") do `summary_by_seed.json` | 7 | São de relatório; aqui se conferem os números |

Diferença de forma em relação ao plano: `execucao` virou um pacote com três
módulos (era um arquivo único no plano), como pedido para esta etapa.

## Evidências

### Critério 1 — métricas

`testes/equivalencia/test_metricas_legado.py`, 71 testes. Igualdade exata de
floats, salvo onde indicado.

| Verificação | Abrangência | Campos conferidos | Resultado |
| --- | --- | --- | --- |
| Sidecar pertence ao conjunto de referência | 24 sidecars (14 do TEST, 10 do DEV) | formato e `y_true` (hash da etapa 1) | 24/24 |
| Métricas do TEST | 9 execuções treinadas (4 baselines, 5 restritos) | macro, micro e weighted-F1, MCC, F1 por classe, `sklearn_report` inteiro, matriz de confusão, `n_candidates` | 9/9 |
| Métricas do DEV na melhor época | 10 (as 9 + Pair-Aware) | melhor época pelo critério do legado, `dev_macro_f1`, `dev_negation_of_f1`, campos "recomputados" do sidecar; no restrito e na Pair-Aware, as métricas do subespaço (`dev_restrito_*`); na Pair-Aware, `dev_best_*` | 10/10 |
| Métricas do subespaço restrito no TEST | 5 restritos | `test_restrito`: macro-F1, F1 por classe, matriz de confusão | 5/5 |
| Resumo da fase 2 | 9 sistemas do `FASE2_test_summary.json` | TP, FP, FN, precisão, recall e macro-F1 exatos; F1 pela fórmula do legado (divergência 1) | 9/9 |
| Trilhas `test_evals` | 10 arquivos, 15 linhas | `n_test`, macro-F1 e F1 de `negation_of` (gravados com 6 casas: compara-se o recalculado arredondado a 6 casas) e `config_sha1` das linhas da fase 2 | 15/15 |
| Agregação entre sementes | `summary_by_seed.json`: 2 modelos × 4 métricas | média, desvio populacional, n e valores por semente, a partir das métricas recalculadas | 8/8 |

### Critério 2 — as 26 comparações

`testes/equivalencia/test_comparacoes_legado.py`: confere que o protocolo tem
exatamente os 26 relatórios do legado, que a semente de cada comparação é a das
execuções, e recalcula cada uma comparando os bytes do relatório com os do
arquivo gravado. Também rodado pelo script, com o mesmo resultado (26 de 26
idênticos).

F1 = F1 de `negation_of` no TEST; IC95 e p(boot) do bootstrap pareado com
10.000 reamostras; p(McNemar) do teste exato. "Sig." = o IC95 exclui o zero.
Nos nomes, "biobertpt s42" é o baseline BioBERTpt semente 42.

| # | A × B | Semente | F1 A | F1 B | A − B | IC95 | p (boot) | p (McNemar) | Sig. | Reprodução |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | biobertpt s42 × bertimbau s42 | 42 | 0.6961 | 0.7110 | -0.0149 | [-0.0509; +0.0209] | 0.4204 | 4.79e-01 | não | idêntico |
| 2 | biobertpt s43 × bertimbau s43 | 43 | 0.7388 | 0.7183 | +0.0204 | [-0.0176; +0.0575] | 0.2878 | 1.42e-01 | não | idêntico |
| 3 | filtro biobertpt s42 × biobertpt s42 | 42 | 0.8299 | 0.6961 | +0.1338 | [+0.0998; +0.1700] | 0.0000 | 1.72e-14 | sim | idêntico |
| 4 | filtro biobertpt s42 × bertimbau s42 | 42 | 0.8299 | 0.7110 | +0.1189 | [+0.0789; +0.1613] | 0.0000 | 1.63e-02 | sim | idêntico |
| 5 | filtro biobertpt s42 × biobertpt s43 | 42 | 0.8299 | 0.7388 | +0.0911 | [+0.0473; +0.1356] | 0.0000 | 2.23e-03 | sim | idêntico |
| 6 | filtro biobertpt s42 × bertimbau s43 | 42 | 0.8299 | 0.7183 | +0.1115 | [+0.0691; +0.1562] | 0.0000 | 1.92e-01 | sim | idêntico |
| 7 | filtro bertimbau s42 × biobertpt s42 | 42 | 0.8168 | 0.6961 | +0.1207 | [+0.0780; +0.1647] | 0.0000 | 5.23e-01 | sim | idêntico |
| 8 | filtro bertimbau s42 × bertimbau s42 | 42 | 0.8168 | 0.7110 | +0.1058 | [+0.0749; +0.1399] | 0.0000 | 1.04e-11 | sim | idêntico |
| 9 | filtro bertimbau s42 × biobertpt s43 | 42 | 0.8168 | 0.7388 | +0.0780 | [+0.0340; +0.1225] | 0.0000 | 1.36e-04 | sim | idêntico |
| 10 | filtro bertimbau s42 × bertimbau s43 | 42 | 0.8168 | 0.7183 | +0.0985 | [+0.0596; +0.1392] | 0.0000 | 9.08e-03 | sim | idêntico |
| 11 | filtro biobertpt s43 × biobertpt s42 | 43 | 0.8509 | 0.6961 | +0.1549 | [+0.1110; +0.2009] | 0.0000 | 1.06e-10 | sim | idêntico |
| 12 | filtro biobertpt s43 × bertimbau s42 | 43 | 0.8509 | 0.7110 | +0.1399 | [+0.0969; +0.1843] | 0.0000 | 4.19e-11 | sim | idêntico |
| 13 | filtro biobertpt s43 × biobertpt s43 | 43 | 0.8509 | 0.7388 | +0.1121 | [+0.0788; +0.1476] | 0.0000 | 1.04e-11 | sim | idêntico |
| 14 | filtro biobertpt s43 × bertimbau s43 | 43 | 0.8509 | 0.7183 | +0.1326 | [+0.0874; +0.1781] | 0.0000 | 4.48e-03 | sim | idêntico |
| 15 | filtro bertimbau s43 × biobertpt s42 | 43 | 0.8267 | 0.6961 | +0.1307 | [+0.0859; +0.1762] | 0.0000 | 7.62e-06 | sim | idêntico |
| 16 | filtro bertimbau s43 × bertimbau s42 | 43 | 0.8267 | 0.7110 | +0.1158 | [+0.0767; +0.1556] | 0.0000 | 5.56e-09 | sim | idêntico |
| 17 | filtro bertimbau s43 × biobertpt s43 | 43 | 0.8267 | 0.7388 | +0.0880 | [+0.0430; +0.1348] | 0.0004 | 1.00e+00 | sim | idêntico |
| 18 | filtro bertimbau s43 × bertimbau s43 | 43 | 0.8267 | 0.7183 | +0.1084 | [+0.0760; +0.1425] | 0.0000 | 2.92e-12 | sim | idêntico |
| 19 | regra pura × biobertpt s42 | 42 | 0.6944 | 0.6961 | -0.0017 | [-0.0588; +0.0540] | 0.9528 | 8.14e-76 | não | idêntico |
| 20 | regra pura × bertimbau s42 | 42 | 0.6944 | 0.7110 | -0.0166 | [-0.0716; +0.0386] | 0.5602 | 6.48e-79 | não | idêntico |
| 21 | regra pura × biobertpt s43 | 43 | 0.6944 | 0.7388 | -0.0444 | [-0.1022; +0.0145] | 0.1384 | 9.05e-58 | não | idêntico |
| 22 | regra pura × bertimbau s43 | 43 | 0.6944 | 0.7183 | -0.0240 | [-0.0818; +0.0334] | 0.4044 | 4.38e-64 | não | idêntico |
| 23 | filtro biobertpt s42 × regra pura | 42 | 0.8299 | 0.6944 | +0.1355 | [+0.0863; +0.1882] | 0.0000 | 1.22e-68 | sim | idêntico |
| 24 | filtro bertimbau s42 × regra pura | 42 | 0.8168 | 0.6944 | +0.1225 | [+0.0745; +0.1731] | 0.0000 | 5.85e-73 | sim | idêntico |
| 25 | filtro biobertpt s43 × regra pura | 43 | 0.8509 | 0.6944 | +0.1566 | [+0.1065; +0.2092] | 0.0000 | 2.26e-52 | sim | idêntico |
| 26 | filtro bertimbau s43 × regra pura | 43 | 0.8267 | 0.6944 | +0.1324 | [+0.0820; +0.1842] | 0.0000 | 3.11e-58 | sim | idêntico |

Os números da tabela são os dos relatórios, que são os mesmos bytes nos dois
lados (legado e código novo).

### Divergências encontradas

1. **Duas fórmulas de F1 no legado.** Os JSONs dos treinos usam o
   scikit-learn, que calcula 2·TP / (2·TP + FP + FN). O
   `FASE2_test_summary.json` usou 2·P·R / (P + R)
   (`make_rule_baseline.score`). Em aritmética de ponto flutuante as duas
   diferem na última casa decimal em 3 dos 9 sistemas:

   | Sistema | Gravado (2·P·R/(P+R)) | Recalculado (2·TP/(2·TP+FP+FN)) |
   | --- | --- | --- |
   | filtro BERTimbau s42 | 0.8168168168168167 | 0.8168168168168168 |
   | filtro BERTimbau s43 | 0.8267477203647415 | 0.8267477203647416 |
   | baseline BioBERTpt s42 | 0.6960784313725491 | 0.696078431372549 |

   A diferença (1,1·10⁻¹⁶) vem só da fórmula: as contagens, a precisão e o
   recall são idênticos, e a fórmula antiga, aplicada a eles, dá exatamente o
   valor gravado (é isso que o teste confere). O código novo usa uma fórmula
   só, a do sklearn, que também é a dos JSONs dos treinos e a do TCC. Nenhum
   número publicado muda nas casas reportadas.
2. **`sum()` do Python a partir da versão 3.12.** A soma de floats passou a
   ser compensada e pode diferir da soma sequencial na última casa. Uma
   primeira versão de `metricas.py` com `sum()` deu 0.7127856751095188 no
   macro-F1 do baseline BioBERTpt s43, contra 0.7127856751095191 do legado; a
   versão entregue faz as médias com o numpy, como o sklearn, e confere
   exatamente em 3.10, 3.12 e 3.13. Fica o registro para as etapas
   seguintes: números que precisam bater com o legado não devem ser somados
   com `sum()`.
3. **Versão do Python declarada.** O README antigo dizia que as execuções
   usaram Python 3.10; o campo `environment` de todas as execuções em
   `results/` registra 3.13.15. Corrigido no README da raiz e no das
   referências.

Registro (não é divergência): o baseline BERTimbau s43 teve o TEST avaliado
duas vezes no legado, com a mesma configuração e os mesmos valores (linhas 1 e
2 da sua trilha). A trilha foi preservada e é o que deve ser declarado na
análise de multiplicidade.

### Os testes acusam desvios

O código foi alterado de propósito e os testes rodados de novo:

| Alteração | Resultado |
| --- | --- |
| Médias com `sum()` do Python em vez do numpy | 6 testes falham (em Python 3.13) |
| F1 por 2·P·R/(P+R) em vez das contagens | 24 testes falham |
| Bootstrap com `default_rng(seed + 1)` | as 2 comparações da fase 1 (as testadas) falham |
| Uma reamostra a menos no bootstrap | idem |
| IC95 com `np.percentile(..., method="nearest")` | idem |
| Regra de semente invertida (B antes de A) | o teste de unidade da regra falha |

Duas alterações não mudaram nada, porque são equivalentes no numpy:
`integers(0, n, n, dtype=np.int32)` e `choice(n, n)` geram os mesmos índices
que `integers(0, n, n)`.

## Testes

| Tipo | Onde | Quantidade | Depende de |
| --- | --- | --- | --- |
| Unidade | `testes/unidade/` | 86 (30 novos nesta etapa, mais 2 do `--help` dos scripts novos) | nada externo |
| Equivalência com as referências | `testes/equivalencia/` | 142 (124 novos; 26 marcados `lento`) | só os arquivos versionados em `testes/referencia/` |
| Execuções com recurso externo | `testes/referencia/gerar_referencias.py` | — | um checkout do legado; fora da suíte |

Nenhum teste da suíte acessa o legado, a rede ou uma GPU.

### Resultados obtidos

Ambiente: Linux (contêiner de nuvem), instalação limpa.

| Python | numpy | scipy | Versões | Suíte completa |
| --- | --- | --- | --- | --- |
| 3.10.20 | 2.2.6 | 1.15.3 | escolhidas pelo pip (as fixadas não existem para 3.10) | 228 passaram |
| 3.12.3 | 2.1.3 | 1.16.3 | fixadas (`-c codigo/requirements.txt`) | 228 passaram |
| 3.13.16 | 2.1.3 | 1.16.3 | fixadas | 228 passaram |

Isso vale para o projeto trabalhado e para o pacote extraído numa pasta vazia
(rotina do [índice](README.md)). As 26 comparações dão os mesmos bytes também
com numpy 2.2.6 e scipy 1.15.3, mas isso não está garantido para qualquer
versão: a reprodução byte a byte é verificada com as versões da tabela.

**Não executado:** nada foi rodado no **Windows**. É o passo final de "Como
aplicar".

## Como aplicar

O pacote foi feito para ser extraído numa **pasta limpa**. No PowerShell:

```powershell
$zip     = "$env:USERPROFILE\Downloads\RECLin-PT-etapa-2-v1.zip"   # ajuste se salvou em outro lugar
$destino = "C:\Users\angeloals\Documents\RECLin-PT"

if (Test-Path $destino) { Rename-Item $destino "RECLin-PT-antes-etapa-2" }   # nada é apagado
Expand-Archive -Path $zip -DestinationPath $destino

cd $destino
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\codigo[testes]" -c codigo\requirements.txt
.\.venv\Scripts\python.exe codigo\scripts\preparar_dados.py conferir
.\.venv\Scripts\python.exe -m pytest codigo
```

O esperado é `conferir` listar as três partições como "confere" e o pytest
terminar com `228 passed` (cerca de 1,5 minuto). Com Python 3.10, rode o
`pip install` sem o `-c codigo\requirements.txt`.

Para ver as 26 comparações sendo refeitas e conferidas uma a uma:

```powershell
.\.venv\Scripts\python.exe codigo\scripts\comparar.py protocolo `
    --resultados codigo\testes\referencia\resultados_legado `
    --saida comparacoes --conferir codigo\testes\referencia\resultados_legado
```

O esperado é a última linha dizer `26 de 26 relatórios idênticos byte a byte à
referência`. A pasta `comparacoes` pode ser apagada depois.

### Registrar no git só as alterações da etapa 2

A partir de um repositório que já tem o commit da etapa 1 (versão 2), com a
cópia de trabalho limpa:

```powershell
cd C:\Users\angeloals\Documents\RECLin-PT
git status --short          # não deve listar nada

# a etapa 2 não remove nenhum arquivo da etapa 1: extrair por cima equivale
# a extrair numa pasta limpa
Expand-Archive -Path "$env:USERPROFILE\Downloads\RECLin-PT-etapa-2-v1.zip" -DestinationPath . -Force

git status --short
git add -A
git commit -m "Etapa 2 (v1): execução e avaliação do RECLin-PT" `
  -m "Formato das execuções (sidecar, diretório, trilha do TEST) e avaliação (métricas, McNemar e bootstrap pareado, protocolo das 26 comparações, agregação entre sementes), validados contra o legado: métricas de todos os sidecars iguais às registradas e as 26 comparações reproduzidas byte a byte." `
  -m "Referências: 72 resultados do legado copiados sem alteração para codigo/testes/referencia/resultados_legado/."

git show --stat --format="%h %s" HEAD
git rev-parse "HEAD^{tree}"
```

O `git status --short` antes do `git add` deve listar exatamente os arquivos de
"Arquivos criados e modificados" abaixo (a pasta `resultados_legado/`, nova,
aparece como uma linha só). Os hashes esperados (da árvore e do commit de
origem) estão na mensagem de entrega: o deste arquivo não pode constar nele
mesmo.

## Arquivos criados e modificados nesta etapa

Criados:

```
codigo/reclin/execucao/      __init__.py · predicoes.py · diretorio.py · trilha.py
codigo/reclin/avaliacao/     __init__.py · metricas.py · significancia.py · protocolo.py · agregacao.py
codigo/scripts/              avaliar.py · comparar.py
codigo/testes/equivalencia/  test_metricas_legado.py · test_comparacoes_legado.py
codigo/testes/unidade/       test_avaliacao.py · test_execucao.py · test_scripts_avaliacao.py
codigo/testes/referencia/resultados_legado/   72 arquivos (cópia de results/ do legado)
codigo/docs/entregas/etapa-2.md
```

Modificados:

| Arquivo | Mudança |
| --- | --- |
| `README.md` | Estado do repositório, versões de Python e bibliotecas (corrigidas), instalação com `-c`, comando das 26 comparações |
| `codigo/README.md` | Instalação, uso dos scripts novos, testes, módulos da etapa 2 e formato de uma execução |
| `codigo/pyproject.toml` | Dependências `numpy` e `scipy`; marcador `lento` |
| `codigo/reclin/util/io.py` | `gravar_texto`: grava texto exatamente como está (LF, sem quebra final), para os relatórios de comparação |
| `codigo/testes/conftest.py` | Fixture `legado` para ler os resultados do legado |
| `codigo/testes/referencia/README.md` | `resultados_legado/` e a versão de Python das execuções |
| `codigo/testes/referencia/gerar_referencias.py` | Parte `resultados` |
| `codigo/testes/referencia/referencias.json` | SHA-256 dos 72 arquivos novos (os anteriores não mudaram) |
| `codigo/docs/entregas/README.md` | Linha da etapa 2 e a instalação com versões fixadas na rotina de verificação |

## Pendências

- **Windows**: confirmar a instalação e os 228 testes no seu computador.
- **`corpus.py`** continua fora do escopo (a etapa 2 não trata do corpus); a
  proposta de incluí-lo numa etapa com teste que só roda com o XML continua
  em aberto.
- **`tcc/`**: precisa entrar antes da etapa 7.
- **Itens adiados** para as etapas que os usam: ver "Fica para as etapas que
  os usam".
- **Próxima etapa (3)**: `negacao/lexico` — não iniciada.
