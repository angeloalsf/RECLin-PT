# Reconstrução dos dados a partir dos XML (v2-texto-fonte) e auditoria da UMLS — antes da etapa 8

Data: 11/10/2026. Alterações feitas direto na pasta de trabalho. **Não inicia a
etapa 8.** Nenhum comando de controle de versão, nenhum acesso à rede, nenhum
download, nenhuma instalação, nenhum treino, nenhuma calibração, nenhuma
avaliação de modelo e nenhum uso dos rótulos do TEST para decisão. A v1 dos
dados, o MANIFEST da v1, as referências do legado e o relatório D9 estão
intactos (seção 7).

Autorização: a reconstrução técnica pela alternativa C1 do D9, em nova versão
rastreável, e a auditoria da UMLS (tarefa de 11/10/2026). Os XML e as
partições foram processados **só no computador do autor** (decisão do autor
nesta tarefa); a nuvem da sessão recebeu só código, testes e documentação.

## Resumo

| Etapa | Situação | Evidência principal |
| --- | --- | --- |
| A — inventário dos XML | concluída | 1.000 XML, 0 erros, conjunto `ed8aae01…8c25` (`sha256-lista-v1`) |
| B — parser e regra C1 | concluída | `reclin/corpus/`; representação intermediária; 421 documentos transformados; 26 anomalias mantidas e registradas |
| C — testes e auditoria dos offsets | concluída | 0 entidades desalinhadas na v2 (TRAIN, DEV e TEST); números do D9 reproduzidos exatamente |
| D — dataset v2 | concluída | `dados/particoes-v2/` + MANIFEST com proveniência; regerado byte a byte |
| E — partições, candidatos e rótulos | concluída | todas as previsões do D9 confirmadas; nenhuma divergência |
| F — integração | concluída (com verificação parcial) | `reclin/versao_dados.py`; v2 é a versão da fase; guardas contra mistura; testes com torch **não executados** |
| G — UMLS / McCray et al. | auditoria concluída | nenhuma implementação encontrada; recomendação: só citar (seção G) |
| H — D4 | proposta (nada executado) | seção H |
| I — testes finais | executados onde foi possível | seção 6 |

---

## 1. Diagnóstico inicial e referências usadas

Lidos antes de começar: o D9 (`d9-alinhamento.md` e o protótipo
`d9/d9_investigacao.py`), as notas da etapa 7 e das pendências, o protocolo
(`docs/protocolo/fase-biobertpt-seed42.md`), `reclin/particoes.py`,
`reclin/tarefa.py`, `reclin/entrada.py`, `reclin/negacao/lexico.py`,
`reclin/analises/alinhamento.py`, `reclin/execucao/{diretorio,inventario}.py`,
`reclin/treino/classificador.py`, `scripts/{preparar_dados,filtro_pistas}.py`
e os testes ligados às partições.

Divergências entre o pedido e a árvore real (resolvidas sem mudar o escopo):

| Ponto | Situação real | Resolução |
| --- | --- | --- |
| Parser do legado (`src/parse_semclinbr.py`) | não está na pasta de trabalho; o repositório original não estava disponível localmente nesta sessão (e não houve rede) | usado o que o D9 registrou do parser e verificado de forma independente: o texto da v1 = texto do parser XML com quebra → espaço em 1.000/1.000 documentos |
| Local dos XML | `codigo/SemClinBr-xml-public-v1/` (o README falava em `codigo/dados/brutos/`) | `caminhos.XML_SEMCLINBR` aponta a pasta real; `--xml` aceita outra; a pasta já é ignorada pelo `.gitignore` |
| `preparar_dados.py` previa `reclin/corpus.py` | — | implementado como pacote `reclin/corpus/` (um módulo por transformação) |
| Ambiente | sem pytest, scipy e torch no computador; sem torch na nuvem; instalação proibida | seção 6 (como os testes foram executados e o que ficou de fora) |

## 2. Etapa A — inventário e proveniência dos XML

`reclin/corpus/fonte.py` (`inventariar`), `scripts/reconstruir_dados.py
inventario`. Só leitura (os arquivos são abertos em modo binário de leitura;
os SHA-256 da pasta foram conferidos antes e depois — seção 7).

| Item | Resultado (verificado) |
| --- | --- |
| Arquivos | 1.000 `*.xml`; nenhum outro arquivo na pasta |
| Identificadores | nome do arquivo (o XML não tem identificador interno); faixa 8906–9935; 30 ausentes na faixa (9360–9389), todos numéricos |
| Duplicatas | 0 identificadores repetidos; 0 arquivos com conteúdo idêntico |
| Documentos ausentes | 0 em relação às partições (1.000 = 1.000) |
| Codificação | declarada UTF-8 em 1.000; UTF-8 válido em 1.000; 0 com BOM; 56 só ASCII |
| Quebras de linha | CRLF na estrutura dos 1.000 arquivos; no `<TEXT>`: 528 com CRLF (7.823 CRLF), 472 sem quebra, 0 CR ou LF isolados; 9 textos com TAB |
| Erros de leitura e estrutura | 0 (raiz, elementos, atributos, inteiros, `<TEXT>` sem marcação interna) |
| Anotações / relações | 45.508 / 11.458; anomalia estrutural: 1 offset negativo (documento 9808, anotação 39461) |
| Hash do conjunto | `ed8aae01ef807c02dd12f68d57c54b081241c996e39a708eda94b348c79d8c25` |
| Regra (`sha256-lista-v1`) | SHA-256 (UTF-8) de uma linha `<nome>\t<sha256 do arquivo>\n` por XML, em ordem numérica do identificador |
| Origem declarada | "declarada, não verificada: SemClinBr (Oliveira et al.), acesso restrito, segundo o README; pasta local `SemClinBr-xml-public-v1`" |
| Data de geração dos derivados | `geracao.data` = 2026-10-11 no MANIFEST da v2 (com Python 3.10.12, CPython) |

O SHA-256 de cada arquivo está em `dados/particoes-v2/FONTE.json`
(`arquivos[]`), o que identifica exatamente os XML usados.

**Limitação:** o hash de conjunto do D9 (`709790d5…`) não pôde ser
reproduzido — o D9 descreve a composição só como "SHA-256 de nome + SHA-256
de cada arquivo, em ordem", e nenhuma das composições plausíveis testadas o
reproduz. Por isso registrei uma regra nova e documentada. A equivalência
com a cópia do D9 é sustentada pelas contagens, todas idênticas às do D9
(sistemas 472/421/101/6, 44.179/1.093/171/39/25/1 por categoria, faixa
9015–9184, 18.078 divergências pela checagem do legado), não pelo hash.
Origem oficial, licença e integridade histórica da cópia **não** são
comprováveis localmente.

## 3. Etapa B — parser e correção C1

### Arquitetura (`reclin/corpus/`, núcleo)

| Módulo | Papel |
| --- | --- |
| `semclinbr.py` | XML → `DocumentoFonte` (representação intermediária): `texto_bruto` (o `<TEXT>` dos bytes, referências XML desfeitas, CRLF preservados), `texto_xml` (o do parser), anotações com todos os atributos e `ordem`, relações com `ordem`, contagem de quebras, SHA-256 do arquivo. Confere `texto_bruto` normalizado == `texto_xml`; qualquer formato inesperado é `ErroFonte` |
| `conversao.py` | `DocumentoFonte` → registro do dataset (mesmo esquema da v1) + registro da conversão (sistema, `transformado`, justificativa, categorias, anomalias) |
| `fonte.py` | inventário (etapa A) |
| `auditoria.py` | cada entidade × anotação de origem, pela identidade |
| `comparacao.py` | invariantes entre versões (documentos, entidades, relações, texto, candidatos, `y_true`, janelas) |
| `geracao.py` | XML → intermediária → dataset → partições + MANIFEST/FONTE/ANOMALIAS; `conferir` (regeração byte a byte) |

### Regra C1 (implementada como no D9, confirmada nos XML locais)

1. Sistema por documento, com todas as anotações, **na posição** (nunca por
   busca): `sem_crlf`; ou conta quantas anotações conferem com CRLF = 2
   (`bruto`) e com quebra = 1 (`lf`); vence a maior; empate com todas
   conferindo → `indistinguivel`; empate com falha → `ambiguo` (erro, nada é
   gerado — 0 casos).
2. Texto: `bruto` → cada CRLF vira **dois espaços** (o texto passa a ter o
   comprimento da convenção dos offsets); `sem_crlf`, `lf`, `indistinguivel`
   → o texto do legado, sem mudança (requisito 4: nada é transformado onde não
   é necessário; os 6 indistinguíveis ficam como na v1, como recomendado no
   D9).
3. Entidades e relações copiadas da fonte, na ordem do arquivo; **nenhum
   offset alterado, nenhuma anotação descartada**. Identificador repetido e
   relação órfã são erro (exigiriam escolher ou descartar; 0 casos).

| Sistema | Corpus | TRAIN | DEV | TEST |
| --- | --- | --- | --- | --- |
| `sem_crlf` | 472 | 365 | 52 | 55 |
| `bruto` (transformados) | 421 | 346 | 36 | 39 |
| `lf` | 101 | 85 | 10 | 6 |
| `indistinguivel` | 6 | 4 | 2 | 0 |
| `ambiguo` | 0 | 0 | 0 | 0 |

A justificativa individual de cada documento transformado (contagens que
decidiram o sistema, número de CRLF) está em
`FONTE.json → conversao_por_documento`.

### As 26 anomalias (mantidas como estão, registradas em `ANOMALIAS.json`)

| Tipo | Corpus | TRAIN | DEV | TEST | Relações que as usam |
| --- | --- | --- | --- | --- | --- |
| atributo com palavras duplicadas | 22 | 19 | 1 | 2 | 0 no TRAIN/DEV |
| atributo divergente do trecho (descontínuo/palavra omitida) | 3 | 0 | 0 | 3 | — |
| offset inválido (`start = -1`, 9808/39461) | 1 | 1 | 0 | 0 | 0 |
| offset entre `\r` e `\n` | 0 | 0 | 0 | 0 | — |

No TEST, a participação em relações não foi inspecionada no arquivo de
anomalias; a contagem estrutural da auditoria (sem tipo) dá 0 relações com
argumento anômalo nas três partições. **Decisão sua (seção 11):** manter as
26 como estão (o que foi feito) ou tratá-las. A entidade com `start = -1`
gera 16 janelas sem marcador (`invalida`, todas `no_relation`) no TRAIN — já
era assim na v1.

## 4. Etapa C — testes do parser e auditoria dos offsets

Auditoria por identidade (`reclin/corpus/auditoria.py`): cada entidade é
ligada à anotação de origem por (`doc_id`, `id`); o trecho do dataset é
comparado com o atributo **na posição**; "anomalia" = nem a fonte, na
convenção do próprio documento, confere; "identidade divergente" = os campos
copiados (start, end, text, tipo) diferem da origem.

| Partição | Versão | Alinhadas | Desalinhadas | Anomalias da fonte | Identidade divergente | Não verificáveis | Relações com argumento desalinhado |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN (36.455) | v1 | 21.948 | 14.487 | 20 | 0 | 0 | 4.302 / 9.221 (708 `negation_of`) |
| | v2 | 36.435 | **0** | 20 | 0 | 0 | **0** |
| DEV (4.513) | v1 | 3.003 | 1.509 | 1 | 0 | 0 | 377 / 987 (58 `negation_of`) |
| | v2 | 4.512 | **0** | 1 | 0 | 0 | **0** |
| TEST (4.540, estrutural) | v1 | 2.990 | 1.545 | 5 | 0 | 0 | 507 / 1.250 (sem tipo) |
| | v2 | 4.535 | **0** | 5 | 0 | 0 | **0** |

As relações são idênticas à fonte (mesmos pares, mesma ordem) em 1.000/1.000
documentos das duas versões; o alinhamento subiu **sem nenhuma associação
alterada** (identidade divergente = 0; entidades e relações v1 = v2 byte a
byte).

Auditoria da etapa 7 (`analises.alinhamento`, sem modificação) — os números
do D9 reproduzidos exatamente:

| Partição | Versão | exato | equivalente | deslocado | ausente | `negation_of` afetados | deriva (sem / monótona / não monótona) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN | v1 | 21.479 | 469 | 13.899 | 608 | 672 / 1.255 | 453 / 223 / 124 |
| | v2 | 35.368 | 1.067 | 1 | 19 | **0** / 1.255 | 799 / 1 / 0 |
| DEV | v1 | 2.970 | 33 | 1.435 | 75 | 58 / 150 | 64 / 21 / 15 |
| | v2 | 4.403 | 109 | 0 | 1 | **0** / 150 | 100 / 0 / 0 |

Os "ausente/deslocado" que restam na v2 são as anomalias da fonte (o único
"monótono" é o documento 9808). Pares cujo trecho marcado diverge do atributo:
v1 72.212 / 7.814 / 7.886 → v2 148 / 4 / 48 (TRAIN/DEV/TEST), todos envolvendo
uma das 26 anomalias (verificado: 0 fora delas). Janelas marcadas: v2 0
`invalida` no DEV (v1: 160) e 16 no TRAIN (v1: 1.450).

Testes criados:

* `testes/unidade/test_corpus.py` (33, sintéticos): LF e CRLF; normalização;
  `start`/`end`; Unicode e acentos; entidades repetidas e sobrepostas;
  relações; ids repetidos e relação órfã (erro); `start = -1`, palavras
  duplicadas, atributo divergente e offset entre `\r\n` (mantidos e
  registrados); documento ambíguo (erro); determinismo; rastreabilidade XML →
  intermediária → dataset → partição; inventário; geração, conferência,
  proteção da v1, parada se a atribuição divergir; comparação e janelas;
  invariantes dos pares; conferência do texto bruto contra o parser.
* `testes/unidade/test_dados_v2.py` (21, dados reais; os de XML são pulados sem a
  pasta): registro × arquivos; MANIFEST; anomalias; previsões da C1 por
  partição; conjunto de referência; partição refeita; léxico; números da
  auditoria; cada entidade contra a origem; texto v1/v2 contra a fonte;
  regeração byte a byte (`lento`).
* `testes/unidade/test_versao_dados.py` (11): seção F.

Mutações (cada proteção desligada de propósito, testes rodados): **13 de 13
detectadas** — C1 desligada, sistema invertido, conferência bruto × parser,
anomalias não registradas, relação órfã aceita, atribuição divergente aceita,
proteção da v1, identidade na auditoria, guarda do `diretorio`, mistura no
inventário, mistura na seleção, derivada de outra versão, `conferir_uma_versao`.

## 5. Etapas D e E — a nova versão e os invariantes

`python codigo/scripts/reconstruir_dados.py gerar --data 2026-10-11`
(no computador, Python 3.10.12):

| Arquivo | SHA-256 |
| --- | --- |
| `dados/particoes-v2/train.jsonl` | `b8aaafce423afb9cda49fa5d116c14b14d6b9c2d09a0521bff4ef8733305d034` |
| `dados/particoes-v2/dev.jsonl` | `19feb024ca9d60f0016491b2e9bc35ab7de9b5cc384e0d22ba34b21a85a9f99c` |
| `dados/particoes-v2/test.jsonl` | `47d40d0b06f23714a8f332e3f8455bff0471bb68fad7b4a835b2aa72812e9666` |
| `dados/particoes-v2/MANIFEST.json` | `18264860a3d8fff7bcc335b8f5703a55d60e08893a09be2afa20b1ea446d3b70` |
| `dados/particoes-v2/FONTE.json` | `15a7e029b087ccf1e38f7db39d033b3e41da1aa377285a3c267a81bc5d8aca60` |
| `dados/particoes-v2/ANOMALIAS.json` | `1d9780a5cb4aeed8d86b15130793edcb9f90f61bc6efc8497affb3edd0f3a094` |
| `dados/processados/dataset-v2-texto-fonte.jsonl` (fora do git) | `9dcbfd105ae12c45e159c8997b987e33e18c6a25fb41e44eb0f219c2a0e5b2c6` |
| `dados/processados/semclinbr-intermediario-v2.jsonl` (fora do git) | `556eb457b7796db9a86839d557cbefac84e747589572c0ee328a3a4a72b4b479` |

Identificador `v2-texto-fonte`, pasta `dados/particoes-v2/` (os propostos;
nada foi renomeado). O MANIFEST mantém as chaves da v1 (`generator`, `seed`,
`source`, `splits`) e acrescenta: `versao`; `conversor` (módulo,
`versao_parser` `semclinbr-xml-2`, regra C1, sistemas, hash da
intermediária); `fonte_xml` (pasta, origem declarada, 1.000 arquivos, regra e
hash do conjunto, hash do `FONTE.json`); `anomalias` (26, por tipo e
partição, tratamento, hash do `ANOMALIAS.json`); `contagens` (documentos,
entidades, relações, candidatos, transformados); `conjuntos_referencia`;
`derivada_de` (pasta, MANIFEST e partições da v1, invariantes por partição);
`geracao` (data, Python). A chave `geracao` é a única que a conferência
ignora.

**Invariantes v1 × v2** (`reconstruir_dados.py comparar`;
`docs/entregas/reconstrucao-v2/comparacao_v1_v2.json`):

| Previsão do D9 | TRAIN | DEV | TEST |
| --- | --- | --- | --- |
| mesmos documentos, mesma ordem | sim (800) | sim (100) | sim (100) |
| sobreposição entre partições | 0 | 0 | 0 |
| entidades / relações idênticas | sim (36.455 / 9.221) | sim (4.513 / 987) | sim (4.540 / 1.250) |
| `text` diferente só nos transformados | 346 = 346 | 36 = 36 | 39 = 39 |
| texto v1 = legado da fonte; v2 = regra aplicada à fonte; comprimento = bruto | sim | sim | sim |
| candidatos (chaves e ordem) | 152.686, idênticos | 19.064, idênticos | 19.210, idênticos |
| rótulos | 1.255 / 7.096 / 144.335 (iguais) | 150 / 763 / 18.151 (iguais) | não contados |
| `y_true` (hash) | idêntico | idêntico | idêntico |
| invariantes dos pares (gap ≤ 25, ordem, sem auto-par/repetição) | 0 violações | 0 | 0 |
| janelas iguais fora dos transformados | sim | sim | sim |
| `TAMANHOS_ESPERADOS` | mantidos | mantidos | mantidos |
| léxico congelado no TRAIN | 11 formas, `70c93fa807de` (igual à v1) | — | — |
| conjunto de referência | só `particao_sha256` muda | idem | idem |
| reprodutibilidade | regerada byte a byte dos XML (`conferir`, saída 0) | | |

A partição foi **refeita** (`particionar`, semente 42) e a atribuição saiu
idêntica à da v1; o código para sem gravar nada se não sair. Nenhum
invariante divergiu: nenhuma etapa foi interrompida.

## 6. Etapa F — integração

| Ponto | Mudança |
| --- | --- |
| `reclin/versao_dados.py` (novo) | registro congelado das duas versões (SHA-256 do MANIFEST e das partições), `VERSAO_DA_FASE = v2-texto-fonte`, identificação de partições/pastas/execuções, `conferir_uma_versao`, `exigir_mesma_versao` |
| `util/caminhos.py` | `PARTICOES_V1`, `PARTICOES_V2`, **`PARTICOES = PARTICOES_V2`** (padrão de todos os scripts), `XML_SEMCLINBR`, `DATASET_V2`, `INTERMEDIARIO_V2` |
| `execucao/diretorio.py` | `gravar_predicoes` recusa predições de outra versão registrada que a do treino ou dos conjuntos já gravados, e grava `versoes_dados` no `config.json` |
| `execucao/inventario.py` | `versao_dados` de cada execução; problema se misturar versões ou se `versoes_dados` não corresponder aos hashes; derivada herda e precisa da mesma versão da base; científica só na v2 (motivo "dados da versão v1…"); a seleção recusa versões diferentes |
| `treino/classificador.py`, `two_stage/composicao.py` | `particoes_congeladas` aceita qualquer versão registrada; a avaliação do TEST recusa partições de outra versão |
| `scripts/filtro_pistas.py` | base, partições e TRAIN do léxico precisam ser da mesma versão |
| `scripts/preparar_dados.py` | padrão = v2; `manifesto`/`particionar` recusam a pasta gerada dos XML |
| `scripts/analisar.py`, `relatorios/tabelas.py` | registram `versao_dados` na origem |
| `scripts/reconstruir_dados.py` (novo), `Makefile` | `inventario`, `gerar`, `conferir`, `auditar`, `comparar` (alvos `inventario-xml`, `reconstruir-v2`, `conferir-v2`, `auditar-v2`, `comparar-v2`, `conferir-dados-v1`) |
| Protocolo | linha "Dados", D9 no registro de alterações, nota da D4, bloco JSON `dados` (versão, pasta, MANIFEST, regra) |
| Testes do legado | apontam explicitamente a v1 (`conftest.py` `documentos`/`manifesto`/`subconjunto`; equivalência; `test_marcadores_offsets`; `test_analises`; janelas do `test_two_stage`; `--particoes` na avaliação do `test_treino_legado`); **nenhum foi enfraquecido ou removido** — continuam comparando a v1 com as mesmas referências. `test_util` passou a afirmar V1, V2 e `PARTICOES == V2`; `test_pacote` passou a exigir os arquivos da v2 e os módulos novos |

Os resultados históricos continuam em `testes/referencia/` e só os testes de
equivalência os leem; o inventário os recusa como antes. Treinos, inferência e
avaliação **não foram executados** para validar a integração (proibido e sem
torch): os caminhos com torch foram verificados só por leitura e compilação.
A identidade do código (D6) muda, como esperado, com os módulos novos.

## 7. Integridade da versão anterior e das referências

Comparação de SHA-256 de todos os arquivos do projeto (fora `.git`) antes e
depois: **1.103 arquivos protegidos** (`dados/particoes/`, `testes/referencia/`,
os 1.000 XML, `docs/entregas/d9*`) **inalterados**; 0 arquivos removidos; MANIFEST
da v1 `a73f291a…f8ae1` e partições v1 iguais aos do registro. Nenhum
`__pycache__` foi gravado no projeto (execuções com `PYTHONDONTWRITEBYTECODE`).

## 8. Testes executados

Ambientes: **computador** (Python 3.10.12, numpy; sem pytest, scipy e torch;
com os dados) e **nuvem da sessão** (Python 3.13, numpy, scipy e um pytest
9.1.1 já presente no cache local, usado sem instalação; sem torch; sem os
dados, por decisão do autor). No computador os testes rodaram com um executor
mínimo no estilo do pytest, só com a biblioteca padrão (fora do projeto; suporta
fixtures, `parametrize`, `skipif`, `raises`, `tmp_path`, `monkeypatch`) — é
evidência mais fraca que o pytest e está registrada como tal; ele foi aferido
contra o pytest na nuvem nos mesmos arquivos (mesmos resultados).

| Conjunto | Onde | Resultado |
| --- | --- | --- |
| `testes/unidade/` (todos, sem `lento`) | computador, dados reais | 411 passaram, 8 pulados; 11 não executáveis **por ambiente**: torch (4 do `test_pacote` e 2 módulos de teste que não importam), scipy (4), `capsys` não suportado pelo executor (1) |
| `testes/unidade/test_dados_v2.py` com o `lento` | computador | 21 passaram (inclui a regeração byte a byte) |
| `testes/unidade/` + `integracao/` + `equivalencia/` | nuvem, pytest | 377 passaram, 22 pulados (torch); falhas e erros só por dados ausentes (partições, referências do legado) ou torch — todos os que dependem de dados passaram no computador, e os que dependem de scipy/`capsys` passaram aqui |
| `testes/equivalencia/` | computador | 161 passaram; 26 comparações `lento` não executadas (scipy ausente onde estão os dados); `test_treino_legado` e `test_estrategias_treinadas_legado` pulados (torch) |
| Regeração byte a byte (`conferir`) | computador | saída 0 |
| Mutações | nuvem | 13/13 detectadas |
| Importação sem efeito colateral | nuvem | 62 módulos; os 7 que exigem torch não importados |
| `--help` dos scripts | nuvem | 13/16; `avaliar_test`, `pretreino`, `treinar` exigem torch |

**Não executados em nenhum ambiente:** todos os testes que exigem torch
(`unidade/test_treino.py`, `test_estrategias_treinadas.py`,
`test_protocolo_fase.py`, `test_relatorios.py`, `integracao/*`,
`equivalencia/test_treino_legado.py`, `test_estrategias_treinadas_legado.py`)
e as 26 comparações do legado (scipy + dados). Os módulos alterados nesse
caminho (`classificador`, `composicao`, `diretorio`, `inventario`) têm as
partes novas testadas sem torch, mas o efeito sobre esses testes só se
confirma rodando-os. **Recomendo rodar a suíte completa no seu ambiente
(`make -C codigo testes-completos`) antes da etapa 8.**

## 9. Etapa G — UMLS e McCray, Burgun e Bodenreider (2001)

### O que existe

| Local | Implementação ou referência encontrada | Papel atual | Relação com o artigo |
| --- | --- | --- | --- |
| XML do SemClinBr, atributo `tag` | nomes de tipos semânticos da UMLS (87 nomes, ex.: `Sign or Symptom`, `Disease or Syndrome`) e 2 rótulos do corpus (`Abbreviation`, `Negation`), unidos por `|`; **sem TUI e sem CUI** (os únicos atributos são `id tag start end text abbr`) | dado da fonte | é o insumo que um agrupamento usaria; a nomenclatura não foi conferida contra a lista oficial (não disponível localmente) |
| `type` das entidades (v1 e v2) | cópia do `tag` | descritivo | nenhuma |
| `reclin/analises/corpus.py`, `relatorios/eda.py` | contagem de entidades por `type` (tabela "Tipos de entidade mais frequentes") | EDA | nenhuma (sem agrupamento) |
| `reclin/analises/falsos_positivos.py` | `tipo_e2` dos falsos positivos | análise descritiva | nenhuma |
| `reclin/entrada.py`, README | marcadores **sem tipo semântico** (Soares et al., 2019) | decisão de projeto | o modelo não usa tipos |
| léxico, estratégias, modelos, avaliação | não usam `type` | — | nenhuma |
| `reclin/corpus/semclinbr.py` (nova) | documentação do formato do `tag` | leitura | nenhuma |

**UMLS, TUI, Semantic Groups, mapeamento tipo→grupo, McCray: não localizados
em nenhum arquivo examinado** (código, testes, scripts, documentação,
notebook, notas de entrega e documentos do projeto). O texto do TCC
(`tcc/`) **não está** na pasta de trabalho: não foi possível auditar se ele
cita o artigo.

Dados do corpus relevantes (estruturais, todas as partições): 45.508
entidades; 42.439 com pelo menos um tipo UMLS; **3.069 só com rótulos do
corpus** (sem tipo UMLS); 5.915 com 2 a 5 tipos UMLS distintos; 1.218 com
partes repetidas no próprio `tag`. Só TRAIN (rótulos permitidos): o `e1` de
1.214 dos 1.255 pares `negation_of` tem `Negation`; o `e2` cobre 42 tipos, a
maioria `Sign or Symptom` (801).

### Avaliação metodológica

Os seis princípios (validade semântica, parcimônia, completude, exclusividade,
naturalidade, utilidade) foram pensados para reduzir os ~134 tipos semânticos
a grupos coerentes **de conceitos**. No RECLin-PT:

* **Completude** falha no corpus: `Negation` e `Abbreviation` não são tipos
  UMLS, e 3.069 entidades não teriam grupo.
* **Exclusividade** vale entre tipos no artigo, mas não por entidade: 5.915
  entidades têm vários tipos, que podem cair em grupos diferentes — exigiria
  uma regra de resolução que o artigo não dá.
* **Utilidade** para a tarefa não está demonstrada: a classe-alvo é definida
  pelo `e1` (pista de negação, já capturada pelo léxico e pelo rótulo
  `Negation`); o grupo poderia informar sobre o `e2` (Disorders/achados),
  mas isso é hipótese.
* O espaço de rótulos é de **relações**; grupos semânticos tipam
  **entidades**.

| Alternativa | Natureza | Avaliação |
| --- | --- | --- |
| 1. Citar o artigo | fundamentação | adequado, desde que o texto diga que o projeto não usa os grupos |
| 2. Implementar o mapeamento | infraestrutura | exige o arquivo oficial de grupos da NLM (não disponível localmente; download proibido) e regras para multi-tipo e rótulos fora da UMLS |
| 3. Grupos como atributo do modelo | **nova estratégia experimental** | precisaria de hipótese, desenho e aprovação; muda a entrada dos modelos |
| 4. Mudar o espaço de rótulos | **outra tarefa** | inadequado: os rótulos são de relação |

**Recomendação: apenas citar o artigo (alternativa 1), sem implementar os
grupos nesta fase.** Se no futuro se quiser testar a hipótese, o desenho
mínimo seria: módulo opcional `reclin/corpus/grupos_semanticos.py` com
`carregar_mapeamento(caminho_semgroups) -> {nome_do_tipo: grupo}` e
`grupos_da_entidade(tag) -> frozenset[str]` (rótulos do corpus num grupo
próprio, multi-tipo devolvendo todos os grupos); testes de cobertura (todo
nome do corpus mapeado ou listado), de determinismo e de multi-tipo; dados: o
arquivo oficial da NLM com versão registrada; hipótese: "marcadores tipados
pelo grupo do `e2` aumentam o F1 de `negation_of` no DEV em relação aos
marcadores sem tipo", avaliada como estratégia nova do protocolo, nunca
alterando as existentes. Nada disso foi implementado.

## 10. Etapa H — proposta para a D4 (`min_freq=3`)

Fatos verificados: o léxico depende só do `text` das entidades e dos
candidatos do TRAIN, que a v2 não muda — o congelado na v2 é idêntico (11
formas, `70c93fa807de`). O valor 3 vem da calibração do legado sobre as
predições do DEV de quatro baselines do legado (nunca do TEST).

| | A — manter `min_freq=3` (fixado a priori) | B — recalibrar com TRAIN e DEV da v2 |
| --- | --- | --- |
| Uso de resultado histórico | sim (a origem do valor é uma calibração do legado) | não |
| Consumo do DEV | nenhum adicional | B-DEV usa o DEV de novo; B-TRAIN não |
| Dependência | nenhuma | B-DEV: o restrito e a Pair-Aware esperam o baseline e a calibração |
| Efeito | nenhum no código | muda o espaço restrito se o valor mudar |

**Recomendação: B-TRAIN** — recalibrar só com o TRAIN da v2, por um critério
independente de modelo, antes do passo 1 da etapa 8. Atende à diretriz de não
usar resultados históricos como entrada, não gasta o DEV (que já escolhe
épocas e limiares) e não cria dependência entre os passos. Procedimento a
pré-registrar (não executado):

* **Conjunto:** só `dados/particoes-v2/train.jsonl`, numa pasta sem DEV e TEST
  (como na calibração da etapa 4).
* **Validação:** 5 partes por documento, com `random.Random(42)` e a mesma
  estratificação por presença de `negation_of` de `particoes.particionar`.
* **Valores:** `min_freq ∈ {1, 2, 3, 5, 10}` (os mesmos da varredura histórica).
* **Métrica:** em cada parte, induzir o léxico das outras quatro e, nos pares
  candidatos (`max_gap` 25) da parte retida, tratar "`e1` é pista" como
  detector de `negation_of`: VP, FP e FN somados nas 5 partes; F1 = 2·VP /
  (2·VP + FP + FN).
* **Desempate:** maior `min_freq` (léxico menor, mais parcimonioso).
* **Vazamento:** o DEV e o TEST não são abertos; nenhum modelo; nenhuma
  predição; resultado gravado com o hash do TRAIN e das partes.
* **Registro:** o valor escolhido induzido do TRAIN inteiro vira o novo
  `CONGELADO` (com a origem "validação cruzada no TRAIN v2"), atualizado no
  protocolo (linha "Léxico" e bloco JSON) **antes** de qualquer treino; se o
  escolhido for 3, nada muda além da origem.
* **Limitação:** o critério mede o léxico, não o filtro aplicado a um modelo;
  com 4/5 do TRAIN nas partes, as contagens são ~20% menores que no TRAIN
  inteiro (o mesmo limiar é usado; declarado).

Alternativas, se preferir: **A** (declarar o valor como parâmetro herdado,
sem recalibrar) ou **B-DEV** (o procedimento da nota das pendências, sobre as
predições do DEV do baseline v2). Nada foi calibrado nem escolhido.

## 11. Decisões que dependem de você

1. **Adotar a v2 como dados da fase.** Implementado (`PARTICOES = v2`,
   protocolo atualizado) conforme a autorização; reversível por uma constante
   (`versao_dados.VERSAO_DA_FASE` / `caminhos.PARTICOES`). Confirme.
2. **26 anomalias da fonte:** mantidas como estão (recomendado; nenhuma em
   relação no TRAIN/DEV, 0 relações estruturais no TEST). Confirme ou peça
   outro tratamento.
3. **6 documentos indistinguíveis:** mantidos com o texto da v1. Confirme.
4. **D4:** A, B-TRAIN (recomendado) ou B-DEV.
5. **UMLS:** só citar (recomendado) ou abrir uma estratégia nova com
   aprovação própria.
6. **Ambiente:** rodar a suíte completa com torch/scipy (seção 8).
7. **Anexos:** manter `docs/entregas/d9/` e `docs/entregas/reconstrucao-v2/`
   (só contagens e hashes).

## 12. Critérios ainda não cumpridos para a etapa 8

* suíte completa com torch não executada nesta tarefa;
* D4 sem decisão (bloqueia o filtro, o restrito e a Pair-Aware);
* confirmação das decisões 1–3;
* pendências anteriores ainda abertas (D5 — redação; D8/D10 — TCC e figuras).

**A etapa 8 não está liberada.**

## Arquivos

**Criados:** `reclin/corpus/{__init__,semclinbr,fonte,conversao,auditoria,comparacao,geracao}.py`,
`reclin/versao_dados.py`, `scripts/reconstruir_dados.py`,
`testes/unidade/{test_corpus,test_dados_v2,test_versao_dados}.py`,
`dados/particoes-v2/{train,dev,test}.jsonl`, `MANIFEST.json`, `FONTE.json`,
`ANOMALIAS.json`, `docs/entregas/reconstrucao-v2.md` (esta nota),
`docs/entregas/reconstrucao-v2/{auditoria_offsets,comparacao_v1_v2}.json`;
fora do git: `dados/processados/dataset-v2-texto-fonte.jsonl`,
`dados/processados/semclinbr-intermediario-v2.jsonl`.

**Modificados:** `README.md` (raiz), `codigo/README.md`, `codigo/Makefile`,
`docs/protocolo/fase-biobertpt-seed42.md`, `docs/reproducao.md`,
`docs/entregas/README.md`, `reclin/util/caminhos.py`,
`reclin/execucao/{diretorio,inventario}.py`, `reclin/treino/classificador.py`,
`reclin/estrategias/two_stage/composicao.py`, `reclin/relatorios/tabelas.py`,
`scripts/{analisar,filtro_pistas,preparar_dados}.py`, `testes/conftest.py`,
`testes/equivalencia/{test_conjunto_referencia,test_estrategias_legado,test_estrategias_treinadas_legado,test_particoes,test_treino_legado}.py`,
`testes/unidade/{test_analises,test_marcadores_offsets,test_pacote,test_two_stage,test_util}.py`.

**Removidos:** nenhum.
