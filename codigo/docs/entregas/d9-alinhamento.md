# D9 — desalinhamento entre os textos e os offsets das entidades

Data: 11/10/2026. Investigação técnica anterior à etapa 8; **não inicia a
etapa 8**. Nenhum treino, nenhuma calibração, nenhum experimento científico,
nenhuma avaliação do TEST, nenhum acesso à rede, nenhuma instalação e nenhum
comando de controle de versão. As partições congeladas, o MANIFEST, as
referências do legado e os resultados históricos **não foram alterados**
(seção 9). Nenhuma correção foi aplicada aos dados ou ao código do projeto: o
que existe é um protótipo isolado e somente leitura.

**Classificação final: Causa identificada e correção recuperável** (seção 7),
com uma pergunta secundária ainda inconclusiva — por que a fonte usa duas
convenções de contagem de quebra de linha — que não altera a correção.

---

## 1. Resumo executivo

- **Causa confirmada.** Os offsets do XML do SemClinBr contam cada quebra de
  linha `\r\n` (CRLF) como **dois** caracteres em 421 dos 528 documentos que
  têm quebras de linha. A conversão do legado (`src/parse_semclinbr.py`) lê o
  `<TEXT>` com um parser XML, que, pela especificação do XML (seção 2.11 do
  XML 1.0), entrega cada CRLF como um único `\n`; em seguida troca `\n` por
  espaço. O texto gravado fica **um caractere mais curto por quebra de
  linha**, e os offsets foram copiados sem ajuste. O deslocamento é de −k para
  uma entidade precedida por k quebras de linha **no mesmo documento**: ele se
  acumula ao longo de cada documento e recomeça do zero no seguinte.
- **Onde entrou:** na etapa 1 do pipeline legado (XML → `dataset.jsonl`),
  antes da partição. Partições, candidatos, janelas e marcadores apenas
  propagam o problema; o código novo reproduz o legado fielmente.
- **O legado detectou e não tratou:** o `logs/pipeline.log` do repositório
  original registra `Offsets divergentes (anotacao != texto): 18079` (17/06/2026).
  Reproduzi o número exato.
- **A fonte não é homogênea:** em 101 documentos com CRLF — exatamente os
  documentos com CRLF de id entre 9015 e 9184 — os offsets contam a quebra
  como **um** caractere; esses já estão corretos no texto atual. O sistema de contagem é determinável por
  documento, sem ambiguidade em nenhum dos 1.000.
- **Recuperação:** com o sistema de contagem do documento, 45.482 das 45.508
  anotações (99,94%) apontam para o próprio texto anotado, **na mesma
  posição**, sem busca textual. As 26 restantes são defeitos da própria
  fonte (atributo `text` com palavras duplicadas, trechos descontínuos, um
  `start = -1`), não do alinhamento.
- **Efeito no TRAIN e no DEV:** pares `negation_of` com alguma entidade
  desalinhada passam de 672/1.255 para 0/1.255 (TRAIN) e de 58/150 para 0/150
  (DEV).
- **Recomendação:** Alternativa C, variante **C1** — nova versão dos dados
  reconstruída do XML com o texto **preservando o comprimento** (cada CRLF
  vira dois espaços) nos documentos que contam CRLF como 2. Offsets,
  entidades, relações, atribuição dos documentos às partições, candidatos e
  `y_true` ficam **idênticos** aos atuais; só o campo `text` de 421 documentos
  muda. Depende da sua aprovação (seção 10).

**Correções aos indícios registrados na etapa 7:** (a) "58 entidades
desalinhadas em 150 verificadas" no DEV são, na verdade, **58 de 150 pares
`negation_of`** com ao menos uma entidade desalinhada; no nível das
entidades, o DEV tem 1.510 de 4.513 entidades nem exatas nem equivalentes;
(b) o deslocamento **não** se acumula entre documentos, e sim dentro de cada
documento.

---

## 2. Arquivos e etapas examinados

Repositório original disponível localmente (somente leitura), incluindo o
corpus em `SemClinBr-xml-public-v1/` (1.000 XML).

| Componente | Caminho | Papel | Como foi examinado |
| --- | --- | --- | --- |
| Corpus de origem | `RECLin-PT-original/SemClinBr-xml-public-v1/*.xml` | `<TEXT>` (texto clínico), `<TAGS>/<annotation id tag start end text>`, `<RELATIONS>/<rel annotation1 annotation2 reltype>` | bytes brutos e parser XML; conjunto identificado por `709790d5…` (SHA-256 de nome + SHA-256 de cada arquivo, em ordem) |
| Conversão XML → JSONL | `RECLin-PT-original/src/parse_semclinbr.py` (`parse_one`, sha `b275efd4…`) | `lxml.etree.XMLParser`; `text = raw.replace("\r\n"," ").replace("\r"," ").replace("\n"," ")`; offsets e `text` das anotações copiados sem ajuste; relações órfãs descartadas | leitura do código e reprodução do resultado |
| Log da conversão | `RECLin-PT-original/logs/pipeline.log` | aviso `Offsets divergentes (anotacao != texto): 18079` | leitura |
| Dataset processado | `RECLin-PT-original/data/processed/dataset.jsonl` (sha `397ef424…`) | saída da conversão; é a `source` do MANIFEST atual | comparado documento a documento com as partições |
| Partições do legado | `RECLin-PT-original/data/splits/{train,dev,test}.jsonl` | mesmo SHA-256 das partições do projeto | hash |
| Partições congeladas | `codigo/dados/particoes/{train,dev,test}.jsonl`, `MANIFEST.json` | entrada de todo o projeto | somente leitura |
| Partição | `codigo/reclin/particoes.py` (`particionar`) | 80/10/10 estratificado por presença de `negation_of`; não usa o texto nem os offsets | reexecutado em memória sobre os dados corrigidos |
| Candidatos | `codigo/reclin/tarefa.py` (`iter_candidate_pairs`, `entity_gap`, `TAMANHOS_ESPERADOS`) | pares dependem **só** de `start`/`end`/`id` (distância ≤ 25) | reexecutado em memória |
| Janelas e marcadores | `codigo/reclin/entrada.py` (`janela_marcada`) | recorte `ctx_chars` e marcadores em volta de `texto[start:end]` | leitura; testes existentes |
| Léxico | `codigo/reclin/negacao/lexico.py` | usa o atributo `text` da entidade, não o texto do documento | leitura |
| Auditoria da etapa 7 | `codigo/reclin/analises/alinhamento.py` (`auditar`, `impacto_nos_pares`, `perfil_da_deriva`, `alinhar_entidade`) | métricas reproduzidas antes e depois da correção | importado somente para leitura |
| Teste dos números | `codigo/testes/unidade/test_marcadores_offsets.py` | fixa os números da auditoria nas partições congeladas | executado |

**Cadeia de transformação reconstruída e verificada (1.000/1.000 documentos):**

1. XML (bytes, CRLF) → parser XML: texto normalizado = bruto com `\r\n` → `\n`
   (nenhuma outra diferença: mesmo conteúdo, `len(bruto) − len(normalizado) =
   nº de CRLF` em todos).
2. `parse_semclinbr.py`: `\n` → espaço; offsets e `text` das anotações
   copiados do XML; relações copiadas (0 órfãs; 11.458 relações = 9.221 +
   987 + 1.250).
3. `dataset.jsonl` (`397ef424…`) → `particionar(seed=42)` → partições, iguais
   byte a byte às do legado e às congeladas do projeto.
4. Verificado: texto de cada partição = texto do parser com `\n` → espaço;
   entidades `(id, start, end, text)` = atributos do XML; relações = XML.

Arquivos desta investigação (fora das entradas dos experimentos):

- `codigo/docs/entregas/d9/d9_investigacao.py` — protótipo consolidado,
  somente leitura, que produz todos os números deste relatório
  (`--xml-dir <pasta dos XML> --codigo <pasta codigo/>`);
- `codigo/docs/entregas/d9/resumo.json` — a saída dele (só contagens e hashes,
  nenhum trecho do corpus).

Os scripts exploratórios e os arquivos corrigidos em memória ficaram numa
pasta temporária da sessão, fora das pastas do projeto e do repositório
original, e não são entrada de nada.

---

## 3. Causa-raiz

### Mecanismo confirmado

Exemplo (documento 8926): o texto começa com uma linha terminada em CRLF e a
anotação 1357 (`PCTE`) tem `start=43, end=47`, que é a posição correta no
texto com CRLF. No texto gravado, o CRLF virou um espaço; `texto[43:47]`
devolve o trecho deslocado de um caractere, e o deslocamento cresce (−1, −2,
… −6) a cada quebra de linha seguinte.

O comentário do parser diz que as quebras são trocadas "preservando o
comprimento". Isso só seria verdade se os offsets contassem cada quebra como
um caractere — o que vale para 101 documentos e não para 421.

### Hipóteses

| Hipótese | Situação | Evidência |
| --- | --- | --- |
| Texto alterado sem atualizar os offsets (normalização de fim de linha) | **Confirmada** | offsets no texto bruto com CRLF: 99,94% das anotações conferem (com o sistema do documento); no texto atual: 60,2% exatas; checagem do próprio legado: 18.079 divergentes, idêntico ao log |
| Remoção/substituição de outros caracteres | Descartada | normalizado = bruto com CRLF→LF em 1.000/1.000; diferença de comprimento = nº de CRLF em 1.000/1.000; 0 CR isolado, 0 LF isolado |
| Normalização Unicode | Descartada | nenhuma alteração além do fim de linha; nenhum caractere fora do BMP (UTF-16 contaria diferente); nenhum combinante |
| Remoção de cabeçalhos/segmentos | Descartada | idem (comprimentos explicados só pelos CRLF) |
| Espaços/pontuação | Descartada como causa | 9 documentos com TAB, preservados; as divergências de espaço estão no **atributo** `text` (1.093 com espaços inseridos em volta da pontuação, 171 só de espaços), com offsets corretos |
| Escape de entidades | Descartada como causa | 39 atributos com escape duplo (`&gt;` literal); texto e offsets corretos |
| Concatenação/separação de documentos; offsets de outro documento | Descartada | 1 XML por documento, mesmos `doc_id`; as entidades de cada documento conferem com o próprio texto em 99,94% |
| Offsets inclusivos × exclusivos | Descartada | `end` exclusivo (fatia `[start:end]`); 0 casos resolvidos com `end` inclusivo |
| Bytes × caracteres | Descartada | índices de caractere; em 39.407 anotações com texto não ASCII antes delas os índices de byte divergiriam e os de caractere conferem; 0 casos resolvidos por bytes |
| Deslocamento criado nas janelas ou pares | Descartada | o desalinhamento já está nas entidades das partições, antes de qualquer janela; `janela_marcada` usa os offsets como estão (testes da etapa 7) |
| **Por que a fonte tem duas convenções** (101 documentos contam a quebra como 1) | **Inconclusiva** | ver abaixo |

### Ponto inconclusivo

Nos 101 documentos de id 9015–9184 com CRLF, os offsets contam a quebra como
um caractere; em todos os outros 421 documentos com CRLF (e só fora dessa
faixa), como dois. Nenhum metadado do XML distingue os dois grupos. Os 1.000
arquivos usam CRLF inclusive nas linhas de estrutura fora do `<TEXT>` (mesmo
os 472 sem nenhuma quebra no texto). Isso é compatível com uma conversão de
fim de linha da cópia local (LF→CRLF em arquivos cujo lote 9015–9184 tinha
LF), mas também com um lote anotado ou exportado por outra ferramenta. Não dá
para decidir sem uma cópia íntegra da distribuição oficial, o que exigiria
acesso externo (fora do escopo). **Isso não muda a correção**: o sistema de
cada documento é determinado pelas suas próprias anotações, e a regra
coincide 100% com a faixa de ids.

---

## 4. Diagnóstico quantitativo

### Sistema de contagem por documento

Regra (determinística, no nível do documento, nunca por entidade): sem CRLF →
`sem_crlf`; com CRLF, conta as anotações que conferem na **mesma posição** no
texto bruto (CRLF = 2) e no texto atual (quebra = 1); vence o maior; empate
com todas conferindo → `indistinguivel`; empate com falhas → `ambiguo`.

| Partição | Docs | sem quebra | CRLF = 2 (`bruto`) | quebra = 1 (`lf`) | indistinguível | ambíguo |
| --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 800 | 365 | 346 | 85 | 4 | 0 |
| DEV | 100 | 52 | 36 | 10 | 2 | 0 |
| TEST (estrutural) | 100 | 55 | 39 | 6 | 0 | 0 |
| Corpus | 1.000 | 472 | 421 | 101 | 6 | 0 |

Os 6 indistinguíveis têm todas as entidades antes da primeira quebra de linha:
a correção não muda nenhum offset neles.

### Entidades — auditoria da etapa 7 reproduzida (TRAIN e DEV)

`alinhamento.auditar` do projeto, sem modificação. "Equivalente" = mesmo
texto a menos de espaços e escapes.

| Partição | Versão | exato | equivalente | deslocado | ausente | exatas |
| --- | --- | --- | --- | --- | --- | --- |
| TRAIN (36.455) | atual | 21.479 | 469 | 13.899 | 608 | 58,9% |
| | B (offsets mapeados) | 35.395 | 1.040 | 1 | 19 | 97,1% |
| | C1 (texto com comprimento preservado) | 35.368 | 1.067 | 1 | 19 | 97,0% |
| DEV (4.513) | atual | 2.970 | 33 | 1.435 | 75 | 65,8% |
| | B | 4.405 | 107 | 0 | 1 | 97,6% |
| | C1 | 4.403 | 109 | 0 | 1 | 97,6% |

Em B e C1 as mesmas entidades ficam alinhadas; a diferença entre "exato" e
"equivalente" vem só de C1 manter dois espaços onde havia CRLF dentro de 27
entidades do TRAIN e 2 do DEV.

### Pares e deriva (TRAIN e DEV)

| Medida | TRAIN atual | TRAIN corrigido | DEV atual | DEV corrigido |
| --- | --- | --- | --- | --- |
| Pares `negation_of` com entidade desalinhada | 672 / 1.255 (53,5%) | **0 / 1.255** | 58 / 150 (38,7%) | **0 / 150** |
| Documentos sem deriva / monótona / não monótona | 453 / 223 / 124 | 799 / 1 / 0 | 64 / 21 / 15 | 100 / 0 / 0 |

O único documento ainda "com deriva" no TRAIN é o da anotação com `start = -1`
(abaixo).

### Tipos de divergência que restam (todas da fonte)

Comparação **na posição** dada pelo sistema do documento:

| Tipo | Corpus | TRAIN | DEV | Offsets | Observação |
| --- | --- | --- | --- | --- | --- |
| exato | 44.179 | 35.368–35.395 | 4.403–4.405 | corretos | — |
| atributo com espaços em volta da pontuação | 1.093 | 892 | 84 | corretos | ex.: `MÉDIA QUANTIDADE .` para `MÉDIA QUANTIDADE.` |
| só espaços | 171 | 116–143 | 21–23 | corretos | — |
| escape duplo no atributo | 39 | 32 | 2 | corretos | ex.: `&gt;` literal |
| atributo com palavras duplicadas | 22 | 19 | 1 | posição consistente com o texto sem duplicação | ex.: `bom bom estado estado geral geral` sobre `bom estado geral` |
| trecho descontínuo / palavra omitida no atributo | 3 | 0 | 0 | — | só no TEST (estrutural) |
| `start = -1` | 1 | 1 | 0 | inválido | documento 9808, anotação 39461 |

Nenhuma das 21 anotações do TRAIN e do DEV nessas três últimas linhas (19 + 1
com palavras duplicadas e a de `start = -1`) participa de relação anotada. Das 5 falhas do TEST não foi
olhada a participação em relações (não é necessário para nenhuma decisão).

### Por que não realinhar por busca textual

Entre as entidades que a correção posicional torna exatas e cujo offset muda
(13.936 no TRAIN, 1.438 no DEV), o texto da entidade aparece mais de uma vez
no documento em 3.431 (TRAIN) e 323 (DEV). Escolher a primeira ocorrência
erraria 2.223 e 210 delas; a busca local da etapa 7 (ocorrência mais próxima
numa janela de 50 caracteres) erraria 225 e 21 e não encontraria 18. Isso
descarta a opção "realinhar procurando o `text` perto dos offsets"
registrada na etapa 7.

### Candidatos (por variante)

| Partição | Atual | B (offsets mapeados) | C1 (comprimento preservado) |
| --- | --- | --- | --- |
| TRAIN | 152.686 (1.255 / 7.096 / 144.335) | 154.878 (1.255 / 7.100 / 146.523) | 152.686, **lista idêntica** |
| DEV | 19.064 (150 / 763 / 18.151) | 19.314 (150 / 763 / 18.401) | 19.064, **lista idêntica** |
| TEST (estrutural) | 19.210 | 19.470 | 19.210, **lista idêntica** |

Rótulos na ordem `negation_of` / `associated_with` / `no_relation`, só TRAIN e
DEV. Em B a distância entre entidades encolhe uma unidade por quebra de linha
entre elas e entram mais pares dentro de `MAX_GAP = 25` (quase todos
`no_relation`; 4 `associated_with` no TRAIN). Em C1 os offsets não mudam,
então os candidatos (chaves, ordem e rótulos) também não mudam.

---

## 5. Viabilidade de recuperação

**Recuperável, de forma inequívoca e rastreável.** As sete ligações pedidas
ficam estabelecidas para cada entidade:

| Elo | Como fica identificado |
| --- | --- |
| 1. Documento original | arquivo `<doc_id>.xml` (mesmo `doc_id` da partição) |
| 2. Texto original | `<TEXT>` bruto, com CRLF |
| 3. Entidade original | `annotation@id` (único no documento; 0 ids duplicados) |
| 4. Offset original | `annotation@start/end`, copiados sem alteração em todas as partições |
| 5. Texto armazenado | `text` da partição = texto do parser com `\n` → espaço |
| 6. Offset atual | = offset original |
| 7. Trecho extraído hoje | `text[start:end]` — deslocado de −k (k = CRLF antes do offset) nos 421 documentos `bruto` |

A recuperação não procura texto: cada offset é transformado **pela sua
posição**, com a regra do sistema do documento. A identidade da anotação
(`id`), o documento e as relações (`e1_id`, `e2_id`, `type`) não são tocados.

Critérios de validação aplicados ao protótipo (TRAIN, DEV e TEST, as duas
variantes):

| Critério | Resultado |
| --- | --- |
| Trecho recuperado = texto anotado | 45.482/45.508 na mesma posição; as 26 restantes listadas acima como defeitos da fonte |
| Convenção dos offsets | caractere, `end` exclusivo, consistente em todas |
| Entidades repetidas e sobrepostas | ordem e sobreposição entre **todos** os pares de entidades de cada documento preservadas (0 alterações); nenhuma escolha entre ocorrências |
| Relações | idênticas, por documento, antes e depois (asserção) |
| Nada descartado ou reassociado | mesmas entidades, mesma ordem, mesmo `id`, `text` e `type` (asserção) |
| Casos ambíguos/irrecuperáveis explícitos | 0 documentos ambíguos; 0 offsets caindo entre `\r` e `\n`; 1 offset inválido e 25 atributos defeituosos listados |
| Determinismo | SHA-256 da saída igual em duas execuções, por partição e variante |
| Rastreabilidade | C1: offset = atributo do XML; B: `novo = original − nº de CRLF antes dele` (inversível) |
| Separação das partições | `particionar(seed=42)` sobre os documentos corrigidos devolve a mesma atribuição e a mesma ordem nas três partições |

---

## 6. Alternativas comparadas

| | **A — manter os dados** | **B — corrigir offsets na preparação** (texto atual, offsets mapeados) | **C — reconstruir da fonte** |
| --- | --- | --- | --- |
| Evidência | limitação quantificada (seção 4) | protótipo: 0 pares `negation_of` afetados | protótipo: idem; duas variantes (C1 e C2) |
| Viabilidade | imediata | alta (regra de 1 linha por offset) | alta (XML disponível localmente) |
| Risco de alterar o significado das anotações | nenhum nos dados, mas o modelo vê os marcadores em volta do trecho errado em ~40% das entidades | baixo: posição, não busca; offsets passam a diferir da fonte (rastreáveis pela fórmula) | C1: nenhum — offsets literalmente os da fonte; C2: igual a B |
| Entidades / relações | inalteradas | offsets mudam nas entidades dos documentos `bruto` precedidas por quebra de linha; relações iguais | C1: inalteradas; C2: como B |
| Janelas / marcadores | ~53% dos pares `negation_of` do TRAIN com marcador deslocado | corretos | corretos |
| Pares candidatos | inalterados | +2.192 (TRAIN), +250 (DEV), +260 (TEST) | C1: **inalterados**; C2: como B |
| TRAIN / DEV / TEST | inalterados | mesmos documentos, conteúdo muda | mesmos documentos; C1 muda só `text` de 346/36/39 documentos |
| MANIFEST, hashes, contagens, testes | inalterados | se feito em tempo de execução, as partições congeladas deixam de descrever a entrada real dos modelos (identidade ambígua); `TAMANHOS_ESPERADOS` e hashes dos candidatos mudam | nova versão com MANIFEST próprio; C1 mantém `TAMANHOS_ESPERADOS`, `candidatos_sha256` e `y_true_sha256`, muda `particao_sha256` |
| Nova versão formal dos dados | não | **sim** (recomendo não fazê-la só "em tempo de execução") | sim |
| Repetir experimentos | não | sim: todos os da fase (nenhum científico foi rodado ainda) | idem |
| Comparabilidade com o legado | total (mesma entrada defeituosa) | conjunto de referência diferente | C1: mesmo conjunto de referência (candidatos e rótulos); só as janelas diferem |

Observações sobre C1. (1) Nos 421 documentos `bruto` a quebra de linha passa a
ocupar dois espaços no texto; os tokenizadores do BERT normalizam espaços, então
a sequência de tokens é a mesma que a de um espaço só. (2) A distância usada
nos candidatos continua contando a quebra como 2 nesses documentos e como 1 nos
101 `lf` — é exatamente a convenção do conjunto de referência atual (e do
legado); B a uniformiza ao custo de mudar o conjunto de referência, e a
diferença entre as duas está quantificada acima.

---

## 7. Recomendação técnica

**Alternativa C, variante C1**, como nova versão formal dos dados.

1. **Fidelidade às anotações.** Os offsets ficam literalmente os do XML; o que
   se corrige é o erro de transformação (o texto), não as anotações. É a única
   alternativa em que a rastreabilidade é por igualdade, não por fórmula.
2. **Validade de construto.** Elimina o marcador em volta do trecho errado: 0
   pares `negation_of` afetados no TRAIN e no DEV, contra 53,5% e 38,7% hoje.
   A estratégia lexical (que já usa o `text` correto da entidade) e a janela
   passam a falar do mesmo trecho.
3. **Comparabilidade.** O conjunto de referência (candidatos, ordem, rótulos,
   `y_true`) é idêntico ao atual e ao do legado; muda só o que o modelo lê na
   janela. A mudança fica isolada na variável que se quer corrigir.
4. **Custo e momento.** Nenhum experimento científico foi rodado (etapa 8 não
   iniciada): adotar a correção agora não invalida nenhum resultado da fase.
5. **Por que não A:** declarar a limitação seria honesto, mas mantém um defeito
   conhecido, mensurado e corrigível na entrada de todos os modelos.
   **Por que não B:** mesma correção das janelas, porém muda o conjunto de
   referência e afasta os offsets da fonte, sem ganho de alinhamento.

---

## 8. Plano de implementação (não aplicado; depende de aprovação)

**Princípio:** a versão atual (v1) fica intocada, com o MANIFEST, os testes de
equivalência e as referências do legado; a versão nova (v2) fica em pasta
própria e passa a ser a da fase científica só depois de aprovada.

**Dados**

- `codigo/dados/particoes-v2/{train,dev,test}.jsonl` — mesmos documentos, na
  mesma ordem; `entities` e `relations` byte a byte iguais aos da v1; `text`
  reconstruído do XML (C1).
- `codigo/dados/particoes-v2/MANIFEST.json` — as chaves atuais mais:
  `versao` (proposta: `v2-texto-fonte`), `derivada_de` (SHA-256 do MANIFEST e
  das partições v1), `fonte_xml` (`709790d5…`, 1.000 arquivos), `regra` (texto
  da regra de C1), `documentos_alterados` por partição, `anomalias_fonte`
  (lista das 26 anotações com o tipo).
- O XML continua fora do versionamento (licença): `codigo/dados/brutos/`, já
  previsto no README das entregas.

**Código (lista provável)**

| Arquivo | Mudança |
| --- | --- |
| `reclin/corpus/semclinbr.py` (novo) | leitura do XML com o `<TEXT>` bruto; sistema por documento; reconstrução C1; registro de anomalias; erro explícito para documento `ambiguo` ou offset entre `\r` e `\n` |
| `scripts/converter_semclinbr.py` (novo) | XML → partições v2 usando a **atribuição de documentos da v1** (não refaz a partição) + MANIFEST v2 |
| `reclin/util/caminhos.py` | `PARTICOES_V2` / versão selecionável |
| `reclin/particoes.py` | conferência do MANIFEST v2 (chaves novas) |
| `scripts/preparar_dados.py` | conferência da v2 |
| `reclin/tarefa.py` | **nada** em C1 (`TAMANHOS_ESPERADOS` iguais); em B, novos tamanhos |
| `docs/protocolo/fase-biobertpt-seed42.md`, `docs/reproducao.md`, READMEs | hashes e versão dos dados da fase |

A identidade do código (D6) muda com o módulo novo, como esperado.

**Testes**

- Sintéticos do conversor: documento com CRLF contando 2; com quebra contando
  1; sem quebra; indistinguível; ambíguo (erro); offset entre `\r` e `\n`
  (erro); `start` negativo, palavras duplicadas e escape duplo (anomalia
  registrada, nada descartado).
- Integridade da v2 contra o MANIFEST v2.
- Invariantes v1 × v2: mesmos `doc_id` e ordem por partição; `entities` e
  `relations` idênticos; `text` igual fora dos documentos `bruto`; nos
  `bruto`, `text` v2 = `<TEXT>` com CRLF → dois espaços e `len(v2) − len(v1)`
  = nº de CRLF.
- Candidatos v2 = v1 (chaves, ordem, `y_true`) nas três partições.
- Alinhamento v2 no TRAIN e no DEV: tudo exato ou equivalente, exceto as
  anomalias listadas; 0 pares `negation_of` afetados.
- `test_marcadores_offsets.py`: números atuais mantidos como registro da v1;
  números da v2 acrescentados.
- Léxico: o `CONGELADO` não deve mudar (usa o `text` das entidades e os mesmos
  candidatos) — conferir o hash.
- Suíte completa; equivalência das janelas com o legado continua valendo só
  para a v1.

**Critérios de aceitação**

Todos os da seção 5 na v2 materializada; MANIFEST v1, partições v1 e
referências do legado com os mesmos SHA-256 de hoje; conversão reexecutada
duas vezes com saída byte a byte igual; protocolo atualizado apontando a v2.

**Separação e vazamento**

Mesma atribuição de documentos da v1 (verificada); a regra usa só o texto e as
anotações de entidade do próprio documento, sem parâmetro ajustado e sem
relação anotada; o TEST passa pela mesma regra sem inspeção dos seus rótulos
de relação e sem nenhum resultado de modelo.

---

## 9. Integridade

Verificado ao final, no computador do autor e na cópia de trabalho:

| Arquivo | SHA-256 | Confere com |
| --- | --- | --- |
| `particoes/train.jsonl` | `36014b2a…a437a` | MANIFEST e legado |
| `particoes/dev.jsonl` | `a9ea3276…c8aeb` | MANIFEST e legado |
| `particoes/test.jsonl` | `86304d8d…1e62` | MANIFEST e legado |
| `particoes/MANIFEST.json` | `a73f291a…f8ae1` | igual ao início da tarefa |
| legado `data/splits/MANIFEST.json` | `9734395…f5c6` | inalterado |
| legado `data/processed/dataset.jsonl` | `397ef424…fd6c` | `source` do MANIFEST |

- Nenhum arquivo do projeto ou do repositório original foi modificado pela
  investigação: a última modificação nas duas pastas, antes da gravação deste
  relatório, é a sincronização final da tarefa anterior (pendências da etapa 7). Os únicos arquivos criados no projeto são este relatório, a pasta
  `d9/` (protótipo e resumo) e a linha nova no índice das entregas.
- Testes executados: `test_marcadores_offsets.py` e `test_particoes.py` —
  **12 passaram**. A suíte completa não foi executada (nenhum código foi
  alterado).
- O protótipo importou `reclin.tarefa` e `reclin.analises.alinhamento` sem
  gravar cache de bytecode no projeto.

---

## 10. Decisões pendentes

1. **Alternativa:** aprovar C1 (nova versão v2, texto com comprimento
   preservado, conjunto de referência idêntico)? Ou prefere B/C2 (quebra
   contando 1 em todos os documentos, conjunto de referência novo, +2.192
   candidatos no TRAIN) ou A (manter e declarar)?
2. **Anomalias da fonte (26 anotações):** manter como estão e registrá-las no
   MANIFEST v2 (recomendado; nenhuma participa de relação no TRAIN e no DEV)
   ou excluí-las? Em particular a entidade com `start = -1` (TRAIN, documento
   9808): mantê-la mantém os candidatos idênticos; excluí-la muda o conjunto
   de referência.
3. **6 documentos indistinguíveis:** manter o texto atual (recomendado; nenhum
   offset muda) ou reconstruí-los também com dois espaços?
4. **Identificador e local:** `v2-texto-fonte` em `codigo/dados/particoes-v2/`,
   com a v1 preservada para o histórico e para os testes de equivalência?
5. **Proveniência da convenção mista:** quer comparar a cópia local do corpus
   com uma cópia íntegra da distribuição oficial (sem conversão de fim de
   linha)? É opcional e não bloqueia a correção.
6. **Anexo:** manter `codigo/docs/entregas/d9/` (protótipo e resumo) no
   projeto ou removê-lo depois da revisão?

D4 (`min_freq=3`) continua pendente e independente desta decisão. A etapa 8
não foi iniciada.
