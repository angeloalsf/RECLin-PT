# Etapa 4 — Auditoria integral do RECLin-PT (13/11/2026)

Escopo: as duas branches (`main` e `resultados-colab`), código, experimentos, TCC e
apresentação. Restrições seguidas: nenhum treino, nenhuma alteração de experimento ou
resultado gravado, nenhum comando Git no repositório local, nenhum commit ou push.
Cada afirmação abaixo traz o rótulo **[verificado]** (conferi nesta etapa),
**[inferido]** ou **[não verificado]**.

## 1. Como a auditoria foi feita

- **Pasta local** `D:\angeloalsf\tcc\RECLin-PT`: copiei para a nuvem 196 arquivos
  (TCC, slides, `reclin/`, arquivos de raiz). Não copiei `codigo/dados/`,
  `SemClinBr-xml-public-v1/` nem `.git/`. O ambiente Linux do desktop não iniciou,
  então todo o trabalho rodou na nuvem.
- **GitHub**, com sua autorização: clone só leitura, raso e sem blobs, de `main` e de
  `resultados-colab`, excluindo `codigo/dados/`, `testes/referencia/resultados_legado/`,
  `testes/referencia/treino/`, a planilha de adjudicação e os PDFs de terceiros.
- **Pasta local × `main` remoto** [verificado]: dos 196 arquivos copiados, todo o
  código é idêntico byte a byte ao `main`; 18 arquivos do TCC diferem (a pasta local
  tem edições de 08–09/11 ainda não enviadas) e 1 PDF só existe localmente.

### Incidente com o TEST (contido)

No primeiro clone da `resultados-colab` usei `git ls-tree -l`. Em clone sem blobs, a
opção de tamanho faz o Git baixar objetos sob demanda, e os dois `test.jsonl` de
`codigo/dados/` vieram como objetos compactados dentro de `.git`. O arquivo **não foi
extraído, aberto, lido nem processado**. Interrompi o comando, apaguei o clone inteiro
e refiz com exclusão explícita e com o download sob demanda desligado logo após o
checkout; conferi, pelo índice dos pacotes, que nenhum objeto excluído estava presente. Nenhuma métrica,
predição ou texto do TEST foi usado em qualquer análise desta etapa.

## 2. Achados críticos

**C1. Repositórios públicos com o corpus de acesso restrito.** [verificado]
`angeloalsf/RECLin-PT` e `angeloalsf/RECLin-PT-original` estão com visibilidade
**pública**. O `main` e a `resultados-colab` versionam `codigo/dados/` (partições
`train/dev/test.jsonl` com texto das notas), e o `.gitignore` diz que isso é
proposital. O README afirma que o corpus "não é distribuído aqui", e o TCC dizia em
nota de rodapé que "o corpus não é redistribuído". O SemClinBr só é obtido por
formulário de solicitação aos autores. Também estão públicos a planilha de adjudicação
(texto de notas, segundo o cofre), as saídas legadas com predições do TEST e os PDFs de
TCCs de terceiros. Consequências: possível violação dos termos do corpus e TEST exposto.
*Ação sua (não executei nada):* tornar os dois repositórios privados agora; depois
decidir sobre a limpeza do histórico. O conteúdo do repositório legado é [inferido].

**C2. Experimento executado e não registrado.** [verificado] A `resultados-colab`
termina em `pair_aware__a3ac671a77e2__t2` (08/11): Pair-Aware Linear, BioBERTpt-all,
semente 42, `class_weight="none"`. O notebook está em
`codigo/notebooks/teste/reclin_pair_aware_sem_pesos.ipynb`. Não constava do protocolo,
do inventário, do cofre nem do TCC. Resultado no DEV: P 0,794, R 0,873, F1 0,832 de
`negation_of` (VP/FP/FN 131/34/19), F1 de `associated_with` 0,448, macro-F1 0,752.
Contra a versão ponderada: F1 neg +0,002 [−0,028; +0,031] e macro +0,011
[−0,003; +0,025], nenhuma detectada (bootstrap por candidato). Os 34 FP são
subconjunto dos 49 da versão ponderada, e os 9 FN desta estão nos 19 da nova.

**C3. O TCC não tem avaliação no TEST e se descrevia como trabalho em andamento.**
[verificado] O quinto objetivo é cumprido em parte. É decisão sua e do orientador
entregar assim ou congelar e avaliar antes (ver plano de evolução).

## 3. Experimentos e resultados

- **Métricas** [verificado]: recalculei, com código próprio, precisão, recall, F1,
  macro-F1 e MCC das 11 execuções com predições de três rótulos na `resultados-colab`
  (17 pastas ao todo; as demais são estágios e artefatos do Two-Stage).
  Todas coincidem com o `metricas.json` gravado (diferença 0) e com a
  `tabelas/dev_sistemas.tex`. O `y_true` é o mesmo em todas. Nenhum `metricas.json`
  tem a chave `test`.
- **Bootstrap** [verificado]: minha réplica independente reproduz o registro
  Pair-Aware − baseline (BioBERTpt-all): +0,0272 [+0,0052; +0,0506].
- **Afirmações numéricas do texto** [verificado contra `comparacoes_dev.json`]: faixas
  Pair-Aware × Two-Stage (−0,001 a −0,007), `associated_with` (+0,071 e +0,074),
  encoders (−0,011 a +0,014), calibrado (+0,020), filtro (+0,011), b/c 249/261.
- **Probabilidades com 4 casas** [verificado]: os empates tornam a precisão média de
  `negation_of` imprecisa (baseline BioBERTpt-all: entre 0,820 e 0,875 conforme o
  desempate). Na de `associated_with`, relatada no TCC, o efeito é ≤ 0,003.
- **Não refiz** [não verificado]: bootstrap por documento, quase duplicatas, controles
  de tipo e léxico, porque dependem das partições, que não copiei.

## 4. Código

- 18,7 mil linhas em `reclin/`, `scripts/` e `diagnosticos/`; 10,9 mil em testes.
  Compila; `ruff` não acusa erro real (classe F). Há 2.283 linhas longas e 28 funções
  de complexidade alta. Métricas e significância estão corretas [verificado por leitura
  e por réplica].
- **Testes de unidade** [verificado, sem torch e sem `dados/`]: 601 passam, 19 são
  ignorados, 43 falham ou dão erro. Quase todos por ambiente. **Uma falha é real**:
  `test_sem_dependencia_do_legado_nem_de_git[reclin/analises/sensibilidade_semente.py]`
  proíbe a palavra `referencia` no pacote, e esse módulo usa uma chave com esse nome.
  A suíte completa (torch, dados, integração) **não foi executada**.
- **Acoplamento**: `reclin.estrategias.pair_aware` importa torch no carregamento, e
  dois módulos de teste de relatórios não são coletados sem torch.
- **Guarda do TEST**: `scripts/avaliar_test.py` registra a avaliação em trilha, mas a
  trilha é registro, não trava. Nada impede tecnicamente uma avaliação acidental.
- **Dependências** [do cofre, não reconferido]: `scikit-learn` fixado e não usado;
  `lxml` e `matplotlib` sem declaração.
- **Documentação defasada**: README de raiz ainda diz que `tcc/` não foi incluída.
- Sem credenciais expostas nos arquivos analisados [verificado por busca de padrões].

## 5. TCC

Compilação [verificado na nuvem, prévia]: 102 páginas antes, 103 depois; 0 referência
indefinida, 0 aviso do BibTeX, 0 caixa estourada.

| # | Achado | Situação |
|---|---|---|
| T1 | "Única diferença" entre baseline e Pair-Aware: há duas (marcadores e pooler) | corrigido |
| T2 | Concordância de 0,57 citada na discussão sem estar no método | incluída na Seção 4.5.3 |
| T3 | Quadro de ambiente sem a revisão do BioBERTpt-clin (`b59beec`) | corrigido |
| T4 | Linguagem de gestão ("arquitetura oficial", "emenda E3", datas de registro) | reescrito |
| T5 | Experimento sem pesos ausente | nova Seção 6.8 e Tabela 15 |
| T6 | Sem seção de organização do trabalho | incluída na Introdução |
| T7 | Rodapé com link para o repositório e afirmação falsa sobre o corpus | reescrito, sem URL |
| T8 | Notas de tabelas prometiam "comparações pré-registradas no teste" | alinhadas (tabelas e gerador) |
| T9 | Banca, coorientadora, data e ficha com texto do modelo ("Fulana de Tal", "XXXX") | **pendente, seu** |
| T10 | `agradecimentos.tex` existe e não é impresso no `main.tex` | **pendente, seu** |
| T11 | 9 tabelas órfãs em `tabelas/`; 1 referência não citada (`sechidis_2011`) | pendente |
| T12 | PDF sem título nem autor nos metadados | pendente |

Referências [verificado]: as 20 entradas citadas existem no `.bib`. Conferi na web a
existência das duas de 2026 (Shimaoka et al., *Research on Biomedical Engineering*;
Almeida et al., arXiv 2603.26510). Os números atribuídos às fontes (contagens de
estudos, F1 de Torres e Ramos) **não foram reconferidos** nesta etapa.

### Marcas d'água e artefatos de escrita

[verificado] Varredura de todos os `.tex`, `.bib`, `.md`, `.py` e notebooks: **nenhum
caractere invisível** (zero-width, BOM, espaços exóticos). Metadados do PDF limpos. A
skill `watermarks-remover` não foi necessária e não foi executada: a parte dela que
remove caracteres ocultos não teria o que remover, e a parte que reescreve texto para
confundir detectores eu não uso nem garanto. O que havia de artificial era estilo, e
foi tratado na reescrita. Restam, por decisão sua: as pastas `Claude outputs/` (duas,
versionadas) e a menção em `tcc/tcc/README.md`. O PPTX antigo trazia "PptxGenJS" nos
metadados; o novo traz título, autor e instituição.

Uso de IA: se o Ifes ou o orientador exigirem declaração de uso de ferramentas de IA
generativa, ela ainda não está no texto. Vale confirmar a regra antes do depósito.

## 6. Apresentação

Deck antigo [verificado]: 23 slides, cerca de 2.700 palavras, corpo em 11–12 pt,
metadados de título dizendo "10 min" e link para o repositório público. Deck novo: 13 slides e 2
de apêndice, corpo em 14 pt ou mais, dois gráficos nativos, notas do orador, sem o
link. Números conferidos contra as tabelas do TCC.

## 7. Arquivos

**Modificados** em `tcc/tcc/src/`: `pre_textuais/resumo.tex`; `textuais/introducao.tex`,
`metodologia.tex`, `proposta_prototipo.tex`, `experimentos_e_resultados.tex`,
`discussao.tex`, `conclusao.tex` (reescrita integral); `apendices/apendice_e_exploratoria.tex`,
`apendice_f_inventario.tex`; `tabelas/dev_sistemas.tex`, `dev_exploratoria_mlp.tex`,
`dev_comparacoes_ts.tex`, `dev_comparacoes_pa.tex`, `dev_comparacoes_encoders.tex`
(só o texto das notas). Em `tcc/analise/`: `gerar_tabelas_etapa3.py` (as mesmas frases).

**Criados**: `tcc/tcc/src/tabelas/dev_sem_pesos.tex`; `tcc/analise/gerar_tabela_sem_pesos.py`;
`tcc/tcc/revisao/TCC_revisado_previa.pdf` e `TCC_revisado_diff.pdf`;
`apresentacao/RECLin-PT_apresentacao_revisada.pptx` e `.pdf`; este relatório e
`etapa4-recomendacoes-e-plano-de-evolucao.md` em `codigo/docs/entregas/`.

**Removidos**: nenhum. Nenhum resultado, dado ou artefato histórico foi sobrescrito.

## 8. Limitações desta etapa

- Os PDFs são **prévias**: a nuvem não tinha o abnTeX2 nem o babel em português.
  Usei o abnTeX2 oficial e arquivos substitutos escritos por mim para o idioma. O PDF
  oficial deve sair do seu Docker.
- O PDF de diferenças é **por parágrafo** (vermelho removido, verde novo), feito com
  código próprio; o `latexdiff` foi bloqueado no meu ambiente.
- As comparações do experimento sem pesos usam só o bootstrap por candidato.
- Não auditei o conteúdo do repositório legado nem rodei a suíte completa de testes.

## 9. Decisões que dependem de você

1. Tornar os repositórios privados e tratar o histórico (C1).
2. Manter ou retirar a Seção 6.8 (experimento sem pesos) do TCC.
3. Ponto de operação da configuração final: perda ponderada ou sem pesos.
4. Quando congelar e avaliar no TEST, e se o TCC é entregue antes disso.
5. Declaração de uso de IA, dados da banca, ficha catalográfica e agradecimentos.
6. Destino das pastas `Claude outputs/` e das tabelas órfãs.
