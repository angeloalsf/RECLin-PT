# Etapa 7 — análises, relatórios e pontos de entrada (e revisão dos critérios do restrito e da Pair-Aware)

> **Atualização (11/10/2026):** as pendências D2, D3, D4, D5, D6 e D9 desta nota foram tratadas em
> [etapa-7-pendencias.md](etapa-7-pendencias.md) (teto do restrito e da Pair-Aware passou a 2 épocas;
> identidade do código; filtro no modo da fase). O texto abaixo é o registro da etapa 7 como entregue.

Data: 11/10/2026. Sem pacote: as alterações foram feitas direto na pasta de
trabalho. Nenhum comando de controle de versão foi executado; nenhum treino
científico, nenhuma avaliação científica do TEST, nenhum acesso à rede e
nenhuma instalação de dependência.

A etapa teve duas frentes, nesta ordem: (1) a revisão dos critérios de seleção
e de parada do restrito e da Pair-Aware para a fase atual (BioBERTpt, seed 42)
e (2) a etapa 7 do plano de migração, adaptada à decisão de escopo: **o
projeto novo não incorpora nem usa dados, resultados ou artefatos do legado**.

## 1. Revisão dos critérios do restrito e da Pair-Aware

Fontes lidas: o protocolo (`docs/protocolo/fase-biobertpt-seed42.md`), a nota
de 11/10 do mapa de migração (C1–C14), `reclin/estrategias/restrito.py`,
`reclin/estrategias/pair_aware/`, `reclin/treino/classificador.py`,
`reclin/treino/laco.py`, `reclin/execucao/subconjunto.py`,
`reclin/negacao/lexico.py`, `scripts/treinar.py`, `scripts/pretreino.py` e as
notas das etapas 5, 6 e 6.1. Nenhum resultado histórico foi usado para julgar
os critérios.

| # | Ponto | Restrito | Pair-Aware |
| --- | --- | --- | --- |
| 1 | Métrica primária da escolha do checkpoint | macro-F1 do DEV **completo**, com as predições do espaço restrito remapeadas ao conjunto completo (fora do espaço: `no_relation`, `[0, 0, 1]`) — `dev_macro_f1`, calculado em `classificador._pontuador` | igual ao restrito |
| 2 | Secundária e desempates | não há métrica secundária; empate → a primeira época (máximo estrito `>` em `laco.treinar`; `laco.melhor_epoca` reconstrói a escolha). `dev_negation_of_f1` e `dev_restrito_*` ficam no histórico só como informação | igual |
| 3 | Épocas máximas | 10 (`ConfigRestrito.epochs`) | 10 (`ConfigPairAware.epochs`) |
| 4 | Parada antecipada | não há: o laço roda todas as épocas; `parar_apos_passo` é só interrupção para retomada, e um treino não concluído não pode ter o TEST avaliado | igual |
| 5 | Dados de cada decisão | léxico: induzido do TRAIN congelado (`min_freq=3`, conferido pelo tamanho e pelo `lexico_sha1`); espaço restrito: pares com e1 pista, em cada partição; pesos `balanced`: TRAIN restrito; época: DEV completo remapeado; TEST: só `avaliar_test`, depois do treino concluído, refazendo a seleção a partir do léxico gravado e sem ler o TRAIN | igual; a contagem de marcadores ausentes é feita no TRAIN e no DEV antes do treino e no TEST só na avaliação |
| 6 | Compatível com a fase (BioBERTpt, seed 42) | sim: os critérios valem por execução e não dependem do número de sementes; os padrões são seed 42 e BioBERTpt; `pretreino.py conferir --execucoes` exige a configuração do protocolo | sim |
| 7 | Dependência de várias sementes ou de decisão pendente | nenhuma de várias sementes: os "portões de pesquisa" do legado (`criterio_parada.py`, `criterio_parada_pair_aware.py`, multissemente) foram retirados pela C10 (`portao_de_pesquisa: false`). Há dois pontos de origem histórica, registrados abaixo (D4, D5) | igual |
| 8 | Protocolo × código | conformes: o bloco JSON do protocolo (10 épocas, `dev_macro_f1`, seleção remapeada, remapeamento) é conferido por `test_protocolo_fase.py`; o comportamento é verificado agora, sem o legado, por `test_criterios_restrito_pair_aware.py` | conformes |
| 9 | Influência do TEST | nenhuma: o treino lê só TRAIN e DEV (verificado com a pasta sem o arquivo do TEST e com a leitura das partições instrumentada); seleção e parada usam só o DEV; `avaliar_test` recusa treino incompleto e execução de validação, e uma segunda avaliação exige `reavaliar` e fica na trilha | igual |

**Conclusão: os critérios estão corretos e conformes ao protocolo; nada foi
alterado** nos critérios, no protocolo ou no código das estratégias.

Evidência nova, independente do legado
(`testes/integracao/test_criterios_restrito_pair_aware.py`, 7 testes): com um
BERT sintético criado no próprio teste e 30/10 linhas do TRAIN e do DEV (sem o
arquivo do TEST), os dois treinos rodam todas as épocas; a época escolhida é o
primeiro máximo de `dev_macro_f1`; esse valor é o macro-F1 recalculado do
sidecar remapeado ao DEV completo (e o `dev_restrito_macro_f1` é o do
subconjunto); só `train` e `dev` são lidos. Nessa execução, o critério do DEV
completo escolhe a época 1 e o do DEV restrito escolheria a 2 — por isso a
mutação "selecionar pelo DEV restrito" foi detectada (2 falhas); antes, essa
propriedade só era coberta pela equivalência com o legado.

## 2. Etapa 7: critério original, escopo aplicado e entregas

**Critério de aceitação original** (mapa, seção 7.1): *"Alimentados com os
sidecars do legado (convertidos só num diretório de teste), geram as tabelas
atuais do TCC com os mesmos números; conferência passa."*

**Divergência de escopo (registrada, não implementada):** esse portão é uma
reprodução histórica e fica **fora do escopo** pela decisão desta etapa (o
projeto novo não usa sidecars nem números do legado). As 26 comparações
históricas também não são critério. A reprodução técnica das etapas 1 a 6
continua só nos testes de equivalência, sem relação com as análises novas.

**Critérios aplicados** (a infraestrutura nova, verificável sem treino
científico e sem TEST):

| Critério | Situação | Evidência |
| --- | --- | --- |
| Análises e relatórios no nível do núcleo, sem importar `negacao`/`estrategias`, importáveis sem efeito colateral | atendido | `test_pacote.py` (descobre os módulos novos sozinho) |
| Só partições congeladas e execuções do código novo com proveniência verificada; recusa das referências e do formato do legado | atendido | `test_inventario.py`; 11 mutações detectadas |
| Números recalculados das predições; sistemas ausentes como `pendente`, sem valores fictícios | atendido | `test_relatorios.py`, `test_relatorios_ponta_a_ponta.py` |
| As tabelas nunca avaliam um TEST pela primeira vez (só com a trilha) | atendido | idem (gerar antes do TEST: tudo pendente; filtro sem trilha: pendente) |
| Naturezas (científica, validação, fora da fase) e sementes nunca misturadas | atendido | idem |
| Comparações do protocolo (seção 7) iguais às de `significancia.comparar` | atendido | ponta a ponta, com execuções reais do código novo em CPU |
| Scripts finos, `Makefile` e notebook único; passos da fase científica só com autorização explícita | atendido | `test_entradas_e_legado.py`, `test_pacote.py` (`--help`) |
| Mecanismo de conferência do TCC (caminhos antigos, afirmações, validade dos valores) | atendido (mecanismo) | `test_relatorios.py`, ponta a ponta |
| Conferência real do texto do TCC | **pendente** | a pasta `tcc/` não está no repositório; a lista de afirmações é da etapa 9 |
| Figuras | **pendente (não verificado)** | o matplotlib não está instalado e não pôde ser instalado; o código importa-o só quando usado e registra a pendência |
| Tabelas atuais do TCC com números do código novo | **pendente** | dependem das execuções da etapa 8 |
| Documentação dos comandos | atendido | `docs/reproducao.md`, `codigo/README.md` |
| Suíte de regressão | ver seção 5 | |

**Entregas, pelo mapeamento da seção 1 do plano** (a finalidade de cada
script do legado, reimplementada a partir do código atual; nada copiado nem
importado):

| Plano | Entregue |
| --- | --- |
| `auditoria_marcadores.py` → `analises/marcadores.py` | marcadores nas janelas (ok, entrelaçados, inválidos) e depois da tokenização (sem `[E1]`/`[E2]` no truncamento), por rótulo |
| `align_entities` → `analises/alinhamento.py` | offsets × texto: exato, equivalente (espaços/entidades HTML), deslocado (com o deslocamento) ou ausente |
| `audit_residual_fp.py` → `analises/falsos_positivos.py` | falsos positivos de `negation_of` de uma execução verificada: origem, pista do e1 (léxico passado de fora), distância, forma, alinhamento |
| `analyze_max_gap.py` → `analises/max_gap.py` | candidatos e cobertura das relações por limite de distância (TRAIN e DEV) |
| `make_tcc_eda.py` → `analises/corpus.py` + `relatorios/eda.py` | estatísticas por partição e 6 tabelas da EDA |
| `_artifacts.py` (formatação) → `util/formatacao.py` + `relatorios/formatacao.py` | números (casas, vírgula, sinal, %); LaTeX `booktabs` ABNT, Markdown e CSV; marcadores `pendente` e `n/c` |
| `_artifacts.py` (`load_*`) → `execucao` | `execucao/inventario.py`: inventário e verificação de proveniência |
| `make_tcc_artifacts.py` → `relatorios/tabelas.py` + `scripts/gerar_tcc.py` | resultados no TEST, diagnóstico no DEV, comparações, limiares, proveniência, partições, registro de valores (`\rv{chave}`) |
| `make_tcc_curves.py` → `relatorios/curvas.py` | histórico por época com a época escolhida |
| `make_figures.py` + 3 `apply_style` → `relatorios/figuras.py` (`estilo(tamanho_fonte)`) | curvas e barras (matplotlib opcional) |
| `make_fase2_report.py` → `relatorios/markdown.py` | relatório da fase em Markdown |
| `check_tcc_numbers.py` → `relatorios/conferencia.py` + `scripts/conferir_tcc.py` | mecanismo: caminhos antigos (lista da seção 5.3), afirmações contra o registro de valores e validade das origens |
| `run.sh` → `codigo/Makefile` | alvos de testes, análises, relatórios e conferência; alvos da fase científica protegidos por `AUTORIZADO=1` |
| 5 notebooks → 1 parametrizado | `codigo/notebooks/reclin.ipynb` (BioBERTpt, seed 42; passos científicos com `AUTORIZADO[...]`) |
| READMEs das frentes → `docs/estrategias/` | `docs/estrategias/README.md`, escrito a partir do código novo (os READMEs do legado não foram copiados) |
| `aggregate_seeds.py` → `scripts/agregar.py` | **não implementado**: a agregação entre sementes saiu da fase (nota de 11/10); `avaliacao/agregacao.py` já existe desde a etapa 2 |

Fora da etapa 7 por definição do plano: critérios de decisão (`avaliacao/decisao`,
`decidir.py`), modelo final (etapa 10), atualização do TCC (etapa 9).

## 3. Arquivos

Criados (32):

| Arquivo | Finalidade |
| --- | --- |
| `codigo/reclin/execucao/inventario.py` | inventário das execuções, verificação de proveniência, naturezas, recusa do legado |
| `codigo/reclin/util/formatacao.py` | formatação de números (sem LaTeX) |
| `codigo/reclin/analises/__init__.py`, `alinhamento.py`, `corpus.py`, `max_gap.py`, `marcadores.py`, `falsos_positivos.py` | análises descritivas |
| `codigo/reclin/relatorios/__init__.py`, `fase.py`, `formatacao.py`, `tabelas.py`, `curvas.py`, `eda.py`, `figuras.py`, `markdown.py`, `conferencia.py` | apresentação dos resultados |
| `codigo/scripts/gerar_tcc.py` | inventário, tabelas, curvas, EDA, valores e relatório |
| `codigo/scripts/conferir_tcc.py` | conferência do texto do TCC |
| `codigo/Makefile` | pontos de entrada |
| `codigo/notebooks/reclin.ipynb` | notebook único parametrizado |
| `codigo/docs/reproducao.md` | comandos, origem dos dados, o que é gravado e as regras |
| `codigo/docs/estrategias/README.md` | resumo das estratégias do código novo |
| `codigo/docs/entregas/etapa-7.md` | esta nota |
| `codigo/testes/integracao/conftest.py` | BERT sintético criado no teste (sem artefatos do legado) |
| `codigo/testes/integracao/test_criterios_restrito_pair_aware.py` | frente 1 (7 testes) |
| `codigo/testes/integracao/test_relatorios_ponta_a_ponta.py` | etapa 7 de ponta a ponta com execuções reais do código novo em CPU, sobre um TEST sintético (9 testes) |
| `codigo/testes/unidade/conftest.py` | partições e execuções sintéticas |
| `codigo/testes/unidade/test_inventario.py`, `test_analises.py`, `test_relatorios.py`, `test_entradas_e_legado.py` | unidades da etapa 7 |

Modificados (4): `README.md` (estado do projeto e comandos da etapa 7),
`codigo/README.md` (seção da etapa 7, dependências opcionais, descrição das
pastas de testes), `codigo/scripts/analisar.py` (subcomandos `corpus`,
`max-gap`, `alinhamento`, `marcadores`, `falsos-positivos`; o `lexico` ficou
como estava) e `codigo/docs/entregas/README.md` (linha da etapa 7). Nenhum
teste existente foi alterado; partições, MANIFEST, referências e protocolo
estão intactos (conferido byte a byte contra a cópia anterior à etapa).

## 4. Proveniência: como as análises não consomem resultados históricos

- Uma única porta de entrada para resultados: `execucao.inventario`. Ela lê só
  `<raiz>/<nome>/config.json` e os sidecars da própria execução; não usa
  `diretorio.caminho_predicoes` (que aceita o formato do legado) e recusa,
  sem opção, `codigo/testes/referencia/` e os nomes de arquivo do legado.
- Cada sidecar é conferido contra o conjunto de referência registrado e a
  partição contra o MANIFEST; as métricas gravadas precisam ser as
  recalculadas; o TEST só com trilha; dependências presentes e coerentes
  (semente, proveniência, marca de validação).
- Cada número publicado leva as execuções de origem (com o SHA-256 dos
  sidecars e do `config.json`) no índice e no registro de valores;
  `conferir_tcc.py` recusa valores cujas execuções mudaram depois da geração.
- `test_entradas_e_legado.py` confere, nos arquivos da etapa 7, a ausência de
  referências às pastas e funções da reprodução do legado e de comandos de
  controle de versão.

## 5. Validações executadas

| Verificação | Resultado |
| --- | --- |
| Testes novos (unidade e integração da etapa 7 e da frente 1) | 84 passaram, 1 pulado (figuras: sem matplotlib), em 1 min 20 s; repetidos na versão final com `test_pacote.py` e `test_protocolo_fase.py`: 184 passaram, 1 pulado |
| Mutações (inventário, tabelas, conferência) | 11/11 detectadas |
| Mutação da seleção do restrito/Pair-Aware (DEV restrito no lugar do remapeado) | detectada |
| `test_pacote.py` (níveis, importação sem efeito colateral, `--help` de todos os scripts) | 95 passaram |
| Suíte completa, numa cópia isolada da pasta (Python 3.13, CPU, 1 thread) | **677 passaram, 2 pulados** em 26 min (antes da etapa: 574 e 1). Por pasta: `unidade/` 370 + 1 pulado, `integracao/` 82, `equivalencia/` 225 + 1 pulado. Os 103 a mais são os 84 testes novos e 19 gerados automaticamente por `test_pacote.py` (17 módulos novos na regra de níveis e 2 scripts novos no `--help`). Depois dessa rodada mudaram só o texto de ajuda e o nome de um alvo do `Makefile` (`testes-sem-equivalencia`) e as notas de entrega; os testes do `Makefile` e do pacote foram repetidos depois disso |
| Partições, MANIFEST, referências e protocolo | idênticos à cópia anterior à etapa (SHA-256 e comparação byte a byte); `preparar_dados.py conferir`: as três partições conferem |
| `gerar_tcc.py tudo` sobre a raiz de execuções vazia (o estado atual) | todas as tabelas de resultados `pendente`; EDA completa; conferência `pendente` (saída 2) |

Os testes de equivalência (`testes/equivalencia/`) rodaram dentro da suíte
como **regressão** das etapas 1 a 6; não são evidência de validação
experimental do projeto novo e não foram adaptados.

Limitações: Windows e GPU não executados; o notebook não foi executado num
Jupyter (as células de parâmetros, preparação e as guardas da fase científica
foram executadas nos testes); o `make` foi testado no Linux; figuras não
verificadas.

## 6. Achado das análises (dados congelados, só TRAIN e DEV)

`analisar.py alinhamento` mostra que muitas entidades não têm o próprio texto
nos offsets: no TRAIN, 58,9% exatas, 1,3% equivalentes (espaços ou entidades
HTML), 38,1% deslocadas (o texto aparece perto, com deslocamentos pequenos e
quase sempre negativos, o padrão de uma deriva de offsets) e 1,7% ausentes; no
DEV, 65,8% exatas. Entre os pares candidatos `negation_of`, 53,6% (TRAIN) e
38,7% (DEV) têm ao menos uma entidade não alinhada. Como as janelas marcadas
usam os offsets como estão, os marcadores desses pares envolvem um trecho
deslocado. **Nada foi alterado** (corrigir mudaria os dados congelados e a
tarefa); é um ponto para o autor avaliar (D9).

## 7. Pendências e decisões que dependem do autor

- **D1 — portão original da etapa 7** (reprodução das tabelas históricas):
  fora do escopo por decisão; registrado aqui e na nota do mapa.
- **D2 — artefatos do legado dentro do projeto novo.**
  `codigo/testes/referencia/` contém 74 arquivos de `results/` do legado
  (`resultados_legado/`), sidecars de treinos de referência e o modelo
  minúsculo, usados pelos testes de equivalência e por testes de integração
  das etapas 5, 6 e 6.1. Isso contraria "o projeto novo não deve conter
  artefatos do legado", mas a instrução também manda não apagar o legado e
  preservar o que não é do escopo. Nada foi movido nem apagado; o inventário
  os recusa e os testes novos não os usam. Opções: (a) manter como referência
  de teste isolada (como está, documentado); (b) tirá-los do repositório e
  tornar a equivalência opcional (pulada sem as referências); (c) outra.
- **D3 — filtro léxico na fase (bloqueia o passo 3 da etapa 8).**
  `scripts/filtro_pistas.py` exige `--calibracao`, e o único arquivo existente
  é o `CALIBRACAO_filtro.json` do legado; gerar outro com `calibrar.py` seria
  a recalibração suspensa pela C4 (e pode escolher um `min_freq` diferente de
  3, que o filtro recusa). Além disso, a avaliação do TEST do filtro
  (`avaliar.py`) não grava trilha, então o inventário não usa o TEST do filtro
  (fica `pendente` nas tabelas). Ajuste possível (não feito, muda código da
  etapa 4 e o fluxo da etapa 8): o filtro usar o léxico congelado sem arquivo
  de calibração e a avaliação do TEST do filtro registrar a trilha.
- **D4 — `min_freq=3`.** O léxico congelado (C4) vem de uma calibração
  histórica sobre predições do legado no DEV. Não é recalibrado nem escolhido
  agora, mas a origem é histórica; afeta restrito, Pair-Aware e filtro.
  Manter a C4 (parâmetro fixado a priori, declarado) ou recalibrar no DEV do
  baseline novo é decisão do autor.
- **D5 — justificativas do protocolo.** O teto de 10 épocas do restrito e da
  Pair-Aware é o da configuração do legado (o protocolo não traz justificativa
  própria, ao contrário do estágio 2), e a seção 3 justifica o teto do
  baseline e do estágio 1 com a "documentação histórica". Nenhuma escolha usa
  resultado histórico agora, mas a redação depende dele; mudar o protocolo
  exige aprovação.
- **D6 — versão do código.** As execuções não registram a versão do código
  (nenhum hash); o inventário não detecta execuções de versões diferentes.
  Registrar um hash do pacote no `config.json` antes da etapa 8 mudaria o
  formato e a conferência da retomada: decisão do autor.
- **D7 — escopo na conferência final.** `pretreino.py conferir --execucoes`
  exige a mesma semente e a configuração do protocolo, mas não exige seed 42
  nem o BioBERTpt; o inventário só classifica como científica uma execução
  com seed 42 e BioBERTpt. Observação, sem mudança.
- **D8 — TCC.** A pasta `tcc/` não está no repositório: conferência real,
  citações de caminhos antigos e lista de afirmações ficam para a etapa 9
  (C11). Os números do texto atual são do legado e não são resultados atuais.
- **D9 — desalinhamento dos offsets** (seção 6).
- **D10 — figuras** não verificadas (matplotlib ausente).

## 8. Separação do legado

Nenhum arquivo do legado foi copiado, importado, recuperado ou recriado;
nenhum sidecar, métrica, limiar ou critério histórico entra nas análises; os
testes novos usam um modelo sintético e partições sintéticas ou subconjuntos
do TRAIN e do DEV congelados (o TEST real não é lido por nenhum teste novo).
O legado continua onde estava (repositório original e
`codigo/testes/referencia/`), sem alteração.

## 9. Estado da etapa 7

**Parcialmente concluída.** A infraestrutura nova está implementada e
verificada (critérios "atendido" da seção 2). Ficam pendentes, por depender de
recursos indisponíveis ou de etapas posteriores: as figuras (matplotlib), a
conferência real do TCC (pasta `tcc/`, etapa 9) e as tabelas com números do
código novo (execuções da etapa 8). O portão original foi retirado do escopo
por decisão (D1).

## 10. Preparação para a etapa 8 (não iniciada)

Pronto: critérios do restrito e da Pair-Aware revisados; `make` e o notebook
com os passos do protocolo, cada um protegido por autorização; o inventário,
as tabelas, as comparações, os limiares, as curvas e o relatório prontos para
ler as execuções conforme forem gravadas (`gerar_tcc.py tudo`; para o passo 1V,
`--natureza validacao`). Bloqueios antes de começar: D3 (filtro) e as decisões
D2, D4, D5 e D6.
