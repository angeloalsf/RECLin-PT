# Etapa 7 — saneamento das pendências antes da etapa 8 (D2, D3, D4, D5, D6, D9)

Data: 11/10/2026. Tarefa de saneamento técnico e metodológico; **não inicia a
etapa 8**. Alterações feitas direto na pasta de trabalho. Nenhum comando de
controle de versão, nenhum acesso à rede, nenhuma instalação, nenhum treino
científico e nenhuma avaliação científica do TEST. Partições, MANIFEST,
referências do legado e resultados existentes não foram alterados.

## Resumo

| Pendência | Estado | Evidências | Arquivos inspecionados | Alterações | Testes | Decisão que depende do autor |
| --- | --- | --- | --- | --- | --- | --- |
| **D2** legado × experimentos | **resolvido** | O inventário já recusava `testes/referencia/` e os nomes do legado; havia um caminho indevido: o modo de reprodução do filtro e da regra pura cria execuções com `config.json` a partir de sidecars e da calibração do legado, e um filtro assim era aceito se a base estivesse no inventário | `execucao/inventario.py`, `scripts/filtro_pistas.py`, `scripts/regra_pura.py`, `scripts/calibrar.py`, `scripts/comparar.py`, `scripts/avaliar.py`, `execucao/diretorio.py`, `testes/equivalencia/*`, `testes/conftest.py` | inventário recusa configuração com arquivo de calibração externo, filtro sem o vínculo `base_sha256` com as predições atuais da base e execução treinada sem `treino.json`; marcador `referencia_legado` e resumo próprio dos testes históricos | unidade e integração (seção 5); 3 mutações detectadas | nenhuma |
| **D3** filtro: execução e registro do TEST | **resolvido (técnico)** | `filtro_pistas.py` exigia `--calibracao` (o único arquivo é o do legado) e não conseguia gravar o TEST numa execução já criada no DEV; o TEST do filtro (avaliado por `avaliar.py`) não gravava trilha | `estrategias/filtro_pistas.py`, `scripts/filtro_pistas.py`, `scripts/avaliar.py`, `scripts/calibrar.py`, `negacao/lexico.py`, protocolo, `testes/equivalencia/test_estrategias_legado.py` | modo da fase em `filtro_pistas.py` (`--base-execucao`): léxico congelado sem arquivo de calibração, DEV e TEST em passos separados, TEST só depois do TEST da base, uma vez, com trilha; `base_sha256` em cada sidecar; o modo de reprodução ficou como estava | integração do filtro de ponta a ponta; equivalência do filtro (byte a byte) | D4 |
| **D4** `min_freq=3` | **bloqueado (decisão metodológica)** | `min_freq=3` vem da calibração do legado no DEV, sobre as predições dos quatro baselines do legado; o protocolo aprovado (C4) o mantém como parâmetro fixo, mas a origem é um resultado histórico | `negacao/lexico.py`, `estrategias/filtro_pistas.py` (`calibrar_min_freq`), `scripts/calibrar.py`, protocolo, nota da etapa 4 | nenhuma (não recalibrado, nenhum valor escolhido) | — | manter a C4 ou recalibrar no DEV do baseline novo (seção D3/D4) |
| **D5** teto de épocas | **resolvido** (com uma divergência para confirmar) | `ConfigRestrito` e `ConfigPairAware` tinham 10 épocas; scripts, `pretreino.py conferir`, Makefile e notebook usam os padrões das classes | `estrategias/restrito.py`, `estrategias/pair_aware/`, `treino/classificador.py`, `treino/laco.py`, `scripts/treinar.py`, `scripts/pretreino.py`, `Makefile`, notebook, protocolo, testes | teto 2 nas duas classes; protocolo (tabela, nota, JSON e estimativas), READMEs; a referência de equivalência do restrito fixa `epochs=10` explicitamente | protocolo × código, critérios de seleção, configuração | confirmar a redação "igual ao limite do Two-Stage" (o Two-Stage tem 3 e 10) |
| **D6** versão do código | **resolvido** | as execuções não registravam a versão do código | `execucao/diretorio.py`, `execucao/trilha.py`, `treino/classificador.py`, `estrategias/two_stage/{treino,composicao}.py`, `estrategias/baseline_calibrado.py`, `scripts/filtro_pistas.py`, `execucao/inventario.py` | `util/identidade_codigo.py`; identidade no `config.json`, nas sessões do treino e na trilha; recusa de retomada, TEST e derivadas de outra versão ou sem identidade; inventário e relatórios conferem e não misturam | 28 unidade + 2 integração; 10 mutações detectadas | nenhuma |
| **D9** alinhamento das entidades | **parcialmente resolvido** (auditado; decisão pendente) | números confirmados no TRAIN e no DEV; textos sem nenhuma quebra de linha; deriva acumulada (monótona) em boa parte dos documentos; os marcadores seguem a especificação (offsets como estão) | `entrada.py`, `tarefa.py`, `analises/alinhamento.py`, partições (só TRAIN e DEV), notas das etapas 1 e 5 | `alinhamento.impacto_nos_pares` e `perfil_da_deriva`; `analisar.py alinhamento` grava os dois; nenhuma correção de dados ou de código de entrada | 6 testes sintéticos e de confirmação | como tratar o desalinhamento antes dos experimentos |

## A. D2 — separação entre o legado e os experimentos atuais

- **Testes de referência preservados.** `testes/equivalencia/` e
  `testes/referencia/` não foram removidos nem enfraquecidos. A única mudança
  num teste histórico é `testes/equivalencia/test_estrategias_treinadas_legado.py`
  passar `epochs=10` explicitamente ao reproduzir a referência do restrito
  (que o legado gerou com 10 épocas), porque o padrão da classe passou a 2
  (D5); a verificação continua a mesma.
- **Identificados e separados.** Um `conftest.py` novo em
  `testes/equivalencia/` marca todos os testes da pasta com
  `referencia_legado` e escreve, no resumo do pytest, a contagem deles com o
  aviso de que demonstram **compatibilidade técnica com o legado, não
  desempenho científico dos modelos atuais**. Dá para rodá-los à parte
  (`-m referencia_legado`) ou excluí-los (`-m "not referencia_legado"`).
- **Restritos à regressão.** Caminhos verificados:
  - inventário: recusa `codigo/testes/referencia/` e os nomes do legado (já
    era assim);
  - **caminho indevido corrigido:** `filtro_pistas.py` e `regra_pura.py` no
    modo de reprodução gravam execuções com `config.json` a partir das
    referências; agora o inventário recusa toda configuração com arquivo de
    calibração externo (`calibracao`) e todo filtro cujos sidecars não apontem,
    pelo SHA-256, as predições atuais da base no mesmo inventário;
  - uma pasta montada à mão com sidecars copiados precisaria ainda de
    `treino.json` (agora exigido nas execuções treinadas), da identidade do
    código consistente (D6) e da trilha; a verificação é de consistência e
    proveniência, não de autenticidade criptográfica contra falsificação
    deliberada (limitação declarada);
  - `comparar.py`, `avaliar.py` e `calibrar.py` aceitam sidecars soltos por
    serem ferramentas da reprodução técnica; nenhum relatório os usa (as
    comparações e métricas dos relatórios são recalculadas a partir do
    inventário).
- **Inventário e relatórios não misturam** execuções de naturezas, sementes,
  proveniências de pesos ou versões do código diferentes.
- **Testes da implementação atual independentes quando apropriado:** os testes
  novos das etapas 7 e deste saneamento usam um BERT sintético e partições
  sintéticas ou subconjuntos do TRAIN e do DEV; os testes de integração das
  etapas 5 a 6.1 continuam usando o modelo minúsculo das referências como
  insumo técnico (não como resultado de comparação), como aprovado nelas.

## B. D5 — limite de épocas

- `ConfigRestrito.epochs = 2` e `ConfigPairAware.epochs = 2`. Não há outro
  ponto com 10: `treinar.py` usa o padrão da classe (o `--epochs` continua
  existindo, mas `pretreino.py conferir --execucoes` recusa uma execução com
  configuração diferente da do protocolo); o Makefile e o notebook não passam
  `--epochs`.
- **Seleção inalterada:** maior macro-F1 do DEV completo com as predições
  remapeadas; empate → primeira época; sem parada antecipada (as 2 épocas
  rodam); o TEST não participa.
- **Two-Stage inalterado:** `ConfigEstagio1` 3 épocas, `ConfigEstagio2` 10
  (`test_two_stage.py` continua conferindo `(3, 10)`).
- **Protocolo:** tabela da seção 3, nota da decisão (D5, fase atual, para
  padronizar o orçamento com o Two-Stage, sem alegar otimalidade), bloco JSON
  (`epocas: 2`), estimativa do passo 4 e o registro da alteração no topo.
- **Divergência registrada:** a decisão diz "igual ao limite do Two-Stage",
  mas os tetos registrados do Two-Stage são 3 (estágio 1) e 10 (estágio 2); 2
  épocas é o do estágio 2 na rodada de validação técnica. Apliquei 2, como
  decidido, e registrei a divergência no protocolo para a sua confirmação.
  **Resolvida em 12/10/2026 (etapa 8.4):** o autor confirmou 3 e 10 para o Two-Stage; o texto do
  protocolo (seção 3) foi corrigido para bater com o código (critério A18).

## C. Integridade dos dados

As três partições e o `MANIFEST.json` estão idênticos (SHA-256) aos de antes
da tarefa, e `preparar_dados.py conferir` confere as três. Nenhum dado foi
corrigido, gerado ou regravado; as referências do legado estão idênticas.

## D. Proveniência do código (D6)

- `reclin/util/identidade_codigo.py`, algoritmo `sha256-manifesto-v1`: SHA-256
  de cada arquivo incluído (com `\r\n` → `\n`), manifesto `{caminho: sha256}`
  em ordem, identidade = SHA-256 do JSON do manifesto. **Incluídos:**
  `reclin/**/*.py`, `scripts/*.py`, `pyproject.toml`. **Excluídos:** análises,
  relatórios, inventário, formatação e os scripts que só leem execuções; nunca
  entram testes, documentação, notebooks, Makefile, dados, resultados,
  relatórios gerados, checkpoints, pesos e caches.
- **Registro:** `config.json` → `codigo` (algoritmo, identidade, número de
  arquivos e manifesto); `treino.json` → cada sessão; cada linha da trilha do
  TEST.
- **Verificação:** retomada do treino, avaliação do TEST (classificador,
  Two-Stage, baseline calibrado, filtro) e execuções derivadas (composição,
  calibração, filtro) recusam execuções de outra versão do código ou sem a
  identidade. O inventário confere o registro (identidade recalculada do
  manifesto), as sessões, a trilha e as dependências; só aceita como científica
  a versão de referência (padrão: a atual; `gerar_tcc.py --codigo-referencia`
  para outra) e nunca mistura versões. A tabela de proveniência ganhou a coluna
  "Código".
- **Execuções antigas, sem identidade:** fora da fase, não recebem artefatos
  novos e não são completadas depois. Hoje não há nenhuma execução científica
  na pasta.
- Documentado em `docs/reproducao.md` e no protocolo (seção 2, item 5; bloco
  JSON `identidade_codigo`).

## D3 e D4 — filtro léxico

**Verificado**

- O léxico é induzido **só do TRAIN** (formas de e1 dos pares candidatos
  `negation_of`, `max_gap` 25, normalização NFKD/casefold) com `min_freq`; o
  registro `CONGELADO` (`min_freq=3`, 11 formas, `lexico_sha1`
  `70c93fa807de`) confere o resultado na carga.
- **Origem de `min_freq=3`:** a calibração do legado no DEV
  (`calibrate_cue_filter.py`): varredura de `min_freq` ∈ {1, 2, 3, 5, 10} pelo
  F1 médio de `negation_of` do filtro aplicado às predições do DEV dos quatro
  baselines do legado, empate pelo menor `min_freq`. O TEST não entrou. O
  arquivo `CALIBRACAO_filtro.json` (em `testes/referencia/resultados_legado/`)
  tem `min_freq`, `lexicon_size` e os parâmetros da regra pura e da porta de
  gap; o filtro usa só `min_freq`.
- O protocolo da fase (seções 1 e 3, decisão C4) fixa o léxico congelado para
  o filtro, o restrito e a Pair-Aware, sem recalibração.
- Antes desta tarefa: o filtro só rodava com um arquivo de calibração (o do
  legado); DEV e TEST eram gravados juntos, e um segundo comando não conseguia
  acrescentar o TEST; a avaliação do TEST do filtro (`avaliar.py`) não gravava
  trilha, e o inventário ignorava o TEST do filtro.

**Corrigido tecnicamente** (autorizado pelo protocolo: léxico congelado, passo
3 no DEV e passo 7 no TEST, uma vez, com trilha)

- `filtro_pistas.py --base-execucao PASTA --particao dev|test` (modo da fase):
  léxico congelado do TRAIN congelado, nenhum arquivo de calibração; base só
  do código novo (baseline, mesma identidade do código), recusando sidecars
  soltos, o formato do legado e `testes/referencia/`; DEV cria a execução e as
  métricas; TEST só numa execução criada no DEV com a mesma configuração,
  depois de o TEST da base ter sido avaliado pelo comando próprio, nunca em
  base de validação, uma vez (`--reavaliar` para de novo, registrado);
  predições, métricas e linha da trilha com a identidade do código; cada
  sidecar com o `base_sha256`. O modo de reprodução continua idêntico (os
  sidecars do legado saem byte a byte).
- O inventário reconhece o TEST do filtro pela trilha e confere o vínculo com
  a base.
- `make filtro-dev` / `make test-filtro` e os passos `filtro_dev` / `test_filtro`
  do notebook, protegidos pela autorização explícita.

**Exige decisão metodológica (D4)**

- O `min_freq=3` é um parâmetro fixado pela C4, mas foi escolhido com
  predições de modelos do legado. Ele define o léxico do filtro **e** o espaço
  restrito do restrito e da Pair-Aware.
- **Opção A — manter a C4:** `min_freq=3` declarado no TCC como parâmetro fixado
  a priori, de origem histórica, sem uso de resultados do legado na fase. Nada
  muda no código.
- **Opção B — recalibrar no DEV do baseline novo** (procedimento proposto, a
  pré-registrar antes de executar): induzir o léxico do TRAIN para `min_freq` ∈
  {1, 2, 3, 5, 10}; para cada um, aplicar o filtro às predições do DEV de
  `baseline_biobertpt_seed42`; critério: maior F1 de `negation_of` pelas
  contagens (a mesma regra da busca de τ e t) ou, para manter o critério
  histórico, 2·P·R/(P+R); empate → menor `min_freq`; só TRAIN e DEV; o
  escolhido vira o novo `CONGELADO` (com a nova origem) antes do passo 4, e o
  restrito e a Pair-Aware passam a usar esse léxico. Consequências: muda o
  espaço restrito se o valor mudar; exige um comando novo e restrito
  (`calibrar.py` também calibra a regra pura e a porta de gap, fora da fase);
  o DEV passa a ser usado também para escolher o léxico.
- **Recomendação:** pela diretriz de não usar resultados históricos para
  calibrar parâmetros, a opção B é a mais coerente; a opção A é defensável se
  o parâmetro for tratado como herdado e declarado. A decisão é sua.

**Antes da execução científica do filtro:** decidir D4; com a opção B,
aprovar o procedimento, implementá-lo e atualizar `CONGELADO` e o protocolo;
com a opção A, nada além da confirmação.

## D9 — alinhamento das entidades

| Item investigado | Achado |
| --- | --- |
| 1. Definição dos offsets | `start`/`end` de cada entidade vêm do `dataset.jsonl` gerado pela conversão do XML do SemClinBr no legado (`parse_semclinbr.py`), copiado nas partições congeladas. O XML e esse script não estão disponíveis aqui (licença do corpus; repositório original sem rede nesta tarefa) |
| 2. Associação texto × offsets | Cada entidade tem `text` (o texto anotado) e os offsets; o código novo não os relaciona — usa só os offsets |
| 3. Janela de contexto | `entrada.janela_marcada`: `ctx_chars` (128) antes e depois do par, a partir dos offsets |
| 4. Inserção dos marcadores | `[E1] … [/E1]` e `[E2] … [/E2]` em volta de `texto[start:end]`, fechamentos antes de aberturas na mesma posição; idêntico ao legado (equivalência das janelas da etapa 5) |
| 5. Origem do desalinhamento | **Não é da implementação atual** (ela reproduz o legado byte a byte e segue a especificação). Nos dados: nenhum texto tem quebra de linha; a deriva é **acumulada** ao longo do documento em boa parte dos casos (por exemplo, em um documento, −1, −2, …, −6, crescendo a cada fim de frase), o padrão de caracteres removidos do texto depois de calculados os offsets, compatível com a normalização de quebras de linha na conversão. Isso é inferência: não dá para confirmar sem o XML e o script de conversão |
| 6. Números confirmados (TRAIN e DEV; TEST não lido) | TRAIN: 21.479 exatas, 469 equivalentes, 13.899 deslocadas, 608 ausentes (58,9% exatas); DEV: 2.970 / 33 / 1.435 / 75 (65,8%). Pares `negation_of` com alguma entidade desalinhada: 672 de 1.255 (TRAIN) e 58 de 150 (DEV). Documentos: TRAIN 453 sem deriva, 223 com deriva monótona, 124 não monótona; DEV 64 / 21 / 15 |
| 7. Marcadores × especificação | Conformes: envolvem exatamente `texto[start:end]` (teste sintético com 200 casos), usam os offsets como estão e não corrigem nada |

**Classificação**

- *Dados de origem:* não é possível afirmar sem o XML.
- *Transformação ou anotação:* hipótese mais provável — a conversão para o
  JSONL alterou o texto (quebras de linha) sem reajustar os offsets.
- *Implementação:* nenhum defeito no código atual dentro do comportamento
  especificado; nenhuma correção feita.
- *Questão metodológica em aberto:* em parte dos pares, os marcadores envolvem
  um trecho deslocado do texto anotado, e o mesmo vale para o léxico (que usa
  o `text` da entidade) × a janela (que usa os offsets). Isso afeta todas as
  estratégias da mesma forma e não autoriza nenhuma conclusão sobre
  desempenho.

**Opções (decisão sua):** (1) manter como está e declarar a limitação no TCC
(comparabilidade com o legado preservada); (2) realinhar os offsets em tempo
de execução (procurar o `text` da entidade perto dos offsets), mudando a
entrada de todas as estratégias e exigindo nova validação; (3) reconverter o
corpus a partir do XML com offsets corrigidos (novas partições e novo
MANIFEST, fora do que pode ser feito agora). Como os experimentos da etapa 8
seriam refeitos em (2) ou (3), a decisão precisa vir antes deles.

## Arquivos

**Criados (6):**

| Arquivo | Finalidade |
| --- | --- |
| `codigo/reclin/util/identidade_codigo.py` | identidade do código (D6) |
| `codigo/testes/unidade/test_identidade_codigo.py` | D6, 28 testes |
| `codigo/testes/integracao/test_proveniencia_codigo.py` | D6 de ponta a ponta (retomada, TEST, derivadas) |
| `codigo/testes/unidade/test_marcadores_offsets.py` | D9, marcadores e números da auditoria |
| `codigo/testes/equivalencia/conftest.py` | D2, marcador `referencia_legado` e resumo |
| `codigo/docs/entregas/etapa-7-pendencias.md` | esta nota |

**Modificados:**

| Arquivo | Mudança |
| --- | --- |
| `codigo/reclin/execucao/diretorio.py` | grava `codigo` no `config.json`; `conferir_codigo` |
| `codigo/reclin/execucao/trilha.py` | identidade do código em cada linha |
| `codigo/reclin/execucao/inventario.py` | identidade do código; calibração externa; vínculo do filtro; `treino.json` obrigatório nas treinadas; modelo das referências fora da fase |
| `codigo/reclin/treino/classificador.py` | identidade nas sessões; recusa de retomada e TEST de outra versão |
| `codigo/reclin/estrategias/two_stage/treino.py` | idem para os estágios |
| `codigo/reclin/estrategias/two_stage/composicao.py` | recusa de estágios e TEST de outra versão |
| `codigo/reclin/estrategias/baseline_calibrado.py` | recusa de base e TEST de outra versão |
| `codigo/reclin/estrategias/filtro_pistas.py` | `base_sha256` opcional no sidecar |
| `codigo/reclin/estrategias/restrito.py`, `codigo/reclin/estrategias/pair_aware/__init__.py` | teto de 2 épocas (D5) |
| `codigo/reclin/analises/alinhamento.py` | `impacto_nos_pares`, `perfil_da_deriva` |
| `codigo/reclin/relatorios/tabelas.py` | coluna "Código" na proveniência |
| `codigo/scripts/filtro_pistas.py` | modo da fase |
| `codigo/scripts/gerar_tcc.py` | `--codigo-referencia` |
| `codigo/scripts/analisar.py` | `alinhamento` grava o impacto nos pares e o perfil da deriva |
| `codigo/docs/protocolo/fase-biobertpt-seed42.md` | D5 (tabela, nota, JSON, estimativa), D6 (seção 2, JSON), comandos do filtro nos passos 3 e 7 |
| `codigo/Makefile`, `codigo/notebooks/reclin.ipynb` | passos do filtro na fase |
| `codigo/pyproject.toml` | marcador `referencia_legado` |
| `codigo/README.md`, `codigo/docs/reproducao.md`, `codigo/docs/estrategias/README.md`, `codigo/docs/entregas/README.md`, `codigo/docs/entregas/etapa-7.md` | documentação (a nota da etapa 7 ganhou só um aviso apontando para esta) |
| testes: `unidade/conftest.py`, `unidade/test_inventario.py`, `unidade/test_estrategias_treinadas.py`, `unidade/test_entradas_e_legado.py`, `integracao/test_criterios_restrito_pair_aware.py`, `integracao/test_relatorios_ponta_a_ponta.py`, `equivalencia/test_estrategias_treinadas_legado.py` | ajustes às decisões D3, D5 e D6 (detalhes acima); nenhum teste removido ou enfraquecido |

## Validações

Antes de rodar, conferi o que a suíte executa: treinos só em CPU com modelos
minúsculos ou sintéticos aleatórios; nenhum download (o teste do pré-treino
usa um baixador falso); nenhum treino científico. Os testes novos desta tarefa
não leem o TEST real (partições sintéticas, subconjuntos do TRAIN e do DEV ou
um TEST sintético feito de linhas do DEV). Alguns testes já aprovados das
etapas 5 e 6 e da equivalência calculam métricas do modelo minúsculo aleatório
sobre o TEST real ou um subconjunto dele: verificação técnica de formato e de
reprodução, sem relação com desempenho científico.

| Verificação | Resultado |
| --- | --- |
| Suíte completa numa cópia isolada (Python 3.13, CPU, 1 thread) | **716 passaram, 2 pulados** em 25 min (antes desta tarefa: 677 e 2). Por pasta: `unidade/` 406 + 1 pulado (figuras: sem matplotlib), `integracao/` 85, `equivalencia/` 225 + 1 pulado (treino completo do modelo minúsculo, opcional). O resumo do pytest separa os 225 testes de referência do legado como compatibilidade técnica. Depois dessa rodada mudaram só estas notas de entrega |
| Testes novos e ajustados | `test_identidade_codigo.py` (28), `test_proveniencia_codigo.py` (2), `test_marcadores_offsets.py` (5), `test_inventario.py` (+2), `test_relatorios_ponta_a_ponta.py` (+1, filtro), `test_pacote.py` (+1, gerado para o módulo novo) |
| Mutações (cada proteção nova desligada de propósito) | 13 de 13 detectadas: normalização de fim de linha, exclusão dos relatórios da identidade, conferência de versão e de ausência da identidade, inventário com outra versão, mistura de versões na seleção, trilha de outra versão, filtro sem vínculo, calibração externa, `treino.json` ausente, retomada, TEST e base do calibrado sem conferir o código |
| Teto de épocas | `ConfigRestrito().epochs == ConfigPairAware().epochs == 2`; protocolo × código conferido por `test_protocolo_fase.py`; Two-Stage `(3, 10)` conferido por `test_two_stage.py`; seleção pelo DEV remapeado conferida por `test_criterios_restrito_pair_aware.py` |
| Equivalência do filtro e da regra (modo de reprodução) | sidecars do legado idênticos byte a byte (`test_estrategias_legado.py`) |
| Partições, MANIFEST e referências | idênticos byte a byte aos de antes; `preparar_dados.py conferir`: 3 partições conferem |

Limitações: Windows e GPU não executados; o notebook foi executado só nas
células de parâmetros, preparação e nas guardas da fase científica; figuras
não verificadas (sem matplotlib); a hipótese sobre a origem do desalinhamento
não pôde ser confirmada (sem o XML e sem acesso ao repositório original).

## E. Decisões que precisam de aprovação antes da etapa 8

1. **D4** — manter `min_freq=3` (C4) ou aprovar o procedimento de
   recalibração no DEV do baseline novo.
2. **D9** — como tratar o desalinhamento de offsets (manter e declarar,
   realinhar em tempo de execução ou reconverter o corpus).
3. **D5** — confirmar a redação: o teto de 2 épocas do restrito e da
   Pair-Aware foi aplicado; o Two-Stage (3 e 10) não mudou.

## F. Prontidão para a etapa 8

**Não apto, por enquanto.** A infraestrutura está tecnicamente pronta:
protocolo íntegro e coerente com o código, proveniência dos pesos e do código,
isolamento do legado, filtro executável sem artefatos do legado, seleção só
pelo DEV. Mas duas decisões metodológicas mudam a entrada dos experimentos e
precisam vir antes deles: D4 (o léxico do filtro, do restrito e da Pair-Aware)
e D9 (o que os marcadores envolvem em parte dos pares). Se a decisão for
manter as duas como estão, declaradas como limitações, a recomendação passa a
**apto com ressalvas**.
