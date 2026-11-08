"""Análises descritivas: do corpus (partições congeladas) e das predições de
execuções verificadas pelo inventário (`execucao.inventario`).

Nível do núcleo: não importa `negacao` nem `estrategias`. O que depende do
léxico de pistas (por exemplo, se o e1 de um falso positivo é pista) recebe a
função de fora — quem chama (os scripts) passa `negacao.lexico.e_pista`.

* `alinhamento` — offsets das entidades contra o texto do documento;
* `corpus` — estatísticas das partições (documentos, entidades, relações,
  candidatos, distâncias);
* `max_gap` — sensibilidade do número de candidatos e da cobertura das
  relações anotadas ao limite de distância entre as entidades;
* `marcadores` — os marcadores de entidade nas janelas e depois da
  tokenização (truncamento em `max_length`);
* `falsos_positivos` — auditoria dos falsos positivos de `negation_of` de uma
  execução.

Diagnósticos da Etapa 2 (linha de comando em `codigo/diagnosticos/`):

* `controles_anotacao` — controles **informados pelas anotações** (tipo
  semântico `Negation`; tipo + posição). Não são configurações do RECLin-PT e
  recusam o TEST;
* `quase_duplicatas` — detector de documentos quase duplicados (só registra);
* `sensibilidade_semente` — a divisão 800/100/100 com outras sementes (só
  contagens; as partições congeladas não mudam);
* `exclusao_documentos` — métricas de um sidecar já gravado sem alguns
  documentos (preparada para o TEST; nele, só com autorização explícita).

Nenhuma análise treina, escolhe parâmetros ou lê resultados do legado. As que
poderiam orientar uma escolha (`max_gap`) usam TRAIN e DEV por padrão; o TEST
só entra quando pedido explicitamente, como descrição.
"""
