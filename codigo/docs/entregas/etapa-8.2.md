# Etapa 8.2 — consolidação das especificações (documental) — ENCERRADA

> **Encerrada em 12/10/2026** com as decisões finais do autor (seções 9 e 10).
> A E1 continua rascunho, `ativa: false`. A etapa 8.3 e o Prompt 2 não foram
> iniciados.

Data: 12/10/2026. **Sem código, testes, treino, download ou Git.** Nenhum dado,
MANIFEST, resultado de referência ou o protocolo vigente foi alterado. A emenda
E1 continua **rascunho, não em vigor, não autorizada para execução**.
O próximo trabalho (Prompt 2 — infraestrutura compartilhada do Colab) só começa por comando do autor.

## 1. Resumo

Foram escritas três especificações novas, revisado o rascunho da emenda E1 e
registradas as observações do legado sobre Hugging Face e Drive:

| Pedido | Onde está |
| --- | --- |
| 1. Identidade por núcleo / estratégia / estágio / composição; scripts, `pyproject.toml`, módulos de identidade; regra para docstrings; detecção localizada de invalidação | `docs/arquitetura/identidade-por-componente.md` |
| 2. Orçamento de até 10 candidatas (proposta), registro por candidata, seleção só no DEV, risco de otimismo | `docs/protocolo/rascunho-emenda-two-stage-desenvolvimento.md`, seções 4–5 |
| 3. Matriz de compatibilidade e chave de reaproveitamento | `docs/arquitetura/compatibilidade-two-stage.md` |
| 4. Desenvolvimento × promoção × avaliação científica; `promocao.json`; rótulos nas comparações | emenda, seções 2 e 7 |
| 5. Consulta ao legado | `docs/arquitetura/integracao-legado-hf-drive.md` |
| 6. D5 | seção 4 deste relatório (proposta; **nada foi alterado**) |

## 2. Principais conclusões técnicas

- **Identidade.** Os conjuntos fecham: estágio = 8 (núcleo comum) + 19 (núcleo
  de treino) + 5 próprios; composição = 8 + 19 + 3. A composição não inclui o
  código de treino dos estágios; a ligação com eles é de **dados** (hash do
  `melhor_modelo/`) e de **campos**.
- **Dois acoplamentos desnecessários encontrados** (por leitura): (a)
  `two_stage/__init__.py` mistura constantes dos dois estágios, então mudar o
  critério do estágio 2 invalida o estágio 1; (b) `two_stage/treino.py`
  importa `ErroExecucao` de `treino/classificador.py`, arrastando o treino do
  baseline para a identidade do estágio. Ambos têm correção proposta (3.4 da
  especificação), aplicadas como implementação mínima da decisão; se inviáveis vale a invalidação
  conservadora.
- **Hoje todo script entra no hash global**; decisão final: cada script entra só
  na identidade dos componentes que ele afeta.
- **`pyproject.toml`** sai do hash de código; versões instaladas e GPU passam
  a ser registradas e a bloquear a **retomada** (aviso no reaproveitamento).
- **Docstrings:** não são presumidas irrelevantes. Fato: nenhum módulo de
  `reclin/` lê `__doc__`. Decisão final: hash integral do arquivo,
  inclusive docstrings; sem normalização de AST.
- **Compatibilidade:** achado relevante — a composição usa o `max_length`, o
  `batch_size` e a `seed` do **estágio 1** para inferir **os dois** estágios, e
  constrói os exemplos com `max_gap`/`ctx_chars` do estágio 1. Por isso o
  `max_length` não pode variar entre estágios sem mudar a composição.
- **Tokenizer** não é comparado entre estágios hoje (lacuna a fechar).
- `registro_estagio` grava `class_weight` como texto fixo "balanced",
  mesmo se `class_weight="none"` (correção de registro necessária para X1).

## 3. Matriz de compatibilidade (resumo; completa no documento)

| Campo | Regra | Evidência |
| --- | --- | --- |
| dados, partições, `max_gap`, `ctx_chars`, representação do exemplo | idêntico | construídos uma vez com a configuração do estágio 1 (`composicao._particao`) — lido |
| encoder, `seed`, pesos (revisão + manifesto), proveniência | idêntico | `CAMPOS_BASE`, `pretreino.mesma_proveniencia` — lido |
| tokenizer | idêntico (a verificar explicitamente) | hoje não comparado — lido/lacuna |
| `max_length` | idêntico (X3 não será implementada) | `Pipeline` usa o do estágio 1 para os dois — lido |
| `batch_size` | idêntico (X2 não será implementada) | idem — lido |
| `epochs` | pode variar (já varia: 3 e 10) | fora de `CAMPOS_BASE` — lido |
| `lr`, `weight_decay`, `warmup_ratio`, `max_grad_norm`, `class_weight` | pode variar (X1) | usados só no treino do estágio — lido |
| critério de seleção, dados de treino do estágio | fixo por estágio | `CAMPO_SELECAO`, `DADOS_TREINO` — lido |
| τ, grade e desempates | da composição | `avaliacao/limiar.py`; protocolo — lido |

## 4. D5 — decisão registrada (protocolo NÃO alterado)

O autor aprovou a redação abaixo, que limita o teto de 2 épocas a `restrito` e
`pair_aware`. Registro documental apenas: `fase-biobertpt-seed42.md` e os demais
documentos que citam o D5 **não foram modificados** nesta etapa.

> **Teto de épocas do restrito e da Pair-Aware (D5, 11/10/2026): 2 épocas.**
> Decisão do autor para a fase experimental atual. Aplica-se **somente** às
> estratégias `restrito` e `pair_aware`. As 2 épocas rodam sempre, sem parada
> antecipada; a época escolhida é a de maior macro-F1 do DEV remapeado ao
> conjunto completo (empate → a primeira). Este teto **não** se aplica ao
> baseline (3 épocas), ao Two-Stage (estágio 1: 3 épocas; estágio 2: 10
> épocas) nem a qualquer configuração de desenvolvimento, e não é estendido a
> eles por extensão ou analogia. Ele substitui o teto anterior de 10 épocas,
> que vinha da configuração da frente no legado. Não se afirma que o teto seja
> ótimo nem que melhore o desempenho. A decisão não autoriza treinamento.

A incorporação desse texto ao protocolo, se desejada, é uma alteração do
protocolo e fica fora desta etapa.

## 5. Arquivos

Criados (4): `docs/arquitetura/identidade-por-componente.md`,
`docs/arquitetura/compatibilidade-two-stage.md`,
`docs/arquitetura/integracao-legado-hf-drive.md`, `docs/entregas/etapa-8.2.md`.

Modificados (2): `docs/protocolo/rascunho-emenda-two-stage-desenvolvimento.md`
(reescrito; continua rascunho, `ativa: false`), `docs/entregas/README.md`
(linha da 8.2). Removidos: nenhum. Código, testes, dados, MANIFEST,
referências e protocolo vigente: **não alterados**.

## 6. Verificações

| Verificação | Resultado |
| --- | --- |
| Leitura de `composicao.py`, `two_stage/__init__.py`, `treino.py`, `config.py`, `modelos.py`, `pyproject.toml`, protocolo (seção 3) | executada (somente leitura) |
| AST dos `__init__.py` (docstring-only) e busca de `__doc__` | executadas (sem importar o projeto) |
| Bloco JSON do registro da emenda é JSON válido, `ativa: false` | executada |
| Nenhum arquivo fora de `docs/` modificado nas últimas 120 min | executada (por data de modificação; **não** é verificação de Git) |
| Consulta ao legado (README, notebooks 01/02, `hf_backup.py`) | executada por leitura remota **resumida por ferramenta**; detalhes numéricos precisam de conferência |
| Testes, treino, download, calibração, TEST, Git | **não executados** |
| Funções realmente usadas de `classificador.py`/`checkpoint.py`/`diretorio.py`, `regra_pura` | levantamento por leitura na 8.3 (seção 10) |

## 7. Critérios de aceitação desta etapa

| Critério | Estado |
| --- | --- |
| Identidade por núcleo/estratégia/estágio/composição, scripts, pyproject, módulos de identidade, regra de docstring documentada | atendido (documental) |
| Detecção de invalidação por estágio sem invalidar tudo | especificado; depende da separação por estágio e da remoção da dependência de `classificador.py` na 8.3 (seção 10) |
| E1 mantida como rascunho; orçamento ≤ 10 como proposta; registro por candidata; risco de otimismo | atendido |
| Nenhum critério, faixa ou desempate novo sem marca de aprovação | atendido; as regras novas (desempate, reinício, `cand-00`) foram aprovadas como proposta documental na seção 9 |
| Matriz de compatibilidade e chave de reaproveitamento | atendido |
| Três regimes e `promocao.json`; promoção não autoriza TEST | atendido |
| Consulta ao legado sem copiar código | atendido, com a limitação da seção 6 |
| D5 sem alteração do protocolo | atendido (proposta apenas) |

## 8. Limitações e riscos

- A leitura do legado foi resumida; "célula 11" é markdown (sem código) com
  contagem a partir de 0, e o código está na 12: **confirmar com você** qual
  célula tinha em mente.
- A classificação de arquivos em classes S/R é proposta, não verificada
  linha a linha.
- O refinamento por AST depende de a Colab e a máquina local terem o mesmo
  `major.minor` do Python; senão cai no hash de bytes (mais invalidação).
- O orçamento de 10 é uma proposta; a escolha entre 10 candidatas no DEV traz
  viés otimista, registrado na emenda.

## 9. Decisões finais do autor aplicadas

| Tema | Decisão aplicada |
| --- | --- |
| Notebooks legados | questão da célula 11 esclarecida; só o comportamento observado vale |
| E1 | rascunho, `ativa: false`; N máx. 10, todas as tentativas e reinícios contam; `cand-00` fora do orçamento por ser só controle; primeira candidata registrada vence empates; nada autoriza experimento/treino/TEST |
| Compatibilidade | X1 aprovada; X2 e X3 não serão implementadas; `max_length` e `batch_size` compartilhados; duas listas explícitas; reaproveitamento exige configuração efetiva, dados, pesos, tokenizer, identidade do componente e integridade do modelo |
| Identidade | por estágio e por composição, com identidade comum; scripts mapeados aos componentes; hash integral com docstrings, sem AST; `pyproject.toml` fora do hash, ambiente registrado e conferido antes de retomar; `__version__` no hash; mudança local não invalida independentes, invalida todos os efetivamente afetados; **O1/O2/O3 encerradas** |
| D5 | redação aprovada (seção 4); protocolo inalterado |

## 10. Escolhas aplicadas para detalhes não especificados (sem ampliar escopo)

1. Classes S/R de arquivo **não** adotadas; todo arquivo do conjunto invalida o
   componente.
2. Separação do código por estágio e remoção da dependência de
   `classificador.py` tratadas como implementação mínima da decisão de não
   invalidar independentes; se inviáveis, vale a invalidação conservadora,
   registrada na 8.3.
3. `regra_pura` e scripts de avaliação/dados levantados na 8.3 pela mesma regra.
4. Rótulos de regime (`exploratoria`, `promovida`, `cientifica_oficial`) e
   comparação lado a lado mantidos como no rascunho da E1.

## 11. Bloqueios para iniciar o próximo prompt

Nenhum bloqueio real identificado. A implementação deve respeitar as
restrições permanentes (sem Git, sem execução automática de testes, sem
treino/download/TEST).
