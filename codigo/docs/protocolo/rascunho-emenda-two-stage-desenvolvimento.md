# RASCUNHO — Emenda E1 ao protocolo da fase BioBERTpt, seed 42: desenvolvimento do Two-Stage

> **Estado: RASCUNHO. NÃO ESTÁ EM VIGOR. NÃO É PRÉ-REGISTRO. NÃO AUTORIZA
> NENHUMA EXECUÇÃO.**
> Este documento não altera `fase-biobertpt-seed42.md` nem nenhum critério de
> aceitação. Nada nele é executado, aplicado ou conferido pelo código. Ele só
> passa a valer depois de (1) revisão pelo autor, (2) preenchimento dos pontos
> marcados **[A DEFINIR]**, (3) aprovação explícita registrada e (4) gravação do
> SHA-256 do texto aprovado. Até lá, nenhuma configuração do Two-Stage além da
> do protocolo vigente pode ser treinada como parte da fase.
>
> **Decisões do autor incorporadas na etapa 8.2 (12/10/2026), como proposta
> documental:** orçamento máximo de 10 candidatas com registro de todas as
> tentativas (inclusive falhas e reinícios); `cand-00` fora do orçamento por
> ser exclusivamente referência de controle; empate resolvido pela primeira
> candidata registrada, aplicado uniformemente; X1 aprovada; X2 e X3 não
> serão implementadas. **Nada disso autoriza experimento, treinamento ou
> avaliação TEST, e o protocolo vigente não foi alterado nem ativado.**
>
> Convenções: **[PROPOSTO]** = sugestão minha para revisão (não vale sem
> aprovação); **[A DEFINIR]** = depende de decisão do autor, sem sugestão;
> **[NOVO — APROVAR]** = regra, critério, faixa ou desempate novo que só entra com
> aprovação expressa (nenhum está pendente além do uso de intervalos/testes,
> seção 5.3).

Rascunhado em 12/10/2026 (etapa 8.1) e revisado em 12/10/2026 (etapa 8.2), a
partir das decisões do autor registradas em `docs/entregas/etapa-8.1.md` e das
instruções da etapa 8.2. Documentos de apoio:
`docs/arquitetura/identidade-por-componente.md` e
`docs/arquitetura/compatibilidade-two-stage.md`.

## 1. Objetivo e escopo

Permitir que o Two-Stage evolua (novas configurações de estágio 1, estágio 2 e
composição) sem enfraquecer o protocolo vigente e sem usar o TEST para escolher
configurações.

- Escopo: Two-Stage (estágio 1, estágio 2, composição). BioBERTpt
  (`pucpr/biobertpt-all`), seed 42, dados `v2-texto-fonte`, mesma revisão dos
  pesos do protocolo.
- Fora de escopo: as demais estratégias, outros encoders ou sementes, mudança de
  dados, de MANIFEST, de `max_gap`, do léxico congelado, das comparações do
  protocolo ou da regra do modelo final (C13).
- Esta emenda **não** torna o Two-Stage a arquitetura oficial (seção 9).
- O teto de 2 épocas (D5) do restrito e da Pair-Aware **não** é afetado por
  esta emenda nem é estendido a ela: os tetos de épocas do Two-Stage seguem o
  protocolo vigente (3 e 10) e, sob a E1, só variam como parâmetro de
  configuração candidata (seção 3), nunca por herança do D5.

## 2. Três regimes, definidos sem ambiguidade

| | **Desenvolvimento** | **Promoção autorizada** | **Avaliação científica** |
| --- | --- | --- | --- |
| O que é | configuração exploratória do Two-Stage sob a E1 | congelamento de **uma** configuração de desenvolvimento, com registro explícito de autorização | execução conforme o protocolo vigente e seus critérios de comparação |
| Seleção | só no DEV, critério da seção 4 | nenhuma: congela o que já foi escolhido | a do protocolo |
| TEST | **bloqueado** (recusado no código pela marca, a implementar na 8.3) | **não autorizado pela promoção**: exige autorização própria e as regras do protocolo | permitido nas regras do protocolo (uma vez por sistema após o congelamento do DEV) |
| Natureza gravada | `desenvolvimento` (permanente) | a execução **continua** `desenvolvimento`; a promoção é um registro à parte (`promocao.json`) | `cientifica` |
| Entra em tabelas oficiais | não | não, até uma revisão do protocolo dizer outra coisa (seção 9) | sim |
| Pode ser apagado/sobrescrito | nunca | nunca | nunca |

Regras comuns:

1. **Usar BioBERTpt e seed 42 não torna uma configuração científica.** Uma
   execução só é `cientifica` se tiver **exatamente** a configuração do
   protocolo vigente. Uma configuração de desenvolvimento **nunca** se converte
   em `cientifica`; ela só pode ser **promovida**, que é outro estado (seção 7).
2. A promoção **não** altera resultados anteriores, **não** sobrescreve
   referências, **não** apaga tentativas e **não** converte retroativamente
   execuções de desenvolvimento em resultados científicos.
3. Execuções `desenvolvimento` não têm o TEST avaliado, não entram em tabelas
   oficiais e nunca sobrescrevem execuções oficiais (que mantêm os nomes do
   protocolo).
4. A marca é gravada na criação da execução e verificada pelo código (não só
   pelo orquestrador): `avaliar-test`, `aplicar-test`, `pretreino.py conferir`
   e `comparar.py` a recusam, como já fazem com a validação técnica.
   *(Implementação prevista na etapa 8.3; sem ela, esta emenda não pode ser
   ativada.)*

## 3. Espaço de configuração permitido **[A DEFINIR PELO AUTOR]**

Nenhum valor, faixa ou passo é proposto aqui. Qualquer faixa de valores é uma
regra nova e entra só depois de apresentada e aprovada neste rascunho.

Campos que **podem** compor o espaço (o autor escolhe quais incluir e quais
valores/faixas cada um tem). A divisão segue a matriz de
`compatibilidade-two-stage.md`:

| Componente | Campos candidatos | Observação |
| --- | --- | --- |
| Estágio 1 | `epochs`, `lr`, `weight_decay`, `warmup_ratio`, `max_grad_norm`, `class_weight` | exceção **X1** a `CAMPOS_BASE` (aprovada na 8.2) |
| Estágio 2 | os mesmos, **independentes** dos do estágio 1 | idem |
| Composição | nenhum: a grade de τ (0,05–0,95, passo 0,05) e a busca são as do protocolo | alterar a grade seria faixa nova |

**Compartilhados e fora do espaço** (idênticos nos dois estágios e não variam
sob a E1): encoder, semente, dados e partições, `max_gap`, **`ctx_chars`**,
**`max_length`**, **`batch_size`**, representação do exemplo, revisão e manifesto dos pesos,
tokenizer, regra de roteamento (P(relacao) arredondada a 4 casas ≥ τ).
`ctx_chars`, `max_length` e `batch_size` saem do espaço candidato (na 8.1
estavam como candidatos): a composição usa a configuração do estágio 1 para os
exemplos e para a inferência dos dois estágios. X2 e X3 **não serão
implementadas**.

## 4. Critério de seleção (só DEV)

**[Critério já especificado; empate entre candidatas decidido na 8.2]**

- Seleção interna de cada estágio: a do protocolo (AP de `relacao` no estágio 1;
  F1 de `negation_of` nos pares com relação anotada no estágio 2), com desempate
  pela primeira época.
- Comparação entre configurações candidatas do pipeline completo: **F1 de
  `negation_of` da saída final de 3 rótulos no DEV**, com τ escolhido no DEV
  pela busca pré-registrada (`reclin/avaliacao/limiar.py`; grade 0,05–0,95;
  desempates do protocolo **dentro da busca de τ**: macro-F1, menor |τ−0,5|,
  maior τ). Métricas complementares (macro-F1, MCC, precisão, recall)
  descritivas.
- O TEST não participa de nenhuma escolha, de nenhuma regra de parada nem de
  nenhuma decisão de continuar, interromper ou aumentar a busca.
- **Empate exato entre candidatas (decisão do autor, 8.2):** vence a
  candidata **registrada primeiro** no registro de candidatas, aplicado
  uniformemente a todas as candidatas. É regra determinística e independente
  dos resultados. (O protocolo define desempate apenas para τ; esta regra vale
  só entre candidatas e só quando a E1 estiver ativa.)
- Qualquer outro critério, faixa ou regra exige alterar esta seção antes da
  ativação.

## 5. Orçamento e multiplicidade no DEV

### 5.1 Proposta de orçamento inicial **[PROPOSTO — para revisão; NÃO é autorização]**

- **Até 10 configurações candidatas de composição**, contadas em conjunto
  durante toda a vigência da E1. O limite é uma **proposta**: não autoriza
  iniciar nenhum experimento, e a emenda continua rascunho.
- **O que é uma configuração candidata de composição:** o conjunto (configuração
  efetiva do estágio 1, configuração efetiva do estágio 2, composição com a
  grade de τ do protocolo). Uma nova combinação de estágios já treinados é uma
  nova candidata.
- **O que consome o orçamento:** cada candidata **registrada** (antes do
  treino), **qualquer que seja o resultado**: treino concluído, interrompido
  sem retomada, falha técnica, resultado inválido, composição sem resultado
  válido, descartada pelo autor. Um **reinício do zero** da mesma configuração
  conta como nova tentativa e consome mais uma unidade (decisão do autor,
  8.2). A
  **retomada** a partir de um checkpoint compatível da mesma execução não é
  nova tentativa.
- **Referência:** a composição do protocolo vigente (estágio 1: 3 épocas;
  estágio 2: 10 épocas) é registrada como `cand-00`, **fora do orçamento**
  porque é exclusivamente referência de controle: não participa da busca nem
  da escolha, e aparece em toda comparação. Se `cand-00` participasse da
  busca, passaria a consumir orçamento.
- **Sem aumento nem redistribuição automáticos.** Não há transferência de
  unidades não usadas, não há "tentativa extra por falha" e nenhuma regra
  aumenta N sozinha. Qualquer mudança é registrada como **proposta de
  alteração** (valor proposto, motivo, data, quem propôs, estado) e só vale
  depois de aprovação expressa do autor, também registrada. A proposta de
  alteração não pode usar o TEST como motivo.

### 5.2 Registro de candidatas

Arquivo de registro apenas de acréscimo (cada mudança de estado é uma nova
linha; nenhuma linha é editada ou apagada). Local e formato exatos: etapa 8.4.
Campos mínimos de cada candidata:

| Campo | Conteúdo |
| --- | --- |
| `id` | `cand-NN` sequencial, nunca reutilizado |
| `parametros` | configuração efetiva completa de cada estágio e da composição; chaves de reaproveitamento (`compatibilidade-two-stage.md`, seção 3); revisão e manifesto dos pesos; versão dos dados; identidade de código por componente |
| `origem` | de onde veio: protocolo, variação de uma candidata anterior (qual campo, de quanto para quanto), proposta do autor; data; justificativa |
| `status` | `planejada` → `em_treino` → `interrompida` / `falhou` / `estagios_prontos` → `composta` → `sem_resultado_valido` / `descartada` / `escolhida` / `promovida` |
| `resultados_dev` | por estágio: métrica de seleção, melhor época, histórico; do pipeline: τ escolhido, F1 de `negation_of`, macro-F1, MCC, precisão, recall; **ausência explícita** (`null` + motivo) quando não houver resultado válido |
| `execucoes` | `run_id`s (com tentativa), nomes de execução, hashes de `melhor_modelo/` e de configuração |
| `tentativa` e `consumo` | número da tentativa; unidades de orçamento consumidas |
| `ambiente` | hardware (GPU), versões, data e hora |

Estágios reaproveitados contam **uma vez** para o treino; cada composição nova
conta como uma candidata.

### 5.3 Risco de otimismo (registro expresso)

O DEV já é usado para escolher a época de cada estágio e τ. Escolher, além
disso, a melhor entre até 10 candidatas pelo **mesmo** DEV expõe a seleção ao
viés do máximo: o F1 de `negation_of` da candidata escolhida é o maior de várias
estimativas correlacionadas e ruidosas e tende a **superestimar** o desempenho
fora do DEV. O estágio 2 é selecionado em 913 pares do DEV com relação anotada
(docstring do Two-Stage) e `negation_of` é a classe minoritária (1.255 dos
8.351 pares com relação no TRAIN); poucos positivos aumentam a variância das
diferenças entre candidatas. Em
consequência:

- o número de candidatas e **todos** os seus resultados DEV acompanham qualquer
  número derivado do DEV;
- a candidata escolhida é apresentada como **diagnóstico de desenvolvimento**,
  não como desempenho esperado;
- diferenças pequenas entre candidatas não são interpretadas como superioridade
  sem outra evidência; **qualquer uso de intervalo, teste ou correção de
  multiplicidade é critério novo e exige aprovação neste rascunho**
  **[NOVO — APROVAR, se desejado]**;
- a seleção não se estende a outra semente, encoder ou partição sem nova emenda.

## 6. Reaproveitamento de estágios

Um estágio concluído pode ser reutilizado numa nova composição sem novo treino
se, e somente se, a **chave de reaproveitamento** coincidir e as verificações no
momento do uso passarem. A definição completa está em
`docs/arquitetura/compatibilidade-two-stage.md` (seção 3). Resumo:

- chave = estágio, configuração efetiva (sem caminhos locais), dados e
  partições, pesos de origem (revisão + manifesto), tokenizer, identidade de
  código **do componente** (`identidade-por-componente.md`) e natureza;
- **a existência de um checkpoint, por si só, não comprova compatibilidade**;
  hash do `melhor_modelo/`, treino concluído, partições, proveniência e natureza
  são conferidos no uso;
- o reaproveitamento cria uma dependência registrada, nunca uma cópia; um
  estágio reaproveitado não pode ser alterado; a composição é uma execução
  própria, com identificador próprio.

## 7. Congelamento, promoção e efeito sobre as comparações

### 7.1 Promoção

Uma configuração `desenvolvimento` só pode ser considerada para avaliação no
TEST depois de um registro de **promoção** (`promocao.json`). **A promoção, por
si só, não autoriza executar o TEST.** Uma avaliação no TEST continua exigindo
(a) **autorização explícita e separada** do autor para aquela configuração e
(b) todas as regras do protocolo vigente (congelamento do DEV, proveniência,
guardas e trilha do TEST). O `promocao.json` registra explicitamente
`"teste_autorizado": false`; a autorização do TEST, se vier, é um registro
distinto que referencia o hash do `promocao.json`.

Campos mínimos do `promocao.json`:

| Campo | Conteúdo |
| --- | --- |
| `emenda` | `E1` e o SHA-256 do texto **aprovado** da emenda aplicável |
| `autorizacao` | quem autorizou, data, referência/texto da autorização explícita, escopo ("promoção da candidata X; não autoriza TEST") |
| `configuracao_congelada` | `cand-NN`; chaves e configurações efetivas dos dois estágios e da composição; τ congelado e vínculos do `limiar.json`; hashes dos `melhor_modelo/`; pesos; dados; manifestos de identidade de código |
| `execucoes_relacionadas` | `run_id`s dos estágios e da composição; **todas** as candidatas e tentativas do orçamento (inclusive descartadas e sem resultado válido) |
| `orcamento` | N aprovado, N usado, hash do registro de candidatas no momento da promoção |
| `dev` | métricas DEV da candidata e das demais; declaração de risco de otimismo |
| `declaracoes` | nenhum resultado de TEST foi visto para decidir; nenhuma execução oficial foi alterada |
| `teste_autorizado` | `false` |
| `substitui` | `null` (nenhuma promoção substitui outra sem nova promoção) |

Depois da promoção a configuração é imutável. Qualquer mudança exige nova
promoção. O `promocao.json` é apenas de criação (hash gravado; não se edita).

### 7.2 Como as comparações distinguem os três regimes

Cada linha de qualquer tabela ou relatório carrega um **rótulo de regime**
derivado do registro, nunca digitado:

| Rótulo | Origem | Onde aparece |
| --- | --- | --- |
| `exploratoria` | natureza `desenvolvimento`, sem `promocao.json` válido | só na tabela **"Desenvolvimento (DEV)"**: todas as candidatas, N usado e aviso de otimismo; nunca em tabelas de resultado final |
| `promovida` | natureza `desenvolvimento` **com** `promocao.json` válido | tabela própria, **"Promovida sob E1 (fora do protocolo vigente)"**; resultados DEV e, se um dia autorizado e executado, TEST com o rótulo explícito de que não é o sistema do protocolo |
| `cientifica_oficial` | natureza `cientifica` (nomes do protocolo) | tabelas do protocolo (seção 7 dele) |

Regras:

- `comparar.py` e os relatórios **não misturam** rótulos numa mesma tabela
  oficial; uma comparação entre `promovida` e `cientifica_oficial` é permitida
  apenas lado a lado, com ambos os rótulos visíveis, e **nunca** sob o rótulo
  "oficial".
- Resultados anteriores permanecem exatamente como estão; a promoção só
  acrescenta registros.
- Não existe operação de "reclassificar como científica".

## 8. Relação com o protocolo vigente

- O protocolo vigente permanece intacto: o Two-Stage dele (estágio 1: 3 épocas;
  estágio 2: 10 épocas; τ pela busca pré-registrada) continua sendo o sistema
  da fase. As execuções oficiais mantêm os nomes do protocolo.
- Uma promoção sob esta emenda **não** altera, por si só, as comparações da
  seção 7 do protocolo, a métrica primária nem o rótulo "validação inicial,
  seed 42". Mudar qualquer um deles exige uma revisão própria do protocolo.
- A avaliação histórica do TEST no legado continua apenas registrada como dado
  histórico, sem uso de números.
- Esta emenda não toca no D5: o teto de 2 épocas do restrito e da Pair-Aware é
  assunto do protocolo, e a redação proposta para ele está no relatório da
  etapa 8.2 (não aplicada).

## 9. Pendências que permanecem fora desta emenda

1. Valores/intervalos da seção 3 e a lista final dos campos que entram no
   espaço (a definir pelo autor antes de qualquer ativação).
2. Se a promoção cria um sistema comparado novo ou substitui o Two-Stage do
   protocolo nas comparações, e a oficialização do Two-Stage como arquitetura
   do RECLin-PT — exige registro e revisão próprios do protocolo.
3. Texto de ativação e local do registro do SHA-256 aprovado.
4. Procedimento de autorização do TEST de uma configuração promovida (registro
   distinto referenciando o hash do `promocao.json`).

Nada nesta lista é ativado por este rascunho.

## 10. Registro verificável (rascunho; não conferido por teste algum)

```json
{
  "emenda": "E1",
  "estado": "rascunho",
  "ativa": false,
  "autoriza_execucao": false,
  "protocolo_base": "docs/protocolo/fase-biobertpt-seed42.md",
  "escopo": "two_stage",
  "natureza_nova": "desenvolvimento",
  "teste_permitido_em_desenvolvimento": false,
  "promocao_autoriza_teste": false,
  "criterio_de_comparacao": "f1_negation_of_dev_pipeline_completo",
  "orcamento_n_proposto": 10,
  "orcamento_n_aprovado": null,
  "orcamento_conta_todas_as_tentativas": true,
  "orcamento_aumento_automatico": false,
  "espaco_de_configuracao": null,
  "desempate_entre_candidatas": "primeira_registrada",
  "cand_00_no_orcamento": false,
  "reinicio_consome_unidade": true,
  "sha256_aprovado": null
}
```
