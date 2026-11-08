# PROPOSTA — Emenda E3 ao protocolo da fase BioBERTpt, seed 42: Pair-Aware linear como arquitetura principal e TEST reservado ao modelo final

> **Estado: PROPOSTA (revisão 2). NÃO ESTÁ EM VIGOR. NÃO AUTORIZA NENHUMA
> AVALIAÇÃO DO TEST.**
>
> **Esta proposta não altera:**
> - `fase-biobertpt-seed42.md` nem o seu bloco JSON (seção 9);
> - nenhuma configuração, partição, predição, checkpoint ou resultado existente.
>
> **Para entrar em vigor, ela precisa de:**
> 1. as decisões ainda pendentes (seção 7) tomadas e escritas no texto;
> 2. aprovação explícita do autor, registrada;
> 3. gravação do SHA-256 do texto aprovado na seção 8 do protocolo, como "E3", ao lado da E2.
>
> Até lá, vale o protocolo como está.
>
> **Convenções:**
> - **[DECIDIDO PELO AUTOR]**: decisão comunicada pelo autor em 02/11/2026, incorporada ao texto. Só vale com a aprovação da emenda.
> - **[PROPOSTO]**: redação sugerida para revisão.
> - **[PENDENTE]**: depende de decisão do autor. Alternativas e recomendação indicadas, nada assumido.

## Histórico desta proposta

| Revisão | Data | Conteúdo |
| --- | --- | --- |
| 1 | 02/11/2026 | Pair-Aware linear como arquitetura principal. Comparações principais no TEST: Pair-Aware linear × baseline e × baseline calibrado. Critério de escolha do encoder entre três encoders e lista de sistemas avaliados no TEST, ambos pendentes |
| 2 | 02/11/2026 | Incorpora as decisões do autor: arquitetura principal; BioBERTpt-all como encoder de referência; configuração de referência não definitiva; etapa de aprimoramento com novos experimentos; TEST **só para o modelo final**, sem comparação de experimentos. As comparações com o baseline e o calibrado passam a ser referências de desenvolvimento no DEV. O critério estatístico (E3-D3) continua pendente |

## 1. Por que uma emenda e em que momento

O protocolo (11/10/2026), rotulado "validação inicial, seed 42", prevê:
- duas comparações principais no TEST, ambas com o **Two-Stage**: contra o baseline padrão e contra o baseline calibrado;
- o TEST de todos os sistemas, uma vez cada (seção 6, passo 7);
- o encoder BioBERTpt-all (seção 1);
- a regra do modelo final adiada (pendência C13).

**O que mudou desde então:**
- A Pair-Aware linear (configuração da D10) passou a ser a arquitetura oficial do RECLin-PT, por decisão do autor, registrada a partir de 01/11/2026, depois de conhecidos os resultados do DEV (`docs/analises/cronologia-arquitetura-oficial.md`).
- Em 02/11/2026, o autor decidiu também:
  - manter o BioBERTpt-all como encoder de referência;
  - continuar a experimentação antes de definir o modelo final;
  - reservar o TEST ao modelo final.

**Natureza: pós-DEV e pré-TEST.**
- Toda escolha apoiada nos resultados do DEV é exploratória.
- Nenhuma configuração atual foi avaliada no TEST (`test_avaliado: false` em todas as execuções; E2).

## 2. O que não muda

| Item | Onde está fixado |
| --- | --- |
| Dados `v2-texto-fonte`, partições, MANIFEST `18264860…` e `max_gap = 25` | seção 1 e E2 |
| Pesos por commit exato e identidade do código | seção 2 |
| Busca de limiares (grade e critério) | seção 4 |
| Métrica primária: F1 de `negation_of` | seção 7 |
| Situação do TEST e análise secundária sem as notas 9534–9536 | E2 |
| Execuções já registradas (configuração, épocas, τ/t, predições e hashes) | inventário da Etapa 3 |

**O bloco JSON da seção 9 do protocolo não muda.** Ele descreve as configurações da fase de validação inicial, que continuam sendo as das execuções registradas.

## 3. Evidência do DEV (descritiva)

Fonte: `codigo/resultados/auditoria_execucoes/`.
- Métricas recalculadas das predições, idênticas às gravadas.
- Comparações pareadas por candidato e por documento.

**O DEV foi usado também para escolher as épocas e τ/t. Nada aqui é confirmatório.**

| Comparação (DEV, semente 42) | Δ F1 `negation_of` | IC95% candidato | IC95% documento | Δ macro-F1 (IC doc.) | Δ F1 `associated_with` (IC doc.) |
| --- | ---: | --- | --- | --- | --- |
| PA linear − Baseline, BioBERTpt-all | +0,0272 | [+0,0052; +0,0506] | [+0,0015; +0,0556] | +0,0219 [+0,010; +0,034] | +0,0285 [+0,009; +0,046] |
| PA linear − Baseline, BioBERTpt-clin | +0,0116 | [−0,0093; +0,0344] | [−0,0082; +0,0325] | +0,0151 [+0,007; +0,023] | +0,0286 [+0,014; +0,042] |
| PA linear − Baseline, BERTimbau-base | +0,0182 | [−0,0090; +0,0463] | [−0,0109; +0,0448] | +0,0134 [+0,004; +0,023] | +0,0173 [+0,005; +0,030] |
| PA linear − Baseline calibrado, BioBERTpt-all | +0,0200 | [0,0000; +0,0416] | [−0,0017; +0,0434] | +0,0196 [+0,009; +0,030] | +0,0290 [+0,009; +0,047] |
| PA linear − Two-Stage, BioBERTpt-all | −0,0013 | [−0,0347; +0,0327] | [−0,0328; +0,0296] | +0,0157 [−0,008; +0,038] | +0,0706 [+0,013; +0,126] |
| PA linear − Two-Stage, BioBERTpt-clin | −0,0062 | [−0,0367; +0,0256] | [−0,0388; +0,0255] | +0,0159 [−0,008; +0,040] | +0,0738 [+0,014; +0,135] |
| PA linear − Two-Stage, BERTimbau-base | −0,0071 | [−0,0361; +0,0219] | [−0,0480; +0,0289] | −0,0151 [−0,033; +0,001] | −0,0260 [−0,051; +0,001] |

**Leitura:**
- Contra o baseline, a Pair-Aware linear tem macro-F1 e F1 de `associated_with` maiores nos três encoders, nos dois níveis de reamostragem.
- Em `negation_of`, a diferença só é detectada com o BioBERTpt-all.
- Contra o calibrado (BioBERTpt-all), a diferença em `negation_of` não é detectada.
- Contra o Two-Stage, nada é detectado em `negation_of`. O Two-Stage depende de dois treinos e de um τ escolhido no DEV, que ficou no extremo da grade com os dois BioBERTpt.

## 4. Alterações

### A1. Arquitetura principal [DECIDIDO PELO AUTOR]

**Texto.** A Pair-Aware linear, na configuração da D10, é a arquitetura principal:
- todos os pares candidatos;
- sem léxico;
- cabeça `[h_CLS ; h_[E1] ; h_[E2]] → Dropout → Linear(3d, 3)`;
- 3 épocas;
- seleção pelo macro-F1 do DEV completo.

| Papel | Estratégia |
| --- | --- |
| Arquitetura principal | Pair-Aware linear |
| Referência experimental | Baseline padrão |
| Controles | Baseline calibrado e filtro léxico |
| Estratégia experimental, não adotada | Two-Stage |
| Variante exploratória | Pair-Aware MLP |

**Antes.** A seção 7 tomava o Two-Stage como objeto das comparações principais; a Pair-Aware era comparação exploratória.

**Justificativa.**
- Decisão do autor: o Two-Stage não apresentou, no DEV, ganho suficientemente relevante para justificar a sua maior complexidade:
  - dois classificadores treinados separadamente;
  - um limiar τ escolhido no DEV, no extremo da grade com os dois BioBERTpt;
  - perda de grande parte de `associated_with`.
- A evidência da seção 3 é compatível com a decisão. A Pair-Aware linear:
  - tem F1 de `negation_of` equivalente ao do Two-Stage;
  - não usa limiar escolhido no DEV;
  - melhora o baseline nas outras classes nos três encoders.

**Limite declarado.** A decisão foi tomada com o DEV conhecido. Ela não afirma superioridade da Pair-Aware linear sobre o Two-Stage na classe-alvo.

### A2. Encoder de referência [DECIDIDO PELO AUTOR]

**Texto.** O BioBERTpt-all (`pucpr/biobertpt-all`, revisão `36b11cf…`) continua sendo o encoder de referência, como no protocolo (seção 1).
- As execuções com o BioBERTpt-clin e o BERTimbau-base continuam registradas como **variantes experimentais de encoder**, fora do pré-registro.
- Esses e outros encoders podem continuar sendo explorados na etapa de aprimoramento (A4).
- Nenhum deles substitui automaticamente o encoder de referência. Uma substituição só ocorre na seleção da configuração final (A5).

**Antes.** O protocolo fixava o BioBERTpt-all e deixava os outros encoders fora do escopo. A revisão 1 desta proposta previa formalizar os três encoders e um critério de escolha entre eles; isso foi retirado.

**Justificativa.**
- Mantém a escolha pré-registrada e evita uma seleção de encoder apoiada no DEV.
- No DEV, nenhuma diferença entre encoders foi detectada em `negation_of`. A diferença detectada a favor do BioBERTpt-clin está em `associated_with`, a classe mais ruidosa da anotação, num DEV pobre nessa classe.

### A3. Configuração de referência atual [DECIDIDO PELO AUTOR]

**Texto.** A configuração de referência atual do RECLin-PT é Pair-Aware linear + BioBERTpt-all + semente 42.
- Execução registrada: `pairaware_biobertpt_seed42`.
- Código: identidade `ecbb98…`, commit `cb88c52`.

É a referência **até o momento**, não a configuração definitiva. Novos experimentos são comparados com ela no DEV.

### A4. Etapa de aprimoramento [DECIDIDO PELO AUTOR; regras de condução PROPOSTAS]

**Texto.** Até o congelamento da configuração final, são permitidos novos treinamentos, alterações de hiperparâmetros e de encoders, e variantes da Pair-Aware linear.

**Regras de condução [PROPOSTO]**, que repetem as salvaguardas que o projeto já aplica:
1. **Só TRAIN e DEV.** O TEST não é lido, nem para diagnóstico. O código já recusa o TEST no treino e na seleção.
2. **Registro de todas as tentativas**, inclusive as interrompidas e as descartadas, cada uma com:
   - a configuração;
   - a identidade do código;
   - a proveniência dos pesos;
   - os resultados no DEV.

   É a mesma exigência adotada no rascunho da E1 para o desenvolvimento do Two-Stage ("registro de todas as tentativas").
3. **Execuções existentes intocadas.** Nenhum resultado, predição, checkpoint ou partição existente é alterado ou apagado. Cada experimento novo grava uma execução nova.
4. **Comparação no DEV.** Cada experimento é comparado com a configuração de referência (A3) e com o baseline e o baseline calibrado do mesmo encoder, quando existirem. Os critérios estatísticos dessa comparação estão pendentes (E3-D3).

**Justificativa.** As decisões de desenvolvimento ficam no DEV e preservam o TEST para uma única estimativa do modelo final (A6).

**Risco declarado.** Cada experimento novo avaliado no mesmo DEV aumenta o otimismo das estimativas do DEV e a chance de escolher uma configuração por acaso. O registro de todas as tentativas permite relatar quantas configurações foram consideradas.

### A5. Modelo final [DECIDIDO PELO AUTOR; critério de seleção PENDENTE]

**Texto.** O modelo final do RECLin-PT só é definido depois da conclusão dos experimentos de aprimoramento e do congelamento da configuração.

**Antes do congelamento, é preciso registrar:**
- o critério de seleção da configuração final, só com o DEV;
- a configuração escolhida;
- a execução correspondente, com os hashes do `melhor_modelo/`.

Isso resolve a pendência C13 do protocolo.

**Antes.** A revisão 1 propunha que o modelo final fosse uma das execuções existentes, sem novos treinos. Isso foi substituído pela decisão de continuar a experimentação.

O critério de seleção é a decisão **E3-D6**, pendente (seção 7).

### A6. TEST reservado ao modelo final [DECIDIDO PELO AUTOR]

**Texto.**
- O TEST é avaliado **uma única vez**, **exclusivamente para o modelo final** do RECLin-PT, depois do congelamento (A5) e com autorização específica do autor.
- Nenhum outro sistema é avaliado no TEST: nem baseline, calibrado, filtro, Two-Stage e MLP, nem as variantes de encoder e os experimentos de aprimoramento.
- O TEST não é usado para comparar experimentos.
- Nenhum resultado do TEST muda a configuração, e o TEST não é reavaliado.

**O que é relatado do TEST [PROPOSTO]:**
- precisão, recall e F1 de `negation_of` (métrica primária);
- F1 de `associated_with` e de `no_relation`, macro-F1 e MCC;
- intervalo de confiança das métricas, com o procedimento definido em E3-D3;
- a análise secundária sem as notas 9534–9536 (E2).

**Antes.**
- O protocolo previa o TEST de todos os sistemas (seção 6, passo 7).
- Previa também comparações principais no TEST (seção 7), centradas no Two-Stage.
- A revisão 1 desta proposta previa comparações principais da Pair-Aware linear contra o baseline e o calibrado no TEST.

**Justificativa (decisão do autor).** Reservar o TEST a uma estimativa única do modelo final, sem que ele participe de nenhuma escolha entre sistemas.

**Consequência declarada.** Sem o baseline e o calibrado no TEST, a fase **não terá comparação confirmatória** entre a Pair-Aware linear e o baseline:
- as diferenças entre sistemas continuam apoiadas só no DEV e são exploratórias;
- o TEST estima o desempenho do modelo final, mas não a vantagem dele sobre outra estratégia.

O TCC deve formular as conclusões sobre estratégias e encoders nesses termos.

### A7. Comparações com o baseline e o baseline calibrado [DECIDIDO PELO AUTOR]

**Texto.** O baseline padrão e o baseline calibrado do mesmo encoder continuam sendo as referências de comparação da arquitetura principal, **no DEV**, durante o desenvolvimento e na escolha da configuração final.
- O calibrado continua sendo o controle que separa o efeito da arquitetura do efeito de recalibrar a decisão (protocolo, seção 7).
- No DEV atual, essa separação não foi obtida para o BioBERTpt-all: a diferença para o calibrado não é detectada.
- As comparações do protocolo que envolvem o Two-Stage permanecem registradas como descritivas do DEV (TCC, Capítulo 6).

**Critérios estatísticos:** pendentes (E3-D3).

## 5. Separação entre o que vem do DEV e o que fica para o TEST

| Decisão | Base | Momento | Pode mudar depois do TEST? |
| --- | --- | --- | --- |
| Configuração da Pair-Aware linear (D10) | protocolo, antes do treino | 13/10/2026 | não |
| Arquitetura principal (A1) e encoder de referência (A2) | decisão do autor, com o DEV conhecido | 01–02/11/2026; formalizadas aqui | não |
| Épocas, τ e t das execuções registradas | DEV (protocolo, seções 3 e 4) | ao fim de cada execução | não |
| Experimentos de aprimoramento (A4) | TRAIN e DEV | até o congelamento | não |
| Critério de seleção e configuração final (A5) | DEV | antes do congelamento | não |
| Desempenho do modelo final | **TEST**, uma vez (A6) | depois do congelamento e da autorização | — (é o resultado) |

## 6. Pré-requisitos antes da avaliação do TEST (nenhum executado)

1. **Aprovação desta emenda**, com E3-D3 e E3-D6 decididas e o SHA-256 registrado na seção 8 do protocolo.
2. **Conclusão dos experimentos de aprimoramento**, com o registro de todas as tentativas (A4).
3. **Congelamento do modelo final** (A5):
   - critério e configuração registrados;
   - execução identificada pelos hashes;
   - `pretreino.py conferir --execucoes` sobre ela.
4. **Proveniência do modelo final:**
   - conferência do `melhor_modelo/` pelos hashes;
   - identidade do código registrada.
5. **Autorização específica do TEST** pelo autor (protocolo, seção 6; E2).

## 7. Decisões

| # | Decisão | Situação |
| --- | --- | --- |
| E3-D1 | Pair-Aware linear como arquitetura principal | **decidida** (02/11/2026): sim (A1) |
| E3-D2 | Encoder | **decidida**: BioBERTpt-all como referência; outros encoders só experimentais (A2) |
| E3-D3 | Critérios estatísticos das comparações no DEV e dos intervalos do TEST | **pendente** |
| E3-D4 | Sistemas avaliados no TEST | **decidida**: só o modelo final (A6) |
| E3-D5 | Modelo final | **decidida**: definido depois dos experimentos e do congelamento (A5) |
| E3-D6 | Critério de seleção da configuração final, no DEV | **pendente** (consequência de E3-D5) |

**Alternativas já levantadas para E3-D3**, nenhuma assumida:
- (a) o procedimento atual do protocolo: bootstrap por candidato, sem correção;
- (b) bootstrap por candidato e por documento, com detecção nos dois níveis e correção de Holm na família de comparações;
- (c) só por documento.

Para o TEST, a escolha define o intervalo de confiança das métricas do modelo final.

**Alternativas para E3-D6**, nenhuma assumida:
- (a) manter a configuração de referência (A3), salvo se um experimento a superar no DEV pelo critério de E3-D3;
- (b) escolher a de maior F1 de `negation_of` no DEV, com desempate pelo macro-F1;
- (c) uma regra hierárquica `negation_of` → `associated_with`, como a da revisão 1.

Em qualquer caso, o critério precisa ser registrado **antes** de conhecidos os resultados dos experimentos que ele vai julgar, ou declarado como pós-DEV.

## 8. Riscos e limitações declarados

- **Emenda pós-DEV.** Arquitetura e encoder de referência foram fixados com o DEV conhecido.
- **Sem comparação confirmatória entre sistemas** (A6). As vantagens da Pair-Aware linear sobre o baseline são evidência do DEV, que continua sendo reutilizado.
- **Reuso do DEV na etapa de aprimoramento** (A4). Aumenta o otimismo das estimativas do DEV.
- **Uma semente e uma partição.** No legado, trocar só a semente mudou o F1 de `negation_of` do baseline em cerca de 0,04 (`entregas/auditoria-tcc-completa.md`, 6.4).
- **O TEST não é inédito** (E2), e três notas dele têm cópia no TRAIN.
- **Proveniência:** execuções retomadas registram só a sessão final; há quatro identidades de código entre as execuções registradas. Ver o inventário da Etapa 3.

## 9. Registro verificável (proposta; não conferido por teste algum)

```json
{
  "emenda": "E3",
  "revisao": 2,
  "estado": "proposta",
  "ativa": false,
  "redigida_em": "2026-11-02",
  "natureza": "pos-DEV, pre-TEST",
  "altera_bloco_json_do_protocolo": false,
  "arquitetura_principal": "pair_aware (configuracao D10)",
  "encoder_de_referencia": "pucpr/biobertpt-all@36b11cfdfa49876b108fdbe88eaeeb9be3b03f1c",
  "configuracao_de_referencia": {"execucao": "pairaware_biobertpt_seed42", "seed": 42, "definitiva": false},
  "etapa_de_aprimoramento": {"permitida": true, "particoes": ["train", "dev"], "registro_de_todas_as_tentativas": true},
  "modelo_final": "definido apos os experimentos e o congelamento; criterio: [PENDENTE: E3-D6]",
  "test": {"sistemas": ["modelo final"], "avaliacoes": 1, "comparacao_entre_sistemas": false},
  "estatistica": "[PENDENTE: E3-D3]",
  "fonte_da_evidencia_dev": "codigo/resultados/auditoria_execucoes/"
}
```
