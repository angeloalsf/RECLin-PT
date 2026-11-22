# Emenda da configuração final (complemento da E3): Pair-Aware Linear sem pesos, P(neg) ≥ 0,80

Redigida em 22/11/2026 a pedido do autor, que decidiu a configuração final e
pediu o registro, a implementação e o congelamento ("Aceito fazer tudo", com a
avaliação do TEST pausada). **Estado: decidida pelo autor; o texto abaixo ainda
precisa da conferência dele** antes de o SHA-256 ser registrado na seção 8 do
protocolo (`fase-biobertpt-seed42.md`) e antes de rodar o notebook da
configuração final. Natureza: **pós-DEV, pré-TEST.** Nenhuma linha do TEST foi
lida para escrever este documento.

Complementa `emenda-e3-pair-aware.md` (revisão 2), cujas decisões E3-D3 e E3-D6
estavam pendentes. Não altera partições, candidatos, `max_gap`, encoder,
semente, hiperparâmetros de treino nem a regra de uma única avaliação do TEST.

## 1. Decisões

| # | Decisão | Texto |
| --- | --- | --- |
| E3-D3 | Critérios estatísticos | Bootstrap pareado (10.000 reamostras, semente 42, IC95% por percentis) **por candidato e por documento**; uma diferença é "detectada" quando o intervalo exclui o zero, lida nos dois níveis (o nível por documento prevalece quando divergem). Sem correção para múltiplas comparações, o que é declarado como limitação. No TEST, o intervalo das métricas do modelo final é o do bootstrap por documento. |
| E3-D6 | Critério de seleção da configuração final, no DEV | **Alternativa nova (d), pós-DEV:** prioridade à precisão de `negation_of`, objetivo declarado do projeto. Entre as configurações já medidas no DEV, escolhe-se a Pair-Aware Linear treinada sem pesos de classe, e o ponto de operação é o limiar t de P(`negation_of`) de **maior precisão de `negation_of` entre os limiares que recuperam pelo menos 80% das negações anotadas (recall ≥ 80%)**, na grade 0,05…0,95 (passo 0,05); desempates: maior F1 de `negation_of`, maior macro-F1, menor limiar. O critério foi formulado **depois** de conhecida a curva do DEV (17/11/2026) e é declarado assim. |

## 2. A configuração final

| Item | Valor |
| --- | --- |
| Modelo | execução `pair_aware__a3ac671a77e2__t2`: Pair-Aware Linear, BioBERTpt-all (`pucpr/biobertpt-all@36b11cfd…`), semente 42, `class_weight = none`, 3 épocas, melhor época 3 (macro-F1 do DEV), mesmos demais hiperparâmetros da referência anterior. **Sem treino novo.** |
| Regra de decisão | `negation_of` se P(`negation_of`) ≥ t; senão, argmax entre `associated_with` e `no_relation` (empate → `associated_with`) |
| t | 0,80, escolhido pelo critério de E3-D6: entre os limiares da grade com recall ≥ 80%, o de maior precisão (0,845) |
| Atenção | t e o piso de recall são grandezas diferentes com o mesmo valor por coincidência: t é a probabilidade mínima de `negation_of` para o modelo afirmar a relação num par; o piso é a fração mínima das negações anotadas que o modelo precisa recuperar no DEV para que t seja aceito |
| Execuções a criar (Colab, só DEV) | `pair_aware_sem_pesos_biobertpt_seed42` (reempacotamento da base sob o código atual; ver §4) e `pair_aware_limiar_sem_pesos_biobertpt_seed42` (o `limiar.json` com t e a tabela da busca) |
| O que é congelado | a derivada, com as impressões da base: configuração, identidade do código, predições do DEV, `limiar.json` e o hash da pasta `melhor_modelo/` |

A configuração de referência anterior (`pairaware_biobertpt_seed42`, perda
ponderada, argmax) deixa de ser a candidata ao TEST. A arquitetura continua a
Pair-Aware Linear e o encoder continua o BioBERTpt-all (E3-D1 e E3-D2).

## 3. Evidência do DEV (descritiva; uma semente)

Fonte: `resultados/diagnosticos/configuracao_final_dev.json`
(`diagnosticos/avaliar_configuracao_final_dev.py`), sobre as predições
versionadas; 19.064 candidatos, 98 documentos.

| Configuração | P neg | R neg | F1 neg | F1 assoc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| **Final: sem pesos, P(neg) ≥ 0,80** | **0,845** | 0,800 | 0,822 | 0,448 | 0,748 |
| Sem pesos, argmax | 0,794 | 0,873 | 0,832 | 0,448 | 0,752 |
| Ponderada, argmax (referência anterior) | 0,742 | 0,940 | 0,829 | 0,436 | 0,741 |
| Two-Stage, τ = 0,95 | 0,798 | 0,867 | 0,831 | 0,366 | 0,725 |
| Baseline | 0,696 | 0,947 | 0,802 | 0,408 | 0,719 |

Diferenças da configuração final (A − B), IC95% por documento:

| B | Δ precisão | Δ recall | Δ F1 neg | Δ macro-F1 |
| --- | --- | --- | --- | --- |
| Sem pesos, argmax | +0,051 [+0,016; +0,094]* | −0,073 [−0,117; −0,034]* | −0,010 [−0,041; +0,020] | −0,003 [−0,014; +0,007] |
| Ponderada, argmax | +0,103 [+0,059; +0,151]* | −0,140 [−0,212; −0,076]* | −0,007 [−0,053; +0,035] | +0,008 [−0,013; +0,028] |
| Two-Stage | +0,048 [+0,013; +0,085]* | −0,067 [−0,114; −0,022]* | −0,009 [−0,043; +0,022] | +0,024 [+0,005; +0,041]* |
| Baseline | +0,149 [+0,095; +0,202]* | −0,147 [−0,214; −0,086]* | +0,020 [−0,031; +0,071] | +0,030 [+0,005; +0,054]* |

\* intervalo exclui o zero.

Leitura: a configuração final não melhora o F1 de `negation_of` de nenhuma
alternativa; ela troca recall por precisão, de forma detectada, num ponto que
o critério fixa em recall ≥ 80%. Na grade, 0,80 é o joelho: de 0,75 a 0,80 os
falsos positivos caem de 28 para 22 com a perda de um verdadeiro positivo; acima
de 0,80 a precisão fica em 0,846–0,848 e o recall continua caindo.

## 4. Reempacotamento (por que há uma execução nova sem treino novo)

A base foi treinada com o código `0526d912…`, anterior à trava do TEST (G2). As
operações sobre uma execução exigem o mesmo código que a criou. Em vez de
afrouxar essa regra, o código atual **reempacota** a base
(`reclin/treino/reempacotamento.py`): recarrega os pesos, prevê o DEV e só
grava a execução nova se as classes previstas forem as mesmas nos 19.064 pares
e as probabilidades diferirem no máximo 0,001; a comparação fica registrada. Os
pesos são uma cópia conferida por hash; a base não é alterada.

## 5. Procedimento até o TEST

1. O autor confere este texto; o SHA-256 dele é registrado na seção 8 do
   protocolo como complemento da E3, ao lado da E2.
2. Commit e push do código (o notebook clona o `main`).
3. Notebook `codigo/notebooks/final/reclin_configuracao_final.ipynb` (Colab,
   GPU, só DEV): confere os SHA-256 dos pesos no Drive, reempacota, escolhe e
   congela t (o notebook exige t = 0,80 com 120/22/30) e grava o registro de
   congelamento no Drive.
4. **Só com autorização explícita do autor**, uma única vez: `avaliar_test.py`
   na reempacotada e `pair_aware_limiar.py aplicar-test` na derivada, com o
   código de liberação. A trava recusa outra ordem, outra execução ou uma
   segunda avaliação. O resultado reportado é o da derivada; as métricas de
   argmax que a base grava no primeiro passo são um subproduto técnico e não
   são usadas para nenhuma escolha.

## 6. Riscos declarados

- **Seleção pós-DEV e otimista.** O modelo, o limiar e o critério foram escolhidos
  com o DEV conhecido; o desempenho no DEV superestima o esperado. O TEST é a
  estimativa honesta.
- **Limiar no limite do piso.** No DEV, t = 0,80 tem recall exatamente 80%
  (120 de 150). Com outra amostra, o recall pode ficar abaixo do piso.
- **Uma semente.** A calibração das probabilidades muda entre sementes; o mesmo
  t pode não ser o mesmo ponto em outro treino.
- **Probabilidades com 4 casas** nos sidecars, como em todas as regras de limiar
  do projeto.
- **Sem comparação confirmatória** entre sistemas (A6 da E3): o TEST estima só o
  modelo final.

## 7. Experimento pré-registrado: 5 épocas (antes do congelamento)

Aprovado pelo autor em 22/11/2026, **antes** do treino. A execução sem pesos é
repetida com `epochs = 5` e nada mais alterado
(`codigo/notebooks/teste/reclin_pair_aware_sem_pesos_5epocas.ipynb`; melhor época
pelo macro-F1 do DEV, entre 1 e 5).

**Regra de decisão:** em cada modelo (3 e 5 épocas), t é escolhido pelo critério
de E3-D6. **A versão de 5 épocas substitui a de 3 só se a precisão de
`negation_of` dela, no seu t, for maior que a da versão de 3 épocas no seu t,
com o IC95% por documento da diferença (bootstrap pareado, 10.000 reamostras,
semente 42) todo acima de zero.** Caso contrário, a configuração final continua a
da §2. Recall, F1 e macro-F1 são relatados, mas não entram na regra.

O congelamento (§5, passo 3) só é feito depois de aplicada esta regra. Se a
escolha for 5 épocas, a §2, o notebook da configuração final e o TCC são
atualizados para a nova base antes do congelamento.

## 8. Registro verificável

```json
{
  "emenda": "E3-complemento-configuracao-final",
  "redigida_em": "2026-11-22",
  "estado": "decidida pelo autor; texto a conferir",
  "natureza": "pos-DEV, pre-TEST",
  "E3-D3": "bootstrap pareado por candidato e por documento, 10000 reamostras, semente 42, IC95% percentis; sem correcao multipla",
  "E3-D6": "(d) pos-DEV: maior precisao de negation_of com recall >= 4/5, grade 0,05-0,95, sobre a Pair-Aware Linear sem pesos",
  "base": "pair_aware__a3ac671a77e2__t2",
  "regra": "P(negation_of) >= t -> negation_of; senao argmax(associated_with, no_relation)",
  "t_esperado": 0.8,
  "dev_esperado": {"tp": 120, "fp": 22, "fn": 30},
  "experimento_5_epocas": "5 epocas so se IC95% por documento da diferenca de precisao (cada modelo no seu t) > 0; senao 3 epocas",
  "test": {"avaliacoes": 1, "autorizacao": "pendente"}
}
```
