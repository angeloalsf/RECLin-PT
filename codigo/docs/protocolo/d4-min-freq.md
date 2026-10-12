# Pré-registro — seleção do `min_freq` do léxico de pistas (pendência D4)

Registrado em 11/10/2026, **antes de calcular qualquer resultado** da
validação cruzada descrita abaixo. Este documento é o que o procedimento
segue; o SHA-256 dele é gravado no resultado
(`docs/entregas/d4/selecao_min_freq.json`), e o script recusa rodar se as
constantes do código diferirem das registradas aqui (bloco JSON no fim).

## Por que

`min_freq=3` (léxico congelado `70c93fa807de`, 11 formas) foi escolhido no
legado por uma calibração sobre as predições do DEV de quatro baselines do
legado. A diretriz da fase é não usar resultados históricos como entrada dos
experimentos novos (pendência D4). Este procedimento escolhe o `min_freq`
**só com o TRAIN da v2**, sem nenhum modelo, nenhuma predição, nenhum
resultado histórico, sem o DEV e sem o TEST.

O que já era conhecido antes deste registro, e não é resultado da validação
cruzada: o tamanho do léxico induzido do TRAIN inteiro para cada valor (51 /
17 / 11 / 9 / 8 formas para 1 / 2 / 3 / 5 / 10, registrado na etapa 3) e que
o léxico com `min_freq=3` é o mesmo na v1 e na v2.

## Procedimento

1. **Dados.** Só `codigo/dados/particoes-v2/train.jsonl`, conferido contra o
   MANIFEST da v2 (`18264860…3b70`) e contra o registro de versões
   (`b8aaafce…d034`). A execução usa uma pasta com só o `train.jsonl` e o
   `MANIFEST.json` da v2: o DEV e o TEST não existem nela.
2. **Unidade de avaliação.** Os pares candidatos da tarefa
   (`tarefa.iter_candidate_pairs`, `max_gap = 25`), com o rótulo anotado.
   Positivo verdadeiro: rótulo `negation_of`.
3. **Detector avaliado.** "O `e1` do par é pista" (`lexico.e_pista`, com a
   normalização do léxico: NFKD, sem diacríticos, espaços colapsados,
   `casefold`). É exatamente o uso do léxico no filtro, no fine-tuning
   restrito e na Pair-Aware (o espaço em que `negation_of` pode ser predito).
4. **Validação cruzada.** 5 partes, por documento (nunca um documento em duas
   partes). Estratificação pela presença de ao menos uma relação anotada
   `negation_of` no documento, como em `particoes.particionar`: os documentos
   ordenados por `doc_id` (chave numérica) são separados em "com" e "sem"
   `negation_of`; um `random.Random(42)` embaralha o grupo "com" e depois o
   grupo "sem"; o i-ésimo documento de cada grupo vai para a parte `i mod 5`.
5. **Candidatos.** `min_freq ∈ {1, 2, 3, 5, 10}` (a grade histórica; nenhum
   outro valor é avaliado).
6. **Para cada valor e cada parte:** induzir o léxico (`lexico.induzir`) das
   outras quatro partes com o `min_freq` do valor e `max_gap = 25`; nos pares
   candidatos da parte retida, contar VP (`negation_of` com `e1` pista), FP
   (outro rótulo com `e1` pista) e FN (`negation_of` com `e1` não pista).
7. **Métrica.** VP, FP e FN somados nas 5 partes; F1 = 2·VP / (2·VP + FP + FN)
   (micro, pelas contagens). A comparação entre valores é exata (frações de
   inteiros, sem arredondamento).
8. **Regra de seleção.** O valor de maior F1. **Desempate:** o maior
   `min_freq` (léxico menor).
9. **Resultado.** Aceito qualquer que seja. Não se testam depois outros
   valores, outras métricas, outro número de partes ou outra semente. Precisão,
   cobertura (recall) e o F1 de cada parte são registrados como descrição, sem
   papel na escolha.
10. **Registro.** O léxico final é o induzido do TRAIN v2 inteiro com o valor
    escolhido. Se o valor for 3, o léxico congelado não muda (11 formas,
    `70c93fa807de`); só a sua origem passa a ser este procedimento. Se for
    outro, `reclin.negacao.lexico.CONGELADO` passa a ser o novo léxico
    (`min_freq`, `n_formas`, `lexico_sha1`, origem), o registro anterior é
    preservado à parte para os testes de regressão do legado, e o protocolo
    da fase é atualizado antes de qualquer treino.
11. **Proveniência gravada.** SHA-256 do `train.jsonl`, do MANIFEST, deste
    documento e dos arquivos de código do procedimento; hash da atribuição das
    partes; contagens por valor e parte; tamanho e `lexico_sha1` do léxico
    final; versão do Python.

## Limitações declaradas

* O critério mede o léxico como detector, não o filtro aplicado a um modelo:
  a escolha não usa nenhum modelo por desenho.
* Cada parte induz o léxico de 4/5 do TRAIN; as frequências ficam ~20%
  menores que no TRAIN inteiro e o mesmo limiar é usado (sem reescala).
* A precisão do detector é baixa por construção (uma pista forma pares com
  vários `e2`); ela entra no F1 da mesma forma para todos os valores.

## Registro verificável

```json
{
  "pendencia": "D4",
  "particao": "train",
  "versao_dados": "v2-texto-fonte",
  "train_sha256": "b8aaafce423afb9cda49fa5d116c14b14d6b9c2d09a0521bff4ef8733305d034",
  "manifest_sha256": "18264860a3d8fff7bcc335b8f5703a55d60e08893a09be2afa20b1ea446d3b70",
  "candidatos_min_freq": [1, 2, 3, 5, 10],
  "max_gap": 25,
  "n_partes": 5,
  "semente": 42,
  "estratificacao": "presença de negation_of no documento",
  "metrica": "F1 = 2VP/(2VP+FP+FN), contagens somadas nas partes",
  "desempate": "maior min_freq"
}
```
