# Divergências da prosa do TCC após a regeneração de 20/09/2026

Gerado depois de rodar, na ordem do Makefile, sobre os 4 JSONs de **17-20/09**:

```
src/significance.py (s42 e s43, --n-boot 10000 --seed <semente>)
scripts/aggregate_seeds.py
scripts/make_tcc_artifacts.py
scripts/make_tcc_curves.py
scripts/make_tcc_eda.py --check-against-results
scripts/check_tcc_numbers.py    -> 50 divergência(s)
```

Nenhum `.tex` de prosa foi tocado. Esta é a lista de edição manual.

---

## 0. Antes de tudo: o `significance_*_seed43.json` estava fora da especificação

O arquivo que estava em `results/` desde 20/09 16:30 foi gerado com **`--seed 42`**
no bootstrap, não com `--seed 43` como manda a regra do Makefile
(`SIGNIF_RULE` → `--seed $(1)`). Confirmado por reprodução: rodar com `--seed 42`
sobre as preds da semente 43 devolve **exatamente** os valores do arquivo antigo
(IC95% `[−0,016374800796585005; +0,05727565041459665]`, p = 0,2804); com `--seed 43`
devolve outros.

O de seed 42 estava correto e saiu **byte-idêntico** na regeneração.

Valores oficiais agora (bootstrap, semente 43): **IC95% [−0,018; +0,057], p = 0,2878**
(antes: `[−0,016; +0,057]`, p = 0,2804). O veredito não muda — não é significativo
nos dois casos — mas **`COMPARACAO_determinismo_20set.md` cita 0,2804 e o IC antigo**
na tabela de significância. Corrigir lá também (o arquivo antigo foi preservado em
`archive_pre_determinismo/` como `*.OFFSPEC-bootstrap-seed42.json`).

Ambiente da regeneração: numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1,
matplotlib 3.10.0 — os pins de `requirements.txt`.

---

## 1. Os números novos, de uma vez

Teste (n = 19.210):

| execução | macro-F1 | F1 `negation_of` | P `neg` | R `neg` | F1 `assoc` | F1 `no_rel` | MCC | acurácia |
|---|---|---|---|---|---|---|---|---|
| BioBERTpt s42 | 0,693 | 0,696 | 0,555 | 0,934 | 0,442 | 0,942 | 0,478 | 0,894 |
| BioBERTpt s43 | 0,713 | 0,739 | 0,617 | 0,921 | 0,453 | 0,946 | 0,485 | 0,902 |
| BERTimbau s42 | 0,698 | 0,711 | 0,582 | 0,914 | 0,441 | 0,941 | 0,476 | 0,893 |
| BERTimbau s43 | 0,704 | 0,718 | 0,592 | 0,914 | 0,447 | 0,945 | 0,478 | 0,900 |

Média e amplitude entre sementes:

| modelo | macro-F1 (méd / ampl) | F1 `negation_of` (méd / ampl) |
|---|---|---|
| BioBERTpt | 0,703 / **0,020** | 0,717 / **0,043** |
| BERTimbau | 0,701 / **0,006** | 0,715 / **0,007** |

Significância:

| | McNemar `b`(Bio) / `c`(BERT) | p | bootstrap IC95% | p | veredito |
|---|---|---|---|---|---|
| semente 42 | 540 / 516 | **0,479** | [−0,051; +0,021] | 0,4204 | Não |
| semente 43 | 559 / 510 | **0,142** | [−0,018; +0,057] | 0,2878 | **Não** |

Acurácia: s42 Bio 0,894 × BERT 0,893 · s43 Bio 0,902 × BERT 0,900.

`dev_history` (F1 `negation_of` na 1ª época): Bio s42 0,643 / s43 0,628 (**Δ 0,015**);
BERT s42 0,674 / s43 0,525 (**Δ 0,149**).

---

## 2. Divergências por arquivo e linha

### `tcc/src/textuais/experimentos_e_resultados.tex`

| linha(s) | o texto diz | os artefatos dizem |
|---|---|---|
| 41–42 | perda de validação do BERTimbau s43 **sobe** de 0,426 (ép. 2) para 0,474 (ép. 3), macro-F1 de 0,654 para 0,680 | dev_loss 0,473 (ép. 2) → **0,469** (ép. 3) — **cai**; macro-F1 0,651 → 0,664 |
| 51 | F1 `negation_of` de 0,436 (s42) contra 0,588 (s43), no BioBERTpt | 0,643 (s42) e 0,628 (s43) |
| 53–54 | distância de 0,152 no BioBERTpt contra 0,056 no BERTimbau | **0,015** no BioBERTpt contra **0,149** no BERTimbau |
| 77 | macro-F1 0,686 (BERTimbau) e 0,675 (BioBERTpt) | 0,698 e 0,693 |
| 78 | diferença de 0,011 a favor do geral | **0,004** |
| 79–80 | F1 `negation_of` 0,694 (BERTimbau) contra 0,677 (BioBERTpt) | 0,711 e 0,696 |
| 82 | recall na negação *idêntico*, 0,901 nos dois | **0,934** (Bio) e **0,914** (BERT) — não são mais iguais |
| 83 | precisão ≈ 0,54–0,56 | 0,555 (Bio) e 0,582 (BERT) → ≈ 0,55–0,58 |
| 84 | F1 de `associated_with` ≈ 0,42–0,43 | 0,442 e 0,441 → ≈ 0,44 |
| 140 | diferença observada de −0,017 | **−0,015** |
| 145 | McNemar **rejeita** H₀, p ≈ 1,1×10⁻⁵ | p = **0,479** — não rejeita |
| 146–147 | 1.235 discordâncias, 540 Bio contra 695 BERT, "assimetria grande demais para o acaso" | **1.056** discordâncias, **540 contra 516** — praticamente simétrica |
| 147–150 | "há diferença mensurável no conjunto dos candidatos, e ela aponta para o *encoder* geral, coerente com a vantagem em macro-F1 e em `associated_with`" | não há diferença mensurável; a contagem favorece o **clínico**; e em `associated_with` o BioBERTpt agora vence (0,442 × 0,441) |
| 154 | IC95% [−0,053; +0,018] | **[−0,051; +0,021]** |
| 155 | p = 0,345 | **0,420** |
| 157–162 | "a vantagem do *encoder* geral é real no agregado…"; "…ao contrário do que aponta o McNemar, que se repete" | não há vantagem agregada; o McNemar não rejeita em nenhuma das duas sementes |
| 175–176 | BioBERTpt: macro-F1 0,675 → 0,712; F1 `neg` 0,677 → 0,754; amplitude 0,078 | 0,693 → 0,713; 0,696 → 0,739; amplitude **0,043** |
| 177–179 | amplitude "quase **cinco vezes maior**" que a diferença de 0,017 na s42 | 0,043 contra 0,015 ≈ **2,9×** |
| 184–185 | BERTimbau: macro-F1 0,686 → 0,704; F1 `neg` 0,694 → 0,702; amplitude 0,008 | 0,698 → 0,704; 0,711 → 0,718; amplitude **0,007** |
| 185–186 | "cerca de **dez vezes** menor que a do BioBERTpt" | 0,007 contra 0,043 ≈ **6×** |
| 195 | margem de 0,053 a favor do clínico na s43 | **0,020** |
| 196–198 | bootstrap "passa a acusar a diferença como **significativa**" — IC95% [+0,016; +0,090], p = 0,0054 | IC95% **[−0,018; +0,057]**, p = **0,288** → **inclui o zero, não é significativa** |
| 202–204 | McNemar "rejeita a hipótese nula nas duas sementes e, nas duas, aponta para o mesmo lado" | não rejeita em nenhuma; aponta para o **clínico** nas duas |
| 205–206 | s43: 1.066 discordâncias, 442 Bio contra 624 BERT, p ≈ 2,7×10⁻⁸ | **1.069**, **559 contra 510**, p = **0,142** |
| 207–208 | s42: 540 contra 695, p ≈ 1,1×10⁻⁵ | 540 contra **516**, p = **0,479** |
| 209 | acurácia 0,880 × 0,888 (s42); 0,898 × 0,908 (s43) | **0,894 × 0,893** (s42); **0,902 × 0,900** (s43) — o clínico fica à frente nas duas |
| 216–219 | "a conclusão do teste pareado alterna entre 'sem diferença' e 'diferença significativa'" | as duas sementes dão **sem diferença**; só o **sinal** inverte |
| 224–226 | "no agregado ela atravessa, pois tanto o McNemar quanto a acurácia apontam para o *encoder* geral nas duas" | os dois apontam para o **clínico** nas duas, sem significância |
| 230 | amplitude 0,078 contra 0,008 | **0,043 contra 0,007** |
| 231–233 | "BioBERTpt marginalmente à frente na métrica-alvo (0,715 × 0,698) e **empatado** em macro-F1 (0,694 × 0,695)" | 0,717 × 0,715 (margem cai de 0,017 para **0,003**) e macro-F1 0,703 × 0,701 — **à frente nas duas**, não empatado |
| 234 | "dispersão uma **ordem de grandeza** maior" | 6×, não 10× |

### `tcc/src/pre_textuais/resumo.tex`

| linha(s) | o texto diz | os artefatos dizem |
|---|---|---|
| 22 | 0,677 contra 0,694 | **0,696 contra 0,711** |
| 23 | IC95% [−0,053; +0,018], p = 0,345 | **[−0,051; +0,021]**, p = **0,420** |
| 24–26 | s43 inverte o sinal e favorece o clínico "de forma **significativa**" (IC95% [+0,016; +0,090]) | IC95% **[−0,018; +0,057]** — inclui o zero. A **inversão de sinal permanece**, a significância **não** |
| 28–30 | McNemar aponta para o *encoder* geral nas duas sementes | aponta para o **clínico** nas duas, e não rejeita H₀ em nenhuma |
| 31–33 | "acompanha o clínico em macro-F1 médio… e é cerca de **dez vezes** mais estável" | macro-F1 médio 0,701 × 0,703 (ainda acompanha); estabilidade ≈ **6×** na métrica-alvo, ≈ **3,4×** em macro-F1 |

### `tcc/src/textuais/discussao.tex`

| linha(s) | o texto diz | os artefatos dizem |
|---|---|---|
| 14–17 | empate na s42 que "não se repete na s43, em que o bootstrap acusa diferença **significativa** a favor do clínico" | empate nas **duas**; na s43 o sinal inverte mas sem significância |
| 19–21 | McNemar rejeita H₀ nas duas e aponta nas duas para o geral | não rejeita em nenhuma; aponta para o **clínico** |
| 22–23 | 540 × 695 (s42); 442 × 624 (s43); "assim como a acurácia" | **540 × 516**; **559 × 510**; a acurácia também inverte de lado |
| 24–26 | "em ambas as sementes, a variação de um mesmo *encoder* supera a diferença medida entre *encoders*" | sobrevive só lido pela amplitude do BioBERTpt (0,043 > 0,015 e > 0,020); pela do BERTimbau (0,007) **não** vale na s43. Vale precisar de qual amplitude se fala |
| 59–60 | "O BERTimbau **iguala** o BioBERTpt em macro-F1 médio (0,695 × 0,694)" | **0,701 × 0,703** — fica 0,002 **atrás**, não iguala |
| 64–65 | amplitude 0,008 do BERTimbau, "cerca de **dez vezes** menor" que 0,078 do BioBERTpt | **0,007** e **0,043** ≈ **6×** |
| 95 | "diferença significativa, e em favor do BioBERTpt clínico" (ameaças à validade) | não significativa |
| 137–138 | "a conclusão de empate na s42 (e a de **diferença significativa** na s43)" | empate nas duas |

### `tcc/src/textuais/conclusao.tex`

| linha(s) | o texto diz | os artefatos dizem |
|---|---|---|
| 30–31 | diferença de 0,017 a favor do geral na s42 | **0,015** |
| 33 | IC95% [−0,053; +0,018], p = 0,345 | **[−0,051; +0,021]**, p = **0,420** |
| 34–36 | s43 inverte e favorece o clínico "de forma **significativa**" (IC95% [+0,016; +0,090], p = 0,0054) | IC95% **[−0,018; +0,057]**, p = **0,288** — não significativa |
| 38 | 0,078 de amplitude no BioBERTpt | **0,043** |
| 40–43 | McNemar rejeita nas duas e aponta nas duas para o geral (540 × 695; 442 × 624) | não rejeita; **540 × 516** e **559 × 510**; aponta para o **clínico** |
| 43–46 | "o *encoder* geral **acompanha** o clínico em macro-F1 médio (0,695 × 0,694)… cerca de **dez vezes** mais estável" | 0,701 × 0,703; ≈ **6×** (métrica-alvo) e ≈ **3,4×** (macro-F1) |
| 61–62 | o clínico fica **0,017** à frente na métrica-alvo (0,715 × 0,698) | **0,003** à frente (0,717 × 0,715) |
| 63 | amplitude do BioBERTpt de 0,078 | **0,043** |
| 72 | "varia cerca de **dez vezes** menos entre sementes na métrica-alvo" | ≈ **6×** |
| 81 | `associated_with` F1 ≈ 0,42–0,43 contra ≈ 0,68–0,69 de `negation_of` | ≈ **0,44** contra ≈ **0,70–0,71** |

---

## 3. O que mudou de conclusão, em ordem de gravidade

**1. Nenhum teste rejeita H₀ em nenhuma semente.** A significância do bootstrap na
semente 43 (p = 0,0078 na rodada arquivada, p = 0,2878 agora) era o único resultado
significativo do TCC. Com ela cai a estrutura retórica de §6.5 e §7.1 — "alterna entre
sem diferença e diferença significativa". O que resta é mais simples e mais defensável:
**a diferença nunca é significativa, e o sinal ainda assim inverte entre sementes.**

**2. O McNemar inverte de lado nas duas sementes e deixa de rejeitar.** A frase
"o padrão global é estável e aponta para o *encoder* geral nas duas sementes" —
que era o contrapeso da inversão da métrica-alvo — está morta em quatro arquivos
(§6.3, §6.5, Cap. 7, Cap. 8, resumo). A acurácia acompanha a inversão.

**3. A "assimetria de dispersão" sobrevive, mas encolhe.** 0,043 × 0,007 na
métrica-alvo (≈ 6×, não 10×) e 0,020 × 0,006 em macro-F1 (≈ 3,4×). A recomendação
pelo BERTimbau continua de pé por custo e previsibilidade — só não com o número "dez".

**4. O macro-F1 médio inverte.** BioBERTpt 0,703 contra BERTimbau 0,701. "O BERTimbau
iguala/acompanha o clínico" vira "fica 0,002 atrás". A folga do clínico na métrica-alvo
cai de 0,017 para 0,003, o que na prática **reforça** o argumento de empate.

**5. §6.2 inverte quem é disperso na 1ª época — de novo, e mais forte.** Agora é o
BERTimbau que dispersa (Δ 0,149) e o BioBERTpt que não (Δ 0,015). O parágrafo das
linhas 48–56 afirma o contrário.

**6. O exemplo das linhas 41–42 deixou de existir.** No BERTimbau s43 a perda de
validação agora **cai** da época 2 para a 3. O fenômeno descrito (macro-F1 sobe
enquanto a perda piora) continua verdadeiro nas 4 execuções — só precisa de outro
par de épocas: BioBERTpt s42 (dev_loss 0,380 → 0,426, macro-F1 0,654 → 0,673) é o
substituto mais limpo.

**7. Tabela 6 (`tab:resultados`) redistribuiu o negrito.** BERTimbau leva macro-F1 e
F1 `negation_of`; BioBERTpt leva `associated_with`, `no_relation`, MCC e acurácia.
O texto de §6.3 que liga a "vantagem em `associated_with`" ao *encoder* geral
(linha 150) ficou invertido.

**8. O recall de `negation_of` deixou de ser idêntico** (0,934 × 0,914). A palavra
*idêntico* na linha 82 precisa sair.

---

## 4. O que **não** mudou — não reescrever à toa

- **A EDA não se moveu**: as 6 tabelas de EDA saíram **byte-idênticas**, `tcc_eda.json`
  também, e `--check-against-results` passou ("candidatos recomputados == n_candidates").
  Todo o Cap. 5 (corpus, distâncias, partições, candidatos, teto de recall) está intacto.
  As 5 figuras de EDA saíram **pixel-idênticas** (só metadados do PNG mudaram) e por isso
  não foram regravadas no repositório.
- Perda de treino **cai monotonicamente** nas 4 execuções (linhas 36–37).
- Perda de validação estabiliza ou volta a subir a partir da 2ª época — vale nas 4,
  com as ressalvas do item 6 acima.
- A **inversão de sinal** da métrica-alvo entre as sementes continua de pé
  (s42 → geral, s43 → clínico).
- `associated_with` continua sendo a classe mais difícil, com alta cobertura e baixa
  precisão; as matrizes de confusão continuam qualitativamente semelhantes.
- "Variação entre sementes do mesmo modelo > diferença entre *encoders*" continua
  verdadeiro pela amplitude do BioBERTpt (ver ressalva em `discussao.tex:24–26`).
- BioBERTpt à frente na negação na semente 43 (0,739 × 0,718).
- `n = 19.210`, `best_epoch = 3` nas 4, `108,9 × 177,9` milhões de parâmetros,
  `max_gap = 25`, splits e SHA-256: inalterados.
- Nenhum caminho fantasma, nenhuma cifra da rodada `max_gap=20` e nenhum
  `\input{tabelas/…}` quebrado — as verificações 3, 4 e 5 do script passaram.

---

## 5. Artefatos regravados nesta rodada

**`results/`**: `significance_biobertpt_vs_bertimbau_seed43.json` (agora com `--seed 43`),
`summary_by_seed.json` (novo — a pasta não tinha).
Byte-idênticos, não regravados: `significance_…_seed42.json`, `tcc_eda.json`.

**`tcc/src/tabelas/`**: `resultados.tex`, `robustez_semente.tex`, `significancia.tex`,
`significancia_seed43.tex`, `dev_history.tex`.

**`tcc/src/imagens/resultados/`**: `f1_por_classe.png`, `cm_biobertpt.png`,
`cm_bertimbau.png`, `curvas_treino.png`, `curvas_treino_biobertpt.png`,
`curvas_treino_bertimbau.png`.

**Não regravados de propósito**: `tcc/src/tabelas/{distribuicao_relacoes,
distancia_por_tipo,particao_relacoes,teto_recall,candidatos_por_particao,
sha256_splits}.tex` e `tcc/src/imagens/eda/*.png` — saída idêntica, regravar só
poluiria o `git status`.
