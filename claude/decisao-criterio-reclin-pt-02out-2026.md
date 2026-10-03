# Critério de decisão do RECLin-PT, fixado antes da análise comparativa (02/10/2026)

**Registro:** 02/10/2026. A hora oficial é a do commit que introduz este arquivo
(`git log --format=%cI -- claude/decisao-criterio-reclin-pt-02out-2026.md`), por ser
a marca verificável. Nenhuma hora foi escrita à mão.

> **Este registro foi feito antes da execução da análise comparativa
> (`scripts/analise_ftr.py` ou equivalente); qualquer resultado obtido depois desta
> data deve ser avaliado contra esta regra, não o contrário.**

## Contexto

Duas abordagens de especialização em `negation_of` foram construídas sobre os
baselines oficiais do Cap. 6 (rodada de 17-20/09/2026). Uma delas será apresentada
como o RECLin-PT, a contribuição principal do TCC, e a outra como alternativa testada
e documentada.

| | Filtro de pista lexical | Fine-tuning no espaço restrito |
|---|---|---|
| fase | Fase 2, concluída | Fase 2-B, em avaliação |
| natureza | pós-processamento determinístico das predições dos baselines | encoder treinado só nos pares com pista de negação |
| pesos próprios | não | sim |
| custo de GPU | 0 h | ~30 min de T4 por execução de 10 épocas (estimado) |
| léxico | `min_freq = 3`, 11 formas, calibrado no DEV | o mesmo |
| estado | 16/16 comparações com IC95 acima de zero contra os baselines, no TEST | análise comparativa pendente |
| referência | `claude/fase2-filtro-lexico-20set-2026.md` | `claude/finetuning-restrito-26set-2026.md`, código em `src/finetuning_restrito/` |

### Relação com o critério de 26/09/2026

O registro de 26/09 dizia que o modelo da Fase 2-B, se batesse os 4 baselines,
viraria candidato a RECLin-PT no lugar do filtro. Bater os baselines no DEV continua
sendo o critério de parada da Fase 2-B, mas deixa de bastar. A partir deste registro,
o modelo precisa superar também o filtro, nos termos abaixo.

## Regra de decisão

**Métrica e conjunto.** F1 da classe `negation_of` no DEV (19.064 pares), com o
remapeamento fixado em 26/09: `y_true` nunca é remapeado e, fora do espaço restrito,
`y_pred = no_relation`. É a única métrica comparável entre os dois sistemas, já que o
macro-F1 remapeado zera `associated_with` por construção.

**Teste estatístico.** Bootstrap pareado com IC95%, mesmo procedimento da Fase 2
(`scripts/run_fase2_significance.py`), aplicado às predições de DEV.

**Execuções que entram.** Sementes 42 e 43 dos dois encoders, as únicas em que o
filtro existe (ele é pós-processamento dos 4 baselines oficiais). Seguindo o protocolo
da Fase 2, entram todas as comparações cruzadas Fase 2-B × filtro (4 × 4 = 16).
Execuções da Fase 2-B em outras sementes, se houver, são reportadas mas não entram na
decisão.

**a.** O sistema escolhido como o RECLin-PT principal é o que superar o outro com
significância estatística, isto é, com IC95% do bootstrap pareado inteiramente do lado
favorável nas 16 comparações, avaliado no conjunto de DESENVOLVIMENTO.

**b.** Se nenhum dos dois superar o outro nesse sentido, há empate, e o filtro de
pista lexical vence por critério de desempate. O motivo é o custo computacional zero
contra o custo de GPU do fine-tuning, pelo princípio de parcimônia já defendido no
Cap. 7 do TCC. Resultado misto (parte das comparações significativa e parte não)
conta como empate.

**c.** Independentemente de qual vencer, os dois sistemas são documentados no TCC. O
vencedor entra como a contribuição principal (RECLin-PT) e o outro como abordagem
alternativa testada e comparada sob o mesmo protocolo estatístico, não como tentativa
descartada.

**d.** Se a Fase 2-B não passar no critério de parada definido em 26/09
(F1_dev(restrito, s) > F1_dev(baseline do mesmo encoder, s) nas sementes 42 e 43), a
comparação com o filtro não é executada e vale (b). A Fase 2-B continua documentada
conforme (c), com o que foi medido.

**e.** Depois da decisão, os dois sistemas são avaliados no TEST e reportados. O
resultado no TEST não reabre a decisão tomada no DEV.

## O que já era conhecido no momento do registro

- F1 de `negation_of` no DEV dos baselines (s42 / s43): BioBERTpt 0,7109 / 0,7224;
  BERTimbau 0,7030 / 0,6904.
- F1 de `negation_of` no DEV do filtro (s42 / s43): filtro(BioBERTpt) 0,8086 / 0,8000;
  filtro(BERTimbau) 0,7778 / 0,7726.
- Da Fase 2-B, nenhum F1 no DEV foi lido, e a comparação pareada com o filtro não foi
  executada.
- Viés de seleção nos dois lados. O `min_freq` do filtro foi escolhido no próprio DEV,
  e a época da Fase 2-B também é selecionada no DEV (por macro-F1 remapeado).

## Fora do escopo deste registro

Nenhuma análise ou script foi executado para produzi-lo. A análise comparativa
(manifest + `analise_ftr.py` sobre o DEV) é tarefa separada, posterior a este commit.
