# Calibracao do filtro de pista e da regra pura — DEV

`20/09/2026` · `max_gap = 25` · `split = dev` · **o TEST nao foi lido por este script**

Tudo aqui sai dos quatro `<out>.dev_preds.json` das execucoes oficiais de
17-20/09/2026, que guardam as predicoes do DEV feitas com os pesos da melhor
epoca. O espaco de candidatos do DEV foi reconstruido a partir de
`src/candidates.py` e conferido contra o `y_true` dos quatro sidecars, com
zero divergencias. Sao 19.064 candidatos e 150 pares `negation_of`
no gold.

## Criterio, fixado antes de rodar

1. `min_freq` do lexico maximiza a **media** do F1 de `negation_of` no DEV sobre
   as quatro execucoes. Empate resolve pelo menor `min_freq`, que e o lexico de
   maior cobertura e portanto o teto de recall mais alto.
2. A regra pura escolhe variante e limiar de gap pelo F1 de `negation_of` no DEV.
   Empate resolve pela regra de menor indice, depois pelo menor limiar.
3. O sistema combinado usa o mesmo criterio do item 1.

Nenhuma dessas escolhas foi revisitada depois de o TEST ser tocado.

## 1. Varredura de `min_freq`

Cobertura e a fracao dos pares `negation_of` do DEV cujo e1 pertence ao lexico.
Ela e o teto de recall que o filtro impoe por construcao.

| `min_freq` | formas | cobertura no dev | BioBERTpt s42 | BERTimbau s42 | BioBERTpt s43 | BERTimbau s43 | media |
|---|---|---|---|---|---|---|---|
| 1 | 51 | 0,9667 | 0,7939 | 0,7651 | 0,7834 | 0,7538 | 0,7740 |
| 2 | 17 | 0,9667 | 0,8037 | 0,7744 | 0,7952 | 0,7678 | 0,7853 |
| **3** | 11 | 0,9467 | 0,8086 | 0,7778 | 0,8000 | 0,7726 | **0,7898** |
| 5 | 9 | 0,9133 | 0,7925 | 0,7687 | 0,7988 | 0,7712 | 0,7828 |
| 10 | 8 | 0,9133 | 0,7950 | 0,7736 | 0,8012 | 0,7736 | 0,7858 |

**Escolhido: `min_freq = 3`**, com 11 formas e cobertura
0,9467 no DEV.

Vale registrar onde o maximo caiu. A analise exploratoria de 09/09, feita
diretamente no TEST, encontrou ganho monotonico com lexicos cada vez mais
restritivos, com o melhor valor em `min_freq = 10`. No DEV o maximo e
interior, em `min_freq = 3`, porque a cobertura comeca a cair antes de o
ganho de precisao compensar. A escolha honesta portanto nao e a que o TEST
premiaria, e a diferenca entre as duas e justamente o que a disciplina de
calibracao custa.

Lexico resultante, com a frequencia no train:

```
   662  sem
   353  nega
   128  nao
    12  ausente
    11  ausencia
    11  s
    10  evacuacao
    10  s/
     5  ausentes
     4  eliminacao fecal
     3  indolor
```

## 2. Regra pura (R1-R4)

Lexico fixado em `min_freq = 3`. `gap` e `candidates.entity_gap`.
R1 e R2 nao tem limiar de gap, entao aparecem uma vez, com a janela do espaco
de candidatos.

| regra | `gap <=` | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| R1 | 25 | 142 | 632 | 8 | 0,1835 | 0,9467 | 0,3074 |
| R2 | 25 | 93 | 97 | 57 | 0,4895 | 0,6200 | 0,5471 |
| R3 | 0 | 0 | 0 | 150 | 0,0000 | 0,0000 | 0,0000 |
| **R3** | 1 | 112 | 61 | 38 | 0,6474 | 0,7467 | **0,6935** |
| R3 | 2 | 112 | 146 | 38 | 0,4341 | 0,7467 | 0,5490 |
| R3 | 3 | 116 | 177 | 34 | 0,3959 | 0,7733 | 0,5237 |
| R3 | 5 | 118 | 201 | 32 | 0,3699 | 0,7867 | 0,5032 |
| R3 | 10 | 124 | 294 | 26 | 0,2967 | 0,8267 | 0,4366 |
| R3 | 25 | 142 | 632 | 8 | 0,1835 | 0,9467 | 0,3074 |
| R4 | 0 | 0 | 0 | 150 | 0,0000 | 0,0000 | 0,0000 |
| R4 | 1 | 93 | 57 | 57 | 0,6200 | 0,6200 | 0,6200 |
| R4 | 2 | 93 | 82 | 57 | 0,5314 | 0,6200 | 0,5723 |
| R4 | 3 | 93 | 91 | 57 | 0,5054 | 0,6200 | 0,5569 |
| R4 | 5 | 93 | 94 | 57 | 0,4973 | 0,6200 | 0,5519 |
| R4 | 10 | 93 | 95 | 57 | 0,4947 | 0,6200 | 0,5503 |
| R4 | 25 | 93 | 97 | 57 | 0,4895 | 0,6200 | 0,5471 |

**Escolhida: R3 com `gap <= 1`**, F1 = 0,6935 no DEV.

## 3. Sistema combinado: filtro + porta de gap

O filtro de pista sozinho nao usa distancia. Esta secao mede o que acontece ao
aplicar tambem a condicao de gap de R3 sobre as predicoes ja filtradas, que e a
leitura literal de combinar o filtro com a regra pura. `gap <= 25` equivale a
desligar a porta, porque nenhum candidato excede a janela.

| `gap <=` | BioBERTpt s42 | BERTimbau s42 | BioBERTpt s43 | BERTimbau s43 | media |
|---|---|---|---|---|---|
| 0 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| 1 | 0,7774 | 0,7616 | 0,7832 | 0,7643 | 0,7716 |
| 2 | 0,7774 | 0,7616 | 0,7832 | 0,7643 | 0,7716 |
| 3 | 0,7917 | 0,7649 | 0,7805 | 0,7616 | 0,7747 |
| 5 | 0,7945 | 0,7639 | 0,7793 | 0,7606 | 0,7746 |
| 10 | 0,7947 | 0,7667 | 0,7841 | 0,7713 | 0,7792 |
| **25** | 0,8086 | 0,7778 | 0,8000 | 0,7726 | **0,7898** |

**Escolhido: `gap <= 25`.**
O limiar vencedor desliga a porta, ou seja, o DEV nao sustenta somar a
condicao de distancia ao filtro. O sistema combinado colapsa no filtro
sozinho e nao sera avaliado como um sistema separado no TEST.

## Configuracao final levada ao TEST

```json
{
  "combined_gap": 25,
  "lexicon_size": 11,
  "min_freq": 3,
  "rule": "R3",
  "rule_gap": 1
}
```

