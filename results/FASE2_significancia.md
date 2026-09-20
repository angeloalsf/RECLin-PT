# Fase 2 sem GPU: filtro de pista e regra pura no TEST

`20/09/2026` · `max_gap = 25` · `n_test = 19210` · `calibracao = {"combined_gap": 25, "lexicon_size": 11, "min_freq": 3, "rule": "R3", "rule_gap": 1}`

Tudo calibrado no DEV (`results/CALIBRACAO_filtro.md`) e so entao aplicado ao
TEST, uma vez por sistema. As linhas de registro estao nos
`results/*.test_evals.jsonl`, com `eval_index_for_config = 1` para cada um dos
cinco sistemas novos.

## Quadro no TEST

| sistema | TP | FP | FN | P | R | **F1 `negation_of`** | macro-F1 |
|---|---|---|---|---|---|---|---|
| BioBERTpt s42 | 142 | 114 | 10 | 0,5547 | 0,9342 | 0,6961 | 0,6932 |
| BERTimbau s42 | 139 | 100 | 13 | 0,5816 | 0,9145 | 0,7110 | 0,6977 |
| BioBERTpt s43 | 140 | 87 | 12 | 0,6167 | 0,9211 | 0,7388 | 0,7128 |
| BERTimbau s43 | 139 | 96 | 13 | 0,5915 | 0,9145 | 0,7183 | 0,7035 |
| filtro(BioBERTpt s42) | 139 | 44 | 13 | 0,7596 | 0,9145 | **0,8299** | 0,7383 |
| filtro(BERTimbau s42) | 136 | 45 | 16 | 0,7514 | 0,8947 | **0,8168** | 0,7334 |
| filtro(BioBERTpt s43) | 137 | 33 | 15 | 0,8059 | 0,9013 | **0,8509** | 0,7506 |
| filtro(BERTimbau s43) | 136 | 41 | 16 | 0,7684 | 0,8947 | **0,8267** | 0,7401 |
| regra pura R3 | 117 | 68 | 35 | 0,6324 | 0,7697 | **0,6944** | 0,5549 |

O filtro so rebaixa predicoes `negation_of` para `no_relation`, entao o F1 de
`associated_with` fica inalterado em todos os digitos nas quatro execucoes. Foi
conferido, e o macro-F1 sobe por conta das outras duas classes.

## As 24 comparacoes

Bootstrap pareado, 10.000 reamostragens, IC95 por percentis 2,5/97,5. A semente
do bootstrap segue a regra do Makefile, estendida ao caso cruzado: e a semente do
lado A, e quando A nao tem semente e a do lado B.

O McNemar aparece so como contexto. Entre um sistema e sua propria versao
filtrada ele e degenerado por construcao, porque a regra so muda predicoes numa
direcao. O criterio de decisao aqui e o IC95 do bootstrap.

| A | B | F1 A | F1 B | diferenca | IC95 | IC95 exclui zero? |
|---|---|---|---|---|---|---|
| filtro(BioBERTpt s42) | BioBERTpt s42 | 0,8299 | 0,6961 | +0,1338 | [+0,0998; +0,1700] | **sim** |
| filtro(BioBERTpt s42) | BERTimbau s42 | 0,8299 | 0,7110 | +0,1189 | [+0,0789; +0,1613] | **sim** |
| filtro(BioBERTpt s42) | BioBERTpt s43 | 0,8299 | 0,7388 | +0,0911 | [+0,0473; +0,1356] | **sim** |
| filtro(BioBERTpt s42) | BERTimbau s43 | 0,8299 | 0,7183 | +0,1115 | [+0,0691; +0,1562] | **sim** |
| filtro(BioBERTpt s42) | regra pura R3 | 0,8299 | 0,6944 | +0,1355 | [+0,0863; +0,1882] | **sim** |
| filtro(BERTimbau s42) | BioBERTpt s42 | 0,8168 | 0,6961 | +0,1207 | [+0,0780; +0,1647] | **sim** |
| filtro(BERTimbau s42) | BERTimbau s42 | 0,8168 | 0,7110 | +0,1058 | [+0,0749; +0,1399] | **sim** |
| filtro(BERTimbau s42) | BioBERTpt s43 | 0,8168 | 0,7388 | +0,0780 | [+0,0340; +0,1225] | **sim** |
| filtro(BERTimbau s42) | BERTimbau s43 | 0,8168 | 0,7183 | +0,0985 | [+0,0596; +0,1392] | **sim** |
| filtro(BERTimbau s42) | regra pura R3 | 0,8168 | 0,6944 | +0,1225 | [+0,0745; +0,1731] | **sim** |
| filtro(BioBERTpt s43) | BioBERTpt s42 | 0,8509 | 0,6961 | +0,1549 | [+0,1110; +0,2009] | **sim** |
| filtro(BioBERTpt s43) | BERTimbau s42 | 0,8509 | 0,7110 | +0,1399 | [+0,0969; +0,1843] | **sim** |
| filtro(BioBERTpt s43) | BioBERTpt s43 | 0,8509 | 0,7388 | +0,1121 | [+0,0788; +0,1476] | **sim** |
| filtro(BioBERTpt s43) | BERTimbau s43 | 0,8509 | 0,7183 | +0,1326 | [+0,0874; +0,1781] | **sim** |
| filtro(BioBERTpt s43) | regra pura R3 | 0,8509 | 0,6944 | +0,1566 | [+0,1065; +0,2092] | **sim** |
| filtro(BERTimbau s43) | BioBERTpt s42 | 0,8267 | 0,6961 | +0,1307 | [+0,0859; +0,1762] | **sim** |
| filtro(BERTimbau s43) | BERTimbau s42 | 0,8267 | 0,7110 | +0,1158 | [+0,0767; +0,1556] | **sim** |
| filtro(BERTimbau s43) | BioBERTpt s43 | 0,8267 | 0,7388 | +0,0880 | [+0,0430; +0,1348] | **sim** |
| filtro(BERTimbau s43) | BERTimbau s43 | 0,8267 | 0,7183 | +0,1084 | [+0,0760; +0,1425] | **sim** |
| filtro(BERTimbau s43) | regra pura R3 | 0,8267 | 0,6944 | +0,1324 | [+0,0820; +0,1842] | **sim** |
| regra pura R3 | BioBERTpt s42 | 0,6944 | 0,6961 | -0,0017 | [-0,0588; +0,0540] | nao |
| regra pura R3 | BERTimbau s42 | 0,6944 | 0,7110 | -0,0166 | [-0,0716; +0,0386] | nao |
| regra pura R3 | BioBERTpt s43 | 0,6944 | 0,7388 | -0,0444 | [-0,1022; +0,0145] | nao |
| regra pura R3 | BERTimbau s43 | 0,6944 | 0,7183 | -0,0240 | [-0,0818; +0,0334] | nao |

## Veredito

As 16 comparacoes de cada execucao filtrada contra cada
baseline oficial dao IC95 acima de zero em 16 delas.
Nenhuma falha, nas duas sementes e nos dois encoders. O pior caso possivel e a
execucao filtrada mais fraca, filtro(BERTimbau s42) = 0,8168, contra o
baseline mais forte, BioBERTpt s43 = 0,7388, e mesmo ele da
+0,0780 com IC95 [+0,0340; +0,1225].

**A meta da fase 2 esta batida sem fine-tuning.**

A regra pura sozinha nao supera baseline nenhum. Ela fica em 0,6944, e contra os quatro o IC95 contem zero, ou seja, ela
empata estatisticamente com os quatro baselines de 11,4 h de GPU. Esse e o
numero que protege o Cap. 6, e ele corta nos dois sentidos: a regra e um piso
alto, e o que o modelo contextual acrescenta so aparece depois que a pista
restringe o espaco.

