# Criterio de parada da Tarefa 2 (biobertpt)

`2026-10-02T13:34:31+0000` · DEV remapeado, 19.064 pares, 150 `negation_of` no gold · teto de recall do espaco restrito 0,9467 (774 pares restritos). O TEST nao foi lido.

**Regra fixada antes de rodar:** passa se o F1 de `negation_of` do restrito superar o do baseline do mesmo encoder e da mesma semente, em todas as sementes.

| semente | restrito F1 (P / R) | baseline F1 (P / R) | Δ | IC95 bootstrap dev | filtro fase 2 no dev | melhor época |
|---|---|---|---|---|---|---|
| 42 | 0,7792 (0,7595 / 0,8000) | 0,7109 (0,5903 / 0,8933) | +0,0683 | [0,0198; 0,1193] | 0,8086 | 9 de 10 |
| 43 | 0,7837 (0,7396 / 0,8333) | 0,7224 (0,6063 / 0,8933) | +0,0613 | [0,0165; 0,1069] | 0,8000 | 10 de 10 |

**Veredito: PASSA: o restrito supera o baseline no DEV em todas as sementes. Prosseguir para a Tarefa 3.**

Δ médio vs baseline: 0,0648. GPU gasta na sanidade: 0,86 h.

## Diagnóstico (não entra na regra)

- semente 42: 2 `negation_of` do dev preditos como `associated_with` dentro do espaço restrito; melhor época é a última: não; F1 por época [0.2353, 0.6791, 0.6807, 0.6959, 0.7593, 0.7516, 0.7485, 0.7664, 0.7792, 0.7752]; 4 aviso(s) de determinismo capturado(s).
- semente 43: 0 `negation_of` do dev preditos como `associated_with` dentro do espaço restrito; melhor época é a última: sim; F1 por época [0.5273, 0.6701, 0.741, 0.7365, 0.7515, 0.7553, 0.7562, 0.7516, 0.7802, 0.7837]; 4 aviso(s) de determinismo capturado(s).

O IC do bootstrap no dev e o filtro são contexto. A decisão usa só a comparação pontual acima, como fixado na tarefa.
