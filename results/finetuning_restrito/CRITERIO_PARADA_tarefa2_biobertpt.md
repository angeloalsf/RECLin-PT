# Criterio de parada da Tarefa 2 (biobertpt)

`2026-10-02T11:23:30+0000` · DEV remapeado, 19.064 pares, 150 `negation_of` no gold · teto de recall do espaco restrito 0,9467 (774 pares restritos). O TEST nao foi lido.

**Regra fixada antes de rodar:** passa se o F1 de `negation_of` do restrito superar o do baseline do mesmo encoder e da mesma semente, em todas as sementes.

| semente | restrito F1 (P / R) | baseline F1 (P / R) | Δ | IC95 bootstrap dev | filtro fase 2 no dev | melhor época |
|---|---|---|---|---|---|---|
| 42 | 0,7029 (0,6150 / 0,8200) | 0,7109 (0,5903 / 0,8933) | -0,0080 | [-0,0534; 0,0385] | 0,8086 | 2 de 3 |

**Veredito: NAO PASSA: o restrito nao supera o baseline no DEV em todas as sementes. PARAR aqui; este e o veredito da frente.**

Δ médio vs baseline: -0,0080. GPU gasta na sanidade: 0,13 h.

## Diagnóstico (não entra na regra)

- semente 42: 0 `negation_of` do dev preditos como `associated_with` dentro do espaço restrito; melhor época é a última: não; F1 por época [0.6211, 0.7029, 0.7008]; 4 aviso(s) de determinismo capturado(s).

O IC do bootstrap no dev e o filtro são contexto. A decisão usa só a comparação pontual acima, como fixado na tarefa.
