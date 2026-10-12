# Pendência D4 — `min_freq` do léxico de pistas, só com o TRAIN da v2

Data: 11/10/2026. Trata **só a D4**. Não inicia a etapa 8. Nenhum comando de
controle de versão, nenhum treino, nenhuma calibração baseada em modelo,
nenhuma predição, nenhum resultado histórico, nenhum acesso à rede, nenhuma
instalação. O DEV e o TEST não foram usados na seleção (veja "Uso do DEV" ao
fim, sobre uma verificação feita por engano depois da escolha).

## Resultado

**`min_freq = 10`** — léxico da fase com 8 formas, `lexico_sha1`
`8af0716633e9` (`sem` 662, `nega` 353, `nao` 128, `ausente` 12, `ausencia`
11, `s` 11, `evacuacao` 10, `s/` 10). Saem do léxico anterior `ausentes` (5),
`eliminacao fecal` (4) e `indolor` (3).

Escolhido pela regra pré-registrada e **sujeito às limitações abaixo**, que
recomendo você ler antes de confirmar.

## Método (pré-registrado antes de qualquer cálculo)

`docs/protocolo/d4-min-freq.md` (SHA-256 `933ed702…d927`), gravado e
conferido no computador antes da primeira execução; o script recusa rodar se
as constantes do código diferirem do bloco JSON do pré-registro.

* Dados: só `dados/particoes-v2/train.jsonl` (`b8aaafce…d034`), numa pasta
  temporária com apenas o `train.jsonl` e o `MANIFEST.json` da v2 (o DEV e o
  TEST não existiam nela; a pasta presente foi gravada na proveniência).
* Unidade: pares candidatos (`max_gap = 25`); positivo = `negation_of`.
* Detector: "o `e1` é pista" — o uso do léxico no filtro, no restrito e na
  Pair-Aware.
* 5 partes por documento, estratificadas pela presença de `negation_of`
  (`random.Random(42)`, como `particoes.particionar`); em cada parte, o léxico
  é induzido das outras quatro.
* Candidatos `{1, 2, 3, 5, 10}`; métrica F1 = 2·VP / (2·VP + FP + FN) com as
  contagens somadas nas 5 partes, comparada exatamente (frações); desempate
  pelo maior `min_freq`; o léxico final é o induzido do TRAIN inteiro.

## Resultados (`docs/entregas/d4/selecao_min_freq.json`)

| `min_freq` | VP | FP | FN | Precisão | Cobertura | F1 (exato) | F1 por parte |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.219 | 8.070 | 36 | 0,1312 | 0,9713 | 0,231222 (1219/5272) | 0,242 0,216 0,210 0,251 0,239 |
| 2 | 1.207 | 5.749 | 48 | 0,1735 | 0,9618 | 0,293996 (142/483) | 0,287 0,300 0,296 0,297 0,291 |
| 3 | 1.203 | 5.474 | 52 | 0,1802 | 0,9586 | 0,303328 (401/1322) | 0,301 0,309 0,296 0,309 0,301 |
| 5 | 1.197 | 5.252 | 58 | 0,1856 | 0,9538 | 0,310748 (133/428) | 0,316 0,309 0,296 0,323 0,310 |
| **10** | 1.149 | 4.673 | 106 | 0,1974 | 0,9155 | **0,324714** (766/2359) | 0,337 0,327 0,314 0,335 0,313 |

Partes: 161/161/160/159/159 documentos (107/107/107/106/106 com
`negation_of`); hash da atribuição `055b83f3…9c89`. Tamanho do léxico nas
partes com `min_freq=10`: 6, 4, 5, 4, 4 formas (o final tem 8). O valor 10 tem
o maior F1 no total e em cada uma das 5 partes; o F1 cresce com o `min_freq`
em toda a grade.

## Limitações (verificadas nos resultados; nenhuma muda a regra)

1. **Escolha na borda da grade.** O F1 é crescente em toda a grade
   `{1, 2, 3, 5, 10}`; o valor escolhido é o maior permitido. A grade foi
   fixada no pré-registro e não foi ampliada; não se sabe se um valor maior
   teria F1 maior por este critério.
2. **Critério que favorece a precisão.** Cada pista forma pares com vários
   `e2`, então os FP (milhares) dominam o F1, e um léxico menor quase sempre
   os reduz mais do que perde VP. O critério mede o léxico como detector, não
   o filtro aplicado a um modelo (limitação declarada no pré-registro).
3. **Limiar sem reescala nas partes.** Com 4/5 do TRAIN, formas com
   frequência 10–12 no TRAIN inteiro (`ausente`, `ausencia`, `s`,
   `evacuacao`, `s/`) ficam com cerca de 8–10 em cada parte e, com
   `min_freq=10`, entram em umas partes e saem de outras: o léxico avaliado
   (4 a 6 formas) é menor que o adotado (8). Também declarado no
   pré-registro. Consequência: a validação cruzada estima o desempenho de um
   léxico mais restrito que o final.
4. **Efeito no espaço restrito** (contagens estruturais, sem rótulos): pares
   cujo `e1` é pista passam de 6.805 / 774 / 800 para 6.449 / 737 / 774 no
   TRAIN / DEV / TEST. Cobertura no TRAIN (dentro da amostra): 0,9633 →
   0,9538. Isso reduz o teto de recall de `negation_of` do filtro, do
   restrito e da Pair-Aware.

Pelas regras do pré-registro (item 9), o resultado foi aceito e nenhum valor,
métrica ou divisão alternativa foi testado. Se você não aceitar estas
limitações, a alternativa honesta é um **novo** pré-registro, com outra grade
ou outra métrica, sabendo que ele seria feito depois de conhecer este
resultado (o risco de escolher o procedimento pelo resultado deve ser
declarado). Também é possível voltar à opção A da seção H de
`reconstrucao-v2.md`: manter `min_freq = 3` declarado como herdado do legado.

## Registro aplicado (item 10 do pré-registro)

* `reclin/negacao/lexico.py`: `CONGELADO` = `min_freq` 10, 8 formas,
  `8af0716633e9`, com a origem D4; `CONGELADO_LEGADO` = o registro anterior
  (3, 11, `70c93fa807de`), usado só pela reprodução técnica;
  `registro_do_lexico(lex)` devolve o registro pelo hash.
* O restrito e a Pair-Aware gravam o `min_freq` do registro do léxico que
  recebem (legado → 3; fase → 10). O filtro no modo da fase, `treinar.py`
  (restrito e Pair-Aware) e `analisar.py lexico` usam o `CONGELADO`.
* O modo de reprodução de `filtro_pistas.py` e `regra_pura.py`, e o rótulo de
  `calibrar.py`, usam `CONGELADO_LEGADO`. Eles reproduzem o legado.
* Protocolo da fase: linha "Léxico", registro das alterações, seção 8 e bloco
  JSON (`filtro_pistas.lexico` = 10 / 8 / `8af0716633e9`).
* Testes do legado apontam explicitamente para `CONGELADO_LEGADO`, com as
  mesmas verificações de antes. O teste de integração que fixava o espaço
  restrito do TEST em 800 (léxico do legado) passou a 774 (contagem
  estrutural com o léxico novo).
* A identidade do código (D6) muda, como esperado.

## Proveniência

| Item | Valor |
| --- | --- |
| Pré-registro `docs/protocolo/d4-min-freq.md` | `933ed702a4a98ffde57b0d2ac0f2b08695158692fa81b54ee373cdc872d0d927` |
| TRAIN v2 / MANIFEST v2 | `b8aaafce…d034` / `18264860…3b70` |
| `reclin/negacao/selecao_min_freq.py` | `2283e0fd466c2bde409e025b8ab2bc2a849deb4d5c0ced78d83b80d4b2a883ea` |
| `scripts/selecionar_min_freq.py` | `30df14b9390a09ae666c6ef3e798302a9a1914aa86e583033aa1db146ef4bf88` |
| `reclin/negacao/lexico.py` (na execução, antes do registro) | `634235b6…8535` |
| `reclin/tarefa.py` | `0e3669e0…c483` |
| Resultado `docs/entregas/d4/selecao_min_freq.json` | `0e9bed3e864522f64ac26ce465905c1f82c88ee3ad53368213fa0284e5f534d8` |
| Ambiente | Python 3.10.12 (computador do autor), 77 s |

Comando: `python codigo/scripts/selecionar_min_freq.py --particoes <pasta só
com train.jsonl e MANIFEST.json da v2> --saida
codigo/docs/entregas/d4/selecao_min_freq.json` (`make selecionar-min-freq`
refaz sobre `dados/particoes-v2/`, que também só abre o TRAIN).

## Uso do DEV

Depois da escolha e do registro do resultado, ao medir o impacto no espaço
restrito, calculei também a cobertura de `negation_of` no DEV (usa rótulos
do DEV). Esse número não entrou na escolha, não está registrado aqui e não
deve ser usado para decidir sobre a D4. Os números de impacto acima são
estruturais (sem rótulos) ou do TRAIN.

## Testes executados

Computador: Python 3.10, dados reais, executor mínimo no estilo do pytest (o
pytest não está instalado lá). Nuvem: pytest 9.1.1, sem os dados.

| Conjunto | Onde | Resultado |
| --- | --- | --- |
| `unidade/test_selecao_min_freq.py` (7, sintéticos: partes, contagem, F1 exato e desempate, isolamento da parte retida, reprodutibilidade, código × pré-registro, recusa de pasta que não é da v2) | nuvem e computador | 7 passaram |
| `unidade/test_d4_registro.py` (6: resultado = `CONGELADO`, regra aplicada, proveniência, protocolo, léxico do TRAIN v2, **reprodução completa no TRAIN real**, `lento`) | computador | 6 passaram (na nuvem, 4, sem dados) |
| `unidade/test_negacao.py` (registro da fase e do legado, `registro_do_lexico`) | computador e nuvem | 21 passaram |
| `equivalencia/test_lexico_legado.py` + `unidade/test_estrategias.py` | computador | 44 passaram |
| `equivalencia/test_estrategias_legado.py` (filtro, regra e calibração do legado byte a byte, com `CONGELADO_LEGADO`) | computador | 19 passaram |
| `unidade/test_dados_v2.py::test_lexico_congelado_igual_na_v2` | computador | passou |
| `unidade/test_entradas_e_legado.py` (notebook e Makefile) + `test_versao_dados.py` | nuvem | passaram (no computador, 1 erro do executor: `capsys`) |
| Regra de níveis do pacote (`test_pacote.py`), `--help` dos scripts tocados | nuvem | passaram |
| Mutação: léxico induzido sem excluir a parte retida | nuvem | detectada |

Não executados (exigem torch, ausente nos dois ambientes):
`equivalencia/test_estrategias_treinadas_legado.py`,
`unidade/test_protocolo_fase.py` (o mesmo conteúdo do bloco JSON é conferido
sem torch por `test_d4_registro.py`) e os testes de integração do restrito e
da Pair-Aware com o léxico novo, inclusive o espaço restrito do TEST = 774.

Integridade: os 1.114 arquivos protegidos (`dados/` v1 e v2, `testes/referencia/`,
XML, notas do D9 e da v2) estão com o mesmo SHA-256 de antes desta tarefa.

## Arquivos

Criados: `docs/protocolo/d4-min-freq.md`, `reclin/negacao/selecao_min_freq.py`,
`scripts/selecionar_min_freq.py`, `testes/unidade/test_selecao_min_freq.py`,
`testes/unidade/test_d4_registro.py`, `docs/entregas/d4/selecao_min_freq.json`,
`docs/entregas/d4-min-freq.md`.

Modificados: `reclin/negacao/lexico.py`, `reclin/estrategias/restrito.py`,
`reclin/estrategias/pair_aware/__init__.py`, `scripts/filtro_pistas.py`,
`scripts/regra_pura.py`, `scripts/calibrar.py`,
`docs/protocolo/fase-biobertpt-seed42.md`, `docs/estrategias/README.md`,
`docs/entregas/README.md`, `codigo/README.md`, `codigo/Makefile`,
`notebooks/reclin.ipynb`, `testes/unidade/test_negacao.py`,
`testes/equivalencia/{test_lexico_legado,test_estrategias_legado,test_estrategias_treinadas_legado}.py`,
`testes/integracao/test_estrategias_treinadas.py`.

Removidos: nenhum.
