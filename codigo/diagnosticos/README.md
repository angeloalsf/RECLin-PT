# Diagnósticos dos dados e controles informados pelas anotações

Área experimental **separada** das quatro configurações do RECLin-PT:

| Configuração | Papel |
| --- | --- |
| Baseline | referência experimental |
| Pair-Aware linear | arquitetura oficial |
| Two-Stage | estratégia experimental |
| Pair-Aware MLP | variante experimental |

Nada aqui é sistema do RECLin-PT, entra nas comparações do protocolo ou muda dados, partições, anotações, predições ou modelos. Nada treina, avalia o TEST ou escolhe parâmetros.

## O que há aqui

| Subcomando | O que mede | Partições lidas | Módulo |
| --- | --- | --- | --- |
| `tipos` | **Controles informados pelas anotações**: (1) tipo `Negation` como identificador de pista (entidade e par); (2) tipo + posição: pista → entidade seguinte | TRAIN, DEV (o TEST é recusado) | `reclin/analises/controles_anotacao.py` |
| `duplicatas` | Documentos quase duplicados (Jaccard de *shingles* do texto sem espaços); só registra | as três (só texto, para comparar) | `reclin/analises/quase_duplicatas.py` |
| `semente` | A divisão 800/100/100 com 200 sementes: contagens por partição e o percentil da semente 42 | as três (só contagens) | `reclin/analises/sensibilidade_semente.py` |
| `max-gap` | Candidatos e cobertura por classe para vários limites; distribuição das distâncias anotadas; reprodução do "11,6%" | TRAIN, DEV; corpus inteiro só no bloco `origem_11_6` (contagem) | `reclin/analises/max_gap.py` |
| `excluir-documentos` | Métricas de um sidecar de predições já gravado, com e sem 9534, 9535 e 9536 | a do sidecar; **TEST só com `--test-autorizado`** | `reclin/analises/exclusao_documentos.py` |

Os controles de tipo são **experimentos informados pelas anotações**. Eles usam o tipo semântico que o SemClinBr dá às pistas, um atributo que as quatro configurações **não** recebem: os marcadores são sem tipo (`reclin/entrada.py`). Servem para medir o tamanho do atalho que o tipo daria, não como referência de desempenho.

## Como rodar

A partir da raiz do repositório:

```bash
python codigo/diagnosticos/diagnosticar.py tudo          # tipos, duplicatas, semente e max-gap -> codigo/resultados/diagnosticos/
python codigo/diagnosticos/diagnosticar.py tipos --saida controles.json
python codigo/diagnosticos/diagnosticar.py excluir-documentos --predicoes <execucao>/predicoes_dev.json --particao dev
```

Antes de ler as partições, cada subcomando confere os arquivos contra o `MANIFEST.json`. A saída registra os SHA-256 das partições lidas. As saídas são determinísticas: sem data, chaves ordenadas.

## O que as saídas contêm

Só números agregados, ids de documentos e parâmetros. Nenhum texto clínico, nem trechos das entidades. O teste `testes/unidade/test_diagnosticos.py::test_quase_duplicatas_reais` confere que nenhum trecho de 30 caracteres de nenhum documento aparece na saída do detector.

## Por que fora de `codigo/scripts/`

`scripts/*.py` entra na identidade do código das execuções (`reclin/util/identidade_codigo.py`). Um script novo ali mudaria a identidade e impediria retomar uma execução interrompida. Os módulos ficam em `reclin/analises/`, excluído da identidade, como as demais análises. Com esta área, a identidade continua a mesma de antes: `0526d9123ecc…`, 71 arquivos.

## Auditoria das execuções (Etapa 3)

`auditar_execucoes.py` lê as pastas de `resultados/execucoes/` da branch `resultados-colab` (e, se informadas, as execuções derivadas só do DEV), sem treinar, rodar modelo ou ler o TEST:

| Saída (`resultados/auditoria_execucoes/`) | Conteúdo |
| --- | --- |
| `inventario.json` | Por pasta: tipo, estratégia, encoder e revisão dos pesos, semente, hashes das partições, identidade do código e commit, épocas, sessões, `test_avaliado`; vínculos composição → estágios do Two-Stage pelos hashes; arquivos que diferem entre as identidades de código |
| `metricas_dev.json` | P, R, F1, suporte, TP, FP, FN por classe, macro-F1, MCC e matriz de confusão recalculados das predições, com a conferência contra `metricas.json` e os hashes do DEV; AP por classe quando há probabilidades |
| `comparacoes_dev.json` | Comparações pareadas descritivas: F1 de `negation_of`, F1 de `associated_with` e macro-F1, bootstrap por candidato (o do protocolo) e por documento, McNemar exato |
| `two_stage_tau.json` | Grade de τ de cada composição, com precisão e recall de `associated_with` |
| `associated_with_dev.json` | Origem dos falsos positivos de `associated_with` (direção invertida, sem relação, `negation_of`) e precisão por distância; exige `--particoes` |

```bash
python codigo/diagnosticos/auditar_execucoes.py \
    --execucoes <checkout de resultados-colab>/codigo/resultados/execucoes \
    --derivadas tcc/analise/execucoes_derivadas_dev/baseline_calibrado_biobertpt_seed42 \
    --derivadas tcc/analise/execucoes_derivadas_dev/filtro_biobertpt_seed42 \
    --particoes codigo/dados/particoes-v2 --origem resultados-colab@<commit>
python tcc/analise/gerar_tabelas_etapa3.py . tcc/tcc/src     # tabelas e figura do TCC
```

Módulo: `reclin/analises/auditoria_execucoes.py` (fora da identidade do código). Testes: `testes/unidade/test_auditoria_execucoes.py`. Nota: `docs/entregas/etapa3-auditoria-experimental.md`.

Resultados e interpretação:
- `docs/analises/controles-tipo-semantico.md`;
- `docs/analises/janela-max-gap.md`;
- `docs/analises/integridade-particoes.md`.
