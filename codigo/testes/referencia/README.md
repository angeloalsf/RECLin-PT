# Referências do legado

Arquivos gerados a partir do projeto anterior (o "legado": repositório
original, https://github.com/angeloalsf/RECLin-PT, commit `a5f055c`) para
validar a nova implementação. Os testes de equivalência do código novo comparam
o que ele produz com estes arquivos, e por isso rodam sem o legado. **Não edite
à mão**: para regerar, veja "Como regerar" abaixo.

| Arquivo | Conteúdo | Valida as etapas |
| --- | --- | --- |
| `dados.json` | Por partição: documentos, candidatos (`max_gap=25`), contagem por rótulo e SHA-256 dos candidatos, das janelas marcadas (`ctx_chars=128`) e do `y_true`, conferido contra os sidecars oficiais. Léxico de pistas por `min_freq` (1, 2, 3, 5, 10), léxico congelado (`min_freq=3`, `lexico_sha1`), índices do espaço restrito e pesos `balanced` dos dois espaços. MANIFEST das partições do legado e configuração registrada dos quatro baselines (`config` do `.json` e `config_sha1` da trilha `test_evals`, que também cobre `weight_decay`, `warmup_ratio` e `splits_dir`) | 1, 3, 5, 6 |
| `modelo_minusculo/` | BERT aleatório (16 dimensões, 2 camadas) e tokenizer de vocabulário sintético, sem os marcadores. Entrada comum dos treinos de referência | 5, 6 |
| `treino/` | Sidecars (`.json`, `.preds.json`, `.dev_preds.json`) do legado treinando o modelo minúsculo em CPU, sem retomada: baseline (2 épocas), fine-tuning restrito (10 épocas) e Pair-Aware (3 épocas), semente 42, `lr=1e-3` | 5, 6 |
| `resultados_legado/` | Cópia sem alteração de 74 arquivos de `results/` do legado, na mesma estrutura de pastas: os quatro baselines (`.json` com as métricas, `.preds.json` do TEST, `.dev_preds.json` do DEV, `.test_evals.jsonl`), os sidecars dos quatro filtros e da regra pura com `FASE2_test_summary.json` e a trilha da regra, a calibração do filtro que congelou o léxico (`CALIBRACAO_filtro.json`) e o seu relatório com a varredura completa (`CALIBRACAO_filtro.md`), as 26 comparações `significance_*.json`, `summary_by_seed.json`, as cinco execuções do fine-tuning restrito e a Pair-Aware (só DEV). Ficam de fora os `archive_*`, as comparações de DEV dos critérios de parada e os `filtro_dev/` | 2, 3, 4 |
| `referencias.json` | Ambiente, comandos, SHA-256 de cada arquivo, checagem de determinismo e resultado dos dois testes de CPU do legado | — |
| `gerar_referencias.py` | O gerador: a única ponte entre o código novo e o legado | — |

Nos registros, `legado/` designa a raiz do projeto legado (onde ficam `src/`,
`data/` e `results/`), e caminhos como `results/baseline_biobertpt_seed42.preds.json`
são relativos a ela.

## Como foram gerados

- Python 3.13 com as versões fixadas em `codigo/requirements.txt`, as mesmas do
  legado (torch 2.11.0, transformers 5.16.1, scikit-learn 1.6.1, numpy 2.1.3,
  scipy 1.16.3). As execuções do legado registradas em `results/` também usaram
  Python 3.13 (campo `environment` de cada uma), no Colab.
- Treinos em CPU com `PYTHONHASHSEED=0` e **uma thread** (`OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`): o resultado de multiplicações em CPU pode mudar nos
  últimos bits com o número de threads.
- Cada treino rodou **duas vezes**; as predições do DEV e do TEST saíram
  idênticas byte a byte nas duas rodadas (registrado em `referencias.json`).
- Os treinos usam `lr=1e-3`, e não o `2e-5` dos experimentos: com o `lr`
  original o modelo minúsculo prevê só `no_relation`, e a referência não
  exercitaria nem o argmax nem a escolha da melhor época. Isso não muda o que é
  testado — o código — só torna o teste mais sensível.

## Como regerar

Só é preciso regerar se o legado ganhar uma referência nova. Num ambiente com
`codigo/requirements.txt` instalado:

```bash
git clone https://github.com/angeloalsf/RECLin-PT RECLin-PT-legado
git -C RECLin-PT-legado checkout a5f055c
python codigo/testes/referencia/gerar_referencias.py --legado RECLin-PT-legado
# ou só uma parte: --partes dados | modelo | treino | testes | resultados
```

O script confere as partições do legado contra o MANIFEST dele e o `y_true`
contra os sidecars oficiais, e falha se algo divergir ou se um treino não for
determinístico. Os dois testes de CPU do legado regravam um arquivo dentro de
`results/`; o script devolve o conteúdo original ao terminar.

## Como usar nos testes de equivalência

- Os testes do código novo devem rodar nas mesmas condições: CPU, uma thread,
  `PYTHONHASHSEED=0`, modelo de `modelo_minusculo/` e os mesmos argumentos
  registrados em `referencias.json`.
- Comparar `y_true`, `y_pred` e `probs` dos sidecars e, nos `.json`, o
  `dev_history` (`train_loss`, `dev_loss`, F1 por época) e a melhor época. Não
  comparar os `.json` inteiros: eles trazem durações e o ambiente da execução.
  O campo `model` traz o caminho absoluto do modelo na máquina onde as
  referências foram geradas e também fica fora da comparação.
- Os hashes de `dados.json` usam `json.dumps(..., ensure_ascii=False,
  separators=(",", ":"))` sobre as listas, na ordem de geração dos candidatos.
