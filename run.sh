#!/usr/bin/env bash
# Pipeline minimo RECLin-PT, fim a fim.
# Compara BioBERTpt (clinico) x BERTimbau (geral) na MESMA tarefa, mudando so o
# modelo -> responde "o pre-treinamento clinico importa para extracao de
# relacoes em textos medicos em portugues?".
set -e
cd "$(dirname "$0")"

# 1) Parse SemClinBr XML -> dataset.jsonl
python src/parse_semclinbr.py --xml-dir SemClinBr-xml-public-v1 \
    --out data/processed/dataset.jsonl

# 2) Splits 80/10/10 (doc-level, seed 42) + data/splits/MANIFEST.json (SHA-256)
#    ATENCAO: sobrescreve os splits versionados; o MANIFEST torna isso visivel
#    no git diff. Para so recalcular os hashes: --manifest-only.
python src/make_splits.py --input data/processed/dataset.jsonl \
    --out-dir data/splits

# 3a) Baseline CLINICO: BioBERTpt (precisa baixar o modelo; GPU recomendada)
#     Hiperparametros do artigo -- NAO sao os defaults do argparse.
python src/baseline_biobertpt.py --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
    --ckpt-dir checkpoints/biobertpt_seed42 \
    --out results/baseline_biobertpt_seed42.json

# 3b) Baseline GERAL: BERTimbau -- MESMOS hiperparametros (paridade total)
python src/baseline_bertimbau.py --splits-dir data/splits \
    --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
    --ckpt-dir checkpoints/bertimbau_seed42 \
    --out results/baseline_bertimbau_seed42.json

# 4) Significancia (teste pareado, usa os *.preds.json gerados em 3a/3b)
python src/significance.py \
    --a results/baseline_biobertpt_seed42.preds.json \
    --b results/baseline_bertimbau_seed42.preds.json \
    --target negation_of --n-boot 10000 --seed 42 \
    --out results/significance_biobertpt_vs_bertimbau_seed42.json

# 5) Artefatos do artigo: tabelas (.tex) e figuras (.pdf) em artigo-sbc/,
#    derivados de results/. Fonte unica -- nao edite os arquivos gerados.
python scripts/make_tables.py
python scripts/make_figures.py

# 6) Agregacao multi-seed -> results/summary_by_seed.json.
#    So roda se as duas seeds existirem; com uma seed so, pule este passo.
python scripts/aggregate_seeds.py || echo "aggregate_seeds: pulado (falta alguma seed)"
