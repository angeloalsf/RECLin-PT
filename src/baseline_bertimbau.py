#!/usr/bin/env python3
"""
Baseline BERTimbau (encoder GERAL em portugues) para extracao de relacao.

Entry-point fino: toda a logica vive em `src/relation_extraction.py` (compartilhado com o
baseline BioBERTpt). Aqui so escolhemos o modelo e os caminhos de saida, para
que a UNICA diferenca entre os dois experimentos seja o checkpoint de pre-treino
-- a condicao necessaria para responder, de forma valida:

    "O pre-treinamento clinico importa para extracao de relacoes em textos
     medicos em portugues?"

BERTimbau (`neuralmind/bert-base-portuguese-cased`) e o lado GERAL da comparacao
(pre-treinado no brWaC, corpus web/geral); BioBERTpt (`pucpr/biobertpt-all`) e o
lado CLINICO. Mesma representacao de entrada, loss, scheduler, early stopping,
salvamento de checkpoints, metricas, seed e logging -- garantidos por usarem o
mesmo nucleo.

Uso (CPU funciona, mas e lento; ideal GPU/Colab):
    python src/baseline_bertimbau.py --splits-dir data/splits \
        --epochs 3 --batch-size 64 --max-gap 20 --max-length 128 --seed 42 \
        --ckpt-dir checkpoints/bertimbau_seed42 \
        --out results/baseline_bertimbau_seed42.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relation_extraction import build_arg_parser, run  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("baseline_bertimbau")

DEFAULT_MODEL = "neuralmind/bert-base-portuguese-cased"
# Sem default fixo no argparse: o caminho so pode ser montado DEPOIS do
# parse_args(), quando a seed real e conhecida -- assim o nome do arquivo nunca
# mente sobre a seed que o gerou (convencao `_seed<N>` do README).
OUT_TEMPLATE = "results/baseline_bertimbau_seed{seed}.json"


def main() -> int:
    ap = build_arg_parser(default_model=DEFAULT_MODEL, default_out=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = OUT_TEMPLATE.format(seed=args.seed)
    return run(args, log)


if __name__ == "__main__":
    raise SystemExit(main())
