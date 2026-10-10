#!/usr/bin/env python3
"""Avalia o TEST de uma execução de treino concluída — operação separada do treino.

    python codigo/scripts/avaliar_test.py --execucao codigo/resultados/execucoes/<nome>

Recarrega o `melhor_modelo/` da execução (a época escolhida pelo DEV), prevê o
TEST e grava `predicoes_test.json`, as métricas do TEST em `metricas.json` e
uma linha em `avaliacoes_test.jsonl` (trilha das avaliações do TEST, com o
`eval_index_for_config`). Recusa uma execução cujo treino não terminou.

O TEST de uma execução é avaliado uma vez. Avaliar de novo exige
`--reavaliar` e fica registrado na trilha como uma segunda avaliação da mesma
configuração, que precisa ser declarada.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin.util import caminhos
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.avaliar_test")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--execucao", type=Path, required=True, help="pasta da execução de treino")
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--reavaliar", action="store_true",
                    help="avalia de novo um TEST já avaliado (registrado na trilha)")
    ap.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--threads", type=int, default=None, help="threads do torch em CPU")
    args = ap.parse_args(argv)
    configurar()

    import torch
    from reclin.treino import classificador, reprodutibilidade
    reprodutibilidade.configurar_ambiente()
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        resultado = classificador.avaliar_test(
            args.execucao, pasta_particoes=args.particoes, reavaliar=args.reavaliar,
            dispositivo=None if args.dispositivo == "auto" else args.dispositivo)
    except classificador.ErroExecucao as erro:
        log.error("%s", erro)
        return 1
    if resultado["trilha"]["eval_index_for_config"] > 1:
        log.warning("Esta é a avaliação %d do TEST para esta configuração: declare a multiplicidade.",
                    resultado["trilha"]["eval_index_for_config"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
