#!/usr/bin/env python3
"""Avalia o TEST de uma execução de treino concluída — operação separada do treino.

    python codigo/scripts/avaliar_test.py --execucao codigo/resultados/execucoes/<nome>

Recarrega o `melhor_modelo/` da execução (a época escolhida pelo DEV), prevê o
TEST com a mesma estratégia que a treinou (lida de `config.json`: baseline,
restrito, Pair-Aware ou o classificador da etapa 5; no restrito e na
Pair-Aware, a seleção dos pares sai do léxico gravado na execução, sem ler o
TRAIN) e grava `predicoes_test.json` (no conjunto completo), as métricas do TEST em `metricas.json` e
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

from reclin.estrategias import baseline, pair_aware, restrito
from reclin.util import caminhos
from reclin.util.io import ler_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.avaliar_test")

ESTRATEGIAS = {m.NOME: m for m in (baseline, restrito, pair_aware)}


def montagem_da_execucao(pasta: Path):
    """A montagem da estratégia que treinou a execução, a partir de config.json."""
    from reclin.treino.montagem import CLASSIFICADOR
    registro = ler_json(pasta / "config.json")["config"]
    nome = registro.get("estrategia", CLASSIFICADOR.estrategia)
    if nome == CLASSIFICADOR.estrategia:
        return CLASSIFICADOR
    if nome not in ESTRATEGIAS:
        raise ValueError(f"estratégia {nome!r} desconhecida em {pasta / 'config.json'}")
    return ESTRATEGIAS[nome].montagem_de_registro(registro)


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
        if not (args.execucao / "config.json").is_file():
            raise classificador.ErroExecucao(f"{args.execucao} não é uma execução de treino (sem config.json)")
        resultado = classificador.avaliar_test(
            args.execucao, pasta_particoes=args.particoes, reavaliar=args.reavaliar,
            montagem=montagem_da_execucao(args.execucao),
            dispositivo=None if args.dispositivo == "auto" else args.dispositivo)
    except (classificador.ErroExecucao, ValueError) as erro:
        log.error("%s", erro)
        return 1
    if resultado["trilha"]["eval_index_for_config"] > 1:
        log.warning("Esta é a avaliação %d do TEST para esta configuração: declare a multiplicidade.",
                    resultado["trilha"]["eval_index_for_config"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
