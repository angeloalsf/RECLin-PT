#!/usr/bin/env python3
"""Avalia o TEST de uma execução de treino concluída — operação separada do treino.

    python codigo/scripts/avaliar_test.py --execucao codigo/resultados/execucoes/<nome>

Recarrega o `melhor_modelo/` da execução (a época escolhida pelo DEV), prevê o
TEST com a mesma estratégia que a treinou (lida de `config.json`: baseline,
restrito, Pair-Aware (com ou sem MLP) ou o classificador da etapa 5; no restrito, a seleção dos pares sai do léxico gravado na execução, sem ler o
TRAIN) e grava `predicoes_test.json` (no conjunto completo), as métricas do TEST em `metricas.json` e
uma linha em `avaliacoes_test.jsonl` (trilha das avaliações do TEST, com o
`eval_index_for_config`). Recusa uma execução cujo treino não terminou.

No TEST científico (partições congeladas do projeto), a avaliação exige a
configuração final congelada por `scripts/congelar.py` e o código de liberação:

    python codigo/scripts/avaliar_test.py --execucao <pasta> --liberacao <código>

Sem o registro de congelamento, com outra execução ou com o código errado, a
avaliação é recusada antes de carregar o modelo ou abrir a partição (a trava é a
primeira instrução de `classificador.avaliar_test`). O código só deve
ser usado depois da autorização explícita do autor.

Quando a configuração congelada é derivada (`pair_aware_limiar`), este comando
é o primeiro passo, na base: a trava aceita a base da configuração congelada,
e o limiar é aplicado depois por `pair_aware_limiar.py aplicar-test`.

O TEST de uma execução é avaliado uma vez. Avaliar de novo exige
`--reavaliar` e fica registrado na trilha como uma segunda avaliação da mesma
configuração, que precisa ser declarada.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin.estrategias import ablacao_pooler, baseline, pair_aware, pair_aware_mlp, restrito
from reclin.util import caminhos
from reclin.util.io import ler_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.avaliar_test")

ESTRATEGIAS = {m.NOME: m for m in (baseline, restrito, pair_aware, pair_aware_mlp, *ablacao_pooler.VARIANTES)}


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
    ap.add_argument("--congelamento", type=Path, default=None,
                    help="registro da configuração final (padrão: resultados/congelamento/configuracao_final.json)")
    ap.add_argument("--liberacao", default=None,
                    help="código de liberação do registro de congelamento (só após autorização do autor)")
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
            dispositivo=None if args.dispositivo == "auto" else args.dispositivo,
            congelamento=args.congelamento, liberacao=args.liberacao)
    except (classificador.ErroExecucao, ValueError) as erro:
        log.error("%s", erro)
        return 1
    if resultado["trilha"]["eval_index_for_config"] > 1:
        log.warning("Esta é a avaliação %d do TEST para esta configuração: declare a multiplicidade.",
                    resultado["trilha"]["eval_index_for_config"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
