#!/usr/bin/env python3
"""Reempacota uma execução treinada por outra versão do código, sob o código atual (só o DEV é lido).

    python codigo/scripts/reempacotar.py --base <execucoes>/pair_aware__a3ac671a77e2__t2 \\
        --nome pair_aware_sem_pesos_biobertpt_seed42 [--saida <execucoes>] [--particoes <pasta>]

Recarrega o `melhor_modelo/` da base com o código atual, prevê o DEV e compara
com o `predicoes_dev.json` gravado pela base (mesma classe em todos os pares;
probabilidades com diferença absoluta <= 0,001). Só se forem equivalentes
cria a execução nova, com a mesma configuração, uma cópia conferida dos pesos
e o registro da comparação (`reempacotamento` em config.json e treino.json).
Com qualquer divergência, nada é gravado. Não treina, não lê o TEST e não
altera a base. Ver `reclin.treino.reempacotamento`.
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

log = logging.getLogger("reclin.scripts.reempacotar")

ESTRATEGIAS = {m.NOME: m for m in (baseline, restrito, pair_aware, pair_aware_mlp, *ablacao_pooler.VARIANTES)}


def montagem_da_execucao(pasta: Path):
    """A montagem da estratégia que treinou a execução (a mesma regra de avaliar_test.py)."""
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
    ap.add_argument("--base", type=Path, required=True, help="pasta da execução treinada")
    ap.add_argument("--nome", required=True, help="nome da execução reempacotada")
    ap.add_argument("--saida", type=Path, default=None, help="raiz das execuções (padrão: a da base)")
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES, help="partições (só o DEV é lido)")
    ap.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args(argv)
    configurar()

    from reclin.treino import reempacotamento, reprodutibilidade
    reprodutibilidade.configurar_ambiente()
    if not (args.base / "config.json").is_file():
        log.error("%s não é uma execução (sem config.json)", args.base)
        return 1
    try:
        resultado = reempacotamento.reempacotar(
            args.base, args.saida or args.base.parent, args.nome, pasta_particoes=args.particoes,
            montagem=montagem_da_execucao(args.base),
            dispositivo=None if args.dispositivo == "auto" else args.dispositivo)
    except (reempacotamento.ErroReempacotamento, ValueError, FileExistsError) as erro:
        log.error("%s", erro)
        return 1
    c = resultado["comparacao"]
    log.info("Equivalente no DEV: %d pares, %d classes divergentes, diferença máxima de probabilidade %.6f",
             c["n"], c["classes_divergentes"], c["max_dif_prob"])
    log.info("Execução reempacotada: %s", resultado["pasta"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
