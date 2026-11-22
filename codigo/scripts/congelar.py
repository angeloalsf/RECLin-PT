#!/usr/bin/env python3
"""Congela a configuração final do RECLin-PT, antes da avaliação única no TEST.

    python codigo/scripts/congelar.py --execucao codigo/resultados/execucoes/<nome> \\
        --responsavel "<autor>" --justificativa "<por que esta configuração>"

Grava `resultados/congelamento/configuracao_final.json` com as impressões
digitais da execução (configuração, predições do DEV, limiar e pesos do melhor
modelo) e o código de liberação. Não lê o TEST, não carrega modelo e não
substitui um congelamento existente.

Depois do congelamento, `avaliar_test.py` (e os comandos equivalentes do
Two-Stage e do baseline calibrado) só aceitam essa execução, e só com
`--liberacao <código>`. O código deve ser usado apenas depois da autorização
explícita do autor: o TEST é avaliado uma única vez.

Uma configuração derivada (um limiar sobre as probabilidades de uma execução
treinada, como `pair_aware_limiar`) é congelada pela pasta da derivada: o
registro inclui as impressões da base (ao lado dela), com o hash dos pesos.
A avaliação, depois, tem dois passos sob a mesma liberação: `avaliar_test.py`
na base e `pair_aware_limiar.py aplicar-test` na derivada.

`--conferir` só verifica um congelamento existente contra a execução.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin.execucao import congelamento
from reclin.util import caminhos
from reclin.util.io import ler_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.congelar")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--execucao", type=Path, required=True, help="pasta da execução escolhida")
    ap.add_argument("--responsavel", default="", help="quem decide o congelamento")
    ap.add_argument("--justificativa", default="", help="critério e evidência da escolha (só DEV)")
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--saida", type=Path, default=None, help=f"padrão: {congelamento.caminho_padrao()}")
    ap.add_argument("--conferir", action="store_true", help="confere um congelamento existente; não grava")
    args = ap.parse_args(argv)
    configurar()
    destino = args.saida or congelamento.caminho_padrao()
    try:
        if args.conferir:
            registro = ler_json(destino)
            iguais = congelamento.impressoes_da_execucao(args.execucao) == registro["execucao"]
            integro = (registro.get("liberacao") or {}).get("codigo") == congelamento.codigo_de_liberacao(registro)
            log.info("congelamento de %s: execução %s; registro %s; avaliações do TEST na trilha final: %d",
                     registro["execucao"]["pasta"], "confere" if iguais else "DIVERGE",
                     "íntegro" if integro else "ALTERADO", len(congelamento.avaliacoes_registradas(destino)))
            return 0 if iguais and integro else 1
        registro = congelamento.criar(args.execucao, responsavel=args.responsavel,
                                      justificativa=args.justificativa, pasta_particoes=args.particoes)
        caminho = congelamento.gravar(registro, destino)
    except (congelamento.ErroCongelamento, FileNotFoundError) as erro:
        log.error("%s", erro)
        return 1
    log.info("Configuração final congelada: %s", registro["execucao"]["pasta"])
    log.info("Registro: %s", caminho)
    log.info("Código de liberação: %s (use em --liberacao só depois da autorização do autor)",
             congelamento.codigo_de_liberacao(registro))
    return 0


if __name__ == "__main__":
    sys.exit(main())
