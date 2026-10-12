#!/usr/bin/env python3
"""Confere o texto do TCC: caminhos antigos, afirmações numéricas e a validade dos valores gerados.

    python codigo/scripts/conferir_tcc.py [--tcc tcc] \\
        [--relatorios codigo/resultados/relatorios/cientifica] [--afirmacoes tcc/afirmacoes.json] \\
        [--execucoes codigo/resultados/execucoes] [--saida conferencia.json]

Três verificações (`reclin.relatorios.conferencia`):

1. o texto (`*.tex` em `--tcc`) não cita caminhos do repositório original;
2. cada afirmação de `--afirmacoes` (lista JSON de `{"id", "arquivo",
   "trecho", "chave"}`, com `{valor}` no trecho) aparece no arquivo com o
   valor atual do registro gerado por `gerar_tcc.py` (`valores/*.json` em
   `--relatorios`);
3. as execuções de onde vieram os valores ainda são as mesmas (sidecars e
   `config.json` pelo SHA-256, número de avaliações do TEST e natureza),
   pelo inventário de `--execucoes`.

Saída 0 se tudo confere; 1 se algo diverge; 2 se a conferência está pendente
(sem a pasta do TCC, sem a lista de afirmações ou sem valores gerados). A
lista de afirmações e o texto são da etapa 9; nenhuma afirmação do texto
antigo é incorporada aqui.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin.execucao import inventario as inv
from reclin.relatorios import conferencia, tabelas
from reclin.util import caminhos
from reclin.util.io import gravar_json, ler_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.conferir_tcc")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tcc", type=Path, default=caminhos.TCC)
    ap.add_argument("--relatorios", type=Path, default=caminhos.RESULTADOS / "relatorios" / "cientifica")
    ap.add_argument("--afirmacoes", type=Path, default=None)
    ap.add_argument("--execucoes", type=Path, default=caminhos.RESULTADOS / "execucoes")
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--saida", type=Path, default=None)
    args = ap.parse_args(argv)
    configurar()

    valores = None
    pendencias = []
    try:
        if (args.relatorios / "valores").is_dir():
            valores = tabelas.ler_valores(args.relatorios)
    except tabelas.ErroTabela as erro:
        log.error("%s", erro)
        return 1
    if valores is None:
        pendencias.append(f"sem valores gerados em {args.relatorios} (rode gerar_tcc.py)")
    afirmacoes = ler_json(args.afirmacoes) if args.afirmacoes and args.afirmacoes.is_file() else None
    resultado = conferencia.conferir(args.tcc, afirmacoes=afirmacoes, valores=valores)

    origens_divergentes = []
    if valores is not None:
        try:
            inventario = inv.inventariar(args.execucoes, pasta_particoes=args.particoes)
            origens_divergentes = inv.conferir_origens(tabelas.ler_origens(args.relatorios), inventario)
        except inv.ErroInventario as erro:
            origens_divergentes = [str(erro)]

    for achado in resultado.caminhos_antigos:
        log.error("caminho antigo em %s:%d: %s", achado["arquivo"], achado["linha"], achado["caminho"])
    for a in resultado.afirmacoes:
        if not a["ok"]:
            log.error("afirmação %s: %s", a["id"], a.get("motivo"))
    for d in origens_divergentes:
        log.error("origem dos valores: %s", d)
    for p in resultado.pendencias + pendencias:
        log.warning("pendente: %s", p)

    status = resultado.status
    if origens_divergentes:
        status = "divergente"
    elif status == "ok" and pendencias:
        status = "pendente"
    if args.saida:
        gravar_json(args.saida, {**resultado.como_dict(), "status": status,
                                 "origens_divergentes": origens_divergentes,
                                 "pendencias": resultado.pendencias + pendencias})
    log.info("Conferência: %s", status)
    return {"ok": 0, "divergente": 1, "pendente": 2}[status]


if __name__ == "__main__":
    sys.exit(main())
