#!/usr/bin/env python3
"""Análises que não são estratégias: por enquanto, o léxico de pistas de negação.

    python codigo/scripts/analisar.py lexico
    python codigo/scripts/analisar.py lexico --min-freq 2 --saida lexico.json

lexico  induz o léxico do TRAIN e mostra as formas com a frequência, o
        `lexico_sha1` e a cobertura em cada partição (fração dos pares
        candidatos `negation_of` cujo e1 é pista). Sem `--min-freq`, carrega o
        léxico congelado e confere que é o registrado em
        `reclin.negacao.lexico.CONGELADO` (falha se não for). Com `--saida`,
        grava tudo num JSON.

As partições são conferidas contra o MANIFEST antes de qualquer conta.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin import particoes
from reclin.negacao import lexico
from reclin.tarefa import MAX_GAP
from reclin.util import caminhos
from reclin.util.io import gravar_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.analisar")


def cmd_lexico(args) -> int:
    divergencias = particoes.conferir_particoes(args.particoes)
    if divergencias:
        for d in divergencias:
            log.error("%s", d)
        return 1
    docs = particoes.ler_particoes(args.particoes)
    if args.min_freq is None:
        lex = lexico.carregar_congelado(docs["train"])
        min_freq, origem = lexico.CONGELADO["min_freq"], "congelado"
    else:
        lex = lexico.induzir(docs["train"], min_freq=args.min_freq, max_gap=args.max_gap)
        min_freq, origem = args.min_freq, "induzido"

    sha = lexico.lexico_sha1(lex)
    log.info("Léxico %s: min_freq=%d, max_gap=%d, %d formas, lexico_sha1=%s",
             origem, min_freq, args.max_gap, len(lex), sha)
    for forma, n in lex.items():
        log.info("  %5d  %s", n, forma)
    coberturas = {}
    for nome in particoes.PARTICOES:
        coberturas[nome] = lexico.cobertura(docs[nome], lex, args.max_gap)
        c = coberturas[nome]
        log.info("Cobertura %-5s: %d de %d pares negation_of com e1 no léxico = %.4f",
                 nome, c["cobertos"], c["negation_of"], c["cobertura"])
    if args.saida:
        gravar_json(args.saida, {"origem": origem, "min_freq": min_freq, "max_gap": args.max_gap,
                                 "n_formas": len(lex), "lexico_sha1": sha, "formas": lex,
                                 "cobertura": coberturas})
        log.info("Gravado em %s", args.saida)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("lexico", help="léxico de pistas de negação")
    p.add_argument("--min-freq", type=int, default=None,
                   help="induz com este limiar em vez de carregar o congelado")
    p.add_argument("--max-gap", type=int, default=MAX_GAP)
    p.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    p.add_argument("--saida", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.comando == "lexico" and args.min_freq is None and args.max_gap != MAX_GAP:
        ap.error("o léxico congelado usa max_gap=25; para outro valor, informe --min-freq")

    configurar()
    return {"lexico": cmd_lexico}[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
