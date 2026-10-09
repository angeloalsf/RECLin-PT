#!/usr/bin/env python3
"""Significância entre sistemas: McNemar exato e bootstrap pareado no F1 de negation_of.

    python codigo/scripts/comparar.py par --a A.preds.json --b B.preds.json --saida X.json
    python codigo/scripts/comparar.py protocolo --resultados DIR --saida DIR [--conferir DIR]

par        compara duas predições (sidecars do TEST). A semente do bootstrap é
           a de A ou, se A não tiver semente, a de B; `--seed` a substitui.
protocolo  roda as 26 comparações do TCC (`reclin.avaliacao.protocolo`) sobre
           as execuções em `--resultados`, no formato novo (diretórios) ou no
           do legado (`<nome>.preds.json`), e grava os relatórios em `--saida`
           com os nomes do legado. Com `--conferir DIR`, compara cada relatório
           byte a byte com o de mesmo nome em DIR e sai com código 1 se algum
           diferir ou faltar.

Exemplo (reproduz as 26 comparações do legado a partir das referências):

    python codigo/scripts/comparar.py protocolo \\
        --resultados codigo/testes/referencia/resultados_legado \\
        --saida comparacoes --conferir codigo/testes/referencia/resultados_legado
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from reclin.avaliacao import protocolo, significancia
from reclin.execucao import diretorio, predicoes
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.comparar")


def cmd_par(args) -> int:
    a, b = predicoes.ler(args.a), predicoes.ler(args.b)
    seed = args.seed if args.seed is not None else protocolo.semente_bootstrap(a["seed"], b["seed"])
    relatorio = significancia.comparar(a, b, seed=seed, alvo=args.alvo, n_boot=args.n_boot)
    significancia.gravar_relatorio(args.saida, relatorio)
    bs, mc = relatorio["paired_bootstrap"], relatorio["mcnemar"]
    log.info("F1(%s): A=%.4f B=%.4f A-B=%+.4f | IC95=[%+.4f, %+.4f] | p(bootstrap)=%.4g | "
             "p(McNemar)=%.4g | semente=%d", args.alvo, relatorio["target_f1"]["a"],
             relatorio["target_f1"]["b"], relatorio["target_f1"]["a_minus_b"],
             bs["ci95_low"], bs["ci95_high"], bs["p_value"], mc["p_value"], seed)
    log.info("Relatório gravado em %s", args.saida)
    return 0


def cmd_protocolo(args) -> int:
    comparacoes = protocolo.comparacoes_tcc()
    if args.so:
        comparacoes = [c for c in comparacoes if c.arquivo in set(args.so)]
    args.saida.mkdir(parents=True, exist_ok=True)
    problemas = 0
    for i, c in enumerate(comparacoes, 1):
        t0 = time.time()
        a = predicoes.ler(diretorio.caminho_predicoes(args.resultados, c.a, "test"))
        b = predicoes.ler(diretorio.caminho_predicoes(args.resultados, c.b, "test"))
        relatorio = significancia.comparar(a, b, seed=c.seed, n_boot=args.n_boot)
        destino = args.saida / c.arquivo
        significancia.gravar_relatorio(destino, relatorio)
        situacao = ""
        if args.conferir:
            referencia = args.conferir / c.arquivo
            if not referencia.is_file():
                situacao, problemas = "SEM REFERÊNCIA", problemas + 1
            elif destino.read_bytes() == referencia.read_bytes():
                situacao = "idêntico"
            else:
                situacao, problemas = "DIFERENTE", problemas + 1
        bs = relatorio["paired_bootstrap"]
        log.info("[%2d/%d] %-62s semente=%d  A-B=%+.4f  IC95=[%+.4f, %+.4f]  p=%.4f  %s  (%.1fs)",
                 i, len(comparacoes), c.arquivo, c.seed, relatorio["target_f1"]["a_minus_b"],
                 bs["ci95_low"], bs["ci95_high"], bs["p_value"], situacao, time.time() - t0)
    if args.conferir:
        log.info("%d de %d relatórios idênticos byte a byte à referência",
                 len(comparacoes) - problemas, len(comparacoes))
    return 1 if problemas else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("par", help="compara duas predições")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--alvo", default=significancia.ALVO)
    p.add_argument("--n-boot", type=int, default=significancia.N_BOOT)
    p = sub.add_parser("protocolo", help="as 26 comparações do TCC")
    p.add_argument("--resultados", type=Path, required=True)
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--conferir", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=significancia.N_BOOT)
    p.add_argument("--so", nargs="+", default=None, help="só estes relatórios (nomes de arquivo)")
    args = ap.parse_args(argv)

    configurar()
    return {"par": cmd_par, "protocolo": cmd_protocolo}[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
