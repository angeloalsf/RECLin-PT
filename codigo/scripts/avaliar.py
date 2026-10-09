#!/usr/bin/env python3
"""Métricas de predições: de sidecars avulsos ou das partições de uma execução.

    python codigo/scripts/avaliar.py ARQUIVO.preds.json [...] [--saida metricas.json]
    python codigo/scripts/avaliar.py --execucao resultados/execucoes/<nome>

Com arquivos, imprime as métricas de cada sidecar e, com `--saida`, grava todas
num JSON. Com `--execucao`, calcula as métricas de cada partição que tiver
predições e grava `metricas.json` no diretório da execução.

As métricas são as de `reclin.avaliacao.metricas`: macro, micro e weighted-F1,
MCC, F1 por classe, relatório por classe e matriz de confusão.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin.avaliacao import metricas
from reclin.execucao import diretorio, predicoes
from reclin.tarefa import LABELS
from reclin.util.io import gravar_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.avaliar")


def resumir(rotulo: str, m: dict) -> None:
    neg = m["por_classe"]["negation_of"]
    log.info("%s | n=%d | macro-F1=%.4f | micro-F1=%.4f | weighted-F1=%.4f | MCC=%.4f",
             rotulo, m["n"], m["macro_f1"], m["micro_f1"], m["weighted_f1"], m["mcc"])
    log.info("  negation_of: TP=%d FP=%d FN=%d | P=%.4f R=%.4f F1=%.4f",
             neg["tp"], neg["fp"], neg["fn"], neg["precision"], neg["recall"], neg["f1"])
    log.info("  F1 por classe: %s", " | ".join(f"{r}={m['f1_per_class'][r]:.4f}" for r in LABELS))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("arquivos", nargs="*", type=Path, help="sidecars de predições")
    ap.add_argument("--saida", type=Path, default=None, help="grava as métricas dos arquivos num JSON")
    ap.add_argument("--execucao", type=Path, default=None, help="diretório de uma execução")
    args = ap.parse_args(argv)
    if not args.arquivos and args.execucao is None:
        ap.error("informe arquivos de predições ou --execucao")

    configurar()
    if args.execucao is not None:
        for particao in diretorio.PARTICOES:
            if (args.execucao / diretorio.arquivo_predicoes(particao)).exists():
                p = diretorio.ler_predicoes(args.execucao, particao)
                m = metricas.avaliar(p["y_true"], p["y_pred"])
                diretorio.gravar_metricas(args.execucao, particao, m)
                resumir(f"{args.execucao.name} [{particao}]", m)
        log.info("Métricas gravadas em %s", args.execucao / diretorio.METRICAS)

    todas = {}
    for arquivo in args.arquivos:
        p = predicoes.ler(arquivo)
        todas[arquivo.as_posix()] = metricas.avaliar(p["y_true"], p["y_pred"])
        resumir(arquivo.name, todas[arquivo.as_posix()])
    if args.saida and todas:
        gravar_json(args.saida, todas)
        log.info("Métricas gravadas em %s", args.saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
