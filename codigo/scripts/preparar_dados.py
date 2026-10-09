#!/usr/bin/env python3
"""Prepara os dados: partições train/dev/test e o MANIFEST que as congela.

    python codigo/scripts/preparar_dados.py conferir
    python codigo/scripts/preparar_dados.py manifesto
    python codigo/scripts/preparar_dados.py particionar --sobrescrever

conferir     confere as partições contra o MANIFEST (código de saída 1 se
             divergirem). É o que se roda antes de qualquer experimento.
manifesto    recalcula o MANIFEST a partir das partições já gravadas, sem
             alterá-las. O SHA-256 do dataset de origem é recalculado se o
             dataset estiver disponível; se não estiver, mantém-se o já
             registrado, porque as partições continuam sendo as mesmas.
particionar  divide `dados/processados/dataset.jsonl` e grava partições e
             MANIFEST. As partições são versionadas e congeladas: o comando
             recusa sobrescrevê-las sem `--sobrescrever`.

A conversão do XML do SemClinBr em `dataset.jsonl` entra aqui como subcomando
`corpus` quando `reclin/corpus.py` for implementado.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin import particoes
from reclin.util import caminhos
from reclin.util.io import ler_jsonl, sha256_arquivo
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.preparar_dados")


def origem_do_dataset(dataset: Path, pasta: Path) -> dict:
    """`source` do MANIFEST: o dataset de onde as partições vieram."""
    caminho = caminhos.relativo_ao_codigo(dataset)
    if dataset.exists():
        return {"path": caminho, "sha256": sha256_arquivo(dataset)}
    anterior = pasta / particoes.ARQUIVO_MANIFESTO
    sha = particoes.ler_manifesto(pasta)["source"].get("sha256") if anterior.exists() else None
    log.warning("Dataset ausente (%s): mantido o SHA-256 de origem já registrado (%s)",
                caminho, sha)
    return {"path": caminho, "sha256": sha}


def cmd_conferir(args) -> int:
    divergencias = particoes.conferir_particoes(args.pasta)
    for d in divergencias:
        log.error("%s", d)
    if divergencias:
        return 1
    for nome, entrada in particoes.ler_manifesto(args.pasta)["splits"].items():
        log.info("%-5s confere | %d documentos | sha256=%s", nome, entrada["n_records"],
                 entrada["sha256"])
    return 0


def cmd_manifesto(args) -> int:
    faltando = [n for n in particoes.PARTICOES
                if not particoes.arquivo_particao(args.pasta, n).exists()]
    if faltando:
        log.error("Partições ausentes em %s: %s", args.pasta, ", ".join(faltando))
        return 2
    arquivo = args.pasta / particoes.ARQUIVO_MANIFESTO
    seed = particoes.ler_manifesto(args.pasta)["seed"] if arquivo.exists() else args.seed
    manifesto = particoes.calcular_manifesto(
        args.pasta, seed=seed, origem=origem_do_dataset(args.dataset, args.pasta))
    caminho = particoes.gravar_manifesto(args.pasta, manifesto)
    log.info("MANIFEST recalculado sem alterar as partições: %s", caminho)
    for nome, entrada in manifesto["splits"].items():
        log.info("  %-5s sha256=%s n_records=%d", nome, entrada["sha256"], entrada["n_records"])
    return 0


def cmd_particionar(args) -> int:
    existentes = [n for n in particoes.PARTICOES
                  if particoes.arquivo_particao(args.pasta, n).exists()]
    if existentes and not args.sobrescrever:
        log.error("Já há partições em %s (%s). Elas são congeladas; use --sobrescrever "
                  "só se for de fato regerá-las.", args.pasta, ", ".join(existentes))
        return 2
    if not args.dataset.exists():
        log.error("Dataset não encontrado: %s", args.dataset)
        return 2
    documentos = list(ler_jsonl(args.dataset))
    log.info("%d documentos lidos de %s", len(documentos), args.dataset)
    divisao = particoes.particionar(documentos, seed=args.seed)
    gravados = particoes.gravar_particoes(divisao, args.pasta)
    for nome, n in gravados.items():
        log.info("  %-5s %d documentos (%.1f%%)", nome, n, 100 * n / len(documentos))
    origem = {"path": caminhos.relativo_ao_codigo(args.dataset),
              "sha256": sha256_arquivo(args.dataset)}
    manifesto = particoes.calcular_manifesto(args.pasta, seed=args.seed, origem=origem)
    log.info("MANIFEST gravado em %s", particoes.gravar_manifesto(args.pasta, manifesto))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pasta", type=Path, default=caminhos.PARTICOES,
                    help="pasta das partições (default: codigo/dados/particoes)")
    ap.add_argument("--dataset", type=Path, default=caminhos.DATASET,
                    help="dataset.jsonl de origem (default: codigo/dados/processados/dataset.jsonl)")
    ap.add_argument("--seed", type=int, default=particoes.SEED)
    sub = ap.add_subparsers(dest="comando", required=True)
    sub.add_parser("conferir", help="confere as partições contra o MANIFEST")
    sub.add_parser("manifesto", help="recalcula o MANIFEST das partições gravadas")
    p = sub.add_parser("particionar", help="gera as partições a partir do dataset")
    p.add_argument("--sobrescrever", action="store_true")
    args = ap.parse_args(argv)

    configurar()
    comandos = {"conferir": cmd_conferir, "manifesto": cmd_manifesto,
                "particionar": cmd_particionar}
    return comandos[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
