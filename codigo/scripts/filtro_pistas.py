#!/usr/bin/env python3
"""Executa o filtro de pistas sobre as predições de execuções de base.

    python codigo/scripts/filtro_pistas.py \\
        --resultados codigo/testes/referencia/resultados_legado \\
        --calibracao codigo/testes/referencia/resultados_legado/CALIBRACAO_filtro.json \\
        --saida execucoes --conferir codigo/testes/referencia/resultados_legado

Para cada base (por padrão, os quatro baselines), cria a execução
`filtro_<encoder>_seed<N>` em `--saida` e grava as predições filtradas de
cada partição pedida (DEV e TEST, por padrão). As predições da base são
procuradas em `--resultados`, no formato novo ou no do legado
(`<nome>.preds.json`, `<nome>.dev_preds.json`).

Nada é escolhido aqui: `min_freq` vem da calibração no DEV (`calibrar.py`), e o
léxico é o congelado da etapa 3 (`reclin.negacao.lexico.CONGELADO`) — o script
falha se a calibração não corresponder a ele. Aplicar o filtro ao TEST só gera
predições; avaliá-las é outro passo (`avaliar.py`).

Com `--conferir PASTA`, compara os bytes de cada sidecar gravado com o da
mesma execução em PASTA (formato novo ou do legado), quando existir; saída 1
se algum diferir.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin import particoes, tarefa
from reclin.estrategias import filtro_pistas
from reclin.execucao import diretorio, predicoes
from reclin.negacao import lexico
from reclin.util import caminhos
from reclin.util.io import ler_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.filtro_pistas")

BASES = ("baseline_biobertpt_seed42", "baseline_bertimbau_seed42",
         "baseline_biobertpt_seed43", "baseline_bertimbau_seed43")


def carregar_lexico(train, calibracao: dict) -> dict[str, int]:
    """O léxico congelado, conferido contra a calibração."""
    if calibracao["min_freq"] != lexico.CONGELADO["min_freq"]:
        raise ValueError(f"a calibração escolheu min_freq={calibracao['min_freq']}, e o léxico "
                         f"congelado tem min_freq={lexico.CONGELADO['min_freq']}: uma recalibração "
                         "precisa atualizar o registro CONGELADO")
    lex = lexico.carregar_congelado(train)
    if len(lex) != calibracao["lexicon_size"]:
        raise ValueError(f"o léxico tem {len(lex)} formas e a calibração registra "
                         f"{calibracao['lexicon_size']}")
    return lex


def conferir(gravado: Path, referencias: Path, nome: str, particao: str) -> bool | None:
    """True/False se há referência para comparar; None se não há."""
    try:
        ref = diretorio.caminho_predicoes(referencias, nome, particao)
    except FileNotFoundError:
        log.info("%s [%s]: sem referência em %s", nome, particao, referencias)
        return None
    igual = gravado.read_bytes() == ref.read_bytes()
    (log.info if igual else log.error)("%s [%s]: %s byte a byte a %s", nome, particao,
                                       "idêntico" if igual else "DIFERENTE", ref)
    return igual


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--resultados", type=Path, required=True,
                    help="pasta com as predições das execuções de base")
    ap.add_argument("--calibracao", type=Path, required=True, help="CALIBRACAO_filtro.json")
    ap.add_argument("--bases", nargs="+", default=list(BASES))
    ap.add_argument("--particao", nargs="+", choices=diretorio.PARTICOES,
                    default=list(diretorio.PARTICOES))
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "execucoes")
    ap.add_argument("--conferir", type=Path, default=None,
                    help="pasta de referência para comparar os sidecars byte a byte")
    args = ap.parse_args(argv)
    configurar()

    lidas = ["train", *args.particao]
    divergencias = particoes.conferir_particoes(args.particoes, lidas)
    if divergencias:
        for d in divergencias:
            log.error("%s", d)
        return 1
    calibracao = ler_json(args.calibracao)
    try:
        lex = carregar_lexico(particoes.ler_particao(args.particoes, "train"), calibracao)
    except ValueError as erro:
        log.error("%s", erro)
        return 1
    log.info("Léxico congelado: %d formas, min_freq=%d (calibração %s)",
             len(lex), calibracao["min_freq"], args.calibracao)

    manifesto = particoes.ler_manifesto(args.particoes)
    conjuntos, candidatos = {}, {}
    for p in args.particao:
        docs = particoes.ler_particao(args.particoes, p)
        candidatos[p] = tarefa.candidatos(docs)
        conjuntos[p] = tarefa.conjunto_referencia(
            p, docs, particao_sha256=manifesto["splits"][p]["sha256"])

    resultados_conferencia = []
    for base in args.bases:
        nome = filtro_pistas.nome_execucao(base)
        pasta = diretorio.criar(args.saida, nome, estrategia=filtro_pistas.NOME, config={
            "base": base, "min_freq": calibracao["min_freq"], "max_gap": tarefa.MAX_GAP,
            "n_formas": len(lex), "lexico_sha1": lexico.lexico_sha1(lex),
            "calibracao": calibracao})
        for p in args.particao:
            origem = diretorio.caminho_predicoes(args.resultados, base, p)
            predicoes_base = predicoes.ler(origem)
            pred = filtro_pistas.montar_predicoes(
                predicoes_base, candidatos[p], lex,
                base_preds=origem.relative_to(args.resultados).as_posix(),
                min_freq=calibracao["min_freq"])
            gravado = diretorio.gravar_predicoes(pasta, p, pred, conjuntos[p])
            antes = sum(1 for y in predicoes_base["y_pred"] if y == tarefa.NEG)
            depois = sum(1 for y in pred["y_pred"] if y == tarefa.NEG)
            log.info("%s [%s]: %d de %d negation_of rebaixados; gravado em %s",
                     nome, p, antes - depois, antes, gravado)
            if args.conferir:
                resultados_conferencia.append(conferir(gravado, args.conferir, nome, p))

    if args.conferir:
        comparados = [r for r in resultados_conferencia if r is not None]
        log.info("Conferência: %d sidecars comparados, %d idênticos", len(comparados), sum(comparados))
        if not all(comparados):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
