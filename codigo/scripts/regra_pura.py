#!/usr/bin/env python3
"""Executa a regra pura (pista lexical + distância, sem modelo).

    python codigo/scripts/regra_pura.py \\
        --calibracao codigo/testes/referencia/resultados_legado/CALIBRACAO_filtro.json \\
        --saida execucoes --conferir codigo/testes/referencia/resultados_legado

Cria a execução `regra_pura` em `--saida` e grava as predições de cada
partição pedida (DEV e TEST, por padrão). Não depende de nenhuma outra
execução nem de outra estratégia.

Nada é escolhido aqui: a regra (`rule`), o limiar (`rule_gap`) e `min_freq`
vêm da calibração no DEV (`calibrar.py`), e o léxico é o congelado da etapa 3
(`reclin.negacao.lexico.CONGELADO`) — o script falha se a calibração não
corresponder a ele. Aplicar a regra ao TEST só gera predições; avaliá-las é
outro passo (`avaliar.py`).

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
from reclin.estrategias import regra_pura
from reclin.execucao import diretorio
from reclin.negacao import lexico
from reclin.util import caminhos
from reclin.util.io import ler_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.regra_pura")


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
    ap.add_argument("--calibracao", type=Path, required=True, help="CALIBRACAO_filtro.json")
    ap.add_argument("--particao", nargs="+", choices=diretorio.PARTICOES,
                    default=list(diretorio.PARTICOES))
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "execucoes")
    ap.add_argument("--conferir", type=Path, default=None,
                    help="pasta de referência para comparar os sidecars byte a byte")
    args = ap.parse_args(argv)
    configurar()

    divergencias = particoes.conferir_particoes(args.particoes, ["train", *args.particao])
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
    regra, gap, nome = calibracao["rule"], calibracao["rule_gap"], regra_pura.NOME
    log.info("Regra %s com gap<=%d (calibrada no DEV), léxico congelado de %d formas",
             regra, gap, len(lex))

    pasta = diretorio.criar(args.saida, nome, estrategia=regra_pura.NOME, config={
        "regra": regra, "max_target_gap": gap, "min_freq": calibracao["min_freq"],
        "max_gap": tarefa.MAX_GAP, "n_formas": len(lex), "lexico_sha1": lexico.lexico_sha1(lex),
        "calibracao": calibracao})
    manifesto = particoes.ler_manifesto(args.particoes)
    resultados_conferencia = []
    for p in args.particao:
        docs = particoes.ler_particao(args.particoes, p)
        conjunto = tarefa.conjunto_referencia(p, docs, particao_sha256=manifesto["splits"][p]["sha256"])
        pred = regra_pura.montar_predicoes(tarefa.candidatos(docs), lex, regra=regra,
                                           max_target_gap=gap, min_freq=calibracao["min_freq"])
        gravado = diretorio.gravar_predicoes(pasta, p, pred, conjunto)
        log.info("%s [%s]: %d negation_of em %d candidatos; gravado em %s", nome, p,
                 sum(1 for y in pred["y_pred"] if y == tarefa.NEG), len(pred["y_pred"]), gravado)
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
