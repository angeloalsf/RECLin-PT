#!/usr/bin/env python3
"""Calibração no DEV do filtro de pistas e da regra pura: `CALIBRACAO_filtro.json`.

    python codigo/scripts/calibrar.py \\
        --resultados codigo/testes/referencia/resultados_legado \\
        --saida calibracao/CALIBRACAO_filtro.json \\
        --conferir codigo/testes/referencia/resultados_legado/CALIBRACAO_filtro.json

Reproduz o procedimento histórico, feito de uma vez para as duas estratégias
e NUNCA sobre o TEST (o arquivo `test.jsonl` não é aberto):

1. `min_freq` do léxico pelo F1 médio do filtro de pistas sobre as predições
   do DEV das execuções de base (`filtro_pistas.calibrar_min_freq`);
2. a regra pura (R1 a R4 e o limiar de gap) com o léxico de (1)
   (`regra_pura.calibrar`);
3. a porta de gap depois do filtro, também com o léxico de (1)
   (`filtro_pistas.calibrar_porta_gap`).

As bases são, por padrão, as quatro execuções do legado na ordem da
calibração original; as predições do DEV de cada uma são procuradas em
`--resultados` no formato novo (`<nome>/predicoes_dev.json`) ou no do legado
(`<nome>.dev_preds.json`). Cada uma precisa ter o `y_true` do DEV.

Grava `--saida` com os cinco parâmetros, no formato do legado (JSON com
indentação 2, chaves ordenadas, sem quebra de linha final); com `--detalhes`,
também a varredura completa num JSON. Com `--conferir`, compara os bytes do
arquivo gravado com os de uma referência (saída 1 se diferirem).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from reclin import particoes, tarefa
from reclin.estrategias import filtro_pistas, regra_pura
from reclin.execucao import diretorio, predicoes
from reclin.negacao import lexico
from reclin.util import caminhos
from reclin.util.io import gravar_json, gravar_texto
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.calibrar")

BASES = ("baseline_biobertpt_seed42", "baseline_bertimbau_seed42",
         "baseline_biobertpt_seed43", "baseline_bertimbau_seed43")
LIDAS = ("train", "dev")          # o TEST não entra na calibração


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--resultados", type=Path, required=True,
                    help="pasta com as predições do DEV das execuções de base")
    ap.add_argument("--bases", nargs="+", default=list(BASES),
                    help="nomes das execuções de base (padrão: os quatro baselines)")
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "CALIBRACAO_filtro.json")
    ap.add_argument("--detalhes", type=Path, default=None, help="grava a varredura completa")
    ap.add_argument("--sobrescrever", action="store_true")
    ap.add_argument("--conferir", type=Path, default=None,
                    help="compara os bytes da saída com este arquivo")
    args = ap.parse_args(argv)
    configurar()

    if args.saida.exists() and not args.sobrescrever:
        log.error("%s já existe; para regravar, use --sobrescrever", args.saida)
        return 1
    divergencias = particoes.conferir_particoes(args.particoes, LIDAS)
    if divergencias:
        for d in divergencias:
            log.error("%s", d)
        return 1
    train = particoes.ler_particao(args.particoes, "train")
    cands = tarefa.candidatos(particoes.ler_particao(args.particoes, "dev"))
    y_true = tarefa.y_true(cands)
    n_gold = sum(1 for t in y_true if t == tarefa.NEG)
    log.info("DEV: %d candidatos, %d negation_of; o TEST não é lido", len(cands), n_gold)

    preds = {}
    for nome in args.bases:
        caminho = diretorio.caminho_predicoes(args.resultados, nome, "dev")
        p = predicoes.ler(caminho)
        if p["y_true"] != y_true:
            log.error("%s não tem o y_true do DEV: não pertence a este conjunto", caminho)
            return 1
        preds[nome] = p["y_pred"]
        log.info("Base %s: %s", nome, caminho)

    # 1. min_freq ------------------------------------------------------------
    cal_lex = filtro_pistas.calibrar_min_freq(lexico.contar_formas(train), cands, y_true, preds)
    for r in cal_lex["varredura"]:
        log.info("min_freq=%2d | %2d formas | cobertura %.4f | F1 %s | média %.4f",
                 r["min_freq"], r["n_formas"], r["cobertura"],
                 " ".join(f"{x['f1']:.4f}" for x in r["por_execucao"].values()), r["f1_medio"])
    min_freq = cal_lex["min_freq"]
    lex = lexico.induzir(train, min_freq=min_freq)
    sha = lexico.lexico_sha1(lex)
    congelado = (min_freq == lexico.CONGELADO["min_freq"] and sha == lexico.CONGELADO["lexico_sha1"])
    log.info("Escolhido min_freq=%d: %d formas, lexico_sha1=%s (%s)", min_freq, len(lex), sha,
             "é o léxico congelado" if congelado else "DIFERENTE do léxico congelado")

    # 2. regra pura ----------------------------------------------------------
    cal_regra = regra_pura.calibrar(cands, y_true, lex)
    for r in cal_regra["varredura"]:
        log.info("%s gap<=%-2d | TP=%3d FP=%3d FN=%3d | P=%.4f R=%.4f F1=%.4f", r["regra"], r["gap"],
                 r["tp"], r["fp"], r["fn"], r["precision"], r["recall"], r["f1"])
    log.info("Escolhida a regra %s com gap<=%d", cal_regra["regra"], cal_regra["gap"])

    # 3. porta de gap --------------------------------------------------------
    cal_porta = filtro_pistas.calibrar_porta_gap(cands, y_true, preds, lex)
    for r in cal_porta["varredura"]:
        log.info("filtro + gap<=%-2d | F1 %s | média %.4f", r["gap"],
                 " ".join(f"{x['f1']:.4f}" for x in r["por_execucao"].values()), r["f1_medio"])
    log.info("Escolhida a porta gap<=%d%s", cal_porta["gap"],
             " (igual à janela de candidatos: porta desligada)" if cal_porta["gap"] >= tarefa.MAX_GAP else "")

    calibracao = {"min_freq": min_freq, "rule": cal_regra["regra"], "rule_gap": cal_regra["gap"],
                  "combined_gap": cal_porta["gap"], "lexicon_size": len(lex)}
    gravar_texto(args.saida, json.dumps(calibracao, ensure_ascii=False, indent=2, sort_keys=True))
    log.info("Calibração gravada em %s: %s", args.saida, calibracao)
    if args.detalhes:
        gravar_json(args.detalhes, {
            "calibracao": calibracao, "particao": "dev", "max_gap": tarefa.MAX_GAP,
            "n_candidatos": len(cands), "n_negation_of": n_gold, "bases": list(args.bases),
            "lexico": {"min_freq": min_freq, "lexico_sha1": sha, "congelado": congelado,
                       "formas": [[forma, n] for forma, n in lex.items()]},
            "min_freq": cal_lex["varredura"], "regra": cal_regra["varredura"],
            "porta_gap": cal_porta["varredura"]})
        log.info("Varredura completa em %s", args.detalhes)

    if args.conferir:
        if args.saida.read_bytes() != args.conferir.read_bytes():
            log.error("A calibração gravada DIFERE de %s", args.conferir)
            return 1
        log.info("A calibração gravada é idêntica, byte a byte, a %s", args.conferir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
