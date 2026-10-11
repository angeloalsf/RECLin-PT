#!/usr/bin/env python3
"""Baseline calibrado: escolhe o limiar t no DEV do baseline e o aplica ao TEST, uma vez.

    python codigo/scripts/baseline_calibrado.py calibrar-dev --base EXEC_BASELINE [--nome N] [--saida DIR]
    python codigo/scripts/baseline_calibrado.py aplicar-test --execucao EXEC_CALIBRADO [--base EXEC_BASELINE]

calibrar-dev   lê as probabilidades do DEV que o baseline gravou
               (predicoes_dev.json), escolhe t na grade pré-registrada (a mesma
               do τ do Two-Stage; F1 de negation_of; desempates do protocolo),
               congela t em limiar.json e grava o DEV calibrado. Não lê o TEST.
               Nome padrão: baseline_calibrado_<encoder>_seed<S> (ou, sobre uma
               base de validação técnica, baseline_calibrado_validacao_<N>ep).
aplicar-test   confere os vínculos de t (base, predições do DEV, proveniência,
               DEV) e aplica t às predições do TEST da base, que precisa ter
               sido avaliado antes (scripts/avaliar_test.py). Uma vez; de novo só
               com --reavaliar, registrado na trilha. Recusa execuções de
               validação técnica.

Regra: P(negation_of) >= t → negation_of; senão o argmax entre
associated_with e no_relation (empate → associated_with). Sem GPU.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin.estrategias import baseline_calibrado
from reclin.execucao import validacao
from reclin.util import caminhos
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.baseline_calibrado")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("calibrar-dev", help="escolhe e congela t no DEV")
    p.add_argument("--base", type=Path, required=True, help="execução do baseline (código novo)")
    p.add_argument("--nome", default=None)
    p.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "execucoes")
    p = sub.add_parser("aplicar-test", help="aplica o t congelado ao TEST, uma vez")
    p.add_argument("--execucao", type=Path, required=True)
    p.add_argument("--base", type=Path, default=None, help="pasta da base, se tiver mudado de lugar")
    p.add_argument("--reavaliar", action="store_true")
    args = ap.parse_args(argv)
    configurar()
    try:
        if args.comando == "calibrar-dev":
            nome = args.nome
            if nome is None:
                marca = validacao.da_execucao(args.base)
                if marca:
                    nome = validacao.nome(baseline_calibrado.NOME, marca["epocas"])
                else:
                    nome = baseline_calibrado.nome_execucao(args.base.name)
            if not (args.base / "config.json").is_file():
                raise baseline_calibrado.ErroCalibracao(f"{args.base} não é uma execução do código novo")
            resultado = baseline_calibrado.calibrar_dev(args.base, args.saida, nome)
            m = resultado["metricas"]
            log.info("t congelado: %.2f | DEV calibrado: F1 negation_of=%.4f | macro-F1=%.4f%s", resultado["t"],
                     m["f1_per_class"]["negation_of"], m["macro_f1"],
                     " | VALIDAÇÃO TÉCNICA (não avalia desempenho)" if resultado["validacao"] else "")
            log.info("Gravado em %s", args.saida / nome)
        else:
            resultado = baseline_calibrado.aplicar_test(args.execucao, base=args.base, reavaliar=args.reavaliar)
            m = resultado["metricas"]
            log.info("TEST calibrado (t=%.2f): F1 negation_of=%.4f | macro-F1=%.4f", resultado["t"],
                     m["f1_per_class"]["negation_of"], m["macro_f1"])
            if resultado["trilha"]["eval_index_for_config"] > 1:
                log.warning("Esta é a avaliação %d do TEST para esta configuração: declare a multiplicidade.",
                            resultado["trilha"]["eval_index_for_config"])
    except (baseline_calibrado.ErroCalibracao, validacao.ExecucaoDeValidacao, ValueError, KeyError) as erro:
        log.error("%s", erro)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
