#!/usr/bin/env python3
"""Pair-Aware com limiar: escolhe t no DEV de uma execução da Pair-Aware e o aplica ao TEST, uma vez.

    python codigo/scripts/pair_aware_limiar.py calibrar-dev --base EXEC_PAIR_AWARE [--nome N]
    python codigo/scripts/pair_aware_limiar.py aplicar-test --execucao EXEC_LIMIAR --liberacao <código>

calibrar-dev   lê as probabilidades do DEV que a base gravou (predicoes_dev.json),
               escolhe t na grade pré-registrada (0,05 a 0,95) pelo critério
               da configuração final — a maior precisão de negation_of entre os
               limiares com recall de negation_of >= 4/5; desempates: maior F1
               de negation_of, maior macro-F1, menor limiar —, congela t em
               limiar.json e grava o DEV com a regra, ao lado da base (mesma
               raiz de execuções). Não lê o TEST. Nome padrão:
               pair_aware_limiar_<resto do nome da base>.
aplicar-test   confere os vínculos de t (base, predições do DEV, proveniência,
               DEV) e aplica t às predições do TEST da base, que precisa ter
               sido avaliada antes sob a mesma liberação (scripts/avaliar_test.py
               na base). Uma vez; de novo só com --reavaliar, registrado na trilha.

Regra: P(negation_of) >= t → negation_of; senão o argmax entre
associated_with e no_relation (empate → associated_with). Sem GPU.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin.estrategias import pair_aware_limiar
from reclin.execucao import congelamento, validacao
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.pair_aware_limiar")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("calibrar-dev", help="escolhe e congela t no DEV")
    p.add_argument("--base", type=Path, required=True, help="execução da Pair-Aware (código atual)")
    p.add_argument("--nome", default=None)
    p = sub.add_parser("aplicar-test", help="aplica o t congelado ao TEST, uma vez")
    p.add_argument("--execucao", type=Path, required=True)
    p.add_argument("--base", type=Path, default=None, help="pasta da base, se não estiver ao lado")
    p.add_argument("--reavaliar", action="store_true")
    p.add_argument("--congelamento", type=Path, default=None, help="registro da configuração final")
    p.add_argument("--liberacao", default=None, help="código de liberação (só após autorização do autor)")
    args = ap.parse_args(argv)
    configurar()
    try:
        if args.comando == "calibrar-dev":
            nome = args.nome or pair_aware_limiar.nome_execucao(args.base.name)
            resultado = pair_aware_limiar.calibrar_dev(args.base, args.base.parent, nome)
            m = resultado["metricas"]
            linha = next(x for x in resultado["tabela"] if x["limiar"] == resultado["t"])
            c = linha["contagens"]["negation_of"]
            log.info("t congelado: %.2f | DEV: negation_of VP=%d FP=%d FN=%d | P=%.4f R=%.4f F1=%.4f | "
                     "macro-F1=%.4f%s", resultado["t"], c["tp"], c["fp"], c["fn"],
                     m["por_classe"]["negation_of"]["precision"], m["por_classe"]["negation_of"]["recall"],
                     m["f1_per_class"]["negation_of"], m["macro_f1"],
                     " | VALIDAÇÃO TÉCNICA (não avalia desempenho)" if resultado["validacao"] else "")
            log.info("Gravado em %s", args.base.parent / nome)
        else:
            resultado = pair_aware_limiar.aplicar_test(args.execucao, base=args.base, reavaliar=args.reavaliar,
                                                       congelamento=args.congelamento, liberacao=args.liberacao)
            m = resultado["metricas"]
            log.info("TEST com limiar (t=%.2f): F1 negation_of=%.4f | macro-F1=%.4f", resultado["t"],
                     m["f1_per_class"]["negation_of"], m["macro_f1"])
            if resultado["trilha"]["eval_index_for_config"] > 1:
                log.warning("Esta é a avaliação %d do TEST para esta configuração: declare a multiplicidade.",
                            resultado["trilha"]["eval_index_for_config"])
    except (pair_aware_limiar.ErroLimiarPairAware, congelamento.ErroCongelamento, validacao.ExecucaoDeValidacao,
            ValueError, KeyError) as erro:
        log.error("%s", erro)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
