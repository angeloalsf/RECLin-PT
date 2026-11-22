#!/usr/bin/env python3
"""A configuração final no DEV: escolha de t, métricas e comparações pareadas (só DEV).

    python codigo/diagnosticos/avaliar_configuracao_final_dev.py \\
        --execucoes <checkout de resultados-colab>/codigo/resultados/execucoes \\
        --particoes codigo/dados/particoes-v2 \\
        --saida codigo/resultados/diagnosticos/configuracao_final_dev.json

Lê as probabilidades do DEV da Pair-Aware sem pesos (`pair_aware__a3ac671a77e2__t2`),
refaz a busca de t da configuração final (`estrategias.pair_aware_limiar`: grade
0,05…0,95; maior precisão de `negation_of` com recall >= 4/5), calcula as
métricas do DEV com a regra e compara a configuração final (A) com outras
execuções do mesmo DEV (B): F1 de `negation_of`, F1 de `associated_with`,
macro-F1 (`analises.auditoria_execucoes.comparar_pareado`) e precisão e recall
de `negation_of` (`comparar_precisao_recall.comparar`), por candidato e por
documento. É o mesmo cálculo que a execução derivada fará no Colab, feito aqui
sobre as predições versionadas, para o TCC e para conferir o notebook. Não
treina, não grava execução e não lê o TEST.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

CODIGO = Path(__file__).resolve().parents[1]
if str(CODIGO) not in sys.path:
    sys.path.insert(0, str(CODIGO))

from reclin import particoes as particoes_mod  # noqa: E402
from reclin import tarefa  # noqa: E402
from reclin.analises import auditoria_execucoes as aud  # noqa: E402
from reclin.avaliacao import limiar as busca_limiar  # noqa: E402
from reclin.avaliacao import metricas  # noqa: E402
from reclin.estrategias import pair_aware_limiar  # noqa: E402
from reclin.util.io import gravar_json, ler_json, sha256_arquivo  # noqa: E402

BASE = "pair_aware__a3ac671a77e2__t2"
COMPARADAS = {
    "pairaware_biobertpt_seed42": "Pair-Aware Linear ponderada, argmax (referência anterior)",
    BASE: "Pair-Aware Linear sem pesos, argmax",
    "two_stage_limiar_BioBERTpt-all": "Two-Stage (τ = 0,95)",
    "baseline_biobertpt_seed42": "Baseline",
}


def _comparar_pr():
    caminho = CODIGO / "diagnosticos" / "comparar_precisao_recall.py"
    spec = importlib.util.spec_from_file_location("comparar_precisao_recall", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.comparar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--execucoes", type=Path, required=True)
    ap.add_argument("--particoes", type=Path, required=True, help="partições v2 (só o DEV é lido)")
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=aud.N_BOOT)
    args = ap.parse_args(argv)

    divergencias = particoes_mod.conferir_particoes(args.particoes, ("dev",))
    if divergencias:
        raise SystemExit("DEV não confere com o MANIFEST: " + "; ".join(divergencias))
    cands = tarefa.candidatos(particoes_mod.ler_particao(args.particoes, "dev"))
    grupos = [c["doc_id"] for c in cands]
    y_ref = tarefa.y_true(cands)

    arquivo_base = args.execucoes / BASE / "predicoes_dev.json"
    base = ler_json(arquivo_base)
    if list(base["y_true"]) != y_ref:
        raise SystemExit("o y_true da base não é o do DEV reconstruído")
    criterio = pair_aware_limiar.criterio()
    t, tabela = busca_limiar.buscar(base["y_true"], lambda x: pair_aware_limiar.prever(base["probs"], x),
                                    criterio=criterio)
    y_final = pair_aware_limiar.prever(base["probs"], t)
    m = metricas.avaliar(base["y_true"], y_final)

    comparar_pr = _comparar_pr()
    comparacoes = {}
    for nome, rotulo in COMPARADAS.items():
        arquivo = args.execucoes / nome / "predicoes_dev.json"
        if not arquivo.is_file():
            comparacoes[nome] = {"rotulo": rotulo, "ausente": True}
            continue
        outra = ler_json(arquivo)
        if list(outra["y_true"]) != y_ref:
            raise SystemExit(f"{nome}: y_true diferente do DEV")
        r = aud.comparar_pareado(base["y_true"], y_final, outra["y_pred"], grupos=grupos, n_boot=args.n_boot)
        pr = comparar_pr(base["y_true"], y_final, outra["y_pred"], grupos, n_boot=args.n_boot)
        comparacoes[nome] = {"rotulo": rotulo, "predicoes_sha256": sha256_arquivo(arquivo),
                             "f1_e_macro": r, "precisao_recall_negation_of": pr,
                             "metricas_b": metricas.avaliar(outra["y_true"], outra["y_pred"])}

    saida = {
        "descricao": "configuração final (Pair-Aware Linear sem pesos, P(neg) >= t) no DEV; A = configuração final",
        "base": BASE, "base_predicoes_sha256": sha256_arquivo(arquivo_base),
        "regra": pair_aware_limiar.REGRA,
        "busca": busca_limiar.configuracao_busca(pair_aware_limiar.REGRA, criterio=criterio),
        "t": t, "tabela": tabela, "metricas_dev": m, "comparacoes": comparacoes,
        "n_boot": args.n_boot, "seed": aud.SEMENTE,
        "n_candidatos": len(cands), "n_documentos": len(set(grupos)), "test_lido": False,
    }
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    gravar_json(args.saida, saida)
    neg = m["por_classe"]["negation_of"]
    print(f"t = {t:.2f} | negation_of P={neg['precision']:.4f} R={neg['recall']:.4f} F1={neg['f1']:.4f} "
          f"(VP {neg['tp']}, FP {neg['fp']}, FN {neg['fn']}) | F1 assoc={m['f1_per_class']['associated_with']:.4f} "
          f"| macro-F1={m['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
