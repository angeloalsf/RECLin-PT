#!/usr/bin/env python3
"""Diferença pareada de precisão e recall de `negation_of` entre duas execuções (só DEV).

    python codigo/diagnosticos/comparar_precisao_recall.py \\
        --a codigo/resultados/execucoes/pair_aware__a3ac671a77e2__t2 \\
        --b codigo/resultados/execucoes/pairaware_biobertpt_seed42 \\
        --saida codigo/resultados/diagnosticos/precisao_recall_sem_pesos_x_ponderada.json

A auditoria (`auditar_execucoes.py`) compara F1; quando duas execuções têm o
mesmo F1 em pontos de operação diferentes, a pergunta passa a ser se a troca
entre precisão e recall é distinguível do ruído. Este script calcula A − B para
precisão, recall e F1 de `negation_of`, com bootstrap pareado por documento
(10.000 reamostras, semente 42, IC95% por percentis) e por candidato. Leitura
descritiva de uma semente. Não treina e não lê o TEST.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from reclin import particoes as particoes_mod
from reclin import tarefa
from reclin.tarefa import NEG
from reclin.util import caminhos
from reclin.util.io import gravar_json, ler_json, sha256_arquivo

N_BOOT, SEMENTE = 10_000, 42


def _prf(vp, fp, fn):
    vp, fp, fn = (np.asarray(v, dtype=float) for v in (vp, fp, fn))
    p = np.divide(vp, vp + fp, out=np.zeros_like(vp), where=(vp + fp) > 0)
    r = np.divide(vp, vp + fn, out=np.zeros_like(vp), where=(vp + fn) > 0)
    f = np.divide(2 * vp, 2 * vp + fp + fn, out=np.zeros_like(vp), where=(2 * vp + fp + fn) > 0)
    return {"precision": p, "recall": r, "f1": f}


def _contagens(y_true, y_pred):
    """Por candidato: (VP, FP, FN) de `negation_of` como 0/1."""
    t, p = np.asarray(y_true) == NEG, np.asarray(y_pred) == NEG
    return np.stack([t & p, ~t & p, t & ~p], axis=1).astype(np.int64)


def comparar(y_true, y_a, y_b, grupos, *, n_boot: int = N_BOOT, seed: int = SEMENTE) -> dict:
    ca, cb = _contagens(y_true, y_a), _contagens(y_true, y_b)
    obs_a, obs_b = _prf(*ca.sum(0)), _prf(*cb.sum(0))
    saida = {"a": {k: float(v) for k, v in obs_a.items()}, "b": {k: float(v) for k, v in obs_b.items()},
             "a_contagens": dict(zip(("tp", "fp", "fn"), map(int, ca.sum(0)))),
             "b_contagens": dict(zip(("tp", "fp", "fn"), map(int, cb.sum(0)))), "n_boot": n_boot, "seed": seed}
    _, inv = np.unique(np.asarray(grupos), return_inverse=True)
    niveis = {"por_documento": inv, "por_candidato": np.arange(len(y_true))}
    for nivel, unidade in niveis.items():
        n = int(unidade.max()) + 1
        sa, sb = np.zeros((n, 3), dtype=np.int64), np.zeros((n, 3), dtype=np.int64)
        np.add.at(sa, unidade, ca)
        np.add.at(sb, unidade, cb)
        rng = np.random.default_rng(seed)
        da = {k: np.empty(n_boot) for k in obs_a}
        for i in range(n_boot):
            pesos = np.bincount(rng.integers(0, n, n), minlength=n)
            ma, mb = _prf(*(pesos @ sa)), _prf(*(pesos @ sb))
            for k in da:
                da[k][i] = ma[k] - mb[k]
        saida[nivel] = {"n_unidades": n}
        for k in da:
            baixo, alto = np.percentile(da[k], [2.5, 97.5])
            saida[nivel][k] = {"diferenca": float(obs_a[k] - obs_b[k]), "ic95": [float(baixo), float(alto)],
                               "leitura": "positiva" if baixo > 0 else "negativa" if alto < 0 else "não detectada"}
    return saida


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--saida", type=Path, required=True)
    args = ap.parse_args(argv)
    divergencias = particoes_mod.conferir_particoes(args.particoes, ["dev"])
    if divergencias:
        raise SystemExit("partição DEV não confere com o MANIFEST:\n  " + "\n  ".join(divergencias))
    cands = tarefa.candidatos(particoes_mod.ler_particao(args.particoes, "dev"))
    pa, pb = (ler_json(p / "predicoes_dev.json") for p in (args.a, args.b))
    if not (pa.get("split") == pb.get("split") == "dev") or list(pa["y_true"]) != list(pb["y_true"]) \
            or list(pa["y_true"]) != list(tarefa.y_true(cands)):
        raise SystemExit("as predições não são do mesmo DEV completo da partição")
    r = comparar(pa["y_true"], pa["y_pred"], pb["y_pred"], [c["doc_id"] for c in cands])
    gravar_json(args.saida, {"descricao": "A − B em precisão, recall e F1 de negation_of (DEV, descritivo)",
                             "a_execucao": args.a.name, "b_execucao": args.b.name,
                             "a_predicoes_sha256": sha256_arquivo(args.a / "predicoes_dev.json"),
                             "b_predicoes_sha256": sha256_arquivo(args.b / "predicoes_dev.json"),
                             "test_lido": False, **r})
    for nivel in ("por_candidato", "por_documento"):
        print(nivel, {k: (round(v["diferenca"], 4), [round(x, 4) for x in v["ic95"]], v["leitura"])
                      for k, v in r[nivel].items() if isinstance(v, dict)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
