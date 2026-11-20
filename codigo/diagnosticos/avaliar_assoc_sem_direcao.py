#!/usr/bin/env python3
"""Avalia `associated_with` sem direção para os sistemas auditados (só DEV).

    python codigo/diagnosticos/avaliar_assoc_sem_direcao.py \\
        --inventario codigo/resultados/auditoria_execucoes/inventario.json \\
        --execucoes codigo/resultados/execucoes \\
        --derivadas tcc/analise/execucoes_derivadas_dev

Lê a partição DEV e as predições do DEV de cada sistema de três rótulos listado
no inventário; grava `resultados/diagnosticos/associated_with_sem_direcao.json`.
Análise secundária e descritiva (`reclin.analises.associated_sem_direcao`). Não
treina e não lê o TEST.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin import particoes as particoes_mod
from reclin import tarefa
from reclin.analises import associated_sem_direcao
from reclin.analises import auditoria_execucoes as aud
from reclin.execucao import predicoes
from reclin.util import caminhos, identidade_codigo
from reclin.util.io import gravar_json, ler_json, sha256_arquivo
from reclin.util.log import configurar

log = logging.getLogger("reclin.diagnosticos.assoc_sem_direcao")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inventario", type=Path, default=caminhos.RESULTADOS / "auditoria_execucoes" / "inventario.json")
    ap.add_argument("--execucoes", type=Path, default=caminhos.RESULTADOS / "execucoes")
    ap.add_argument("--derivadas", type=Path, default=None, help="pasta com as execuções derivadas no DEV")
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "diagnosticos" / "associated_with_sem_direcao.json")
    args = ap.parse_args(argv)
    configurar()

    divergencias = particoes_mod.conferir_particoes(args.particoes, ["dev"])
    if divergencias:
        raise SystemExit("partição DEV não confere com o MANIFEST:\n  " + "\n  ".join(divergencias))
    cands = tarefa.candidatos(particoes_mod.ler_particao(args.particoes, "dev"))
    y_true = tarefa.y_true(cands)

    sistemas = {}
    for r in ler_json(args.inventario)["execucoes"]:
        if "predicoes_dev.json" not in r.get("arquivos", []) or r.get("tipo") not in ("execucao", "composicao_two_stage"):
            continue
        pasta = next((p for p in (args.execucoes / r["pasta"], (args.derivadas or args.execucoes) / r["pasta"])
                      if (p / "predicoes_dev.json").is_file()), None)
        if pasta is None:
            log.warning("%s: predições do DEV não encontradas; ignorada", r["pasta"])
            continue
        sidecar = predicoes.ler(pasta / "predicoes_dev.json")
        if sidecar.get("split") != "dev" or list(sidecar["y_true"]) != list(y_true):
            log.info("%s: não é um sistema de três rótulos sobre todos os candidatos do DEV; ignorada", r["pasta"])
            continue
        chave = f"{aud.chave_estrategia(r)}|{r['encoder']}"
        sistemas[chave] = {"pasta": r["pasta"], "predicoes_sha256": sha256_arquivo(pasta / "predicoes_dev.json"),
                           **associated_sem_direcao.avaliar(cands, sidecar["y_pred"])}
        s = sistemas[chave]
        log.info("%-40s F1 assoc com direção %.3f | sem direção %.3f (%+.3f)", chave,
                 s["com_direcao"]["f1"], s["sem_direcao"]["f1"], s["diferenca_f1"])
    gravar_json(args.saida, {
        "descricao": "associated_with avaliada sobre pares não ordenados (análise secundária, DEV)",
        "definicao": associated_sem_direcao.__doc__.split("Definição")[1].split("É uma análise")[0].strip(),
        "proveniencia": {"particao": "dev", "particao_sha256": sha256_arquivo(particoes_mod.arquivo_particao(args.particoes, "dev")),
                         "n_candidatos": len(cands), "codigo": identidade_codigo.atual()["identidade"],
                         "test_lido": False},
        "sistemas": sistemas})
    log.info("gravado %s (%d sistemas)", args.saida, len(sistemas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
