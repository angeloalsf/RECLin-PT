"""Ordem de trabalho da adjudicação cega, em dois lotes (análise secundária).

A planilha cega de `reclin.analises.adjudicacao_negation_of` mistura os falsos
positivos de `negation_of` da execução adjudicada com controles (verdadeiros
positivos sorteados e os falsos negativos). Quando interessa revisar primeiro os
falsos positivos que outra execução também comete (os que persistem noutro
ponto de operação), entregar ao adjudicador só a lista desses casos revelaria
que todos são falsos positivos. Este módulo divide os casos em dois lotes sem
quebrar a cegueira:

* **lote 1**: os falsos positivos comuns às duas execuções, mais controles
  sorteados na mesma proporção da planilha;
* **lote 2**: os demais casos.

Cada lote mantém aproximadamente a proporção de falsos positivos da planilha
inteira, então pertencer a um lote não informa o grupo do caso. O arquivo cego
(`ordem_de_trabalho.csv`) traz só lote, ordem, caso e documento neutro; o
vínculo com os grupos fica no arquivo restrito (`lotes.json`). A planilha não é
alterada: as decisões continuam sendo registradas nela.

    python -m reclin.analises.adjudicacao_lotes \\
        --rastreabilidade <...>/restrito/rastreabilidade.json \\
        --outra-execucao codigo/resultados/execucoes/pair_aware__a3ac671a77e2__t2 \\
        --saida-cega <...>/cega/ordem_de_trabalho.csv \\
        --saida-restrita <...>/restrito/lotes.json

Só lê predições do DEV. Não treina e não lê o TEST.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reclin.analises.adjudicacao_negation_of import CONTROLE_FN, CONTROLE_VP, FALSO_POSITIVO
from reclin.tarefa import NEG
from reclin.util.io import gravar_json, ler_json, sha256_arquivo

SEMENTE = 42


def montar(casos: Sequence[Mapping[str, Any]], y_true: Sequence[int], y_pred_outra: Sequence[int], *,
           semente: int = SEMENTE) -> dict[str, Any]:
    """Divide os casos em dois lotes; o lote 1 contém os falsos positivos comuns à outra execução."""
    fp_outra = {i for i, (t, p) in enumerate(zip(y_true, y_pred_outra)) if int(p) == NEG and int(t) != NEG}
    fps = [c for c in casos if c["grupo"] == FALSO_POSITIVO]
    comuns = [c for c in fps if c["indice_candidato"] in fp_outra]
    if not comuns or len(comuns) == len(fps):
        raise ValueError(f"{len(comuns)} de {len(fps)} falsos positivos são comuns: não há dois lotes a formar")
    fracao = len(comuns) / len(fps)
    sorteio = random.Random(semente)
    lote1 = {c["caso"] for c in comuns}
    for grupo in (CONTROLE_VP, CONTROLE_FN):
        do_grupo = sorted(c["caso"] for c in casos if c["grupo"] == grupo)
        lote1.update(sorteio.sample(do_grupo, round(fracao * len(do_grupo))))
    lotes = {c["caso"]: 1 if c["caso"] in lote1 else 2 for c in casos}
    resumo = {}
    for n in (1, 2):
        do_lote = [c for c in casos if lotes[c["caso"]] == n]
        por_grupo = {g: sum(1 for c in do_lote if c["grupo"] == g) for g in (FALSO_POSITIVO, CONTROLE_VP, CONTROLE_FN)}
        resumo[str(n)] = {"casos": len(do_lote), "por_grupo": por_grupo,
                          "proporcao_de_falsos_positivos": por_grupo[FALSO_POSITIVO] / len(do_lote)}
    return {"semente": semente, "fp_da_execucao_adjudicada": len(fps), "fp_comuns_a_outra_execucao": len(comuns),
            "fp_da_outra_execucao_fora_da_planilha": len(fp_outra - {c["indice_candidato"] for c in fps}),
            "lotes": resumo, "lote_por_caso": dict(sorted(lotes.items())),
            "fp_comum_por_caso": {c["caso"]: c["indice_candidato"] in fp_outra for c in fps}}


def gravar_cego(caminho: Path, casos: Sequence[Mapping[str, Any]], lote_por_caso: Mapping[str, int]) -> None:
    """Só lote, ordem, caso e documento neutro (nenhum grupo, rótulo, predição ou texto)."""
    linhas = sorted(((lote_por_caso[c["caso"]], c["caso"], c["documento_neutro"]) for c in casos))
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(["lote", "ordem_no_lote", "caso", "documento"])
        ordem = {1: 0, 2: 0}
        for lote, caso, documento in linhas:
            ordem[lote] += 1
            w.writerow([lote, ordem[lote], caso, documento])


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reclin.analises.adjudicacao_lotes", description=__doc__.split("\n")[0])
    ap.add_argument("--rastreabilidade", required=True, type=Path)
    ap.add_argument("--outra-execucao", required=True, type=Path, help="pasta da execução cujos FP definem o lote 1")
    ap.add_argument("--saida-cega", required=True, type=Path)
    ap.add_argument("--saida-restrita", required=True, type=Path)
    ap.add_argument("--semente", type=int, default=SEMENTE)
    args = ap.parse_args(list(argv) if argv is not None else None)
    for saida in (args.saida_cega, args.saida_restrita):
        if saida.exists():
            raise SystemExit(f"{saida} já existe: os lotes não são regerados por cima (apague à mão se for o caso)")
    rast = ler_json(args.rastreabilidade)
    sidecar = ler_json(args.outra_execucao / "predicoes_dev.json")
    if sidecar.get("split") != "dev":
        raise SystemExit("as predições da outra execução não são do DEV")
    for c in rast["casos"]:                       # o mesmo conjunto do DEV: os rótulos anotados conferem caso a caso
        anotado_neg = c["grupo"] != FALSO_POSITIVO
        if (int(sidecar["y_true"][c["indice_candidato"]]) == NEG) != anotado_neg:
            raise SystemExit(f"{c['caso']}: o rótulo anotado não confere com o conjunto da outra execução")
    resultado = montar(rast["casos"], sidecar["y_true"], sidecar["y_pred"], semente=args.semente)
    gravar_cego(args.saida_cega, rast["casos"], resultado["lote_por_caso"])
    gravar_json(args.saida_restrita, {
        "natureza": "ordem de trabalho da adjudicação cega; não altera a planilha nem as métricas",
        "rastreabilidade_sha256": sha256_arquivo(args.rastreabilidade),
        "execucao_adjudicada": rast["origem"]["execucao"], "outra_execucao": args.outra_execucao.name,
        "outra_execucao_predicoes_sha256": sha256_arquivo(args.outra_execucao / "predicoes_dev.json"),
        "ordem_de_trabalho_sha256": sha256_arquivo(args.saida_cega), **resultado})
    for n, r in resultado["lotes"].items():
        print(f"lote {n}: {r['casos']} casos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
