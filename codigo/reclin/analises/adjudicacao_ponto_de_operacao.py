"""Leva as decisões da adjudicação a outros pontos de operação (análise secundária).

A adjudicação cega (`reclin.analises.adjudicacao_negation_of`) revisa os falsos
positivos de `negation_of` de **uma** execução, a Pair-Aware linear ponderada.
Outras execuções avaliadas no mesmo DEV (por exemplo, a treinada sem pesos de
classe) erram, em parte, nos mesmos pares. Este módulo responde, para cada
execução informada: dos seus falsos positivos, quantos foram adjudicados, e como
ficam precisão, recall e F1 de `negation_of` se os pares adjudicados como
negação válida (decisão principal `1`) passarem a contar como positivos.

Referência adjudicada (parcial): os positivos são os `negation_of` anotados mais
os pares adjudicados como `1`. Ela só corrige omissões encontradas entre os
falsos positivos da execução adjudicada; não revê os demais pares. Por isso:

* um falso positivo que não está entre os casos adjudicados continua falso
  positivo e é contado em `fp_fora_da_adjudicacao`;
* um par adjudicado como `1` que a execução não previu passa a ser falso
  negativo dela;
* os valores são secundários e exploratórios, não substituem as métricas
  oficiais e não entram em critério de seleção sem decisão registrada do autor.

Uso (depois de `congelar` as decisões):

    python -m reclin.analises.adjudicacao_ponto_de_operacao \\
        --rastreabilidade <...>/restrito/rastreabilidade.json \\
        --congeladas <...>/restrito/decisoes_congeladas.json \\
        --execucao codigo/resultados/execucoes/pairaware_biobertpt_seed42 \\
        --execucao codigo/resultados/execucoes/pair_aware__a3ac671a77e2__t2 \\
        --saida <...>/restrito/pontos_de_operacao.json

Só lê predições do DEV. Não treina e não lê o TEST.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reclin.analises.adjudicacao_negation_of import FALSO_POSITIVO
from reclin.tarefa import NEG
from reclin.util.io import gravar_json, ler_json, sha256_arquivo

NEGACAO_VALIDA = "1"


def _prf(vp: int, fp: int, fn: int) -> dict[str, Any]:
    return {"tp": vp, "fp": fp, "fn": fn,
            "precision": vp / (vp + fp) if vp + fp else None,
            "recall": vp / (vp + fn) if vp + fn else None,
            "f1": 2 * vp / (2 * vp + fp + fn) if 2 * vp + fp + fn else None}


def projetar(casos: Sequence[Mapping[str, Any]], decisoes: Mapping[str, Mapping[str, Any]],
             y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Métricas de `negation_of` de uma execução sob a referência adjudicada (parcial)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} rótulos e {len(y_pred)} predições")
    adjudicados = {c["indice_candidato"]: decisoes[c["caso"]]["principal"]
                   for c in casos if c["grupo"] == FALSO_POSITIVO and c["caso"] in decisoes}
    validos = {i for i, d in adjudicados.items() if d == NEGACAO_VALIDA}
    if any(y_true[i] == NEG for i in adjudicados):
        raise ValueError("um caso do grupo de falsos positivos tem negation_of anotado: rastreabilidade inconsistente")
    previstos = {i for i, p in enumerate(y_pred) if int(p) == NEG}
    anotados = {i for i, t in enumerate(y_true) if int(t) == NEG}
    fps = previstos - anotados
    oficial = _prf(len(previstos & anotados), len(fps), len(anotados - previstos))
    positivos = anotados | validos
    secundaria = _prf(len(previstos & positivos), len(previstos - positivos), len(positivos - previstos))
    return {
        "oficial": oficial,
        "sob_referencia_adjudicada": secundaria,
        "fp_da_execucao": len(fps),
        "fp_adjudicados": len(fps & adjudicados.keys()),
        "fp_fora_da_adjudicacao": len(fps - adjudicados.keys()),
        "fp_adjudicados_como_negacao_valida": len(fps & validos),
        "negacoes_validas_adjudicadas_nao_previstas": len(validos - previstos),
        "por_decisao_entre_os_fp_adjudicados": {
            d: sum(1 for i in fps & adjudicados.keys() if adjudicados[i] == d)
            for d in sorted(set(adjudicados.values()))},
    }


def analisar(rastreabilidade: str | Path, congeladas: str | Path, execucoes: Iterable[str | Path]) -> dict[str, Any]:
    rastreabilidade, congeladas = Path(rastreabilidade), Path(congeladas)
    rast, cong = ler_json(rastreabilidade), ler_json(congeladas)
    if cong.get("rastreabilidade_sha256") != sha256_arquivo(rastreabilidade):
        raise ValueError("as decisões congeladas foram conferidas contra outra rastreabilidade")
    fontes = dict(cong["decisoes"]["individuais"])
    if cong["decisoes"].get("consenso"):
        fontes["consenso"] = cong["decisoes"]["consenso"]
    saida: dict[str, Any] = {
        "natureza": "análise secundária exploratória: não altera rótulos, partições nem métricas oficiais",
        "referencia": "negation_of anotados + pares adjudicados como negação válida (parcial; ver o módulo)",
        "rastreabilidade_sha256": sha256_arquivo(rastreabilidade), "congeladas_sha256": sha256_arquivo(congeladas),
        "execucao_adjudicada": rast["origem"].get("execucao") if isinstance(rast.get("origem"), dict) else None,
        "execucoes": {}}
    for pasta in map(Path, execucoes):
        sidecar = ler_json(pasta / "predicoes_dev.json")
        if sidecar.get("split") != "dev":
            raise ValueError(f"{pasta.name}: as predições não são do DEV")
        saida["execucoes"][pasta.name] = {
            "predicoes_sha256": sha256_arquivo(pasta / "predicoes_dev.json"),
            "por_fonte_de_decisao": {nome: projetar(rast["casos"], d, sidecar["y_true"], sidecar["y_pred"])
                                     for nome, d in fontes.items()}}
    return saida


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reclin.analises.adjudicacao_ponto_de_operacao",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--rastreabilidade", required=True, type=Path)
    ap.add_argument("--congeladas", required=True, type=Path)
    ap.add_argument("--execucao", required=True, type=Path, action="append", help="pasta de execução (repetível)")
    ap.add_argument("--saida", required=True, type=Path)
    args = ap.parse_args(list(argv) if argv is not None else None)
    resultado = analisar(args.rastreabilidade, args.congeladas, args.execucao)
    gravar_json(args.saida, resultado)
    for nome, r in resultado["execucoes"].items():
        for fonte, m in r["por_fonte_de_decisao"].items():
            o, s = m["oficial"], m["sob_referencia_adjudicada"]
            print(f"{nome} [{fonte}]: P {o['precision']:.3f} -> {s['precision']:.3f} | R {o['recall']:.3f} -> "
                  f"{s['recall']:.3f} | F1 {o['f1']:.3f} -> {s['f1']:.3f} | FP fora da adjudicação: "
                  f"{m['fp_fora_da_adjudicacao']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
