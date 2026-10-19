#!/usr/bin/env python3
"""Consolida, de forma determinística, as métricas de DEV das execuções do
branch `resultados-colab` do RECLin-PT e as comparações pareadas DESCRITIVAS
no DEV. Não treina, não avalia o TEST e não altera nenhum artefato de origem.

Uso:
    PYTHONPATH=<repo>/codigo python3 analise_dev.py <repo>/codigo/resultados/execucoes \
        <pasta das execuções derivadas no DEV> <saida.json>

As comparações usam `reclin.avaliacao.significancia.comparar` (bootstrap pareado
do F1 de negation_of, 10.000 reamostras, semente 42; McNemar exato), a mesma
função que o protocolo prevê para o TEST. No DEV elas são só descritivas: o τ do
Two-Stage e o t do baseline calibrado foram escolhidos nesse mesmo DEV.
"""
import hashlib
import json
import sys
from pathlib import Path

from reclin.avaliacao import significancia

ROTULOS = ["negation_of", "associated_with", "no_relation"]

SISTEMAS = [
    # chave, pasta, origem, rótulo, situação no protocolo
    ("baseline_bio", "E", "baseline_biobertpt_seed42", "Baseline (BioBERTpt)", "protocolo"),
    ("calibrado_bio", "D", "baseline_calibrado_biobertpt_seed42", "Baseline calibrado (BioBERTpt)", "protocolo; t calculado nesta revisão"),
    ("filtro_bio", "D", "filtro_biobertpt_seed42", "Filtro léxico (BioBERTpt)", "protocolo; calculado nesta revisão"),
    ("pairaware_bio", "E", "pairaware_biobertpt_seed42", "Pair-Aware (BioBERTpt)", "protocolo (D10)"),
    ("twostage_bio", "E", "two_stage_limiar_BioBERTpt-all", "Two-Stage (BioBERTpt)", "protocolo"),
    ("pamlp_bio", "E", "pair_aware_mlp_BioBERTpt-all", "Pair-Aware MLP (BioBERTpt)", "exploratória, fora do protocolo"),
    ("baseline_bert", "E", "baseline_BERTimbau", "Baseline (BERTimbau)", "comparação de encoders, fora do escopo pré-registrado"),
]

COMPARACOES = [
    ("twostage_bio", "baseline_bio", "principal no protocolo"),
    ("twostage_bio", "calibrado_bio", "principal no protocolo"),
    ("calibrado_bio", "baseline_bio", "exploratória no protocolo"),
    ("twostage_bio", "filtro_bio", "exploratória no protocolo"),
    ("twostage_bio", "pairaware_bio", "exploratória no protocolo"),
    ("pairaware_bio", "baseline_bio", "adicional"),
    ("twostage_bio", "pamlp_bio", "adicional (estratégia fora do protocolo)"),
    ("baseline_bert", "baseline_bio", "adicional (encoders, mesmo caminho do baseline)"),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def contagens(yt, yp):
    out = {}
    for c in ROTULOS:
        tp = sum(1 for a, b in zip(yt, yp) if a == c and b == c)
        fp = sum(1 for a, b in zip(yt, yp) if a != c and b == c)
        fn = sum(1 for a, b in zip(yt, yp) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
                  "support": tp + fn}
    return out


def main(execucoes: Path, derivadas: Path, saida: Path) -> None:
    resultado = {"sistemas": {}, "comparacoes_dev_descritivas": [], "tau_two_stage": None}
    preds = {}
    yt_ref = None
    for chave, onde, pasta, rotulo, situacao in SISTEMAS:
        base = (execucoes if onde == "E" else derivadas) / pasta
        arq = base / "predicoes_dev.json"
        p = json.loads(arq.read_text())
        labs = p["labels"]
        yt = [labs[v] if isinstance(v, int) else v for v in p["y_true"]]
        yp = [labs[v] if isinstance(v, int) else v for v in p["y_pred"]]
        if yt_ref is None:
            yt_ref = yt
        assert yt == yt_ref, f"{pasta}: y_true diferente"
        c = contagens(yt, yp)
        macro = sum(c[k]["f1"] for k in ROTULOS) / 3
        met = json.loads((base / "metricas.json").read_text())["dev"]
        assert abs(macro - met["macro_f1"]) < 1e-12, f"{pasta}: macro-F1 não confere"
        for k in ROTULOS:
            assert abs(c[k]["f1"] - met["por_classe"][k]["f1"]) < 1e-12, f"{pasta}: F1 {k}"
        matriz = [[sum(1 for a, b in zip(yt, yp) if a == r and b == q) for q in ROTULOS] for r in ROTULOS]
        resultado["sistemas"][chave] = {
            "rotulo": rotulo, "situacao": situacao, "pasta": pasta,
            "origem": "branch resultados-colab" if onde == "E" else "derivada no DEV nesta revisão (scripts do repositório)",
            "predicoes_sha256": sha256(arq), "metricas_sha256": sha256(base / "metricas.json"),
            "n": len(yt), "por_classe": c, "macro_f1": macro, "mcc": met["mcc"],
            "acuracia": met["micro_f1"], "matriz_confusao": {"rotulos": ROTULOS, "matriz": matriz},
        }
        preds[chave] = p
    for a, b, tipo in COMPARACOES:
        r = significancia.comparar(preds[a], preds[b], seed=42)
        resultado["comparacoes_dev_descritivas"].append({
            "a": a, "b": b, "tipo": tipo,
            "f1_a": r["target_f1"]["a"], "f1_b": r["target_f1"]["b"], "diferenca": r["target_f1"]["a_minus_b"],
            "bootstrap": r["paired_bootstrap"], "mcnemar": r["mcnemar"]})
    lim = json.loads((execucoes / "two_stage_limiar_BioBERTpt-all" / "limiar.json").read_text())
    resultado["tau_two_stage"] = {"tau": lim["limiar"], "limiar_sha256": sha256(execucoes / "two_stage_limiar_BioBERTpt-all" / "limiar.json"),
                                  "tabela": [{"tau": t["limiar"], "f1_negation_of": t["f1_negation_of"],
                                              "macro_f1": t["macro_f1"], "n_roteados": t["n_roteados"],
                                              "contagens": t["contagens"]} for t in lim["tabela"]]}
    cal = json.loads((derivadas / "baseline_calibrado_biobertpt_seed42" / "limiar.json").read_text())
    resultado["t_calibrado"] = {"t": cal["limiar"], "tabela": [{"t": t["limiar"], "f1_negation_of": t["f1_negation_of"],
                                                                 "macro_f1": t["macro_f1"]} for t in cal["tabela"]]}
    ts = json.loads((execucoes / "two_stage_limiar_BioBERTpt-all" / "metricas.json").read_text())
    resultado["diagnostico_estagios_two_stage"] = ts.get("dev_diagnostico_estagios")
    hist = {}
    for pasta in ["baseline_biobertpt_seed42", "baseline_BERTimbau", "pairaware_biobertpt_seed42",
                  "pair_aware_mlp_BioBERTpt-all", "twostage_e1_biobertpt_seed42", "twostage_e2_biobertpt_seed42"]:
        t = json.loads((execucoes / pasta / "treino.json").read_text())
        hist[pasta] = {"melhor_epoca": t.get("melhor_epoca"), "campo_selecao": t.get("campo_selecao", "dev_macro_f1"),
                       "epocas": t["config"]["epochs"], "dev_history": t["dev_history"],
                       "n_parametros": t["identidade_modelo"]["n_parametros"]}
    resultado["historico_treino"] = hist
    saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=1, sort_keys=True))
    print(f"gravado {saida}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
