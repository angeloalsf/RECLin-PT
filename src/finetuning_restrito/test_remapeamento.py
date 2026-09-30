#!/usr/bin/env python3
"""
Teste rapido em CPU (sem GPU, sem pesos, sem torch) do espaco restrito e do
remapeamento. Tarefa 1, item 4.

O que e verificado (cada item e um `assert`; qualquer falha aborta):

 1. O lexico induzido com min_freq=3 bate com a calibracao congelada da fase 2.
 2. O espaco completo reconstruido tem 19.064 (dev) e 19.210 (test) pares, e o
    `y_true` e identico, posicao a posicao, ao dos 4 sidecars oficiais de cada
    split (dev_preds e preds dos baselines).
 3. Cada janela do espaco restrito e byte-identica a janela que
    `relation_extraction.build_dataset` gera para o mesmo par (prova de que o
    dataset restrito e um subconjunto do dataset dos baselines, nao uma
    reimplementacao).
 4. `pesos_balanced` aplicado ao train COMPLETO reproduz os pesos que o nucleo
    registrou (40,55 / 7,17 / 0,3526); aplicado ao train RESTRITO, da os pesos
    efetivos desta frente.
 5. Remapeamento de uma predicao FICTICIA (sorteada) e de uma predicao oraculo:
    tamanho preservado, fora do espaco = no_relation, dentro = predicao do
    modelo; o recall do oraculo remapeado e exatamente o teto.
 6. Validacao cruzada com a fase 2: remapear as predicoes de cada baseline
    restritas ao espaco produz EXATAMENTE o mesmo conjunto de predicoes
    `negation_of` (e o mesmo F1) que o `filtro_*.preds.json` oficial.
 7. `src/significance.py`, sem nenhuma alteracao, roda sobre a predicao ficticia
    remapeada contra um baseline e devolve codigo 0; o F1 que ele calcula para
    o lado A bate com o calculado aqui.

Os arquivos da predicao ficticia vao para um diretorio temporario e sao
apagados. O unico arquivo gravado no repositorio e o registro da verificacao
(`--registro`), dentro de results/finetuning_restrito/.

Uso:
    python src/finetuning_restrito/test_remapeamento.py \
        --registro results/finetuning_restrito/tarefa1_verificacao_cpu.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import runpy  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolamento  # noqa: E402
from _isolamento import REPO, SRC, exigir_saida_isolada  # noqa: E402

from relation_extraction import LABELS, build_dataset, read_jsonl  # noqa: E402
from restricted_space import (MIN_FREQ, NEG, NOREL,  # noqa: E402
                              carregar_espacos, carregar_lexico, lexico_sha1,
                              pesos_balanced, remapear)

log = _isolamento.get()
ENC = ("biobertpt", "bertimbau")
SEEDS = (42, 43)


def f1_neg(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == NEG and p == NEG)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != NEG and p == NEG)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == NEG and p != NEG)
    den = 2 * tp + fp + fn
    return (2 * tp / den if den else 0.0), tp, fp, fn


def rodar_significance(a, b, out, seed):
    """Executa src/significance.py INALTERADO, como __main__, no mesmo processo.

    No mesmo processo (e nao via subprocess) porque o logger "significance" ja
    foi desviado por _isolamento; um subprocesso abriria logs/pipeline.log.
    """
    argv = sys.argv
    sys.argv = ["significance.py", "--a", str(a), "--b", str(b),
                "--target", "negation_of", "--n-boot", "2000",
                "--seed", str(seed), "--out", str(out)]
    try:
        runpy.run_path(str(SRC / "significance.py"), run_name="__main__")
        code = 0
    except SystemExit as e:
        code = int(e.code or 0)
    finally:
        sys.argv = argv
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=str(REPO / "data" / "splits"))
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    ap.add_argument("--registro", default=None)
    args = ap.parse_args()
    if args.registro:
        exigir_saida_isolada(args.registro)
    t0 = time.time()
    res_dir = Path(args.results_dir)
    reg = {"data": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checagens": {}}
    chk = reg["checagens"]

    # 1. lexico
    lex = carregar_lexico(read_jsonl(Path(args.splits_dir) / "train.jsonl"),
                          min_freq=MIN_FREQ, guarda=res_dir / "CALIBRACAO_filtro.json")
    assert len(lex) == 11, len(lex)
    chk["1_lexico"] = {"min_freq": MIN_FREQ, "n_formas": len(lex),
                       "sha1": lexico_sha1(lex), "formas": list(lex)}

    # 2. espaco completo e y_true
    esp = carregar_espacos(args.splits_dir, lex)
    assert esp["dev"].n_full == 19064 and esp["test"].n_full == 19210
    conferidos = []
    for enc in ENC:
        for s in SEEDS:
            for split, suf in (("dev", "dev_preds"), ("test", "preds")):
                p = res_dir / f"baseline_{enc}_seed{s}.{suf}.json"
                d = json.loads(p.read_text(encoding="utf-8"))
                assert d["labels"] == LABELS
                assert d["y_true"] == esp[split].y_true_full, p
                conferidos.append(p.name)
    chk["2_y_true_identico_aos_sidecars"] = conferidos

    # 3. janelas identicas as do build_dataset dos baselines
    for split in ("dev", "test"):
        X, y = build_dataset(list(read_jsonl(Path(args.splits_dir) / f"{split}.jsonl")),
                             25, 128)
        e = esp[split]
        assert y == e.y_true_full
        assert [X[i] for i in e.indices] == e.textos
    Xtr, ytr = build_dataset(list(read_jsonl(Path(args.splits_dir) / "train.jsonl")), 25, 128)
    assert ytr == esp["train"].y_true_full
    assert [Xtr[i] for i in esp["train"].indices] == esp["train"].textos
    chk["3_janelas_identicas_ao_build_dataset"] = {
        s: esp[s].n_restrito for s in ("train", "dev", "test")}

    # 4. pesos
    w_full = pesos_balanced(esp["train"].y_true_full)
    assert [round(x, 2) for x in w_full] == [40.55, 7.17, 0.35], w_full
    w_r = pesos_balanced(esp["train"].y_restrito)
    chk["4_pesos"] = {"completo_reproduz_nucleo": dict(zip(LABELS, [round(x, 4) for x in w_full])),
                      "restrito_efetivo": dict(zip(LABELS, [round(x, 4) for x in w_r]))}

    # 5. remapeamento de predicoes ficticias
    rng = random.Random(20260926)
    fict = {}
    for split in ("dev", "test"):
        e = esp[split]
        y_r = [rng.choice([NEG, NOREL, NEG, NOREL, 1]) for _ in e.indices]
        probs_r = [[0.5, 0.1, 0.4] for _ in e.indices]
        y_full, probs_full = e.remapear(y_r, probs_r)
        assert len(y_full) == e.n_full and len(probs_full) == e.n_full
        dentro = set(e.indices)
        assert all(y_full[i] == NOREL for i in range(e.n_full) if i not in dentro)
        assert all(probs_full[i] == [0.0, 0.0, 1.0] for i in range(e.n_full) if i not in dentro)
        assert [y_full[i] for i in e.indices] == y_r
        oraculo, _ = e.remapear(e.y_restrito)
        f, tp, fp, fn = f1_neg(e.y_true_full, oraculo)
        assert fp == 0 and abs(tp / (tp + fn) - e.teto_recall()) < 1e-12
        fict[split] = {"n_completo": e.n_full, "n_restrito": e.n_restrito,
                       "tamanho_preservado": True,
                       "oraculo_recall": tp / (tp + fn), "oraculo_f1": f,
                       "teto_recall": e.teto_recall()}
        fict[split]["_y_full"] = y_full
    try:
        remapear([0, 5], 3, [NEG, NEG])
        raise AssertionError("indice fora do espaco nao foi recusado")
    except ValueError:
        pass
    chk["5_remapeamento_ficticio"] = {k: {kk: vv for kk, vv in v.items() if kk != "_y_full"}
                                      for k, v in fict.items()}

    # 6. validacao cruzada com os filtros oficiais da fase 2
    cruz = {}
    for enc in ENC:
        for s in SEEDS:
            base = json.loads((res_dir / f"baseline_{enc}_seed{s}.preds.json").read_text("utf-8"))
            filt = json.loads((res_dir / f"filtro_{enc}_seed{s}.preds.json").read_text("utf-8"))
            e = esp["test"]
            rem, _ = e.remapear([base["y_pred"][i] for i in e.indices])
            neg_rem = {i for i, p in enumerate(rem) if p == NEG}
            neg_fil = {i for i, p in enumerate(filt["y_pred"]) if p == NEG}
            assert neg_rem == neg_fil, (enc, s)
            f_r = f1_neg(e.y_true_full, rem)[0]
            f_f = f1_neg(filt["y_true"], filt["y_pred"])[0]
            assert f_r == f_f
            cruz[f"{enc}_seed{s}"] = round(f_r, 6)
    chk["6_remap_reproduz_filtro_fase2_f1_negation_of"] = cruz

    # 7. significance.py inalterado sobre a predicao ficticia remapeada
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        payload = {"model": "FICTICIO", "seed": 0, "labels": LABELS,
                   "y_true": esp["test"].y_true_full,
                   "y_pred": fict["test"]["_y_full"]}
        a = tmp / "ficticio.preds.json"
        a.write_text(json.dumps(payload), encoding="utf-8")
        b = res_dir / "baseline_bertimbau_seed42.preds.json"
        out = tmp / "sig.json"
        code = rodar_significance(a, b, out, 42)
        assert code == 0, code
        sig = json.loads(out.read_text(encoding="utf-8"))
        f_a = f1_neg(esp["test"].y_true_full, fict["test"]["_y_full"])[0]
        assert abs(sig["target_f1"]["a"] - f_a) < 1e-12
        assert sig["n_test"] == 19210
        chk["7_significance_py_inalterado"] = {
            "codigo_saida": code, "n_test": sig["n_test"],
            "f1_a_confere": True,
            "chaves_de_saida": sorted(sig),
            "nota": "predicao FICTICIA sorteada; numeros sem significado, so contrato"}

    reg["espaco_restrito"] = {s: e.resumo() for s, e in esp.items()}
    reg["pesos_classe_efetivos_restrito"] = dict(zip(LABELS, w_r))
    reg["resultado"] = "OK: 7 checagens"
    reg["duracao_s"] = round(time.time() - t0, 1)
    log.info("OK: 7 checagens do espaco restrito e do remapeamento (%.1fs)",
             reg["duracao_s"])
    if args.registro:
        p = Path(args.registro)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(reg, ensure_ascii=False, indent=2, sort_keys=True),
                     encoding="utf-8", newline="\n")
        log.info("Registro gravado em %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
