"""Auditoria das execuções gravadas (Etapa 3): `reclin.analises.auditoria_execucoes`.

Tudo sintético, montado aqui em pastas temporárias, sem texto clínico, sem
modelo e sem o TEST. Fixa:

* o recálculo das métricas e a conferência com `metricas.json`;
* a equivalência do bootstrap por candidato com `significancia.comparar`
  (mesma sequência de reamostras, mesmo resultado no F1 de `negation_of`);
* o bootstrap por documento (reamostra documentos inteiros);
* o vínculo das composições do Two-Stage com os estágios pelos hashes, mesmo
  quando o nome registrado no limiar difere do nome da pasta;
* a origem dos falsos positivos de `associated_with` (erro de direção);
* a AP igual à do `sklearn` em casos com empates.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from reclin.analises import auditoria_execucoes as aud
from reclin.avaliacao import metricas as met
from reclin.avaliacao import significancia
from reclin.tarefa import ASSOC, LABELS, NEG, NOREL
from reclin.util.io import sha256_json


def _gravar(pasta, nome, obj):
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / nome).write_text(json.dumps(obj), encoding="utf-8")


def _preds(n=400, seed=0, erro=0.2):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.1, 0.2, 0.7])
    p = np.where(rng.random(n) < erro, rng.integers(0, 3, n), y)
    return y.tolist(), p.tolist()


def _execucao(pasta, y, p, *, com_metricas=True, probs=None):
    sidecar = {"labels": list(LABELS), "y_true": y, "y_pred": p, "model": "m", "seed": 42}
    if probs is not None:
        sidecar["probs"] = probs
    _gravar(pasta, "predicoes_dev.json", sidecar)
    if com_metricas:
        m = met.avaliar(y, p)
        _gravar(pasta, "metricas.json", {"dev": {"classification_report": m["classification_report"],
                                                 "confusion_matrix": m["confusion_matrix"]}})


def test_metricas_recalculadas_conferem(tmp_path):
    y, p = _preds()
    _execucao(tmp_path / "a", y, p)
    r = aud.metricas_sistema(tmp_path / "a", {"y_true_sha256": sha256_json(y)})
    assert r["confere_metricas_json"] and r["confere_matriz_gravada"] and r["confere_y_true_sha256"]
    neg = r["por_classe"]["negation_of"]
    tp = sum(1 for a, b in zip(y, p) if a == b == NEG)
    assert neg["tp"] == tp
    assert neg["f1"] == pytest.approx(2 * tp / (2 * tp + neg["fp"] + neg["fn"]))


def test_metricas_detecta_divergencia(tmp_path):
    y, p = _preds()
    _execucao(tmp_path / "a", y, p)
    q = list(p)
    q[0] = (q[0] + 1) % 3                       # o sidecar muda depois das métricas gravadas
    _gravar(tmp_path / "a", "predicoes_dev.json", {"labels": list(LABELS), "y_true": y, "y_pred": q})
    assert aud.metricas_sistema(tmp_path / "a")["confere_metricas_json"] is False


def test_bootstrap_por_candidato_igual_ao_do_protocolo():
    y, a = _preds(seed=1)
    _, b = _preds(seed=2)
    r = aud.comparar_pareado(y, a, b, n_boot=300)
    ref = significancia.comparar({"labels": list(LABELS), "y_true": y, "y_pred": a},
                                 {"labels": list(LABELS), "y_true": y, "y_pred": b}, seed=42, n_boot=300)
    neg = r["por_candidato"]["f1_negation_of"]
    pb = ref["paired_bootstrap"]
    assert neg["diferenca"] == pb["observed_diff"]
    assert neg["media_bootstrap"] == pb["mean_diff"]
    assert neg["ic95"] == [pb["ci95_low"], pb["ci95_high"]]
    assert neg["p_bilateral"] == pb["p_value"]
    assert r["mcnemar"] == ref["mcnemar"]


def test_bootstrap_por_documento():
    y, a = _preds(seed=3)
    _, b = _preds(seed=4)
    grupos = [str(i // 20) for i in range(len(y))]          # 20 documentos de 20 candidatos
    r = aud.comparar_pareado(y, a, b, grupos=grupos, n_boot=200)
    assert r["n_documentos"] == 20
    for k in aud.METRICAS_COMPARADAS:
        d, c = r["por_documento"][k], r["por_candidato"][k]
        assert d["diferenca"] == c["diferenca"]           # o observado é o mesmo
        assert d["ic95"][0] <= d["diferenca"] <= d["ic95"][1] or d["diferenca"] == 0


def test_bootstrap_identicos_nao_detecta():
    y, a = _preds(seed=5)
    r = aud.comparar_pareado(y, a, a, grupos=[str(i % 7) for i in range(len(y))], n_boot=100)
    for k in aud.METRICAS_COMPARADAS:
        assert r["por_candidato"][k]["diferenca"] == 0.0
        assert r["por_documento"][k]["diferenca_na_seed42"] == "não detectada"
    assert r["mcnemar"]["p_value"] == 1.0


def test_tamanhos_diferentes_recusados():
    with pytest.raises(ValueError):
        aud.comparar_pareado([0, 1], [0, 1, 2], [0, 1])


def test_average_precision_como_sklearn():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.9, 0.8, 0.5, 0.5, 0.4, 0.4, 0.1])
    # valor do sklearn.metrics.average_precision_score para este caso (com empates)
    precisao = [1 / 2, 2 / 3, 3 / 5, 4 / 7, 4 / 8]
    recall = [1 / 4, 2 / 4, 3 / 4, 4 / 4, 4 / 4]
    esperado = sum((r - r0) * p for r, r0, p in zip(recall, [0] + recall[:-1], precisao))
    assert aud.average_precision(y, s) == pytest.approx(esperado)
    assert aud.average_precision(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3])) == 0.0


def test_origem_falsos_positivos_assoc():
    pares = [("d", "a", "b"), ("d", "b", "a"), ("d", "a", "c"), ("d", "p", "a")]
    y = [ASSOC, NOREL, NOREL, NEG]
    p = [ASSOC, ASSOC, ASSOC, ASSOC]
    r = aud.origem_falsos_positivos_assoc(y, p, pares, gaps=[0, 0, 12, 1])
    assert r["fp_total"] == 3
    assert r["fp_por_origem"] == {"verdadeiro_negation_of": 1, "inverso_associated_with": 1,
                                  "inverso_negation_of": 0, "sem_relacao_nas_duas_direcoes": 1}
    assert r["por_distancia"]["0-1"]["tp"] == 1 and r["por_distancia"]["0-1"]["fp"] == 2
    assert r["por_distancia"]["11-25"]["fp"] == 1


def _estagio(pasta, estrategia, nome, melhor, hash_modelo):
    config = {"config": {"config": {"seed": 42}, "estrategia": estrategia,
                         "proveniencia_pesos": {"repositorio": "neuralmind/bert-base-portuguese-cased"}},
              "estrategia": estrategia, "nome": nome}
    _gravar(pasta, "config.json", config)
    _gravar(pasta, "treino.json", {"concluido": True, "melhor_epoca": melhor, "campo_selecao": "x",
                                   "dev_history": [{"x": 0.1}, {"x": 0.3}][:melhor],
                                   "final": {"sha256_melhor_modelo": hash_modelo}})
    return aud._hash_obj(config["config"])


def test_vinculo_two_stage_pelos_hashes(tmp_path):
    h1 = _estagio(tmp_path / "e1_renomeado", "two_stage_e1", "e1_renomeado", 2, "m1")
    h2 = _estagio(tmp_path / "e2_x", "two_stage_e2", "e2_x", 1, "m2")
    _gravar(tmp_path / "comp", "limiar.json", {
        "limiar": 0.5, "tabela": [],
        "vinculos": {"seed": 42, "proveniencia_pesos": {"repositorio": "neuralmind/bert-base-portuguese-cased"},
                     "estagio1": {"nome": "e1_nome_antigo", "sha256_config": h1,
                                  "sha256_melhor_modelo": "m1", "melhor_epoca": 2},
                     "estagio2": {"nome": "e2_x", "sha256_config": h2,
                                  "sha256_melhor_modelo": "m2", "melhor_epoca": 1}}})
    inv = aud.inventariar(tmp_path)
    (v,) = inv["vinculos_two_stage"]
    assert v["estagio1"]["pastas"] == ["e1_renomeado"] and v["estagio1"]["nome_confere"] is False
    assert v["estagio1"]["melhor_epoca_confere"]
    assert v["estagio2"]["nome_confere"] and v["estagio2"]["melhor_epoca_confere"]
    tipos = {r["pasta"]: r["tipo"] for r in inv["execucoes"]}
    assert tipos == {"comp": "composicao_two_stage", "e1_renomeado": "estagio_two_stage",
                     "e2_x": "estagio_two_stage"}
    comp = next(r for r in inv["execucoes"] if r["pasta"] == "comp")
    assert comp["encoder"] == "BERTimbau" and comp["estrategia"] == "two_stage"


def test_tabela_tau():
    lim = {"tabela": [{"limiar": 0.5, "n_roteados": 10, "f1_negation_of": 0.8, "macro_f1": 0.7,
                       "contagens": {"associated_with": {"tp": 3, "fp": 1, "fn": 6},
                                     "negation_of": {"tp": 4, "fp": 1, "fn": 1},
                                     "no_relation": {"tp": 9, "fp": 1, "fn": 1}}}]}
    (linha,) = aud.tabela_tau(lim)
    assert linha["assoc_precision"] == 0.75 and linha["assoc_recall"] == pytest.approx(1 / 3)
