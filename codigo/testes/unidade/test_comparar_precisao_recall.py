"""Diferença pareada de precisão e recall de negation_of (script de diagnóstico)."""
import importlib.util

import numpy as np

from reclin.tarefa import NEG, NOREL
from reclin.util.caminhos import CODIGO

spec = importlib.util.spec_from_file_location("comparar_pr", CODIGO / "diagnosticos" / "comparar_precisao_recall.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_contagens_e_diferencas_observadas():
    y_true = [NEG, NEG, NEG, NOREL, NOREL, NOREL, NOREL, NEG]
    y_a = [NEG, NEG, NOREL, NOREL, NOREL, NOREL, NOREL, NEG]         # VP 3, FP 0, FN 1
    y_b = [NEG, NEG, NEG, NEG, NEG, NOREL, NOREL, NEG]               # VP 4, FP 2, FN 0
    r = script.comparar(y_true, y_a, y_b, ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"], n_boot=200)
    assert r["a_contagens"] == {"tp": 3, "fp": 0, "fn": 1} and r["b_contagens"] == {"tp": 4, "fp": 2, "fn": 0}
    assert np.isclose(r["por_documento"]["precision"]["diferenca"], 1.0 - 4 / 6)
    assert np.isclose(r["por_candidato"]["recall"]["diferenca"], 0.75 - 1.0)
    assert r["por_documento"]["n_unidades"] == 4 and r["por_candidato"]["n_unidades"] == 8


def test_execucoes_identicas_nao_tem_diferenca():
    y = [NEG, NOREL, NEG, NOREL]
    r = script.comparar(y, y, y, ["a", "a", "b", "b"], n_boot=100)
    for nivel in ("por_documento", "por_candidato"):
        for k in ("precision", "recall", "f1"):
            assert r[nivel][k]["diferenca"] == 0 and r[nivel][k]["ic95"] == [0.0, 0.0] and r[nivel][k]["leitura"] == "não detectada"
