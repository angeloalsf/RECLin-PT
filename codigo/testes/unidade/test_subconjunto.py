"""Predições de um subconjunto levadas ao conjunto completo (`reclin.execucao.subconjunto`)."""
from __future__ import annotations

import pytest

from reclin.execucao import subconjunto
from reclin.tarefa import ASSOC, NEG, NOREL


def test_remapear_preds_e_probs():
    preds, probs = subconjunto.remapear([1, 3], 5, [NEG, ASSOC], [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    assert preds == [NOREL, NEG, NOREL, ASSOC, NOREL]
    assert probs == [[0.0, 0.0, 1.0], [0.7, 0.2, 0.1], [0.0, 0.0, 1.0], [0.1, 0.8, 0.1], [0.0, 0.0, 1.0]]
    assert subconjunto.remapear([1, 3], 5, [NEG, ASSOC])[1] is None
    assert subconjunto.remapear([], 2, []) == ([NOREL, NOREL], None)


@pytest.mark.parametrize("indices,n,preds,mensagem", [
    ([1, 1], 5, [0, 0], "fora de ordem ou repetidos"),
    ([3, 1], 5, [0, 0], "fora de ordem ou repetidos"),
    ([1, 5], 5, [0, 0], "fora de 0..4"),
    ([-1, 2], 5, [0, 0], "fora de 0..4"),
    ([1, 2], 5, [0], "1 predições para 2 posições"),
])
def test_remapear_recusa(indices, n, preds, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        subconjunto.remapear(indices, n, preds)


def test_remapear_recusa_probs_desalinhadas():
    with pytest.raises(ValueError, match="linhas de probs"):
        subconjunto.remapear([0], 2, [NEG], [[1, 0, 0], [1, 0, 0]])


def test_resumo():
    y = [NEG, NEG, ASSOC, NOREL, NOREL, NOREL]
    r = subconjunto.resumo([0, 3, 4], y)
    assert r == {"n_completo": 6, "n_restrito": 3, "fracao": 0.5,
                 "contagens_completo": {"negation_of": 2, "associated_with": 1, "no_relation": 3},
                 "contagens_restrito": {"negation_of": 1, "associated_with": 0, "no_relation": 2},
                 "teto_recall_negation_of": 0.5, "teto_recall_associated_with": 0.0,
                 "razao_no_relation_por_negation_of": 2.0}
    assert subconjunto.resumo([3], y)["razao_no_relation_por_negation_of"] is None
