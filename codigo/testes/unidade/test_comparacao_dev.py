"""`reclin.experimento.comparacao_dev`: só sidecars sintéticos em memória/tmp; sem rede, GPU, pesos, treino, TEST nem Git.
Escrito junto do módulo e **não executado** pelo autor da etapa."""
from __future__ import annotations

import pytest

from reclin.execucao import predicoes
from reclin.experimento import comparacao_dev as cd


def _sc(y_true, y_pred):
    return predicoes.montar(model="m", seed=42, y_true=y_true, y_pred=y_pred)


def test_compara_mesmo_universo_e_calcula_por_classe():
    yt = [0, 0, 1, 2, 2, 2]
    r = cd.comparar({"a": _sc(yt, [0, 0, 2, 2, 2, 2]), "b": _sc(yt, [0, 2, 1, 2, 2, 0])})
    assert r["n"] == 6 and [l["execucao"] for l in r["linhas"]] == ["a", "b"]
    a = r["linhas"][0]
    assert a["negation_of.precisao"] == 1.0 and a["negation_of.recall"] == 1.0
    assert a["associated_with.f1"] == 0.0 and a["no_relation.suporte"] == 3
    assert abs(a["acuracia"] - 5 / 6) < 1e-12


def test_recusa_universos_diferentes():
    with pytest.raises(cd.ErroComparacao):
        cd.comparar({"a": _sc([0, 1, 2], [0, 1, 2]), "b": _sc([0, 1, 1], [0, 1, 2])})
    with pytest.raises(cd.ErroComparacao):
        cd.comparar({"a": _sc([0, 1, 2], [0, 1, 2]), "b": _sc([0, 1], [0, 1])})


def test_ausentes_sao_listados_e_nao_inventados(tmp_path):
    r = cd.carregar_e_comparar([tmp_path], ["nao_existe"])
    assert r["linhas"] == [] and r["ausentes"] == ["nao_existe"]
    assert "NÃO ENCONTRADAS" in cd.formatar(r)


def test_particao_fixa_em_dev():
    assert cd.PARTICAO == "dev"
