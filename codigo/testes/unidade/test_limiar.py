"""Busca de limiar pré-registrada e congelamento (`avaliacao.limiar`), usada
pelo τ do Two-Stage e pelo t do baseline calibrado (A8, A19)."""
from __future__ import annotations

from fractions import Fraction

import pytest

from reclin.avaliacao import limiar
from reclin.tarefa import ASSOC, NEG, NOREL
from reclin.util.io import gravar_json, ler_json


def test_grade_pre_registrada():
    assert limiar.GRADE == tuple(round(0.05 * i, 2) for i in range(1, 20))
    assert len(limiar.GRADE) == 19 and limiar.GRADE[0] == 0.05 and limiar.GRADE[-1] == 0.95
    assert 0.5 in limiar.GRADE
    assert limiar.DESEMPATES == ("maior macro-F1 (contagens)", "menor |limiar - 0,5|", "maior limiar")


def test_pontuar_pelas_contagens():
    y_true = [NEG, NEG, ASSOC, NOREL, NOREL]
    y_pred = [NEG, NOREL, ASSOC, NEG, NOREL]
    p = limiar.pontuar(y_true, y_pred)
    assert p["contagens"]["negation_of"] == {"tp": 1, "fp": 1, "fn": 1}
    assert p["_f1_exato"] == Fraction(1, 2)
    assert p["_macro_exato"] == (Fraction(1, 2) + 1 + Fraction(1, 2)) / 3     # no_relation: TP 1, FP 1, FN 1


def _linha(t, f1, macro):
    return {"limiar": t, "_f1_exato": Fraction(f1), "_macro_exato": Fraction(macro)}


def test_criterio_e_desempates_na_ordem():
    # 1) maior F1 de negation_of
    assert limiar.escolher([_linha(0.3, "1/2", "1/3"), _linha(0.7, "3/5", "0")])["limiar"] == 0.7
    # 2) empate no F1 -> maior macro-F1
    assert limiar.escolher([_linha(0.3, "1/2", "2/3"), _linha(0.5, "1/2", "1/3")])["limiar"] == 0.3
    # 3) empate nos dois -> menor |t - 0,5|
    assert limiar.escolher([_linha(0.2, "1/2", "1/3"), _linha(0.6, "1/2", "1/3")])["limiar"] == 0.6
    # 4) ainda empatado (simétricos) -> maior t
    assert limiar.escolher([_linha(0.45, "1/2", "1/3"), _linha(0.55, "1/2", "1/3")])["limiar"] == 0.55


def test_comparacao_exata_sem_tolerancia():
    # 1/3 + 1/3 + 1/3 em float não é 1; em frações, é: o empate é reconhecido exatamente
    a = _linha(0.4, "1/2", Fraction(1, 3) + Fraction(1, 3) + Fraction(1, 3))
    b = _linha(0.5, "1/2", "1")
    assert limiar.escolher([a, b])["limiar"] == 0.5


def test_buscar_avalia_cada_ponto_da_grade():
    y_true = [NEG, NOREL, NOREL, NEG]
    escores = [0.9, 0.35, 0.2, 0.42]
    chamados = []

    def prever(t):
        chamados.append(t)
        return [NEG if e >= t else NOREL for e in escores]

    t, tabela = limiar.buscar(y_true, prever)
    assert chamados == list(limiar.GRADE) and len(tabela) == 19
    assert t == 0.4                      # pega as duas negações sem o falso positivo de 0,35
    assert all(not k.startswith("_") for linha in tabela for k in linha)


def test_congelar_e_conferir(tmp_path):
    y_true = [NEG, NOREL, NEG]
    t, tabela = limiar.buscar(y_true, lambda t: [NEG if p >= t else NOREL for p in (0.8, 0.3, 0.6)])
    vinculos = {"modelo": "abc", "dev": {"y_true_sha256": "x"}}
    caminho = tmp_path / limiar.ARQUIVO
    limiar.congelar(caminho, limiar=t, regra="r", vinculos=vinculos, tabela=tabela)
    assert limiar.conferir_congelado(caminho, regra="r", vinculos=vinculos) == t
    with pytest.raises(limiar.ErroLimiar, match="já existe"):
        limiar.congelar(caminho, limiar=t, regra="r", vinculos=vinculos, tabela=tabela)


@pytest.mark.parametrize("alteracao", ["vinculo", "regra", "limiar_editado", "tabela_editada", "ausente"])
def test_conferir_recusa_incompatibilidades(tmp_path, alteracao):
    y_true = [NEG, NOREL, NEG]
    t, tabela = limiar.buscar(y_true, lambda t: [NEG if p >= t else NOREL for p in (0.8, 0.3, 0.6)])
    vinculos = {"modelo": "abc"}
    caminho = tmp_path / limiar.ARQUIVO
    limiar.congelar(caminho, limiar=t, regra="r", vinculos=vinculos, tabela=tabela)
    regra = "r"
    if alteracao == "vinculo":
        vinculos = {"modelo": "outro"}
    elif alteracao == "regra":
        regra = "outra regra"
    elif alteracao == "limiar_editado":
        registro = ler_json(caminho)
        registro["limiar"] = 0.05
        gravar_json(caminho, registro)
    elif alteracao == "tabela_editada":
        registro = ler_json(caminho)
        registro["tabela"][0]["contagens"]["negation_of"]["tp"] += 5
        gravar_json(caminho, registro)
    else:
        caminho.unlink()
    with pytest.raises(limiar.ErroLimiar):
        limiar.conferir_congelado(caminho, regra=regra, vinculos=vinculos)


def test_congelar_recusa_limiar_que_a_regra_nao_escolhe(tmp_path):
    y_true = [NEG, NOREL, NEG]
    t, tabela = limiar.buscar(y_true, lambda t: [NEG if p >= t else NOREL for p in (0.8, 0.3, 0.6)])
    outro = next(g for g in limiar.GRADE if g != t)
    with pytest.raises(limiar.ErroLimiar, match="não é o que a regra escolhe"):
        limiar.congelar(tmp_path / limiar.ARQUIVO, limiar=outro, regra="r", vinculos={}, tabela=tabela)
    with pytest.raises(limiar.ErroLimiar, match="fora da grade"):
        limiar.congelar(tmp_path / limiar.ARQUIVO, limiar=0.33, regra="r", vinculos={}, tabela=tabela)
