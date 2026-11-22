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


# --------------------------------------------------------------------------- #
# Critério alternativo: maior precisão com piso de recall (configuração final) #
# --------------------------------------------------------------------------- #
def _linha_neg(t, tp, fp, fn):
    """Uma linha de tabela com as contagens de negation_of (as outras classes fixas)."""
    return {"limiar": t, "contagens": {"negation_of": {"tp": tp, "fp": fp, "fn": fn},
                                       "associated_with": {"tp": 5, "fp": 5, "fn": 5},
                                       "no_relation": {"tp": 50, "fp": 5, "fn": 5}}}


def test_configuracao_padrao_inalterada_pelo_criterio_opcional():
    cfg = limiar.configuracao_busca("r")
    assert "criterio_alternativo" not in cfg and cfg["criterio"] == limiar.CRITERIO
    assert limiar.hash_configuracao("r") == limiar.hash_configuracao("r", limiar.GRADE, None)
    alt = limiar.PrecisaoComPisoDeRecall(Fraction(4, 5))
    assert limiar.hash_configuracao("r", criterio=alt) != limiar.hash_configuracao("r")
    assert limiar.criterio_do_registro(limiar.configuracao_busca("r", criterio=alt)) == alt
    assert limiar.criterio_do_registro(cfg) is None


def test_precisao_com_piso_de_recall():
    alt = limiar.PrecisaoComPisoDeRecall(Fraction(4, 5))
    tabela = [_linha_neg(0.5, 131, 33, 19),      # R 0,873; P 0,799
              _linha_neg(0.75, 121, 28, 29),     # R 0,807; P 0,812
              _linha_neg(0.8, 120, 22, 30),      # R 0,800 (no piso); P 0,845
              _linha_neg(0.85, 115, 21, 35),     # R 0,767 (abaixo do piso); P 0,846
              _linha_neg(0.95, 95, 10, 55)]      # R 0,633; P 0,905
    assert limiar.escolhido_da_tabela(tabela, alt) == 0.8
    assert limiar.escolhido_da_tabela(tabela) == 0.5            # o critério padrão (F1) escolhe outro
    with pytest.raises(limiar.ErroLimiar, match="piso de recall"):
        limiar.escolhido_da_tabela([_linha_neg(0.9, 10, 0, 90)], alt)
    # empate em precisão e F1 -> maior macro-F1 -> menor limiar
    empate = [_linha_neg(0.6, 80, 20, 20), _linha_neg(0.4, 80, 20, 20)]
    assert limiar.escolhido_da_tabela(empate, alt) == 0.4
    with pytest.raises(ValueError, match="piso"):
        limiar.PrecisaoComPisoDeRecall(Fraction(0))


def test_congelar_e_conferir_com_o_criterio_alternativo(tmp_path):
    alt = limiar.PrecisaoComPisoDeRecall(Fraction(1, 2))
    y_true = [NEG, NEG, NEG, NEG, NOREL, NOREL, NOREL, ASSOC]
    probs = [0.9, 0.7, 0.6, 0.3, 0.65, 0.2, 0.1, 0.05]

    def prever(t):
        return [NEG if p >= t else NOREL for p in probs]

    t, tabela = limiar.buscar(y_true, prever, criterio=alt)
    assert t == 0.7                       # recall 1/2 com precisão 1 (0,65 < 0,70 deixa de ser FP)
    caminho = tmp_path / limiar.ARQUIVO
    limiar.congelar(caminho, limiar=t, regra="r", vinculos={"v": 1}, tabela=tabela, criterio=alt)
    gravado = ler_json(caminho)
    assert gravado["configuracao_busca"]["criterio_alternativo"] == {"nome": alt.NOME, "piso_recall": "1/2"}
    assert limiar.conferir_congelado(caminho, regra="r", vinculos={"v": 1}, criterio=alt) == 0.7
    with pytest.raises(limiar.ErroLimiar, match="configuração da busca"):
        limiar.conferir_congelado(caminho, regra="r", vinculos={"v": 1})          # sem o critério: recusa
