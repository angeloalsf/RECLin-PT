"""Projeção das decisões de adjudicação sobre outro ponto de operação."""
import pytest

from reclin.analises.adjudicacao_negation_of import CONTROLE_VP, FALSO_POSITIVO
from reclin.analises.adjudicacao_ponto_de_operacao import projetar
from reclin.tarefa import ASSOC, NEG, NOREL

#            0    1    2      3      4      5
Y_TRUE = [NEG, NEG, NOREL, NOREL, ASSOC, NOREL]
CASOS = [{"caso": "C1", "grupo": FALSO_POSITIVO, "indice_candidato": 2},
         {"caso": "C2", "grupo": FALSO_POSITIVO, "indice_candidato": 3},
         {"caso": "C3", "grupo": FALSO_POSITIVO, "indice_candidato": 4},
         {"caso": "C4", "grupo": CONTROLE_VP, "indice_candidato": 0}]
DECISOES = {"C1": {"principal": "1"}, "C2": {"principal": "2"}, "C3": {"principal": "1"}, "C4": {"principal": "1"}}


def test_execucao_adjudicada():
    r = projetar(CASOS, DECISOES, Y_TRUE, [NEG, NEG, NEG, NEG, NEG, NOREL])        # FP em 2, 3 e 4
    assert (r["oficial"]["tp"], r["oficial"]["fp"], r["oficial"]["fn"]) == (2, 3, 0)
    s = r["sob_referencia_adjudicada"]
    assert (s["tp"], s["fp"], s["fn"]) == (4, 1, 0)                                # 2 e 4 viram positivos
    assert r["fp_adjudicados"] == 3 and r["fp_fora_da_adjudicacao"] == 0
    assert r["fp_adjudicados_como_negacao_valida"] == 2 and r["negacoes_validas_adjudicadas_nao_previstas"] == 0


def test_outro_ponto_de_operacao_ganha_fn_e_mantem_fp_nao_adjudicado():
    r = projetar(CASOS, DECISOES, Y_TRUE, [NEG, NOREL, NEG, NOREL, NOREL, NEG])    # FP em 2 (válido) e 5 (fora)
    assert (r["oficial"]["tp"], r["oficial"]["fp"], r["oficial"]["fn"]) == (1, 2, 1)
    s = r["sob_referencia_adjudicada"]
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 2)                                # 4 era válido e não foi previsto
    assert r["fp_fora_da_adjudicacao"] == 1 and r["negacoes_validas_adjudicadas_nao_previstas"] == 1


def test_rastreabilidade_inconsistente_e_recusada():
    with pytest.raises(ValueError, match="rastreabilidade inconsistente"):
        projetar([{"caso": "C1", "grupo": FALSO_POSITIVO, "indice_candidato": 0}], DECISOES, Y_TRUE, Y_TRUE)
