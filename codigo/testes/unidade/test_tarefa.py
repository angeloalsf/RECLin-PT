"""Rótulos e enumeração de candidatos (`reclin.tarefa`) sobre um documento pequeno."""
from __future__ import annotations

import inspect

import pytest

from reclin import tarefa

# "nega febre. tosse ... dor"
#  0   4 5   10 12  17      40 43
TEXTO = "nega febre. tosse" + " " * 23 + "dor"


def entidade(id_, start, end, tipo="Sign or Symptom"):
    return {"id": id_, "start": start, "end": end, "text": TEXTO[start:end], "type": tipo}


def documento(relacoes=None, entidades=None):
    return {
        "doc_id": "1",
        "text": TEXTO,
        "entities": entidades if entidades is not None else [
            entidade("3", 12, 17),          # tosse
            entidade("1", 0, 4, "Negation"),  # nega
            entidade("2", 5, 10),            # febre
            entidade("4", 40, 43),           # dor (longe)
        ],
        "relations": relacoes if relacoes is not None else [
            {"e1_id": "1", "e2_id": "2", "type": "negation_of"},
            {"e1_id": "3", "e2_id": "2", "type": "associated_with"},
        ],
    }


def pares(cands):
    return [(c["e1"]["id"], c["e2"]["id"], c["label"]) for c in cands]


def test_rotulos_na_ordem_do_contrato():
    assert tarefa.LABELS == ("negation_of", "associated_with", "no_relation")
    assert (tarefa.NEG, tarefa.ASSOC, tarefa.NOREL) == (0, 1, 2)
    assert all(tarefa.ID2LABEL[tarefa.LABEL2ID[r]] == r for r in tarefa.LABELS)


def test_entity_gap():
    nega, febre, tosse = entidade("1", 0, 4), entidade("2", 5, 10), entidade("3", 12, 17)
    assert tarefa.entity_gap(nega, febre) == 1
    assert tarefa.entity_gap(febre, nega) == 1
    assert tarefa.entity_gap(nega, tosse) == 8
    assert tarefa.entity_gap(entidade("a", 0, 4), entidade("b", 4, 8)) == 0   # encostadas
    assert tarefa.entity_gap(entidade("a", 0, 6), entidade("b", 2, 8)) == 0   # sobrepostas


def test_pares_ordenados_com_rotulo_direcional():
    cands = list(tarefa.iter_candidate_pairs(documento(), max_gap=25))
    rotulos = {(a, b): r for a, b, r in pares(cands)}
    assert rotulos[("1", "2")] == "negation_of"
    assert rotulos[("2", "1")] == "no_relation"          # a direção importa
    assert rotulos[("3", "2")] == "associated_with"
    assert all(c["doc_id"] == "1" for c in cands)


def test_ordem_por_start_end_id_e_janela():
    cands = list(tarefa.iter_candidate_pairs(documento(), max_gap=25))
    # entidades ordenadas: nega(1), febre(2), tosse(3), dor(4); dor fica a 23
    # caracteres de tosse, a 30 de febre e a 36 de nega
    assert [(a, b) for a, b, _ in pares(cands)] == [
        ("1", "2"), ("1", "3"),
        ("2", "1"), ("2", "3"),
        ("3", "1"), ("3", "2"), ("3", "4"),
        ("4", "3"),
    ]


def test_max_gap_limita_a_distancia():
    assert len(list(tarefa.iter_candidate_pairs(documento(), max_gap=0))) == 0
    assert len(list(tarefa.iter_candidate_pairs(documento(), max_gap=1))) == 2
    assert len(list(tarefa.iter_candidate_pairs(documento(), max_gap=36))) == 12


def test_sem_auto_par_nem_span_identico():
    ents = [entidade("1", 0, 4), entidade("9", 0, 4), entidade("2", 5, 10)]
    cands = list(tarefa.iter_candidate_pairs(documento(relacoes=[], entidades=ents)))
    assert ("1", "9", "no_relation") not in pares(cands)
    assert ("9", "1", "no_relation") not in pares(cands)
    assert all(c["e1"] is not c["e2"] for c in cands)
    assert len(cands) == 4


def test_empate_de_span_desfeito_pelo_id():
    """O SemClinBr tem 40 entidades com span repetido; nas partições elas já vêm
    em ordem de id, então só este teste garante o desempate."""
    ents = [entidade("9", 0, 4), entidade("1", 0, 4), entidade("2", 5, 10)]
    cands = list(tarefa.iter_candidate_pairs(documento(relacoes=[], entidades=ents)))
    assert [(a, b) for a, b, _ in pares(cands)] == [("1", "2"), ("9", "2"), ("2", "1"), ("2", "9")]


def test_default_de_max_gap_e_o_dos_experimentos():
    padrao = inspect.signature(tarefa.iter_candidate_pairs).parameters["max_gap"].default
    assert padrao == tarefa.MAX_GAP == 25


def test_y_true_e_contagem():
    cands = tarefa.candidatos([documento()], max_gap=25)
    assert tarefa.y_true(cands)[:2] == [tarefa.NEG, tarefa.NOREL]
    assert tarefa.contagem_por_rotulo(cands) == {
        "negation_of": 1, "associated_with": 1, "no_relation": 6}
    assert tarefa.contagem_por_rotulo([]) == dict.fromkeys(tarefa.LABELS, 0)


def test_conjunto_referencia_confere_os_tamanhos_congelados():
    with pytest.raises(ValueError, match="19064"):
        tarefa.conjunto_referencia("dev", [documento()], particao_sha256="x")
    # fora das partições congeladas, ou com outra janela, não há tamanho esperado
    assert tarefa.conjunto_referencia("outra", [documento()], particao_sha256="x")["n_candidatos"] == 8
    assert tarefa.conjunto_referencia("dev", [documento()], particao_sha256="x",
                                      max_gap=10)["n_candidatos"] == 6


def test_identidade_muda_com_o_rotulo_mas_nao_com_o_numero():
    a = tarefa.conjunto_referencia("outra", [documento()], particao_sha256="x")
    outra = documento(relacoes=[{"e1_id": "2", "e2_id": "1", "type": "negation_of"}])
    b = tarefa.conjunto_referencia("outra", [outra], particao_sha256="x")
    assert a["n_candidatos"] == b["n_candidatos"]
    assert a["candidatos_sha256"] != b["candidatos_sha256"]
    assert a["y_true_sha256"] != b["y_true_sha256"]
