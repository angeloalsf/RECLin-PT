"""Léxico de pistas de negação (`reclin.negacao.lexico`) sobre documentos pequenos."""
from __future__ import annotations

import hashlib
import json

import pytest

from reclin.negacao import lexico


@pytest.mark.parametrize("texto,forma", [
    ("NÃO", "nao"), ("NAO", "nao"), ("Não", "nao"),
    ("AUSÊNCIA", "ausencia"), ("ausência", "ausencia"),
    ("  Eliminação\t  fecal\n", "eliminacao fecal"),
    ("S/", "s/"), ("INDOLOR", "indolor"),
    ("ﬁm", "fim"),                    # NFKD também desfaz ligaduras
    ("STRASSE", "strasse"), ("Straße", "strasse"),   # casefold, não lower
])
def test_normalizar(texto, forma):
    assert lexico.normalizar(texto) == forma


def ent(id_, start, end, texto):
    return {"id": id_, "start": start, "end": end, "text": texto, "type": "x"}


def doc(doc_id, entidades, relacoes):
    return {"doc_id": doc_id, "text": "x" * 200, "entities": entidades, "relations": relacoes}


def neg(e1, e2):
    return {"e1_id": e1, "e2_id": e2, "type": "negation_of"}


# Três documentos: "NEGA" como e1 de negation_of 2 vezes, "Não" 1 vez, "SEM" 1
# vez, e uma negação cujo par fica fora da janela (não conta).
DOCS = [
    doc("1", [ent("a", 0, 4, "NEGA"), ent("b", 5, 10, "febre"), ent("c", 11, 14, "SEM"),
              ent("d", 15, 19, "dor")],
        [neg("a", "b"), neg("c", "d")]),
    doc("2", [ent("a", 0, 4, "nega"), ent("b", 5, 10, "tosse"), ent("c", 100, 103, "Não"),
              ent("d", 104, 108, "dor"), ent("e", 190, 195, "edema")],
        [neg("a", "b"), neg("c", "d"), neg("a", "e")]),          # a→e: 186 caracteres, fora
    doc("3", [ent("a", 0, 3, "não"), ent("b", 4, 8, "dor")],
        [{"e1_id": "b", "e2_id": "a", "type": "associated_with"}]),
]


def test_contar_formas_so_dos_candidatos_negation_of():
    assert lexico.contar_formas(DOCS) == {"nega": 2, "sem": 1, "nao": 1}
    # com uma janela maior, o par a→e do documento 2 vira candidato e conta
    assert lexico.contar_formas(DOCS, max_gap=200)["nega"] == 3


def test_induzir_aplica_o_limiar_e_ordena():
    assert lexico.induzir(DOCS, min_freq=1) == {"nega": 2, "nao": 1, "sem": 1}
    assert list(lexico.induzir(DOCS, min_freq=1)) == ["nega", "nao", "sem"]
    assert lexico.induzir(DOCS, min_freq=2) == {"nega": 2}
    assert lexico.induzir(DOCS, min_freq=3) == {}


def test_e_pista_normaliza_a_entidade():
    lex = lexico.induzir(DOCS, min_freq=1)
    assert lexico.e_pista(ent("x", 0, 3, "NÃO"), lex)
    assert lexico.e_pista(ent("x", 0, 3, "Nega"), {"nega"})          # aceita conjunto
    assert not lexico.e_pista(ent("x", 0, 3, "febre"), lex)


def test_lexico_sha1_cobre_formas_e_contagens():
    lex = {"sem": 2, "nao": 1}
    esperado = hashlib.sha1(json.dumps([["nao", 1], ["sem", 2]], ensure_ascii=False)
                            .encode("utf-8")).hexdigest()[:12]
    assert lexico.lexico_sha1(lex) == esperado
    assert lexico.lexico_sha1({"nao": 1, "sem": 2}) == esperado           # ordem não importa
    assert lexico.lexico_sha1({"nao": 1, "sem": 3}) != esperado           # contagem importa


def test_cobertura():
    assert lexico.cobertura(DOCS, {"nega"}) == {"negation_of": 4, "cobertos": 2, "cobertura": 0.5}
    assert lexico.cobertura(DOCS[2:], {"nega"}) == {"negation_of": 0, "cobertos": 0, "cobertura": None}


def test_carregar_congelado_confere_o_registro():
    lex = lexico.induzir(DOCS, min_freq=1)
    registro = {"min_freq": 1, "max_gap": 25, "n_formas": 3, "lexico_sha1": lexico.lexico_sha1(lex)}
    assert lexico.carregar_congelado(DOCS, registro) == lex
    with pytest.raises(ValueError, match="deixariam de ser comparáveis"):
        lexico.carregar_congelado(DOCS, dict(registro, lexico_sha1="000000000000"))
    with pytest.raises(ValueError):
        lexico.carregar_congelado(DOCS)                                   # o registro real


def test_registro_congelado():
    assert lexico.CONGELADO["min_freq"] == 3 and lexico.CONGELADO["max_gap"] == 25
    assert lexico.CONGELADO["n_formas"] == 11 and lexico.CONGELADO["lexico_sha1"] == "70c93fa807de"


def test_resultado_deterministico():
    """Mesma entrada, mesmo léxico, na mesma ordem e com o mesmo hash, em
    chamadas repetidas e com os documentos em outra ordem."""
    a = lexico.induzir(DOCS, min_freq=1)
    b = lexico.induzir(list(reversed(DOCS)), min_freq=1)
    assert a == b and list(a) == list(b) and lexico.lexico_sha1(a) == lexico.lexico_sha1(b)
    assert lexico.induzir(DOCS, min_freq=1) == a


def test_modulo_nao_le_arquivos():
    """O léxico só vê os documentos que recebe: o módulo não lê partições nem
    conhece caminhos, então quem chama decide que só o TRAIN entra."""
    import inspect
    fonte = inspect.getsource(lexico)
    for proibido in ("open(", "particoes", "caminhos", "ler_jsonl", "ler_json"):
        assert proibido not in fonte, proibido
