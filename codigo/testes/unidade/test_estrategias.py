"""Estratégias sem GPU (`reclin.estrategias.filtro_pistas` e `regra_pura`)
sobre exemplos pequenos: as regras, o formato dos sidecars, os critérios de
desempate da calibração e a independência entre as estratégias."""
from __future__ import annotations

import ast
import inspect

import pytest

from reclin.avaliacao import metricas
from reclin.estrategias import filtro_pistas, regra_pura
from reclin.tarefa import LABELS, NEG, NOREL, candidatos, y_true

ASSOC = LABELS.index("associated_with")
LEX = {"nega": 3, "sem": 2}


def ent(id_, start, end, texto):
    return {"id": id_, "start": start, "end": end, "text": texto, "type": "x"}


def doc(doc_id, entidades, relacoes=()):
    return {"doc_id": doc_id, "text": "x" * 200, "entities": entidades, "relations": list(relacoes)}


def rel(e1, e2, tipo="negation_of"):
    return {"e1_id": e1, "e2_id": e2, "type": tipo}


# Documento 1: a pista "NEGA" com dois alvos, a 1 e a 11 caracteres; "tosse" e
# "febre" não são pistas. Documento 2: a pista "Sem" com dois alvos a 1
# caractere, um antes ("dor", início 10) e um depois ("edema", início 20).
DOCS = [
    doc("1", [ent("1", 0, 4, "NEGA"), ent("2", 5, 10, "tosse"), ent("3", 15, 20, "febre")],
        [rel("1", "2"), rel("1", "3")]),
    doc("2", [ent("10", 10, 15, "dor"), ent("4", 16, 19, "Sem"), ent("9", 20, 25, "edema")],
        [rel("4", "9")]),
]
CANDS = candidatos(DOCS)
Y_TRUE = y_true(CANDS)


def indice(doc_id, e1, e2):
    return next(i for i, c in enumerate(CANDS)
                if (c["doc_id"], c["e1"]["id"], c["e2"]["id"]) == (doc_id, e1, e2))


# --------------------------------------------------------------------------- #
# Filtro de pistas                                                            #
# --------------------------------------------------------------------------- #
def test_filtrar_rebaixa_so_negation_of_sem_pista():
    y = [NEG] * len(CANDS)
    y[indice("1", "3", "2")] = ASSOC
    filtrado = filtro_pistas.filtrar(CANDS, y, LEX)
    for i, c in enumerate(CANDS):
        if c["e1"]["text"] in ("NEGA", "Sem"):
            assert filtrado[i] == NEG                       # pista: fica
        elif y[i] == ASSOC:
            assert filtrado[i] == ASSOC                     # outro rótulo: fica
        else:
            assert filtrado[i] == NOREL                     # sem pista: rebaixado
    assert y[0] == NEG                                      # a entrada não muda


def test_filtrar_exige_o_mesmo_numero_de_predicoes():
    with pytest.raises(ValueError, match="não são deste conjunto"):
        filtro_pistas.filtrar(CANDS, [NEG], LEX)


def test_porta_gap():
    y = [NEG] * len(CANDS)
    com_porta = filtro_pistas.porta_gap(CANDS, y, 1)
    assert com_porta[indice("1", "1", "2")] == NEG          # gap 1
    assert com_porta[indice("1", "1", "3")] == NOREL        # gap 11
    assert filtro_pistas.porta_gap(CANDS, y, 25) == y       # a janela inteira: porta desligada


def test_nome_execucao():
    assert filtro_pistas.nome_execucao("baseline_biobertpt_seed42") == "filtro_biobertpt_seed42"
    assert filtro_pistas.nome_execucao("restrito_biobertpt_seed43") == "filtro_restrito_biobertpt_seed43"


def test_montar_predicoes_do_filtro():
    base = {"model": "enc/x", "seed": 7, "labels": list(LABELS), "y_true": Y_TRUE,
            "y_pred": [NEG] * len(CANDS), "probs": [[1.0, 0.0, 0.0]] * len(CANDS)}
    p = filtro_pistas.montar_predicoes(base, CANDS, LEX, base_preds="b.preds.json", min_freq=2)
    assert list(p) == ["model", "seed", "labels", "y_true", "y_pred", "base_preds", "postproc"]
    assert p["model"] == "filtro(enc/x s7)" and p["seed"] == 7 and p["base_preds"] == "b.preds.json"
    assert p["postproc"] == {"kind": "cue_filter", "min_freq": 2, "lexicon_size": 2, "max_gap": 25,
                             "demote_to": "no_relation", "calibrated_on": "dev"}
    assert p["y_pred"] == filtro_pistas.filtrar(CANDS, base["y_pred"], LEX)


def test_montar_predicoes_recusa_base_de_outro_conjunto():
    base = {"model": "m", "seed": 1, "labels": list(LABELS), "y_true": [NOREL] * len(CANDS),
            "y_pred": [NOREL] * len(CANDS)}
    with pytest.raises(ValueError, match="não pertence"):
        filtro_pistas.montar_predicoes(base, CANDS, LEX, base_preds="b", min_freq=1)


# --------------------------------------------------------------------------- #
# Regra pura                                                                  #
# --------------------------------------------------------------------------- #
def positivos(y):
    return {(CANDS[i]["doc_id"], CANDS[i]["e1"]["id"], CANDS[i]["e2"]["id"])
            for i, p in enumerate(y) if p == NEG}


def test_regras():
    r1 = regra_pura.prever(CANDS, LEX, "R1", 0)
    assert positivos(r1) == {("1", "1", "2"), ("1", "1", "3"),
                             ("2", "4", "10"), ("2", "4", "9")}
    # R2: um alvo por pista; em "Sem", os dois alvos estão a 1 caractere, vence
    # o de menor e2.start ("10", em 10 < 20)
    assert positivos(regra_pura.prever(CANDS, LEX, "R2", 0)) == {("1", "1", "2"), ("2", "4", "10")}
    assert positivos(regra_pura.prever(CANDS, LEX, "R3", 1)) == {("1", "1", "2"), ("2", "4", "10"),
                                                                  ("2", "4", "9")}
    assert positivos(regra_pura.prever(CANDS, LEX, "R3", 0)) == set()
    assert positivos(regra_pura.prever(CANDS, LEX, "R4", 1)) == {("1", "1", "2"), ("2", "4", "10")}
    assert all(p in (NEG, NOREL) for p in r1)              # nunca associated_with


def test_mais_proximo_desempata_por_id_inteiro():
    """Mesmo gap e mesmo início: vence o menor id como inteiro ("9" < "10")."""
    d = doc("3", [ent("1", 0, 3, "sem"), ent("10", 4, 8, "dor"), ent("9", 4, 9, "dores")])
    cands = candidatos([d])
    y = regra_pura.prever(cands, LEX, "R2", 25)
    assert [(c["e1"]["id"], c["e2"]["id"]) for c, p in zip(cands, y) if p == NEG] == [("1", "9")]


def test_regra_desconhecida():
    with pytest.raises(ValueError, match="regra desconhecida"):
        regra_pura.prever(CANDS, LEX, "R5", 1)


def test_montar_predicoes_da_regra():
    p = regra_pura.montar_predicoes(CANDS, LEX, regra="R3", max_target_gap=1, min_freq=2)
    assert list(p) == ["model", "seed", "labels", "y_true", "y_pred", "postproc"]
    assert p["model"] == "regra R3 (gap<=1, min_freq=2)" and p["seed"] is None
    assert p["y_true"] == Y_TRUE
    assert p["postproc"] == {"kind": "pure_rule", "rule": "R3", "max_target_gap": 1, "min_freq": 2,
                             "lexicon_size": 2, "max_gap": 25, "calibrated_on": "dev"}


# --------------------------------------------------------------------------- #
# Calibração: critério e desempates                                           #
# --------------------------------------------------------------------------- #
def test_resumo_alvo_pr_usa_a_formula_do_legado():
    r = metricas.resumo_alvo_pr([0, 0, 0, 2, 2, 1], [0, 2, 0, 0, 0, 1])
    assert (r["tp"], r["fp"], r["fn"]) == (2, 2, 1)
    assert r["f1"] == 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)
    assert metricas.f1_pr(0.0, 0.0) == 0.0


def test_calibrar_min_freq_empate_pelo_menor():
    # as duas formas cobrem os mesmos candidatos com min_freq 1 e 2: empate
    contagem = {"nega": 3, "sem": 2, "nunca": 1}
    preds = {"a": [NEG] * len(CANDS), "b": Y_TRUE}
    cal = filtro_pistas.calibrar_min_freq(contagem, CANDS, Y_TRUE, preds, min_freqs=(2, 1, 3))
    f1 = {r["min_freq"]: r["f1_medio"] for r in cal["varredura"]}
    assert f1[1] == f1[2] > f1[3] and cal["min_freq"] == 1
    assert [r["n_formas"] for r in cal["varredura"]] == [2, 3, 1]
    assert cal["varredura"][0]["cobertura"] == 1.0


def test_calibrar_regra_empate_pela_regra_e_pelo_gap():
    cal = regra_pura.calibrar(CANDS, Y_TRUE, LEX, gaps=(1, 25))
    f1 = {(r["regra"], r["gap"]): r["f1"] for r in cal["varredura"]}
    assert list(f1) == [("R1", 25), ("R2", 25), ("R3", 1), ("R3", 25), ("R4", 1), ("R4", 25)]
    melhor = max(f1.values())
    empatados = [k for k, v in f1.items() if v == melhor]
    assert (cal["regra"], cal["gap"]) == empatados[0]       # ordem de REGRAS, depois menor gap


def test_calibrar_porta_gap_empate_pelo_menor():
    preds = {"a": [NEG] * len(CANDS)}
    cal = filtro_pistas.calibrar_porta_gap(CANDS, Y_TRUE, preds, LEX, gaps=(25, 11, 0))
    f1 = {r["gap"]: r["f1_medio"] for r in cal["varredura"]}
    assert f1[11] == f1[25] and cal["gap"] == 11


# --------------------------------------------------------------------------- #
# Independência                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modulo,outra", [(filtro_pistas, "regra_pura"), (regra_pura, "filtro_pistas")])
def test_uma_estrategia_nao_importa_a_outra(modulo, outra):
    arvore = ast.parse(inspect.getsource(modulo))
    importados = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.ImportFrom):
            importados.add(no.module or "")
            importados.update(f"{no.module}.{a.name}" for a in no.names)
        elif isinstance(no, ast.Import):
            importados.update(a.name for a in no.names)
    assert not any(outra in nome for nome in importados)
    assert all(nome.startswith(("reclin.tarefa", "reclin.execucao", "reclin.avaliacao",
                                "reclin.negacao", "__future__", "math", "typing"))
               for nome in importados), importados


@pytest.mark.parametrize("modulo", [filtro_pistas, regra_pura])
def test_estrategia_nao_le_arquivos(modulo):
    """As estratégias recebem candidatos, predições e o léxico; quem lê as
    partições (e decide que a calibração só vê o DEV) são os scripts."""
    fonte = inspect.getsource(modulo)
    for proibido in ("open(", "particoes", "caminhos", "ler_json", "ler_jsonl", "read_"):
        assert proibido not in fonte, proibido
