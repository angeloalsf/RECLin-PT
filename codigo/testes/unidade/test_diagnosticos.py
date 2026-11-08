"""Diagnósticos da Etapa 2: controles por tipo semântico, quase duplicatas,
sensibilidade à semente e ao `max_gap`, e a exclusão de documentos preparada.

Duas partes:

* exemplos sintéticos, montados aqui com rótulos de marcação (`p1`, `c1`, …),
  sem texto clínico, para fixar o comportamento de cada função;
* os dados reais da fase (`dados/particoes-v2/`), só TRAIN e DEV nos controles,
  conferindo os números registrados em `docs/analises/controles-tipo-semantico.md`.
  Esses testes são pulados quando as partições não estão presentes.

Nenhum teste avalia o TEST, treina, ou altera partições, anotações ou predições.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from reclin import particoes
from reclin.analises import controles_anotacao as ca
from reclin.analises import exclusao_documentos as ex
from reclin.analises import max_gap, quase_duplicatas as qd, sensibilidade_semente as ss
from reclin.tarefa import LABEL2ID, NEG, NOREL, candidatos
from reclin.util import caminhos
from reclin.util.io import sha256_arquivo


def ent(i, start, end, tipo="Finding"):
    return {"id": i, "start": start, "end": end, "text": "x" * (end - start), "type": tipo}


def doc_sintetico(doc_id="1"):
    """Duas pistas e três conceitos, com uma negação anotada e uma associação.

    posições:  p1[0,3) c1[4,8)  c2[10,14)  p2[30,33) c3[34,38)
    p1 -> c1 negation_of (anotada); p2 -> c3 sem relação anotada (omissão);
    c1 -> c2 associated_with.
    """
    return {"doc_id": doc_id, "text": "y" * 40,
            "entities": [ent("p1", 0, 3, "Negation"), ent("c1", 4, 8), ent("c2", 10, 14),
                         ent("p2", 30, 33, "Negation|Abbreviation"), ent("c3", 34, 38)],
            "relations": [{"e1_id": "p1", "e2_id": "c1", "type": "negation_of"},
                          {"e1_id": "c1", "e2_id": "c2", "type": "associated_with"}]}


# --------------------------------------------------------------------------- #
# Controles por tipo                                                          #
# --------------------------------------------------------------------------- #
def test_tipos_le_o_campo_type_com_barra():
    assert ca.tipos(ent("a", 0, 1, "Negation|Abbreviation")) == {"Negation", "Abbreviation"}
    assert ca.e_pista_pelo_tipo(ent("a", 0, 1, "Abbreviation|Negation"))
    assert not ca.e_pista_pelo_tipo(ent("a", 0, 1, "Sign or Symptom"))


def test_controle_tipo_nivel_entidade_e_par():
    r = ca.controle_tipo([doc_sintetico()])
    e = r["nivel_entidade"]
    assert (e["tp"], e["fp"], e["fn"]) == (1, 1, 0)          # p1 acerta; p2 é pista sem relação
    assert e["precision"] == 0.5 and e["recall"] == 1.0
    p = r["nivel_par"]
    # candidatos com e1 Negation e e2 sem (distância simétrica <= 25): (p1,c1) (p1,c2)
    # (p2,c1) (p2,c2) (p2,c3); só (p1,c1) é negation_of
    assert (p["tp"], p["fp"], p["fn"]) == (1, 4, 0)


def test_alvo_seguinte_pula_pistas_e_entidades_anteriores():
    d = doc_sintetico()
    por_id = {e["id"]: e for e in d["entities"]}
    assert ca.alvo_seguinte(por_id["p1"], d["entities"])["id"] == "c1"
    assert ca.alvo_seguinte(por_id["p2"], d["entities"])["id"] == "c3"
    d["entities"].append(ent("p3", 39, 40, "Negation"))
    assert ca.alvo_seguinte(d["entities"][-1], d["entities"]) is None


def test_controle_tipo_posicao_conta_tp_fp_fn():
    d = doc_sintetico()
    r = ca.controle_tipo_posicao([d])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)          # (p1,c1) certo; (p2,c3) sem anotação
    assert r["contagens_da_regra"]["pistas"] == 2
    assert r["recall_sobre_anotadas"] == 1.0


def test_controle_tipo_posicao_respeita_a_janela():
    d = {"doc_id": "1", "text": "y" * 50,
         "entities": [ent("p1", 0, 3, "Negation"), ent("c1", 40, 44)],   # alvo a 37 caracteres da pista
         "relations": [{"e1_id": "p1", "e2_id": "c1", "type": "negation_of"}]}
    r = ca.controle_tipo_posicao([d], max_gap=25)
    assert r["contagens_da_regra"]["alvo_fora_da_janela"] == 1
    assert r["fn"] == 0 and r["tp"] == 0                       # o par anotado nem é candidato
    assert r["recall_sobre_anotadas"] == 0.0


def test_controles_alinhados_com_os_candidatos_da_tarefa():
    d = doc_sintetico()
    y_true, y_pred, _ = ca.predicoes_tipo_posicao([d])
    assert y_true == [LABEL2ID[c["label"]] for c in candidatos([d])]
    assert set(y_pred) <= {NEG, NOREL}


def test_controles_recusam_o_test_e_nao_mudam_os_documentos():
    d = doc_sintetico()
    antes = copy.deepcopy(d)
    with pytest.raises(ValueError, match="TEST"):
        ca.relatorio({"test": [d]})
    ca.relatorio({"dev": [d]})
    assert d == antes


# --------------------------------------------------------------------------- #
# Quase duplicatas                                                            #
# --------------------------------------------------------------------------- #
def _doc_texto(doc_id, texto):
    return {"doc_id": doc_id, "text": texto, "entities": [], "relations": []}


def test_detecta_copia_que_difere_so_em_espacos_e_nao_grava_texto():
    base = " ".join(f"tok{i}" for i in range(60))
    copia = base.replace(" tok5", "tok5").replace("tok9 ", "tok9  ")
    outro = " ".join(f"zz{i}" for i in range(60))
    r = qd.detectar({"train": [_doc_texto("1", base), _doc_texto("3", outro)], "test": [_doc_texto("2", copia)]})
    assert r["n_pares"] == 1
    p = r["pares"][0]
    assert {p["doc_a"], p["doc_b"]} == {"1", "2"} and p["entre_particoes"]
    assert p["identicos_sem_espacos"] and not p["identicos_byte_a_byte"]
    assert p["diferencas"]["so_espacos"]
    assert qd.documentos_de_test_com_copia_no_treino(r) == ["2"]
    assert "tok" not in json.dumps(r)


def test_detector_nao_altera_as_particoes():
    docs = {"train": [_doc_texto("1", "a b c " * 20)], "dev": [_doc_texto("2", "a b c " * 20)]}
    antes = copy.deepcopy(docs)
    qd.detectar(docs)
    assert docs == antes


def test_detector_recusa_doc_id_repetido_e_limiar_invalido():
    with pytest.raises(ValueError):
        qd.detectar({"train": [_doc_texto("1", "a")], "dev": [_doc_texto("1", "b")]})
    with pytest.raises(ValueError):
        qd.detectar({"train": []}, limiar=0)


def test_concordancia_conta_sem_texto():
    a, b = doc_sintetico("1"), doc_sintetico("2")
    b["relations"] = b["relations"][:1]
    c = qd.concordancia(a, b)
    assert c["relacoes"] == {"a": 2, "b": 1, "em_comum": 1}
    assert c["relacoes_por_tipo"]["negation_of"]["em_comum"] == 1


# --------------------------------------------------------------------------- #
# Sensibilidade à semente e ao max_gap                                        #
# --------------------------------------------------------------------------- #
def _corpus_sintetico(n=40):
    docs = []
    for i in range(n):
        d = doc_sintetico(str(1000 + i))
        if i % 3:
            d["relations"] = d["relations"][1:]                # sem negation_of
        docs.append(d)
    return docs


def test_sensibilidade_semente_reproduz_particionar():
    docs = _corpus_sintetico()
    r = ss.sensibilidade(docs, sementes=range(5))
    assert r["n_sementes"] == 6                                # 0..4 e a 42
    esperado = ss.contar(particoes.particionar(docs, seed=42))
    assert r["referencia"] == esperado
    assert r["resumo"]["dev"]["negation_of"]["semente_de_referencia"] == esperado["dev"]["negation_of"]


def test_distancias_anotadas_simetricas_e_contagem_acima_do_limite():
    d = doc_sintetico()
    d["relations"].append({"e1_id": "c3", "e2_id": "p2", "type": "negation_of"})   # invertida
    r = max_gap.distancias_anotadas([d], limites=(0, 1))
    assert r["negation_of"]["n"] == 2 and r["negation_of"]["ate_1"] == 2   # p1-c1 e c3-p2: distância 1
    assert r["associated_with"]["acima_de"]["1"] == 1                     # c1-c2: distância 2
    assert r["todas"]["n"] == 3


# --------------------------------------------------------------------------- #
# Exclusão de documentos (preparada para o TEST, não executada nele)          #
# --------------------------------------------------------------------------- #
def _sidecar(docs, acerta=True):
    cands = candidatos(docs)
    y_true = [LABEL2ID[c["label"]] for c in cands]
    return {"y_true": y_true, "y_pred": list(y_true) if acerta else [NOREL] * len(y_true)}


def test_metricas_excluindo_documento():
    docs = [doc_sintetico("1"), doc_sintetico("2")]
    pred = _sidecar(docs)
    pred["y_pred"] = [NOREL if i < len(candidatos(docs[:1])) else y for i, y in enumerate(pred["y_pred"])]
    r = ex.metricas_excluindo(pred, docs, particao="dev", excluir=["1"])
    assert r["completo"]["negation_of"]["fn"] == 1 and r["completo"]["negation_of"]["tp"] == 1
    assert r["sem_excluidos"]["negation_of"] == {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0,
                                                 "recall": 1.0, "f1": 1.0}
    assert r["contagens"]["excluidos"] == len(candidatos(docs[:1]))


def test_exclusao_recusa_test_sem_autorizacao_e_y_true_diferente():
    docs = [doc_sintetico("1"), doc_sintetico("2")]
    pred = _sidecar(docs)
    with pytest.raises(PermissionError):
        ex.metricas_excluindo(pred, docs, particao="test", excluir=["1"])
    ruim = dict(pred, y_true=[NOREL] * len(pred["y_true"]))
    with pytest.raises(ValueError, match="y_true"):
        ex.metricas_excluindo(ruim, docs, particao="dev", excluir=["1"])
    with pytest.raises(ValueError, match="não estão"):
        ex.metricas_excluindo(pred, docs, particao="dev", excluir=["999"])


def test_documentos_preparados_para_exclusao():
    assert ex.DOCUMENTOS_TEST_COM_COPIA_NO_TREINO == ("9534", "9535", "9536")


# --------------------------------------------------------------------------- #
# Script                                                                      #
# --------------------------------------------------------------------------- #
SCRIPT = caminhos.CODIGO / "diagnosticos" / "diagnosticar.py"


def test_script_responde_a_ajuda():
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()


def test_script_recusa_test_na_exclusao(tmp_path):
    pred = tmp_path / "p.json"
    pred.write_text("{}", encoding="utf-8")
    r = subprocess.run([sys.executable, str(SCRIPT), "excluir-documentos", "--predicoes", str(pred),
                        "--particao", "test", "--saida", str(tmp_path / "s.json")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "TEST recusado" in (r.stderr + r.stdout)
    assert not (tmp_path / "s.json").exists()


# --------------------------------------------------------------------------- #
# Dados reais da fase (TRAIN e DEV nos controles)                             #
# --------------------------------------------------------------------------- #
V2 = caminhos.PARTICOES_V2
com_dados = pytest.mark.skipif(not (V2 / "dev.jsonl").is_file(), reason="partições v2 ausentes")


@pytest.fixture(scope="module")
def reais():
    return {n: particoes.ler_particao(V2, n) for n in ("train", "dev")}


@com_dados
def test_controles_reais_dev(reais):
    r = ca.relatorio({"dev": reais["dev"]})["particoes"]["dev"]
    e = r["controle_tipo"]["nivel_entidade"]
    assert (e["tp"], e["fp"], e["fn"]) == (136, 94, 5)
    p = r["controle_tipo"]["nivel_par"]
    assert (p["tp"], p["fp"], p["fn"]) == (145, 751, 5)
    tp = r["controle_tipo_posicao"]
    assert (tp["tp"], tp["fp"], tp["fn"]) == (130, 92, 20)
    assert round(tp["f1"], 4) == 0.6989


@com_dados
def test_controles_reais_train(reais):
    r = ca.relatorio({"train": reais["train"]})["particoes"]["train"]
    e = r["controle_tipo"]["nivel_entidade"]
    assert (e["tp"], e["fp"], e["fn"]) == (1125, 716, 42)
    tp = r["controle_tipo_posicao"]
    assert (tp["tp"], tp["fp"], tp["fn"]) == (1089, 684, 166)


@com_dados
def test_controles_reais_nao_alteram_as_particoes(reais):
    antes = {n: sha256_arquivo(V2 / f"{n}.jsonl") for n in ("train", "dev")}
    ca.relatorio(reais)
    assert {n: sha256_arquivo(V2 / f"{n}.jsonl") for n in ("train", "dev")} == antes


@com_dados
@pytest.mark.lento
def test_quase_duplicatas_reais():
    docs = particoes.ler_particoes(V2)
    r = qd.detectar(docs)
    pares = {tuple(sorted((p["doc_a"], p["doc_b"]))) for p in r["pares"]}
    assert pares == {("9271", "9280"), ("9531", "9632"), ("9532", "9633"), ("9533", "9634"),
                     ("9534", "9635"), ("9535", "9636"), ("9536", "9637")}
    assert qd.documentos_de_test_com_copia_no_treino(r) == list(ex.DOCUMENTOS_TEST_COM_COPIA_NO_TREINO)
    assert not any(p["identicos_byte_a_byte"] for p in r["pares"])
    textos = [d["text"] for ds in docs.values() for d in ds if len(d["text"]) >= 30]
    saida = json.dumps(r, ensure_ascii=False)
    assert not any(t[:30] in saida for t in textos)


@com_dados
def test_origem_do_11_6_por_cento():
    docs = [d for ds in particoes.ler_particoes(V2).values() for d in ds]
    r = max_gap.distancias_anotadas(docs, limites=(20,))
    assert (r["todas"]["acima_de"]["20"], r["todas"]["n"]) == (1324, 11458)
    assert round(100 * r["todas"]["fracao_acima_de"]["20"], 1) == 11.6
