"""D4 — seleção do `min_freq` só com o TRAIN (`reclin.negacao.selecao_min_freq`
e `scripts/selecionar_min_freq.py`), sobre documentos sintéticos. A execução
registrada sobre o TRAIN real da v2 é conferida em `test_d4_registro.py`."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from fractions import Fraction

import pytest

from reclin import particoes
from reclin.negacao import lexico, selecao_min_freq as sel
from reclin.util import caminhos

SCRIPT = caminhos.CODIGO / "scripts" / "selecionar_min_freq.py"


def documento(doc_id: str, pista: str | None, alvo: str = "FEBRE", extra: str = "DOR") -> dict:
    """'<pista> <alvo> <extra>' com negation_of pista→alvo (se houver pista)."""
    partes = [p for p in (pista, alvo, extra) if p]
    texto, entidades, pos = "", [], 0
    for i, p in enumerate(partes):
        entidades.append({"id": str(i), "start": pos, "end": pos + len(p), "text": p, "type": "x"})
        texto += p + " "
        pos += len(p) + 1
    rel = [{"e1_id": "0", "e2_id": "1", "type": "negation_of"}] if pista else []
    return {"doc_id": doc_id, "text": texto.strip(), "entities": entidades, "relations": rel}


def corpus(n: int = 40, semente: int = 3) -> list[dict]:
    rng = random.Random(semente)
    formas = ["SEM"] * 10 + ["NEGA"] * 6 + ["AUSENCIA"] * 2 + ["RARA"]
    return [documento(str(100 + i), rng.choice(formas) if rng.random() < 0.7 else None) for i in range(n)]


def test_partes_deterministicas_estratificadas_e_disjuntas():
    docs = corpus()
    partes = sel.atribuir_partes(docs)
    assert partes == sel.atribuir_partes(list(reversed(docs)))           # independe da ordem de entrada
    assert set(partes) == {d["doc_id"] for d in docs} and set(partes.values()) == set(range(5))
    com = [sum(1 for d in docs if d["relations"] and partes[d["doc_id"]] == k) for k in range(5)]
    tamanhos = [sum(1 for v in partes.values() if v == k) for k in range(5)]
    assert max(com) - min(com) <= 1 and max(tamanhos) - min(tamanhos) <= 2
    assert sel.atribuir_partes(docs, semente=7) != partes
    with pytest.raises(ValueError, match="repetido"):
        sel.atribuir_partes(docs + docs[:1])


def test_contagem_do_detector():
    docs = [documento("1", "SEM"), documento("2", "NEGA"), documento("3", None)]
    c = sel.contar(docs, {"sem"})
    # doc 1: par (SEM,FEBRE) VP; (SEM,DOR) FP. doc 2: (NEGA,FEBRE) FN
    assert c == {"vp": 1, "fp": 1, "fn": 1}
    assert sel.f1(c) == Fraction(2, 4)
    assert sel.f1({"vp": 0, "fp": 0, "fn": 0}) == 0


def test_escolha_pelo_f1_exato_e_desempate_pelo_maior():
    r = {1: {"vp": 2, "fp": 2, "fn": 0}, 3: {"vp": 1, "fp": 0, "fn": 1}, 5: {"vp": 1, "fp": 0, "fn": 1}}
    assert sel.f1(r[1]) == sel.f1(r[3]) == Fraction(2, 3)
    assert sel.escolher(r) == 5
    r[2] = {"vp": 3, "fp": 0, "fn": 1}
    assert sel.escolher(r) == 2


def test_lexico_de_cada_parte_nao_ve_a_parte_retida(monkeypatch):
    """O léxico que avalia uma parte é induzido só das outras; o final, do TRAIN inteiro."""
    docs = corpus(40)
    partes = sel.atribuir_partes(docs)
    chamadas, contadas = [], []
    original_induzir, original_contar = lexico.induzir, sel.contar
    monkeypatch.setattr(lexico, "induzir", lambda d, **kw: chamadas.append({x["doc_id"] for x in d})
                        or original_induzir(d, **kw))
    monkeypatch.setattr(sel, "contar", lambda d, lex, mg: contadas.append({x["doc_id"] for x in d})
                        or original_contar(d, lex, mg))
    sel.selecionar(docs)
    assert len(chamadas) == 5 * 5 + 1 and len(contadas) == 25
    for induzidos, avaliados in zip(chamadas, contadas):
        assert induzidos.isdisjoint(avaliados) and induzidos | avaliados == set(partes)
        assert len({partes[i] for i in avaliados}) == 1
    assert chamadas[-1] == set(partes)


def test_selecao_completa_e_reprodutivel():
    docs = corpus(60)
    r = sel.selecionar(docs)
    assert r["candidatos"] == [1, 2, 3, 5, 10] and r["n_partes"] == 5 and r["semente"] == 42
    assert r["min_freq_escolhido"] in r["candidatos"] and r["ordem_por_f1"][0] == r["min_freq_escolhido"]
    assert sum(r["documentos_por_parte"]) == 60
    totais = {int(m): v["total"] for m, v in r["resultados"].items()}
    for m, t in totais.items():
        partes = r["resultados"][str(m)]["partes"]
        assert t["vp"] == sum(p["vp"] for p in partes) and t["fn"] == sum(p["fn"] for p in partes)
        assert t["vp"] + t["fn"] == totais[1]["vp"] + totais[1]["fn"]     # positivos não dependem do léxico
    melhor = max(Fraction(*map(int, v["total"]["f1_fracao"].split("/"))) for v in r["resultados"].values())
    assert Fraction(*map(int, r["resultados"][str(r["min_freq_escolhido"])]["total"]["f1_fracao"].split("/"))) == melhor
    final = lexico.induzir(docs, min_freq=r["min_freq_escolhido"])
    assert r["lexico_final"]["lexico_sha1"] == lexico.lexico_sha1(final)
    assert json.dumps(sel.selecionar(docs), sort_keys=True) == json.dumps(r, sort_keys=True)


def test_codigo_segue_o_pre_registro():
    sys.path.insert(0, str(caminhos.CODIGO / "scripts"))
    try:
        import selecionar_min_freq as script
    finally:
        sys.path.pop(0)
    pre = script.registro_do_pre_registro()
    assert script.conferir_constantes(pre) == []
    assert script.conferir_constantes(dict(pre, candidatos_min_freq=[1, 2, 3])) != []


def test_script_recusa_pasta_que_nao_e_a_v2(tmp_path):
    docs = {"train": corpus(10), "dev": [], "test": []}
    particoes.gravar_particoes(docs, tmp_path)
    particoes.gravar_manifesto(tmp_path, particoes.calcular_manifesto(tmp_path, seed=42,
                                                                      origem={"path": "x", "sha256": None}))
    r = subprocess.run([sys.executable, str(SCRIPT), "--particoes", str(tmp_path), "--saida",
                        str(tmp_path / "s.json")], capture_output=True, text=True)
    assert r.returncode == 2 and "v2" in (r.stdout + r.stderr)
    assert not (tmp_path / "s.json").exists()
