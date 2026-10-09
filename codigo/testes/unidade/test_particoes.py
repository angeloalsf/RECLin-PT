"""Particionamento por documento e MANIFEST (`reclin.particoes`) sobre dados sintéticos."""
from __future__ import annotations

import random

import pytest

from reclin import particoes


def docs_sinteticos(n_com_negacao=10, n_sem=20):
    docs = []
    for i in range(n_com_negacao + n_sem):
        tipo = "negation_of" if i < n_com_negacao else "associated_with"
        docs.append({"doc_id": str(100 + i), "text": f"documento {i}", "entities": [],
                     "relations": [{"e1_id": "1", "e2_id": "2", "type": tipo}]})
    return docs


def ids(lista):
    return [d["doc_id"] for d in lista]


def test_80_10_10_dentro_de_cada_grupo():
    divisao = particoes.particionar(docs_sinteticos())
    tamanhos = {nome: len(docs) for nome, docs in divisao.items()}
    assert tamanhos == {"train": 8 + 16, "dev": 1 + 2, "test": 1 + 2}
    for nome in particoes.PARTICOES:
        assert any(particoes._tem_negacao(d) for d in divisao[nome]), nome


def test_particoes_disjuntas_completas_e_ordenadas():
    docs = docs_sinteticos()
    divisao = particoes.particionar(docs)
    todos = [i for nome in particoes.PARTICOES for i in ids(divisao[nome])]
    assert sorted(todos) == sorted(ids(docs))
    assert len(set(todos)) == len(todos)
    for nome in particoes.PARTICOES:
        assert ids(divisao[nome]) == sorted(ids(divisao[nome]), key=int)


def test_deterministico_e_independente_da_ordem_de_entrada():
    docs = docs_sinteticos()
    embaralhados = docs[:]
    random.Random(7).shuffle(embaralhados)
    assert particoes.particionar(docs) == particoes.particionar(embaralhados)
    assert particoes.particionar(docs, seed=42) != particoes.particionar(docs, seed=43)


def test_doc_id_repetido_falha():
    docs = docs_sinteticos()
    with pytest.raises(ValueError, match="repetido"):
        particoes.particionar(docs + [dict(docs[0])])


def test_particao_desconhecida_falha(tmp_path):
    with pytest.raises(ValueError, match="desconhecida"):
        particoes.arquivo_particao(tmp_path, "validacao")


def test_manifesto_ida_e_volta(tmp_path):
    particoes.gravar_particoes(particoes.particionar(docs_sinteticos()), tmp_path)
    origem = {"path": "dados/processados/dataset.jsonl", "sha256": "abc"}
    manifesto = particoes.calcular_manifesto(tmp_path, seed=42, origem=origem)
    assert manifesto["generator"] == particoes.GERADOR
    assert manifesto["source"] == origem
    assert manifesto["splits"]["dev"]["n_records"] == 3
    assert manifesto["splits"]["train"]["n_relations"] == {"associated_with": 16, "negation_of": 8}

    caminho = particoes.gravar_manifesto(tmp_path, manifesto)
    bytes_1 = caminho.read_bytes()
    particoes.gravar_manifesto(tmp_path, particoes.calcular_manifesto(tmp_path, seed=42, origem=origem))
    assert caminho.read_bytes() == bytes_1          # sem timestamp: estável byte a byte
    assert particoes.ler_manifesto(tmp_path) == manifesto
    assert particoes.conferir_particoes(tmp_path) == []


def test_conferir_acusa_alteracao_e_ausencia(tmp_path):
    particoes.gravar_particoes(particoes.particionar(docs_sinteticos()), tmp_path)
    particoes.gravar_manifesto(tmp_path, particoes.calcular_manifesto(
        tmp_path, seed=42, origem={"path": "x", "sha256": None}))

    dev = particoes.arquivo_particao(tmp_path, "dev")
    dev.write_bytes(dev.read_bytes().replace(b"documento", b"Documento", 1))
    divergencias = particoes.conferir_particoes(tmp_path)
    assert [d.split(":")[0] for d in divergencias] == ["dev.sha256"]

    particoes.arquivo_particao(tmp_path, "test").unlink()
    assert "test.jsonl: arquivo ausente" in particoes.conferir_particoes(tmp_path)
