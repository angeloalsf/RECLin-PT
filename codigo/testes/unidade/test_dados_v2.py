"""A versão v2-texto-fonte dos dados, materializada em `codigo/dados/particoes-v2/`.

Confere o registro de versões contra os arquivos, as previsões da regra C1
(D9) contra a v1 e os números da auditoria do alinhamento depois da correção.
Rótulos por tipo só no TRAIN e no DEV; no TEST, só identidade estrutural
(documentos, entidades, relações, candidatos e o hash de `y_true`).

Os testes que precisam dos XML do SemClinBr (fora do git, licença) são
pulados quando a pasta não está presente; a regeração completa é `lento`.
"""
from __future__ import annotations

import json

import pytest

from reclin import particoes, tarefa, versao_dados as vd
from reclin.analises import alinhamento
from reclin.corpus import auditoria, comparacao, conversao, fonte, geracao, semclinbr
from reclin.negacao import lexico
from reclin.util import caminhos
from reclin.util.io import ler_json, sha256_arquivo

V1, V2 = caminhos.PARTICOES_V1, caminhos.PARTICOES_V2
XML = caminhos.XML_SEMCLINBR
com_xml = pytest.mark.skipif(not XML.is_dir(), reason="XML do SemClinBr ausente (fora do git: licença)")


@pytest.fixture(scope="module")
def v1():
    return particoes.ler_particoes(V1)


@pytest.fixture(scope="module")
def v2():
    return particoes.ler_particoes(V2)


@pytest.fixture(scope="module")
def manifesto_v2():
    return particoes.ler_manifesto(V2)


@pytest.fixture(scope="module")
def fontes():
    return {f["doc_id"]: f for f in semclinbr.ler_pasta(XML)}


# --------------------------------------------------------------------------- #
# Registro, MANIFEST e companheiros                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("versao", vd.VERSOES)
def test_registro_confere_com_os_arquivos(versao):
    pasta = vd.PASTAS[versao]
    assert particoes.conferir_particoes(pasta) == []
    assert sha256_arquivo(pasta / "MANIFEST.json") == vd.REGISTRO[versao]["manifest_sha256"]
    for nome in particoes.PARTICOES:
        assert sha256_arquivo(pasta / f"{nome}.jsonl") == vd.REGISTRO[versao]["particoes"][nome]
    assert vd.versao_da_pasta(pasta) == versao


def test_manifesto_v2(manifesto_v2):
    m = manifesto_v2
    assert m["versao"] == vd.V2 == geracao.VERSAO and m["seed"] == particoes.SEED
    assert m["conversor"]["versao_parser"] == semclinbr.VERSAO_PARSER
    assert m["conversor"]["sistemas"] == {"bruto": 421, "indistinguivel": 6, "lf": 101, "sem_crlf": 472}
    assert m["fonte_xml"]["n_arquivos"] == 1000 and m["fonte_xml"]["regra_conjunto"] == fonte.REGRA_CONJUNTO
    assert m["fonte_xml"]["inventario_sha256"] == sha256_arquivo(V2 / geracao.ARQUIVO_FONTE)
    assert m["anomalias"]["arquivo_sha256"] == sha256_arquivo(V2 / geracao.ARQUIVO_ANOMALIAS)
    assert m["anomalias"]["n"] == 26
    assert m["anomalias"]["por_tipo"] == {conversao.PALAVRAS_DUPLICADAS: 22, conversao.TRECHO_DIVERGENTE: 3,
                                          conversao.OFFSET_INVALIDO: 1}
    assert m["anomalias"]["por_particao"] == {"dev": 1, "test": 5, "train": 20}
    assert {n: c["documentos_transformados"] for n, c in m["contagens"].items()} == \
        {"train": 346, "dev": 36, "test": 39}
    assert m["derivada_de"]["manifest_sha256"] == vd.REGISTRO[vd.V1]["manifest_sha256"]
    for nome, inv in m["derivada_de"]["invariantes"].items():
        assert all(v for k, v in inv.items() if isinstance(v, bool)), (nome, inv)
        assert inv["invariantes_dos_pares_violados"] == 0
    for nome in particoes.PARTICOES:
        assert m["conjuntos_referencia"][nome]["n_candidatos"] == tarefa.TAMANHOS_ESPERADOS[nome]


def test_anomalias_e_fonte_sem_texto_do_corpus(v2):
    anomalias = ler_json(V2 / geracao.ARQUIVO_ANOMALIAS)["anomalias"]
    assert len(anomalias) == 26
    ids = {(d["doc_id"], e["id"]) for nome in particoes.PARTICOES for d in v2[nome] for e in d["entities"]}
    assert all((a["doc_id"], a["anotacao"]) in ids for a in anomalias)          # mantidas no dataset
    assert [a for a in anomalias if a["tipo"] == conversao.OFFSET_INVALIDO] == [
        {"doc_id": "9808", "particao": "train", "anotacao": "39461", "tipo": "offset_invalido",
         "start": -1, "end": 132, "relacoes_que_a_usam": 0}]
    assert all(a["relacoes_que_a_usam"] == 0 for a in anomalias if a["particao"] != "test")
    texto = json.dumps(ler_json(V2 / geracao.ARQUIVO_FONTE), ensure_ascii=False) + json.dumps(anomalias)
    amostra = v2["dev"][0]["text"][:40]
    assert amostra not in texto


# --------------------------------------------------------------------------- #
# Previsões da regra C1 contra a v1                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nome", particoes.PARTICOES)
def test_mesmos_documentos_entidades_relacoes_candidatos_e_y_true(nome, v1, v2):
    r = comparacao.comparar_particao(v1[nome], v2[nome], com_rotulos=(nome != "test"))
    assert r["documentos"]["mesmos_e_mesma_ordem"]
    assert r["entidades_identicas"] and r["relacoes_identicas"]
    assert r["candidatos"]["mesmas_chaves_e_ordem"] and r["candidatos"]["y_true_identico"]
    assert r["candidatos"]["depois"] == tarefa.TAMANHOS_ESPERADOS[nome]
    assert r["candidatos"]["invariantes_dos_pares"] == []
    assert r["texto"]["outras_chaves_iguais"]
    assert r["texto"]["documentos_com_texto_diferente"] == {"train": 346, "dev": 36, "test": 39}[nome]
    if nome != "test":
        assert r["candidatos"]["rotulos"]["antes"] == r["candidatos"]["rotulos"]["depois"]


@pytest.mark.parametrize("nome", particoes.PARTICOES)
def test_conjunto_de_referencia_muda_so_no_hash_da_particao(nome, v1, v2):
    a = tarefa.conjunto_referencia(nome, v1[nome], particao_sha256=vd.REGISTRO[vd.V1]["particoes"][nome])
    b = tarefa.conjunto_referencia(nome, v2[nome], particao_sha256=vd.REGISTRO[vd.V2]["particoes"][nome])
    assert {k: v for k, v in a.items() if k != "particao_sha256"} == {k: v for k, v in b.items()
                                                                     if k != "particao_sha256"}


def test_particao_refeita_e_a_mesma_e_sem_sobreposicao(v2):
    todos = [d for nome in particoes.PARTICOES for d in v2[nome]]
    refeita = particoes.particionar(todos, seed=particoes.SEED)
    for nome in particoes.PARTICOES:
        assert [d["doc_id"] for d in refeita[nome]] == [d["doc_id"] for d in v2[nome]]
    conjuntos = [{d["doc_id"] for d in v2[n]} for n in particoes.PARTICOES]
    assert sum(map(len, conjuntos)) == len(set().union(*conjuntos)) == 1000


def test_lexico_congelado_igual_na_v2(v2):
    lex = lexico.carregar_congelado(v2["train"])
    assert len(lex) == lexico.CONGELADO["n_formas"]


def test_textos_sem_quebra_de_linha(v2):
    assert not any("\n" in d["text"] or "\r" in d["text"] for nome in particoes.PARTICOES for d in v2[nome])


# --------------------------------------------------------------------------- #
# Auditoria do alinhamento (TRAIN e DEV; números do D9 depois da correção)    #
# --------------------------------------------------------------------------- #
ESPERADO_V2 = {
    "train": {"status": {"exato": 35368, "equivalente": 1067, "deslocado": 1, "ausente": 19},
              "afetados_negation_of": (0, 1255), "deriva": {"sem_deriva": 799, "monotona": 1, "nao_monotona": 0}},
    "dev": {"status": {"exato": 4403, "equivalente": 109, "deslocado": 0, "ausente": 1},
            "afetados_negation_of": (0, 150), "deriva": {"sem_deriva": 100, "monotona": 0, "nao_monotona": 0}},
}


@pytest.mark.parametrize("nome", ["train", "dev"])
def test_numeros_da_auditoria_na_v2(nome, v2):
    docs = v2[nome]
    assert alinhamento.auditar(docs)["status"] == ESPERADO_V2[nome]["status"]
    neg = alinhamento.impacto_nos_pares(docs)["negation_of"]
    assert (neg["afetados"], neg["pares"]) == ESPERADO_V2[nome]["afetados_negation_of"]
    assert alinhamento.perfil_da_deriva(docs)["documentos"] == ESPERADO_V2[nome]["deriva"]


# --------------------------------------------------------------------------- #
# Contra os XML                                                               #
# --------------------------------------------------------------------------- #
@com_xml
def test_inventario_dos_xml(manifesto_v2):
    inv = fonte.inventariar(XML)
    assert inv["n_arquivos_xml"] == 1000 and not inv["erros"] and not inv["doc_ids_repetidos"]
    assert inv["conjunto_sha256"] == manifesto_v2["fonte_xml"]["conjunto_sha256"]
    assert inv["totais"]["utf8_valido"] == 1000 and inv["totais"]["texto_com_crlf"] == 528
    assert inv["totais"]["texto_com_cr_isolado"] == inv["totais"]["texto_com_lf_isolado"] == 0
    assert fonte.comparar_ids(inv, [d["doc_id"] for n in particoes.PARTICOES
                                    for d in particoes.ler_particao(V2, n)]) == {"ausentes": [], "excedentes": []}


@com_xml
@pytest.mark.parametrize("nome", particoes.PARTICOES)
def test_cada_entidade_contra_a_anotacao_de_origem(nome, v1, v2, fontes):
    com_rotulos = nome != "test"
    a1 = auditoria.auditar(v1[nome], fontes, com_rotulos=com_rotulos)
    a2 = auditoria.auditar(v2[nome], fontes, com_rotulos=com_rotulos)
    anomalias = {"train": 20, "dev": 1, "test": 5}[nome]
    assert a2["entidades"] == {"alinhada": a2["n_entidades"] - anomalias, "desalinhada": 0,
                               "anomalia_da_fonte": anomalias, "identidade_divergente": 0, "nao_verificavel": 0}
    assert a1["entidades"]["identidade_divergente"] == 0 and a1["entidades"]["anomalia_da_fonte"] == anomalias
    assert a1["entidades"]["desalinhada"] == {"train": 14487, "dev": 1509, "test": 1545}[nome]
    assert a2["relacoes"]["com_argumento_nao_alinhado"] == 0
    assert a2["relacoes"]["documentos_com_relacoes_identicas_a_fonte"] == len(v2[nome])


@com_xml
def test_texto_v1_e_o_do_legado_e_v2_e_a_regra_aplicada(v1, v2, fontes):
    transformados = {i: conversao.converter(f)[1]["transformado"] for i, f in fontes.items()}
    for nome in particoes.PARTICOES:
        r = comparacao.comparar_particao(v1[nome], v2[nome], fontes=fontes, transformados=transformados,
                                         com_rotulos=False)
        assert r["texto"]["antes_igual_ao_legado_da_fonte"]
        assert r["texto"]["depois_igual_a_regra_aplicada_a_fonte"]
        assert r["texto"]["igual_aos_transformados_previstos"]
        assert r["texto"]["comprimento_igual_ao_bruto_nos_transformados"]


@com_xml
@pytest.mark.lento
def test_v2_reproduzida_byte_a_byte_dos_xml():
    assert geracao.conferir(XML, V2, pasta_anterior=V1) == []
