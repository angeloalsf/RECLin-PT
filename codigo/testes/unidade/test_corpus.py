"""Leitura dos XML, regra C1, inventário, auditoria e geração da v2, sobre XML
sintéticos (nenhum arquivo do corpus é lido aqui; os números do corpus real
estão em `test_dados_v2.py`).

Cada XML sintético imita o formato do SemClinBr: estrutura com CRLF, `<TEXT>`
com o fim de linha escolhido e anotações cujos offsets seguem a convenção
pedida (CRLF contando 2 ou a quebra contando 1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reclin import particoes, tarefa
from reclin.corpus import auditoria, comparacao, conversao, fonte, geracao, semclinbr
from reclin.util import caminhos
from reclin.util.io import sha256_arquivo


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def xml(texto: str, anotacoes, relacoes=(), *, nl: str = "\r\n") -> bytes:
    """`anotacoes`: (id, tag, start, end, text); `relacoes`: (a1, a2, tipo)."""
    linhas = ["<?xml version='1.0' encoding='UTF-8'?>", "<ANNOTATIONS>", f"<TEXT>{_esc(texto)}</TEXT>", "<TAGS>"]
    linhas += [f'<annotation id="{i}" tag="{_esc(tg)}" start="{s}" end="{e}" text="{_esc(t)}" abbr="" />'
               for i, tg, s, e, t in anotacoes]
    linhas += ["</TAGS>", "<RELATIONS>"]
    linhas += [f'<rel annotation1="{a}" annotation2="{b}" reltype="{r}" />' for a, b, r in relacoes]
    linhas += ["</RELATIONS>", "</ANNOTATIONS>", ""]
    return nl.join(linhas).encode("utf-8")


def anotar(texto: str, palavras, *, convencao: str, inicio: int = 1):
    """Anotações das `palavras` (na ordem, primeira ocorrência depois da
    anterior) com os offsets na convenção pedida: 'crlf2' (posição no texto
    com CRLF) ou 'lf1' (posição com cada CRLF contando 1)."""
    alvo = texto if convencao == "crlf2" else texto.replace("\r\n", "\n")
    saida, pos = [], 0
    for k, p in enumerate(palavras):
        s = alvo.index(p, pos)
        saida.append((str(inicio + k), "Finding" if p != "NEGA" else "Negation", s, s + len(p), p))
        pos = s + len(p)
    return saida


TEXTO = "PACIENTE ESTÁVEL.\r\nNEGA FEBRE.\r\nREFERE DOR EM AÇÃO.\r\nSEM EDEMA."
PALAVRAS = ["PACIENTE", "NEGA", "FEBRE", "DOR", "AÇÃO", "EDEMA"]


def doc(texto=TEXTO, *, convencao="crlf2", relacoes=(("2", "3", "negation_of"),), extra=(), doc_id="100"):
    return semclinbr.ler_bytes(xml(texto, anotar(texto, PALAVRAS, convencao=convencao) + list(extra), relacoes),
                               doc_id=doc_id, arquivo=f"{doc_id}.xml")


# --------------------------------------------------------------------------- #
# Leitura (representação intermediária)                                       #
# --------------------------------------------------------------------------- #
def test_texto_bruto_preserva_crlf_e_texto_xml_normaliza():
    d = doc()
    assert d["texto_bruto"] == TEXTO
    assert d["texto_xml"] == TEXTO.replace("\r\n", "\n")
    assert d["quebras"] == {"crlf": 3, "cr_isolado": 0, "lf_isolado": 0}
    assert [a["ordem"] for a in d["anotacoes"]] == list(range(len(PALAVRAS)))
    assert d["relacoes"] == [{"annotation1": "2", "annotation2": "3", "reltype": "negation_of", "ordem": 0}]
    assert len(d["sha256"]) == 64 and d["doc_id"] == "100"


def test_referencias_do_xml_desfeitas_no_texto_bruto():
    texto = "PA > 120 & FC < 90 \"ok\"\r\nSEM DOR"
    d = semclinbr.ler_bytes(xml(texto, []), doc_id="1")
    assert d["texto_bruto"] == texto and d["texto_xml"] == texto.replace("\r\n", "\n")
    assert semclinbr.desfazer_referencias("&#233;&#xE9;&apos;") == "éé'"
    with pytest.raises(semclinbr.ErroFonte, match="não declarada"):
        semclinbr.desfazer_referencias("&nbsp;")


def test_texto_lf_tambem_e_lido():
    texto = TEXTO.replace("\r\n", "\n")
    d = semclinbr.ler_bytes(xml(texto, [], nl="\n"), doc_id="1")
    assert d["texto_bruto"] == d["texto_xml"] == texto
    assert d["quebras"] == {"crlf": 0, "cr_isolado": 0, "lf_isolado": 3}


@pytest.mark.parametrize("conteudo, erro", [
    (b"\xef\xbb\xbf" + xml("A", []), "BOM"),
    (xml("A", []).replace(b"</TAGS>", b""), "malformado"),
    (xml("A", []).replace(b"<TEXT>A</TEXT>", b"<TEXT><![CDATA[A]]></TEXT>"), "marcação"),
    (xml("A", []).replace(b"<TAGS>", b"<OUTRO/><TAGS>"), "inesperados"),
    (xml("A", []).replace(b"ANNOTATIONS>", b"DOC>"), "raiz"),
    (xml("A", [("1", "x", "0a", "1", "A")]), "não é inteiro"),
    (xml("A", [("1", "x", 0, 1, "A")]).replace(b' abbr=""', b' cui="C1"'), "inesperados"),
    ("ÁÉ".encode("latin-1"), "UTF-8"),
], ids=["bom", "malformado", "cdata", "elemento", "raiz", "offset", "atributo", "codificacao"])
def test_fora_do_formato_e_erro_explicito(conteudo, erro):
    with pytest.raises(semclinbr.ErroFonte, match=erro):
        semclinbr.ler_bytes(conteudo, doc_id="1")


def test_ler_pasta_em_ordem_numerica_e_somente_leitura(tmp_path):
    for i in (10, 9, 100):
        (tmp_path / f"{i}.xml").write_bytes(xml("A B", [("1", "x", 0, 1, "A")]))
    antes = {p.name: sha256_arquivo(p) for p in tmp_path.iterdir()}
    assert [d["doc_id"] for d in semclinbr.ler_pasta(tmp_path)] == ["9", "10", "100"]
    assert {p.name: sha256_arquivo(p) for p in tmp_path.iterdir()} == antes


# --------------------------------------------------------------------------- #
# Sistema de offsets e regra C1                                               #
# --------------------------------------------------------------------------- #
def test_crlf_contando_2_vira_dois_espacos_sem_mudar_offsets():
    d = doc(convencao="crlf2")
    registro, conv = conversao.converter(d)
    assert conv["sistema"] == conversao.BRUTO and conv["transformado"]
    assert registro["text"] == TEXTO.replace("\r\n", "  ") and len(registro["text"]) == len(TEXTO)
    for e, a in zip(registro["entities"], d["anotacoes"]):
        assert (e["id"], e["start"], e["end"], e["text"], e["type"]) == (a["id"], a["start"], a["end"], a["text"],
                                                                          a["tag"])
        assert registro["text"][e["start"]:e["end"]] == e["text"]          # alinhada, na posição
    # o texto do legado (quebra = 1 espaço) desalinha as entidades depois da 1ª quebra (deriva acumulada)
    legado = conversao.texto_legado(d)
    deltas = [legado.find(e["text"], max(0, e["start"] - 5)) - e["start"] for e in registro["entities"]]
    assert deltas == [0, -1, -1, -2, -2, -3]


def test_quebra_contando_1_mantem_o_texto_do_legado():
    d = doc(convencao="lf1")
    registro, conv = conversao.converter(d)
    assert conv["sistema"] == conversao.LF and not conv["transformado"]
    assert registro["text"] == conversao.texto_legado(d) == TEXTO.replace("\r\n", " ")
    assert all(registro["text"][e["start"]:e["end"]] == e["text"] for e in registro["entities"])


def test_sem_quebra_e_indistinguivel_nao_sao_transformados():
    texto = "NEGA FEBRE E DOR"
    d = semclinbr.ler_bytes(xml(texto, anotar(texto, ["NEGA", "FEBRE"], convencao="crlf2")), doc_id="1")
    r, c = conversao.converter(d)
    assert c["sistema"] == conversao.SEM_CRLF and not c["transformado"] and r["text"] == texto
    texto = "NEGA FEBRE.\r\nOUTRA LINHA"                       # entidades antes da 1ª quebra
    d = semclinbr.ler_bytes(xml(texto, anotar(texto, ["NEGA", "FEBRE"], convencao="crlf2")), doc_id="1")
    r, c = conversao.converter(d)
    assert c["sistema"] == conversao.INDISTINGUIVEL and not c["transformado"]
    assert r["text"] == "NEGA FEBRE. OUTRA LINHA"


def test_documento_ambiguo_e_erro_e_nao_e_convertido():
    texto = "AB\r\nCD"
    # uma anotação confere só com CRLF=2, outra só com quebra=1: empate com falhas
    d = semclinbr.ler_bytes(xml(texto, [("1", "x", 4, 6, "CD"), ("2", "x", 3, 5, "CD")]), doc_id="1")
    assert conversao.decidir_sistema(d)["sistema"] == conversao.AMBIGUO
    with pytest.raises(conversao.ErroConversao, match="decisão necessária"):
        conversao.converter(d)


def test_unicode_e_acentos_contam_como_um_caractere():
    d = doc()
    r, _ = conversao.converter(d)
    e = next(e for e in r["entities"] if e["text"] == "AÇÃO")
    assert e["end"] - e["start"] == 4 and r["text"][e["start"]:e["end"]] == "AÇÃO"


def test_entidades_repetidas_e_sobrepostas_preservadas_na_ordem():
    texto = "DOR\r\nDOR TORÁCICA\r\nDOR"
    alvo = texto
    anot = [("1", "x", 0, 3, "DOR"), ("2", "x", alvo.index("DOR", 3), alvo.index("DOR", 3) + 3, "DOR"),
            ("3", "x", alvo.index("DOR", 3), alvo.index("DOR", 3) + 12, "DOR TORÁCICA"),   # sobreposta
            ("4", "x", alvo.rindex("DOR"), alvo.rindex("DOR") + 3, "DOR")]
    d = semclinbr.ler_bytes(xml(texto, anot, [("1", "4", "associated_with")]), doc_id="1")
    r, c = conversao.converter(d)
    assert c["sistema"] == conversao.BRUTO and c["anomalias"] == []
    assert [(e["id"], e["start"], e["end"]) for e in r["entities"]] == [(a[0], a[2], a[3]) for a in anot]
    assert all(r["text"][e["start"]:e["end"]] == e["text"] for e in r["entities"])
    assert r["relations"] == [{"e1_id": "1", "e2_id": "4", "type": "associated_with"}]


def test_identificador_repetido_e_relacao_orfa_sao_erros():
    texto = "NEGA FEBRE"
    d = semclinbr.ler_bytes(xml(texto, [("1", "x", 0, 4, "NEGA"), ("1", "x", 5, 10, "FEBRE")]), doc_id="1")
    with pytest.raises(conversao.ErroConversao, match="repetidos"):
        conversao.converter(d)
    d = semclinbr.ler_bytes(xml(texto, [("1", "x", 0, 4, "NEGA")], [("1", "9", "negation_of")]), doc_id="1")
    with pytest.raises(conversao.ErroConversao, match="inexistentes"):
        conversao.converter(d)


def test_anotacoes_defeituosas_mantidas_e_registradas():
    defeitos = [("90", "x", -1, 8, "PACIENTE"),                         # start = -1
                ("91", "x", 0, 8, "PACIENTE PACIENTE"),                 # palavras duplicadas
                ("92", "x", 0, 8, "ENFERMO")]                           # divergente
    d = doc(extra=defeitos)
    r, c = conversao.converter(d)
    assert [e["id"] for e in r["entities"]][-3:] == ["90", "91", "92"]  # nada descartado
    assert {(e["id"], e["start"]) for e in r["entities"]} >= {("90", -1)}  # offset inválido intocado
    tipos = {a["anotacao"]: a["tipo"] for a in c["anomalias"]}
    assert tipos == {"90": conversao.OFFSET_INVALIDO, "91": conversao.PALAVRAS_DUPLICADAS,
                     "92": conversao.TRECHO_DIVERGENTE}
    assert c["sistema"] == conversao.BRUTO                              # a maioria decide o documento


def test_offset_entre_cr_e_lf_registrado():
    texto = "AB\r\nCD\r\nEF"
    anot = anotar(texto, ["CD", "EF"], convencao="crlf2") + [("9", "x", 3, 6, "\nCD")]
    d = semclinbr.ler_bytes(xml(texto, anot), doc_id="1")
    r, c = conversao.converter(d)
    assert any(a["tipo"] == conversao.OFFSET_DENTRO_DE_CRLF and a["anotacao"] == "9" for a in c["anomalias"])
    assert r["entities"][-1]["start"] == 3


def test_conversao_deterministica():
    a = conversao.converter(doc())
    b = conversao.converter(doc())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_categorias_na_posicao():
    assert conversao.categoria("A > B", "A &gt; B") == conversao.ESCAPE
    assert conversao.categoria("DOR  NO PEITO", "DOR NO PEITO") == conversao.SO_ESPACOS
    assert conversao.categoria("35,7ºC", "35 , 7ºC") == conversao.TOKENIZADO
    assert conversao.categoria("FEBRE", "TOSSE") == conversao.FALHA


# --------------------------------------------------------------------------- #
# Auditoria por identidade                                                    #
# --------------------------------------------------------------------------- #
def test_auditoria_distingue_desalinhada_anomalia_e_identidade():
    d = doc(extra=[("90", "x", 0, 8, "ENFERMO")])
    v2, _ = conversao.converter(d)
    v1 = dict(v2, text=conversao.texto_legado(d))
    r1 = auditoria.auditar([v1], {"100": d})
    r2 = auditoria.auditar([v2], {"100": d})
    assert r2["entidades"][auditoria.ALINHADA] == 6 and r2["entidades"][auditoria.ANOMALIA] == 1
    assert r2["entidades"][auditoria.DESALINHADA] == 0
    assert r1["entidades"][auditoria.DESALINHADA] == 5                 # todas depois da 1ª quebra
    assert r1["relacoes"]["com_argumento_desalinhado"] == 1 and r2["relacoes"]["com_argumento_nao_alinhado"] == 0
    # trocar um offset por busca de texto seria uma associação incorreta: detectada pela identidade
    errado = json.loads(json.dumps(v1))
    errado["entities"][1]["start"] -= 1
    errado["entities"][1]["end"] -= 1
    r = auditoria.auditar([errado], {"100": d})
    assert r["entidades"][auditoria.IDENTIDADE] == 1
    assert auditoria.auditar([v2], {})["entidades"][auditoria.NAO_VERIFICAVEL] == 7


def test_auditoria_sem_rotulos_no_test():
    d = doc()
    v2, _ = conversao.converter(d)
    r = auditoria.auditar([v2], {"100": d}, com_rotulos=False)
    assert "relacoes_nao_alinhadas_por_tipo" not in r


# --------------------------------------------------------------------------- #
# Inventário                                                                  #
# --------------------------------------------------------------------------- #
def test_inventario_hashes_quebras_e_erros(tmp_path):
    (tmp_path / "2.xml").write_bytes(xml(TEXTO, anotar(TEXTO, PALAVRAS, convencao="crlf2")))
    (tmp_path / "1.xml").write_bytes(xml("SEM QUEBRA", [("1", "x", -1, 3, "SEM")]))
    (tmp_path / "3.xml").write_bytes(b"<ANNOTATIONS><TEXT>")
    (tmp_path / "leiame.txt").write_text("x")
    inv = fonte.inventariar(tmp_path)
    assert inv["n_arquivos_xml"] == 3 and inv["outros_arquivos"] == ["leiame.txt"]
    assert [a["doc_id"] for a in inv["arquivos"]] == ["1", "2", "3"]
    assert inv["totais"]["com_erro"] == 1 and inv["erros"][0]["arquivo"] == "3.xml"
    assert inv["totais"]["texto_com_crlf"] == 1 and inv["totais"]["texto_sem_quebra"] == 1
    assert inv["anomalias_por_tipo"] == {"offset_negativo": 1}
    esperado = fonte.sha256_conjunto((a["arquivo"], sha256_arquivo(tmp_path / a["arquivo"])) for a in inv["arquivos"])
    assert inv["conjunto_sha256"] == esperado
    assert fonte.comparar_ids(inv, ["1", "2", "4"]) == {"ausentes": ["4"], "excedentes": ["3"]}
    assert "não verificada" in inv["origem_declarada"]
    assert "PACIENTE" not in json.dumps(inv)                           # nenhum trecho do texto
    (tmp_path / "1.xml").write_bytes(xml("SEM QUEBRA!", []))
    assert fonte.inventariar(tmp_path)["conjunto_sha256"] != inv["conjunto_sha256"]


# --------------------------------------------------------------------------- #
# Geração, conferência e invariantes entre versões                            #
# --------------------------------------------------------------------------- #
def _corpus_sintetico(pasta: Path, n: int = 20) -> None:
    pasta.mkdir()
    for i in range(n):
        convencao = "crlf2" if i % 3 else "lf1"
        texto = TEXTO if i % 4 else "NEGA FEBRE E DOR. PACIENTE ESTÁVEL. AÇÃO. EDEMA"
        palavras = PALAVRAS if i % 4 else ["NEGA", "FEBRE", "DOR", "PACIENTE", "AÇÃO", "EDEMA"]
        rel = (("2", "3", "negation_of"),) if i % 2 else (("1", "2", "associated_with"),)
        (pasta / f"{500 + i}.xml").write_bytes(xml(texto, anotar(texto, palavras, convencao=convencao), rel))


def _v1_de(pasta_xml: Path, destino: Path) -> None:
    """Uma 'v1' sintética: o texto do legado com os mesmos offsets."""
    docs = [dict(conversao.converter(f)[0], text=conversao.texto_legado(f)) for f in semclinbr.ler_pasta(pasta_xml)]
    particoes.gravar_particoes(particoes.particionar(docs), destino)
    particoes.gravar_manifesto(destino, particoes.calcular_manifesto(destino, seed=42,
                                                                      origem={"path": "x", "sha256": None}))


def test_geracao_completa_deterministica_e_rastreavel(tmp_path):
    xmls, v1, v2 = tmp_path / "xml", tmp_path / "v1", tmp_path / "v2"
    _corpus_sintetico(xmls)
    _v1_de(xmls, v1)
    m = geracao.gerar(xmls, v2, dataset=tmp_path / "ds.jsonl", intermediario=tmp_path / "int.jsonl",
                      pasta_anterior=v1, data="2026-01-01")
    assert particoes.conferir_particoes(v2) == []
    assert m["versao"] == geracao.VERSAO and m["fonte_xml"]["n_arquivos"] == 20
    assert m["fonte_xml"]["inventario_sha256"] == sha256_arquivo(v2 / "FONTE.json")
    assert m["anomalias"]["n"] == 0 and m["geracao"]["data"] == "2026-01-01"
    for nome, inv in m["derivada_de"]["invariantes"].items():
        assert inv["mesmos_documentos_mesma_ordem"] and inv["entidades_identicas"] and inv["relacoes_identicas"]
        assert inv["candidatos_mesmas_chaves_e_ordem"] and inv["y_true_identico"]
        assert inv["texto_diferente_so_nos_transformados"] and inv["texto_v2_igual_a_regra_aplicada_a_fonte"]
    # rastreabilidade XML -> intermediária -> dataset -> partição
    fonte_json = json.loads((v2 / "FONTE.json").read_text("utf-8"))
    for nome in particoes.PARTICOES:
        for d in particoes.ler_particao(v2, nome):
            registro = fonte_json["conversao_por_documento"][d["doc_id"]]
            assert registro["sha256_xml"] == sha256_arquivo(xmls / registro["arquivo"])
            assert registro["transformado"] == (registro["sistema"] == conversao.BRUTO)
    assert geracao.conferir(xmls, v2, pasta_anterior=v1) == []
    # o dataset e a intermediária têm o hash registrado
    assert m["source"]["sha256"] == sha256_arquivo(tmp_path / "ds.jsonl")
    assert m["conversor"]["representacao_intermediaria"]["sha256"] == sha256_arquivo(tmp_path / "int.jsonl")


def test_conferir_detecta_alteracao(tmp_path):
    xmls, v2 = tmp_path / "xml", tmp_path / "v2"
    _corpus_sintetico(xmls)
    geracao.gerar(xmls, v2, dataset=tmp_path / "ds.jsonl", intermediario=tmp_path / "int.jsonl",
                  pasta_anterior=None, data="d")
    assert geracao.conferir(xmls, v2, pasta_anterior=None) == []
    (xmls / "501.xml").write_bytes((xmls / "501.xml").read_bytes().replace(b"EDEMA", b"EDEMX", 1))
    assert geracao.conferir(xmls, v2, pasta_anterior=None)


def test_geracao_protege_a_v1_e_nao_sobrescreve(tmp_path):
    xmls = tmp_path / "xml"
    _corpus_sintetico(xmls)
    kw = dict(dataset=tmp_path / "ds.jsonl", intermediario=tmp_path / "int.jsonl", pasta_anterior=None, data="d")
    with pytest.raises(geracao.ErroGeracao, match="congelada"):
        geracao.gerar(xmls, caminhos.PARTICOES_V1, **kw)
    geracao.gerar(xmls, tmp_path / "v2", **kw)
    with pytest.raises(geracao.ErroGeracao, match="sobrescrever"):
        geracao.gerar(xmls, tmp_path / "v2", **kw)


def test_geracao_para_se_a_atribuicao_divergir(tmp_path):
    xmls, v1 = tmp_path / "xml", tmp_path / "v1"
    _corpus_sintetico(xmls)
    _v1_de(xmls, v1)
    trocadas = particoes.ler_particoes(v1)
    trocadas["train"][0], trocadas["dev"][0] = trocadas["dev"][0], trocadas["train"][0]
    particoes.gravar_particoes(trocadas, v1)
    particoes.gravar_manifesto(v1, particoes.calcular_manifesto(v1, seed=42, origem={"path": "x", "sha256": None}))
    with pytest.raises(geracao.ErroGeracao, match="mesmos documentos"):
        geracao.gerar(xmls, tmp_path / "v2", dataset=tmp_path / "ds.jsonl", intermediario=tmp_path / "int.jsonl",
                      pasta_anterior=v1, data="d")
    assert not (tmp_path / "v2").exists()


def test_comparacao_acusa_divergencias_sem_forcar():
    d = doc()
    v2, _ = conversao.converter(d)
    v1 = dict(v2, text=conversao.texto_legado(d))
    ok = comparacao.comparar_particao([v1], [v2], fontes={"100": d}, transformados={"100": True})
    assert ok["entidades_identicas"] and ok["candidatos"]["mesmas_chaves_e_ordem"] and ok["candidatos"]["y_true_identico"]
    assert ok["texto"]["depois_igual_a_regra_aplicada_a_fonte"]
    outro = json.loads(json.dumps(v2))
    outro["entities"][2]["start"] += 30                     # muda os candidatos
    r = comparacao.comparar_particao([v1], [outro])
    assert not r["entidades_identicas"] and r["documentos_com_entidades_diferentes"] == ["100"]
    assert not r["candidatos"]["mesmas_chaves_e_ordem"]
    janelas = comparacao.comparar_janelas([v1], [v2], transformados={"100": True})
    assert janelas["trecho_marcado_diverge_do_atributo"]["depois"] == 0
    assert janelas["trecho_marcado_diverge_do_atributo"]["antes"] > 0


def test_invariantes_dos_pares():
    d = doc()
    v2, _ = conversao.converter(d)
    cands = tarefa.candidatos([v2])
    assert comparacao.invariantes_de_pares(cands) == []
    assert comparacao.invariantes_de_pares(list(reversed(cands)))     # ordem quebrada é acusada


def test_texto_bruto_conferido_contra_o_parser():
    """Uma referência `&#13;` no <TEXT> vira um `\\r` que o parser preserva, mas
    que a normalização do fim de linha transformaria: a leitura acusa a
    divergência em vez de seguir com dois textos diferentes."""
    conteudo = xml("A", []).replace(b"<TEXT>A</TEXT>", b"<TEXT>A&#13;B</TEXT>")
    with pytest.raises(semclinbr.ErroFonte, match="difere do texto do parser"):
        semclinbr.ler_bytes(conteudo, doc_id="1")
