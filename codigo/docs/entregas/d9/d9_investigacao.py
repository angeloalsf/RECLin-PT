"""D9 — investigação do desalinhamento entre textos e offsets (protótipo isolado).

Somente leitura: lê os XML do SemClinBr e as partições congeladas, não grava
nenhum arquivo e não altera nada no projeto. Imprime um resumo em JSON.

Uso:
    python d9_investigacao.py --xml-dir <pasta dos XML> --codigo <pasta codigo/ do RECLin-PT>

TEST: só contagens estruturais (documentos, sistema de coordenadas, número de
candidatos e igualdade da lista de candidatos). Nenhuma relação do TEST é
contada por rótulo e nada do TEST entra em decisão.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import copy
import glob
import hashlib
import html
import json
import os
import re
import sys
from xml.etree import ElementTree as ET

OK = {"exato", "escape_no_atributo", "so_espacos", "tokenizado"}


# --------------------------------------------------------------------------- #
# Leitura da fonte                                                            #
# --------------------------------------------------------------------------- #
def ler_xml(caminho: str):
    """(bruto, normalizado, anotações, relações) de um XML.

    bruto       — conteúdo de <TEXT> como está nos bytes do arquivo, com as
                  entidades XML desfeitas e os CRLF preservados;
    normalizado — <TEXT> como um parser XML o entrega (fim de linha
                  normalizado para LF, XML 1.0 §2.11), o que o legado usou.
    """
    b = open(caminho, "rb").read()
    s = b.decode("utf-8")
    conteudo = re.search(r"<TEXT>(.*?)</TEXT>", s, flags=re.S).group(1)
    assert not conteudo.startswith("<![CDATA[")
    bruto = html.unescape(conteudo)
    raiz = ET.fromstring(b)
    normalizado = raiz.find("TEXT").text
    tags = raiz.find("TAGS")
    anns = [(a.get("id"), int(a.get("start")), int(a.get("end")), a.get("text") or "")
            for a in (tags.findall("annotation") if tags is not None else [])]
    rels = raiz.find("RELATIONS")
    rels = [(r.get("annotation1"), r.get("annotation2"), r.get("reltype"))
            for r in (rels.findall("rel") if rels is not None else [])]
    return bruto, normalizado, anns, rels


def categoria(trecho: str, t: str) -> str:
    """Compara o trecho apontado pelos offsets com o atributo `text` da anotação,
    na MESMA posição (nunca procura o texto em outro lugar)."""
    u = html.unescape(t)
    if trecho == t:
        return "exato"
    if trecho == u:
        return "escape_no_atributo"
    if " ".join(trecho.split()) == " ".join(u.split()):
        return "so_espacos"
    if re.sub(r"\s+", "", trecho) == re.sub(r"\s+", "", u):
        return "tokenizado"
    return "falha"


def sistema(bruto: str, atual: str, anns) -> str:
    """Sistema de coordenadas dos offsets, decidido por DOCUMENTO com todas as
    suas anotações: 'bruto' (CRLF conta 2), 'lf' (quebra conta 1),
    'sem_crlf', 'indistinguivel' ou 'ambiguo'."""
    if "\r\n" not in bruto:
        return "sem_crlf"
    validas = [(s, e, t) for _, s, e, t in anns if s >= 0]
    r = sum(categoria(bruto[s:e], t) in OK for s, e, t in validas)
    l_ = sum(categoria(atual[s:e], t) in OK for s, e, t in validas)
    if r == l_:
        return "indistinguivel" if r == len(validas) else "ambiguo"
    return "bruto" if r > l_ else "lf"


# --------------------------------------------------------------------------- #
# Correções candidatas (em memória)                                           #
# --------------------------------------------------------------------------- #
def corrigir(doc: dict, bruto: str, sis: str, variante: str):
    novo = copy.deepcopy(doc)
    flags = []
    if sis != "bruto":
        return novo, flags
    if variante == "C":     # texto que preserva o comprimento; offsets da fonte intocados
        novo["text"] = bruto.replace("\r\n", "  ")
        assert len(novo["text"]) == len(bruto) and "\r" not in novo["text"] and "\n" not in novo["text"]
        return novo, flags
    # variante B: texto atual intocado; offsets levados a ele por posição
    crlf = [m.start() for m in re.finditer("\r\n", bruto)]

    def mapa(o: int):
        if o < 0:
            return o, "offset_negativo"
        k = bisect.bisect_left(crlf, o)              # CRLF inteiramente antes de o
        return o - k, ("dentro_de_crlf" if k and crlf[k - 1] == o - 1 else None)

    for e in novo["entities"]:
        (s2, f1), (e2, f2) = mapa(e["start"]), mapa(e["end"])
        if f1 or f2:
            flags.append((doc["doc_id"], e["id"], f1 or f2))
        e["start"], e["end"] = s2, e2
    return novo, flags


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--codigo", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.codigo)
    from reclin import tarefa                      # noqa: E402  (somente leitura)
    from reclin.analises import alinhamento as al  # noqa: E402

    fonte = {}
    for p in sorted(glob.glob(os.path.join(a.xml_dir, "*.xml"))):
        fonte[os.path.basename(p)[:-4]] = ler_xml(p)
    pasta = os.path.join(a.codigo, "dados", "particoes")
    particoes = {p: [json.loads(l) for l in open(os.path.join(pasta, f"{p}.jsonl"), encoding="utf-8")]
                 for p in ("train", "dev", "test")}
    saida: dict = {"n_xml": len(fonte)}

    # 1. Cadeia XML -> partições
    cadeia = collections.Counter()
    for p, docs in particoes.items():
        for d in docs:
            bruto, norm, anns, rels = fonte[d["doc_id"]]
            cadeia["norm_igual_bruto_com_crlf_em_lf"] += norm == bruto.replace("\r\n", "\n")
            cadeia["texto_particao_igual_norm_com_lf_em_espaco"] += d["text"] == norm.replace("\n", " ")
            cadeia["entidades_iguais_ao_xml"] += sorted((e["id"], e["start"], e["end"], e["text"]) for e in d["entities"]) == sorted(anns)
            cadeia["relacoes_iguais_ao_xml"] += sorted((r["e1_id"], r["e2_id"], r["type"]) for r in d["relations"]) == sorted(rels)
            cadeia["docs"] += 1
    saida["cadeia"] = dict(cadeia)

    # 2. Sistema por documento e coerência com a faixa de ids
    sis = {did: sistema(b, n.replace("\n", " "), an) for did, (b, n, an, _) in fonte.items()}
    lf = sorted(int(d) for d, s in sis.items() if s == "lf")
    lo, hi = lf[0], lf[-1]
    saida["sistema"] = dict(collections.Counter(sis.values()))
    saida["faixa_lf"] = [lo, hi]
    saida["faixa_lf_composicao"] = dict(collections.Counter(s for d, s in sis.items() if lo <= int(d) <= hi))
    saida["fora_da_faixa_composicao"] = dict(collections.Counter(s for d, s in sis.items() if not lo <= int(d) <= hi))

    # 3. Hipóteses (em todo o corpus, sem rótulos de relação)
    h = collections.Counter()
    for did, (bruto, norm, anns, _) in fonte.items():
        atual = norm.replace("\n", " ")
        alvo = bruto if sis[did] == "bruto" else atual
        h["docs_com_TAB"] += "\t" in bruto
        h["docs_com_LF_isolado"] += bool(re.search(r"(?<!\r)\n", bruto))
        h["docs_com_CR_isolado"] += bool(re.search(r"\r(?!\n)", bruto))
        h["docs_fora_do_BMP"] += any(ord(ch) > 0xFFFF for ch in bruto)
        for _, s, e, t in anns:
            h["anotacoes"] += 1
            if s < 0:
                h["start_negativo"] += 1
                continue
            c_alvo, c_atual = categoria(alvo[s:e], t), categoria(atual[s:e], t)
            h["alvo_" + c_alvo] += 1
            h["atual_" + c_atual] += 1
            if c_alvo not in OK:
                h["inclusivo_resolveria"] += categoria(alvo[s:e + 1], t) in OK
                h["bytes_resolveria"] += alvo.encode("utf-8")[s:e].decode("utf-8", "replace") == t
            elif alvo[:e] != alvo[:e].encode("utf-8").decode("latin-1"):
                h["chars_certo_bytes_errado"] += 1
            h["checagem_legado_divergente"] += " ".join(atual[s:e].split()) != " ".join(t.split())
    saida["hipoteses"] = dict(h)

    # 4. Correções candidatas por partição
    for p, docs in particoes.items():
        r = {"docs": len(docs), "sistema": dict(collections.Counter(sis[d["doc_id"]] for d in docs))}
        base = tarefa.candidatos(docs)
        r["candidatos_atual"] = len(base)
        if p != "test":
            r["auditoria_etapa7_atual"] = al.auditar(docs)["status"]
            r["negation_of_afetados_atual"] = al.impacto_nos_pares(docs)["negation_of"]
            r["rotulos_atual"] = tarefa.contagem_por_rotulo(base)
        for v in ("B", "C"):
            novos, flags = [], []
            for d in docs:
                n, f = corrigir(d, fonte[d["doc_id"]][0], sis[d["doc_id"]], v)
                novos.append(n)
                flags += f
            for x, y in zip(docs, novos):    # identidade das anotações e relações
                assert x["doc_id"] == y["doc_id"] and x["relations"] == y["relations"]
                assert [(e["id"], e["text"], e["type"]) for e in x["entities"]] == \
                       [(e["id"], e["text"], e["type"]) for e in y["entities"]]
            ordem = 0
            for x, y in zip(docs, novos):    # ordem e sobreposição entre todos os pares de entidades
                rel = lambda u, w: (u["start"] < w["start"], u["start"] == w["start"], u["end"] <= w["start"], u["end"] == w["end"])
                ordem += sum(rel(e1, e2) != rel(f1, f2) for e1, f1 in zip(x["entities"], y["entities"])
                             for e2, f2 in zip(x["entities"], y["entities"]))
            cats = collections.Counter(categoria(d["text"][e["start"]:e["end"]], e["text"]) if e["start"] >= 0
                                       else "offset_negativo" for d in novos for e in d["entities"])
            cands = tarefa.candidatos(novos)
            rv = {"categorias": dict(cats), "flags": len(flags), "ordem_ou_sobreposicao_alterada": ordem,
                  "candidatos": len(cands),
                  "candidatos_iguais_ao_atual": [tarefa.chave_candidato(c) for c in cands] == [tarefa.chave_candidato(c) for c in base],
                  "sha": sha_json(novos),
                  "sha_repeticao": sha_json([corrigir(d, fonte[d["doc_id"]][0], sis[d["doc_id"]], v)[0] for d in docs])}
            if p != "test":
                rv["auditoria_etapa7"] = al.auditar(novos)["status"]
                rv["negation_of_afetados"] = al.impacto_nos_pares(novos)["negation_of"]
                rv["perfil_da_deriva"] = al.perfil_da_deriva(novos)["documentos"]
                rv["rotulos"] = tarefa.contagem_por_rotulo(cands)
            if p != "test" and v == "B":      # risco de realinhar por busca textual
                risco = collections.Counter()
                originais = {d["doc_id"]: {e["id"]: e for e in d["entities"]} for d in docs}
                for d in novos:
                    for e in d["entities"]:
                        o = originais[d["doc_id"]][e["id"]]
                        if e["start"] < 0 or e["start"] == o["start"] or d["text"][e["start"]:e["end"]] != e["text"]:
                            continue
                        risco["corrigidas_exatas"] += 1
                        occ = [m.start() for m in re.finditer(re.escape(e["text"]), d["text"])]
                        risco["texto_repetido_no_documento"] += len(occ) > 1
                        risco["primeira_ocorrencia_erraria"] += bool(occ) and occ[0] != e["start"]
                        b = al.alinhar_entidade(d["text"], o)
                        risco["busca_local_etapa7_erraria"] += b["status"] == "deslocado" and o["start"] + b["delta"] != e["start"]
                        risco["busca_local_etapa7_nao_acharia"] += b["status"] == "ausente"
                rv["risco_busca_textual"] = dict(risco)
            r[v] = rv
        saida[p] = r
    print(json.dumps(saida, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
