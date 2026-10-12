"""Invariantes entre duas versões dos dados (por exemplo, v1 × v2).

A regra C1 prevê (D9) que a nova versão mantém, em cada partição: os mesmos
documentos na mesma ordem; entidades e relações idênticas; os mesmos
candidatos (chaves e ordem), os mesmos rótulos e o mesmo `y_true`; e muda só o
campo `text` dos documentos transformados, com o mesmo comprimento do texto
bruto. `comparar_particao` confere cada uma dessas previsões e devolve os
resultados lado a lado, **sem forçar igualdade**: uma divergência aparece como
`False` e com exemplos (só identificadores), para ser investigada.

Os rótulos por tipo (distribuição) só são calculados com `com_rotulos=True`
(TRAIN e DEV). No TEST a comparação é estrutural: igualdade das sequências e
dos hashes, sem contagem por rótulo.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from reclin import entrada, tarefa
from reclin.corpus import conversao
from reclin.corpus.semclinbr import DocumentoFonte
from reclin.tarefa import Documento
from reclin.util.io import sha256_json


def _identidade_candidatos(docs: Sequence[Documento]) -> dict[str, Any]:
    cands = tarefa.candidatos(docs)
    chaves = [tarefa.chave_candidato(c) for c in cands]
    return {"n": len(cands), "chaves": chaves, "candidatos_sha256": sha256_json(chaves),
            "y_true_sha256": sha256_json(tarefa.y_true(cands)), "cands": cands}


def invariantes_de_pares(cands: Sequence[tarefa.Candidato], max_gap: int = tarefa.MAX_GAP) -> list[str]:
    """Propriedades da enumeração que valem em qualquer versão: sem auto-par,
    sem span idêntico, distância dentro de `max_gap`, pares em ordem
    estritamente crescente de ((start, end, id) de e1, (start, end, id) de e2)
    dentro de cada documento, sem par repetido."""
    problemas = []
    vistos = set()
    for i, c in enumerate(cands):
        e1, e2 = c["e1"], c["e2"]
        chave = tuple(tarefa.chave_candidato(c)[:3])
        if e1["id"] == e2["id"]:
            problemas.append(f"auto-par em {c['doc_id']}:{e1['id']}")
        if (e1["start"], e1["end"]) == (e2["start"], e2["end"]):
            problemas.append(f"span idêntico em {c['doc_id']}:{e1['id']},{e2['id']}")
        if tarefa.entity_gap(e1, e2) > max_gap:
            problemas.append(f"distância acima de {max_gap} em {c['doc_id']}:{e1['id']},{e2['id']}")
        if chave in vistos:
            problemas.append(f"par repetido {chave}")
        vistos.add(chave)
        if i and cands[i - 1]["doc_id"] == c["doc_id"]:
            a = cands[i - 1]
            k = lambda e: (e["start"], e["end"], e["id"])            # noqa: E731
            if (k(a["e1"]), k(a["e2"])) >= (k(e1), k(e2)):
                problemas.append(f"ordem (start, end, id) dos pares quebrada em {c['doc_id']}")
        if len(problemas) >= 20:
            break
    return problemas


def comparar_particao(antes: Sequence[Documento], depois: Sequence[Documento], *,
                      fontes: Mapping[str, DocumentoFonte] | None = None,
                      transformados: Mapping[str, bool] | None = None,
                      com_rotulos: bool = True) -> dict[str, Any]:
    """Previsões da regra C1 para uma partição, conferidas uma a uma.

    Com `fontes` (representação intermediária dos XML), confere também o texto
    de cada documento contra a fonte, de forma independente da conversão: o
    texto de `antes` é o do legado (quebra = 1 espaço) e o de `depois` é o
    bruto com CRLF → dois espaços nos `transformados` e o do legado nos demais."""
    ids_a, ids_d = [d["doc_id"] for d in antes], [d["doc_id"] for d in depois]
    r: dict[str, Any] = {"documentos": {"antes": len(antes), "depois": len(depois),
                                        "mesmos_e_mesma_ordem": ids_a == ids_d}}
    pares = list(zip(antes, depois)) if ids_a == ids_d else []
    entidades_dif = [a["doc_id"] for a, d in pares if a["entities"] != d["entities"]]
    relacoes_dif = [a["doc_id"] for a, d in pares if a["relations"] != d["relations"]]
    r["entidades_identicas"] = bool(pares) and not entidades_dif
    r["relacoes_identicas"] = bool(pares) and not relacoes_dif
    r["documentos_com_entidades_diferentes"] = entidades_dif[:20]
    r["documentos_com_relacoes_diferentes"] = relacoes_dif[:20]
    r["n_entidades"] = {"antes": sum(len(d["entities"]) for d in antes),
                        "depois": sum(len(d["entities"]) for d in depois)}
    r["n_relacoes"] = {"antes": sum(len(d["relations"]) for d in antes),
                       "depois": sum(len(d["relations"]) for d in depois)}

    texto_mudou = [a["doc_id"] for a, d in pares if a["text"] != d["text"]]
    r["texto"] = {"documentos_com_texto_diferente": len(texto_mudou),
                  "outras_chaves_iguais": all(set(a) == set(d) for a, d in pares)}
    if transformados is not None:
        previstos = sorted((i for i, t in transformados.items() if t and i in set(ids_a)),
                           key=lambda i: ids_a.index(i))
        r["texto"]["igual_aos_transformados_previstos"] = texto_mudou == previstos
        r["texto"]["transformados_previstos"] = len(previstos)
    if fontes is not None and transformados is not None:
        def esperado_depois(fonte: DocumentoFonte) -> str:
            if transformados[fonte["doc_id"]]:
                return fonte["texto_bruto"].replace("\r\n", "  ")
            return conversao.texto_legado(fonte)
        r["texto"]["antes_igual_ao_legado_da_fonte"] = all(
            a["text"] == conversao.texto_legado(fontes[a["doc_id"]]) for a, _ in pares)
        r["texto"]["depois_igual_a_regra_aplicada_a_fonte"] = all(
            d["text"] == esperado_depois(fontes[d["doc_id"]]) for _, d in pares)
        r["texto"]["comprimento_igual_ao_bruto_nos_transformados"] = all(
            len(d["text"]) == len(fontes[d["doc_id"]]["texto_bruto"]) for _, d in pares if transformados[d["doc_id"]])
        r["texto"]["sem_quebra_de_linha"] = not any("\n" in d["text"] or "\r" in d["text"] for d in depois)

    ca, cd = _identidade_candidatos(antes), _identidade_candidatos(depois)
    r["candidatos"] = {"antes": ca["n"], "depois": cd["n"], "mesmas_chaves_e_ordem": ca["chaves"] == cd["chaves"],
                       "candidatos_sha256": {"antes": ca["candidatos_sha256"], "depois": cd["candidatos_sha256"]},
                       "y_true_sha256": {"antes": ca["y_true_sha256"], "depois": cd["y_true_sha256"]},
                       "y_true_identico": ca["y_true_sha256"] == cd["y_true_sha256"],
                       "invariantes_dos_pares": invariantes_de_pares(cd["cands"])}
    if com_rotulos:
        r["candidatos"]["rotulos"] = {"antes": tarefa.contagem_por_rotulo(ca["cands"]),
                                      "depois": tarefa.contagem_por_rotulo(cd["cands"])}
    return r


def comparar_janelas(antes: Sequence[Documento], depois: Sequence[Documento], *,
                     transformados: Mapping[str, bool], ctx_chars: int = entrada.CTX_CHARS) -> dict[str, Any]:
    """Janelas marcadas (`entrada.janela_marcada`) das duas versões, par a par
    na ordem do conjunto de referência: iguais nos documentos não
    transformados; e, em cada versão, quantos pares têm o trecho marcado de e1
    ou de e2 diferente do texto anotado da entidade (na posição). Não usa os
    rótulos."""
    if [d["doc_id"] for d in antes] != [d["doc_id"] for d in depois]:
        raise ValueError("as duas versões não têm os mesmos documentos na mesma ordem")
    r: dict[str, Any] = {"pares": 0, "janelas_iguais_fora_dos_transformados": True,
                         "pares_em_documentos_transformados": 0, "janelas_diferentes_nos_transformados": 0,
                         "trecho_marcado_diverge_do_atributo": {"antes": 0, "depois": 0}}

    def diverge(texto: str, e: tarefa.Entidade) -> bool:
        return not conversao.offset_valido(e, len(texto)) or \
            conversao.categoria(texto[e["start"]:e["end"]], e["text"]) not in conversao.CONFERE

    for a, d in zip(antes, depois):
        ca, cd = list(tarefa.iter_candidate_pairs(a)), list(tarefa.iter_candidate_pairs(d))
        if [tarefa.chave_candidato(c) for c in ca] != [tarefa.chave_candidato(c) for c in cd]:
            raise ValueError(f"{a['doc_id']}: candidatos diferentes entre as versões")
        mudou = transformados.get(d["doc_id"], False)
        for x, y in zip(ca, cd):
            r["pares"] += 1
            ja = entrada.janela_marcada(a["text"], x["e1"], x["e2"], ctx_chars)
            jd = entrada.janela_marcada(d["text"], y["e1"], y["e2"], ctx_chars)
            if mudou:
                r["pares_em_documentos_transformados"] += 1
                r["janelas_diferentes_nos_transformados"] += ja != jd
            elif ja != jd:
                r["janelas_iguais_fora_dos_transformados"] = False
            r["trecho_marcado_diverge_do_atributo"]["antes"] += diverge(a["text"], x["e1"]) or diverge(a["text"], x["e2"])
            r["trecho_marcado_diverge_do_atributo"]["depois"] += diverge(d["text"], y["e1"]) or diverge(d["text"], y["e2"])
    return r
