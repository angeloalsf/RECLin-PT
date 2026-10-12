"""Auditoria das entidades de um dataset contra as anotações de origem.

Cada entidade do dataset é ligada à anotação de origem pela **identidade**
(`doc_id` e `id`), nunca por busca de texto, e recebe um estado:

* `alinhada` — a identidade confere (mesmo `start`, `end`, `text` e tipo da
  anotação de origem) e o trecho `text[start:end]` do dataset confere com o
  atributo da anotação (`conversao.CONFERE`, na posição);
* `desalinhada` — a identidade confere, a anotação de origem confere com o
  texto da fonte na convenção do documento, mas o trecho do dataset não: o
  erro está na transformação do texto;
* `anomalia_da_fonte` — nem na fonte, na convenção do próprio documento, o
  trecho confere com o atributo (atributo defeituoso ou offset inválido):
  irrecuperável sem inventar um offset;
* `identidade_divergente` — a entidade não tem anotação de origem com o mesmo
  `id`, ou os campos copiados diferem (uma associação incorreta);
* `nao_verificavel` — o documento não está na fonte ou o sistema de offsets
  dele é ambíguo.

Para as relações, confere se os dois argumentos existem no documento e se
são as mesmas anotações da relação de origem (mesma ordem), e conta as
relações com algum argumento não alinhado. O tipo da relação só é usado
quando `com_rotulos=True` (TRAIN e DEV); no TEST só se contam relações.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from reclin.corpus import conversao
from reclin.corpus.semclinbr import DocumentoFonte
from reclin.tarefa import Documento

ALINHADA, DESALINHADA, ANOMALIA, IDENTIDADE, NAO_VERIFICAVEL = (
    "alinhada", "desalinhada", "anomalia_da_fonte", "identidade_divergente", "nao_verificavel")
ESTADOS = (ALINHADA, DESALINHADA, ANOMALIA, IDENTIDADE, NAO_VERIFICAVEL)


def texto_da_convencao(fonte: DocumentoFonte, sistema: str) -> str | None:
    """O texto da fonte em que os offsets do documento valem (None se ambíguo)."""
    if sistema == conversao.BRUTO:
        return fonte["texto_bruto"]
    if sistema in (conversao.SEM_CRLF, conversao.LF, conversao.INDISTINGUIVEL):
        return conversao.texto_legado(fonte)
    return None


def auditar_documento(doc: Documento, fonte: DocumentoFonte | None, *, com_rotulos: bool = True
                      ) -> dict[str, Any]:
    """Estado de cada entidade e as relações afetadas de um documento."""
    estados: dict[str, str] = {}
    if fonte is None:
        estados = {e["id"]: NAO_VERIFICAVEL for e in doc["entities"]}
        return {"estados": estados, "relacoes": _relacoes(doc, None, estados, com_rotulos), "sistema": None}
    sistema = conversao.decidir_sistema(fonte)["sistema"]
    referencia = texto_da_convencao(fonte, sistema)
    origem = {a["id"]: a for a in fonte["anotacoes"]}
    for e in doc["entities"]:
        a = origem.get(e["id"])
        if a is None or (a["start"], a["end"], a["text"], a["tag"]) != (e["start"], e["end"], e["text"], e["type"]):
            estados[e["id"]] = IDENTIDADE
        elif referencia is None:
            estados[e["id"]] = NAO_VERIFICAVEL
        elif conversao.anomalia_da_anotacao(referencia, a) in (conversao.OFFSET_INVALIDO,
                                                               conversao.PALAVRAS_DUPLICADAS,
                                                               conversao.TRECHO_DIVERGENTE):
            estados[e["id"]] = ANOMALIA
        elif conversao.offset_valido(e, len(doc["text"])) and \
                conversao.categoria(doc["text"][e["start"]:e["end"]], e["text"]) in conversao.CONFERE:
            estados[e["id"]] = ALINHADA
        else:
            estados[e["id"]] = DESALINHADA
    return {"estados": estados, "relacoes": _relacoes(doc, fonte, estados, com_rotulos), "sistema": sistema}


def _relacoes(doc: Documento, fonte: DocumentoFonte | None, estados: Mapping[str, str],
              com_rotulos: bool) -> dict[str, Any]:
    ids = {e["id"] for e in doc["entities"]}
    contagem: Counter[str] = Counter()
    por_tipo: Counter[str] = Counter()
    if fonte is not None:
        origem = [(r["annotation1"], r["annotation2"], r["reltype"]) for r in fonte["relacoes"]]
        atual = [(r["e1_id"], r["e2_id"], r["type"]) for r in doc["relations"]]
        contagem["documentos_com_relacoes_identicas_a_fonte"] = int(origem == atual)
    for r in doc["relations"]:
        contagem["relacoes"] += 1
        if r["e1_id"] not in ids or r["e2_id"] not in ids:
            contagem["argumento_inexistente"] += 1
            continue
        e1, e2 = estados.get(r["e1_id"]), estados.get(r["e2_id"])
        if e1 != ALINHADA or e2 != ALINHADA:
            contagem["com_argumento_nao_alinhado"] += 1
            if com_rotulos:
                por_tipo[r["type"]] += 1
        if DESALINHADA in (e1, e2):
            contagem["com_argumento_desalinhado"] += 1
        if ANOMALIA in (e1, e2):
            contagem["com_argumento_anomalo"] += 1
    saida: dict[str, Any] = dict(contagem)
    if com_rotulos:
        saida["com_argumento_nao_alinhado_por_tipo"] = dict(sorted(por_tipo.items()))
    return saida


def auditar(documentos: Iterable[Documento], fontes: Mapping[str, DocumentoFonte], *,
            com_rotulos: bool = True) -> dict[str, Any]:
    """Resumo da auditoria de uma partição: entidades por estado, relações
    afetadas e a lista das entidades não alinhadas (só identificadores)."""
    estados: Counter[str] = Counter()
    relacoes: Counter[str] = Counter()
    por_tipo: Counter[str] = Counter()
    sistemas: Counter[str] = Counter()
    nao_alinhadas = []
    n_docs = 0
    for doc in documentos:
        n_docs += 1
        r = auditar_documento(doc, fontes.get(doc["doc_id"]), com_rotulos=com_rotulos)
        sistemas[r["sistema"] or "sem_fonte"] += 1
        for ident, estado in r["estados"].items():
            estados[estado] += 1
            if estado != ALINHADA:
                nao_alinhadas.append({"doc_id": doc["doc_id"], "id": ident, "estado": estado})
        for k, v in r["relacoes"].items():
            if k == "com_argumento_nao_alinhado_por_tipo":
                por_tipo.update(v)
            else:
                relacoes[k] += v
    total = sum(estados.values())
    saida = {
        "n_documentos": n_docs, "n_entidades": total,
        "sistemas": dict(sorted(sistemas.items())),
        "entidades": {e: estados.get(e, 0) for e in ESTADOS},
        "fracao_alinhada": estados.get(ALINHADA, 0) / total if total else None,
        "relacoes": {k: relacoes.get(k, 0) for k in ("relacoes", "documentos_com_relacoes_identicas_a_fonte",
                                                       "argumento_inexistente", "com_argumento_nao_alinhado",
                                                       "com_argumento_desalinhado", "com_argumento_anomalo")},
        "nao_alinhadas": nao_alinhadas if len(nao_alinhadas) <= 100 else None,
        "n_nao_alinhadas_listadas": len(nao_alinhadas) if len(nao_alinhadas) <= 100 else 0,
    }
    if com_rotulos:
        saida["relacoes_nao_alinhadas_por_tipo"] = dict(sorted(por_tipo.items()))
    return saida
