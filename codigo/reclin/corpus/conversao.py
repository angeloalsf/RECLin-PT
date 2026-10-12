"""Da representação intermediária ao registro do dataset (regra C1).

O PROBLEMA (investigação D9)
----------------------------
Nos XML locais, os offsets de cada documento seguem uma de duas convenções:
contam cada `\\r\\n` do `<TEXT>` como **dois** caracteres (`bruto`) ou como
**um** (`lf`). O legado gravava o texto do parser XML (quebra = 1 caractere)
com a quebra trocada por espaço, e copiava os offsets: nos documentos `bruto`,
cada entidade fica deslocada de −k, sendo k o número de quebras antes dela.

SISTEMA DE CADA DOCUMENTO (`decidir_sistema`)
---------------------------------------------
Decidido por documento, com todas as suas anotações, sempre **na posição**
dada pelos offsets (nunca procurando o texto em outro lugar):

* `sem_crlf` — o `<TEXT>` não tem `\\r\\n`;
* conta as anotações válidas cujo trecho em `texto_bruto[start:end]` confere
  com o atributo `text` (`bruto`) e as que conferem em
  `texto_legado[start:end]` (`lf`); "confere" = categoria em `CONFERE`;
* vence a maior contagem; empate com todas conferindo → `indistinguivel`
  (todas as entidades estão antes da primeira quebra: nenhum offset muda);
  empate com alguma falha → `ambiguo` (erro: exige decisão, nada é gerado).

TEXTO DA VERSÃO v2 (`texto_v2`)
-------------------------------
* `bruto`: cada `\\r\\n` do texto bruto vira **dois espaços** — o texto passa
  a ter o comprimento da convenção dos offsets, e os offsets da fonte valem
  sem alteração;
* `sem_crlf`, `lf` e `indistinguivel`: o texto do legado, sem mudança
  (`texto_legado`: o texto do parser com cada quebra trocada por um espaço).
  Nada é transformado onde a transformação não é necessária.

ENTIDADES E RELAÇÕES
--------------------
Copiadas da fonte, na ordem do arquivo: `id`, `start`, `end`, `text` e
`type` (= `tag`) da anotação; `e1_id`, `e2_id` e `type` (= `reltype`) da
relação. **Nenhum offset é alterado, nenhuma anotação é descartada.** As
anotações cujo trecho não confere com o atributo nem na convenção do próprio
documento são defeitos da fonte: ficam no dataset como estão e são listadas
em `anomalias` com o tipo. Identificador de anotação repetido ou relação para
anotação inexistente são erros (`ErroConversao`): exigiriam escolher qual
anotação usar ou descartar uma relação.

Um offset que cai entre o `\\r` e o `\\n` de uma quebra, num documento `bruto`,
é registrado como anomalia (`offset_dentro_de_crlf`); com os dois espaços da
regra C1 ele continua apontando a mesma posição.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from typing import Any, Iterable

from reclin.corpus.semclinbr import AnotacaoFonte, DocumentoFonte
from reclin.tarefa import Documento

SEM_CRLF, BRUTO, LF, INDISTINGUIVEL, AMBIGUO = "sem_crlf", "bruto", "lf", "indistinguivel", "ambiguo"
SISTEMAS = (SEM_CRLF, BRUTO, LF, INDISTINGUIVEL, AMBIGUO)

EXATO, ESCAPE, SO_ESPACOS, TOKENIZADO, FALHA = "exato", "escape_no_atributo", "so_espacos", "tokenizado", "falha"
CATEGORIAS = (EXATO, ESCAPE, SO_ESPACOS, TOKENIZADO, FALHA)
CONFERE = frozenset({EXATO, ESCAPE, SO_ESPACOS, TOKENIZADO})

PALAVRAS_DUPLICADAS = "atributo_com_palavras_duplicadas"
TRECHO_DIVERGENTE = "atributo_divergente_do_trecho"
OFFSET_INVALIDO = "offset_invalido"
OFFSET_DENTRO_DE_CRLF = "offset_dentro_de_crlf"
TIPOS_DE_ANOMALIA = (PALAVRAS_DUPLICADAS, TRECHO_DIVERGENTE, OFFSET_INVALIDO, OFFSET_DENTRO_DE_CRLF)

REGRA_C1 = ("C1 (v2-texto-fonte): sistema de offsets decidido por documento, na posição, com todas as "
            "anotações; nos documentos 'bruto' (CRLF conta 2) cada CRLF do <TEXT> bruto vira dois espaços; "
            "nos demais ('sem_crlf', 'lf', 'indistinguivel') o texto é o do legado (texto do parser XML com cada "
            "quebra trocada por um espaço); offsets, entidades e relações copiados da fonte sem alteração; "
            "anotações defeituosas da fonte mantidas e registradas")


class ErroConversao(ValueError):
    """O documento não pode ser convertido sem uma decisão que a regra não toma."""


# --------------------------------------------------------------------------- #
# Comparação na posição                                                       #
# --------------------------------------------------------------------------- #
def categoria(trecho: str, atributo: str) -> str:
    """Compara o trecho apontado pelos offsets com o atributo `text`, na mesma
    posição: idêntico; igual com as referências HTML desfeitas no atributo;
    igual a menos de espaços repetidos; igual sem nenhum espaço; ou falha."""
    desfeito = html.unescape(atributo)
    if trecho == atributo:
        return EXATO
    if trecho == desfeito:
        return ESCAPE
    if " ".join(trecho.split()) == " ".join(desfeito.split()):
        return SO_ESPACOS
    if re.sub(r"\s+", "", trecho) == re.sub(r"\s+", "", desfeito):
        return TOKENIZADO
    return FALHA


def offset_valido(a: AnotacaoFonte | dict, comprimento: int) -> bool:
    return 0 <= a["start"] <= a["end"] <= comprimento


def texto_legado(doc: DocumentoFonte) -> str:
    """O texto que a conversão do legado gravava (quebra de linha = 1 espaço)."""
    return doc["texto_xml"].replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def texto_preservando_comprimento(doc: DocumentoFonte) -> str:
    """O texto bruto com cada `\\r\\n` trocado por dois espaços."""
    if doc["quebras"]["cr_isolado"] or doc["quebras"]["lf_isolado"]:
        raise ErroConversao(f"{doc['doc_id']}: quebra de linha isolada (\\r ou \\n) no <TEXT>; "
                            "a regra C1 só trata \\r\\n")
    return doc["texto_bruto"].replace("\r\n", "  ")


def _confere(texto: str, anotacoes: Iterable[AnotacaoFonte]) -> tuple[int, int]:
    validas = [a for a in anotacoes if offset_valido(a, len(texto))]
    return sum(categoria(texto[a["start"]:a["end"]], a["text"]) in CONFERE for a in validas), len(validas)


def decidir_sistema(doc: DocumentoFonte) -> dict[str, Any]:
    """Sistema de offsets do documento e as contagens que o decidiram."""
    if doc["quebras"]["crlf"] == 0:
        return {"sistema": SEM_CRLF, "conferem_bruto": None, "conferem_lf": None}
    bruto, n_bruto = _confere(doc["texto_bruto"], doc["anotacoes"])
    lf, n_lf = _confere(texto_legado(doc), doc["anotacoes"])
    if bruto == lf:
        sistema = INDISTINGUIVEL if bruto == n_bruto == n_lf else AMBIGUO
    else:
        sistema = BRUTO if bruto > lf else LF
    return {"sistema": sistema, "conferem_bruto": bruto, "conferem_lf": lf,
            "anotacoes_validas": max(n_bruto, n_lf)}


def texto_v2(doc: DocumentoFonte, sistema: str) -> str:
    if sistema == BRUTO:
        return texto_preservando_comprimento(doc)
    if sistema in (SEM_CRLF, LF, INDISTINGUIVEL):
        return texto_legado(doc)
    raise ErroConversao(f"{doc['doc_id']}: sistema de offsets {sistema!r}: decisão necessária "
                        "(a regra C1 não escolhe a convenção de um documento ambíguo)")


# --------------------------------------------------------------------------- #
# Anomalias                                                                   #
# --------------------------------------------------------------------------- #
def _sem_duplicacao(texto: str) -> str:
    """Remove palavras repetidas em sequência ('bom bom estado estado' → 'bom estado')."""
    saida: list[str] = []
    for palavra in texto.split():
        if not saida or saida[-1] != palavra:
            saida.append(palavra)
    return " ".join(saida)


def anomalia_da_anotacao(texto: str, a: AnotacaoFonte | dict) -> str | None:
    """Tipo do defeito da anotação no texto da versão (na posição), ou None."""
    if not offset_valido(a, len(texto)):
        return OFFSET_INVALIDO
    trecho = texto[a["start"]:a["end"]]
    if categoria(trecho, a["text"]) in CONFERE:
        return None
    if _sem_duplicacao(html.unescape(a["text"])) == " ".join(trecho.split()):
        return PALAVRAS_DUPLICADAS
    return TRECHO_DIVERGENTE


def _crlf_partidos(doc: DocumentoFonte) -> list[dict[str, Any]]:
    bruto = doc["texto_bruto"]
    dentro = {m.start() + 1 for m in re.finditer("\r\n", bruto)}
    saida = []
    for a in doc["anotacoes"]:
        for campo in ("start", "end"):
            if a[campo] in dentro:
                saida.append({"anotacao": a["id"], "tipo": OFFSET_DENTRO_DE_CRLF, "campo": campo,
                              "start": a["start"], "end": a["end"]})
    return saida


# --------------------------------------------------------------------------- #
# Conversão                                                                   #
# --------------------------------------------------------------------------- #
def converter(doc: DocumentoFonte) -> tuple[Documento, dict[str, Any]]:
    """(registro do dataset, registro da conversão) de um documento.

    O registro da conversão tem o sistema, se o texto foi transformado e por
    quê, as contagens por categoria na posição e as anomalias (sem nenhum
    trecho do texto)."""
    ids = Counter(a["id"] for a in doc["anotacoes"])
    repetidos = sorted(i for i, n in ids.items() if n > 1)
    if repetidos:
        raise ErroConversao(f"{doc['doc_id']}: identificadores de anotação repetidos {repetidos[:5]}")
    orfas = [r["ordem"] for r in doc["relacoes"] if r["annotation1"] not in ids or r["annotation2"] not in ids]
    if orfas:
        raise ErroConversao(f"{doc['doc_id']}: relações {orfas[:5]} apontam para anotações inexistentes")

    decisao = decidir_sistema(doc)
    sistema = decisao["sistema"]
    texto = texto_v2(doc, sistema)
    legado = texto_legado(doc)
    transformado = texto != legado
    if transformado and len(texto) != len(doc["texto_bruto"]):
        raise ErroConversao(f"{doc['doc_id']}: o texto transformado não tem o comprimento do <TEXT> bruto")

    registro: Documento = {
        "doc_id": doc["doc_id"],
        "text": texto,
        "entities": [{"id": a["id"], "start": a["start"], "end": a["end"], "text": a["text"], "type": a["tag"]}
                     for a in doc["anotacoes"]],
        "relations": [{"e1_id": r["annotation1"], "e2_id": r["annotation2"], "type": r["reltype"]}
                      for r in doc["relacoes"]],
    }

    categorias: Counter[str] = Counter()
    anomalias = []
    for a in doc["anotacoes"]:
        tipo = anomalia_da_anotacao(texto, a)
        if offset_valido(a, len(texto)):
            categorias[categoria(texto[a["start"]:a["end"]], a["text"])] += 1
        else:
            categorias[OFFSET_INVALIDO] += 1
        if tipo is not None:
            anomalias.append({"anotacao": a["id"], "tipo": tipo, "start": a["start"], "end": a["end"]})
    if sistema == BRUTO:
        anomalias += _crlf_partidos(doc)

    if sistema == BRUTO:
        justificativa = (f"offsets contam CRLF como 2 ({decisao['conferem_bruto']} anotações conferem na posição "
                         f"com CRLF=2 contra {decisao['conferem_lf']} com quebra=1): "
                         f"{doc['quebras']['crlf']} CRLF trocados por dois espaços")
    elif sistema == LF:
        justificativa = (f"offsets contam a quebra como 1 ({decisao['conferem_lf']} contra "
                         f"{decisao['conferem_bruto']}): texto do legado mantido")
    elif sistema == INDISTINGUIVEL:
        justificativa = "todas as anotações conferem nas duas convenções (antes da primeira quebra): texto do legado mantido"
    else:
        justificativa = "sem CRLF no <TEXT>: texto do legado mantido"

    conversao = {
        "doc_id": doc["doc_id"], "arquivo": doc["arquivo"], "sha256_xml": doc["sha256"],
        "sistema": sistema, "conferem_bruto": decisao["conferem_bruto"], "conferem_lf": decisao["conferem_lf"],
        "n_crlf": doc["quebras"]["crlf"], "transformado": transformado, "justificativa": justificativa,
        "n_entidades": len(registro["entities"]), "n_relacoes": len(registro["relations"]),
        "categorias": dict(sorted(categorias.items())), "anomalias": anomalias,
    }
    return registro, conversao


def converter_todos(documentos: Iterable[DocumentoFonte]) -> tuple[list[Documento], list[dict[str, Any]]]:
    registros, conversoes = [], []
    for doc in documentos:
        r, c = converter(doc)
        registros.append(r)
        conversoes.append(c)
    return registros, conversoes
