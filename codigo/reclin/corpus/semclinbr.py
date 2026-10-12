"""Leitura de um XML do SemClinBr na representação intermediária.

FORMATO DA FONTE
----------------
Cada arquivo `<doc_id>.xml` tem a raiz `<ANNOTATIONS>` com:

* `<TEXT>` — o texto clínico, sem elementos filhos;
* `<TAGS>` — `<annotation id tag start end text [abbr]/>`: `start`/`end` são
  índices de caractere com `end` exclusivo; `tag` é o tipo (nomes de tipos
  semânticos da UMLS e rótulos próprios do corpus, como `Negation` e
  `Abbreviation`, unidos por `|`); `text` é o trecho anotado;
* `<RELATIONS>` — `<rel annotation1 annotation2 reltype/>`.

POR QUE DUAS VERSÕES DO TEXTO
-----------------------------
Um parser XML entrega o texto com o fim de linha normalizado (XML 1.0,
seção 2.11: `\\r\\n` e `\\r` viram `\\n`). Os offsets de parte dos documentos
contam cada `\\r\\n` como **dois** caracteres (investigação D9). Por isso a
representação intermediária guarda:

* `texto_bruto` — o conteúdo de `<TEXT>` tal como está nos bytes do arquivo,
  com as referências de entidade do XML desfeitas e os `\\r\\n` preservados;
* `texto_xml` — o mesmo conteúdo como o parser o entrega (o que o legado usou).

A leitura confere que `texto_bruto` com o fim de linha normalizado é
exatamente `texto_xml`: a extração do trecho bruto não muda nenhum outro
caractere. Qualquer divergência, ou uma estrutura fora do formato acima,
levanta `ErroFonte` — nada é ajustado silenciosamente.

Os atributos das anotações e relações são copiados como o parser os entrega
(a normalização de valores de atributo do XML vale também para o legado), na
ordem do arquivo, com a posição registrada em `ordem`. Os offsets são os do
XML, inteiros, sem nenhuma validação contra o texto (a validação é da
`conversao` e da `auditoria`, que registram os defeitos em vez de corrigi-los).

Origem no legado (só a finalidade): `parse_one` de `src/parse_semclinbr.py`,
que lia o XML com `lxml` e gravava o texto já com as quebras de linha
trocadas por espaço.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, TypedDict
from xml.etree import ElementTree as ET

VERSAO_PARSER = "semclinbr-xml-2"
"""Versão do leitor e da conversão (`conversao`). Muda quando a saída mudaria
para a mesma entrada."""

RAIZ = "ANNOTATIONS"
ELEMENTOS = ("TEXT", "TAGS", "RELATIONS")
ATRIBUTOS_ANOTACAO = ("id", "tag", "start", "end", "text")
ATRIBUTOS_OPCIONAIS_ANOTACAO = ("abbr",)
ATRIBUTOS_RELACAO = ("annotation1", "annotation2", "reltype")

_ENTIDADES_XML = {"lt": "<", "gt": ">", "amp": "&", "quot": '"', "apos": "'"}
_REFERENCIA = re.compile(r"&(#x[0-9A-Fa-f]+|#[0-9]+|[A-Za-z][A-Za-z0-9]*);")


class ErroFonte(ValueError):
    """O arquivo não está no formato esperado; a leitura não continua."""


class AnotacaoFonte(TypedDict):
    id: str
    tag: str
    start: int
    end: int
    text: str
    abbr: str | None
    ordem: int


class RelacaoFonte(TypedDict):
    annotation1: str
    annotation2: str
    reltype: str
    ordem: int


class DocumentoFonte(TypedDict):
    """Representação intermediária de um XML (ver o módulo)."""
    doc_id: str
    arquivo: str
    sha256: str
    texto_bruto: str
    texto_xml: str
    anotacoes: list[AnotacaoFonte]
    relacoes: list[RelacaoFonte]
    quebras: dict[str, int]


def desfazer_referencias(trecho: str) -> str:
    """Desfaz as referências do XML (`&lt;`, `&#233;`, `&#xE9;`...). Qualquer
    outro `&` é erro: o conteúdo de um elemento XML não admite `&` solto nem
    entidades não declaradas."""
    def trocar(m: re.Match) -> str:
        nome = m.group(1)
        if nome.startswith("#x"):
            return chr(int(nome[2:], 16))
        if nome.startswith("#"):
            return chr(int(nome[1:]))
        if nome in _ENTIDADES_XML:
            return _ENTIDADES_XML[nome]
        raise ErroFonte(f"entidade XML não declarada: &{nome};")

    resultado = _REFERENCIA.sub(trocar, trecho)
    if "&" in _REFERENCIA.sub("", trecho):
        raise ErroFonte("'&' fora de uma referência de entidade no <TEXT>")
    return resultado


def contar_quebras(texto: str) -> dict[str, int]:
    """`\\r\\n`, `\\r` isolado e `\\n` isolado."""
    crlf = texto.count("\r\n")
    return {"crlf": crlf, "cr_isolado": texto.count("\r") - crlf, "lf_isolado": texto.count("\n") - crlf}


def _texto_bruto(decodificado: str) -> str:
    aberturas = [m.start() for m in re.finditer(r"<TEXT>", decodificado)]
    fechamentos = [m.start() for m in re.finditer(r"</TEXT>", decodificado)]
    if len(aberturas) != 1 or len(fechamentos) != 1:
        raise ErroFonte(f"esperado exatamente um <TEXT>...</TEXT> literal; há {len(aberturas)} "
                        f"aberturas e {len(fechamentos)} fechamentos")
    inicio, fim = aberturas[0] + len("<TEXT>"), fechamentos[0]
    if fim < inicio:
        raise ErroFonte("</TEXT> antes de <TEXT>")
    conteudo = decodificado[inicio:fim]
    if "<" in conteudo:
        raise ErroFonte("<TEXT> com marcação dentro (CDATA, comentário ou elemento)")
    return desfazer_referencias(conteudo)


def _inteiro(valor: str | None, campo: str, anotacao: str | None) -> int:
    try:
        return int(valor)                                   # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ErroFonte(f"anotação {anotacao!r}: {campo}={valor!r} não é inteiro") from None


def ler_bytes(conteudo: bytes, *, doc_id: str, arquivo: str = "") -> DocumentoFonte:
    """Representação intermediária de um XML já lido (ver o módulo)."""
    if conteudo.startswith(b"\xef\xbb\xbf"):
        raise ErroFonte("arquivo com BOM UTF-8 (fora do formato da fonte)")
    try:
        decodificado = conteudo.decode("utf-8")
    except UnicodeDecodeError as erro:
        raise ErroFonte(f"não é UTF-8 válido: {erro}") from None
    try:
        raiz = ET.fromstring(conteudo)
    except ET.ParseError as erro:
        raise ErroFonte(f"XML malformado: {erro}") from None
    if raiz.tag != RAIZ:
        raise ErroFonte(f"raiz {raiz.tag!r}, esperado {RAIZ!r}")
    filhos = [f.tag for f in raiz]
    inesperados = sorted(set(filhos) - set(ELEMENTOS))
    if inesperados:
        raise ErroFonte(f"elementos inesperados em <{RAIZ}>: {inesperados}")
    for nome in ELEMENTOS:
        if filhos.count(nome) > 1 or (nome == "TEXT" and filhos.count(nome) != 1):
            raise ErroFonte(f"<{nome}> aparece {filhos.count(nome)} vezes")
    texto_el = raiz.find("TEXT")
    if len(texto_el) or texto_el.attrib:
        raise ErroFonte("<TEXT> com elementos filhos ou atributos")
    texto_xml = texto_el.text or ""
    texto_bruto = _texto_bruto(decodificado)
    if texto_bruto.replace("\r\n", "\n").replace("\r", "\n") != texto_xml:
        raise ErroFonte("o <TEXT> bruto, com o fim de linha normalizado, difere do texto do parser XML")

    anotacoes: list[AnotacaoFonte] = []
    tags = raiz.find("TAGS")
    for ordem, a in enumerate(tags if tags is not None else []):
        if a.tag != "annotation":
            raise ErroFonte(f"elemento inesperado em <TAGS>: {a.tag!r}")
        faltam = [c for c in ATRIBUTOS_ANOTACAO if c not in a.attrib]
        sobram = sorted(set(a.attrib) - set(ATRIBUTOS_ANOTACAO) - set(ATRIBUTOS_OPCIONAIS_ANOTACAO))
        if faltam or sobram:
            raise ErroFonte(f"anotação {a.get('id')!r}: atributos ausentes {faltam}, inesperados {sobram}")
        anotacoes.append({"id": a.get("id"), "tag": a.get("tag"),
                          "start": _inteiro(a.get("start"), "start", a.get("id")),
                          "end": _inteiro(a.get("end"), "end", a.get("id")),
                          "text": a.get("text"), "abbr": a.get("abbr"), "ordem": ordem})

    relacoes: list[RelacaoFonte] = []
    rels = raiz.find("RELATIONS")
    for ordem, r in enumerate(rels if rels is not None else []):
        if r.tag != "rel":
            raise ErroFonte(f"elemento inesperado em <RELATIONS>: {r.tag!r}")
        if set(r.attrib) != set(ATRIBUTOS_RELACAO):
            raise ErroFonte(f"relação {ordem}: atributos {sorted(r.attrib)}, esperado {list(ATRIBUTOS_RELACAO)}")
        relacoes.append({"annotation1": r.get("annotation1"), "annotation2": r.get("annotation2"),
                         "reltype": r.get("reltype"), "ordem": ordem})

    return {"doc_id": doc_id, "arquivo": arquivo, "sha256": hashlib.sha256(conteudo).hexdigest(),
            "texto_bruto": texto_bruto, "texto_xml": texto_xml, "anotacoes": anotacoes,
            "relacoes": relacoes, "quebras": contar_quebras(texto_bruto)}


def doc_id_do_arquivo(caminho: str | Path) -> str:
    """O identificador do documento é o nome do arquivo sem `.xml` (como no
    legado; o XML não traz um identificador interno)."""
    return Path(caminho).stem


def ler_arquivo(caminho: str | Path) -> DocumentoFonte:
    caminho = Path(caminho)
    with open(caminho, "rb") as f:                          # somente leitura
        conteudo = f.read()
    return ler_bytes(conteudo, doc_id=doc_id_do_arquivo(caminho), arquivo=caminho.name)


def arquivos_xml(pasta: str | Path) -> list[Path]:
    """Os `*.xml` da pasta, em ordem numérica do identificador (e pelo nome
    para identificadores não numéricos, depois dos numéricos)."""
    def chave(p: Path) -> tuple[int, int, str]:
        return (0, int(p.stem), p.name) if p.stem.isdigit() else (1, 0, p.name)
    return sorted((p for p in Path(pasta).iterdir() if p.is_file() and p.suffix.lower() == ".xml"), key=chave)


def ler_pasta(pasta: str | Path) -> list[DocumentoFonte]:
    """Representação intermediária de todos os XML da pasta, na ordem de
    `arquivos_xml`. O primeiro arquivo fora do formato interrompe a leitura."""
    documentos = []
    for caminho in arquivos_xml(pasta):
        try:
            documentos.append(ler_arquivo(caminho))
        except ErroFonte as erro:
            raise ErroFonte(f"{caminho.name}: {erro}") from None
    return documentos


def como_registro(doc: DocumentoFonte) -> dict[str, Any]:
    """Forma serializável (JSON) da representação intermediária."""
    return dict(doc)
