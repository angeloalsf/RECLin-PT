"""Inventário e proveniência da pasta de XML do SemClinBr (só leitura).

O inventário registra só o que dá para comprovar localmente:

* os arquivos (`*.xml`), o identificador de cada documento (nome do arquivo),
  identificadores repetidos ou não numéricos, outros arquivos na pasta;
* o SHA-256 e o tamanho de cada arquivo, e o hash do **conjunto**, pela regra
  `REGRA_CONJUNTO` (abaixo); arquivos com conteúdo idêntico;
* a codificação (declaração XML, BOM, UTF-8 válido, só ASCII);
* as quebras de linha do arquivo inteiro e do `<TEXT>` (`\\r\\n`, `\\r` e `\\n`
  isolados);
* erros de leitura e anomalias estruturais (`semclinbr.ErroFonte`);
* contagens de anotações e relações e anomalias de anotação que dispensam o
  texto: identificador repetido no documento, offset negativo, `end < start`,
  offset além do fim do texto, relação apontando para anotação inexistente.

Origem, licença e integridade histórica da cópia **não** são verificáveis
aqui: o inventário só repete o nome da pasta e a declaração do README do
projeto, marcada como declaração.

REGRA DO HASH DO CONJUNTO (`sha256-lista-v1`)
---------------------------------------------
SHA-256, em UTF-8, do texto formado por uma linha `<nome do arquivo>\\t<sha256
do arquivo>\\n` para cada XML, na ordem de `semclinbr.arquivos_xml` (numérica
pelo identificador). Muda se qualquer arquivo mudar, entrar ou sair, ou se um
arquivo for renomeado.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from reclin.corpus import semclinbr

REGRA_CONJUNTO = "sha256-lista-v1"
DESCRICAO_REGRA_CONJUNTO = ("SHA-256 (UTF-8) de uma linha '<nome do arquivo>\\t<sha256 do arquivo>\\n' por XML, "
                            "em ordem numérica do identificador")
ORIGEM_DECLARADA = ("declarada, não verificada: corpus SemClinBr (Oliveira et al.), de acesso restrito, "
                    "segundo o README do projeto; nome da pasta local: {pasta}")

_DECLARACAO = re.compile(rb"^<\?xml[^>]*encoding=['\"]([^'\"]+)['\"]")


def sha256_conjunto(arquivos: Iterable[tuple[str, str]]) -> str:
    """Hash do conjunto a partir de pares (nome, sha256) já na ordem da regra."""
    texto = "".join(f"{nome}\t{sha}\n" for nome, sha in arquivos)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _codificacao(conteudo: bytes) -> dict[str, Any]:
    m = _DECLARACAO.match(conteudo.lstrip(b"\xef\xbb\xbf"))
    try:
        conteudo.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    return {"declarada": m.group(1).decode("ascii", "replace") if m else None,
            "bom_utf8": conteudo.startswith(b"\xef\xbb\xbf"), "utf8_valido": utf8,
            "so_ascii": all(b < 128 for b in conteudo)}


def _anomalias_de_anotacao(doc: semclinbr.DocumentoFonte) -> list[dict[str, Any]]:
    """Anomalias que não dependem de decidir a convenção dos offsets."""
    anomalias = []
    ids = Counter(a["id"] for a in doc["anotacoes"])
    for ident, n in sorted(ids.items()):
        if n > 1:
            anomalias.append({"tipo": "id_de_anotacao_repetido", "anotacao": ident, "vezes": n})
    limite = len(doc["texto_bruto"])
    for a in doc["anotacoes"]:
        if a["start"] < 0 or a["end"] < 0:
            anomalias.append({"tipo": "offset_negativo", "anotacao": a["id"], "start": a["start"], "end": a["end"]})
        elif a["end"] < a["start"]:
            anomalias.append({"tipo": "end_antes_de_start", "anotacao": a["id"], "start": a["start"], "end": a["end"]})
        elif a["end"] > limite:
            anomalias.append({"tipo": "offset_alem_do_texto", "anotacao": a["id"], "start": a["start"], "end": a["end"]})
    existentes = set(ids)
    for r in doc["relacoes"]:
        for campo in ("annotation1", "annotation2"):
            if r[campo] not in existentes:
                anomalias.append({"tipo": "relacao_orfa", "relacao": r["ordem"], "campo": campo, "anotacao": r[campo]})
    return anomalias


def descrever_arquivo(caminho: str | Path) -> dict[str, Any]:
    """Entrada do inventário para um arquivo (nenhum trecho do texto)."""
    caminho = Path(caminho)
    with open(caminho, "rb") as f:
        conteudo = f.read()
    entrada: dict[str, Any] = {
        "arquivo": caminho.name, "doc_id": semclinbr.doc_id_do_arquivo(caminho),
        "sha256": hashlib.sha256(conteudo).hexdigest(), "n_bytes": len(conteudo),
        "codificacao": _codificacao(conteudo),
        "quebras_arquivo": semclinbr.contar_quebras(conteudo.decode("utf-8", "replace")),
    }
    try:
        doc = semclinbr.ler_bytes(conteudo, doc_id=entrada["doc_id"], arquivo=caminho.name)
    except semclinbr.ErroFonte as erro:
        entrada.update(erro=str(erro), quebras_texto=None, n_caracteres_texto=None,
                       n_anotacoes=None, n_relacoes=None, anomalias=[])
        return entrada
    entrada.update(erro=None, quebras_texto=doc["quebras"], n_caracteres_texto=len(doc["texto_bruto"]),
                   tem_tab=("\t" in doc["texto_bruto"]),
                   n_anotacoes=len(doc["anotacoes"]), n_relacoes=len(doc["relacoes"]),
                   anomalias=_anomalias_de_anotacao(doc))
    return entrada


def inventariar(pasta: str | Path) -> dict[str, Any]:
    """Inventário completo da pasta (ver o módulo). Só lê."""
    pasta = Path(pasta)
    xml = semclinbr.arquivos_xml(pasta)
    outros = sorted(p.name for p in pasta.iterdir() if p not in set(xml))
    arquivos = [descrever_arquivo(p) for p in xml]

    ids = Counter(a["doc_id"] for a in arquivos)
    por_conteudo: dict[str, list[str]] = {}
    for a in arquivos:
        por_conteudo.setdefault(a["sha256"], []).append(a["arquivo"])
    numericos = sorted(int(a["doc_id"]) for a in arquivos if a["doc_id"].isdigit())
    lacunas = sorted(set(range(numericos[0], numericos[-1] + 1)) - set(numericos)) if numericos else []

    totais: Counter[str] = Counter()
    for a in arquivos:
        totais["arquivos"] += 1
        totais["com_erro"] += a["erro"] is not None
        totais["utf8_valido"] += a["codificacao"]["utf8_valido"]
        totais["bom_utf8"] += a["codificacao"]["bom_utf8"]
        totais["so_ascii"] += a["codificacao"]["so_ascii"]
        if a["erro"] is None:
            q = a["quebras_texto"]
            totais["texto_com_crlf"] += q["crlf"] > 0
            totais["texto_com_cr_isolado"] += q["cr_isolado"] > 0
            totais["texto_com_lf_isolado"] += q["lf_isolado"] > 0
            totais["texto_sem_quebra"] += sum(q.values()) == 0
            totais["texto_com_tab"] += a["tem_tab"]
            totais["anotacoes"] += a["n_anotacoes"]
            totais["relacoes"] += a["n_relacoes"]
            totais["crlf_no_texto"] += q["crlf"]
        qa = a["quebras_arquivo"]
        totais["arquivo_com_crlf"] += qa["crlf"] > 0
        totais["arquivo_com_lf_isolado"] += qa["lf_isolado"] > 0
    anomalias = Counter(x["tipo"] for a in arquivos for x in a["anomalias"])

    return {
        "pasta": pasta.name,
        "origem_declarada": ORIGEM_DECLARADA.format(pasta=pasta.name),
        "versao_parser": semclinbr.VERSAO_PARSER,
        "n_arquivos_xml": len(arquivos),
        "outros_arquivos": outros,
        "doc_ids_repetidos": sorted(i for i, n in ids.items() if n > 1),
        "doc_ids_nao_numericos": sorted(i for i in ids if not i.isdigit()),
        "faixa_doc_ids": [numericos[0], numericos[-1]] if numericos else None,
        "lacunas_na_faixa": {"n": len(lacunas), "primeiras": lacunas[:20]},
        "conteudo_identico": sorted(v for v in por_conteudo.values() if len(v) > 1),
        "codificacoes_declaradas": dict(Counter(a["codificacao"]["declarada"] for a in arquivos)),
        "totais": dict(sorted(totais.items())),
        "anomalias_por_tipo": dict(sorted(anomalias.items())),
        "erros": [{"arquivo": a["arquivo"], "erro": a["erro"]} for a in arquivos if a["erro"]],
        "regra_conjunto": REGRA_CONJUNTO,
        "descricao_regra_conjunto": DESCRICAO_REGRA_CONJUNTO,
        "conjunto_sha256": sha256_conjunto((a["arquivo"], a["sha256"]) for a in arquivos),
        "arquivos": arquivos,
    }


def comparar_ids(inventario: dict[str, Any], esperados: Iterable[str]) -> dict[str, list[str]]:
    """Documentos esperados (por exemplo, os das partições anteriores) sem XML,
    e XML sem documento esperado."""
    presentes = {a["doc_id"] for a in inventario["arquivos"]}
    esperados = set(esperados)
    chave = lambda i: (0, int(i), i) if i.isdigit() else (1, 0, i)          # noqa: E731
    return {"ausentes": sorted(esperados - presentes, key=chave),
            "excedentes": sorted(presentes - esperados, key=chave)}
