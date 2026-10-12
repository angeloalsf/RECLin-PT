"""Geração de uma versão dos dados a partir dos XML (v2-texto-fonte).

    XML (somente leitura)
      └─ semclinbr.ler_pasta ........ representação intermediária
           └─ conversao.converter ... dataset (regra C1) + registro da conversão
                └─ particoes.particionar(seed=42) ... train/dev/test
                     └─ MANIFEST.json, FONTE.json, ANOMALIAS.json

Arquivos gravados:

* `<destino>/{train,dev,test}.jsonl` — partições (forma canônica de
  `util.io.gravar_jsonl`, a mesma da v1);
* `<destino>/MANIFEST.json` — as chaves do MANIFEST da v1 (`generator`,
  `seed`, `source`, `splits`, conferidas por `particoes.conferir_particoes`)
  e as da proveniência: versão, fonte XML, regra de conversão, anomalias,
  contagens, conjuntos de referência, relação com a versão anterior e a
  geração (data e ferramentas);
* `<destino>/FONTE.json` — inventário dos XML (`fonte.inventariar`) e o
  registro da conversão de cada documento (sistema, se o texto foi
  transformado e a justificativa); só identificadores, hashes e contagens;
* `<destino>/ANOMALIAS.json` — as anotações defeituosas da fonte, mantidas no
  dataset, com o tipo e a partição;
* `dataset` e `intermediario` (fora do controle de versão: têm o texto do
  corpus) — o dataset completo antes da partição e a representação
  intermediária, para rastrear XML → intermediária → dataset → partição.

Tudo é determinístico: a mesma pasta de XML produz os mesmos bytes, exceto a
chave `geracao` do MANIFEST (data e versões das ferramentas), que
`conferir` ignora.

A partição é refeita com `particoes.particionar` (semente 42) e, quando a
versão anterior é informada, a atribuição dos documentos às partições tem de
ser idêntica à dela: se não for, a geração para (`ErroGeracao`) sem gravar
nada, porque a regra C1 prevê a mesma atribuição.
"""
from __future__ import annotations

import copy
import platform
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from reclin import particoes, tarefa
from reclin.corpus import comparacao, conversao, fonte, semclinbr
from reclin.util import caminhos
from reclin.util.io import gravar_json, gravar_jsonl, sha256_arquivo, sha256_json

VERSAO = "v2-texto-fonte"
ARQUIVO_FONTE = "FONTE.json"
ARQUIVO_ANOMALIAS = "ANOMALIAS.json"
ARQUIVOS = ("train.jsonl", "dev.jsonl", "test.jsonl", particoes.ARQUIVO_MANIFESTO, ARQUIVO_FONTE, ARQUIVO_ANOMALIAS)


class ErroGeracao(RuntimeError):
    """A geração não pode continuar (destino protegido, invariante violado)."""


def _relativo(caminho: Path) -> str:
    return caminhos.relativo_ao_codigo(caminho)


def _anomalias(conversoes: list[dict[str, Any]], atribuicao: dict[str, str],
               registros: dict[str, tarefa.Documento]) -> list[dict[str, Any]]:
    saida = []
    for c in conversoes:
        particao = atribuicao[c["doc_id"]]
        for a in c["anomalias"]:
            item = {"doc_id": c["doc_id"], "particao": particao, **a}
            if particao == "test":
                item["relacoes_que_a_usam"] = "não inspecionado (TEST)"
            else:
                rels = registros[c["doc_id"]]["relations"]
                item["relacoes_que_a_usam"] = sum(a["anotacao"] in (r["e1_id"], r["e2_id"]) for r in rels)
            saida.append(item)
    return saida


def gerar(pasta_xml: str | Path, destino: str | Path, *, dataset: str | Path, intermediario: str | Path,
          pasta_anterior: str | Path | None = caminhos.PARTICOES_V1, data: str,
          seed: int = particoes.SEED, sobrescrever: bool = False) -> dict[str, Any]:
    """Gera a versão em `destino` e devolve o MANIFEST (ver o módulo)."""
    pasta_xml, destino = Path(pasta_xml), Path(destino)
    dataset, intermediario = Path(dataset), Path(intermediario)
    if pasta_anterior is not None and destino.resolve() == Path(pasta_anterior).resolve():
        raise ErroGeracao(f"o destino {destino} é a pasta da versão anterior, que é congelada")
    if destino.resolve() == caminhos.PARTICOES_V1.resolve():
        raise ErroGeracao("a pasta da v1 (dados/particoes) é congelada e nunca é destino de uma geração")
    existentes = [n for n in ARQUIVOS if (destino / n).exists()]
    if existentes and not sobrescrever:
        raise ErroGeracao(f"{destino} já tem {existentes}: regerar exige sobrescrever")

    inventario = fonte.inventariar(pasta_xml)
    if inventario["erros"]:
        raise ErroGeracao(f"XML com erro de leitura: {inventario['erros'][:3]}")
    fontes = semclinbr.ler_pasta(pasta_xml)
    registros, conversoes = conversao.converter_todos(fontes)
    divisao = particoes.particionar(registros, seed=seed)
    atribuicao = {d["doc_id"]: nome for nome in particoes.PARTICOES for d in divisao[nome]}

    comparacoes = None
    if pasta_anterior is not None:
        if particoes.conferir_particoes(pasta_anterior):
            raise ErroGeracao(f"a versão anterior em {pasta_anterior} não confere com o próprio MANIFEST")
        anteriores = particoes.ler_particoes(pasta_anterior)
        for nome in particoes.PARTICOES:
            if [d["doc_id"] for d in anteriores[nome]] != [d["doc_id"] for d in divisao[nome]]:
                raise ErroGeracao(f"a partição {nome} refeita não tem os mesmos documentos, na mesma ordem, da "
                                  "versão anterior: a regra C1 prevê a mesma atribuição; nada foi gravado")
        transformados = {c["doc_id"]: c["transformado"] for c in conversoes}
        por_id = {f["doc_id"]: f for f in fontes}
        comparacoes = {nome: comparacao.comparar_particao(anteriores[nome], divisao[nome], fontes=por_id,
                                                          transformados=transformados,
                                                          com_rotulos=(nome != "test"))
                       for nome in particoes.PARTICOES}

    # ----- gravação --------------------------------------------------------- #
    destino.mkdir(parents=True, exist_ok=True)
    gravar_jsonl(intermediario, (semclinbr.como_registro(f) for f in fontes))
    gravar_jsonl(dataset, registros)
    particoes.gravar_particoes(divisao, destino)
    por_doc = {c["doc_id"]: {k: v for k, v in c.items() if k != "doc_id"} for c in conversoes}
    gravar_json(destino / ARQUIVO_FONTE, {
        "versao": VERSAO, "versao_parser": semclinbr.VERSAO_PARSER,
        "inventario": {k: v for k, v in inventario.items() if k != "arquivos"},
        "arquivos": inventario["arquivos"],
        "conversao_por_documento": por_doc,
    })
    lista_anomalias = _anomalias(conversoes, atribuicao, {r["doc_id"]: r for r in registros})
    gravar_json(destino / ARQUIVO_ANOMALIAS, {
        "versao": VERSAO,
        "descricao": "anotações da fonte cujo trecho, na posição dos offsets e na convenção do próprio "
                     "documento, não confere com o atributo text; mantidas no dataset sem alteração",
        "tipos": {conversao.PALAVRAS_DUPLICADAS: "atributo com palavras repetidas em sequência; o trecho nos "
                                                 "offsets é o texto sem a repetição",
                  conversao.TRECHO_DIVERGENTE: "atributo diferente do trecho (trecho descontínuo ou palavra "
                                               "omitida)",
                  conversao.OFFSET_INVALIDO: "offset fora do texto (start negativo, end antes de start ou "
                                             "além do fim)",
                  conversao.OFFSET_DENTRO_DE_CRLF: "offset entre o \\r e o \\n de uma quebra de linha"},
        "anomalias": lista_anomalias,
    })

    manifesto = particoes.calcular_manifesto(destino, seed=seed, origem={"path": _relativo(dataset),
                                                                         "sha256": sha256_arquivo(dataset)})
    sistemas = Counter(c["sistema"] for c in conversoes)
    contagens, conjuntos = {}, {}
    for nome in particoes.PARTICOES:
        docs = divisao[nome]
        cands = tarefa.candidatos(docs)
        contagens[nome] = {"documentos": len(docs), "entidades": sum(len(d["entities"]) for d in docs),
                           "relacoes": sum(len(d["relations"]) for d in docs), "candidatos": len(cands),
                           "documentos_transformados": sum(por_doc[d["doc_id"]]["transformado"] for d in docs)}
        conjuntos[nome] = {"max_gap": tarefa.MAX_GAP, "n_candidatos": len(cands),
                           "candidatos_sha256": sha256_json([tarefa.chave_candidato(c) for c in cands]),
                           "y_true_sha256": sha256_json(tarefa.y_true(cands))}
    manifesto.update({
        "versao": VERSAO,
        "conversor": {"modulo": "reclin/corpus", "versao_parser": semclinbr.VERSAO_PARSER,
                      "regra": conversao.REGRA_C1, "sistemas": dict(sorted(sistemas.items())),
                      "representacao_intermediaria": {"path": _relativo(intermediario),
                                                      "sha256": sha256_arquivo(intermediario)}},
        "fonte_xml": {"pasta": inventario["pasta"], "origem_declarada": inventario["origem_declarada"],
                      "n_arquivos": inventario["n_arquivos_xml"], "regra_conjunto": fonte.REGRA_CONJUNTO,
                      "descricao_regra_conjunto": fonte.DESCRICAO_REGRA_CONJUNTO,
                      "conjunto_sha256": inventario["conjunto_sha256"],
                      "inventario": ARQUIVO_FONTE, "inventario_sha256": sha256_arquivo(destino / ARQUIVO_FONTE)},
        "anomalias": {"n": len(lista_anomalias),
                      "por_tipo": dict(sorted(Counter(a["tipo"] for a in lista_anomalias).items())),
                      "por_particao": dict(sorted(Counter(a["particao"] for a in lista_anomalias).items())),
                      "tratamento": "mantidas no dataset como estão (nenhuma descartada ou corrigida)",
                      "arquivo": ARQUIVO_ANOMALIAS, "arquivo_sha256": sha256_arquivo(destino / ARQUIVO_ANOMALIAS)},
        "contagens": contagens,
        "conjuntos_referencia": conjuntos,
        "geracao": {"data": data, "python": platform.python_version(), "implementacao": sys.implementation.name},
    })
    if pasta_anterior is not None:
        pasta_anterior = Path(pasta_anterior)
        manifesto["derivada_de"] = {
            "pasta": _relativo(pasta_anterior),
            "manifest_sha256": sha256_arquivo(pasta_anterior / particoes.ARQUIVO_MANIFESTO),
            "particoes_sha256": {n: s["sha256"] for n, s in particoes.ler_manifesto(pasta_anterior)["splits"].items()},
            "invariantes": {nome: _resumo_invariantes(r) for nome, r in comparacoes.items()},
        }
    particoes.gravar_manifesto(destino, manifesto)
    return manifesto


def _resumo_invariantes(r: dict[str, Any]) -> dict[str, Any]:
    texto = r["texto"]
    return {"mesmos_documentos_mesma_ordem": r["documentos"]["mesmos_e_mesma_ordem"],
            "entidades_identicas": r["entidades_identicas"], "relacoes_identicas": r["relacoes_identicas"],
            "documentos_com_texto_diferente": texto["documentos_com_texto_diferente"],
            "texto_diferente_so_nos_transformados": texto.get("igual_aos_transformados_previstos"),
            "texto_v1_igual_ao_legado_da_fonte": texto.get("antes_igual_ao_legado_da_fonte"),
            "texto_v2_igual_a_regra_aplicada_a_fonte": texto.get("depois_igual_a_regra_aplicada_a_fonte"),
            "comprimento_igual_ao_bruto_nos_transformados": texto.get("comprimento_igual_ao_bruto_nos_transformados"),
            "candidatos_mesmas_chaves_e_ordem": r["candidatos"]["mesmas_chaves_e_ordem"],
            "y_true_identico": r["candidatos"]["y_true_identico"],
            "invariantes_dos_pares_violados": len(r["candidatos"]["invariantes_dos_pares"])}


def sem_geracao(manifesto: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifesto.items() if k != "geracao"}


def conferir(pasta_xml: str | Path, pasta: str | Path, *,
             pasta_anterior: str | Path | None = caminhos.PARTICOES_V1) -> list[str]:
    """Regera a versão numa pasta temporária a partir dos XML e compara com a
    gravada, byte a byte (o MANIFEST sem a chave `geracao`). Devolve as
    divergências (vazio = reproduzida). Não grava nada fora da pasta temporária."""
    pasta = Path(pasta)
    gravado = particoes.ler_manifesto(pasta)
    divergencias = [f"{d} (contra o próprio MANIFEST)" for d in particoes.conferir_particoes(pasta)]
    with tempfile.TemporaryDirectory(prefix="reclin_v2_") as tmp:
        tmp = Path(tmp)
        refeito = gerar(pasta_xml, tmp / "particoes", dataset=tmp / "dataset.jsonl",
                        intermediario=tmp / "intermediario.jsonl", pasta_anterior=pasta_anterior,
                        data=gravado.get("geracao", {}).get("data", ""))
        for nome in ARQUIVOS:
            if nome == particoes.ARQUIVO_MANIFESTO:
                continue
            if not (pasta / nome).is_file():
                divergencias.append(f"{nome}: ausente em {pasta}")
            elif sha256_arquivo(pasta / nome) != sha256_arquivo(tmp / "particoes" / nome):
                divergencias.append(f"{nome}: regerado com outro conteúdo")
        a, b = _sem_caminhos(sem_geracao(gravado)), _sem_caminhos(sem_geracao(refeito))
        if a != b:
            chaves = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            divergencias.append(f"MANIFEST.json: chaves diferentes na regeração: {chaves}")
        shutil.rmtree(tmp, ignore_errors=True)
    return divergencias


def _sem_caminhos(manifesto: dict[str, Any]) -> dict[str, Any]:
    """O MANIFEST sem os caminhos dos arquivos fora do controle de versão
    (que, na regeração, ficam numa pasta temporária); os hashes continuam."""
    m = copy.deepcopy(manifesto)
    m.get("source", {}).pop("path", None)
    m.get("conversor", {}).get("representacao_intermediaria", {}).pop("path", None)
    return m

