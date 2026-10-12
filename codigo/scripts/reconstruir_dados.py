#!/usr/bin/env python3
"""Reconstrução dos dados a partir dos XML do SemClinBr (versão v2-texto-fonte).

    python codigo/scripts/reconstruir_dados.py inventario [--saida inventario.json]
    python codigo/scripts/reconstruir_dados.py gerar --data AAAA-MM-DD [--sobrescrever]
    python codigo/scripts/reconstruir_dados.py conferir
    python codigo/scripts/reconstruir_dados.py auditar [--saida auditoria.json]
    python codigo/scripts/reconstruir_dados.py comparar [--saida comparacao.json]

inventario  inventário e proveniência da pasta de XML (só lê): arquivos,
            identificadores, SHA-256 de cada arquivo e do conjunto,
            codificação, quebras de linha, erros e anomalias estruturais.
gerar       XML → representação intermediária → dataset (regra C1) →
            partições (semente 42) em `dados/particoes-v2/` com MANIFEST,
            FONTE.json e ANOMALIAS.json; o dataset e a representação
            intermediária vão para `dados/processados/` (fora do git).
            Recusa gravar sobre a v1 e sobre uma v2 existente sem
            `--sobrescrever`; para sem gravar nada se a atribuição dos
            documentos às partições divergir da v1.
conferir    regera a v2 numa pasta temporária e compara byte a byte com a
            gravada (o MANIFEST sem a data de geração). Saída 1 se divergir.
auditar     cada entidade de cada versão contra a anotação de origem (pela
            identidade, nunca por busca de texto) e a auditoria da etapa 7
            (`analises.alinhamento`) antes e depois. TEST: só estrutural.
comparar    invariantes v1 × v2 por partição (documentos, entidades,
            relações, texto, candidatos, rótulos, `y_true`, janelas
            marcadas), conjuntos de referência e o léxico congelado no
            TRAIN das duas versões.

Nenhum comando treina, calibra ou avalia modelos. No TEST, nenhuma contagem
por rótulo é calculada: só identidade de sequências e hashes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from reclin import particoes, tarefa
from reclin.analises import alinhamento, marcadores
from reclin.corpus import auditoria, comparacao, conversao, fonte, geracao, semclinbr
from reclin.negacao import lexico
from reclin.util import caminhos
from reclin.util.io import gravar_json, sha256_arquivo
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.reconstruir_dados")


def _gravar_ou_mostrar(args, obj) -> None:
    if args.saida:
        gravar_json(args.saida, obj)
        log.info("Gravado em %s", args.saida)
    else:
        print(json.dumps(obj, ensure_ascii=False, indent=1))


def cmd_inventario(args) -> int:
    inv = fonte.inventariar(args.xml)
    if args.saida:
        gravar_json(args.saida, inv)
    resumo = {k: v for k, v in inv.items() if k != "arquivos"}
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    return 1 if inv["erros"] else 0


def cmd_gerar(args) -> int:
    try:
        manifesto = geracao.gerar(args.xml, args.destino, dataset=args.dataset, intermediario=args.intermediario,
                                  pasta_anterior=args.anterior, data=args.data, sobrescrever=args.sobrescrever)
    except (geracao.ErroGeracao, conversao.ErroConversao, semclinbr.ErroFonte) as erro:
        log.error("%s", erro)
        return 2
    for nome, s in manifesto["splits"].items():
        log.info("%-5s %d documentos sha256=%s", nome, s["n_records"], s["sha256"])
    log.info("MANIFEST: %s", args.destino / particoes.ARQUIVO_MANIFESTO)
    return 0


def cmd_conferir(args) -> int:
    divergencias = geracao.conferir(args.xml, args.pasta, pasta_anterior=args.anterior)
    for d in divergencias:
        log.error("%s", d)
    if not divergencias:
        log.info("v2 reproduzida byte a byte a partir dos XML (%s)", args.pasta)
    return 1 if divergencias else 0


def _fontes(pasta_xml: Path) -> dict:
    return {f["doc_id"]: f for f in semclinbr.ler_pasta(pasta_xml)}


def cmd_auditar(args) -> int:
    fontes = _fontes(args.xml)
    saida: dict = {"xml_conjunto_sha256": fonte.inventariar(args.xml)["conjunto_sha256"], "versoes": {}}
    for rotulo, pasta in (("v1", args.anterior), ("v2", args.pasta)):
        if particoes.conferir_particoes(pasta):
            log.error("%s não confere com o MANIFEST", pasta)
            return 1
        saida["versoes"][rotulo] = {"pasta": caminhos.relativo_ao_codigo(pasta),
                                    "manifest_sha256": sha256_arquivo(pasta / particoes.ARQUIVO_MANIFESTO)}
        for nome in particoes.PARTICOES:
            docs = particoes.ler_particao(pasta, nome)
            com_rotulos = nome != "test"
            r = {"origem": auditoria.auditar(docs, fontes, com_rotulos=com_rotulos)}
            if com_rotulos:
                etapa7 = alinhamento.auditar(docs)
                r["etapa7"] = {"status": etapa7["status"], "fracao_exata": etapa7["fracao_exata"],
                               "negation_of_afetados": alinhamento.impacto_nos_pares(docs)["negation_of"],
                               "perfil_da_deriva": alinhamento.perfil_da_deriva(docs)["documentos"]}
            saida["versoes"][rotulo][nome] = r
    _gravar_ou_mostrar(args, saida)
    return 0


def cmd_comparar(args) -> int:
    fontes = _fontes(args.xml)
    convs = {f: conversao.converter(d)[1] for f, d in fontes.items()}
    transformados = {f: c["transformado"] for f, c in convs.items()}
    saida: dict = {"particoes": {}, "conjuntos_referencia": {}, "lexico": {}}
    for pasta in (args.anterior, args.pasta):
        if particoes.conferir_particoes(pasta):
            log.error("%s não confere com o MANIFEST", pasta)
            return 1
    v1, v2 = particoes.ler_particoes(args.anterior), particoes.ler_particoes(args.pasta)
    shas = {r: {n: s["sha256"] for n, s in particoes.ler_manifesto(p)["splits"].items()}
            for r, p in (("v1", args.anterior), ("v2", args.pasta))}
    for nome in particoes.PARTICOES:
        r = comparacao.comparar_particao(v1[nome], v2[nome], fontes=fontes, transformados=transformados,
                                         com_rotulos=(nome != "test"))
        r["sistemas"] = dict(sorted(Counter(convs[d["doc_id"]]["sistema"] for d in v2[nome]).items()))
        r["janelas"] = comparacao.comparar_janelas(v1[nome], v2[nome], transformados=transformados)
        if nome != "test":
            r["marcadores"] = {versao: marcadores.auditar_janelas(docs[nome]) for versao, docs in (("v1", v1),
                                                                                                    ("v2", v2))}
        saida["particoes"][nome] = r
        saida["conjuntos_referencia"][nome] = {
            versao: tarefa.conjunto_referencia(nome, docs[nome], particao_sha256=shas[versao][nome])
            for versao, docs in (("v1", v1), ("v2", v2))}
    sobreposicao = {}
    for a in particoes.PARTICOES:
        for b in particoes.PARTICOES:
            if a < b:
                sobreposicao[f"{a}x{b}"] = len({d["doc_id"] for d in v2[a]} & {d["doc_id"] for d in v2[b]})
    saida["sobreposicao_de_documentos_v2"] = sobreposicao
    for versao, docs in (("v1", v1), ("v2", v2)):
        try:
            lex = lexico.carregar_congelado(docs["train"])
            saida["lexico"][versao] = {"confere_com_o_congelado": True, "n_formas": len(lex),
                                       "lexico_sha1": lexico.lexico_sha1(lex)}
        except ValueError as erro:
            saida["lexico"][versao] = {"confere_com_o_congelado": False, "erro": str(erro)}
    _gravar_ou_mostrar(args, saida)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--xml", type=Path, default=caminhos.XML_SEMCLINBR,
                    help="pasta dos XML (somente leitura; default: codigo/SemClinBr-xml-public-v1)")
    ap.add_argument("--anterior", type=Path, default=caminhos.PARTICOES_V1,
                    help="partições da versão anterior, congeladas (default: codigo/dados/particoes)")
    sub = ap.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("inventario", help="inventário e proveniência dos XML")
    p.add_argument("--saida", type=Path)
    p = sub.add_parser("gerar", help="gera a v2 (partições, MANIFEST, FONTE, ANOMALIAS)")
    p.add_argument("--destino", type=Path, default=caminhos.PARTICOES_V2)
    p.add_argument("--dataset", type=Path, default=caminhos.DATASET_V2)
    p.add_argument("--intermediario", type=Path, default=caminhos.INTERMEDIARIO_V2)
    p.add_argument("--data", required=True, help="data da geração (AAAA-MM-DD), registrada no MANIFEST")
    p.add_argument("--sobrescrever", action="store_true")
    for nome, ajuda in (("conferir", "regera a v2 e compara byte a byte"),
                        ("auditar", "entidades × anotações de origem, v1 e v2"),
                        ("comparar", "invariantes v1 × v2")):
        p = sub.add_parser(nome, help=ajuda)
        p.add_argument("--pasta", type=Path, default=caminhos.PARTICOES_V2)
        if nome != "conferir":
            p.add_argument("--saida", type=Path)
    args = ap.parse_args(argv)

    configurar()
    comandos = {"inventario": cmd_inventario, "gerar": cmd_gerar, "conferir": cmd_conferir,
                "auditar": cmd_auditar, "comparar": cmd_comparar}
    return comandos[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
