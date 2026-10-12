#!/usr/bin/env python3
"""Tabelas, curvas, figuras, valores e relatório da fase, a partir de execuções verificadas.

    python codigo/scripts/gerar_tcc.py inventario [--saida inventario.json]
    python codigo/scripts/gerar_tcc.py resultados [--natureza cientifica]
    python codigo/scripts/gerar_tcc.py curvas
    python codigo/scripts/gerar_tcc.py eda
    python codigo/scripts/gerar_tcc.py relatorio
    python codigo/scripts/gerar_tcc.py tudo

Opções comuns: `--execucoes` (padrão `codigo/resultados/execucoes`),
`--particoes` (as congeladas), `--codigo-referencia` (a identidade do código
exigida das execuções científicas; padrão: a do código atual, D6), `--destino` (padrão
`codigo/resultados/relatorios/<natureza>`), `--natureza` (`cientifica`, o
padrão; `validacao` para a rodada de validação técnica, com os nomes do
protocolo; `fora_da_fase` para execuções técnicas, como as do modelo minúsculo).

inventario  lista as execuções encontradas, a natureza de cada uma, os
            problemas de proveniência e os itens recusados (por exemplo,
            arquivos no formato do legado). Sai com 1 se alguma execução tiver
            problemas.
resultados  tabelas de resultados no TEST (só execuções com o TEST avaliado
            pelo comando próprio), diagnóstico no DEV, comparações do
            protocolo no TEST (bootstrap pareado e McNemar, recalculados das
            predições), limiares τ e t, proveniência e partições.
curvas      histórico por época de cada execução treinada (e a figura, se o
            matplotlib estiver instalado).
eda         tabelas (e figuras) do corpus: partições, cobertura, distâncias,
            tipos de entidade, proporção dos rótulos e sensibilidade ao
            max_gap (TRAIN e DEV).
relatorio   relatório Markdown da fase (`relatorio.md`), com o inventário, as
            tabelas e as pendências.
tudo        eda, resultados, curvas e relatorio.

Um sistema sem execução, sem a partição avaliada ou só com execução de outra
natureza aparece como `pendente` — nada é preenchido com valores de outra
origem, e execuções de naturezas diferentes nunca entram no mesmo destino.
Nenhum resultado do legado é lido: o inventário recusa as referências do
legado e os arquivos no formato antigo.

Gravado em `--destino`: `tabelas/<nome>.tex|.csv|.md`, `valores/<grupo>.json|.tex`
(o registro de valores que o texto do TCC cita com `\\rv{chave}`),
`indice/<grupo>.json` (status, pendências e proveniência de cada tabela),
`comparacoes/*.json`, `figuras/`, `inventario.json` e `relatorio.md`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin import particoes
from reclin.analises import corpus, max_gap
from reclin.avaliacao import significancia
from reclin.execucao import inventario as inv
from reclin.relatorios import curvas, eda, fase, figuras, markdown, tabelas
from reclin.util import caminhos
from reclin.util.io import gravar_json, gravar_texto, sha256_arquivo
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.gerar_tcc")


def _inventario(args) -> inv.Inventario:
    inventario = inv.inventariar(args.execucoes, pasta_particoes=args.particoes,
                                 codigo_referencia=args.codigo_referencia)
    gravar_json(args.destino / "inventario.json", inventario.resumo())
    return inventario


def _sistemas(args):
    return fase.sistemas_validacao() if args.natureza == "validacao" else fase.sistemas()


def _treinadas(args):
    return fase.treinadas_validacao() if args.natureza == "validacao" else fase.treinadas()


def cmd_inventario(args) -> int:
    inventario = inv.inventariar(args.execucoes, pasta_particoes=args.particoes,
                                 codigo_referencia=args.codigo_referencia)
    resumo = inventario.resumo()
    log.info("Identidade do código de referência: %s", resumo["codigo_referencia"])
    if args.saida:
        gravar_json(args.saida, resumo)
    for nome, e in resumo["execucoes"].items():
        log.info("%-40s %-18s %-12s %s%s", nome, e["estrategia"], e["natureza"],
                 ",".join(e["particoes"]) or "-", "" if e["valida"] else "  PROBLEMAS: " + "; ".join(e["problemas"]))
    for r in resumo["recusados"]:
        log.warning("recusado: %s (%s)", r["caminho"], r["motivo"])
    if not resumo["execucoes"]:
        log.info("Nenhuma execução do código novo em %s", args.execucoes)
    return 0 if all(e["valida"] for e in resumo["execucoes"].values()) else 1


def gerar_resultados(args, inventario: inv.Inventario) -> list[tabelas.Tabela]:
    sistemas = _sistemas(args)
    lista = [tabelas.resultados(inventario, "test", natureza=args.natureza, sistemas=sistemas),
             tabelas.resultados(inventario, "dev", natureza=args.natureza, sistemas=sistemas)]
    comp, relatorios = tabelas.comparacoes(inventario, natureza=args.natureza, sistemas=sistemas,
                                           n_boot=args.n_boot)
    lista += [comp, tabelas.limiares(inventario, natureza=args.natureza, sistemas=sistemas)]
    usadas = [inventario.execucoes[o["execucao"]] for t in lista for o in t.origens]
    usadas = list({ex.nome: ex for ex in usadas}.values())
    lista += [tabelas.proveniencia(usadas, args.natureza), tabelas.particoes_tabela(args.particoes)]
    for nome, r in relatorios.items():
        significancia.gravar_relatorio(args.destino / "comparacoes" / f"{nome}.json", r)
    tabelas.gravar(lista, args.destino, grupo="resultados",
                   extras={"n_boot": args.n_boot, "codigo_referencia": inventario.codigo_referencia})
    for t in lista:
        log.info("%-20s %-9s %s", t.nome, t.status, "; ".join(t.pendencias))
    return lista


def gerar_curvas(args, inventario: inv.Inventario) -> list[tabelas.Tabela]:
    lista, historicos = curvas.tabelas(inventario, natureza=args.natureza, treinadas=_treinadas(args))
    extras = {"figuras": []}
    if historicos and figuras.disponivel():
        extras["figuras"] = [caminhos.relativo_ao_codigo(c) for c in
                             figuras.curvas(historicos, args.destino / "figuras")]
    elif historicos:
        extras["pendencias"] = ["figuras das curvas: matplotlib não instalado"]
        log.warning("matplotlib não instalado: figuras das curvas não geradas")
    tabelas.gravar(lista, args.destino, grupo="curvas", extras=extras)
    for t in lista:
        log.info("%-40s %-9s %s", t.nome, t.status, "; ".join(t.pendencias))
    return lista


def gerar_eda(args) -> list[tabelas.Tabela]:
    divergencias = particoes.conferir_particoes(args.particoes)
    if divergencias:
        raise tabelas.ErroTabela("; ".join(divergencias))
    docs = particoes.ler_particoes(args.particoes)
    sha = sha256_arquivo(args.particoes / "MANIFEST.json")
    estat = corpus.estatisticas(docs)
    sens = max_gap.sensibilidade(docs["train"] + docs["dev"])
    lista = [eda.corpus(estat, sha), eda.cobertura(estat, sha), eda.distancias(estat, sha, "train"),
             eda.entidades(estat, sha, "train"), eda.candidatos_por_rotulo(estat, sha),
             eda.max_gap(sens, sha, "train_dev")]
    gravar_json(args.destino / "analises" / "corpus.json", {"manifest_sha256": sha, "estatisticas": estat})
    gravar_json(args.destino / "analises" / "max_gap.json", {"manifest_sha256": sha, "particoes": ["train", "dev"],
                                                             "sensibilidade": sens})
    extras = {"figuras": []}
    if figuras.disponivel():
        contagens = {p.upper(): {r: estat[p]["candidatos_por_rotulo"][r] for r in ("negation_of", "associated_with")}
                     for p in particoes.PARTICOES}
        extras["figuras"] = [caminhos.relativo_ao_codigo(c) for c in figuras.barras(
            contagens, args.destino / "figuras", nome="eda_relacoes_candidatas", rotulo_y="Pares candidatos")]
    else:
        extras["pendencias"] = ["figuras da EDA: matplotlib não instalado"]
        log.warning("matplotlib não instalado: figuras da EDA não geradas")
    tabelas.gravar(lista, args.destino, grupo="eda", extras=extras)
    for t in lista:
        log.info("%-28s %s", t.nome, t.status)
    return lista


def _executar(args, *, resultados: bool, curvas_: bool, eda_: bool, relatorio: bool) -> int:
    try:
        inventario = _inventario(args)
        geradas = []
        if eda_:
            geradas += gerar_eda(args)
        if resultados or relatorio:
            geradas += gerar_resultados(args, inventario)
        if curvas_ or relatorio:
            geradas += gerar_curvas(args, inventario)
        if relatorio:
            ordem = [t for t in geradas if t.natureza != "dados"] + [t for t in geradas if t.natureza == "dados"]
            texto = markdown.relatorio(inventario, ordem, natureza=args.natureza,
                                       titulo="Relatório da fase BioBERTpt, seed 42")
            gravar_texto(args.destino / "relatorio.md", texto)
            log.info("Relatório em %s", args.destino / "relatorio.md")
    except (tabelas.ErroTabela, inv.ErroInventario) as erro:
        log.error("%s", erro)
        return 1
    log.info("Gravado em %s", args.destino)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="comando", required=True)
    comandos = {}
    for nome, ajuda in (("inventario", "execuções encontradas e a proveniência de cada uma"),
                        ("resultados", "tabelas de resultados, comparações, limiares e proveniência"),
                        ("curvas", "curvas por época das execuções treinadas"),
                        ("eda", "tabelas e figuras do corpus"),
                        ("relatorio", "relatório Markdown da fase"),
                        ("tudo", "eda, resultados, curvas e relatório")):
        p = sub.add_parser(nome, help=ajuda)
        p.add_argument("--execucoes", type=Path, default=caminhos.RESULTADOS / "execucoes")
        p.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
        p.add_argument("--natureza", choices=inv.NATUREZAS, default="cientifica")
        p.add_argument("--destino", type=Path, default=None)
        p.add_argument("--n-boot", type=int, default=significancia.N_BOOT, help=argparse.SUPPRESS)
        p.add_argument("--codigo-referencia", default=None, metavar="ID",
                       help="identidade do código exigida das execuções científicas (padrão: a do código atual)")
        comandos[nome] = p
    comandos["inventario"].add_argument("--saida", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.destino is None:
        args.destino = caminhos.RESULTADOS / "relatorios" / args.natureza
    configurar()
    if args.comando == "inventario":
        return cmd_inventario(args)
    return _executar(args, resultados=args.comando in ("resultados", "tudo"),
                     curvas_=args.comando in ("curvas", "tudo"), eda_=args.comando in ("eda", "tudo"),
                     relatorio=args.comando in ("relatorio", "tudo"))


if __name__ == "__main__":
    sys.exit(main())
