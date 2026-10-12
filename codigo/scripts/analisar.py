#!/usr/bin/env python3
"""Análises descritivas: léxico, corpus, max_gap, alinhamento, marcadores e falsos positivos.

    python codigo/scripts/analisar.py lexico
    python codigo/scripts/analisar.py lexico --min-freq 2 --saida lexico.json
    python codigo/scripts/analisar.py corpus --saida analises/corpus.json
    python codigo/scripts/analisar.py max-gap --saida analises/max_gap.json
    python codigo/scripts/analisar.py alinhamento --saida analises/alinhamento.json
    python codigo/scripts/analisar.py marcadores [--modelo PASTA] --saida analises/marcadores.json
    python codigo/scripts/analisar.py falsos-positivos --execucao filtro_biobertpt_seed42 \\
        [--execucoes codigo/resultados/execucoes] [--particao dev] --saida analises/fp.json

lexico            induz o léxico do TRAIN e mostra as formas com a frequência, o
                  `lexico_sha1` e a cobertura em cada partição (fração dos pares
                  candidatos `negation_of` cujo e1 é pista). Sem `--min-freq`,
                  carrega o léxico congelado e confere que é o registrado em
                  `reclin.negacao.lexico.CONGELADO` (falha se não for).
corpus            estatísticas das três partições (`reclin.analises.corpus`).
max-gap           sensibilidade ao limite de distância, sobre TRAIN e DEV (o TEST
                  só com `--incluir-test`; o `max_gap` da tarefa já está fixado).
alinhamento       offsets das entidades contra o texto (TRAIN e DEV por padrão).
marcadores        os marcadores nas janelas e, com `--modelo PASTA` (um
                  checkpoint local; com `PROVENIENCIA.json`, conferido e
                  carregado sem rede), depois da tokenização.
falsos-positivos  falsos positivos de `negation_of` de uma execução do código
                  novo, verificada pelo inventário (`reclin.execucao.inventario`):
                  por origem, pista do e1 (o léxico congelado, induzido do TRAIN
                  congelado), distância e forma.
                  O TEST só se a execução teve o TEST avaliado pelo comando próprio.

As partições são conferidas contra o MANIFEST antes de qualquer conta. Nenhum
subcomando lê resultados do legado; o inventário recusa as referências do
legado e os arquivos no formato antigo. Com `--saida`, grava um JSON com o
resultado e a origem (SHA-256 do MANIFEST e, para falsos positivos, a
proveniência da execução).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin import entrada, particoes, tarefa, versao_dados
from reclin.analises import alinhamento, corpus, falsos_positivos, marcadores, max_gap
from reclin.execucao import inventario
from reclin.negacao import lexico
from reclin.tarefa import MAX_GAP
from reclin.util import caminhos
from reclin.util.io import gravar_json, sha256_arquivo
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.analisar")


def _conferidas(pasta: Path, nomes) -> dict[str, list] | None:
    divergencias = particoes.conferir_particoes(pasta, nomes)
    if divergencias:
        for d in divergencias:
            log.error("%s", d)
        return None
    return {n: particoes.ler_particao(pasta, n) for n in nomes}


def _origem(pasta: Path, nomes) -> dict:
    return {"particoes": list(nomes), "manifest_sha256": sha256_arquivo(pasta / "MANIFEST.json"),
            "particoes_congeladas": sha256_arquivo(pasta / "MANIFEST.json")
            == sha256_arquivo(caminhos.PARTICOES / "MANIFEST.json"),
            "versao_dados": versao_dados.versao_da_pasta(pasta)}


def _gravar(args, resultado: dict) -> None:
    if args.saida:
        gravar_json(args.saida, resultado)
        log.info("Gravado em %s", args.saida)


def cmd_lexico(args) -> int:
    divergencias = particoes.conferir_particoes(args.particoes)
    if divergencias:
        for d in divergencias:
            log.error("%s", d)
        return 1
    docs = particoes.ler_particoes(args.particoes)
    if args.min_freq is None:
        lex = lexico.carregar_congelado(docs["train"])
        min_freq, origem = lexico.CONGELADO["min_freq"], "congelado"
    else:
        lex = lexico.induzir(docs["train"], min_freq=args.min_freq, max_gap=args.max_gap)
        min_freq, origem = args.min_freq, "induzido"

    sha = lexico.lexico_sha1(lex)
    log.info("Léxico %s: min_freq=%d, max_gap=%d, %d formas, lexico_sha1=%s",
             origem, min_freq, args.max_gap, len(lex), sha)
    for forma, n in lex.items():
        log.info("  %5d  %s", n, forma)
    coberturas = {}
    for nome in particoes.PARTICOES:
        coberturas[nome] = lexico.cobertura(docs[nome], lex, args.max_gap)
        c = coberturas[nome]
        log.info("Cobertura %-5s: %d de %d pares negation_of com e1 no léxico = %.4f",
                 nome, c["cobertos"], c["negation_of"], c["cobertura"])
    if args.saida:
        gravar_json(args.saida, {"origem": origem, "min_freq": min_freq, "max_gap": args.max_gap,
                                 "n_formas": len(lex), "lexico_sha1": sha, "formas": lex,
                                 "cobertura": coberturas})
        log.info("Gravado em %s", args.saida)
    return 0


def cmd_corpus(args) -> int:
    docs = _conferidas(args.particoes, particoes.PARTICOES)
    if docs is None:
        return 1
    estat = corpus.estatisticas(docs, max_gap=args.max_gap)
    for nome in particoes.PARTICOES:
        s = estat[nome]
        log.info("%-5s: %d documentos, %d entidades, relações %s, %d candidatos %s", nome.upper(),
                 s["n_documentos"], s["n_entidades"], s["relacoes_por_tipo"], s["n_candidatos"],
                 s["candidatos_por_rotulo"])
    _gravar(args, {"origem": _origem(args.particoes, particoes.PARTICOES), "estatisticas": estat})
    return 0


def _particoes_pedidas(args) -> list[str]:
    nomes = list(args.particao)
    if getattr(args, "incluir_test", False) and "test" not in nomes:
        nomes.append("test")
    if "test" in nomes and not getattr(args, "incluir_test", False):
        raise SystemExit("o TEST só entra com --incluir-test")
    return nomes


def cmd_max_gap(args) -> int:
    nomes = _particoes_pedidas(args)
    docs = _conferidas(args.particoes, nomes)
    if docs is None:
        return 1
    resultado = max_gap.sensibilidade([d for n in nomes for d in docs[n]], args.limites)
    for linha in resultado["linhas"]:
        cob = linha["cobertura"]
        log.info("max_gap=%-4d %8d candidatos | cobertura negation_of %.4f associated_with %.4f%s",
                 linha["max_gap"], linha["n_candidatos"], cob.get("negation_of", {}).get("fracao") or 0,
                 cob.get("associated_with", {}).get("fracao") or 0, "  (tarefa)" if linha["da_tarefa"] else "")
    _gravar(args, {"origem": _origem(args.particoes, nomes), "sensibilidade": resultado})
    return 0


def cmd_alinhamento(args) -> int:
    nomes = _particoes_pedidas(args)
    docs = _conferidas(args.particoes, nomes)
    if docs is None:
        return 1
    resultado = {n: {**alinhamento.auditar(docs[n], janela=args.janela),
                     "pares": alinhamento.impacto_nos_pares(docs[n], janela=args.janela),
                     "perfil_da_deriva": alinhamento.perfil_da_deriva(docs[n], janela=args.janela)}
                 for n in nomes}
    for n, r in resultado.items():
        log.info("%-5s: %d entidades | %s | exatas %.4f", n.upper(), r["n_entidades"], r["status"],
                 r["fracao_exata"] or 0)
        log.info("%-5s: pares com entidade desalinhada %s | documentos %s", n.upper(),
                 {k: v["afetados"] for k, v in r["pares"].items()}, r["perfil_da_deriva"]["documentos"])
    _gravar(args, {"origem": _origem(args.particoes, nomes), "alinhamento": resultado})
    return 0


def cmd_marcadores(args) -> int:
    nomes = _particoes_pedidas(args)
    docs = _conferidas(args.particoes, nomes)
    if docs is None:
        return 1
    resultado = {"janelas": {n: marcadores.auditar_janelas(docs[n]) for n in nomes}}
    for n in nomes:
        log.info("Janelas %-5s: %s", n.upper(), resultado["janelas"][n])
    origem = _origem(args.particoes, nomes)
    if args.modelo is not None:
        from reclin import modelos, pretreino
        try:
            registro = pretreino.registro_de(args.modelo)
        except pretreino.ErroProveniencia as erro:
            log.error("%s", erro)
            return 1
        if registro is not None:
            pretreino.ativar_offline()
        origem["modelo"] = {"pasta": caminhos.relativo_ao_codigo(args.modelo), "proveniencia_pesos": registro}
        tokenizer = modelos.carregar_tokenizer(args.modelo)
        resultado["tokenizacao"] = {}
        for n in nomes:
            ex = entrada.exemplos(docs[n])
            resultado["tokenizacao"][n] = marcadores.auditar_tokenizacao(
                tokenizer, ex.textos, ex.rotulos, max_length=args.max_length)
            r = resultado["tokenizacao"][n]
            log.info("Tokenização %-5s (max_length=%d): sem [E1] %s | sem [E2] %s", n.upper(), args.max_length,
                     r["sem_e1"], r["sem_e2"])
    _gravar(args, {"origem": origem, "marcadores": resultado})
    return 0


def cmd_falsos_positivos(args) -> int:
    try:
        inv = inventario.inventariar(args.execucoes, pasta_particoes=args.particoes)
        ex = inv.obter(args.execucao)
    except inventario.ErroInventario as erro:
        log.error("%s", erro)
        return 1
    if not ex.valida:
        log.error("%s tem problemas: %s", ex.nome, "; ".join(ex.problemas))
        return 1
    if not ex.tem(args.particao):
        log.error("%s não tem predições utilizáveis de %s%s", ex.nome, args.particao,
                  " (o TEST precisa ter sido avaliado pelo comando próprio)" if args.particao == "test" else "")
        return 1
    docs = _conferidas(args.particoes, [args.particao])
    congeladas = _conferidas(caminhos.PARTICOES, ["train"])
    if docs is None or congeladas is None:
        return 1
    lex = lexico.carregar_congelado(congeladas["train"])      # o léxico congelado, do TRAIN congelado
    cands = tarefa.candidatos(docs[args.particao], max_gap=MAX_GAP)
    por_id = {d["doc_id"]: d for d in docs[args.particao]}
    resultado = falsos_positivos.auditar(cands, ex.predicoes(args.particao), por_id,
                                         e_pista=lambda e: lexico.e_pista(e, lex))
    r = resultado["resumo"]
    log.info("%s [%s, natureza %s]: TP=%d FP=%d | por origem %s | e1 pista %s", ex.nome, args.particao,
             ex.natureza, r["tp"], r["fp"], r["por_origem"], r.get("e1_pista"))
    _gravar(args, {"origem": {**_origem(args.particoes, [args.particao]), "execucao": ex.origem(),
                              "lexico_sha1": lexico.lexico_sha1(lex)},
                   "particao": args.particao, "falsos_positivos": resultado})
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("lexico", help="léxico de pistas de negação")
    p.add_argument("--min-freq", type=int, default=None,
                   help="induz com este limiar em vez de carregar o congelado")
    p.add_argument("--max-gap", type=int, default=MAX_GAP)

    p_corpus = sub.add_parser("corpus", help="estatísticas das partições")
    p_corpus.add_argument("--max-gap", type=int, default=MAX_GAP)

    p_gap = sub.add_parser("max-gap", help="sensibilidade ao limite de distância")
    p_gap.add_argument("--limites", type=int, nargs="+", default=list(max_gap.LIMITES))

    p_al = sub.add_parser("alinhamento", help="offsets das entidades contra o texto")
    p_al.add_argument("--janela", type=int, default=alinhamento.JANELA)

    p_mk = sub.add_parser("marcadores", help="marcadores nas janelas e depois da tokenização")
    p_mk.add_argument("--modelo", type=Path, default=None, help="pasta local de um checkpoint (tokenizer)")
    p_mk.add_argument("--max-length", type=int, default=128)

    p_fp = sub.add_parser("falsos-positivos", help="falsos positivos de negation_of de uma execução")
    p_fp.add_argument("--execucao", required=True, help="nome da execução (pasta em --execucoes)")
    p_fp.add_argument("--execucoes", type=Path, default=caminhos.RESULTADOS / "execucoes")
    p_fp.add_argument("--particao", choices=("dev", "test"), default="dev")

    for s in (p_gap, p_al, p_mk):
        s.add_argument("--particao", nargs="+", choices=particoes.PARTICOES, default=["train", "dev"])
        s.add_argument("--incluir-test", action="store_true", help="inclui o TEST (só descrição)")
    for s in (p, p_corpus, p_gap, p_al, p_mk, p_fp):
        s.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
        s.add_argument("--saida", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.comando == "lexico" and args.min_freq is None and args.max_gap != MAX_GAP:
        ap.error("o léxico congelado usa max_gap=25; para outro valor, informe --min-freq")

    configurar()
    return {"lexico": cmd_lexico, "corpus": cmd_corpus, "max-gap": cmd_max_gap, "alinhamento": cmd_alinhamento,
            "marcadores": cmd_marcadores, "falsos-positivos": cmd_falsos_positivos}[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
