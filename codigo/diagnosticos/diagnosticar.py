#!/usr/bin/env python3
"""Diagnósticos reproduzíveis dos dados e controles informados pelas anotações.

Área experimental separada das quatro configurações do RECLin-PT (baseline,
Pair-Aware linear, Two-Stage e Pair-Aware MLP): nada aqui treina, avalia o
TEST, escolhe parâmetros ou altera partições, anotações, predições ou modelos.
As saídas são JSON só com números agregados e ids de documentos (nenhum texto
clínico). Ver `diagnosticos/README.md`.

    python codigo/diagnosticos/diagnosticar.py tudo
    python codigo/diagnosticos/diagnosticar.py tipos            # controles por tipo semântico (TRAIN, DEV)
    python codigo/diagnosticos/diagnosticar.py duplicatas       # detector de quase duplicatas
    python codigo/diagnosticos/diagnosticar.py semente --n 200  # sensibilidade à semente da partição
    python codigo/diagnosticos/diagnosticar.py max-gap          # sensibilidade ao max_gap e origem do "11,6%"
    python codigo/diagnosticos/diagnosticar.py excluir-documentos --predicoes <predicoes_dev.json> --particao dev

Subcomandos:

tipos               controle por tipo (`Negation` como pista) e controle por
                    tipo + posição (pista -> entidade seguinte), só TRAIN e DEV.
duplicatas          pares de documentos com Jaccard >= limiar entre shingles do
                    texto sem espaços, nas três partições; só registra.
semente             o algoritmo de partição com N sementes; contagens por partição
                    e a posição da semente 42.
max-gap             candidatos e cobertura por classe para vários limites (TRAIN e
                    DEV) e a distribuição das distâncias anotadas; o bloco
                    `origem_11_6` conta as relações do corpus inteiro (inclui as
                    anotações do TEST, só como contagem) acima de 20 caracteres.
excluir-documentos  métricas de um sidecar de predições já gravado, com e sem os
                    documentos 9534, 9535 e 9536 (ou outros); no TEST só com
                    `--test-autorizado`, depois da avaliação autorizada do TEST.
tudo                tipos, duplicatas, semente e max-gap em `--pasta`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

CODIGO = Path(__file__).resolve().parents[1]
if str(CODIGO) not in sys.path:
    sys.path.insert(0, str(CODIGO))

from reclin import particoes as particoes_mod  # noqa: E402
from reclin.analises import controles_anotacao, exclusao_documentos, max_gap, quase_duplicatas, \
    sensibilidade_semente  # noqa: E402
from reclin.tarefa import MAX_GAP  # noqa: E402
from reclin.util import caminhos  # noqa: E402
from reclin.util.io import gravar_json, ler_json, sha256_arquivo  # noqa: E402

log = logging.getLogger("reclin.diagnosticos")
PASTA_PADRAO = CODIGO / "resultados" / "diagnosticos"


def _carregar(pasta: Path, nomes) -> tuple[dict, dict]:
    """Partições conferidas contra o MANIFEST e a proveniência (hashes)."""
    divergencias = particoes_mod.conferir_particoes(pasta, nomes)
    if divergencias:
        raise SystemExit("partições não conferem com o MANIFEST:\n  " + "\n  ".join(divergencias))
    docs = {n: particoes_mod.ler_particao(pasta, n) for n in nomes}
    prov = {"pasta_particoes": pasta.relative_to(CODIGO).as_posix() if pasta.is_relative_to(CODIGO) else str(pasta),
            "manifest_sha256": sha256_arquivo(pasta / particoes_mod.ARQUIVO_MANIFESTO),
            "particoes_sha256": {n: sha256_arquivo(particoes_mod.arquivo_particao(pasta, n)) for n in nomes}}
    return docs, prov


def tipos(args) -> dict:
    nomes = tuple(args.particoes)
    for n in nomes:
        controles_anotacao.conferir_particao(n)
    docs, prov = _carregar(args.particoes_dir, nomes)
    return {"proveniencia": prov, **controles_anotacao.relatorio(docs, max_gap=args.max_gap)}


def duplicatas(args) -> dict:
    docs, prov = _carregar(args.particoes_dir, particoes_mod.PARTICOES)
    r = quase_duplicatas.detectar(docs, limiar=args.limiar, k=args.k)
    r["test_com_copia_no_treino"] = quase_duplicatas.documentos_de_test_com_copia_no_treino(r)
    return {"proveniencia": prov, **r}


def semente(args) -> dict:
    docs, prov = _carregar(args.particoes_dir, particoes_mod.PARTICOES)
    todos = [d for n in particoes_mod.PARTICOES for d in docs[n]]
    logging.getLogger("reclin.particoes").setLevel(logging.WARNING)   # uma linha por semente, sem utilidade
    r = sensibilidade_semente.sensibilidade(todos, sementes=range(args.n))
    reproduz = {n: [d["doc_id"] for d in particoes_mod.particionar(todos, seed=particoes_mod.SEED)[n]]
                == [d["doc_id"] for d in docs[n]] for n in particoes_mod.PARTICOES}
    if not args.por_semente:
        r.pop("por_semente")
    return {"proveniencia": prov, "semente_42_reproduz_as_particoes_congeladas": reproduz, **r}


def max_gap_cmd(args) -> dict:
    docs, prov = _carregar(args.particoes_dir, particoes_mod.PARTICOES)
    sel = {n: docs[n] for n in ("train", "dev")}
    corpus = [d for n in particoes_mod.PARTICOES for d in docs[n]]
    origem = max_gap.distancias_anotadas(corpus, limites=(20, 25, 50))
    return {"proveniencia": prov, "max_gap_da_tarefa": MAX_GAP,
            "definicao": "entity_gap simétrico: caracteres entre o fim do primeiro span no texto e o início do "
                         "segundo, qualquer que seja o e1; 0 se os spans se tocam ou se sobrepõem",
            "sensibilidade": {n: max_gap.sensibilidade(d) for n, d in sel.items()},
            "sensibilidade_train_dev": max_gap.sensibilidade(sel["train"] + sel["dev"]),
            "distancias_anotadas": {n: max_gap.distancias_anotadas(d) for n, d in sel.items()},
            "origem_11_6": {"nota": "corpus inteiro (TRAIN+DEV+TEST), relações anotadas com distância simétrica "
                                    "acima de 20; só contagem descritiva, nenhuma escolha",
                            "distancias": origem}}


def excluir_documentos(args) -> dict:
    if args.particao == "test" and not args.test_autorizado:
        raise SystemExit("TEST recusado: a análise de sensibilidade no TEST só roda depois da avaliação "
                         "autorizada do TEST (passe --test-autorizado)")
    docs, prov = _carregar(args.particoes_dir, (args.particao,))
    pred = ler_json(args.predicoes)
    r = exclusao_documentos.metricas_excluindo(pred, docs[args.particao], particao=args.particao,
                                               excluir=args.excluir, permitir_test=args.test_autorizado)
    return {"proveniencia": prov | {"predicoes_sha256": sha256_arquivo(args.predicoes)}, **r}


COMANDOS = {"tipos": (tipos, "controles_tipo_semantico.json"),
            "duplicatas": (duplicatas, "quase_duplicatas.json"),
            "semente": (semente, "sensibilidade_semente.json"),
            "max-gap": (max_gap_cmd, "sensibilidade_max_gap.json")}


def montar_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--particoes-dir", type=Path, default=caminhos.PARTICOES,
                    help="pasta das partições (padrão: a versão da fase)")
    sub = ap.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("tipos")
    p.add_argument("--particoes", nargs="+", default=list(controles_anotacao.PARTICOES_PERMITIDAS),
                   choices=controles_anotacao.PARTICOES_PERMITIDAS)
    p.add_argument("--max-gap", type=int, default=MAX_GAP)
    p = sub.add_parser("duplicatas")
    p.add_argument("--limiar", type=float, default=quase_duplicatas.LIMIAR)
    p.add_argument("--k", type=int, default=quase_duplicatas.K)
    p = sub.add_parser("semente")
    p.add_argument("--n", type=int, default=200, help="sementes 0..n-1 (a 42 entra sempre)")
    p.add_argument("--por-semente", action="store_true", help="grava também as contagens de cada semente")
    sub.add_parser("max-gap")
    p = sub.add_parser("excluir-documentos")
    p.add_argument("--predicoes", type=Path, required=True)
    p.add_argument("--particao", choices=("dev", "test"), required=True)
    p.add_argument("--excluir", nargs="+", default=list(exclusao_documentos.DOCUMENTOS_TEST_COM_COPIA_NO_TREINO))
    p.add_argument("--test-autorizado", action="store_true")
    for p in sub.choices.values():
        p.add_argument("--saida", type=Path, help="arquivo JSON de saída")
    p = sub.add_parser("tudo")
    p.add_argument("--pasta", type=Path, default=PASTA_PADRAO)
    p.add_argument("--n", type=int, default=200)
    return ap


def main(argv=None) -> int:
    args = montar_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.comando == "tudo":
        padrao = {"particoes": list(controles_anotacao.PARTICOES_PERMITIDAS), "max_gap": MAX_GAP,
                  "limiar": quase_duplicatas.LIMIAR, "k": quase_duplicatas.K, "por_semente": False}
        for nome, (func, arquivo) in COMANDOS.items():
            res = func(argparse.Namespace(**padrao, particoes_dir=args.particoes_dir, n=args.n))
            gravar_json(args.pasta / arquivo, res)
            log.info("%s -> %s", nome, args.pasta / arquivo)
        return 0
    if args.comando == "excluir-documentos":
        res = excluir_documentos(args)
    else:
        res = COMANDOS[args.comando][0](args)
    saida = args.saida or (PASTA_PADRAO / (COMANDOS[args.comando][1] if args.comando in COMANDOS
                                           else "exclusao_documentos.json"))
    gravar_json(saida, res)
    log.info("-> %s", saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
