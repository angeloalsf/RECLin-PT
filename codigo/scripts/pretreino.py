#!/usr/bin/env python3
"""Pesos pré-treinados com proveniência: download num commit exato e conferências.

    python codigo/scripts/pretreino.py baixar --revisao <SHA de 40 caracteres> \\
        [--repo pucpr/biobertpt-all] [--destino codigo/modelos_pretreinados/biobertpt-all@<SHA>]
    python codigo/scripts/pretreino.py conferir --pasta codigo/modelos_pretreinados/biobertpt-all@<SHA>
    python codigo/scripts/pretreino.py conferir --execucoes DIR [DIR ...]

baixar     baixa o snapshot do commit (só SHA completo; branch e tag são
           recusados) e grava `PROVENIENCIA.json` com o commit, o SHA-256 e o
           tamanho de cada arquivo. É o único comando que acessa o Hugging Face
           Hub. A pasta de destino fica fora do git (.gitignore).
conferir   --pasta: recalcula os hashes de todos os arquivos contra o
           manifesto. --execucoes: a checagem antes da comparação final — cada
           execução precisa ser do código novo, não pode ser de validação
           técnica, precisa ter a configuração do protocolo (a de cada
           estratégia; todas com a mesma semente) e todas precisam ter o mesmo
           commit e o mesmo hash de manifesto. Execuções derivadas (filtro,
           baseline calibrado, composição do Two-Stage) são conferidas pelas
           execuções de que dependem. Sai com código 1 em qualquer divergência.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from reclin import pretreino
from reclin.config import Config
from reclin.estrategias import pair_aware, restrito, two_stage
from reclin.execucao import validacao
from reclin.util import caminhos
from reclin.util.io import ler_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.pretreino")

# Configuração do protocolo de cada estratégia treinada (a semente é conferida à parte).
PROTOCOLO = {
    "baseline": Config, "restrito": restrito.ConfigRestrito, "pair_aware": pair_aware.ConfigPairAware,
    "two_stage_e1": two_stage.ConfigEstagio1, "two_stage_e2": two_stage.ConfigEstagio2,
}
DERIVADAS = ("filtro_pistas", "baseline_calibrado", "two_stage")


class Divergencia(ValueError):
    pass


def _config_execucao(pasta: Path) -> dict:
    if not (pasta / "config.json").is_file():
        raise Divergencia(f"{pasta} não é uma execução do código novo (sem config.json)")
    return ler_json(pasta / "config.json")


def _treinada(pasta: Path, sementes: dict) -> dict:
    """Confere uma execução treinada e devolve a proveniência dela."""
    registro = _config_execucao(pasta)["config"]
    estrategia = registro.get("estrategia")
    if estrategia not in PROTOCOLO:
        raise Divergencia(f"{pasta.name}: estratégia {estrategia!r} sem configuração de protocolo")
    if validacao.da_config(registro):
        raise validacao.ExecucaoDeValidacao(f"{pasta.name} é uma execução de validação técnica")
    classe = PROTOCOLO[estrategia]
    gravada = registro["config"]
    esperada = dataclasses.replace(classe(), seed=gravada.get("seed"), encoder=gravada.get("encoder", "biobertpt"))
    esperada = esperada.como_dict()
    if gravada != esperada:
        campos = sorted(k for k in set(gravada) | set(esperada) if gravada.get(k) != esperada.get(k))
        raise Divergencia(f"{pasta.name}: configuração diferente da do protocolo em {campos}")
    sementes[pasta.name] = gravada["seed"]
    return registro.get("proveniencia_pesos")


def conferir_execucoes(pastas: list[Path]) -> dict:
    registros, sementes = {}, {}
    por_nome = {p.name: p for p in pastas}
    for pasta in pastas:
        conteudo = _config_execucao(pasta)
        registro = conteudo["config"]
        estrategia = conteudo.get("estrategia", registro.get("estrategia"))
        if validacao.da_config(registro):
            raise validacao.ExecucaoDeValidacao(f"{pasta.name} é uma execução de validação técnica")
        if estrategia in DERIVADAS:
            if estrategia == "filtro_pistas":
                dependencias = [registro["base"]]
            elif estrategia == "baseline_calibrado":
                dependencias = [registro["base"]["nome"]]
            else:
                dependencias = [registro["estagios"]["1"]["nome"], registro["estagios"]["2"]["nome"]]
            faltando = [d for d in dependencias if d not in por_nome]
            if faltando:
                raise Divergencia(f"{pasta.name} depende de {faltando}: inclua essas execuções na conferência")
            continue
        registros[pasta.name] = _treinada(pasta, sementes)
    if len(set(sementes.values())) > 1:
        raise Divergencia(f"as execuções têm sementes diferentes: {sementes}")
    return {"proveniencia": pretreino.mesma_proveniencia(registros), "sementes": sorted(set(sementes.values()))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("baixar", help="baixa os pesos num commit exato e grava o manifesto")
    p.add_argument("--repo", default=pretreino.REPOSITORIO_BIOBERTPT)
    p.add_argument("--revisao", required=True, help="SHA completo do commit (40 caracteres hexadecimais)")
    p.add_argument("--destino", type=Path, default=None,
                   help="pasta nova (padrão: codigo/modelos_pretreinados/<repo>@<SHA>)")
    p = sub.add_parser("conferir", help="confere uma pasta de pesos ou as execuções da comparação final")
    grupo = p.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--pasta", type=Path)
    grupo.add_argument("--execucoes", type=Path, nargs="+")
    args = ap.parse_args(argv)
    configurar()
    try:
        if args.comando == "baixar":
            pretreino.validar_commit(args.revisao)
            destino = args.destino or (caminhos.CODIGO / "modelos_pretreinados"
                                       / f"{args.repo.split('/')[-1]}@{args.revisao}")
            manifesto = pretreino.baixar(args.repo, args.revisao, destino)
            log.info("Baixado %s@%s em %s: %d arquivos, manifesto %s", args.repo, args.revisao, destino,
                     len(manifesto["arquivos"]), manifesto["hash_manifesto"])
        elif args.pasta is not None:
            registro = pretreino.conferir(args.pasta)
            log.info("Confere: %s@%s, manifesto %s", registro["repositorio"], registro["commit"],
                     registro["hash_manifesto"])
        else:
            resultado = conferir_execucoes(args.execucoes)
            prov = resultado["proveniencia"]
            log.info("As %d execuções conferem: mesmo commit %s, mesmo manifesto %s, semente %s",
                     len(args.execucoes), prov["commit"], prov["hash_manifesto"], resultado["sementes"])
    except (pretreino.ErroProveniencia, validacao.ExecucaoDeValidacao, Divergencia, KeyError) as erro:
        log.error("%s", erro)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
