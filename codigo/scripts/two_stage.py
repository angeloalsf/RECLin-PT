#!/usr/bin/env python3
"""Two-Stage: verificação da inicialização, treino de cada estágio, escolha de τ no DEV e TEST.

    python codigo/scripts/two_stage.py verificar-inicializacao --modelo PASTA_DO_PRETREINO [--saida R.json]
    python codigo/scripts/two_stage.py treinar --estagio 1 --modelo PASTA_DO_PRETREINO [--retomar] ...
    python codigo/scripts/two_stage.py treinar --estagio 2 --modelo PASTA_DO_PRETREINO [--retomar] ...
    python codigo/scripts/two_stage.py compor-dev --estagio1 EXEC_E1 --estagio2 EXEC_E2
    python codigo/scripts/two_stage.py avaliar-test --execucao EXEC_COMPOSICAO

verificar-inicializacao  monta os dois estágios como o treino montaria (mesma
                         semente) e faz as três verificações: encoder igual
                         ao arquivo de pesos do commit, cabeça nova, marcadores
                         com os ids do baseline. Não treina nada.
treinar                  treina (ou retoma) um estágio a partir dos pesos
                         ORIGINAIS (pasta baixada por scripts/pretreino.py, com
                         PROVENIENCIA.json; um checkpoint ajustado é recusado).
                         Lê só TRAIN e DEV. Nome padrão:
                         twostage_e<N>_<encoder>_seed<S>.
compor-dev               roda o pipeline completo no DEV para cada τ da grade
                         pré-registrada, escolhe τ (F1 de negation_of; desempates
                         do protocolo), congela-o em limiar.json e grava o DEV.
                         Não lê o TEST. Nome padrão: twostage_<encoder>_seed<S>.
avaliar-test             confere os vínculos do τ congelado e avalia o TEST uma
                         vez (de novo só com --reavaliar, registrado na trilha).

Validação técnica: `treinar --validacao-epocas N` treina N épocas no lugar das
do protocolo e marca a execução (nome padrão twostage_e<N>_validacao_<N>ep). A
composição de estágios de validação também é marcada (nome padrão
twostage_validacao) e nunca tem o TEST avaliado.

Saída 0 concluído; 3 treino interrompido (há checkpoint para retomar); 1 erro.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from reclin.config import CLASS_WEIGHTS, ENCODERS
from reclin.estrategias import two_stage
from reclin.util import caminhos
from reclin.util.io import gravar_json
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.two_stage")

INTERROMPIDO = 3


def _tipo(campo: dataclasses.Field, valor):
    if valor is not None:
        return type(valor)
    return int if "int" in str(campo.type) else float


def _opcoes_config(p: argparse.ArgumentParser, classe: type) -> None:
    padrao = classe()
    for campo in dataclasses.fields(classe):
        opcao = "--" + campo.name.replace("_", "-")
        valor = getattr(padrao, campo.name)
        if campo.name == "encoder":
            p.add_argument(opcao, choices=sorted(ENCODERS), default=valor)
        elif campo.name == "class_weight":
            p.add_argument(opcao, choices=CLASS_WEIGHTS, default=valor)
        else:
            p.add_argument(opcao, type=_tipo(campo, valor), default=valor, help=f"padrão: {valor}")


def montar_parser(estagio: int = 1) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("verificar-inicializacao", help="verifica a inicialização dos dois estágios")
    p.add_argument("--modelo", type=Path, required=True, help="pasta do pré-treino (com PROVENIENCIA.json)")
    p.add_argument("--seed", type=int, default=two_stage.ConfigEstagio1().seed)
    p.add_argument("--saida", type=Path, default=None, help="grava o relatório neste JSON")

    p = sub.add_parser("treinar", help="treina (ou retoma) um estágio")
    p.add_argument("--estagio", type=int, choices=two_stage.ESTAGIOS, required=True)
    p.add_argument("--modelo", type=Path, required=True, help="pasta do pré-treino (com PROVENIENCIA.json)")
    p.add_argument("--nome", default=None)
    p.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "execucoes")
    p.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    _opcoes_config(p, two_stage.CONFIGS[estagio])
    p.add_argument("--validacao-epocas", type=int, default=None, metavar="N",
                   help="execução de validação técnica com N épocas (marcada; nunca avalia o TEST)")
    p.add_argument("--retomar", action="store_true")
    p.add_argument("--checkpoint-a-cada", type=int, default=None, metavar="N")
    p.add_argument("--parar-apos-passo", type=int, default=None, metavar="N")
    p.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=None)

    p = sub.add_parser("compor-dev", help="escolhe e congela τ no DEV")
    p.add_argument("--estagio1", type=Path, required=True)
    p.add_argument("--estagio2", type=Path, required=True)
    p.add_argument("--nome", default=None)
    p.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "execucoes")
    p.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    p.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=None)

    p = sub.add_parser("avaliar-test", help="avalia o TEST com o τ congelado, uma vez")
    p.add_argument("--execucao", type=Path, required=True)
    p.add_argument("--estagio1", type=Path, default=None, help="pasta do estágio 1, se tiver mudado de lugar")
    p.add_argument("--estagio2", type=Path, default=None, help="pasta do estágio 2, se tiver mudado de lugar")
    p.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    p.add_argument("--reavaliar", action="store_true")
    p.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=None)
    return ap


def _preparar(args) -> None:
    configurar()
    from reclin import pretreino
    pretreino.ativar_offline()
    from reclin.treino import reprodutibilidade
    reprodutibilidade.configurar_ambiente()
    if getattr(args, "threads", None):
        import torch
        torch.set_num_threads(args.threads)


def cmd_verificar(args) -> int:
    from reclin.estrategias.two_stage import treino
    from reclin.treino import reprodutibilidade
    relatorio = {}
    for estagio in two_stage.ESTAGIOS:
        reprodutibilidade.fixar_sementes(args.seed)
        _, _, rel, prov = treino.carregar_modelo(args.modelo, estagio)
        relatorio[f"estagio{estagio}"] = rel
        relatorio["proveniencia_pesos"] = prov
        log.info("Estágio %d: encoder igual ao arquivo de pesos (%d tensores, %s…); cabeça nova %s; "
                 "marcadores %s; pooler novo: %s", estagio, rel["encoder"]["n_tensores"],
                 rel["encoder"]["sha256"][:12], rel["cabeca"]["formato"], rel["marcadores"]["ids"],
                 rel["pooler"]["novo"])
    relatorio["seed"] = args.seed
    if relatorio["estagio1"]["encoder"]["sha256"] != relatorio["estagio2"]["encoder"]["sha256"]:
        log.error("os dois estágios não partem do mesmo encoder")
        return 1
    log.info("Proveniência: %s@%s (manifesto %s)", relatorio["proveniencia_pesos"]["repositorio"],
             relatorio["proveniencia_pesos"]["commit"], relatorio["proveniencia_pesos"]["hash_manifesto"])
    if args.saida:
        gravar_json(args.saida, relatorio)
        log.info("Relatório gravado em %s", args.saida)
    return 0


def cmd_treinar(args) -> int:
    from reclin.estrategias.two_stage import treino
    from reclin.execucao import validacao
    classe = two_stage.CONFIGS[args.estagio]
    try:
        config = classe(**{c.name: getattr(args, c.name) for c in dataclasses.fields(classe)})
    except ValueError as erro:
        log.error("Configuração inválida: %s", erro)
        return 1
    for opcao in ("checkpoint_a_cada", "parar_apos_passo", "threads"):
        valor = getattr(args, opcao)
        if valor is not None and valor <= 0:
            log.error("--%s precisa ser positivo: %s", opcao.replace("_", "-"), valor)
            return 1
    marca = None
    if args.validacao_epocas is not None:
        if args.validacao_epocas <= 0:
            log.error("--validacao-epocas precisa ser positivo: %s", args.validacao_epocas)
            return 1
        if config.epochs != classe().epochs:
            log.error("--validacao-epocas não se combina com --epochs: o protocolo (%d épocas) não muda",
                      classe().epochs)
            return 1
        marca = validacao.bloco(args.validacao_epocas, classe().epochs)
        config = dataclasses.replace(config, epochs=args.validacao_epocas)
        nome = args.nome or validacao.nome(two_stage.PREFIXO_VALIDACAO[args.estagio], args.validacao_epocas)
    else:
        nome = args.nome or two_stage.nome_execucao(args.estagio, config)
    registro = treino.treinar_estagio(
        args.saida, nome, args.estagio, config, modelo=args.modelo, pasta_particoes=args.particoes,
        retomar=args.retomar, checkpoint_a_cada=args.checkpoint_a_cada, parar_apos_passo=args.parar_apos_passo,
        dispositivo_pedido=None if args.dispositivo == "auto" else args.dispositivo, marca_validacao=marca)
    if not registro["concluido"]:
        log.info("Interrompido em %s: para continuar, rode o mesmo comando com --retomar", registro["posicao"])
        return INTERROMPIDO
    log.info("Estágio %d concluído: melhor época %s (%s=%.4f)", args.estagio, registro["melhor_epoca"],
             registro["campo_selecao"], registro["melhor_valor_selecao"])
    return 0


def cmd_compor(args) -> int:
    from reclin.estrategias.two_stage import composicao, treino
    from reclin.util.io import ler_json
    nome = args.nome
    if nome is None:
        registro = ler_json(args.estagio1 / "config.json")["config"] if (args.estagio1 / "config.json").is_file() \
            else None
        if registro and registro.get("validacao"):
            nome = two_stage.NOME_COMPOSICAO_VALIDACAO
        elif registro:
            nome = two_stage.nome_composicao(two_stage.ConfigEstagio1(**{
                k: v for k, v in registro["config"].items() if k not in ("tipo", "modelo")}))
        else:
            nome = "twostage"
    resultado = composicao.compor_dev(args.estagio1, args.estagio2, args.saida, nome,
                                      pasta_particoes=args.particoes,
                                      dispositivo=treino.dispositivo(None if args.dispositivo == "auto"
                                                                     else args.dispositivo))
    m = resultado["metricas"]
    log.info("τ congelado: %.2f | DEV (fluxo real): F1 negation_of=%.4f | macro-F1=%.4f%s", resultado["tau"],
             m["f1_per_class"]["negation_of"], m["macro_f1"],
             " | VALIDAÇÃO TÉCNICA (não avalia desempenho)" if resultado["validacao"] else "")
    log.info("Gravado em %s", args.saida / nome)
    return 0


def cmd_avaliar(args) -> int:
    from reclin.estrategias.two_stage import composicao, treino
    resultado = composicao.avaliar_test(
        args.execucao, estagio1=args.estagio1, estagio2=args.estagio2, pasta_particoes=args.particoes,
        reavaliar=args.reavaliar,
        dispositivo=treino.dispositivo(None if args.dispositivo == "auto" else args.dispositivo))
    if resultado["trilha"]["eval_index_for_config"] > 1:
        log.warning("Esta é a avaliação %d do TEST para esta configuração: declare a multiplicidade.",
                    resultado["trilha"]["eval_index_for_config"])
    return 0


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr): getattr(fluxo, "reconfigure", lambda **_: None)(errors="replace")  # "τ" no --help sob cp1252
    previo = argparse.ArgumentParser(add_help=False)
    previo.add_argument("--estagio", type=int, choices=two_stage.ESTAGIOS, default=1)
    estagio = previo.parse_known_args(argv)[0].estagio
    args = montar_parser(estagio).parse_args(argv)
    _preparar(args)
    from reclin import pretreino
    from reclin.estrategias.two_stage import inicializacao
    from reclin.execucao import validacao
    from reclin.treino import checkpoint, classificador
    comandos = {"verificar-inicializacao": cmd_verificar, "treinar": cmd_treinar, "compor-dev": cmd_compor,
                "avaliar-test": cmd_avaliar}
    try:
        return comandos[args.comando](args)
    except (classificador.ErroExecucao, checkpoint.ErroRetomada, inicializacao.ErroInicializacao,
            pretreino.ErroProveniencia, validacao.ExecucaoDeValidacao, ValueError) as erro:
        log.error("%s", erro)
        return 1


if __name__ == "__main__":
    sys.exit(main())
