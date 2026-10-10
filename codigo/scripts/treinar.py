#!/usr/bin/env python3
"""Treina (ou retoma) o classificador da tarefa. Não lê nem avalia o TEST.

    python codigo/scripts/treinar.py --nome classificador_biobertpt_seed42 \\
        --encoder biobertpt --seed 42 --checkpoint-a-cada 500
    python codigo/scripts/treinar.py --nome classificador_biobertpt_seed42 \\
        --encoder biobertpt --seed 42 --checkpoint-a-cada 500 --retomar

Cria a execução `<saida>/<nome>` (padrão: `codigo/resultados/execucoes/`) com
`config.json`, `treino.json` (histórico do DEV, melhor época, sessões),
`checkpoints/` (estado de retomada e `melhor_modelo/`) e, ao concluir,
`predicoes_dev.json` (DEV na melhor época) e as métricas do DEV.

Os hiperparâmetros têm como padrão os de `reclin.config.Config` (os dos
experimentos); cada um pode ser mudado pela opção de mesmo nome. `--modelo`
usa um checkpoint local no lugar do de `--encoder` (por exemplo, o modelo
minúsculo das referências, para testar em CPU).

Retomada: um checkpoint é gravado no fim de cada época e, com
`--checkpoint-a-cada N`, a cada N passos de treino. Se o treino for
interrompido, rode o MESMO comando com `--retomar`: ele continua do último
checkpoint e produz o mesmo resultado que uma execução sem interrupção (no
mesmo ambiente). Sem `--retomar`, uma execução existente não é tocada; com
`--retomar`, uma configuração diferente da gravada é recusada.
`--parar-apos-passo N` interrompe de propósito depois do passo global N (para
testar a retomada).

O TEST é avaliado por outro comando, depois de concluído o treino:
`scripts/avaliar_test.py --execucao <saida>/<nome>`.

Saída 0 com o treino concluído; 3 com o treino interrompido (há checkpoint para
retomar); 1 em erro de configuração ou de retomada.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from reclin.config import CLASS_WEIGHTS, ENCODERS, Config
from reclin.util import caminhos
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.treinar")

INTERROMPIDO = 3


def montar_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--nome", required=True, help="nome da execução (pasta dentro de --saida)")
    ap.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "execucoes")
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--modelo", default=None,
                    help="checkpoint local (pasta) no lugar do de --encoder")
    padrao = Config()
    for campo in dataclasses.fields(Config):
        opcao = "--" + campo.name.replace("_", "-")
        valor = getattr(padrao, campo.name)
        if campo.name == "encoder":
            ap.add_argument(opcao, choices=sorted(ENCODERS), default=valor)
        elif campo.name == "class_weight":
            ap.add_argument(opcao, choices=CLASS_WEIGHTS, default=valor)
        else:
            ap.add_argument(opcao, type=type(valor), default=valor, help=f"padrão: {valor}")
    ap.add_argument("--retomar", action="store_true", help="continua a execução do último checkpoint")
    ap.add_argument("--checkpoint-a-cada", type=int, default=None, metavar="N",
                    help="grava também um checkpoint a cada N passos de treino")
    ap.add_argument("--parar-apos-passo", type=int, default=None, metavar="N",
                    help="interrompe depois do passo global N, com checkpoint (para testar a retomada)")
    ap.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--threads", type=int, default=None,
                    help="threads do torch em CPU (as referências usam 1)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = montar_parser().parse_args(argv)
    configurar()
    from reclin.treino import reprodutibilidade
    reprodutibilidade.configurar_ambiente()
    try:
        config = Config(**{c.name: getattr(args, c.name) for c in dataclasses.fields(Config)})
    except ValueError as erro:
        log.error("Configuração inválida: %s", erro)
        return 1
    for opcao in ("checkpoint_a_cada", "parar_apos_passo", "threads"):
        valor = getattr(args, opcao)
        if valor is not None and valor <= 0:
            log.error("--%s precisa ser positivo: %s", opcao.replace("_", "-"), valor)
            return 1
    if args.modelo is not None and not Path(args.modelo).is_dir():
        log.error("--modelo %s não é uma pasta com um checkpoint", args.modelo)
        return 1

    import torch
    from reclin.treino import checkpoint, classificador
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        registro = classificador.treinar_execucao(
            args.saida, args.nome, config, modelo=args.modelo, pasta_particoes=args.particoes,
            retomar=args.retomar, checkpoint_a_cada=args.checkpoint_a_cada,
            parar_apos_passo=args.parar_apos_passo,
            dispositivo=None if args.dispositivo == "auto" else args.dispositivo)
    except (classificador.ErroExecucao, checkpoint.ErroRetomada) as erro:
        log.error("%s", erro)
        return 1
    if not registro["concluido"]:
        log.info("Interrompido em %s: para continuar, rode o mesmo comando com --retomar",
                 registro["posicao"])
        return INTERROMPIDO
    log.info("Melhor época: %s (dev_macro_f1=%.4f). Para avaliar o TEST: "
             "scripts/avaliar_test.py --execucao %s", registro["melhor_epoca"],
             registro["melhor_dev_macro_f1"], args.saida / args.nome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
