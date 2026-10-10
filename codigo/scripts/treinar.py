#!/usr/bin/env python3
"""Treina (ou retoma) uma estratégia treinada. Não lê nem avalia o TEST.

    python codigo/scripts/treinar.py --estrategia baseline --encoder biobertpt --seed 42 \\
        --checkpoint-a-cada 500
    python codigo/scripts/treinar.py --estrategia restrito --encoder biobertpt --seed 42 --retomar
    python codigo/scripts/treinar.py --estrategia pair_aware --help    # opções da Pair-Aware

Estratégias (`reclin.estrategias`): `baseline` (todos os candidatos, cabeça
[CLS]), `restrito` (só os pares cujo e1 é pista do léxico congelado; época
escolhida no DEV remapeado) e `pair_aware` (o restrito com a cabeça
[h_cls ; h_E1 ; h_E2]). O nome padrão da execução é o do legado:
`baseline_<encoder>_seed<N>`, `restrito_<encoder>_seed<N>`,
`pairaware_<encoder>_seed<N>`.

Cria a execução `<saida>/<nome>` (padrão: `codigo/resultados/execucoes/`) com
`config.json`, `treino.json` (histórico do DEV, melhor época, sessões),
`checkpoints/` (estado de retomada e `melhor_modelo/`) e, ao concluir,
`predicoes_dev.json` (DEV na melhor época, no conjunto completo) e as
métricas do DEV.

Os hiperparâmetros têm como padrão os da configuração da estratégia
(`Config`, `ConfigRestrito`, `ConfigPairAware`: os dos experimentos); cada um
pode ser mudado pela opção de mesmo nome, e `--help` depois de `--estrategia`
mostra os da estratégia escolhida. `--modelo` usa um checkpoint local no
lugar do de `--encoder` (por exemplo, o modelo minúsculo das referências,
para testar em CPU).

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

from reclin.config import CLASS_WEIGHTS, ENCODERS
from reclin.estrategias import baseline, pair_aware, restrito
from reclin.util import caminhos
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.treinar")

INTERROMPIDO = 3

ESTRATEGIAS = {
    baseline.NOME: (baseline, baseline.Config),
    restrito.NOME: (restrito, restrito.ConfigRestrito),
    pair_aware.NOME: (pair_aware, pair_aware.ConfigPairAware),
}


def _tipo(campo: dataclasses.Field, valor):
    """O tipo da opção: o do valor padrão; para campos opcionais (None), o anotado."""
    if valor is not None:
        return type(valor)
    return int if "int" in str(campo.type) else float


def montar_parser(estrategia: str = baseline.NOME) -> argparse.ArgumentParser:
    classe = ESTRATEGIAS[estrategia][1]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--estrategia", choices=sorted(ESTRATEGIAS), default=baseline.NOME,
                    help="estratégia treinada (padrão: baseline)")
    ap.add_argument("--nome", default=None,
                    help="nome da execução (pasta dentro de --saida); padrão: <estratégia>_<encoder>_seed<N>")
    ap.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "execucoes")
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--modelo", default=None,
                    help="checkpoint local (pasta) no lugar do de --encoder")
    padrao = classe()
    for campo in dataclasses.fields(classe):
        opcao = "--" + campo.name.replace("_", "-")
        valor = getattr(padrao, campo.name)
        if campo.name == "encoder":
            ap.add_argument(opcao, choices=sorted(ENCODERS), default=valor)
        elif campo.name == "class_weight":
            ap.add_argument(opcao, choices=CLASS_WEIGHTS, default=valor)
        else:
            ap.add_argument(opcao, type=_tipo(campo, valor), default=valor, help=f"padrão: {valor}")
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
    previo = argparse.ArgumentParser(add_help=False)
    previo.add_argument("--estrategia", choices=sorted(ESTRATEGIAS), default=baseline.NOME)
    escolhida = previo.parse_known_args(argv)[0].estrategia
    args = montar_parser(escolhida).parse_args(argv)
    modulo, classe = ESTRATEGIAS[args.estrategia]
    configurar()
    from reclin.treino import reprodutibilidade
    reprodutibilidade.configurar_ambiente()
    try:
        config = classe(**{c.name: getattr(args, c.name) for c in dataclasses.fields(classe)})
    except ValueError as erro:
        log.error("Configuração inválida: %s", erro)
        return 1
    nome = args.nome or modulo.nome_execucao(config)
    for opcao in ("checkpoint_a_cada", "parar_apos_passo", "threads"):
        valor = getattr(args, opcao)
        if valor is not None and valor <= 0:
            log.error("--%s precisa ser positivo: %s", opcao.replace("_", "-"), valor)
            return 1
    if args.modelo is not None and not Path(args.modelo).is_dir():
        log.error("--modelo %s não é uma pasta com um checkpoint", args.modelo)
        return 1

    import torch
    from reclin import particoes
    from reclin.negacao import lexico
    from reclin.treino import checkpoint, classificador
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        if modulo is baseline:
            montagem = baseline.montagem(config)
        else:     # restrito e Pair-Aware: o léxico congelado, induzido do TRAIN
            divergencias = particoes.conferir_particoes(args.particoes, ("train",))
            if divergencias:
                raise classificador.ErroExecucao("partições não conferem com o MANIFEST: "
                                                 + "; ".join(divergencias))
            lex = lexico.carregar_congelado(particoes.ler_particao(args.particoes, "train"))
            montagem = modulo.montagem(config, lexico=lex)
        registro = classificador.treinar_execucao(
            args.saida, nome, config, modelo=args.modelo, pasta_particoes=args.particoes,
            retomar=args.retomar, checkpoint_a_cada=args.checkpoint_a_cada,
            parar_apos_passo=args.parar_apos_passo, montagem=montagem,
            dispositivo=None if args.dispositivo == "auto" else args.dispositivo)
    except (classificador.ErroExecucao, checkpoint.ErroRetomada, ValueError) as erro:
        log.error("%s", erro)
        return 1
    if not registro["concluido"]:
        log.info("Interrompido em %s: para continuar, rode o mesmo comando com --retomar",
                 registro["posicao"])
        return INTERROMPIDO
    log.info("Melhor época: %s (dev_macro_f1=%.4f). Para avaliar o TEST: "
             "scripts/avaliar_test.py --execucao %s", registro["melhor_epoca"],
             registro["melhor_dev_macro_f1"], args.saida / nome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
