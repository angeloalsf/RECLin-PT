#!/usr/bin/env python3
"""Seleção do `min_freq` do léxico de pistas só com o TRAIN da v2 (pendência D4).

    python codigo/scripts/selecionar_min_freq.py --particoes PASTA --saida resultado.json

Segue o pré-registro `codigo/docs/protocolo/d4-min-freq.md` (validação
cruzada em 5 partes por documento; detector "o e1 é pista" nos pares
candidatos; F1 pelas contagens somadas; empate → maior `min_freq`). Antes de
calcular, confere:

* que as constantes do código (`reclin.negacao.selecao_min_freq`) são as do
  bloco JSON do pré-registro;
* que a pasta é da v2 (`versao_dados`) e que o `train.jsonl` confere com o
  MANIFEST e com o registro de versões.

Só o `train.jsonl` é aberto: a pasta pode (e, na execução registrada, deve)
conter apenas o `train.jsonl` e o `MANIFEST.json`. Nenhum modelo, nenhuma
predição, nenhum resultado histórico, nenhum DEV ou TEST. O resultado grava
as contagens, o valor escolhido, o léxico final e a proveniência (SHA-256 do
TRAIN, do MANIFEST, do pré-registro e dos arquivos de código).
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
from pathlib import Path

from reclin import particoes, versao_dados
from reclin.negacao import selecao_min_freq as sel
from reclin.util import caminhos
from reclin.util.io import gravar_json, sha256_arquivo
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.selecionar_min_freq")

PRE_REGISTRO = caminhos.CODIGO / "docs" / "protocolo" / "d4-min-freq.md"
CODIGO_DO_PROCEDIMENTO = ("reclin/negacao/selecao_min_freq.py", "reclin/negacao/lexico.py", "reclin/tarefa.py",
                          "scripts/selecionar_min_freq.py")


class ErroSelecao(ValueError):
    pass


def registro_do_pre_registro(caminho: Path = PRE_REGISTRO) -> dict:
    blocos = re.findall(r"```json\n(.*?)\n```", caminho.read_text(encoding="utf-8"), flags=re.S)
    if len(blocos) != 1:
        raise ErroSelecao(f"{caminho}: esperado um bloco JSON verificável")
    return json.loads(blocos[0])


def conferir_constantes(pre: dict) -> list[str]:
    """Divergências entre o pré-registro e as constantes do código."""
    esperado = {"candidatos_min_freq": list(sel.CANDIDATOS), "max_gap": sel.MAX_GAP_SELECAO,
                "n_partes": sel.N_PARTES, "semente": sel.SEMENTE, "estratificacao": sel.ESTRATIFICACAO,
                "metrica": sel.METRICA, "desempate": sel.DESEMPATE, "particao": "train",
                "versao_dados": versao_dados.V2,
                "train_sha256": versao_dados.REGISTRO[versao_dados.V2]["particoes"]["train"],
                "manifest_sha256": versao_dados.REGISTRO[versao_dados.V2]["manifest_sha256"]}
    return [f"{k}: pré-registro {pre.get(k)!r}, código {v!r}" for k, v in esperado.items() if pre.get(k) != v]


def conferir_pasta(pasta: Path) -> str:
    """A pasta é da v2 e o TRAIN confere; devolve o SHA-256 do TRAIN."""
    if versao_dados.versao_da_pasta(pasta) != versao_dados.V2:
        raise ErroSelecao(f"{pasta}: o MANIFEST não é o da v2 ({versao_dados.V2})")
    divergencias = particoes.conferir_particoes(pasta, ["train"])
    if divergencias:
        raise ErroSelecao("; ".join(divergencias))
    sha = sha256_arquivo(particoes.arquivo_particao(pasta, "train"))
    if sha != versao_dados.REGISTRO[versao_dados.V2]["particoes"]["train"]:
        raise ErroSelecao("o train.jsonl não é o registrado para a v2")
    return sha


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES_V2,
                    help="pasta com o train.jsonl e o MANIFEST.json da v2 (só o TRAIN é lido)")
    ap.add_argument("--saida", type=Path, required=True, help="JSON com o resultado e a proveniência")
    ap.add_argument("--pre-registro", type=Path, default=PRE_REGISTRO)
    args = ap.parse_args(argv)
    configurar()

    try:
        pre = registro_do_pre_registro(args.pre_registro)
        divergencias = conferir_constantes(pre)
        if divergencias:
            raise ErroSelecao("o código não segue o pré-registro: " + "; ".join(divergencias))
        sha_train = conferir_pasta(args.particoes)
    except (ErroSelecao, OSError, ValueError) as erro:
        log.error("%s", erro)
        return 2
    lidas = [p.name for p in sorted(args.particoes.iterdir()) if p.suffix == ".jsonl"]
    train = particoes.ler_particao(args.particoes, "train")
    resultado = sel.selecionar(train)
    resultado["proveniencia"] = {
        "pendencia": "D4", "versao_dados": versao_dados.V2,
        "train_sha256": sha_train,
        "manifest_sha256": sha256_arquivo(args.particoes / particoes.ARQUIVO_MANIFESTO),
        "particoes_presentes_na_pasta": lidas,
        "pre_registro": caminhos.relativo_ao_codigo(args.pre_registro),
        "pre_registro_sha256": sha256_arquivo(args.pre_registro),
        "codigo_sha256": {r: sha256_arquivo(caminhos.CODIGO / r) for r in CODIGO_DO_PROCEDIMENTO},
        "python": platform.python_version(),
        "modelos_ou_predicoes_usados": False,
    }
    gravar_json(args.saida, resultado)
    for m in resultado["candidatos"]:
        t = resultado["resultados"][str(m)]["total"]
        log.info("min_freq=%-2d VP=%d FP=%d FN=%d F1=%.6f (%s)", m, t["vp"], t["fp"], t["fn"], t["f1"],
                 t["f1_fracao"])
    lf = resultado["lexico_final"]
    log.info("Escolhido min_freq=%d: léxico do TRAIN inteiro com %d formas, lexico_sha1 %s",
             resultado["min_freq_escolhido"], lf["n_formas"], lf["lexico_sha1"])
    log.info("Gravado em %s", args.saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
