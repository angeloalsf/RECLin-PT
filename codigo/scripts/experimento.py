#!/usr/bin/env python3
"""Operações de infraestrutura dos notebooks de experimento (Colab): ambiente, execuções, Drive, Hugging Face.

    python codigo/scripts/experimento.py ambiente [--modo minimo|fixado] [--exigir-gpu] [--instalar]
    python codigo/scripts/experimento.py listar [--raiz-local DIR] [--raiz-drive DIR]
    python codigo/scripts/experimento.py planejar --chave ARQUIVO.json [--alias NOME_OFICIAL] [--nova-tentativa] [--raiz-local DIR] [--raiz-drive DIR]
    python codigo/scripts/experimento.py estado NOME [--raiz-local DIR]
    python codigo/scripts/experimento.py retomada NOME [--raiz-local DIR] [--raiz-drive DIR]
    python codigo/scripts/experimento.py sincronizar NOME [--raiz-local DIR] [--raiz-drive DIR] [--sem-ultimo]
    python codigo/scripts/experimento.py restaurar NOME [--raiz-local DIR] [--raiz-drive DIR] [--substituir]
    python codigo/scripts/experimento.py publicar NOME --repositorio USUARIO/REPO [--executar] [--incluir-ultimo] [--incluir-test]
    python codigo/scripts/experimento.py recuperar NOME --raiz-local DIR [--executar]

Nenhum comando treina, avalia, calibra nem acessa o TEST. Os que mudam algo
fora do dry-run pedem ação explícita:

ambiente      confere Python, dependências (contra pyproject.toml e
              requirements.txt) e GPU; imprime o plano de instalação.
              `--instalar` executa o pip; sem ele nada é instalado.
listar        execuções locais e do espelho no Drive, com o estado de cada lado.
planejar      dry-run: dado o arquivo JSON da chave de configuração, calcula o
              `config_id`, procura execuções compatíveis (local e Drive) e diz se
              retoma, reaproveita ou cria a próxima tentativa. Não escreve nada.
estado        o `experimento.json` e o estado derivado dos arquivos.
retomada      dry-run: de onde retomar e se ambiente e código são compatíveis.
sincronizar   espelha a execução no Drive (cópia verificada). Escreve só no Drive.
restaurar     traz a execução do Drive para o disco local (cópia verificada).
publicar      plano da publicação no Hugging Face (repositório privado). Só
              com `--executar` envia; precisa do Secret HF_TOKEN com escrita.
recuperar     baixa do Hugging Face uma execução já publicada (a partir da
              referência em `experimento.json`). Só com `--executar`.

Os Secrets (GITHUB_TOKEN, HF_TOKEN) vêm do Colab e nunca são impressos.
Saída 0 = ok; 1 = há problemas ou bloqueios; 2 = erro de uso.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from reclin.experimento import ambiente, ids, persistencia, plano, proveniencia, publicacao_hf, raiz, segredos
from reclin.util import caminhos

RAIZ_DRIVE_PADRAO = os.environ.get("RECLIN_DRIVE", "/content/drive/MyDrive/RECLin-PT")


def _imprimir(objeto) -> None:
    print(segredos.limpar(json.dumps(objeto, ensure_ascii=False, indent=2, sort_keys=True, default=str)))


def _raizes(ap: argparse.ArgumentParser, drive: bool = True) -> None:
    ap.add_argument("--raiz-local", type=Path, default=caminhos.RESULTADOS / "execucoes",
                    help="pasta das execuções no disco local (padrão: codigo/resultados/execucoes)")
    if drive:
        ap.add_argument("--raiz-drive", type=Path, default=Path(RAIZ_DRIVE_PADRAO),
                        help="pasta de persistência no Drive (padrão: $RECLIN_DRIVE ou %(default)s)")


def cmd_ambiente(args) -> int:
    codigo = raiz.localizar_codigo()
    relatorio = ambiente.verificar(codigo, modo=args.modo, exigir_gpu=args.exigir_gpu)
    if args.instalar:
        relatorio["instalacao"] = ambiente.instalar(relatorio["plano"], executar=True)
        relatorio = {**ambiente.verificar(codigo, modo=args.modo, exigir_gpu=args.exigir_gpu),
                     "instalacao": relatorio["instalacao"]}
    relatorio["codigo"] = str(codigo)
    _imprimir(relatorio)
    return 0 if relatorio["ok"] else 1


def cmd_listar(args) -> int:
    drive = args.raiz_drive if args.raiz_drive.is_dir() else None
    _imprimir(persistencia.listar(args.raiz_local, drive))
    return 0


def cmd_planejar(args) -> int:
    chave = json.loads(args.chave.read_text(encoding="utf-8"))
    resultado = plano.planejar(chave, args.raiz_local, args.raiz_drive if args.raiz_drive.is_dir() else None,
                               alias_oficial=args.alias, nova_tentativa=args.nova_tentativa)
    _imprimir(resultado)
    return 0


def cmd_estado(args) -> int:
    pasta = args.raiz_local / args.nome
    saida = {"derivado": proveniencia.derivar_estado(pasta), "problemas": proveniencia.conferir(pasta)}
    try:
        saida["registro"] = proveniencia.ler(pasta)
    except proveniencia.ErroProveniencia as erro:
        saida["registro"] = str(erro)
    _imprimir(saida)
    return 0 if not saida["problemas"] else 1


def cmd_retomada(args) -> int:
    from reclin.util import identidade_codigo  # noqa: PLC0415

    diag = persistencia.diagnosticar_retomada(args.nome, args.raiz_local,
                                              args.raiz_drive if args.raiz_drive.is_dir() else None,
                                              identidade_atual=identidade_codigo.atual()["identidade"])
    _imprimir(diag)
    return 0 if diag["pode_retomar"] else 1


def cmd_sincronizar(args) -> int:
    _imprimir(persistencia.sincronizar_e_registrar(args.raiz_local / args.nome, args.raiz_drive,
                                                   incluir_ultimo=not args.sem_ultimo, progresso=print))
    return 0


def cmd_restaurar(args) -> int:
    _imprimir(persistencia.restaurar(args.raiz_drive, args.nome, args.raiz_local, substituir=args.substituir,
                                     progresso=print))
    return 0


def cmd_publicar(args) -> int:
    pasta = args.raiz_local / args.nome
    credencial = segredos.para_operacao("publicar_hf")[segredos.HUGGING_FACE] if args.executar else None
    resultado = publicacao_hf.publicar(pasta, args.repositorio, credencial=credencial, confirmar=args.executar,
                                       incluir_ultimo=args.incluir_ultimo, incluir_test=args.incluir_test)
    resultado.pop("manifesto", None)
    _imprimir(resultado)
    return 0 if resultado.get("pode_publicar", True) else 1


def cmd_recuperar(args) -> int:
    pasta = args.raiz_local / args.nome
    publicacoes = [p for p in proveniencia.ler(pasta).get("publicacoes", []) if p.get("destino") == "hf"] \
        if (pasta / proveniencia.ARQUIVO).is_file() else []
    if not publicacoes:
        print(f"{args.nome}: sem referência de publicação no Hugging Face em {proveniencia.ARQUIVO}", file=sys.stderr)
        return 1
    referencia = publicacoes[-1]
    credencial = segredos.para_operacao("publicar_hf")[segredos.HUGGING_FACE] if args.executar else None
    _imprimir(publicacao_hf.recuperar(referencia, args.raiz_local, credencial=credencial, confirmar=args.executar))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("ambiente", help="confere ambiente e GPU; plano de instalação")
    p.add_argument("--modo", choices=ambiente.MODOS, default="minimo")
    p.add_argument("--exigir-gpu", action="store_true")
    p.add_argument("--instalar", action="store_true", help="executa o pip do plano (sem isto, só informa)")
    p.set_defaults(func=cmd_ambiente)

    p = sub.add_parser("listar", help="execuções locais e do Drive")
    _raizes(p)
    p.set_defaults(func=cmd_listar)

    p = sub.add_parser("planejar", help="dry-run: config_id, execuções compatíveis e o que fazer")
    p.add_argument("--chave", type=Path, required=True, help="JSON da chave de configuração (plano.chave_da_execucao)")
    p.add_argument("--alias", help="nome oficial do protocolo a atribuir, se a pasta estiver livre")
    p.add_argument("--nova-tentativa", action="store_true", help="recomeçar do zero: a próxima tentativa")
    _raizes(p)
    p.set_defaults(func=cmd_planejar)

    p = sub.add_parser("estado", help="registro e estado derivado de uma execução")
    p.add_argument("nome")
    _raizes(p, drive=False)
    p.set_defaults(func=cmd_estado)

    p = sub.add_parser("retomada", help="dry-run da retomada de uma execução")
    p.add_argument("nome")
    _raizes(p)
    p.set_defaults(func=cmd_retomada)

    p = sub.add_parser("sincronizar", help="espelha a execução no Drive")
    p.add_argument("nome")
    _raizes(p)
    p.add_argument("--sem-ultimo", action="store_true", help="não copia checkpoints/ultimo.pt")
    p.set_defaults(func=cmd_sincronizar)

    p = sub.add_parser("restaurar", help="traz a execução do Drive para o disco local")
    p.add_argument("nome")
    _raizes(p)
    p.add_argument("--substituir", action="store_true", help="o Drive prevalece sobre arquivos locais diferentes")
    p.set_defaults(func=cmd_restaurar)

    p = sub.add_parser("publicar", help="plano (ou envio) para o Hugging Face")
    p.add_argument("nome")
    p.add_argument("--repositorio", required=True, help="usuario/repositorio (privado)")
    _raizes(p, drive=False)
    p.add_argument("--executar", action="store_true", help="envia de fato; sem isto é só o plano")
    p.add_argument("--incluir-ultimo", action="store_true", help="inclui checkpoints/ultimo.pt (não é o padrão)")
    p.add_argument("--incluir-test", action="store_true", help="inclui artefatos do TEST (não é o padrão)")
    p.set_defaults(func=cmd_publicar)

    p = sub.add_parser("recuperar", help="baixa do Hugging Face uma execução publicada")
    p.add_argument("nome")
    _raizes(p, drive=False)
    p.add_argument("--executar", action="store_true", help="baixa de fato; sem isto é só o plano")
    p.set_defaults(func=cmd_recuperar)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (persistencia.ErroSincronizacao, publicacao_hf.ErroPublicacao, segredos.ErroSegredo,
            proveniencia.ErroProveniencia, ambiente.ErroAmbiente, raiz.ErroRaiz, plano.ErroPlano,
            ids.ErroId) as erro:
        print(segredos.limpar(f"erro: {erro}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
