#!/usr/bin/env python3
"""Executa o filtro de pistas sobre as predições de uma execução de base.

Dois modos, que não se misturam:

**Modo da fase atual** (`--base-execucao`), o único usado nos experimentos novos:

    python codigo/scripts/filtro_pistas.py --base-execucao codigo/resultados/execucoes/baseline_biobertpt_seed42 \
        --particao dev        # passo 3 do protocolo: cria filtro_biobertpt_seed42 com o DEV
    python codigo/scripts/filtro_pistas.py --base-execucao codigo/resultados/execucoes/baseline_biobertpt_seed42 \
        --particao test       # passo 7: o TEST, uma vez, depois de o TEST da base ter sido avaliado

O léxico é o congelado (`reclin.negacao.lexico.CONGELADO`, induzido do TRAIN
congelado e conferido pelo `lexico_sha1`), como fixa o protocolo da fase
(seções 1 e 3); nenhum arquivo de calibração é lido. A base precisa ser uma
execução do baseline do código novo (`config.json`, mesma identidade do código
atual); sidecars soltos, o formato do legado e `codigo/testes/referencia/` são
recusados. Cada sidecar do filtro grava o SHA-256 das predições da base de onde
veio (`base_sha256`). O DEV cria a execução e grava as métricas; o TEST só é
aplicado a uma execução já criada no DEV, com a mesma configuração, se o TEST da
base já foi avaliado pelo comando próprio (trilha), nunca numa base de
validação técnica, uma vez (de novo só com `--reavaliar`): grava as predições,
as métricas e a linha da trilha `avaliacoes_test.jsonl` — é a avaliação do TEST
do filtro.

**Modo de reprodução do legado** (`--resultados` e `--calibracao`), só para a
reprodução técnica das etapas anteriores (testes de equivalência):

    python codigo/scripts/filtro_pistas.py \
        --resultados codigo/testes/referencia/resultados_legado \
        --calibracao codigo/testes/referencia/resultados_legado/CALIBRACAO_filtro.json \
        --saida execucoes --conferir codigo/testes/referencia/resultados_legado

Para cada base (por padrão, os quatro baselines do legado), cria a execução
`filtro_<encoder>_seed<N>` em `--saida` e grava as predições filtradas de cada
partição pedida (DEV e TEST, por padrão), procurando as da base em
`--resultados`, no formato novo ou no do legado. `min_freq` vem do arquivo de
calibração, conferido contra o léxico congelado. As execuções gravadas nesse
modo trazem o arquivo de calibração no `config.json` e são recusadas pelo
inventário dos experimentos atuais (`reclin.execucao.inventario`). Com
`--conferir PASTA`, compara os bytes de cada sidecar com o da mesma execução em
PASTA; saída 1 se algum diferir.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reclin import particoes, tarefa, versao_dados
from reclin.avaliacao import metricas
from reclin.estrategias import filtro_pistas
from reclin.execucao import diretorio, inventario, predicoes, trilha, validacao
from reclin.negacao import lexico
from reclin.util import caminhos, identidade_codigo
from reclin.util.io import ler_json, sha256_arquivo
from reclin.util.log import configurar

log = logging.getLogger("reclin.scripts.filtro_pistas")

BASES = ("baseline_biobertpt_seed42", "baseline_bertimbau_seed42",
         "baseline_biobertpt_seed43", "baseline_bertimbau_seed43")


def carregar_lexico(train, calibracao: dict) -> dict[str, int]:
    """O léxico do legado (`CONGELADO_LEGADO`), conferido contra a calibração:
    o modo de reprodução reproduz o legado, nunca o léxico da fase."""
    if calibracao["min_freq"] != lexico.CONGELADO_LEGADO["min_freq"]:
        raise ValueError(f"a calibração escolheu min_freq={calibracao['min_freq']}, e o léxico "
                         f"congelado do legado tem min_freq={lexico.CONGELADO_LEGADO['min_freq']}: uma "
                         "recalibração precisa atualizar o registro CONGELADO_LEGADO")
    lex = lexico.carregar_congelado(train, lexico.CONGELADO_LEGADO)
    if len(lex) != calibracao["lexicon_size"]:
        raise ValueError(f"o léxico tem {len(lex)} formas e a calibração registra "
                         f"{calibracao['lexicon_size']}")
    return lex


def conferir(gravado: Path, referencias: Path, nome: str, particao: str) -> bool | None:
    """True/False se há referência para comparar; None se não há."""
    try:
        ref = diretorio.caminho_predicoes(referencias, nome, particao)
    except FileNotFoundError:
        log.info("%s [%s]: sem referência em %s", nome, particao, referencias)
        return None
    igual = gravado.read_bytes() == ref.read_bytes()
    (log.info if igual else log.error)("%s [%s]: %s byte a byte a %s", nome, particao,
                                       "idêntico" if igual else "DIFERENTE", ref)
    return igual


def _congeladas(pasta: Path) -> bool:
    """MANIFEST de uma versão congelada das partições (`versao_dados`)."""
    return versao_dados.congelada(pasta)


class ErroFiltro(ValueError):
    pass


def _ler_base(pasta: Path) -> dict:
    """A execução de base no modo da fase: baseline do código novo, mesma
    identidade do código atual, fora das referências do legado."""
    motivo = inventario.recusar_caminho(pasta)
    if motivo:
        raise ErroFiltro(f"{pasta}: {motivo}")
    if not (pasta / diretorio.CONFIG).is_file():
        raise ErroFiltro(f"{pasta} não é uma execução do código novo (sem config.json); sidecars soltos e o "
                         "formato do legado não são aceitos no modo da fase")
    registro = diretorio.ler_config(pasta)["config"]
    if registro.get("estrategia") != "baseline":
        raise ErroFiltro(f"{pasta.name} é da estratégia {registro.get('estrategia')!r}; o filtro da fase "
                         "pós-processa o baseline padrão")
    try:
        diretorio.conferir_codigo(pasta, "o uso como base do filtro")
    except identidade_codigo.ErroCodigo as erro:
        raise ErroFiltro(str(erro)) from erro
    return {"pasta": pasta, "nome": pasta.name, "registro": registro}


def registro_da_fase(base: dict, lex: dict) -> dict:
    return {"estrategia": filtro_pistas.NOME, "base": base["nome"],
            "base_pasta": caminhos.relativo_ao_codigo(base["pasta"]),
            "min_freq": lexico.CONGELADO["min_freq"], "max_gap": tarefa.MAX_GAP, "n_formas": len(lex),
            "lexico_sha1": lexico.lexico_sha1(lex),
            "lexico": "congelado (reclin.negacao.lexico.CONGELADO; protocolo da fase, seções 1 e 3)"}


def modo_fase(args) -> int:
    if len(args.particao) != 1:
        raise ErroFiltro("no modo da fase, informe uma partição por vez (--particao dev, depois --particao test)")
    particao = args.particao[0]
    base = _ler_base(args.base_execucao)
    divergencias = particoes.conferir_particoes(caminhos.PARTICOES, ["train"]) + \
        particoes.conferir_particoes(args.particoes, [particao])
    if divergencias:
        raise ErroFiltro("; ".join(divergencias))
    try:      # a base, as partições informadas e o TRAIN do léxico: uma só versão dos dados
        versao_dados.exigir_mesma_versao(diretorio.ler_config(base["pasta"]), args.particoes, "o filtro")
        versao_dados.exigir_mesma_versao(diretorio.ler_config(base["pasta"]), caminhos.PARTICOES,
                                         "o léxico do filtro (TRAIN da fase)")
    except versao_dados.ErroVersaoDados as erro:
        raise ErroFiltro(str(erro)) from erro
    lex = lexico.carregar_congelado(particoes.ler_particao(caminhos.PARTICOES, "train"))
    registro = registro_da_fase(base, lex)
    nome = filtro_pistas.nome_execucao(base["nome"])
    pasta = args.saida / nome
    origem = base["pasta"] / diretorio.arquivo_predicoes(particao)

    if particao == "dev":
        if (pasta / diretorio.CONFIG).exists():
            raise ErroFiltro(f"{pasta} já existe: o filtro do DEV desta base já foi gravado")
        if not origem.is_file():
            raise ErroFiltro(f"a base {base['nome']} não tem {origem.name}")
    else:
        if not (pasta / diretorio.CONFIG).is_file():
            raise ErroFiltro(f"{pasta} não existe: grave antes o filtro do DEV (--particao dev)")
        if diretorio.ler_config(pasta)["config"] != registro:
            raise ErroFiltro(f"{pasta.name} foi criada com outra configuração (léxico ou base): o TEST não é "
                             "aplicado")
        try:
            diretorio.conferir_codigo(pasta, "a avaliação do TEST do filtro")
            validacao.recusar_execucao(base["pasta"], "a avaliação do TEST do filtro")
        except (identidade_codigo.ErroCodigo, validacao.ExecucaoDeValidacao) as erro:
            raise ErroFiltro(str(erro)) from erro
        if not origem.is_file() or not trilha.ler(base["pasta"] / diretorio.TRILHA):
            raise ErroFiltro(f"o TEST da base {base['nome']} ainda não foi avaliado pelo comando próprio "
                             "(avaliar_test.py): o filtro não é aplicado ao TEST antes disso")
        if (pasta / diretorio.arquivo_predicoes("test")).exists() and not args.reavaliar:
            raise ErroFiltro("o TEST deste filtro já foi avaliado; uma nova avaliação exige --reavaliar e fica "
                             "registrada na trilha como reavaliação")

    docs = particoes.ler_particao(args.particoes, particao)
    manifesto = particoes.ler_manifesto(args.particoes)
    conjunto = tarefa.conjunto_referencia(particao, docs, particao_sha256=manifesto["splits"][particao]["sha256"],
                                          conferir_tamanho=_congeladas(args.particoes))
    predicoes_base = predicoes.ler(origem)
    sha_base = sha256_arquivo(origem)
    pred = filtro_pistas.montar_predicoes(predicoes_base, tarefa.candidatos(docs), lex,
                                          base_preds=f"{base['nome']}/{origem.name}",
                                          min_freq=lexico.CONGELADO["min_freq"], base_sha256=sha_base)
    if particao == "dev":
        diretorio.criar(args.saida, nome, estrategia=filtro_pistas.NOME, config=registro)
    diretorio.gravar_predicoes(pasta, particao, pred, conjunto, sobrescrever=args.reavaliar)
    m = metricas.avaliar(pred["y_true"], pred["y_pred"])
    diretorio.gravar_metricas(pasta, particao, m)
    if particao == "test":
        linha = trilha.registrar(pasta / diretorio.TRILHA, execucao=nome, model=pred["model"], seed=pred["seed"],
                                 config_sha1=trilha.sha1_config(registro), n_test=len(pred["y_true"]),
                                 macro_f1=m["macro_f1"], negation_of_f1=m["f1_per_class"]["negation_of"],
                                 extra={"estrategia": filtro_pistas.NOME, "base": base["nome"],
                                        "base_sha256": sha_base})
        log.info("Avaliação do TEST registrada na trilha (eval_index_for_config=%d)",
                 linha["eval_index_for_config"])
    antes = sum(1 for y in predicoes_base["y_pred"] if y == tarefa.NEG)
    depois = sum(1 for y in pred["y_pred"] if y == tarefa.NEG)
    log.info("%s [%s]: léxico congelado (%d formas, %s); %d de %d negation_of rebaixados; F1 negation_of=%.4f",
             nome, particao, len(lex), registro["lexico_sha1"], antes - depois, antes,
             m["f1_per_class"]["negation_of"])
    return 0


def modo_reproducao(args) -> int:
    lidas = ["train", *args.particao]
    divergencias = particoes.conferir_particoes(args.particoes, lidas)
    if divergencias:
        for d in divergencias:
            log.error("%s", d)
        return 1
    calibracao = ler_json(args.calibracao)
    try:
        lex = carregar_lexico(particoes.ler_particao(args.particoes, "train"), calibracao)
    except ValueError as erro:
        log.error("%s", erro)
        return 1
    log.info("Léxico congelado: %d formas, min_freq=%d (calibração %s)",
             len(lex), calibracao["min_freq"], args.calibracao)

    manifesto = particoes.ler_manifesto(args.particoes)
    conjuntos, candidatos = {}, {}
    for p in args.particao:
        docs = particoes.ler_particao(args.particoes, p)
        candidatos[p] = tarefa.candidatos(docs)
        conjuntos[p] = tarefa.conjunto_referencia(
            p, docs, particao_sha256=manifesto["splits"][p]["sha256"])

    resultados_conferencia = []
    for base in args.bases:
        nome = filtro_pistas.nome_execucao(base)
        pasta = diretorio.criar(args.saida, nome, estrategia=filtro_pistas.NOME, config={
            "base": base, "min_freq": calibracao["min_freq"], "max_gap": tarefa.MAX_GAP,
            "n_formas": len(lex), "lexico_sha1": lexico.lexico_sha1(lex),
            "calibracao": calibracao})
        for p in args.particao:
            origem = diretorio.caminho_predicoes(args.resultados, base, p)
            predicoes_base = predicoes.ler(origem)
            pred = filtro_pistas.montar_predicoes(
                predicoes_base, candidatos[p], lex,
                base_preds=origem.relative_to(args.resultados).as_posix(),
                min_freq=calibracao["min_freq"])
            gravado = diretorio.gravar_predicoes(pasta, p, pred, conjuntos[p])
            antes = sum(1 for y in predicoes_base["y_pred"] if y == tarefa.NEG)
            depois = sum(1 for y in pred["y_pred"] if y == tarefa.NEG)
            log.info("%s [%s]: %d de %d negation_of rebaixados; gravado em %s",
                     nome, p, antes - depois, antes, gravado)
            if args.conferir:
                resultados_conferencia.append(conferir(gravado, args.conferir, nome, p))

    if args.conferir:
        comparados = [r for r in resultados_conferencia if r is not None]
        log.info("Conferência: %d sidecars comparados, %d idênticos", len(comparados), sum(comparados))
        if not all(comparados):
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--base-execucao", type=Path, default=None,
                    help="modo da fase: a execução do baseline do código novo (léxico congelado)")
    ap.add_argument("--reavaliar", action="store_true", help="modo da fase: nova avaliação do TEST (na trilha)")
    ap.add_argument("--resultados", type=Path, default=None,
                    help="modo de reprodução: pasta com as predições das execuções de base")
    ap.add_argument("--calibracao", type=Path, default=None, help="modo de reprodução: CALIBRACAO_filtro.json")
    ap.add_argument("--bases", nargs="+", default=None, help="modo de reprodução: nomes das bases")
    ap.add_argument("--particao", nargs="+", choices=diretorio.PARTICOES, default=None)
    ap.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    ap.add_argument("--saida", type=Path, default=caminhos.RESULTADOS / "execucoes")
    ap.add_argument("--conferir", type=Path, default=None,
                    help="modo de reprodução: pasta de referência para comparar os sidecars byte a byte")
    args = ap.parse_args(argv)
    if args.base_execucao is not None:
        if args.resultados or args.calibracao or args.bases or args.conferir:
            ap.error("--base-execucao (modo da fase) não aceita --resultados, --calibracao, --bases nem --conferir")
        if args.particao is None:
            ap.error("no modo da fase, informe --particao dev ou --particao test")
        configurar()
        try:
            return modo_fase(args)
        except (ErroFiltro, ValueError) as erro:
            log.error("%s", erro)
            return 1
    if args.resultados is None or args.calibracao is None:
        ap.error("informe --base-execucao (modo da fase) ou --resultados e --calibracao (modo de reprodução)")
    if args.reavaliar:
        ap.error("--reavaliar só vale no modo da fase")
    args.bases = args.bases or list(BASES)
    args.particao = args.particao or list(diretorio.PARTICOES)
    configurar()
    return modo_reproducao(args)


if __name__ == "__main__":
    sys.exit(main())
