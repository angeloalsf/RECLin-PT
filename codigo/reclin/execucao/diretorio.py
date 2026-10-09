"""O diretório de uma execução.

    <raiz>/<nome>/
        config.json            nome, estratégia, configuração e a identidade dos
                               conjuntos de referência de cada partição gravada
        predicoes_dev.json     sidecar do DEV (`predicoes`)
        predicoes_test.json    sidecar do TEST
        metricas.json          métricas por partição, gravadas por quem avalia
        avaliacoes_test.jsonl  trilha das avaliações do TEST (`trilha`)

O nome segue o identificador do legado: `<estratégia>_<encoder>_seed<N>`
(`baseline_biobertpt_seed42`, `restrito_biobertpt_seed43`) ou só o nome da
estratégia quando ela não tem encoder nem semente (`regra_pura`).

As predições de uma partição são gravadas uma vez. Gravar de novo exige
`sobrescrever=True`: o TEST é avaliado uma única vez por configuração, e uma
regravação silenciosa esconderia uma reavaliação.

O legado gravava tudo lado a lado em `results/`: `<nome>.json`,
`<nome>.preds.json` (TEST), `<nome>.dev_preds.json` (DEV) e
`<nome>.test_evals.jsonl`. `caminho_predicoes` encontra as predições nos dois
formatos, para que comparações e relatórios leiam igual as execuções novas e
as do legado.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from reclin.execucao import predicoes as sidecar
from reclin.tarefa import ConjuntoReferencia
from reclin.util.io import gravar_json, ler_json

PARTICOES = ("dev", "test")
CONFIG = "config.json"
METRICAS = "metricas.json"
TRILHA = "avaliacoes_test.jsonl"
SUFIXOS_LEGADO = {"dev": ".dev_preds.json", "test": ".preds.json"}


def arquivo_predicoes(particao: str) -> str:
    if particao not in PARTICOES:
        raise ValueError(f"partição {particao!r} não é avaliada; opções: {PARTICOES}")
    return f"predicoes_{particao}.json"


def criar(raiz: str | Path, nome: str, *, estrategia: str, config: dict[str, Any]) -> Path:
    """Cria o diretório da execução com o `config.json`. Falha se já existir."""
    pasta = Path(raiz) / nome
    if (pasta / CONFIG).exists():
        raise FileExistsError(f"a execução {nome} já existe em {pasta}")
    gravar_json(pasta / CONFIG, {"nome": nome, "estrategia": estrategia, "config": config,
                                 "conjuntos": {}})
    return pasta


def ler_config(pasta: str | Path) -> dict[str, Any]:
    return ler_json(Path(pasta) / CONFIG)


def gravar_predicoes(pasta: str | Path, particao: str, predicoes: dict[str, Any],
                     conjunto: ConjuntoReferencia, *, sobrescrever: bool = False) -> Path:
    """Grava as predições de uma partição depois de conferi-las contra o
    conjunto de referência, e registra a identidade do conjunto no config."""
    pasta = Path(pasta)
    destino = pasta / arquivo_predicoes(particao)
    if conjunto["particao"] != particao:
        raise ValueError(f"conjunto de {conjunto['particao']} usado para predições de {particao}")
    if destino.exists() and not sobrescrever:
        raise FileExistsError(f"{destino} já existe; regravar exige sobrescrever=True")
    divergencias = sidecar.conferir_conjunto(predicoes, conjunto)
    if divergencias:
        raise ValueError("; ".join(divergencias))
    config = ler_config(pasta)
    sidecar.gravar(destino, predicoes)
    config["conjuntos"][particao] = dict(conjunto)
    gravar_json(pasta / CONFIG, config)
    return destino


def ler_predicoes(pasta: str | Path, particao: str) -> dict[str, Any]:
    return sidecar.ler(Path(pasta) / arquivo_predicoes(particao))


def gravar_metricas(pasta: str | Path, particao: str, metricas: dict[str, Any]) -> None:
    """Acrescenta (ou substitui) as métricas de uma partição em `metricas.json`."""
    caminho = Path(pasta) / METRICAS
    todas = ler_json(caminho) if caminho.exists() else {}
    todas[particao] = metricas
    gravar_json(caminho, todas)


def ler_metricas(pasta: str | Path) -> dict[str, Any]:
    return ler_json(Path(pasta) / METRICAS)


def caminho_predicoes(raiz: str | Path, nome: str, particao: str) -> Path:
    """Predições de `nome` em `raiz`: no formato novo (`<nome>/predicoes_<p>.json`)
    ou no do legado (`<nome>.preds.json`, `<nome>.dev_preds.json`)."""
    raiz = Path(raiz)
    candidatos = [raiz / nome / arquivo_predicoes(particao),
                  raiz / f"{nome}{SUFIXOS_LEGADO[particao]}"]
    for caminho in candidatos:
        if caminho.is_file():
            return caminho
    raise FileNotFoundError(f"predições de {particao} de {nome} não encontradas: "
                            + ", ".join(str(c) for c in candidatos))
