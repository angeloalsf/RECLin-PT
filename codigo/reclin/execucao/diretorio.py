"""O diretório de uma execução.

    <raiz>/<nome>/
        config.json            nome, estratégia, configuração, a identidade dos
                               conjuntos de referência de cada partição gravada,
                               a versão dos dados de cada um (`versoes_dados`,
                               `reclin.versao_dados`) e a identidade do código
                               que criou a execução (`codigo`, `util.identidade_codigo`)
        predicoes_dev.json     sidecar do DEV (`predicoes`)
        predicoes_test.json    sidecar do TEST
        metricas.json          métricas por partição, gravadas por quem avalia
        avaliacoes_test.jsonl  trilha das avaliações do TEST (`trilha`)

O nome segue o identificador do legado: `<estratégia>_<encoder>_seed<N>`
(`baseline_biobertpt_seed42`, `restrito_biobertpt_seed43`) ou só o nome da
estratégia quando ela não tem encoder nem semente (`regra_pura`).

Identidade do código (D6): `criar` grava em `config.json` o registro de
`util.identidade_codigo.atual()`. `conferir_codigo` é o que as operações que
acrescentam artefatos a uma execução existente (retomada do treino, avaliação
do TEST, execuções derivadas) chamam antes de agir: o código atual precisa ser
o mesmo que criou a execução. Uma execução sem a identidade (criada antes da
D6) é recusada nessas operações; seus metadados nunca são completados depois.

Versão dos dados: `gravar_predicoes` recusa (`versao_dados.ErroVersaoDados`)
predições cujo conjunto de referência seja de outra versão registrada dos
dados que a das partições do treino ou dos conjuntos já gravados na execução
(um modelo da v1 avaliado no TEST da v2, por exemplo).

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

from reclin import versao_dados
from reclin.execucao import predicoes as sidecar
from reclin.tarefa import ConjuntoReferencia
from reclin.util import identidade_codigo
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
                                 "conjuntos": {}, "codigo": identidade_codigo.atual()})
    return pasta


def identidade_da_execucao(pasta: str | Path) -> str | None:
    """A identidade do código gravada no `config.json` (None se ausente)."""
    return (ler_config(pasta).get("codigo") or {}).get("identidade")


def conferir_codigo(pasta: str | Path, operacao: str) -> str:
    """Falha (`identidade_codigo.ErroCodigo`) se a execução não tiver a
    identidade do código, se o registro gravado for inconsistente ou se o
    código atual for outro; devolve a identidade."""
    pasta = Path(pasta)
    registro = ler_config(pasta).get("codigo")
    if not registro:
        raise identidade_codigo.ErroCodigo(
            f"{pasta.name} não tem a identidade do código que a criou (execução anterior à D6): "
            f"não pode receber {operacao} do código atual")
    problemas = identidade_codigo.conferir_registro(registro)
    if problemas:
        raise identidade_codigo.ErroCodigo(f"{pasta.name}: " + "; ".join(problemas))
    atual = identidade_codigo.atual()
    if registro["identidade"] != atual["identidade"]:
        mudaram = identidade_codigo.diferencas(registro, atual)
        raise identidade_codigo.ErroCodigo(
            f"{pasta.name} foi criada por outra versão do código ({registro['identidade'][:12]}…, atual "
            f"{atual['identidade'][:12]}…; arquivos diferentes: {', '.join(mudaram[:10]) or '—'}): "
            f"{operacao} recusada para não misturar versões numa execução")
    return atual["identidade"]


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
    versoes = versao_dados.versoes_do_registro(config)
    versoes[f"conjunto.{particao}"] = versao_dados.versao_da_particao(conjunto.get("particao_sha256"))
    misturadas = versao_dados.conferir_uma_versao(versoes)
    if misturadas:
        raise versao_dados.ErroVersaoDados(f"{pasta.name}: " + "; ".join(misturadas)
                                           + ": as predições não são gravadas")
    sidecar.gravar(destino, predicoes)
    config["conjuntos"][particao] = dict(conjunto)
    # versão dos dados do conjunto (`versao_dados`), legível; None = partição fora do registro
    config.setdefault("versoes_dados", {})[particao] = versao_dados.versao_da_particao(conjunto.get("particao_sha256"))
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
