"""Persistência no Google Drive: sincronizar, restaurar, listar e diagnosticar a retomada.

Modelo (decisão 4 da etapa 8): treina-se no disco local do Colab e o Drive guarda
uma cópia **verificada** de cada execução, para sobreviver ao fim da sessão.

    <raiz_drive>/execucoes/<nome>/       espelho da pasta da execução local
        ... os mesmos arquivos (config.json, treino.json, experimento.json,
            predicoes_*.json, metricas.json, checkpoints/ultimo.pt, ultimo.json,
            melhor_modelo/)
        SYNC.json                        manifesto da última sincronização completa

Regras:

* cada arquivo é copiado para `<nome>.sync-tmp`, verificado por SHA-256 contra
  a origem e só então renomeado (`os.replace`): uma queda no meio deixa o
  arquivo anterior intacto, nunca um arquivo pela metade com o nome final;
* os marcadores de estado vão por último (`ultimo.pt` → `ultimo.json` →
  `treino.json` → `experimento.json`) e o `SYNC.json` por último de todos: **sem
  `SYNC.json` o espelho não é um instantâneo completo** e a restauração o recusa;
* arquivos iguais (mesmo SHA-256 do manifesto anterior e mesmo tamanho) não são
  copiados de novo;
* nunca apaga nada no destino, nunca regrava um instantâneo mais avançado com um
  mais antigo (`ErroSincronizacao` se o Drive estiver à frente: restaure antes);
* a restauração não sobrescreve arquivo local diferente sem `substituir=True` e
  confere todos os hashes depois de copiar.

Nenhuma função fala com a rede: o Drive é uma pasta montada. Montar o Drive é
`colab.montar_drive`, chamado pelo notebook.
"""
from __future__ import annotations

import hashlib
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from reclin.experimento import ambiente as ambiente_mod
from reclin.experimento import ids, proveniencia
from reclin.util.io import gravar_json, ler_json

MANIFESTO = "SYNC.json"
SUFIXO_TEMPORARIO = ".sync-tmp"
FORMATO = 1
PASTA_EXECUCOES = "execucoes"
ULTIMO = "checkpoints/ultimo.pt"
ULTIMO_RESUMO = "checkpoints/ultimo.json"
IGNORADOS = ("__pycache__", ".ipynb_checkpoints")
ULTIMOS_ARQUIVOS = (ULTIMO, ULTIMO_RESUMO, "treino.json", proveniencia.ARQUIVO)


class ErroSincronizacao(RuntimeError):
    """A sincronização ou a restauração foi recusada para não perder nem misturar estados."""


def raiz_das_execucoes(raiz_drive: str | Path) -> Path:
    return Path(raiz_drive) / PASTA_EXECUCOES


def _sha(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(4 * 1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def arquivos_da_execucao(pasta: str | Path) -> list[str]:
    """Arquivos da execução (POSIX, relativos), na ordem de cópia: os de dados
    primeiro, os marcadores de estado por último. Sem temporários nem o manifesto."""
    pasta = Path(pasta)
    todos = []
    for caminho in sorted(pasta.rglob("*")):
        relativo = caminho.relative_to(pasta).as_posix()
        partes = relativo.split("/")
        if not caminho.is_file() or relativo == MANIFESTO or relativo.endswith((SUFIXO_TEMPORARIO, ".tmp")) \
                or any(p in IGNORADOS for p in partes):
            continue
        todos.append(relativo)
    comuns = [r for r in todos if r not in ULTIMOS_ARQUIVOS]
    return comuns + [r for r in ULTIMOS_ARQUIVOS if r in todos]


def ler_manifesto(pasta: str | Path) -> dict[str, Any] | None:
    caminho = Path(pasta) / MANIFESTO
    return ler_json(caminho) if caminho.is_file() else None


def _copiar_verificado(origem: Path, destino: Path, sha_origem: str) -> None:
    """Copia para `<destino>.sync-tmp`, confere o SHA-256 lendo o que foi
    gravado e renomeia. O temporário é removido em qualquer falha."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + SUFIXO_TEMPORARIO)
    try:
        with open(origem, "rb") as entrada, open(temporario, "wb") as saida:
            for bloco in iter(lambda: entrada.read(4 * 1024 * 1024), b""):
                saida.write(bloco)
            saida.flush()
            os.fsync(saida.fileno())
        if _sha(temporario) != sha_origem:
            raise ErroSincronizacao(f"verificação falhou ao copiar {origem.name}: o SHA-256 do destino difere da origem")
        os.replace(temporario, destino)
    finally:
        if temporario.exists():
            temporario.unlink()


def _progresso(pasta: Path) -> tuple[int, bool]:
    """(passo global do último checkpoint, treino concluído) de uma pasta de execução."""
    passo, concluido = -1, False
    resumo = pasta / ULTIMO_RESUMO
    if resumo.is_file():
        try:
            passo = int((ler_json(resumo).get("posicao") or {}).get("passo_global", -1))
        except (ValueError, OSError):
            passo = -1
    treino = pasta / "treino.json"
    if treino.is_file():
        try:
            concluido = bool(ler_json(treino).get("concluido"))
        except (ValueError, OSError):
            concluido = False
    return passo, concluido


def conferir_sentido(origem: Path, destino: Path) -> None:
    """Recusa regravar um destino mais avançado com uma origem mais antiga."""
    if not destino.is_dir():
        return
    passo_o, concluido_o = _progresso(origem)
    passo_d, concluido_d = _progresso(destino)
    if concluido_d and not concluido_o:
        raise ErroSincronizacao(f"{destino} já tem o treino concluído e a origem não: restaure do Drive antes de sincronizar")
    if passo_d > passo_o and not concluido_o:
        raise ErroSincronizacao(f"{destino} está mais avançado (passo {passo_d}) que a origem (passo {passo_o}): "
                                "restaure do Drive antes de sincronizar")


def sincronizar(origem: str | Path, raiz_drive: str | Path, *, incluir_ultimo: bool = True,
                progresso: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Espelha a execução `origem` em `<raiz_drive>/execucoes/<nome>/`.

    `incluir_ultimo=False` deixa `checkpoints/ultimo.pt` de fora (útil para
    um espelho só de resultados). Devolve o resumo gravado em
    `experimento.json["checkpoints"]["drive"]` por quem chamar `registrar`."""
    origem = Path(origem)
    if not (origem / "config.json").is_file():
        raise ErroSincronizacao(f"{origem} não é uma execução (sem config.json)")
    nome = ids.validar_nome(origem.name)
    destino = raiz_das_execucoes(raiz_drive) / nome
    conferir_sentido(origem, destino)
    anterior = (ler_manifesto(destino) or {}).get("arquivos", {})
    arquivos, copiados, iguais = {}, 0, 0
    for relativo in arquivos_da_execucao(origem):
        if not incluir_ultimo and relativo == ULTIMO:
            continue
        fonte, alvo = origem / relativo, destino / relativo
        sha, tamanho = _sha(fonte), fonte.stat().st_size
        previo = anterior.get(relativo)
        if previo and previo["sha256"] == sha and previo["tamanho"] == tamanho and alvo.is_file() \
                and alvo.stat().st_size == tamanho:
            iguais += 1
        else:
            if progresso:
                progresso(f"copiando {relativo} ({tamanho / 1e6:.1f} MB)")
            _copiar_verificado(fonte, alvo, sha)
            copiados += 1
        arquivos[relativo] = {"sha256": sha, "tamanho": tamanho}
    manifesto = {"formato": FORMATO, "nome": nome, "arquivos": arquivos, "incluiu_ultimo": incluir_ultimo,
                 "sincronizado_em": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "maquina": platform.node() or None}
    gravar_json(destino / MANIFESTO, manifesto)
    return {"destino": str(destino), "n_arquivos": len(arquivos), "copiados": copiados, "inalterados": iguais,
            "sincronizado_em": manifesto["sincronizado_em"], "incluiu_ultimo": incluir_ultimo,
            "manifesto_sha256": _sha(destino / MANIFESTO)}


def verificar_espelho(pasta: str | Path) -> list[str]:
    """Problemas do espelho (vazio se todos os arquivos do manifesto existem e conferem)."""
    pasta = Path(pasta)
    manifesto = ler_manifesto(pasta)
    if manifesto is None:
        return [f"{pasta} não tem {MANIFESTO}: a sincronização não terminou ou nunca ocorreu"]
    problemas = []
    for relativo, info in sorted(manifesto["arquivos"].items()):
        caminho = pasta / relativo
        if not caminho.is_file():
            problemas.append(f"ausente: {relativo}")
        elif caminho.stat().st_size != info["tamanho"] or _sha(caminho) != info["sha256"]:
            problemas.append(f"alterado ou corrompido: {relativo}")
    return problemas


def restaurar(raiz_drive: str | Path, nome: str, raiz_local: str | Path, *, substituir: bool = False,
              progresso: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Copia `<raiz_drive>/execucoes/<nome>/` para `<raiz_local>/<nome>/`, com
    verificação. Recusa um espelho sem `SYNC.json` ou com hash divergente, e
    arquivo local diferente (a menos que `substituir=True`)."""
    nome = ids.validar_nome(nome)
    origem = raiz_das_execucoes(raiz_drive) / nome
    manifesto = ler_manifesto(origem)
    if manifesto is None:
        raise ErroSincronizacao(f"{origem} não tem {MANIFESTO}: o espelho pode estar incompleto e não será restaurado")
    destino = Path(raiz_local) / nome
    conflitos, a_copiar = [], []
    for relativo, info in manifesto["arquivos"].items():
        fonte, alvo = origem / relativo, destino / relativo
        if not fonte.is_file() or fonte.stat().st_size != info["tamanho"]:
            raise ErroSincronizacao(f"{fonte} ausente ou com tamanho diferente do manifesto")
        if alvo.is_file():
            if alvo.stat().st_size == info["tamanho"] and _sha(alvo) == info["sha256"]:
                continue
            conflitos.append(relativo)
        a_copiar.append(relativo)
    if conflitos and not substituir:
        raise ErroSincronizacao(f"arquivos locais diferentes do Drive em {destino}: {conflitos[:5]}"
                                f"{'…' if len(conflitos) > 5 else ''}; use substituir=True se o Drive deve prevalecer")
    ordem = [r for r in a_copiar if r not in ULTIMOS_ARQUIVOS] + [r for r in ULTIMOS_ARQUIVOS if r in a_copiar]
    for relativo in ordem:
        if progresso:
            progresso(f"restaurando {relativo}")
        _copiar_verificado(origem / relativo, destino / relativo, manifesto["arquivos"][relativo]["sha256"])
    return {"destino": str(destino), "n_arquivos": len(manifesto["arquivos"]), "copiados": len(ordem),
            "substituidos": sorted(conflitos), "sincronizado_em": manifesto.get("sincronizado_em")}


def _resumo_da_execucao(pasta: Path) -> dict[str, Any] | None:
    if not pasta.is_dir() or not any(pasta.iterdir()):
        return None
    derivado = proveniencia.derivar_estado(pasta)
    passo, concluido = _progresso(pasta)
    registro = None
    if (pasta / proveniencia.ARQUIVO).is_file():
        try:
            registro = proveniencia.ler(pasta)
        except (proveniencia.ErroProveniencia, ValueError):
            registro = None
    return {"estado": derivado["estado"], "ambiguo": derivado["ambiguo"], "passo_global": passo,
            "concluido": concluido, "tem_ultimo": (pasta / ULTIMO).is_file(),
            "run_id": (registro or {}).get("run_id"), "alias_oficial": (registro or {}).get("alias_oficial"),
            "n_publicacoes": len((registro or {}).get("publicacoes") or [])}


def listar(raiz_local: str | Path, raiz_drive: str | Path | None) -> list[dict[str, Any]]:
    """Inventário leve das execuções locais e do espelho do Drive (sem as
    verificações do `execucao.inventario`, que continuam sendo a fonte para
    números): uma linha por nome, com o estado de cada lado e se estão em dia."""
    local, drive = Path(raiz_local), (raiz_das_execucoes(raiz_drive) if raiz_drive else None)
    nomes = set()
    for raiz in (local, drive):
        if raiz is not None and raiz.is_dir():
            nomes |= {p.name for p in raiz.iterdir() if p.is_dir() and not p.name.startswith(".")}
    linhas = []
    for nome in sorted(nomes):
        l = _resumo_da_execucao(local / nome)
        d = _resumo_da_execucao(drive / nome) if drive else None
        manifesto = ler_manifesto(drive / nome) if drive and (drive / nome).is_dir() else None
        if l and d:
            situacao = "em_dia" if manifesto and l["passo_global"] == d["passo_global"] \
                and l["concluido"] == d["concluido"] else "divergente"
            if l["passo_global"] < d["passo_global"] or (d["concluido"] and not l["concluido"]):
                situacao = "drive_adiantado"
        else:
            situacao = "so_local" if l else "so_drive"
        origem = l or d
        linhas.append({"nome": nome, "situacao": situacao, "local": l, "drive": d,
                       "espelho_completo": manifesto is not None,
                       "run_id": origem["run_id"], "alias_oficial": origem["alias_oficial"],
                       "retomavel": bool(origem["tem_ultimo"] and not origem["concluido"])})
    return linhas


def diagnosticar_retomada(nome: str, raiz_local: str | Path, raiz_drive: str | Path | None, *,
                          ambiente_atual: Mapping[str, Any] | None = None,
                          identidade_atual: str | None = None) -> dict[str, Any]:
    """Dry-run da retomada: onde está o estado mais avançado, se a execução
    estava interrompida e se o ambiente e o código atuais são compatíveis. Não
    copia nem altera nada.

    Bloqueios: execução concluída (nada a retomar); sem checkpoint; espelho do
    Drive incompleto ou corrompido quando é dele que se retomaria; ambiente
    incompatível (`ambiente.comparar_para_retomada`); identidade do código
    diferente da gravada no `config.json` (`diretorio.conferir_codigo` repete
    esta checagem no momento de retomar)."""
    nome = ids.validar_nome(nome)
    pasta_local = Path(raiz_local) / nome
    pasta_drive = (raiz_das_execucoes(raiz_drive) / nome) if raiz_drive else None
    l, d = _resumo_da_execucao(pasta_local), (_resumo_da_execucao(pasta_drive) if pasta_drive else None)
    bloqueios: list[str] = []
    fonte, pasta = "nenhuma", None
    if l and l["tem_ultimo"] and (not d or l["passo_global"] >= d["passo_global"]):
        fonte, pasta = "local", pasta_local
    elif d and d["tem_ultimo"]:
        fonte, pasta = "drive", pasta_drive
        problemas = verificar_espelho(pasta_drive)
        if problemas:
            bloqueios.append("espelho do Drive não confere: " + "; ".join(problemas[:3]))
    else:
        bloqueios.append("não há checkpoint (ultimo.pt) local nem no Drive")
    alvo = l if fonte == "local" else d
    if alvo and alvo["concluido"]:
        bloqueios.append("o treino já está concluído: não há o que retomar")
    avisos: list[str] = []
    if pasta is not None and not bloqueios:
        treino = ler_json(pasta / "treino.json") if (pasta / "treino.json").is_file() else None
        registro = ler_json(pasta / proveniencia.ARQUIVO) if (pasta / proveniencia.ARQUIVO).is_file() else None
        gravado = ambiente_mod.ambiente_da_ultima_sessao(treino, registro)
        atual = dict(ambiente_atual) if ambiente_atual is not None else ambiente_mod.coletar()
        comparacao = ambiente_mod.comparar_para_retomada(gravado, atual)
        bloqueios += [f"ambiente: {b}" for b in comparacao["bloqueios"]]
        avisos += comparacao["avisos"]
        if identidade_atual is not None and (pasta / "config.json").is_file():
            gravada = (ler_json(pasta / "config.json").get("codigo") or {}).get("identidade")
            if not gravada:
                bloqueios.append("a execução não tem a identidade do código gravada")
            elif gravada != identidade_atual:
                bloqueios.append(f"código diferente do que criou a execução ({gravada[:12]}… ≠ {identidade_atual[:12]}…)")
    return {"nome": nome, "fonte": fonte, "pode_retomar": not bloqueios, "bloqueios": bloqueios, "avisos": avisos,
            "local": l, "drive": d,
            "acao": {"restaurar_do_drive": fonte == "drive", "retomar_com": "scripts/treinar.py --retomar (ou two_stage.py)"}}
