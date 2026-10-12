"""Publicação e recuperação de artefatos num repositório PRIVADO do Hugging Face.

Nada é publicado, baixado ou verificado na rede sem ação explícita: `planejar`
e `plano_recuperacao` só leem arquivos locais; `publicar` e `recuperar` exigem
`confirmar=True`.

Estrutura no repositório (um por fase, privado), um prefixo por execução:

    runs/<nome>/
        MANIFESTO_PUBLICACAO.json     arquivos, tamanhos, SHA-256, run_id,
                                      config_id, revisão do código, pesos, ambiente
        config.json  treino.json  metricas.json  experimento.json
        predicoes_dev.json
        checkpoints/ultimo.json
        checkpoints/melhor_modelo/…   pesos e tokenizer da melhor época

Padrões (todos alteráveis por argumento explícito):

* `checkpoints/ultimo.pt` **não** é publicado (estado de retomada, grande e sem
  valor fora da sessão); `incluir_ultimo=True` o inclui;
* os artefatos do TEST (`predicoes_test.json`, `avaliacoes_test.jsonl`) **não**
  são publicados; `incluir_test=True` os inclui;
* só se publica execução `concluida`, com `experimento.json` consistente;
* um prefixo que já existe no repositório **nunca** é sobrescrito: uma nova
  tentativa tem outro `run_id` (e portanto outro prefixo);
* um repositório que não seja privado é recusado; se ele não existir, é criado
  privado (só então), de modo que um token sem permissão de criação funciona
  num repositório já criado;
* o envio é **um único commit** (as operações de adição e o manifesto), de modo
  que a execução aparece inteira ou não aparece;
* depois do commit, o remoto é verificado: tamanho de todos os arquivos,
  SHA-256 dos arquivos LFS pelos metadados do Hub e SHA-256 dos arquivos
  pequenos baixados de volta. A publicação é registrada em `experimento.json`
  mesmo quando a verificação falha (com `verificada: false`), para a referência
  não se perder.

O token (`segredos.HUGGING_FACE`) é passado às chamadas da biblioteca e nunca
escrito em registro, manifesto, log ou mensagem de erro.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Mapping

from reclin.experimento import ids, persistencia, proveniencia, segredos
from reclin.util.io import ler_json, sha256_arquivo

PASTA_RUNS = "runs"
MANIFESTO_PUBLICACAO = "MANIFESTO_PUBLICACAO.json"
FORMATO = 1
ULTIMO_PT = "checkpoints/ultimo.pt"
ARTEFATOS_DO_TEST = ("predicoes_test.json", "avaliacoes_test.jsonl")
LIMITE_PEQUENO = 8 * 1024 * 1024          # arquivos até aqui são baixados de volta para conferir o SHA-256


class ErroPublicacao(RuntimeError):
    """A publicação ou a recuperação foi recusada ou falhou."""


def _importar_hf():
    """Isolado para que os testes possam trocar o módulo (e para o import ser preguiçoso)."""
    import huggingface_hub  # noqa: PLC0415

    return huggingface_hub


def prefixo_da_execucao(nome: str) -> str:
    return f"{PASTA_RUNS}/{ids.validar_nome(nome)}"


def selecionar_arquivos(pasta: str | Path, *, incluir_ultimo: bool = False, incluir_test: bool = False
                        ) -> tuple[list[str], list[dict[str, str]]]:
    """(arquivos a publicar, excluídos com o motivo). Ordem: a de cópia da persistência."""
    incluidos, excluidos = [], []
    for relativo in persistencia.arquivos_da_execucao(pasta):
        if relativo == ULTIMO_PT and not incluir_ultimo:
            excluidos.append({"arquivo": relativo, "motivo": "estado de retomada: não publicado por padrão"})
        elif relativo in ARTEFATOS_DO_TEST and not incluir_test:
            excluidos.append({"arquivo": relativo, "motivo": "artefato do TEST: não publicado por padrão"})
        else:
            incluidos.append(relativo)
    return incluidos, excluidos


def montar_manifesto(pasta: str | Path, arquivos: list[str], *, repositorio: str,
                     excluidos: list[dict[str, str]] | None = None) -> dict[str, Any]:
    pasta = Path(pasta)
    registro = proveniencia.ler(pasta)
    detalhes = {r: {"sha256": sha256_arquivo(pasta / r), "tamanho": (pasta / r).stat().st_size} for r in arquivos}
    return {"formato": FORMATO, "repositorio": repositorio, "prefixo": prefixo_da_execucao(pasta.name),
            "nome": pasta.name, "run_id": registro["run_id"], "config_id": registro["config_id"],
            "alias_oficial": registro.get("alias_oficial"), "estrategia": registro["estrategia"],
            "seed": registro["seed"], "codigo": registro["codigo"], "pesos_pretreinados": registro["pesos_pretreinados"],
            "ambiente": registro.get("ambiente"), "estado": registro["estado"],
            "arquivos": detalhes, "excluidos": excluidos or [],
            "criado_em": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def planejar(pasta: str | Path, repositorio: str, *, incluir_ultimo: bool = False, incluir_test: bool = False,
             permitir_nao_concluida: bool = False) -> dict[str, Any]:
    """Dry-run: o que seria publicado, onde, e o que impede. Só lê arquivos locais."""
    pasta = Path(pasta)
    problemas: list[str] = []
    if "/" not in repositorio or repositorio.count("/") != 1:
        problemas.append(f"repositório {repositorio!r}: use o formato usuario/nome")
    problemas += proveniencia.conferir(pasta)
    registro = None
    try:
        registro = proveniencia.ler(pasta)
    except proveniencia.ErroProveniencia:
        pass
    derivado = proveniencia.derivar_estado(pasta)
    if registro and registro["estado"] != "concluida" and not permitir_nao_concluida:
        problemas.append(f"estado {registro['estado']!r}: só execuções concluídas são publicadas")
    if derivado["estado"] != "concluida" and not permitir_nao_concluida:
        problemas.append(f"os arquivos evidenciam {derivado['estado']!r}: {derivado['evidencia']}")
    arquivos, excluidos = selecionar_arquivos(pasta, incluir_ultimo=incluir_ultimo, incluir_test=incluir_test)
    if not any(r.startswith("checkpoints/melhor_modelo/") for r in arquivos):
        problemas.append("sem checkpoints/melhor_modelo/: nada de pesos para publicar")
    manifesto = None
    if registro and not problemas:
        manifesto = montar_manifesto(pasta, arquivos, repositorio=repositorio, excluidos=excluidos)
        problemas += [f"manifesto: {m}" for m in segredos.contem_segredo(manifesto)]
    tamanhos = {r: (pasta / r).stat().st_size for r in arquivos}
    return {"executado": False, "repositorio": repositorio, "prefixo": prefixo_da_execucao(pasta.name),
            "n_arquivos": len(arquivos), "bytes": sum(tamanhos.values()), "arquivos": tamanhos,
            "excluidos": excluidos, "problemas": problemas, "pode_publicar": not problemas, "manifesto": manifesto,
            "requer": "HF_TOKEN com escrita (segredos.OPERACOES['publicar_hf'])"}


def _sha_bytes(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def _garantir_repositorio_privado(api, repositorio: str, token: str) -> None:
    """O repositório existe e é privado. Só tenta criá-lo (privado) se não existir,
    para que um token sem permissão de criação funcione num repositório já criado."""
    try:
        info = api.model_info(repositorio, token=token)
    except Exception as erro:  # noqa: BLE001 — a classe do "não encontrado" varia entre versões da biblioteca
        if type(erro).__name__ != "RepositoryNotFoundError":
            raise
        api.create_repo(repositorio, token=token, private=True, repo_type="model", exist_ok=True)
        info = api.model_info(repositorio, token=token)
    if not info.private:
        raise ErroPublicacao(f"o repositório {repositorio} não é privado: publicação recusada")


def _verificar_remoto(hf, api, repositorio: str, commit: str, prefixo: str, manifesto: Mapping[str, Any],
                      credencial: segredos.Credencial) -> dict[str, Any]:
    caminhos = [f"{prefixo}/{r}" for r in manifesto["arquivos"]] + [f"{prefixo}/{MANIFESTO_PUBLICACAO}"]
    informacoes = {i.path: i for i in api.get_paths_info(repositorio, caminhos, expand=True, revision=commit,
                                                         repo_type="model", token=credencial.valor())}
    falhas, sha_lfs, sha_baixado = [], 0, 0
    for relativo, esperado in manifesto["arquivos"].items():
        caminho = f"{prefixo}/{relativo}"
        info = informacoes.get(caminho)
        if info is None:
            falhas.append(f"ausente no remoto: {relativo}")
            continue
        if info.size != esperado["tamanho"]:
            falhas.append(f"tamanho diferente no remoto: {relativo}")
            continue
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            sha_lfs += 1
            if getattr(lfs, "sha256", None) != esperado["sha256"]:
                falhas.append(f"SHA-256 (LFS) diferente no remoto: {relativo}")
        elif esperado["tamanho"] <= LIMITE_PEQUENO:
            local = hf.hf_hub_download(repositorio, caminho, revision=commit, repo_type="model",
                                       token=credencial.valor())
            sha_baixado += 1
            if sha256_arquivo(local) != esperado["sha256"]:
                falhas.append(f"SHA-256 diferente após baixar de volta: {relativo}")
        else:
            falhas.append(f"não foi possível verificar o SHA-256 de {relativo} (grande e fora do LFS)")
    if f"{prefixo}/{MANIFESTO_PUBLICACAO}" not in informacoes:
        falhas.append("manifesto de publicação ausente no remoto")
    return {"verificada": not falhas, "falhas": falhas, "arquivos_conferidos": len(manifesto["arquivos"]),
            "sha256_lfs": sha_lfs, "sha256_baixados": sha_baixado}


def publicar(pasta: str | Path, repositorio: str, *, credencial: segredos.Credencial | None = None,
             confirmar: bool = False, incluir_ultimo: bool = False, incluir_test: bool = False,
             permitir_nao_concluida: bool = False) -> dict[str, Any]:
    """Publica a execução. Sem `confirmar=True` devolve o plano (nada sai da máquina)."""
    pasta = Path(pasta)
    plano = planejar(pasta, repositorio, incluir_ultimo=incluir_ultimo, incluir_test=incluir_test,
                     permitir_nao_concluida=permitir_nao_concluida)
    if not confirmar:
        return plano
    if plano["problemas"]:
        raise ErroPublicacao("publicação recusada: " + "; ".join(plano["problemas"]))
    if credencial is None:
        credencial = segredos.para_operacao("publicar_hf")[segredos.HUGGING_FACE]
    hf = _importar_hf()
    api = hf.HfApi()
    token, prefixo = credencial.valor(), plano["prefixo"]
    try:
        segredos.verificar_escrita_hf(credencial)
        _garantir_repositorio_privado(api, repositorio, token)
        existentes = [f for f in api.list_repo_files(repositorio, repo_type="model", token=token)
                      if f.startswith(prefixo + "/")]
        if existentes:
            raise ErroPublicacao(f"o prefixo {prefixo}/ já existe no repositório ({len(existentes)} arquivos): "
                                 "execuções publicadas nunca são sobrescritas; use outra tentativa (outro run_id)")
        manifesto = plano["manifesto"]
        operacoes = [hf.CommitOperationAdd(path_in_repo=f"{prefixo}/{r}", path_or_fileobj=str(pasta / r))
                     for r in manifesto["arquivos"]]
        corpo = (json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        operacoes.append(hf.CommitOperationAdd(path_in_repo=f"{prefixo}/{MANIFESTO_PUBLICACAO}",
                                               path_or_fileobj=io.BytesIO(corpo)))
        info = api.create_commit(repositorio, operacoes, commit_message=f"{manifesto['run_id']}: artefatos de {pasta.name}",
                                 repo_type="model", token=token)
        commit = info.oid
        verificacao = _verificar_remoto(hf, api, repositorio, commit, prefixo, manifesto, credencial)
    except ErroPublicacao:
        raise
    except segredos.ErroSegredo:
        raise
    except Exception as erro:  # noqa: BLE001
        raise ErroPublicacao(segredos.limpar(f"falha ao publicar ({type(erro).__name__}): {erro}", credencial)) from None
    referencia = {"destino": "hf", "repositorio": repositorio, "prefixo": prefixo, "commit": commit,
                  "manifesto_sha256": _sha_bytes(corpo), "n_arquivos": len(manifesto["arquivos"]),
                  "bytes": plano["bytes"], "publicado_em": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "incluiu_ultimo": incluir_ultimo, "incluiu_test": incluir_test, **verificacao}
    proveniencia.registrar_publicacao(pasta, referencia)
    if not verificacao["verificada"]:
        raise ErroPublicacao("publicada, mas a verificação do remoto falhou (registrada em experimento.json): "
                             + "; ".join(verificacao["falhas"]))
    return {**plano, "executado": True, "referencia": referencia}


def plano_recuperacao(referencia: Mapping[str, Any], raiz_local: str | Path) -> dict[str, Any]:
    """Dry-run: o que `recuperar` baixaria (a partir da referência gravada em `experimento.json`)."""
    nome = Path(referencia["prefixo"]).name
    return {"executado": False, "repositorio": referencia["repositorio"], "prefixo": referencia["prefixo"],
            "revisao": referencia["commit"], "destino": str(Path(raiz_local) / ids.validar_nome(nome)),
            "arquivos_esperados": referencia.get("n_arquivos"), "bytes_esperados": referencia.get("bytes")}


def recuperar(referencia: Mapping[str, Any], raiz_local: str | Path, *, credencial: segredos.Credencial | None,
              confirmar: bool = False) -> dict[str, Any]:
    """Baixa a execução publicada no commit gravado e confere cada arquivo contra
    o manifesto. Exige `confirmar=True`; recusa destino já existente e não vazio."""
    plano = plano_recuperacao(referencia, raiz_local)
    if not confirmar:
        return plano
    destino = Path(plano["destino"])
    if destino.exists() and any(destino.iterdir()):
        raise ErroPublicacao(f"{destino} já existe e não está vazio: nada é sobrescrito")
    hf = _importar_hf()
    token = credencial.valor() if credencial else None
    prefixo = referencia["prefixo"]
    try:
        local = Path(hf.snapshot_download(referencia["repositorio"], repo_type="model", revision=referencia["commit"],
                                          allow_patterns=[f"{prefixo}/*", f"{prefixo}/**"], token=token,
                                          local_dir=str(Path(raiz_local) / ".hf-recuperacao")))
        origem = local / prefixo
        manifesto = ler_json(origem / MANIFESTO_PUBLICACAO)
    except Exception as erro:  # noqa: BLE001
        raise ErroPublicacao(segredos.limpar(f"falha ao baixar ({type(erro).__name__}): {erro}", credencial)) from None
    falhas = [r for r, i in manifesto["arquivos"].items()
              if not (origem / r).is_file() or sha256_arquivo(origem / r) != i["sha256"]]
    if falhas:
        raise ErroPublicacao(f"arquivos baixados que não conferem com o manifesto: {falhas[:5]}")
    destino.mkdir(parents=True, exist_ok=True)
    for relativo in manifesto["arquivos"]:
        alvo = destino / relativo
        alvo.parent.mkdir(parents=True, exist_ok=True)
        (origem / relativo).replace(alvo)
    return {**plano, "executado": True, "n_arquivos": len(manifesto["arquivos"]),
            "manifesto_sha256": sha256_arquivo(origem / MANIFESTO_PUBLICACAO)}
