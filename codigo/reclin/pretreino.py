"""Proveniência dos pesos pré-treinados: um commit exato, conferido a cada carga.

O BioBERTpt (`pucpr/biobertpt-all`) é baixado uma vez, num commit exato do
Hugging Face Hub, para uma pasta local; todas as execuções da fase científica
carregam os pesos dessa pasta, nunca pelo identificador do Hub. Assim é
possível comprovar que o baseline e os dois estágios do Two-Stage partiram dos
mesmos bytes.

* `baixar(repo, commit, destino)` — aceita só um SHA de commit de 40
  caracteres hexadecimais (nomes de branch e tags são recusados antes de
  qualquer acesso à rede), baixa o snapshot desse commit e grava o manifesto;
* `PROVENIENCIA.json` — o repositório, o commit, o SHA-256 e o tamanho de cada
  arquivo da pasta e o hash do manifesto (`hash_manifesto`);
* `conferir(pasta)` — recalcula os hashes de todos os arquivos e falha se
  algum mudou, sumiu ou apareceu; devolve o registro curto
  (`repositorio`, `commit`, `hash_manifesto`) que vai para o `config.json`
  de cada execução;
* `registro_de(modelo)` — o registro conferido de uma pasta com manifesto, ou
  None quando o modelo não tem manifesto (um identificador do Hub, o modelo
  minúsculo dos testes): nesse caso nada muda no comportamento de quem chama;
* `mesma_proveniencia(registros)` — a checagem antes da comparação final:
  todas as execuções com proveniência, o mesmo commit e o mesmo hash de
  manifesto;
* `ativar_offline()` — `HF_HUB_OFFLINE=1` (e `TRANSFORMERS_OFFLINE=1`), para
  que nenhuma carga consulte a rede. Os scripts chamam antes de carregar.

Limitação aceita (versão enxuta): a ligação entre os arquivos locais e o
commit depende do `snapshot_download(revision=<commit>)` no momento do
download; não há conferência posterior contra os metadados do Hub.

Arquivos e pastas ocultos (como `.cache/`, que o `snapshot_download` cria na
pasta de destino) não entram no manifesto.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from reclin.util.io import gravar_json, ler_json, sha256_arquivo

ARQUIVO = "PROVENIENCIA.json"
FORMATO = 1
PADRAO_COMMIT = re.compile(r"^[0-9a-f]{40}$")
REPOSITORIO_BIOBERTPT = "pucpr/biobertpt-all"


class ErroProveniencia(ValueError):
    """A pasta não corresponde ao manifesto, ou as proveniências divergem."""


def validar_commit(commit: str) -> str:
    """O SHA de 40 caracteres hexadecimais (minúsculos); recusa branch e tag."""
    if not isinstance(commit, str) or not PADRAO_COMMIT.fullmatch(commit):
        raise ErroProveniencia(f"revisão {commit!r} recusada: use o SHA completo do commit "
                               "(40 caracteres hexadecimais), não um nome de branch ou tag")
    return commit


def _arquivos(pasta: Path) -> list[Path]:
    """Os arquivos da pasta que entram no manifesto, em ordem."""
    saida = []
    for caminho in sorted(pasta.rglob("*")):
        relativo = caminho.relative_to(pasta)
        if any(parte.startswith(".") for parte in relativo.parts) or relativo.as_posix() == ARQUIVO:
            continue
        if caminho.is_file():
            saida.append(caminho)
    return saida


def _hash_canonico(obj: Any) -> str:
    """SHA-256 do JSON com chaves ordenadas: não depende da ordem de leitura."""
    texto = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def descrever_arquivos(pasta: str | Path) -> dict[str, dict[str, Any]]:
    """SHA-256 e tamanho de cada arquivo da pasta (caminho relativo, com /)."""
    pasta = Path(pasta)
    return {c.relative_to(pasta).as_posix(): {"sha256": sha256_arquivo(c), "tamanho": c.stat().st_size}
            for c in _arquivos(pasta)}


def hash_pasta(pasta: str | Path) -> str:
    """Hash do conteúdo de uma pasta (nomes, tamanhos e SHA-256 dos arquivos).
    Serve também para identificar o `melhor_modelo/` de uma execução."""
    return _hash_canonico(descrever_arquivos(pasta))


def montar_manifesto(pasta: str | Path, *, repositorio: str, commit: str) -> dict[str, Any]:
    arquivos = descrever_arquivos(pasta)
    if not arquivos:
        raise ErroProveniencia(f"{pasta} não tem arquivos para registrar")
    corpo = {"repositorio": repositorio, "commit": validar_commit(commit), "arquivos": arquivos}
    return {"formato": FORMATO, **corpo, "hash_manifesto": _hash_canonico(corpo)}


def gravar(pasta: str | Path, *, repositorio: str, commit: str) -> dict[str, Any]:
    """Grava o `PROVENIENCIA.json` da pasta e devolve o manifesto."""
    manifesto = montar_manifesto(pasta, repositorio=repositorio, commit=commit)
    manifesto["gravado_em"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    gravar_json(Path(pasta) / ARQUIVO, manifesto)
    return manifesto


def tem_manifesto(modelo: str | Path) -> bool:
    return Path(modelo).is_dir() and (Path(modelo) / ARQUIVO).is_file()


def conferir(pasta: str | Path) -> dict[str, str]:
    """Confere todos os arquivos contra o manifesto e devolve o registro curto.
    Falha se algum arquivo mudou, sumiu ou apareceu, ou se o manifesto foi
    editado (o `hash_manifesto` não confere com o conteúdo)."""
    pasta = Path(pasta)
    if not (pasta / ARQUIVO).is_file():
        raise ErroProveniencia(f"{pasta} não tem {ARQUIVO}")
    manifesto = ler_json(pasta / ARQUIVO)
    corpo = {k: manifesto.get(k) for k in ("repositorio", "commit", "arquivos")}
    validar_commit(corpo["commit"])
    if _hash_canonico(corpo) != manifesto.get("hash_manifesto"):
        raise ErroProveniencia(f"{pasta / ARQUIVO}: o hash_manifesto não confere com o conteúdo")
    atuais = descrever_arquivos(pasta)
    registrados = corpo["arquivos"] or {}
    problemas = [f"{n}: ausente" for n in sorted(set(registrados) - set(atuais))]
    problemas += [f"{n}: não registrado no manifesto" for n in sorted(set(atuais) - set(registrados))]
    problemas += [f"{n}: SHA-256 ou tamanho divergente" for n in sorted(set(atuais) & set(registrados))
                  if atuais[n] != registrados[n]]
    if problemas:
        raise ErroProveniencia(f"os pesos em {pasta} não conferem com {ARQUIVO}: " + "; ".join(problemas))
    return {"repositorio": corpo["repositorio"], "commit": corpo["commit"],
            "hash_manifesto": manifesto["hash_manifesto"]}


def registro_de(modelo: str | Path | None) -> dict[str, str] | None:
    """O registro conferido de uma pasta com manifesto; None se não houver
    manifesto (identificador do Hub, ou uma pasta sem `PROVENIENCIA.json`)."""
    if modelo is None or not tem_manifesto(modelo):
        return None
    return conferir(modelo)


def mesma_proveniencia(registros: Mapping[str, Mapping[str, str] | None]) -> dict[str, str]:
    """Falha se alguma execução não tiver proveniência ou se os commits ou os
    hashes de manifesto divergirem; devolve a proveniência comum."""
    if not registros:
        raise ErroProveniencia("nenhuma execução para conferir")
    sem = sorted(nome for nome, r in registros.items() if not r)
    if sem:
        raise ErroProveniencia("execuções sem proveniência dos pesos: " + ", ".join(sem))
    chaves = {nome: (r["commit"], r["hash_manifesto"]) for nome, r in registros.items()}
    if len(set(chaves.values())) > 1:
        detalhe = "; ".join(f"{n}: commit {c[:12]}…, manifesto {h[:12]}…" for n, (c, h) in sorted(chaves.items()))
        raise ErroProveniencia("as execuções não usam a mesma revisão dos pesos: " + detalhe)
    primeiro = next(iter(registros.values()))
    return {"repositorio": primeiro["repositorio"], "commit": primeiro["commit"],
            "hash_manifesto": primeiro["hash_manifesto"]}


def ativar_offline() -> None:
    """Nenhuma consulta ao Hub a partir daqui. Se a biblioteca do Hub já foi
    importada (a variável é lida no import), a constante também é ligada."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    constantes = sys.modules.get("huggingface_hub.constants")
    if constantes is not None:
        constantes.HF_HUB_OFFLINE = True


def baixar(repositorio: str, commit: str, destino: str | Path, *, baixador: Any = None) -> dict[str, Any]:
    """Baixa o snapshot do commit para `destino` (que precisa estar vazio ou
    não existir) e grava o manifesto. `baixador` substitui o
    `huggingface_hub.snapshot_download` (nos testes)."""
    validar_commit(commit)
    destino = Path(destino)
    if destino.exists() and any(destino.iterdir()):
        raise ErroProveniencia(f"{destino} já existe e não está vazio: escolha uma pasta nova")
    if baixador is None:
        from huggingface_hub import snapshot_download as baixador
    destino.mkdir(parents=True, exist_ok=True)
    baixador(repo_id=repositorio, revision=commit, local_dir=str(destino))
    return gravar(destino, repositorio=repositorio, commit=commit)
