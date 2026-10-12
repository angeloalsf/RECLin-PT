"""Obtenção do código: clone de uma branch (notebook atual) ou tarball de um SHA exato (sem Git).

Duas formas, ambas só com a biblioteca padrão:

* `clonar_codigo` — clone raso de uma **branch** (`git clone --depth 1 --branch`), gravando o commit clonado
  (HEAD) em `REVISAO.json` (`origem: "git"`). É o que o notebook do baseline usa: o código pode mudar entre sessões
  e quem garante a coerência é a identidade do código, conferida na retomada (`identidade_codigo`). O token, se
  houver (repositório privado), vai ao Git só por `GIT_ASKPASS` (script temporário, 700, apagado em `finally`) no
  ambiente do subprocesso: nunca em URL, argv, `.git/config` ou `os.environ` da sessão;
* `preparar_codigo` — tarball de um SHA completo (descrito abaixo), mantido para quem precisa de uma revisão fixa.

Este módulo usa **só a biblioteca padrão** e não importa nada de `reclin`: a
primeira célula de um notebook baixa este único arquivo na revisão desejada
(`CELULA_INICIAL`, em `docs/arquitetura/infraestrutura-colab.md`) e o usa para
baixar e extrair o resto do projeto.

Regras:

* a revisão é o SHA completo do commit (40 caracteres hexadecimais); nome de
  branch ou tag é recusado antes de qualquer acesso à rede;
* o tarball vem de `api.github.com/repos/<dono>/<repo>/tarball/<sha>`; o
  token, quando existe (repositório privado), vai só no cabeçalho
  `Authorization` do primeiro pedido, nunca em URL, nunca é repassado ao
  destino do redirecionamento (outro servidor) e nunca aparece em mensagem de
  erro, registro ou arquivo;
* o nome da pasta de topo do tarball carrega os 7 primeiros caracteres do SHA
  que o GitHub serviu: se não forem os do SHA pedido, a extração é recusada;
* a extração recusa caminhos absolutos, `..`, links e arquivos especiais, e
  acontece numa pasta temporária que só vira o destino por renomeação — não
  existe pasta de código "meio extraída";
* um destino já extraído para o mesmo SHA é reaproveitado (sem novo download),
  desde que o `REVISAO.json` e a lista de arquivos conferam.

`REVISAO.json` (na pasta de topo extraída) é o registro de proveniência do
código: repositório, SHA, SHA-256 do tarball, número de arquivos e a lista
`{caminho: tamanho}`. Não contém credencial.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

API = "https://api.github.com"
ARQUIVO_REVISAO = "REVISAO.json"
PADRAO_SHA = re.compile(r"^[0-9a-f]{40}$")
PADRAO_REPOSITORIO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PADRAO_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
LIMITE_TARBALL = 512 * 1024 * 1024        # recusa um tarball maior que isto
LIMITE_REDIRECIONAMENTOS = 3
FORMATO = 1


class ErroCodigoColab(RuntimeError):
    """O código não pôde ser obtido ou verificado."""


def validar_sha(sha: str) -> str:
    """O SHA completo (40 hexadecimais minúsculos); recusa branch, tag e SHA curto."""
    if not isinstance(sha, str) or not PADRAO_SHA.fullmatch(sha):
        raise ErroCodigoColab(f"revisão {sha!r} recusada: use o SHA completo do commit "
                              "(40 caracteres hexadecimais minúsculos), não um nome de branch, tag ou SHA curto")
    return sha


def validar_repositorio(repositorio: str) -> str:
    if not isinstance(repositorio, str) or not PADRAO_REPOSITORIO.fullmatch(repositorio):
        raise ErroCodigoColab(f"repositório {repositorio!r} inválido; use o formato dono/repositorio")
    return repositorio


def _limpar(texto: str, token: str | None) -> str:
    return texto.replace(token, "***") if token else texto


class _SemRedirecionar(urllib.request.HTTPRedirectHandler):
    """Não segue redirecionamentos sozinho: quem chama decide quais cabeçalhos repassar."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def _abrir(url: str, token: str | None, timeout: float):
    """Abre `url` e segue até `LIMITE_REDIRECIONAMENTOS` redirecionamentos; o
    cabeçalho de autorização só vai ao primeiro pedido (api.github.com)."""
    abridor = urllib.request.build_opener(_SemRedirecionar)
    atual, primeiro = url, True
    for _ in range(LIMITE_REDIRECIONAMENTOS + 1):
        pedido = urllib.request.Request(atual, headers={
            "User-Agent": "reclin-colab", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"})
        if primeiro and token:
            pedido.add_header("Authorization", f"Bearer {token}")
        try:
            return abridor.open(pedido, timeout=timeout)
        except urllib.error.HTTPError as erro:
            if erro.code in (301, 302, 303, 307, 308) and erro.headers.get("Location"):
                atual, primeiro = erro.headers["Location"], False
                continue
            dicas = {401: "token inválido ou expirado", 403: "token sem permissão (ou limite de requisições)",
                     404: "repositório ou revisão inexistente — ou token ausente/sem acesso a um repositório privado"}
            raise ErroCodigoColab(f"GitHub respondeu {erro.code}: "
                                  f"{dicas.get(erro.code, erro.reason)}") from None
        except urllib.error.URLError as erro:
            raise ErroCodigoColab(f"sem acesso ao GitHub: {_limpar(str(erro.reason), token)}") from None
    raise ErroCodigoColab("redirecionamentos demais ao baixar o tarball")


def baixar_tarball(repositorio: str, sha: str, destino: str | Path, *, token: str | None = None,
                   timeout: float = 120.0) -> dict[str, Any]:
    """Baixa o tarball do commit `sha` para o arquivo `destino` (gravado em
    `.tmp` e renomeado). Devolve `{sha256, bytes}`."""
    validar_repositorio(repositorio)
    validar_sha(sha)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    resumo, total = hashlib.sha256(), 0
    resposta = _abrir(f"{API}/repos/{repositorio}/tarball/{sha}", token, timeout)
    try:
        with resposta, open(temporario, "wb") as f:
            while True:
                bloco = resposta.read(1024 * 1024)
                if not bloco:
                    break
                total += len(bloco)
                if total > LIMITE_TARBALL:
                    raise ErroCodigoColab(f"tarball maior que {LIMITE_TARBALL // (1024 * 1024)} MiB: recusado")
                resumo.update(bloco)
                f.write(bloco)
        os.replace(temporario, destino)
    finally:
        if temporario.exists():
            temporario.unlink()
    return {"sha256": resumo.hexdigest(), "bytes": total}


def _seguro(nome: str) -> bool:
    partes = Path(nome).parts
    return bool(partes) and not Path(nome).is_absolute() and ".." not in partes and "\\" not in nome


def extrair_seguro(tarball: str | Path, destino: str | Path, *, sha: str) -> dict[str, Any]:
    """Extrai o tarball em `destino` (pasta nova), sem a pasta de topo.

    Só arquivos regulares e pastas; recusa caminhos absolutos ou com `..`, e
    ignora (contando) links simbólicos e arquivos especiais. Confere que a pasta
    de topo termina nos 7 primeiros caracteres do `sha`. Devolve
    `{topo, n_arquivos, arquivos: {caminho: tamanho}, ignorados}`."""
    validar_sha(sha)
    destino = Path(destino)
    arquivos: dict[str, int] = {}
    ignorados = 0
    with tarfile.open(tarball, mode="r:*") as tar:
        membros = tar.getmembers()
        topos = {Path(m.name).parts[0] for m in membros if Path(m.name).parts}
        if len(topos) != 1:
            raise ErroCodigoColab(f"tarball com {len(topos)} pastas de topo (esperada 1)")
        topo = topos.pop()
        if not topo.rsplit("-", 1)[-1] or not sha.startswith(topo.rsplit("-", 1)[-1]) \
                or len(topo.rsplit("-", 1)[-1]) < 7:
            raise ErroCodigoColab(f"a pasta de topo {topo!r} não corresponde ao SHA pedido {sha[:7]}…: "
                                  "o conteúdo não é o da revisão solicitada")
        destino.mkdir(parents=True, exist_ok=False)
        for m in membros:
            relativo = Path(*Path(m.name).parts[1:]).as_posix() if len(Path(m.name).parts) > 1 else ""
            if not relativo:
                continue
            if not _seguro(relativo):
                raise ErroCodigoColab(f"caminho inseguro no tarball: {m.name!r}")
            alvo = destino / relativo
            if m.isdir():
                alvo.mkdir(parents=True, exist_ok=True)
            elif m.isreg():
                alvo.parent.mkdir(parents=True, exist_ok=True)
                origem = tar.extractfile(m)
                with open(alvo, "wb") as f:
                    shutil.copyfileobj(origem, f)
                os.chmod(alvo, 0o755 if m.mode & 0o111 else 0o644)
                arquivos[relativo] = m.size
            else:
                ignorados += 1
    return {"topo": topo, "n_arquivos": len(arquivos), "arquivos": arquivos, "ignorados": ignorados}


def _agora() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ler_revisao(pasta: str | Path) -> dict[str, Any] | None:
    """O `REVISAO.json` de uma pasta de código extraída, ou None."""
    caminho = Path(pasta) / ARQUIVO_REVISAO
    if not caminho.is_file():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def _confere(pasta: Path, revisao: dict[str, Any]) -> bool:
    """Reaproveitamento: o registro existe e cada arquivo listado está lá com o
    mesmo tamanho (conferência barata; não recalcula hashes)."""
    try:
        return all((pasta / c).is_file() and (pasta / c).stat().st_size == t
                   for c, t in revisao["arquivos"].items())
    except (KeyError, OSError, AttributeError):
        return False


def preparar_codigo(repositorio: str, sha: str, *, token: str | None = None,
                    destino: str | Path = "/content/reclin_codigo", subpasta: str = "codigo",
                    arquivo_bootstrap: str | Path | None = None) -> dict[str, Any]:
    """Garante o código de `repositorio` na revisão `sha` em
    `<destino>/<dono>__<repo>@<sha>/` e devolve o registro de revisão mais
    `raiz` (a pasta extraída) e `codigo` (a subpasta `codigo/`).

    Reaproveita uma extração anterior do mesmo SHA. `arquivo_bootstrap`, se
    dado, é o arquivo deste módulo baixado pela primeira célula: precisa ser
    idêntico ao da árvore extraída (senão a revisão usada para obter o código
    não é a que foi extraída)."""
    validar_repositorio(repositorio)
    validar_sha(sha)
    destino = Path(destino)
    pasta = destino / f"{repositorio.replace('/', '__')}@{sha}"
    revisao = ler_revisao(pasta)
    if revisao and revisao.get("sha") == sha and revisao.get("repositorio") == repositorio and _confere(pasta, revisao):
        revisao = {**revisao, "reaproveitado": True}
    else:
        if pasta.exists():
            shutil.rmtree(pasta)
        destino.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destino, prefix=".tmp-") as tmp:
            tmp = Path(tmp)
            baixado = baixar_tarball(repositorio, sha, tmp / "codigo.tar.gz", token=token)
            extraido = extrair_seguro(tmp / "codigo.tar.gz", tmp / "extraido", sha=sha)
            revisao = {"formato": FORMATO, "repositorio": repositorio, "sha": sha, "origem": "tarball",
                       "tarball_sha256": baixado["sha256"], "tarball_bytes": baixado["bytes"],
                       "n_arquivos": extraido["n_arquivos"], "arquivos": extraido["arquivos"],
                       "links_e_especiais_ignorados": extraido["ignorados"], "extraido_em": _agora()}
            (tmp / "extraido" / ARQUIVO_REVISAO).write_text(
                json.dumps(revisao, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp / "extraido", pasta)
        revisao = {**revisao, "reaproveitado": False}
    codigo = pasta / subpasta
    if not (codigo / "pyproject.toml").is_file() or not (codigo / "reclin" / "__init__.py").is_file():
        raise ErroCodigoColab(f"{codigo} não parece o diretório `codigo/` do RECLin-PT "
                              "(sem pyproject.toml ou reclin/__init__.py)")
    if arquivo_bootstrap is not None:
        propria = codigo / "reclin" / "experimento" / "codigo_colab.py"
        if not propria.is_file() or propria.read_bytes() != Path(arquivo_bootstrap).read_bytes():
            raise ErroCodigoColab("o arquivo usado para baixar o código difere do mesmo arquivo na revisão "
                                  "extraída: use o SHA completo de uma revisão que contenha este módulo")
    return {**revisao, "raiz": str(pasta), "codigo": str(codigo)}


# --------------------------------------------------------------------------- #
# Clone de uma branch                                                         #
# --------------------------------------------------------------------------- #
def validar_branch(branch: str) -> str:
    if not isinstance(branch, str) or not PADRAO_BRANCH.fullmatch(branch) or ".." in branch \
            or branch.endswith(("/", ".lock")):
        raise ErroCodigoColab(f"branch {branch!r} inválida")
    return branch


@contextlib.contextmanager
def askpass(token: str) -> Iterator[dict[str, str]]:
    """Ambiente do Git com `GIT_ASKPASS`: um script temporário (700) que lê o token da variável
    `RECLIN_GIT_TOKEN` do **subprocesso**. O token não é gravado no script nem em `os.environ`; a pasta é
    apagada ao sair (também em erro)."""
    pasta = Path(tempfile.mkdtemp(prefix="reclin-askpass-"))
    try:
        os.chmod(pasta, 0o700)
        script = pasta / "askpass.sh"
        script.write_text('#!/bin/sh\ncase "$1" in\n  *sername*) printf "%s" "x-access-token" ;;\n'
                          '  *) printf "%s" "$RECLIN_GIT_TOKEN" ;;\nesac\n', encoding="utf-8")
        os.chmod(script, 0o700)
        yield {"GIT_ASKPASS": str(script), "RECLIN_GIT_TOKEN": token}
    finally:
        shutil.rmtree(pasta, ignore_errors=True)


def _git(argumentos: list[str], cwd: Path | None, env_extra: dict[str, str], token: str | None) -> str:
    r = subprocess.run(["git", "-c", "credential.helper=", *argumentos], cwd=cwd, check=False,
                       env={**os.environ, **env_extra, "GIT_TERMINAL_PROMPT": "0"}, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    saida = _limpar(((r.stdout or "") + (r.stderr or "")).strip(), token)
    if r.returncode != 0:
        raise ErroCodigoColab(f"git {argumentos[0]} falhou ({r.returncode})" + (f": {saida.splitlines()[-1]}" if saida else "")
                              + ". Se o repositório for privado, confira o Secret GITHUB_TOKEN")
    return saida


def clonar_codigo(repositorio: str, branch: str, destino: str | Path = "/content/RECLin-PT", *,
                  token: str | None = None, subpasta: str = "codigo",
                  git: Callable[[list[str], "Path | None", "dict[str, str]", "str | None"], str] = _git) -> dict[str, Any]:
    """Garante um clone raso de `repositorio` na `branch` em `destino` e devolve o registro de revisão
    (`sha` = HEAD clonado) mais `raiz`, `codigo` e `reaproveitado`.

    Um clone já existente é reaproveitado **sem** atualizar (o código não muda no meio da sessão), desde que
    aponte para o mesmo repositório e a mesma branch e não tenha arquivos rastreados alterados. Grava
    `REVISAO.json` na raiz do clone (`origem: "git"`)."""
    validar_repositorio(repositorio)
    validar_branch(branch)
    url = f"https://github.com/{repositorio}.git"
    destino = Path(destino)
    with contextlib.ExitStack() as pilha:
        env = pilha.enter_context(askpass(token)) if token else {}

        def rodar(*argumentos: str, cwd: Path | None = None) -> str:
            return git(list(argumentos), cwd, env, token)

        reaproveitado = (destino / ".git").is_dir()
        if reaproveitado:
            if rodar("remote", "get-url", "origin", cwd=destino) != url:
                raise ErroCodigoColab(f"{destino} é um clone de outro repositório: remova a pasta ou corrija REPOSITORIO")
            if rodar("rev-parse", "--abbrev-ref", "HEAD", cwd=destino) != branch:
                raise ErroCodigoColab(f"{destino} está em outra branch: remova a pasta ou corrija BRANCH_CODIGO")
            if rodar("status", "--porcelain", "--untracked-files=no", cwd=destino):
                raise ErroCodigoColab(f"{destino} tem arquivos alterados: o código não é o da branch. Remova a pasta")
        else:
            if destino.exists() and any(destino.iterdir()):
                raise ErroCodigoColab(f"{destino} existe, não é um clone e não está vazio: remova-o antes")
            destino.parent.mkdir(parents=True, exist_ok=True)
            rodar("clone", "--depth", "1", "--branch", branch, url, str(destino))
            if rodar("rev-parse", "--abbrev-ref", "HEAD", cwd=destino) != branch:
                raise ErroCodigoColab(f"{branch!r} não é uma branch (é tag ou outra referência): informe uma branch")
        sha = rodar("rev-parse", "HEAD", cwd=destino)
    if not PADRAO_SHA.fullmatch(sha):
        raise ErroCodigoColab("não consegui ler o commit clonado (HEAD)")
    anterior = ler_revisao(destino)
    if not (anterior and anterior.get("origem") == "git" and anterior.get("sha") == sha
            and anterior.get("branch") == branch and anterior.get("repositorio") == repositorio):
        revisao = {"formato": FORMATO, "origem": "git", "repositorio": repositorio, "branch": branch, "sha": sha,
                   "tarball_sha256": None, "clonado_em": _agora()}
        (destino / ARQUIVO_REVISAO).write_text(json.dumps(revisao, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
    else:
        revisao = anterior
    codigo = destino / subpasta
    if not (codigo / "pyproject.toml").is_file() or not (codigo / "reclin" / "__init__.py").is_file():
        raise ErroCodigoColab(f"{codigo} não parece o diretório `codigo/` do RECLin-PT "
                              "(sem pyproject.toml ou reclin/__init__.py) na branch informada")
    return {**revisao, "raiz": str(destino), "codigo": str(codigo), "reaproveitado": reaproveitado}
