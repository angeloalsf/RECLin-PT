"""Localização robusta da pasta `codigo/` do projeto e sua ativação.

`localizar_codigo` procura, nesta ordem: o argumento `inicio`, a variável de
ambiente `RECLIN_CODIGO`, o diretório de trabalho e a pasta deste arquivo; em
cada ponto sobe pelos pais (e olha também as subpastas `codigo/` e
`RECLin-PT/codigo/`) até achar uma pasta com `pyproject.toml` do projeto
`reclin` e `reclin/__init__.py`. Não depende do nome da pasta de checkout nem
de onde o Colab extraiu o código.

`ativar` põe a pasta em `sys.path` e devolve o ambiente (`PYTHONPATH`) que os
subprocessos precisam para `import reclin` quando o pacote não está instalado;
`rodar_script` chama um script de `codigo/scripts/` com esse ambiente. Nada disso
acontece ao importar o módulo.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping

VARIAVEL = "RECLIN_CODIGO"
NOME_PROJETO = re.compile(r'^\s*name\s*=\s*["\']reclin["\']\s*$', re.M)


class ErroRaiz(RuntimeError):
    """A pasta `codigo/` do projeto não foi encontrada."""


def e_codigo(pasta: str | Path) -> bool:
    """A pasta é o `codigo/` do RECLin-PT: `pyproject.toml` do projeto `reclin` e o pacote."""
    pasta = Path(pasta)
    pyproject = pasta / "pyproject.toml"
    if not pyproject.is_file() or not (pasta / "reclin" / "__init__.py").is_file():
        return False
    try:
        return bool(NOME_PROJETO.search(pyproject.read_text(encoding="utf-8")))
    except OSError:
        return False


def _candidatos(ponto: Path) -> list[Path]:
    ponto = ponto.resolve()
    saida: list[Path] = []
    for pai in (ponto, *ponto.parents):
        saida += [pai, pai / "codigo", pai / "RECLin-PT" / "codigo"]
    return saida


def localizar_codigo(inicio: str | Path | None = None) -> Path:
    pontos: list[Path] = []
    if inicio is not None:
        pontos.append(Path(inicio))
    if os.environ.get(VARIAVEL):
        pontos.append(Path(os.environ[VARIAVEL]))
    pontos += [Path.cwd(), Path(__file__).resolve().parent]
    vistos: list[str] = []
    for ponto in pontos:
        for candidato in _candidatos(ponto):
            if e_codigo(candidato):
                return candidato.resolve()
            vistos.append(str(candidato))
    raise ErroRaiz("não encontrei a pasta `codigo/` do RECLin-PT (pyproject.toml do projeto `reclin` + "
                   f"reclin/__init__.py). Pontos de partida: {[str(p) for p in pontos]}. "
                   f"Informe o caminho ou defina {VARIAVEL}.")


def ativar(codigo: str | Path) -> dict[str, str]:
    """Põe `codigo` no início de `sys.path` (uma vez) e devolve um ambiente
    para subprocessos com `PYTHONPATH` e `RECLIN_CODIGO` apontando para ele."""
    codigo = Path(codigo).resolve()
    if not e_codigo(codigo):
        raise ErroRaiz(f"{codigo} não é a pasta `codigo/` do RECLin-PT")
    texto = str(codigo)
    if texto not in sys.path:
        sys.path.insert(0, texto)
    return ambiente_para_subprocessos(codigo)


def ambiente_para_subprocessos(codigo: str | Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Cópia de `base` (padrão: `os.environ`) com o projeto no `PYTHONPATH`."""
    codigo = str(Path(codigo).resolve())
    ambiente = dict(os.environ if base is None else base)
    anterior = [p for p in ambiente.get("PYTHONPATH", "").split(os.pathsep) if p and p != codigo]
    ambiente["PYTHONPATH"] = os.pathsep.join([codigo, *anterior])
    ambiente[VARIAVEL] = codigo
    return ambiente


def rodar_script(codigo: str | Path, script: str, *argumentos: str, ambiente: Mapping[str, str] | None = None,
                 verificar: bool = True, log: str | Path | None = None,
                 periodico: tuple[float, Callable[[], None]] | None = None) -> int:
    """Roda `codigo/scripts/<script>` com o interpretador atual e devolve o código
    de saída. É como os notebooks chamam os scripts reais, sem reimplementar nada.

    Sem `log` nem `periodico`, a saída é herdada (aparece na célula). Com `log`,
    a saída (stdout e stderr juntos) continua aparecendo na célula e é também
    acrescentada ao arquivo, linha a linha, depois de passar por
    `segredos.limpar`. Com `periodico=(segundos, funcao)`, `funcao` é chamada a
    cada `segundos` enquanto o script roda (por exemplo, para espelhar a
    execução no Drive); um erro nela é impresso e não interrompe o script.

    As credenciais nunca vão como argumento: quem precisa delas (publicação) as
    lê dos Secrets dentro do próprio script. `verificar=True` levanta
    `ErroRaiz` se o script terminar com erro (código ≠ 0)."""
    codigo = Path(codigo)
    arquivo = codigo / "scripts" / script
    if not arquivo.is_file():
        raise ErroRaiz(f"script inexistente: {arquivo}")
    env = ambiente_para_subprocessos(codigo, ambiente)
    comando = [sys.executable, str(arquivo), *argumentos]
    if log is None and periodico is None:
        resultado = subprocess.run(comando, env=env)
        codigo_saida = resultado.returncode
    else:
        codigo_saida = _rodar_com_log(comando, env, Path(log) if log is not None else None, periodico)
    if verificar and codigo_saida != 0:
        raise ErroRaiz(f"{script} {' '.join(argumentos)} terminou com código {codigo_saida}")
    return codigo_saida


def _rodar_com_log(comando: list[str], env: Mapping[str, str], log: Path | None,
                   periodico: tuple[float, Callable[[], None]] | None) -> int:
    from reclin.experimento import segredos  # noqa: PLC0415

    arquivo = None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        arquivo = open(log, "a", encoding="utf-8")                      # noqa: SIM115 — fechado no finally
    ambiente_filho = {**env, "PYTHONUNBUFFERED": "1"}           # sem isto a saída em pipe só aparece em blocos
    processo = subprocess.Popen(comando, env=ambiente_filho, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)

    def copiar() -> None:
        for linha in processo.stdout:                                  # type: ignore[union-attr]
            linha = segredos.limpar(linha)
            sys.stdout.write(linha)
            sys.stdout.flush()
            if arquivo is not None:
                arquivo.write(linha)
                arquivo.flush()

    leitor = threading.Thread(target=copiar, daemon=True)
    leitor.start()
    try:
        proximo = time.monotonic() + periodico[0] if periodico else None
        while True:
            try:
                processo.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if periodico and time.monotonic() >= proximo:
                    try:
                        periodico[1]()
                    except Exception as erro:  # noqa: BLE001 — o espelhamento não pode derrubar o treino
                        print(segredos.limpar(f"[aviso] tarefa periódica falhou: {type(erro).__name__}: {erro}"))
                    proximo = time.monotonic() + periodico[0]
    except BaseException:
        processo.terminate()
        raise
    finally:
        leitor.join(timeout=10)
        if arquivo is not None:
            arquivo.close()
    return processo.returncode
