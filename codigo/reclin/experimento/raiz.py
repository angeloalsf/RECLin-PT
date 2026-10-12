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
from pathlib import Path
from typing import Mapping

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
                 verificar: bool = True) -> int:
    """Roda `codigo/scripts/<script>` com o interpretador atual, herdando a saída
    (aparece na célula do notebook), e devolve o código de saída. É como os
    notebooks chamam os scripts reais, sem reimplementar nada.

    As credenciais nunca vão como argumento: quem precisa delas (publicação) as
    lê dos Secrets dentro do próprio script. `verificar=True` levanta
    `ErroRaiz` se o script terminar com erro (código ≠ 0)."""
    codigo = Path(codigo)
    arquivo = codigo / "scripts" / script
    if not arquivo.is_file():
        raise ErroRaiz(f"script inexistente: {arquivo}")
    env = ambiente_para_subprocessos(codigo, ambiente)
    resultado = subprocess.run([sys.executable, str(arquivo), *argumentos], env=env)
    if verificar and resultado.returncode != 0:
        raise ErroRaiz(f"{script} {' '.join(argumentos)} terminou com código {resultado.returncode}")
    return resultado.returncode
