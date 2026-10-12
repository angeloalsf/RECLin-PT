"""Ambiente de execução: versões, GPU, plano de instalação e compatibilidade para retomar.

Não assume que o Colab é igual à máquina local. Lê as exigências do próprio
projeto — `pyproject.toml` (`[project].dependencies`, `requires-python`) e
`requirements.txt` (versões fixadas dos experimentos) — e compara com o que
está instalado, sem importar os pacotes (usa `importlib.metadata`).

Dois modos de instalação:

* `minimo` (padrão): instala só o que **falta** ou está **abaixo do mínimo**
  de `pyproject.toml`; um pacote que satisfaz o mínimo não é reinstalado,
  mesmo que difira da versão fixada (a diferença é apenas registrada como
  aviso: resultados numéricos podem variar). É o que evita reinstalar o
  torch do Colab a cada sessão;
* `fixado`: leva todos os pacotes de `requirements.txt` exigidos pelo treino
  às versões fixadas.

`instalar` só executa o `pip` com `executar=True`; sem isso devolve o plano
(dry-run). A coleta (`coletar`) é a mesma estrutura que as sessões de treino
gravam (`treino.reprodutibilidade.ambiente`: `python`, `plataforma`,
`versoes`, `cuda`, `cudnn`, `gpu`), mais `gpus` e `colab`.

Retomada (`comparar_para_retomada`): divergência de Python (minor), torch,
transformers, tokenizers, numpy ou do modelo de GPU **bloqueia**; as demais
diferenças viram aviso. Reaproveitar um estágio concluído (só inferência) usa
`comparar_para_reaproveitar`, que nunca bloqueia por versões ou GPU.
"""
from __future__ import annotations

import importlib.metadata as metadata
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from reclin.experimento import colab

PACOTES_REGISTRADOS = ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy", "scipy")
# só o treino e a avaliação precisam destes; os demais de requirements.txt servem a corpus e figuras
PACOTES_DE_TREINO = ("numpy", "scipy", "torch", "transformers", "tokenizers", "safetensors", "huggingface_hub")
BLOQUEIAM_RETOMADA = ("torch", "transformers", "tokenizers", "numpy")
GPU_REFERENCIA = ("T4",)
GPU_PERMITIDAS = ("L4", "A100")
MODOS = ("minimo", "fixado")


class ErroAmbiente(RuntimeError):
    """O ambiente não atende ao projeto ou a uma operação."""


# ----------------------------------------------------------------- versões

def _versao(pacote: str) -> str | None:
    try:
        return metadata.version(pacote)
    except metadata.PackageNotFoundError:
        return None


def _numeros(versao: str) -> tuple[int, ...]:
    return tuple(int(m) for m in re.findall(r"\d+", versao.split("+")[0].split("-")[0])[:4])


def _cmp(a: str, b: str) -> int:
    x, y = _numeros(a), _numeros(b)
    n = max(len(x), len(y))
    x, y = x + (0,) * (n - len(x)), y + (0,) * (n - len(y))
    return (x > y) - (x < y)


def _nome_normalizado(nome: str) -> str:
    return re.sub(r"[-_.]+", "-", nome).lower()


# ---------------------------------------------------------- exigências

def requisitos_do_projeto(codigo: str | Path) -> dict[str, Any]:
    """`{python, minimos: {pacote: versao}, fixados: {pacote: versao}}`, lidos de
    `pyproject.toml` e `requirements.txt` da pasta `codigo/`. Aceita os
    operadores `>=` e `==` (os que o projeto usa); outros viram erro explícito."""
    codigo = Path(codigo)
    texto = (codigo / "pyproject.toml").read_text(encoding="utf-8")
    python = re.search(r'requires-python\s*=\s*"\s*>=\s*([\d.]+)\s*"', texto)
    bloco = re.search(r"^dependencies\s*=\s*\[(.*?)\]", texto, re.M | re.S)
    if not bloco:
        raise ErroAmbiente("não encontrei `dependencies` em [project] de pyproject.toml")
    minimos = {}
    for item in re.findall(r'"([^"]+)"', bloco.group(1)):
        m = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)\s*(>=|==)\s*([0-9][0-9A-Za-z.+-]*)\s*", item)
        if not m:
            raise ErroAmbiente(f"dependência não prevista em pyproject.toml: {item!r} (aceito: >= ou ==)")
        minimos[_nome_normalizado(m.group(1))] = m.group(3)
    fixados = {}
    arquivo = codigo / "requirements.txt"
    if arquivo.is_file():
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            linha = linha.split("#")[0].strip()
            m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.+-]*)", linha)
            if m:
                fixados[_nome_normalizado(m.group(1))] = m.group(2)
    return {"python": python.group(1) if python else None, "minimos": minimos, "fixados": fixados}


def avaliar_pacotes(requisitos: Mapping[str, Any], *, modo: str = "minimo",
                    instalado: Mapping[str, str | None] | None = None) -> list[dict[str, Any]]:
    """Uma linha por pacote exigido: `estado` é `ok`, `falta`, `abaixo_do_minimo`
    ou `difere_do_fixado` (só aviso no modo mínimo; ação no modo fixado)."""
    if modo not in MODOS:
        raise ErroAmbiente(f"modo {modo!r} desconhecido; opções: {MODOS}")
    minimos, fixados = requisitos["minimos"], requisitos["fixados"]
    nomes = sorted(set(minimos) | (set(fixados) & {_nome_normalizado(p) for p in PACOTES_DE_TREINO}))
    linhas = []
    for nome in nomes:
        atual = instalado[nome] if instalado is not None and nome in instalado else _versao(nome)
        minimo, fixado = minimos.get(nome), fixados.get(nome)
        if atual is None:
            estado = "falta"
        elif minimo and _cmp(atual, minimo) < 0:
            estado = "abaixo_do_minimo"
        elif fixado and _cmp(atual, fixado) != 0:
            estado = "difere_do_fixado"
        else:
            estado = "ok"
        linhas.append({"pacote": nome, "instalado": atual, "minimo": minimo, "fixado": fixado, "estado": estado})
    return linhas


def plano_instalacao(requisitos: Mapping[str, Any], *, modo: str = "minimo",
                     instalado: Mapping[str, str | None] | None = None,
                     modulos_carregados: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """O que instalar (e por quê), sem instalar. `reiniciar` indica que algum
    pacote a trocar já está importado nesta sessão (o Colab precisa reiniciar)."""
    linhas = avaliar_pacotes(requisitos, modo=modo, instalado=instalado)
    acionaveis = {"falta", "abaixo_do_minimo"} | ({"difere_do_fixado"} if modo == "fixado" else set())
    especificacoes, motivos = [], []
    for l in linhas:
        if l["estado"] in acionaveis:
            alvo = l["fixado"] or l["minimo"]
            operador = "==" if l["fixado"] else ">="
            especificacoes.append(f"{l['pacote']}{operador}{alvo}")
            motivos.append(f"{l['pacote']}: {l['estado']} (instalado: {l['instalado']}, alvo: {operador}{alvo})")
    avisos = [f"{l['pacote']} {l['instalado']} difere da versão fixada dos experimentos ({l['fixado']}); "
              "resultados numéricos podem variar" for l in linhas
              if l["estado"] == "difere_do_fixado" and modo == "minimo"]
    carregados = modulos_carregados if modulos_carregados is not None else sys.modules
    ja_importados = [e.split("=")[0].split(">")[0].replace("-", "_") for e in especificacoes
                     if e.split("=")[0].split(">")[0].replace("-", "_") in carregados]
    return {"modo": modo, "pacotes": linhas, "instalar": especificacoes, "motivos": motivos, "avisos": avisos,
            "reiniciar": bool(ja_importados), "ja_importados": ja_importados,
            "comando": [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *especificacoes]
            if especificacoes else []}


def instalar(plano: Mapping[str, Any], *, executar: bool = False) -> dict[str, Any]:
    """Executa o `pip` do plano só com `executar=True`; sem isso é um dry-run.
    Nunca repassa credenciais: o pip roda com o ambiente do processo, sem token."""
    if not plano["comando"]:
        return {"executado": False, "motivo": "nada a instalar", "reiniciar": False}
    if not executar:
        return {"executado": False, "motivo": "dry-run (passe executar=True para instalar)",
                "comando": list(plano["comando"]), "reiniciar": plano["reiniciar"]}
    r = subprocess.run(plano["comando"], capture_output=True, text=True)
    if r.returncode != 0:
        raise ErroAmbiente("falha ao instalar as dependências:\n" + (r.stderr or r.stdout)[-2000:])
    return {"executado": True, "comando": list(plano["comando"]), "reiniciar": plano["reiniciar"]}


# --------------------------------------------------------------------- GPU

def gpus() -> list[dict[str, Any]]:
    """GPUs vistas por `nvidia-smi` (sem importar o torch); lista vazia se não houver."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return []
    saida = []
    for linha in r.stdout.strip().splitlines() if r.returncode == 0 else []:
        partes = [p.strip() for p in linha.split(",")]
        if len(partes) >= 3:
            saida.append({"nome": partes[0], "memoria_mib": int(partes[1]) if partes[1].isdigit() else None,
                          "driver": partes[2]})
    return saida


def classificar_gpu(nome: str | None) -> dict[str, str]:
    """`referencia` (T4), `permitida` (L4, A100) ou `nao_prevista`, com a mensagem de aviso."""
    if not nome:
        return {"classe": "sem_gpu", "aviso": "nenhuma GPU: o treino usaria a CPU (inviável para o BioBERTpt)"}
    for sigla in GPU_REFERENCIA:
        if re.search(rf"\b{sigla}\b", nome):
            return {"classe": "referencia", "aviso": ""}
    for sigla in GPU_PERMITIDAS:
        if re.search(rf"\b{sigla}\b", nome):
            return {"classe": "permitida", "aviso": f"GPU {nome}: a referência é a T4; resultados numéricos podem "
                                                    "variar, e o hardware fica registrado na execução"}
    return {"classe": "nao_prevista", "aviso": f"GPU {nome} não prevista (referência T4; permitidas L4 e A100): "
                                               "confirme antes de uma execução que será comparada"}


# ----------------------------------------------------------------- coleta

def coletar(*, incluir_torch: bool = True) -> dict[str, Any]:
    """O ambiente atual no formato gravado pelas sessões de treino, mais `gpus` e `colab`.
    Com `incluir_torch`, importa o torch (se instalado) para ler CUDA/cuDNN."""
    placas = gpus()
    registro: dict[str, Any] = {
        "python": sys.version.split()[0], "plataforma": platform.platform(),
        "versoes": {p: _versao(p) for p in PACOTES_REGISTRADOS},
        "cuda": None, "cudnn": None, "gpu": placas[0]["nome"] if placas else None,
        "gpus": placas, "colab": colab.em_colab()}
    if incluir_torch and registro["versoes"]["torch"]:
        try:
            import torch  # noqa: PLC0415

            registro["cuda"] = torch.version.cuda
            registro["cudnn"] = torch.backends.cudnn.version()
            if torch.cuda.is_available():
                registro["gpu"] = torch.cuda.get_device_name(0)
        except Exception as erro:  # noqa: BLE001
            registro["erro_torch"] = type(erro).__name__
    return registro


def verificar(codigo: str | Path, *, modo: str = "minimo", exigir_gpu: bool = False,
              instalado: Mapping[str, str | None] | None = None, ambiente_atual: Mapping[str, Any] | None = None
              ) -> dict[str, Any]:
    """Relatório do ambiente para a operação `ambiente` dos notebooks: `ok`,
    `erros` (impedem), `avisos`, o `plano` de instalação e a classe da GPU."""
    requisitos = requisitos_do_projeto(codigo)
    atual = dict(ambiente_atual) if ambiente_atual is not None else coletar(incluir_torch=False)
    plano = plano_instalacao(requisitos, modo=modo, instalado=instalado)
    erros, avisos = [], list(plano["avisos"])
    minimo_python = requisitos["python"]
    if minimo_python and _cmp(atual["python"], minimo_python) < 0:
        erros.append(f"Python {atual['python']} abaixo do mínimo do projeto ({minimo_python})")
    if plano["instalar"]:
        erros.append("dependências a instalar: " + "; ".join(plano["motivos"]))
    if plano["reiniciar"]:
        avisos.append("há pacotes a trocar que já estão importados nesta sessão: reinicie o ambiente depois de instalar")
    gpu = classificar_gpu(atual.get("gpu"))
    if gpu["classe"] == "sem_gpu" and exigir_gpu:
        erros.append(gpu["aviso"])
    elif gpu["aviso"]:
        avisos.append(gpu["aviso"])
    return {"ok": not erros, "erros": erros, "avisos": avisos, "plano": plano, "gpu": gpu, "ambiente": atual,
            "requisitos": {"python": minimo_python, "n_minimos": len(requisitos["minimos"]),
                           "n_fixados": len(requisitos["fixados"])}}


# ------------------------------------------------- compatibilidade (retomar)

def _minor(versao: str | None) -> str | None:
    return ".".join(versao.split(".")[:2]) if versao else None


def _diferencas(gravado: Mapping[str, Any], atual: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(divergências que bloqueiam a retomada, divergências só de aviso)."""
    bloqueios, avisos = [], []
    if _minor(gravado.get("python")) != _minor(atual.get("python")):
        bloqueios.append(f"Python {gravado.get('python')} → {atual.get('python')}")
    gv, av = gravado.get("versoes") or {}, atual.get("versoes") or {}
    for pacote in sorted(set(gv) | set(av)):
        if gv.get(pacote) != av.get(pacote):
            (bloqueios if pacote in BLOQUEIAM_RETOMADA else avisos).append(f"{pacote} {gv.get(pacote)} → {av.get(pacote)}")
    if gravado.get("gpu") != atual.get("gpu"):
        bloqueios.append(f"GPU {gravado.get('gpu')} → {atual.get('gpu')}")
    for campo in ("cuda", "cudnn"):
        if gravado.get(campo) != atual.get(campo):
            avisos.append(f"{campo} {gravado.get(campo)} → {atual.get(campo)}")
    return bloqueios, avisos


def comparar_para_retomada(gravado: Mapping[str, Any] | None, atual: Mapping[str, Any]) -> dict[str, Any]:
    """Compara o ambiente gravado (última sessão) com o atual antes de retomar.
    Sem ambiente gravado não há como conferir: bloqueia, não presume."""
    if not gravado:
        return {"pode_retomar": False, "bloqueios": ["a execução não tem ambiente gravado para comparar"], "avisos": []}
    bloqueios, avisos = _diferencas(gravado, atual)
    return {"pode_retomar": not bloqueios, "bloqueios": bloqueios, "avisos": avisos}


def comparar_para_reaproveitar(gravado: Mapping[str, Any] | None, atual: Mapping[str, Any]) -> dict[str, Any]:
    """Reaproveitar um estágio concluído é só inferência: nenhuma diferença de
    versão ou de GPU bloqueia; todas viram aviso registrado."""
    if not gravado:
        return {"avisos": ["a execução não tem ambiente gravado para comparar"]}
    bloqueios, avisos = _diferencas(gravado, atual)
    return {"avisos": [*(f"(variação numérica possível) {b}" for b in bloqueios), *avisos]}


def ambiente_da_ultima_sessao(treino: Mapping[str, Any] | None, experimento: Mapping[str, Any] | None = None
                              ) -> dict[str, Any] | None:
    """O ambiente da última sessão de `treino.json` (`sessoes[-1].ambiente`), ou o
    registrado em `experimento.json`, ou None."""
    sessoes = (treino or {}).get("sessoes") or []
    if sessoes and sessoes[-1].get("ambiente"):
        return sessoes[-1]["ambiente"]
    return (experimento or {}).get("ambiente")
