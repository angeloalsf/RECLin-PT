"""Pontos de entrada da etapa 7 (Makefile e notebook único) e a separação do
legado: nada da infraestrutura de análise e relatório lê resultados do legado,
e os passos da fase científica só rodam com autorização explícita."""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess

import pytest

from reclin.util import caminhos

CODIGO = caminhos.CODIGO
NOTEBOOK = CODIGO / "notebooks" / "reclin.ipynb"
MAKEFILE = CODIGO / "Makefile"
ETAPA7 = sorted([*(CODIGO / "reclin" / "analises").glob("*.py"), *(CODIGO / "reclin" / "relatorios").glob("*.py"),
                 CODIGO / "reclin" / "execucao" / "inventario.py", CODIGO / "reclin" / "util" / "formatacao.py",
                 CODIGO / "scripts" / "gerar_tcc.py", CODIGO / "scripts" / "conferir_tcc.py",
                 CODIGO / "scripts" / "analisar.py", MAKEFILE, NOTEBOOK])
# o que só a reprodução técnica do legado (testes de equivalência, etapas 2 a 6) usa
PROIBIDOS = ("resultados_legado", "caminho_predicoes", "SUFIXOS_LEGADO", "RESULTADOS_LEGADO", "protocolo.comparacoes_tcc",
             "modelo_minusculo", "referencias.json")
GIT = re.compile(r"(^|[\s;&|(\"'])git(\s|$)", re.M)


@pytest.mark.parametrize("arquivo", ETAPA7, ids=lambda p: p.relative_to(CODIGO).as_posix())
def test_sem_dependencia_do_legado_nem_de_git(arquivo):
    texto = arquivo.read_text(encoding="utf-8")
    assert not [p for p in PROIBIDOS if p in texto], arquivo
    if arquivo.name != "inventario.py":          # o inventário cita a pasta só para recusá-la
        assert "testes/referencia" not in texto and '"referencia"' not in texto
    assert not GIT.search(texto), "nenhum comando de controle de versão"


def _celulas():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["language"] == "python"
    return nb["cells"]


def test_notebook_e_valido_e_parametrizado():
    celulas = _celulas()
    codigo = [c for c in celulas if c["cell_type"] == "code"]
    for c in codigo:
        ast.parse("".join(c["source"]))
        assert c["outputs"] == [] and c["execution_count"] is None
    parametros = [c for c in codigo if "parameters" in c["metadata"].get("tags", [])]
    assert len(parametros) == 1
    ns: dict = {}
    exec("".join(parametros[0]["source"]), ns)
    assert (ns["ENCODER"], ns["SEED"]) == ("biobertpt", 42) and not any(ns["AUTORIZADO"].values())
    scripts = set(re.findall(r'rodar\("([a-z_]+\.py)"', "".join("".join(c["source"]) for c in codigo)))
    assert scripts and all((CODIGO / "scripts" / s).is_file() for s in scripts), scripts


def test_notebook_recusa_outro_escopo_e_passos_sem_autorizacao(monkeypatch, capsys):
    codigo = [c for c in _celulas() if c["cell_type"] == "code"]
    parametros, preparacao = "".join(codigo[0]["source"]), "".join(codigo[1]["source"])
    monkeypatch.chdir(CODIGO)
    ns: dict = {}
    exec(parametros, ns)
    exec(preparacao, ns)
    assert ns["CODIGO"] == CODIGO
    chamadas = []
    ns["rodar"] = lambda *a: chamadas.append(a)
    for celula in codigo[2:]:
        texto = "".join(celula["source"])
        if "autorizado(" in texto:
            exec(texto, ns)
    assert chamadas == [] and "não autorizado" in capsys.readouterr().out
    for encoder, seed in (("bertimbau", 42), ("biobertpt", 43)):
        ns2: dict = {}
        exec(parametros, ns2)
        ns2.update(ENCODER=encoder, SEED=seed)
        with pytest.raises(ValueError, match="fora do escopo"):
            exec(preparacao, ns2)


def _alvos():
    texto = MAKEFILE.read_text(encoding="utf-8")
    return texto, dict(re.findall(r"^([a-z0-9_ -]+):(.*)$", texto, flags=re.M))


def test_makefile_chama_scripts_existentes_e_protege_a_fase_cientifica():
    texto, alvos = _alvos()
    scripts = set(re.findall(r"\$\(SCRIPTS\)/([a-z_]+\.py)", texto))
    assert scripts and all((CODIGO / "scripts" / s).is_file() for s in scripts)
    cientificos = ["pretreino", "verificar-inicializacao", "validacao-1v", "baseline", "calibrado-dev", "restrito",
                   "pair-aware", "estagio1", "estagio2", "compor-dev", "test-baseline", "test-calibrado",
                   "test-restrito", "test-pair-aware", "test-two-stage", "filtro-dev", "test-filtro"]
    for alvo in cientificos:
        assert "_autorizado" in alvos[alvo].split(), alvo
    # nenhum alvo de análise ou relatório depende de um alvo da fase científica
    for alvo in ("analises", "relatorios", "eda resultados curvas", "inventario", "conferir-tcc", "testes"):
        assert not set(alvos[alvo].split()) & set(cientificos)


@pytest.mark.skipif(shutil.which("make") is None, reason="make não instalado")
def test_makefile_recusa_sem_autorizacao():
    for alvo in ("baseline", "test-baseline", "filtro-dev", "test-filtro"):
        r = subprocess.run(["make", "-s", "-C", str(CODIGO), alvo], capture_output=True, text=True)
        assert r.returncode != 0
    r = subprocess.run(["make", "-s", "-C", str(CODIGO), "baseline"], capture_output=True, text=True)
    assert "AUTORIZADO=1" in r.stdout
    r = subprocess.run(["make", "-n", "-C", str(CODIGO), "relatorios"], capture_output=True, text=True)
    assert r.returncode == 0 and "gerar_tcc.py tudo" in r.stdout
