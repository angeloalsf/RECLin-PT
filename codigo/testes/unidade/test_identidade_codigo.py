"""Identidade do código que produz as execuções (D6), sem Git.

Determinística, sensível só aos arquivos que podem mudar o que uma execução
grava, registrada em cada execução e conferida pelo inventário e pelas
operações que acrescentam artefatos a uma execução existente."""
from __future__ import annotations

import json
import re

import pytest

from reclin.execucao import diretorio, inventario as inv
from reclin.util import caminhos, identidade_codigo as ic
from reclin.util.io import gravar_json, ler_json


@pytest.fixture
def arvore(tmp_path):
    """Uma cópia mínima de `codigo/` com arquivos incluídos e excluídos."""
    arquivos = {
        "pyproject.toml": "[project]\nname='x'\n",
        "reclin/__init__.py": "",
        "reclin/tarefa.py": "X = 1\n",
        "reclin/treino/laco.py": "def f():\n    return 1\n",
        "scripts/treinar.py": "print('t')\n",
        # fora da identidade
        "reclin/relatorios/tabelas.py": "R = 1\n",
        "reclin/analises/corpus.py": "A = 1\n",
        "reclin/execucao/inventario.py": "I = 1\n",
        "reclin/util/formatacao.py": "F = 1\n",
        "scripts/gerar_tcc.py": "print('g')\n",
        "scripts/analisar.py": "print('a')\n",
        "scripts/conferir_tcc.py": "print('c')\n",
        "testes/unidade/test_x.py": "def test(): pass\n",
        "docs/protocolo/p.md": "# p\n",
        "Makefile": "x:\n\techo\n",
        "notebooks/n.ipynb": "{}\n",
        "resultados/execucoes/e/config.json": "{}\n",
        "resultados/relatorios/cientifica/tabelas/t.tex": "% t\n",
        "reclin/__pycache__/tarefa.cpython-313.pyc": "lixo",
        "checkpoints/ultimo.pt": "pesos",
    }
    for relativo, conteudo in arquivos.items():
        (tmp_path / relativo).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relativo).write_bytes(conteudo.encode("utf-8"))
    return tmp_path


def test_arquivos_incluidos_e_determinismo(arvore):
    assert ic.arquivos(arvore) == ["pyproject.toml", "reclin/__init__.py", "reclin/tarefa.py",
                                   "reclin/treino/laco.py", "scripts/treinar.py"]
    a, b = ic.calcular(arvore), ic.calcular(arvore)
    assert a == b and a["algoritmo"] == ic.ALGORITMO and a["n_arquivos"] == 5
    assert re.fullmatch(r"[0-9a-f]{64}", a["identidade"]) and ic.conferir_registro(a) == []
    assert a["identidade"] == ic.identidade_de(a["arquivos"])


@pytest.mark.parametrize("relativo", ["reclin/tarefa.py", "scripts/treinar.py", "pyproject.toml"])
def test_alterar_um_arquivo_incluido_muda_a_identidade(arvore, relativo):
    antes = ic.calcular(arvore)
    (arvore / relativo).write_text((arvore / relativo).read_text() + "# mudança\n")
    depois = ic.calcular(arvore)
    assert depois["identidade"] != antes["identidade"] and ic.diferencas(antes, depois) == [relativo]


def test_arquivo_novo_ou_removido_muda_a_identidade(arvore):
    antes = ic.calcular(arvore)
    (arvore / "reclin" / "novo.py").write_text("Y = 2\n")
    assert ic.diferencas(antes, ic.calcular(arvore)) == ["reclin/novo.py"]
    (arvore / "reclin" / "novo.py").unlink()
    (arvore / "reclin" / "tarefa.py").unlink()
    assert ic.diferencas(antes, ic.calcular(arvore)) == ["reclin/tarefa.py"]


@pytest.mark.parametrize("relativo", ["reclin/relatorios/tabelas.py", "reclin/analises/corpus.py",
                                      "reclin/execucao/inventario.py", "reclin/util/formatacao.py",
                                      "scripts/gerar_tcc.py", "scripts/analisar.py", "scripts/conferir_tcc.py",
                                      "testes/unidade/test_x.py", "docs/protocolo/p.md", "Makefile",
                                      "notebooks/n.ipynb", "resultados/execucoes/e/config.json",
                                      "resultados/relatorios/cientifica/tabelas/t.tex",
                                      "reclin/__pycache__/tarefa.cpython-313.pyc", "checkpoints/ultimo.pt"])
def test_arquivos_que_nao_influenciam(arvore, relativo):
    antes = ic.calcular(arvore)["identidade"]
    (arvore / relativo).write_bytes((arvore / relativo).read_bytes() + b"\nmudou\n")
    assert ic.calcular(arvore)["identidade"] == antes


def test_fim_de_linha_nao_muda_a_identidade(arvore):
    antes = ic.calcular(arvore)["identidade"]
    caminho = arvore / "reclin" / "treino" / "laco.py"
    caminho.write_bytes(caminho.read_bytes().replace(b"\n", b"\r\n"))
    assert ic.calcular(arvore)["identidade"] == antes


def test_registro_incompleto_ou_adulterado(arvore):
    registro = ic.calcular(arvore)
    assert ic.conferir_registro(None) == ["sem identidade do código"]
    assert ic.conferir_registro({}) == ["sem identidade do código"]
    sem_manifesto = {k: v for k, v in registro.items() if k != "arquivos"}
    assert any("não confere" in p for p in ic.conferir_registro(sem_manifesto))
    adulterado = {**registro, "arquivos": {**registro["arquivos"], "reclin/tarefa.py": "0" * 64}}
    assert any("não confere" in p for p in ic.conferir_registro(adulterado))
    assert any("algoritmo" in p for p in ic.conferir_registro({**registro, "algoritmo": "outro"}))


def test_codigo_instalado():
    atual = ic.atual()
    assert atual == ic.calcular(caminhos.CODIGO) and ic.conferir_registro(atual) == []
    assert "reclin/treino/laco.py" in atual["arquivos"] and "scripts/treinar.py" in atual["arquivos"]
    assert not any(r.startswith(("reclin/relatorios/", "reclin/analises/", "testes/")) for r in atual["arquivos"])


def test_protocolo_registra_o_algoritmo():
    texto = (caminhos.CODIGO / "docs" / "protocolo" / "fase-biobertpt-seed42.md").read_text("utf-8")
    registro = json.loads(re.findall(r"```json\n(.*?)\n```", texto, flags=re.S)[0])
    assert registro["identidade_codigo"]["algoritmo"] == ic.ALGORITMO


# --------------------------------------------------------------------------- #
# Registro nas execuções e conferência                                        #
# --------------------------------------------------------------------------- #
def test_execucao_grava_a_identidade_e_e_conferida(tmp_path):
    pasta = diretorio.criar(tmp_path, "e", estrategia="baseline", config={"x": 1})
    assert diretorio.ler_config(pasta)["codigo"] == ic.atual()
    assert diretorio.conferir_codigo(pasta, "teste") == ic.atual()["identidade"]


def _trocar_codigo(pasta, registro):
    """Simula uma execução inteira de outra versão do código (ou anterior à
    D6): o config.json e as sessões do treino, de forma consistente."""
    conteudo = ler_json(pasta / "config.json")
    if registro is None:
        conteudo.pop("codigo")
    else:
        conteudo["codigo"] = registro
    gravar_json(pasta / "config.json", conteudo)
    if (pasta / "treino.json").is_file():
        treino = ler_json(pasta / "treino.json")
        treino["sessoes"] = [{} if registro is None else {"codigo": registro["identidade"]}]
        gravar_json(pasta / "treino.json", treino)


def test_execucao_de_outra_versao_ou_sem_identidade_e_recusada(tmp_path, arvore):
    pasta = diretorio.criar(tmp_path, "e", estrategia="baseline", config={})
    _trocar_codigo(pasta, ic.calcular(arvore))                  # outra versão, registro consistente
    with pytest.raises(ic.ErroCodigo, match="outra versão do código"):
        diretorio.conferir_codigo(pasta, "a avaliação do TEST")
    _trocar_codigo(pasta, {**ic.atual(), "identidade": "0" * 64})   # adulterado
    with pytest.raises(ic.ErroCodigo, match="não confere"):
        diretorio.conferir_codigo(pasta, "a avaliação do TEST")
    _trocar_codigo(pasta, None)                                 # anterior à D6
    with pytest.raises(ic.ErroCodigo, match="anterior à D6"):
        diretorio.conferir_codigo(pasta, "a retomada do treino")


def test_inventario_classifica_e_nao_mistura_versoes(fabrica_execucoes, particoes_sinteticas, ferramentas, arvore):
    fabrica_execucoes("atual", "baseline", ferramentas.registro_treinado("baseline"))
    outra = fabrica_execucoes("outra", "baseline", ferramentas.registro_treinado("baseline"), avaliar_test=False)
    _trocar_codigo(outra, ic.calcular(arvore))
    antiga = fabrica_execucoes("antiga", "baseline", ferramentas.registro_treinado("baseline"), avaliar_test=False)
    _trocar_codigo(antiga, None)
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    assert i.codigo_referencia == ic.atual()["identidade"]
    assert i.obter("atual").codigo == i.codigo_referencia and i.obter("atual").valida
    assert any("outra versão do código" in m for m in i.obter("outra").motivos_fora_da_fase)
    assert any("anterior à D6" in m for m in i.obter("antiga").motivos_fora_da_fase)
    assert all(e.natureza == "fora_da_fase" for e in i.execucoes.values())      # partições sintéticas
    with pytest.raises(inv.ErroInventario, match="versões diferentes do código"):
        i.selecionar(["atual", "outra"], natureza="fora_da_fase")
    # com a identidade da outra versão como referência, é a "atual" que fica fora
    i2 = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas,
                         codigo_referencia=ic.calcular(arvore)["identidade"])
    assert any("outra versão do código" in m for m in i2.obter("atual").motivos_fora_da_fase)
    assert not any("outra versão do código" in m for m in i2.obter("outra").motivos_fora_da_fase)


def test_inventario_confere_consistencia_interna(fabrica_execucoes, particoes_sinteticas, ferramentas, arvore):
    pasta = fabrica_execucoes("t", "baseline", ferramentas.registro_treinado("baseline"))
    linha = json.loads((pasta / diretorio.TRILHA).read_text("utf-8"))
    (pasta / diretorio.TRILHA).write_text(json.dumps({**linha, "codigo": "0" * 64}) + "\n", "utf-8")
    ferramentas.gravar_treino(pasta, [], None)
    treino = ler_json(pasta / "treino.json")
    gravar_json(pasta / "treino.json", {**treino, "sessoes": [{"codigo": ic.atual()["identidade"]}, {}]})
    adulterada = fabrica_execucoes("a", "baseline", ferramentas.registro_treinado("baseline"))
    _trocar_codigo(adulterada, {**ic.atual(), "identidade": ic.calcular(arvore)["identidade"]})
    base = fabrica_execucoes("base", "baseline", ferramentas.registro_treinado("baseline"))
    derivada = fabrica_execucoes("cal", "baseline_calibrado", {"base": {"nome": "base"}, "seed": 42})
    _trocar_codigo(base, ic.calcular(arvore))
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    problemas = i.obter("t").problemas
    assert any("trilha do TEST" in p for p in problemas) and any("sessões" in p for p in problemas)
    assert any("identidade do código: a identidade gravada não confere" in p for p in i.obter("a").problemas)
    assert any("identidade do código diferente da de base" in p for p in i.obter("cal").problemas)
    assert derivada.exists()
