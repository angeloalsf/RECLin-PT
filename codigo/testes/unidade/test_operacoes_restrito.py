"""Notebook `reclin_restrito.ipynb`.

Estáticos: nada de rede, GPU, pesos, corpus, treino, TEST nem Git. Escritos junto do notebook e **não executados**
pelo autor da etapa (a suíte só roda por autorização)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from reclin.experimento import operacoes as ops
from reclin.util import caminhos

NOTEBOOK = caminhos.CODIGO / "notebooks" / "reclin_restrito.ipynb"


def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _celulas() -> dict[str, str]:
    return {c["id"]: "".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"}


def test_notebook_valido_compila_sem_saidas_e_com_ids_unicos():
    nb = _nb()
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse("\n".join("pass" if linha.lstrip().startswith("!") else linha
                                for linha in "".join(c["source"]).splitlines()))


def test_notebook_usa_o_restrito_com_as_epocas_do_protocolo():
    c5a, c7 = _celulas()["c5a"], _celulas()["c7"]
    assert "restrito.NOME" in c5a and "restrito.ConfigRestrito" in c5a and "script='treinar.py'" in c5a
    assert "carregar_congelado" in c5a and "ler_particao(pasta, 'train')" in c5a
    assert "'epochs': 2" in c7 and "reclin-pt-restrito-" in _celulas()["c6"]


def test_notebook_nao_avalia_nem_le_o_test_e_nao_expoe_segredos():
    codigo = "\n".join(_celulas().values())
    for proibido in ("avaliar_test", "predicoes_test", "avaliacoes_test", "--particao test", "OP_TEST"):
        assert proibido not in codigo, proibido
    assert not re.search(r"(ghp_|github_pat_|hf_[A-Za-z0-9]{20})", codigo)
    assert [k for k, v in _celulas().items() if "GITHUB_PAT" in v] == ["c1", "c10"]


def test_especificacao_do_restrito_e_o_nome_oficial(tmp_path):
    restrito = pytest.importorskip("reclin.estrategias.restrito")
    espec = ops.Especificacao(estrategia=restrito.NOME, script="treinar.py", config_cls=restrito.ConfigRestrito)
    assert espec.alias_oficial == "restrito_biobertpt_seed42" and espec.config_protocolo().epochs == 2
    ctx = ops.Contexto(codigo=Path("/c"), raiz_local=tmp_path, raiz_drive=None, espec=espec,
                       pasta_pesos=Path("/p"), pasta_particoes=Path("/d"))
    args = ops.argumentos_de_treino(ctx, ops.config_da_operacao(espec, {}), "n", retomar=False)
    assert args[:2] == ["--estrategia", "restrito"] and args[args.index("--epochs") + 1] == "2"
