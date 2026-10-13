"""Notebook `reclin_pair_aware.ipynb` (Pair-Aware da D10: todos os pares, sem MLP, sem léxico) e o ajuste de
`operacoes.argumentos_de_treino` para campos opcionais.

Estáticos ou com diretórios temporários: nada de rede, GPU, pesos, corpus, treino, TEST nem Git. Escritos junto
do notebook e **não executados** pelo autor da etapa (a suíte só roda por autorização)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from reclin.experimento import operacoes as ops
from reclin.util import caminhos

NOTEBOOK = caminhos.CODIGO / "notebooks" / "reclin_pair_aware.ipynb"


def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _celulas() -> dict[str, str]:
    return {c["id"]: "".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"}


def _codigo() -> str:
    return "\n".join(_celulas().values())


# ------------------------------------------------------------------ notebook
def test_notebook_valido_compila_sem_saidas_e_com_ids_unicos():
    nb = _nb()
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse("\n".join("pass" if linha.lstrip().startswith("!") else linha
                                for linha in "".join(c["source"]).splitlines()))


def test_notebook_usa_a_pair_aware_do_protocolo_sem_lexico_nem_mlp():
    c5a, c7 = _celulas()["c5a"], _celulas()["c7"]
    assert "pair_aware.NOME" in c5a and "pair_aware.ConfigPairAware" in c5a and "script='treinar.py'" in c5a
    assert "lexico" not in c5a.replace("sem léxico", "") and "carregar_congelado" not in c5a
    assert "'epochs': 3" in c7 and "'batch_size': 64" in c7 and "'head_dropout': None" in c7
    assert "mlp_hidden" not in _codigo() and "NOVA_EXECUCAO = False" in c7
    assert "reclin-pt-pairaware-" in _celulas()["c6"]


def test_notebook_tem_metricas_por_classe_e_comparacao_no_dev():
    celulas = _celulas()
    assert "comparacao_dev" in celulas["c8b"] and "comparacao_dev" in celulas["c8c"]
    assert "predicoes_dev.json" in celulas["c8b"] and "matriz_confusao" in celulas["c8b"]
    assert "'test'" not in celulas["c8c"] and "predicoes_test" not in celulas["c8c"]


def test_notebook_nao_avalia_nem_le_o_test_e_nao_expoe_segredos():
    codigo = _codigo()
    for proibido in ("avaliar_test", "predicoes_test", "avaliacoes_test", "--particao test", "OP_TEST"):
        assert proibido not in codigo, proibido
    assert not re.search(r"(ghp_|github_pat_|hf_[A-Za-z0-9]{20})", codigo)
    assert [k for k, v in _celulas().items() if "GITHUB_PAT" in v] == ["c1", "c10"]
    assert "os.environ[" not in codigo


# ------------------------------------------------------------------ operações
def test_argumentos_omitem_campos_opcionais_none_e_levam_os_demais(tmp_path):
    pair_aware = pytest.importorskip("reclin.estrategias.pair_aware")
    espec = ops.Especificacao(estrategia=pair_aware.NOME, script="treinar.py", config_cls=pair_aware.ConfigPairAware)
    ctx = ops.Contexto(codigo=Path("/c"), raiz_local=tmp_path, raiz_drive=None, espec=espec,
                       pasta_pesos=Path("/p"), pasta_particoes=Path("/d"))
    args = ops.argumentos_de_treino(ctx, ops.config_da_operacao(espec, {}), "n", retomar=False)
    assert args[:2] == ["--estrategia", "pair_aware"]
    assert "--head-dropout" not in args and "--mlp-hidden" not in args and "None" not in args
    assert args[args.index("--epochs") + 1] == "3" and args[args.index("--batch-size") + 1] == "64"
    outros = ops.argumentos_de_treino(ctx, ops.config_da_operacao(espec, {"head_dropout": 0.2}), "n", retomar=False)
    assert outros[outros.index("--head-dropout") + 1] == "0.2"
    assert espec.alias_oficial == "pairaware_biobertpt_seed42"
    assert espec.config_protocolo().encoder == "biobertpt" and espec.config_protocolo().seed == 42
