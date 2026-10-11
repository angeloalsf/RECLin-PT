"""Marca de execução de validação técnica (`execucao.validacao`) (A20, A21)."""
from __future__ import annotations

import pytest

from reclin.execucao import validacao
from reclin.util.io import gravar_json


def test_bloco_e_nome():
    assert validacao.bloco(1, 3) == {"epocas": 1, "epocas_protocolo": 3, "rotulo": "validação técnica, 1 época"}
    assert validacao.bloco(2, 10)["rotulo"] == "validação técnica, 2 épocas"
    assert validacao.nome("baseline", 1) == "baseline_validacao_1ep"
    assert validacao.nome("twostage_e2", 2) == "twostage_e2_validacao_2ep"
    for invalido in (0, -1, 1.5):
        with pytest.raises(ValueError):
            validacao.bloco(invalido, 3)


def test_marca_lida_do_config(tmp_path):
    gravar_json(tmp_path / "v" / "config.json", {"config": {"validacao": validacao.bloco(1, 3)}})
    gravar_json(tmp_path / "c" / "config.json", {"config": {"estrategia": "baseline"}})
    assert validacao.da_execucao(tmp_path / "v")["epocas"] == 1
    assert validacao.da_execucao(tmp_path / "c") is None
    assert validacao.da_execucao(tmp_path / "sem_config") is None


def test_recusas(tmp_path):
    gravar_json(tmp_path / "v" / "config.json", {"config": {"validacao": validacao.bloco(1, 3)}})
    gravar_json(tmp_path / "c" / "config.json", {"config": {}})
    with pytest.raises(validacao.ExecucaoDeValidacao, match="validação técnica"):
        validacao.recusar_execucao(tmp_path / "v", "avaliação do TEST")
    with pytest.raises(validacao.ExecucaoDeValidacao):
        validacao.recusar_sidecar(tmp_path / "v" / "predicoes_dev.json", "comparações")
    validacao.recusar_execucao(tmp_path / "c", "avaliação do TEST")
    validacao.recusar_sidecar(tmp_path / "solto.preds.json", "comparações")     # sidecar solto (legado)


def test_heranca_e_mistura():
    marca = validacao.bloco(1, 3)
    assert validacao.herdar({"a": None, "b": None}) is None
    herdada = validacao.herdar({"a": marca, "b": validacao.bloco(2, 10)})
    assert set(herdada["origens"]) == {"a", "b"}
    with pytest.raises(validacao.ExecucaoDeValidacao, match="misturar"):
        validacao.herdar({"a": marca, "b": None})
