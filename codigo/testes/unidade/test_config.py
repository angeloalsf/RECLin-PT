"""Configuração base (`reclin.config`)."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields

import pytest

from reclin import tarefa
from reclin.config import ENCODERS, Config


def test_defaults_e_checkpoint():
    c = Config()
    assert c.encoder == "biobertpt" and c.modelo == "pucpr/biobertpt-all"
    assert Config(encoder="bertimbau").modelo == "neuralmind/bert-base-portuguese-cased"
    assert c.max_gap == tarefa.MAX_GAP
    assert set(ENCODERS) == {"biobertpt", "bertimbau"}


def test_congelada():
    with pytest.raises(FrozenInstanceError):
        Config().lr = 1e-3


@pytest.mark.parametrize("mudanca", [
    {"encoder": "roberta"}, {"class_weight": "auto"}, {"epochs": 0},
    {"lr": 0.0}, {"max_gap": -1}, {"warmup_ratio": 1.0},
])
def test_valores_invalidos_falham(mudanca):
    with pytest.raises(ValueError):
        Config(**mudanca)


def test_estrategia_herda_e_muda_so_o_que_declara():
    @dataclass(frozen=True)
    class ConfigExemplo(Config):
        epochs: int = 10
        min_freq: int = 3

    base, exemplo = Config(), ConfigExemplo()
    diferentes = {f.name for f in fields(Config) if getattr(base, f.name) != getattr(exemplo, f.name)}
    assert diferentes == {"epochs"}
    assert exemplo.como_dict()["tipo"] == "ConfigExemplo"
    assert exemplo.como_dict()["min_freq"] == 3
    with pytest.raises(ValueError):
        ConfigExemplo(encoder="roberta")         # a validação da base continua valendo


def test_como_dict_registra_tudo():
    d = Config(seed=43).como_dict()
    assert d["tipo"] == "Config" and d["modelo"] == "pucpr/biobertpt-all" and d["seed"] == 43
    assert set(d) == {"tipo", "modelo"} | {f.name for f in fields(Config)}
