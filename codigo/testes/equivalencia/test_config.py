"""Portão da etapa 1: os defaults de `Config` são os valores dos experimentos.

Confere contra a configuração registrada nos quatro baselines do legado:

* o `config` gravado no `.json` de cada baseline (batch, janela, épocas, lr...);
* o `config_sha1` da trilha `test_evals`, que o legado calculava sobre um
  dicionário que também inclui `weight_decay`, `warmup_ratio` e `splits_dir`.
  Recalcular o hash com a receita do legado a partir de `Config` confere esses
  três valores, que não aparecem em `config`.

`max_grad_norm` não aparece em nenhum registro: era fixo em 1,0 no laço de treino.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from reclin.config import Config


def _encoder_e_seed(nome: str) -> tuple[str, int]:
    _, encoder, seed = nome.split("_")
    return encoder, int(seed.removeprefix("seed"))


def _config_sha1_legado(c: Config) -> str:
    """Receita de `relation_extraction.append_test_eval_log` no legado."""
    cfg = {
        "model": c.modelo, "epochs": c.epochs, "batch_size": c.batch_size,
        "max_length": c.max_length, "max_gap": c.max_gap, "ctx_chars": c.ctx_chars,
        "lr": c.lr, "seed": c.seed,
        "class_weight": c.class_weight, "weight_decay": c.weight_decay,
        "warmup_ratio": c.warmup_ratio,
        "splits_dir": "data/splits",   # caminho das partições no legado
    }
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _baselines(referencia_dados):
    return sorted(referencia_dados["config_baselines"].items())


def test_ha_quatro_baselines(referencia_dados):
    assert len(_baselines(referencia_dados)) == 4


@pytest.mark.parametrize("indice", range(4))
def test_defaults_iguais_ao_config_gravado(indice, referencia_dados):
    nome, registro = _baselines(referencia_dados)[indice]
    encoder, seed = _encoder_e_seed(nome)
    c = Config(encoder=encoder, seed=seed)
    assert c.modelo == registro["model"]
    assert c.seed == registro["seed"]
    for campo, valor in registro["config"].items():
        assert getattr(c, campo) == valor, campo


@pytest.mark.parametrize("indice", range(4))
def test_config_sha1_do_legado(indice, referencia_dados):
    nome, registro = _baselines(referencia_dados)[indice]
    encoder, seed = _encoder_e_seed(nome)
    assert _config_sha1_legado(Config(encoder=encoder, seed=seed)) == registro["config_sha1"]
