"""Sidecars, diretório de execução e trilha (`reclin.execucao`)."""
from __future__ import annotations

import json

import pytest

from reclin.execucao import diretorio, predicoes, trilha
from reclin.tarefa import LABELS
from reclin.util.io import sha256_json

Y_TRUE = [0, 1, 2, 2]
Y_PRED = [0, 2, 2, 2]


def conjunto(particao="test", y_true=Y_TRUE):
    return {"particao": particao, "max_gap": 25, "particao_sha256": "x",
            "n_candidatos": len(y_true), "candidatos_sha256": "y",
            "y_true_sha256": sha256_json(y_true)}


def test_montar_na_ordem_do_legado():
    p = predicoes.montar(model="m", seed=42, y_true=Y_TRUE, y_pred=Y_PRED,
                         probs=[[0.123456, 0.2, 0.676544]] * 4, extra={"split": "test"})
    assert list(p) == ["model", "seed", "labels", "y_true", "y_pred", "split", "probs"]
    assert p["labels"] == list(LABELS) and p["probs"][0] == [0.1235, 0.2, 0.6765]


@pytest.mark.parametrize("estrago,mensagem", [
    (lambda p: p.pop("y_pred"), "chaves"),
    (lambda p: p.update(labels=["a", "b", "c"]), "rótulos"),
    (lambda p: p.update(y_pred=[0, 1]), "elementos"),
    (lambda p: p.update(y_pred=[0, 1, 2, 3]), "fora"),
    (lambda p: p.update(probs=[[1.0, 0.0]] * 4), "probs"),
])
def test_validar_recusa_sidecar_malformado(estrago, mensagem):
    p = predicoes.montar(model="m", seed=None, y_true=Y_TRUE, y_pred=Y_PRED)
    estrago(p)
    with pytest.raises(ValueError, match=mensagem):
        predicoes.validar(p)


def test_gravar_e_ler(tmp_path):
    p = predicoes.montar(model="m", seed=None, y_true=Y_TRUE, y_pred=Y_PRED)
    predicoes.gravar(tmp_path / "x.preds.json", p)
    bruto = (tmp_path / "x.preds.json").read_bytes()
    assert bruto == json.dumps(p, ensure_ascii=False).encode("utf-8")      # compacto, sem \n final
    assert predicoes.ler(tmp_path / "x.preds.json") == p


def test_conferir_conjunto_e_mesmo_conjunto():
    p = predicoes.montar(model="m", seed=1, y_true=Y_TRUE, y_pred=Y_PRED)
    assert predicoes.conferir_conjunto(p, conjunto()) == []
    assert "y_true diferente" in predicoes.conferir_conjunto(p, conjunto(y_true=[0, 1, 2, 1]))[0]
    assert "predições para" in predicoes.conferir_conjunto(p, conjunto(y_true=[0, 1]))[0]
    outra = predicoes.montar(model="n", seed=2, y_true=Y_TRUE, y_pred=Y_TRUE)
    assert predicoes.mesmo_conjunto(p, outra)
    assert not predicoes.mesmo_conjunto(p, dict(outra, y_true=[0, 0, 2, 2]))


# --------------------------------------------------------------------------- #
def test_diretorio_de_execucao(tmp_path):
    pasta = diretorio.criar(tmp_path, "baseline_x_seed1", estrategia="baseline", config={"lr": 2e-5})
    p = predicoes.montar(model="m", seed=1, y_true=Y_TRUE, y_pred=Y_PRED)
    diretorio.gravar_predicoes(pasta, "test", p, conjunto())
    assert diretorio.ler_predicoes(pasta, "test") == p
    config = diretorio.ler_config(pasta)
    assert config["estrategia"] == "baseline" and config["conjuntos"]["test"]["n_candidatos"] == 4

    diretorio.gravar_metricas(pasta, "test", {"macro_f1": 0.5})
    diretorio.gravar_metricas(pasta, "dev", {"macro_f1": 0.4})
    assert diretorio.ler_metricas(pasta) == {"test": {"macro_f1": 0.5}, "dev": {"macro_f1": 0.4}}

    with pytest.raises(FileExistsError):
        diretorio.criar(tmp_path, "baseline_x_seed1", estrategia="baseline", config={})
    with pytest.raises(FileExistsError, match="sobrescrever"):
        diretorio.gravar_predicoes(pasta, "test", p, conjunto())
    with pytest.raises(ValueError, match="conjunto de dev"):
        diretorio.gravar_predicoes(pasta, "test", p, conjunto("dev"), sobrescrever=True)
    with pytest.raises(ValueError, match="y_true diferente"):
        diretorio.gravar_predicoes(pasta, "test", p, conjunto(y_true=[0, 1, 2, 1]), sobrescrever=True)
    with pytest.raises(ValueError, match="não é avaliada"):
        diretorio.arquivo_predicoes("train")


def test_caminho_predicoes_nos_dois_formatos(tmp_path):
    p = predicoes.montar(model="m", seed=1, y_true=Y_TRUE, y_pred=Y_PRED)
    pasta = diretorio.criar(tmp_path, "nova", estrategia="baseline", config={})
    diretorio.gravar_predicoes(pasta, "test", p, conjunto())
    predicoes.gravar(tmp_path / "antiga.preds.json", p)
    predicoes.gravar(tmp_path / "antiga.dev_preds.json", p)
    assert diretorio.caminho_predicoes(tmp_path, "nova", "test") == pasta / "predicoes_test.json"
    assert diretorio.caminho_predicoes(tmp_path, "antiga", "test") == tmp_path / "antiga.preds.json"
    assert diretorio.caminho_predicoes(tmp_path, "antiga", "dev") == tmp_path / "antiga.dev_preds.json"
    with pytest.raises(FileNotFoundError):
        diretorio.caminho_predicoes(tmp_path, "nova", "dev")


# --------------------------------------------------------------------------- #
def test_sha1_config_com_a_receita_do_legado():
    # configuração do baseline BioBERTpt semente 42, cujo hash está na trilha do legado
    cfg = {"model": "pucpr/biobertpt-all", "epochs": 3, "batch_size": 64, "max_length": 128,
           "max_gap": 25, "ctx_chars": 128, "lr": 2e-5, "seed": 42, "class_weight": "balanced",
           "weight_decay": 0.01, "warmup_ratio": 0.1, "splits_dir": "data/splits"}
    assert trilha.sha1_config(cfg) == "ae69a0267d65"


def test_trilha_conta_avaliacoes(tmp_path):
    caminho = tmp_path / "x" / "avaliacoes_test.jsonl"
    assert trilha.ler(caminho) == []
    kw = dict(execucao="e", model="m", seed=1, n_test=4, macro_f1=0.12345678, negation_of_f1=0.5)
    l1 = trilha.registrar(caminho, config_sha1="aaa", **kw)
    l2 = trilha.registrar(caminho, config_sha1="bbb", **kw)
    l3 = trilha.registrar(caminho, config_sha1="aaa", extra={"nota": "reavaliação"}, **kw)
    assert [l["eval_index"] for l in (l1, l2, l3)] == [1, 2, 3]
    assert [l["eval_index_for_config"] for l in (l1, l2, l3)] == [1, 1, 2]
    assert l1["test_macro_f1"] == 0.123457 and l3["nota"] == "reavaliação"
    assert trilha.ler(caminho) == [l1, l2, l3]
    assert trilha.avaliacoes_da_config(caminho, "aaa") == 2
    assert b"\r" not in caminho.read_bytes()
