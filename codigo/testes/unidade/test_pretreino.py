"""Proveniência dos pesos (`reclin.pretreino`): commit exato, manifesto,
conferência antes da carga, registro e checagem entre execuções (A17)."""
from __future__ import annotations

import os
import shutil
import sys

import pytest

from reclin import pretreino
from reclin.util.caminhos import CODIGO
from reclin.util.io import gravar_json, ler_json

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def pasta(tmp_path):
    destino = tmp_path / "pre"
    shutil.copytree(MODELO_MINUSCULO, destino)
    pretreino.gravar(destino, repositorio="teste/minusculo", commit=COMMIT)
    return destino


@pytest.mark.parametrize("revisao", ["main", "v1.0", "0123456", COMMIT.upper(), COMMIT + "0", "", None])
def test_revisao_recusa_branch_tag_e_sha_incompleto(revisao):
    with pytest.raises(pretreino.ErroProveniencia, match="SHA completo"):
        pretreino.validar_commit(revisao)


def test_manifesto_tem_commit_hash_e_tamanho_de_cada_arquivo(pasta):
    manifesto = ler_json(pasta / pretreino.ARQUIVO)
    assert manifesto["commit"] == COMMIT and manifesto["repositorio"] == "teste/minusculo"
    nomes = sorted(p.name for p in MODELO_MINUSCULO.iterdir())
    assert sorted(manifesto["arquivos"]) == nomes
    for nome, info in manifesto["arquivos"].items():
        assert info["tamanho"] == (pasta / nome).stat().st_size and len(info["sha256"]) == 64
    assert pretreino.conferir(pasta) == {"repositorio": "teste/minusculo", "commit": COMMIT,
                                         "hash_manifesto": manifesto["hash_manifesto"]}


def test_ocultos_e_o_proprio_manifesto_ficam_fora(pasta):
    (pasta / ".cache").mkdir()
    (pasta / ".cache" / "x").write_text("cache do download")
    assert pretreino.conferir(pasta)["commit"] == COMMIT


@pytest.mark.parametrize("alteracao", ["byte", "removido", "novo", "manifesto_editado"])
def test_conferencia_recusa_qualquer_divergencia(pasta, alteracao):
    if alteracao == "byte":
        with open(pasta / "model.safetensors", "r+b") as f:
            f.seek(-1, os.SEEK_END)
            ultimo = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([ultimo[0] ^ 1]))
    elif alteracao == "removido":
        (pasta / "tokenizer.json").unlink()
    elif alteracao == "novo":
        (pasta / "extra.bin").write_bytes(b"x")
    else:
        manifesto = ler_json(pasta / pretreino.ARQUIVO)
        manifesto["commit"] = "f" * 40
        gravar_json(pasta / pretreino.ARQUIVO, manifesto)
    with pytest.raises(pretreino.ErroProveniencia):
        pretreino.conferir(pasta)


def test_sem_manifesto_nada_muda():
    assert pretreino.registro_de(None) is None
    assert pretreino.registro_de(MODELO_MINUSCULO) is None          # pasta sem manifesto
    assert pretreino.registro_de("pucpr/biobertpt-all") is None     # identificador do Hub
    assert not pretreino.tem_manifesto(MODELO_MINUSCULO)


def test_mesma_proveniencia(pasta):
    r = pretreino.conferir(pasta)
    assert pretreino.mesma_proveniencia({"a": r, "b": dict(r)}) == r
    with pytest.raises(pretreino.ErroProveniencia, match="sem proveniência"):
        pretreino.mesma_proveniencia({"a": r, "b": None})
    with pytest.raises(pretreino.ErroProveniencia, match="mesma revisão"):
        pretreino.mesma_proveniencia({"a": r, "b": {**r, "commit": "f" * 40}})
    with pytest.raises(pretreino.ErroProveniencia, match="mesma revisão"):
        pretreino.mesma_proveniencia({"a": r, "b": {**r, "hash_manifesto": "0" * 64}})


def test_baixar_usa_o_commit_exato_e_grava_o_manifesto(tmp_path):
    chamadas = []

    def falso(*, repo_id, revision, local_dir):
        chamadas.append((repo_id, revision))
        shutil.copytree(MODELO_MINUSCULO, local_dir, dirs_exist_ok=True)

    manifesto = pretreino.baixar("pucpr/biobertpt-all", COMMIT, tmp_path / "d", baixador=falso)
    assert chamadas == [("pucpr/biobertpt-all", COMMIT)]
    assert pretreino.conferir(tmp_path / "d")["hash_manifesto"] == manifesto["hash_manifesto"]
    with pytest.raises(pretreino.ErroProveniencia, match="não está vazio"):
        pretreino.baixar("pucpr/biobertpt-all", COMMIT, tmp_path / "d", baixador=falso)


def test_baixar_recusa_branch_antes_de_acessar_a_rede(tmp_path):
    def nunca(**_):
        raise AssertionError("não deveria acessar a rede")
    with pytest.raises(pretreino.ErroProveniencia):
        pretreino.baixar("pucpr/biobertpt-all", "main", tmp_path / "d", baixador=nunca)
    assert not (tmp_path / "d").exists()


def test_ativar_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    constantes = sys.modules.get("huggingface_hub.constants")
    anterior = getattr(constantes, "HF_HUB_OFFLINE", None) if constantes else None
    try:
        pretreino.ativar_offline()
        assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
        if constantes is not None:
            assert constantes.HF_HUB_OFFLINE is True
    finally:
        if constantes is not None:
            constantes.HF_HUB_OFFLINE = anterior


def test_hash_pasta_muda_com_o_conteudo(tmp_path):
    shutil.copytree(MODELO_MINUSCULO, tmp_path / "a")
    antes = pretreino.hash_pasta(tmp_path / "a")
    assert antes == pretreino.hash_pasta(MODELO_MINUSCULO)
    (tmp_path / "a" / "config.json").write_text("{}")
    assert pretreino.hash_pasta(tmp_path / "a") != antes
