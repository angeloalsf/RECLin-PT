"""Utilitários (`reclin.util`)."""
from __future__ import annotations

import hashlib
import logging

from reclin.util import caminhos
from reclin.util.io import (gravar_json, gravar_jsonl, ler_json, ler_jsonl, linha_jsonl,
                            sha256_arquivo, sha256_json)
from reclin.util.log import configurar


def test_jsonl_ida_e_volta_com_acentos_e_lf(tmp_path):
    registros = [{"b": "não", "a": 1}, {"texto": "febre\ttosse"}]
    caminho = tmp_path / "x.jsonl"
    assert gravar_jsonl(caminho, registros) == 2
    assert list(ler_jsonl(caminho)) == registros
    bruto = caminho.read_bytes()
    assert b"\r" not in bruto and "não".encode() in bruto
    assert bruto.splitlines()[0] == b'{"a":1,"b":"n\xc3\xa3o"}'      # chaves ordenadas, compacto
    assert linha_jsonl(registros[0]) == '{"a":1,"b":"não"}'


def test_ler_jsonl_ignora_linhas_em_branco(tmp_path):
    caminho = tmp_path / "x.jsonl"
    caminho.write_text('{"a":1}\n\n  \n{"a":2}\n', encoding="utf-8")
    assert list(ler_jsonl(caminho)) == [{"a": 1}, {"a": 2}]


def test_gravar_json_estavel_e_atomico(tmp_path):
    caminho = tmp_path / "sub" / "x.json"
    gravar_json(caminho, {"b": [1, 2], "a": "ç"})
    primeira = caminho.read_bytes()
    gravar_json(caminho, {"a": "ç", "b": [1, 2]})
    assert caminho.read_bytes() == primeira
    assert primeira.endswith(b"}\n") and b"\r" not in primeira
    assert ler_json(caminho) == {"a": "ç", "b": [1, 2]}
    assert [p.name for p in caminho.parent.iterdir()] == ["x.json"]   # nenhum .tmp sobrando


def test_hashes(tmp_path):
    caminho = tmp_path / "x.bin"
    caminho.write_bytes(b"abc")
    assert sha256_arquivo(caminho) == hashlib.sha256(b"abc").hexdigest()
    assert sha256_json([1, "não"]) == hashlib.sha256('[1,"não"]'.encode()).hexdigest()
    # a ordem faz parte da identidade
    assert sha256_json({"a": 1, "b": 2}) != sha256_json({"b": 2, "a": 1})


def test_configurar_log_nao_duplica_handlers(tmp_path):
    arquivo = tmp_path / "logs" / "x.log"
    configurar()
    logger = configurar(arquivo)
    assert len(logger.handlers) == 2
    logging.getLogger("reclin.teste").info("mensagem de teste")
    for h in logger.handlers:
        h.flush()
    assert "mensagem de teste" in arquivo.read_text(encoding="utf-8")
    configurar()
    assert len(logging.getLogger("reclin").handlers) == 1


def test_caminhos():
    assert (caminhos.CODIGO / "reclin" / "__init__.py").exists()
    assert caminhos.PARTICOES == caminhos.CODIGO / "dados" / "particoes"
    assert caminhos.relativo_ao_codigo(caminhos.DATASET) == "dados/processados/dataset.jsonl"

