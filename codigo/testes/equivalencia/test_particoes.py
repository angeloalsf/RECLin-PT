"""Portão da etapa 1: partições copiadas, MANIFEST regerado e algoritmo de partição.

* Os arquivos de `codigo/dados/particoes/` são os do legado (mesmo SHA-256).
* O MANIFEST regerado pelo código novo traz os mesmos hashes e contagens do
  antigo; mudam só o gerador e o caminho do dataset, que seguem o layout novo.
* `particionar`, aplicado à união dos 1.000 documentos, reproduz as três
  partições byte a byte. A ordem de entrada não importa (os documentos são
  ordenados por `doc_id` antes do sorteio), então a união das partições é uma
  entrada equivalente ao `dataset.jsonl` original, que não é versionado.
"""
from __future__ import annotations

import hashlib
import random
import subprocess
import sys

from reclin import particoes
from reclin.util import caminhos
from reclin.util.io import gravar_jsonl, linha_jsonl


def test_arquivos_iguais_aos_do_legado(manifesto, referencia_dados):
    legado = referencia_dados["manifesto_particoes"]
    for nome in particoes.PARTICOES:
        assert manifesto["splits"][nome]["sha256"] == legado["splits"][nome]["sha256"]
    assert particoes.conferir_particoes(caminhos.PARTICOES) == []


def test_manifesto_regerado_igual_ao_antigo(manifesto, referencia_dados):
    legado = referencia_dados["manifesto_particoes"]
    assert manifesto["splits"] == legado["splits"]
    assert manifesto["seed"] == legado["seed"] == particoes.SEED
    assert manifesto["source"]["sha256"] == legado["source"]["sha256"]
    # diferenças esperadas: o layout novo
    assert manifesto["generator"] == particoes.GERADOR
    assert manifesto["source"]["path"] == "dados/processados/dataset.jsonl"
    assert set(manifesto) == set(legado)


def test_manifesto_e_o_que_o_codigo_novo_calcula(manifesto):
    recalculado = particoes.calcular_manifesto(caminhos.PARTICOES, seed=manifesto["seed"],
                                               origem=manifesto["source"])
    assert recalculado == manifesto


def test_particionar_reproduz_as_particoes_congeladas(documentos, manifesto):
    todos = [doc for nome in particoes.PARTICOES for doc in documentos[nome]]
    assert len(todos) == 1000
    divisao = particoes.particionar(todos, seed=manifesto["seed"])
    for nome in particoes.PARTICOES:
        conteudo = "".join(linha_jsonl(doc) + "\n" for doc in divisao[nome]).encode("utf-8")
        assert hashlib.sha256(conteudo).hexdigest() == manifesto["splits"][nome]["sha256"], nome


def test_script_preparar_dados_de_ponta_a_ponta(documentos, manifesto, tmp_path):
    """`preparar_dados.py particionar` sobre um dataset embaralhado grava as
    partições congeladas; `conferir` aceita o resultado; e o script recusa
    sobrescrever partições existentes sem `--sobrescrever`."""
    todos = [doc for nome in particoes.PARTICOES for doc in documentos[nome]]
    random.Random(0).shuffle(todos)
    dataset = tmp_path / "dataset.jsonl"
    gravar_jsonl(dataset, todos)
    pasta = tmp_path / "particoes"
    script = caminhos.CODIGO / "scripts" / "preparar_dados.py"

    def rodar(*args):
        return subprocess.run([sys.executable, str(script), "--pasta", str(pasta),
                               "--dataset", str(dataset), *args],
                              capture_output=True, text=True).returncode

    assert rodar("particionar") == 0
    gerado = particoes.ler_manifesto(pasta)
    assert gerado["splits"] == manifesto["splits"]
    assert rodar("conferir") == 0
    assert rodar("particionar") == 2
