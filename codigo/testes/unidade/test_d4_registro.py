"""D4 — o `min_freq` registrado, contra o resultado gravado, o pré-registro, o
protocolo e o TRAIN real da v2 (nunca o DEV nem o TEST)."""
from __future__ import annotations

import json
import re

import pytest

from reclin import particoes, versao_dados
from reclin.negacao import lexico, selecao_min_freq as sel
from reclin.util import caminhos
from reclin.util.io import ler_json, sha256_arquivo

RESULTADO = caminhos.CODIGO / "docs" / "entregas" / "d4" / "selecao_min_freq.json"
PRE_REGISTRO = caminhos.CODIGO / "docs" / "protocolo" / "d4-min-freq.md"
PROTOCOLO = caminhos.CODIGO / "docs" / "protocolo" / "fase-biobertpt-seed42.md"


@pytest.fixture(scope="module")
def resultado():
    return ler_json(RESULTADO)


@pytest.fixture(scope="module")
def train_v2():
    pasta = caminhos.PARTICOES_V2
    assert particoes.conferir_particoes(pasta, ["train"]) == []
    return particoes.ler_particao(pasta, "train")


def test_resultado_registrado_e_o_congelado(resultado):
    final = resultado["lexico_final"]
    assert resultado["min_freq_escolhido"] == final["min_freq"] == lexico.CONGELADO["min_freq"] == 10
    assert final["n_formas"] == lexico.CONGELADO["n_formas"] and final["lexico_sha1"] == lexico.CONGELADO["lexico_sha1"]
    assert final["max_gap"] == lexico.CONGELADO["max_gap"]
    assert lexico.lexico_sha1(final["formas"]) == final["lexico_sha1"]


def test_resultado_segue_a_regra_pre_registrada(resultado):
    from fractions import Fraction
    assert resultado["candidatos"] == list(sel.CANDIDATOS) and resultado["n_partes"] == sel.N_PARTES
    assert resultado["semente"] == sel.SEMENTE and resultado["max_gap"] == sel.MAX_GAP_SELECAO
    f1 = {int(m): Fraction(*map(int, v["total"]["f1_fracao"].split("/"))) for m, v in resultado["resultados"].items()}
    contagens = {int(m): {k: v["total"][k] for k in ("vp", "fp", "fn")} for m, v in resultado["resultados"].items()}
    assert all(f1[m] == sel.f1(contagens[m]) for m in f1)
    assert sel.escolher(contagens) == resultado["min_freq_escolhido"]


def test_proveniencia(resultado):
    p = resultado["proveniencia"]
    assert p["pendencia"] == "D4" and p["versao_dados"] == versao_dados.V2 and p["modelos_ou_predicoes_usados"] is False
    assert p["train_sha256"] == versao_dados.REGISTRO[versao_dados.V2]["particoes"]["train"]
    assert p["manifest_sha256"] == versao_dados.REGISTRO[versao_dados.V2]["manifest_sha256"]
    assert p["particoes_presentes_na_pasta"] == ["train.jsonl"]            # DEV e TEST ausentes na execução
    assert p["pre_registro_sha256"] == sha256_arquivo(PRE_REGISTRO)       # o pré-registro não mudou depois


def test_protocolo_aponta_o_lexico_da_d4():
    texto = PROTOCOLO.read_text(encoding="utf-8")
    bloco = json.loads(re.findall(r"```json\n(.*?)\n```", texto, flags=re.S)[0])
    assert bloco["estrategias"]["filtro_pistas"]["lexico"] == \
        {k: lexico.CONGELADO[k] for k in ("min_freq", "n_formas", "lexico_sha1")}


def test_congelado_e_o_induzido_do_train_v2(train_v2):
    lex = lexico.carregar_congelado(train_v2)
    assert len(lex) == 8
    # o registro do legado continua válido para o TRAIN (reprodução técnica)
    assert len(lexico.carregar_congelado(train_v2, lexico.CONGELADO_LEGADO)) == 11


@pytest.mark.lento
def test_selecao_reproduzida_no_train_v2(train_v2, resultado):
    """Refaz o procedimento inteiro (cerca de 1 a 2 minutos) e compara com o gravado."""
    refeito = sel.selecionar(train_v2)
    gravado = {k: v for k, v in resultado.items() if k != "proveniencia"}
    assert json.loads(json.dumps(refeito, ensure_ascii=False, sort_keys=True)) == \
        json.loads(json.dumps(gravado, ensure_ascii=False, sort_keys=True))
