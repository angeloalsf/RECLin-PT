"""O protocolo pré-registrado da fase (`docs/protocolo/fase-biobertpt-seed42.md`)
confere com o código (A18): épocas, chaves de seleção, desempate, remapeamento,
léxico, grade e regras dos limiares, ausência de parada antecipada e de portão
de pesquisa, e os nomes da rodada de validação técnica.

Se um critério mudar no código sem que o registro mude (ou o contrário), este
teste falha: o registro é o que foi aprovado antes de qualquer treino.
"""
from __future__ import annotations

import json
import re

import pytest

from reclin.avaliacao import limiar
from reclin.config import Config
from reclin.estrategias import baseline_calibrado, pair_aware, restrito, two_stage
from reclin.execucao import validacao
from reclin.negacao.lexico import CONGELADO
from reclin.util.caminhos import CODIGO

DOCUMENTO = CODIGO / "docs" / "protocolo" / "fase-biobertpt-seed42.md"


@pytest.fixture(scope="module")
def registro():
    texto = DOCUMENTO.read_text(encoding="utf-8")
    blocos = re.findall(r"```json\n(.*?)\n```", texto, flags=re.S)
    assert len(blocos) == 1, "o protocolo precisa ter exatamente um bloco JSON verificável"
    return json.loads(blocos[0])


def test_escopo(registro):
    assert registro["fase"] == "validação inicial, seed 42"
    assert (registro["encoder"], registro["seed"]) == (Config().encoder, Config().seed)
    assert registro["modelo"] == Config().modelo == "pucpr/biobertpt-all"
    assert registro["parada_antecipada"] is False and registro["portao_de_pesquisa"] is False


def test_criterios_das_estrategias_treinadas(registro):
    from reclin.treino import laco
    e = registro["estrategias"]
    assert e["baseline"] == {"epocas": Config().epochs, "campo_selecao": laco.CAMPO_SELECAO}
    for nome, classe, modulo in (("restrito", restrito.ConfigRestrito, restrito),
                                 ("pair_aware", pair_aware.ConfigPairAware, pair_aware)):
        assert e[nome]["epocas"] == classe().epochs and e[nome]["campo_selecao"] == laco.CAMPO_SELECAO
        reg = modulo.montagem(classe(), lexico={"nao": 1}).registro
        assert reg["selecao_melhor_epoca"] == e[nome]["selecao_melhor_epoca"]
        assert reg["remapeamento"] == e[nome]["remapeamento"]
    for estagio in (1, 2):
        assert e[f"two_stage_e{estagio}"] == {"epocas": two_stage.CONFIGS[estagio]().epochs,
                                              "campo_selecao": two_stage.CAMPO_SELECAO[estagio]}
    assert e["filtro_pistas"]["lexico"] == {k: CONGELADO[k] for k in ("min_freq", "n_formas", "lexico_sha1")}
    assert e["baseline_calibrado"]["regra"] == baseline_calibrado.REGRA


def test_desempate_das_epocas_e_o_do_laco(registro):
    from reclin.treino import laco
    assert registro["desempate_de_epocas"] == "primeira época (máximo estrito)"
    historico = [{"epoch": 1, "x": 0.5}, {"epoch": 2, "x": 0.7}, {"epoch": 3, "x": 0.7}]
    assert laco.melhor_epoca(historico, "x") == (2, 0.7)


def test_limiares(registro):
    from reclin.estrategias.two_stage import composicao
    lim = registro["limiar"]
    assert lim["grade"] == list(limiar.GRADE)
    assert lim["criterio"] == limiar.CRITERIO and lim["desempates"] == list(limiar.DESEMPATES)
    assert lim["regra_two_stage"] == composicao.REGRA
    assert lim["casas_roteamento"] == two_stage.CASAS_ROTEAMENTO == 4


def test_rodada_de_validacao(registro):
    v = registro["validacao_tecnica"]
    assert v["epocas"] == {"baseline": 1, "two_stage_e1": 1, "two_stage_e2": 2}
    assert v["nomes"]["baseline"] == validacao.nome("baseline", v["epocas"]["baseline"])
    assert v["nomes"]["two_stage_e1"] == validacao.nome(two_stage.PREFIXO_VALIDACAO[1], v["epocas"]["two_stage_e1"])
    assert v["nomes"]["two_stage_e2"] == validacao.nome(two_stage.PREFIXO_VALIDACAO[2], v["epocas"]["two_stage_e2"])
    assert v["nomes"]["two_stage"] == two_stage.NOME_COMPOSICAO_VALIDACAO
    assert v["nomes"]["baseline_calibrado"] == validacao.nome(baseline_calibrado.NOME, v["epocas"]["baseline"])
