"""Portão da etapa 3: o léxico de pistas do código novo é o do legado.

Sobre as partições versionadas, com `max_gap=25`, comparado com IGUALDADE
EXATA (formas, contagens, hashes, frações em ponto flutuante) contra:

* `referencia/dados.json`, gerado pelo próprio código do legado
  (`count_cue_forms`, `induce_lexicon`, `carregar_lexico` com a guarda e
  `construir_espaco`);
* os resultados do legado em `referencia/resultados_legado/`: a calibração do
  filtro que congelou o léxico e o bloco `restricted_space` das cinco
  execuções do fine-tuning restrito e da Pair-Aware;
* o número citado no TCC (cobertura de 0,9467 no DEV, com 4 casas).
"""
from __future__ import annotations

import pytest

from reclin import particoes, tarefa
from reclin.negacao import lexico
from reclin.util.io import sha256_json

MIN_FREQS = (1, 2, 3, 5, 10)
TAMANHOS = {1: 51, 2: 17, 3: 11, 5: 9, 10: 8}
FORMAS_CONGELADAS = {"sem": 662, "nega": 353, "nao": 128, "ausente": 12, "ausencia": 11,
                     "s": 11, "evacuacao": 10, "s/": 10, "ausentes": 5,
                     "eliminacao fecal": 4, "indolor": 3}
EXECUCOES_COM_LEXICO = ([f"finetuning_restrito/restrito_biobertpt_seed{s}" for s in range(42, 47)]
                        + ["pair_aware/pairaware_biobertpt_seed42"])


@pytest.fixture(scope="module")
def congelado(documentos):
    return lexico.carregar_congelado(documentos["train"])


def test_contagem_das_formas_no_train(documentos, referencia_dados):
    contagem = lexico.contar_formas(documentos["train"])
    assert dict(contagem) == referencia_dados["lexico"]["contagem_formas_train"]
    assert len(contagem) == 51 and sum(contagem.values()) == 1255


@pytest.mark.parametrize("min_freq", MIN_FREQS)
def test_lexico_por_min_freq(min_freq, documentos, referencia_dados):
    lex = lexico.induzir(documentos["train"], min_freq=min_freq)
    assert sorted(lex) == referencia_dados["lexico"]["por_min_freq"][str(min_freq)]
    assert len(lex) == TAMANHOS[min_freq]


def test_as_11_formas_congeladas(congelado, referencia_dados):
    ref = referencia_dados["lexico"]["congelado"]
    assert congelado == FORMAS_CONGELADAS == ref["formas"]
    assert list(congelado) == list(FORMAS_CONGELADAS)        # ordem: frequência, depois forma
    assert lexico.lexico_sha1(congelado) == ref["lexico_sha1"] == "70c93fa807de"
    assert ref["min_freq"] == lexico.CONGELADO["min_freq"] == 3
    assert sum(congelado.values()) == 1209


def test_registro_confere_com_a_calibracao_do_legado(legado):
    calibracao = legado("CALIBRACAO_filtro.json")
    assert calibracao["min_freq"] == lexico.CONGELADO["min_freq"]
    assert calibracao["lexicon_size"] == lexico.CONGELADO["n_formas"]
    assert calibracao["combined_gap"] == lexico.CONGELADO["max_gap"] == tarefa.MAX_GAP


@pytest.mark.parametrize("execucao", EXECUCOES_COM_LEXICO)
def test_mesmo_lexico_das_execucoes_do_legado(execucao, congelado, legado):
    registro = legado(execucao + ".json")
    gravado = registro["restricted_space"]["lexico"]
    assert gravado["formas"] == congelado
    assert gravado["sha1"] == lexico.lexico_sha1(congelado)
    assert gravado["n_formas"] == len(congelado)
    assert gravado["min_freq"] == lexico.CONGELADO["min_freq"]
    assert gravado["max_gap"] == lexico.CONGELADO["max_gap"]
    assert registro["config"]["lexicon_sha1"] == lexico.lexico_sha1(congelado)


@pytest.mark.parametrize("particao", particoes.PARTICOES)
def test_cobertura_por_particao(particao, congelado, documentos, referencia_dados, legado):
    c = lexico.cobertura(documentos[particao], congelado)
    resumo = referencia_dados["espaco_restrito"][particao]["resumo"]
    assert c["negation_of"] == resumo["contagens_completo"]["negation_of"]
    assert c["cobertos"] == resumo["contagens_restrito"]["negation_of"]
    assert c["cobertura"] == resumo["teto_recall_negation_of"]
    # o mesmo valor gravado nas execuções do restrito
    por_split = legado(EXECUCOES_COM_LEXICO[0] + ".json")["restricted_space"]["por_split"][particao]
    assert c["cobertura"] == por_split["teto_recall_negation_of"]


def test_cobertura_do_dev_citada_no_tcc(congelado, documentos):
    c = lexico.cobertura(documentos["dev"], congelado)
    assert (c["cobertos"], c["negation_of"]) == (142, 150)
    assert c["cobertura"] == 142 / 150
    assert round(c["cobertura"], 4) == 0.9467


@pytest.mark.parametrize("particao", particoes.PARTICOES)
def test_e_pista_seleciona_os_mesmos_candidatos(particao, congelado, documentos, referencia_dados):
    """Aplicado a todos os candidatos (190.960 nas três partições), `e_pista`
    escolhe exatamente os mesmos pares que o legado, na mesma ordem."""
    cands = tarefa.candidatos(documentos[particao])
    indices = [i for i, c in enumerate(cands) if lexico.e_pista(c["e1"], congelado)]
    ref = referencia_dados["espaco_restrito"][particao]
    assert len(indices) == ref["n_restrito"]
    assert sha256_json(indices) == ref["indices_sha256"]
    contagem = tarefa.contagem_por_rotulo(cands[i] for i in indices)
    assert contagem == ref["resumo"]["contagens_restrito"]


# --------------------------------------------------------------------------- #
# Só o TRAIN entra no léxico, e o resultado não depende da ordem              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("acrescimo", [("dev",), ("test",), ("dev", "test")])
def test_dev_ou_test_mudariam_o_lexico(acrescimo, documentos):
    """O valor de referência só sai do TRAIN: acrescentar o DEV ou o TEST à
    indução muda as formas e o hash. Logo, reproduzir 11 formas e
    `70c93fa807de` é evidência de que nenhum dos dois foi usado."""
    docs = list(documentos["train"])
    for nome in acrescimo:
        docs += documentos[nome]
    lex = lexico.induzir(docs, min_freq=3)
    assert len(lex) != 11 and lexico.lexico_sha1(lex) != "70c93fa807de"
    with pytest.raises(ValueError):
        lexico.carregar_congelado(docs)


def test_mudar_dev_e_test_nao_altera_o_lexico(documentos):
    """A indução recebe só os documentos do TRAIN: esvaziar o DEV e o TEST não
    muda nada."""
    original = lexico.carregar_congelado(documentos["train"])
    sem_dev_test = dict(documentos, dev=[], test=[])
    assert lexico.carregar_congelado(sem_dev_test["train"]) == original


def test_lexico_independe_da_ordem_dos_documentos(documentos):
    import random
    docs = list(documentos["train"])
    random.Random(0).shuffle(docs)
    lex = lexico.induzir(docs, min_freq=3)
    assert lex == FORMAS_CONGELADAS and list(lex) == list(FORMAS_CONGELADAS)
    assert lexico.lexico_sha1(lex) == "70c93fa807de"
