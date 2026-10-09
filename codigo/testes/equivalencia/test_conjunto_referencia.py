"""Portão da etapa 1: o conjunto de referência do código novo é o do legado.

Sobre as partições versionadas em `codigo/dados/particoes/`, com `max_gap=25`:
mesmo número de documentos e de candidatos (152.686 / 19.064 / 19.210), mesma
contagem por rótulo e a mesma sequência de candidatos e de rótulos, elemento a
elemento. O `y_true` de referência do DEV e do TEST foi conferido, na geração,
contra os sidecars do baseline BioBERTpt semente 42 do legado.
"""
from __future__ import annotations

import pytest

from reclin import particoes, tarefa


@pytest.mark.parametrize("nome", particoes.PARTICOES)
def test_conjunto_de_referencia_igual_ao_do_legado(nome, documentos, manifesto, referencia_dados):
    ref = referencia_dados["particoes"][nome]
    assert referencia_dados["max_gap"] == tarefa.MAX_GAP
    assert referencia_dados["labels"] == list(tarefa.LABELS)

    docs = documentos[nome]
    assert len(docs) == ref["n_documentos"]

    conjunto = tarefa.conjunto_referencia(nome, docs, particao_sha256=manifesto["splits"][nome]["sha256"])
    assert conjunto["n_candidatos"] == ref["n_candidatos"] == tarefa.TAMANHOS_ESPERADOS[nome]
    assert conjunto["particao_sha256"] == ref["arquivo_sha256"]
    assert conjunto["candidatos_sha256"] == ref["candidatos_sha256"]
    assert conjunto["y_true_sha256"] == ref["y_true_sha256"]
    assert tarefa.contagem_por_rotulo(tarefa.candidatos(docs)) == ref["contagem_por_rotulo"]


def test_y_true_do_dev_e_do_test_veio_dos_sidecars(referencia_dados):
    """Garante que a referência usada acima é a dos sidecars oficiais."""
    for nome in ("dev", "test"):
        assert referencia_dados["particoes"][nome]["y_true_confere_com"].startswith(
            "results/baseline_biobertpt_seed42")
