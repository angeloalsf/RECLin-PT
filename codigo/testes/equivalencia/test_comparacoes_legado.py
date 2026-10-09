"""Portão da etapa 2 (2/2): as 26 comparações de significância do legado.

Cada comparação do protocolo (`reclin.avaliacao.protocolo`) é recalculada a
partir dos sidecars do TEST do legado, com a semente do protocolo e 10.000
reamostras, e o relatório serializado é comparado BYTE A BYTE com o
`significance_*.json` gravado pelo legado. Nenhuma tolerância: qualquer
diferença em qualquer casa decimal falha o teste.

São os testes mais lentos da suíte (cerca de 4 s cada): `-m "not lento"` os
pula.
"""
from __future__ import annotations

import pytest

from reclin.avaliacao import protocolo, significancia
from reclin.execucao import diretorio, predicoes
from reclin.util.caminhos import CODIGO

RESULTADOS_LEGADO = CODIGO / "testes" / "referencia" / "resultados_legado"

COMPARACOES = protocolo.comparacoes_tcc()


def ler_test(nome: str) -> dict:
    return predicoes.ler(diretorio.caminho_predicoes(RESULTADOS_LEGADO, nome, "test"))


def test_protocolo_tem_as_26_comparacoes_do_legado():
    gravadas = sorted(p.name for p in RESULTADOS_LEGADO.glob("significance_*.json"))
    assert len(COMPARACOES) == 26 == len(gravadas)
    assert sorted(c.arquivo for c in COMPARACOES) == gravadas


@pytest.mark.parametrize("comparacao", COMPARACOES, ids=[c.arquivo for c in COMPARACOES])
def test_semente_do_protocolo_e_a_das_execucoes(comparacao):
    assert ler_test(comparacao.a)["seed"] == comparacao.seed_a
    assert ler_test(comparacao.b)["seed"] == comparacao.seed_b


@pytest.mark.lento
@pytest.mark.parametrize("comparacao", COMPARACOES, ids=[c.arquivo for c in COMPARACOES])
def test_comparacao_reproduzida_byte_a_byte(comparacao):
    relatorio = significancia.comparar(ler_test(comparacao.a), ler_test(comparacao.b),
                                       seed=comparacao.seed)
    gravado = (RESULTADOS_LEGADO / comparacao.arquivo).read_bytes()
    assert significancia.relatorio_json(relatorio).encode("utf-8") == gravado
