"""D9 — marcadores e offsets das entidades.

Testes sintéticos e determinísticos do comportamento especificado dos
marcadores (`entrada.janela_marcada`): eles envolvem exatamente
`texto[start:end]` de cada entidade, usando os offsets como estão, sem
nenhuma correção. E a confirmação, no TRAIN e no DEV congelados (nunca no
TEST), dos números da auditoria de alinhamento registrados na etapa 7."""
from __future__ import annotations

import random

import pytest

from reclin import entrada, particoes, tarefa
from reclin.analises import alinhamento
from reclin.entrada import E1_ABRE, E1_FECHA, E2_ABRE, E2_FECHA
from reclin.util.caminhos import PARTICOES_V1 as PARTICOES     # os números registrados são da v1


def _entre(janela: str, abre: str, fecha: str) -> str:
    return janela[janela.index(abre) + len(abre):janela.index(fecha)].strip()


def test_marcadores_envolvem_exatamente_os_offsets():
    rng = random.Random(3)
    palavras = ["nega", "febre", "dor", "tosse", "sem", "edema", "refere", "dispneia"]
    for _ in range(200):
        texto = " ".join(rng.choice(palavras) for _ in range(30))
        posicoes = [i for i, c in enumerate(texto) if c != " " and (i == 0 or texto[i - 1] == " ")]
        a, b = rng.sample(posicoes, 2)
        e1 = {"start": a, "end": texto.index(" ", a) if " " in texto[a:] else len(texto)}
        e2 = {"start": b, "end": texto.index(" ", b) if " " in texto[b:] else len(texto)}
        janela = entrada.janela_marcada(texto, e1, e2, ctx_chars=20)
        assert _entre(janela, E1_ABRE, E1_FECHA) == texto[e1["start"]:e1["end"]]
        assert _entre(janela, E2_ABRE, E2_FECHA) == texto[e2["start"]:e2["end"]]
        # o contexto vai até ctx_chars antes e depois do par, sem passar dos limites do texto
        limpo = janela
        for m in (E1_ABRE + " ", " " + E1_FECHA, E2_ABRE + " ", " " + E2_FECHA):
            limpo = limpo.replace(m, "")
        ini = max(0, min(a, b) - 20)
        assert limpo == texto[ini:min(len(texto), max(e1["end"], e2["end"]) + 20)]


def test_offsets_usados_como_estao():
    """Com offsets deslocados em relação ao texto da entidade, os marcadores
    envolvem o trecho dos offsets (não há realinhamento em tempo de execução)."""
    texto = "PACIENTE NEGA FEBRE HOJE"
    e1 = {"start": 10, "end": 14, "text": "NEGA"}          # deslocado: NEGA começa em 9
    e2 = {"start": 14, "end": 19, "text": "FEBRE"}
    janela = entrada.janela_marcada(texto, e1, e2)
    assert _entre(janela, E1_ABRE, E1_FECHA) == "EGA"
    assert alinhamento.alinhar_entidade(texto, e1) == {"status": "deslocado", "delta": -1}


def test_deriva_acumulada_quando_caracteres_somem_depois_dos_offsets():
    """Ilustração do mecanismo suspeito: offsets calculados num texto com
    quebras de linha de 2 caracteres, que depois viram 1 espaço. Cada quebra
    antes de uma entidade desloca-a de mais 1 (deriva monótona)."""
    original = "DOR ABDOMINAL.\r\nNEGA FEBRE.\r\nREFERE TOSSE."
    nomes = ["DOR", "NEGA", "FEBRE", "TOSSE"]
    entidades = [{"id": str(i), "start": original.index(n), "end": original.index(n) + len(n), "text": n,
                  "type": "x"} for i, n in enumerate(nomes)]
    doc = {"doc_id": "d", "text": original.replace("\r\n", " "), "entities": entidades, "relations": []}
    deltas = [alinhamento.alinhar_entidade(doc["text"], e)["delta"] for e in entidades]
    assert deltas == [None, -1, -1, -2]
    assert alinhamento.perfil_da_deriva([doc])["documentos"] == {"sem_deriva": 0, "monotona": 1, "nao_monotona": 0}


# Números registrados na etapa 7 (TRAIN e DEV congelados; o TEST não é lido)
ESPERADO = {
    "train": {"status": {"exato": 21479, "equivalente": 469, "deslocado": 13899, "ausente": 608},
              "afetados_negation_of": (672, 1255)},
    "dev": {"status": {"exato": 2970, "equivalente": 33, "deslocado": 1435, "ausente": 75},
            "afetados_negation_of": (58, 150)},
}


@pytest.mark.parametrize("particao", ["train", "dev"])
def test_numeros_da_auditoria_confirmados(particao):
    docs = particoes.ler_particao(PARTICOES, particao)
    r = alinhamento.auditar(docs)
    assert r["status"] == ESPERADO[particao]["status"]
    pares = alinhamento.impacto_nos_pares(docs)
    neg = pares["negation_of"]
    assert (neg["afetados"], neg["pares"]) == ESPERADO[particao]["afetados_negation_of"]
    assert sum(v["pares"] for v in pares.values()) == tarefa.TAMANHOS_ESPERADOS[particao]
    assert not any("\n" in d["text"] for d in docs)        # nenhum texto tem quebra de linha
