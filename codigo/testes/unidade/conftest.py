"""Fixtures das unidades que não dependem de nenhum artefato do legado.

`particoes_sinteticas` cria partições inteiramente sintéticas (documentos
inventados, com MANIFEST próprio): os testes de inventário e de relatórios as
usam para que nada, nem o TEST real, seja lido. `fabrica_execucoes` grava
execuções no formato do código novo (`execucao.diretorio`) com predições
sintéticas sobre essas partições.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from reclin import particoes, tarefa
from reclin.avaliacao import metricas
from reclin.execucao import diretorio, predicoes, trilha
from reclin.util import identidade_codigo
from reclin.util.io import gravar_json

PALAVRAS = ["paciente", "nega", "febre", "tosse", "dor", "sem", "refere", "dispneia", "edema", "hoje"]


def documento(doc_id: str, rng: random.Random) -> dict:
    """Um texto com 8 palavras-entidade e algumas relações entre vizinhas."""
    texto, entidades = "", []
    for i in range(8):
        palavra = rng.choice(PALAVRAS)
        inicio = len(texto)
        texto += palavra + " "
        entidades.append({"id": f"{doc_id}-{i}", "start": inicio, "end": inicio + len(palavra), "text": palavra,
                          "type": "Negation" if palavra in ("nega", "sem") else "Sign"})
    relacoes = []
    for i in range(7):
        sorteio = rng.random()
        if sorteio < 0.25:
            relacoes.append({"e1_id": entidades[i]["id"], "e2_id": entidades[i + 1]["id"], "type": "negation_of"})
        elif sorteio < 0.5:
            relacoes.append({"e1_id": entidades[i]["id"], "e2_id": entidades[i + 1]["id"],
                             "type": "associated_with"})
    return {"doc_id": doc_id, "text": texto.strip(), "entities": entidades, "relations": relacoes}


@pytest.fixture(scope="session")
def particoes_sinteticas(tmp_path_factory):
    rng = random.Random(7)
    pasta = tmp_path_factory.mktemp("particoes_sinteticas")
    docs = {nome: [documento(f"{nome}{i}", rng) for i in range(n)]
            for nome, n in (("train", 6), ("dev", 4), ("test", 4))}
    particoes.gravar_particoes(docs, pasta)
    particoes.gravar_manifesto(pasta, particoes.calcular_manifesto(
        pasta, seed=42, origem={"path": "sintético (testes)", "sha256": None}))
    return pasta


@pytest.fixture
def fabrica_execucoes(particoes_sinteticas, tmp_path):
    """`criar(nome, estrategia, registro, *, particoes=("dev",), avaliar_test=True, semente=0, extras=None)`
    grava uma execução com predições aleatórias (reprodutíveis) sobre as
    partições sintéticas; com `avaliar_test`, grava a trilha do TEST."""
    raiz = tmp_path / "execucoes"
    manifesto = particoes.ler_manifesto(particoes_sinteticas)

    def criar(nome, estrategia, registro, *, particoes_=("dev", "test"), avaliar_test=True, semente=0,
              extras=None, metricas_=True):
        pasta = diretorio.criar(raiz, nome, estrategia=estrategia, config=registro)
        if isinstance(registro.get("config"), dict):          # execução treinada: o treino.json mínimo
            gravar_json(pasta / "treino.json", {"concluido": True, "dev_history": [], "melhor_epoca": None,
                                                "sessoes": [{"codigo": identidade_codigo.atual()["identidade"]}]})
        rng = random.Random(semente)
        for p in particoes_:
            docs = particoes.ler_particao(particoes_sinteticas, p)
            y = tarefa.y_true(tarefa.candidatos(docs))
            y_pred = [yt if rng.random() < 0.7 else rng.randrange(3) for yt in y]
            sidecar = predicoes.montar(model=nome, seed=42, y_true=y, y_pred=y_pred,
                                       extra={"split": p, **(extras or {}).get(p, {})})
            diretorio.gravar_predicoes(pasta, p, sidecar, tarefa.conjunto_referencia(
                p, docs, particao_sha256=manifesto["splits"][p]["sha256"], conferir_tamanho=False))
            if metricas_:
                diretorio.gravar_metricas(pasta, p, metricas.avaliar(y, y_pred))
            if p == "test" and avaliar_test:
                m = metricas.avaliar(y, y_pred)
                trilha.registrar(pasta / diretorio.TRILHA, execucao=nome, model=nome, seed=42, config_sha1="x",
                                 n_test=len(y), macro_f1=m["macro_f1"],
                                 negation_of_f1=m["f1_per_class"]["negation_of"])
        return pasta

    criar.raiz = raiz
    return criar


def registro_treinado(estrategia: str, *, seed: int = 42, encoder: str = "biobertpt", proveniencia=True,
                      validacao=None) -> dict:
    r = {"estrategia": estrategia, "config": {"seed": seed, "encoder": encoder}, "modelo": "x",
         "particoes": {}, "estrategia_config": {}}
    if proveniencia:
        r["proveniencia_pesos"] = {"repositorio": "pucpr/biobertpt-all", "commit": "a" * 40, "hash_manifesto": "b"}
    if validacao:
        r["validacao"] = validacao
    return r


def gravar_treino(pasta, historico, melhor, campo=None, concluido=True):
    conteudo = {"dev_history": historico, "melhor_epoca": melhor, "concluido": concluido}
    if campo:
        conteudo["campo_selecao"] = campo
    gravar_json(pasta / "treino.json", conteudo)


@pytest.fixture
def ferramentas():
    """As funções auxiliares acima, para os testes (o modo de importação do
    pytest não deixa importar o conftest diretamente)."""
    return SimpleNamespace(registro_treinado=registro_treinado, gravar_treino=gravar_treino, documento=documento)
