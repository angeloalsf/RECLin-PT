"""Critérios de seleção e de parada do restrito e da Pair-Aware (D10: todos os pares), verificados no
código novo por si mesmos (revisão da etapa 7), sem nenhum artefato do legado.

Com um BERT sintético criado no teste (`modelo_sintetico`) e as primeiras
30/10 linhas do TRAIN e do DEV congelados (a pasta não tem o TEST):

* o treino roda todas as épocas configuradas (sem parada antecipada);
* a época escolhida é o primeiro máximo estrito de `dev_macro_f1`;
* `dev_macro_f1` é o macro-F1 do DEV COMPLETO com as predições remapeadas
  (no restrito, fora do espaço restrito: `no_relation`; na Pair-Aware, nada é
  remapeado: todos os pares são pontuados), recalculado aqui a partir do
  sidecar gravado, e não o macro-F1 do DEV restrito;
* só TRAIN e DEV são lidos; o TEST não existe na pasta.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from reclin import particoes, tarefa  # noqa: E402
from reclin.avaliacao import metricas  # noqa: E402
from reclin.estrategias import pair_aware, restrito  # noqa: E402
from reclin.execucao import predicoes  # noqa: E402
from reclin.negacao import lexico as modulo_lexico  # noqa: E402
from reclin.tarefa import NOREL  # noqa: E402
from reclin.treino import classificador, laco  # noqa: E402
from reclin.util.caminhos import PARTICOES  # noqa: E402

EPOCAS = 3
PEQUENA = {"epochs": EPOCAS, "batch_size": 4, "max_length": 64, "lr": 1e-3, "seed": 42}
LOTE = {"restrito": 4, "pair_aware": 256}      # a Pair-Aware treina nos 5.310 pares do TRAIN pequeno
MODULOS = {"restrito": (restrito, restrito.ConfigRestrito), "pair_aware": (pair_aware, pair_aware.ConfigPairAware)}


@pytest.fixture(scope="module")
def lexico():
    return modulo_lexico.carregar_congelado(particoes.ler_particao(PARTICOES, "train"))


@pytest.fixture(scope="module")
def treinadas(tmp_path_factory, subconjunto, lexico, modelo_sintetico):
    pasta_particoes = subconjunto(30, 10, 10, sem_test=True)
    raiz = tmp_path_factory.mktemp("criterios")
    lidas = {}
    original = particoes.ler_particao
    for qual, (modulo, classe) in MODULOS.items():
        config = classe(**{**PEQUENA, "batch_size": LOTE[qual]})
        lidas[qual] = []

        def ler(pasta, nome, _lista=lidas[qual]):
            _lista.append(nome)
            return original(pasta, nome)
        particoes.ler_particao = ler
        try:
            classificador.treinar_execucao(raiz, qual, config, modelo=modelo_sintetico,
                                           pasta_particoes=pasta_particoes,
                                           montagem=modulo.montagem(config, lexico=lexico) if qual == "restrito"
                                           else modulo.montagem(config))
        finally:
            particoes.ler_particao = original
    return raiz, pasta_particoes, lidas


def test_padroes_do_protocolo():
    assert restrito.ConfigRestrito().epochs == 2          # teto da fase atual (D5)
    assert pair_aware.ConfigPairAware().epochs == 3       # D10: o do baseline
    assert laco.CAMPO_SELECAO == "dev_macro_f1"


@pytest.mark.parametrize("qual", list(MODULOS))
def test_todas_as_epocas_sem_parada_antecipada(qual, treinadas):
    raiz, _, _ = treinadas
    treino = json.loads((raiz / qual / "treino.json").read_text("utf-8"))
    assert treino["concluido"] and treino["config"]["epochs"] == EPOCAS
    assert [h["epoch"] for h in treino["dev_history"]] == list(range(1, EPOCAS + 1))


@pytest.mark.parametrize("qual", list(MODULOS))
def test_epoca_pelo_primeiro_maximo_do_dev_remapeado(qual, treinadas, lexico):
    raiz, pasta_particoes, _ = treinadas
    treino = json.loads((raiz / qual / "treino.json").read_text("utf-8"))
    historico = treino["dev_history"]
    valores = [h["dev_macro_f1"] for h in historico]
    primeira_max = valores.index(max(valores)) + 1
    assert treino["melhor_epoca"] == primeira_max and treino["melhor_dev_macro_f1"] == max(valores)

    # o valor da época escolhida é o macro-F1 do DEV completo, recalculado do sidecar remapeado
    sidecar = predicoes.ler(raiz / qual / "predicoes_dev.json")
    cands = tarefa.candidatos(particoes.ler_particao(pasta_particoes, "dev"))
    completo = metricas.avaliar(sidecar["y_true"], sidecar["y_pred"])
    assert completo["macro_f1"] == historico[primeira_max - 1]["dev_macro_f1"]
    assert completo["f1_per_class"]["negation_of"] == historico[primeira_max - 1]["dev_negation_of_f1"]
    if qual == "pair_aware":       # todos os pares, sem remapeamento
        assert sidecar["y_true"] == tarefa.y_true(cands) and "restricted_indices" not in sidecar
        return
    indices = restrito.selecionar(cands, lexico)
    assert sidecar["y_true"] == tarefa.y_true(cands) and len(indices) < len(cands)
    fora = set(range(len(cands))) - set(indices)
    assert all(sidecar["y_pred"][i] == NOREL for i in fora)
    # e não o do espaço restrito, que fica no histórico só como informação
    restrito_m = metricas.avaliar([sidecar["y_true"][i] for i in indices], [sidecar["y_pred"][i] for i in indices])
    assert restrito_m["macro_f1"] == historico[primeira_max - 1]["dev_restrito_macro_f1"]
    assert completo["f1_per_class"]["negation_of"] == historico[primeira_max - 1]["dev_negation_of_f1"]


@pytest.mark.parametrize("qual", list(MODULOS))
def test_so_train_e_dev_sao_lidos(qual, treinadas):
    raiz, pasta_particoes, lidas = treinadas
    assert not (pasta_particoes / "test.jsonl").exists()
    assert lidas[qual] == ["train", "dev"]
    assert not (raiz / qual / "predicoes_test.json").exists()
    config = json.loads((raiz / qual / "config.json").read_text("utf-8"))
    assert set(config["conjuntos"]) == {"dev"}
    assert config["config"]["estrategia_config"]["selecao_melhor_epoca"] == \
        ("dev_macro_f1 remapeado ao espaco completo" if qual == "restrito" else "dev_macro_f1 no DEV completo")
