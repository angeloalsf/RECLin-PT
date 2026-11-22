"""Configuração final de ponta a ponta, em CPU, com o modelo minúsculo e partições sintéticas:

treino da Pair-Aware sem pesos → reempacotamento (o código atual reproduz o DEV) →
limiar em P(negation_of) escolhido no DEV (maior precisão com piso de recall) →
congelamento (com os pesos da base) → trava do TEST: a base primeiro, depois a
derivada, uma vez cada, só com a liberação.

As partições sintéticas fazem o papel das oficiais (o "TEST científico" é
simulado com `monkeypatch`); nenhum arquivo do projeto é lido como dado.
"""
from __future__ import annotations

import json
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")

from reclin.estrategias import pair_aware, pair_aware_limiar  # noqa: E402
from reclin.execucao import congelamento as cg  # noqa: E402
from reclin.execucao import diretorio, predicoes  # noqa: E402
from reclin.treino import classificador, reempacotamento  # noqa: E402
from reclin.util import caminhos  # noqa: E402
from reclin.util.caminhos import CODIGO  # noqa: E402
from reclin.util.io import ler_json  # noqa: E402

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"
CONFIG = pair_aware.ConfigPairAware(epochs=1, batch_size=8, max_length=64, lr=1e-3, seed=42, class_weight="none")
PISO = Fraction(1, 10)          # o modelo minúsculo prevê pouco: um piso baixo garante um ponto elegível


@pytest.fixture(scope="module")
def base_treinada(tmp_path_factory, particoes_sinteticas):
    raiz = tmp_path_factory.mktemp("execucoes")
    montagem = pair_aware.montagem(CONFIG)
    registro = classificador.treinar_execucao(raiz, "pair_aware_sem_pesos_t", CONFIG, modelo=MODELO_MINUSCULO,
                                              pasta_particoes=particoes_sinteticas, montagem=montagem)
    assert registro["concluido"] and registro["pesos_classe"] is None
    return raiz, raiz / "pair_aware_sem_pesos_t", montagem


def test_reempacotamento_reproduz_o_dev_e_copia_os_pesos(base_treinada, particoes_sinteticas):
    raiz, base, montagem = base_treinada
    antes = {p.name: p.read_bytes() for p in base.iterdir() if p.is_file()}
    r = reempacotamento.reempacotar(base, raiz, "pair_aware_reempacotada", pasta_particoes=particoes_sinteticas,
                                    montagem=montagem)
    nova = r["pasta"]
    assert r["comparacao"]["equivalente"] and r["comparacao"]["classes_divergentes"] == 0
    assert {p.name: p.read_bytes() for p in base.iterdir() if p.is_file()} == antes          # base intacta
    cfg_base, cfg_nova = diretorio.ler_config(base), diretorio.ler_config(nova)
    assert cfg_nova["config"] == cfg_base["config"]                                           # mesma configuração
    assert cfg_nova["conjuntos"]["dev"] == cfg_base["conjuntos"]["dev"]
    assert cfg_nova["reempacotamento"]["base"] == base.name
    treino = ler_json(nova / "treino.json")
    assert treino["concluido"] and not treino["test_avaliado"] and treino["reempacotamento"]["comparacao_dev"]["equivalente"]
    assert predicoes.ler(nova / "predicoes_dev.json")["y_pred"] == predicoes.ler(base / "predicoes_dev.json")["y_pred"]
    with pytest.raises(reempacotamento.ErroReempacotamento, match="já existe"):
        reempacotamento.reempacotar(base, raiz, "pair_aware_reempacotada", pasta_particoes=particoes_sinteticas,
                                    montagem=montagem)


def test_reempacotamento_recusa_quando_o_dev_nao_bate(base_treinada, particoes_sinteticas, tmp_path):
    raiz, base, montagem = base_treinada
    import shutil
    copia = tmp_path / "execucoes" / base.name
    shutil.copytree(base, copia)
    sidecar = predicoes.ler(copia / "predicoes_dev.json")
    sidecar["y_pred"][0] = (sidecar["y_pred"][0] + 1) % 3
    (copia / "predicoes_dev.json").write_text(json.dumps(sidecar), "utf-8")
    with pytest.raises(reempacotamento.ErroReempacotamento, match="não reproduz"):
        reempacotamento.reempacotar(copia, copia.parent, "nao_deve_existir", pasta_particoes=particoes_sinteticas,
                                    montagem=montagem)
    assert not (copia.parent / "nao_deve_existir").exists()


def test_comparar_sidecars():
    a = {"y_true": [0, 1, 2], "y_pred": [0, 1, 2], "probs": [[0.9, 0.05, 0.05]] * 3}
    assert reempacotamento.comparar_sidecars(a, {**a, "probs": [[0.9004, 0.05, 0.0496]] * 3})["equivalente"]
    assert not reempacotamento.comparar_sidecars(a, {**a, "probs": [[0.89, 0.06, 0.05]] * 3})["equivalente"]
    assert not reempacotamento.comparar_sidecars(a, {**a, "y_pred": [0, 2, 2]})["equivalente"]
    assert not reempacotamento.comparar_sidecars(a, {**a, "y_true": [0, 1, 1]})["equivalente"]


def test_limiar_congelamento_e_trava_do_test(base_treinada, particoes_sinteticas, monkeypatch, tmp_path):
    raiz, base, montagem = base_treinada
    reemp = raiz / "pair_aware_reempacotada"
    if not reemp.exists():
        reempacotamento.reempacotar(base, raiz, reemp.name, pasta_particoes=particoes_sinteticas, montagem=montagem)
    nome = pair_aware_limiar.nome_execucao(reemp.name)
    r = pair_aware_limiar.calibrar_dev(reemp, raiz, nome, piso_recall=PISO)
    der = raiz / nome
    gravado = ler_json(der / "limiar.json")
    assert gravado["limiar"] == r["t"] and gravado["configuracao_busca"]["criterio_alternativo"]["piso_recall"] == "1/10"
    linha = next(x for x in r["tabela"] if x["limiar"] == r["t"])
    c = linha["contagens"]["negation_of"]
    assert Fraction(c["tp"], c["tp"] + c["fn"]) >= PISO
    assert predicoes.ler(der / "predicoes_dev.json")["y_pred"] == pair_aware_limiar.prever(
        predicoes.ler(reemp / "predicoes_dev.json")["probs"], r["t"])
    with pytest.raises(pair_aware_limiar.ErroLimiarPairAware, match="ao lado da base"):
        pair_aware_limiar.calibrar_dev(reemp, tmp_path, "outro", piso_recall=PISO)

    # o que vai para a branch de resultados (só DEV), antes do TEST
    from reclin.experimento import versionamento
    assert versionamento.plano_de_salvamento(reemp, "reempacotada")["problemas"] == []
    assert versionamento.plano_de_salvamento(der, "derivada")["problemas"] == []

    # as partições sintéticas passam a ser "as oficiais": a trava vale
    monkeypatch.setattr(caminhos, "PARTICOES_V2", particoes_sinteticas)
    monkeypatch.setattr(caminhos, "PARTICOES_V1", tmp_path / "inexistente")
    destino = tmp_path / "congelamento" / cg.ARQUIVO
    registro = cg.criar(der, responsavel="Autor", justificativa="teste", pasta_particoes=particoes_sinteticas)
    assert registro["execucao"]["base"]["pasta"] == reemp.name
    cg.gravar(registro, destino)
    codigo = cg.codigo_de_liberacao(registro)

    with pytest.raises(pair_aware_limiar.ErroLimiarPairAware, match="ainda não foi avaliado"):
        pair_aware_limiar.aplicar_test(der, congelamento=destino, liberacao=codigo)
    with pytest.raises(cg.ErroCongelamento, match="código de liberação"):
        classificador.avaliar_test(reemp, pasta_particoes=particoes_sinteticas, montagem=montagem,
                                   congelamento=destino)
    with pytest.raises(cg.ErroCongelamento, match="não 'pair_aware_sem_pesos_t'"):
        classificador.avaliar_test(base, pasta_particoes=particoes_sinteticas, montagem=montagem,
                                   congelamento=destino, liberacao=codigo)

    classificador.avaliar_test(reemp, pasta_particoes=particoes_sinteticas, montagem=montagem,
                               congelamento=destino, liberacao=codigo)
    resultado = pair_aware_limiar.aplicar_test(der, congelamento=destino, liberacao=codigo)
    assert resultado["t"] == r["t"]
    trilha_final = cg.avaliacoes_registradas(destino)
    assert [(a["execucao"], a["papel"]) for a in trilha_final] == [(reemp.name, cg.PAPEL_BASE), (der.name, cg.PAPEL_FINAL)]
    with pytest.raises(pair_aware_limiar.ErroLimiarPairAware, match="já foi avaliado"):
        pair_aware_limiar.aplicar_test(der, congelamento=destino, liberacao=codigo)


def test_notebook_da_configuracao_final_nao_toca_o_test():
    nb = json.loads((CODIGO / "notebooks" / "final" / "reclin_configuracao_final.ipynb").read_text(encoding="utf-8"))
    codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for proibido in ("avaliar_test", "aplicar_test", "aplicar-test", "test.jsonl", "predicoes_test", "--liberacao"):
        if proibido == "predicoes_test":
            assert codigo.count(proibido) == 1 and "assert not (BASE / 'predicoes_test.json').exists()" in codigo
            continue
        assert proibido not in codigo, proibido
    assert "PUBLICAR_NO_GITHUB = False" in codigo and "(0.8, 120, 22, 30)" in codigo
