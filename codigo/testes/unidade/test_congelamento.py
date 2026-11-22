"""Trava da avaliação do TEST: congelamento da configuração final e liberação.

Sem torch e sem partições: as pastas são temporárias e o "TEST científico" é
simulado por um MANIFEST com um hash fictício."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from reclin.execucao import congelamento as cg
from reclin.util import caminhos

CODIGO = caminhos.CODIGO


def _particoes(pasta: Path, sha_test: str) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "MANIFEST.json").write_text(json.dumps({"splits": {"test": {"sha256": sha_test}}}), "utf-8")
    return pasta


def _execucao(pasta: Path, *, concluido: bool = True) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "config.json").write_text(json.dumps({"nome": pasta.name, "estrategia": "pair_aware",
                                                   "config": {"seed": 42}}), "utf-8")
    (pasta / "predicoes_dev.json").write_text(json.dumps({"split": "dev", "y_true": [0, 1], "y_pred": [0, 2]}), "utf-8")
    (pasta / "treino.json").write_text(json.dumps({"concluido": concluido, "test_avaliado": False}), "utf-8")
    melhor = pasta / cg.PASTA_CHECKPOINTS / cg.MELHOR_MODELO
    melhor.mkdir(parents=True, exist_ok=True)
    (melhor / "model.safetensors").write_bytes(b"pesos do melhor modelo")
    return pasta


@pytest.fixture
def cientificas(tmp_path, monkeypatch):
    """Uma pasta que o módulo reconhece como as partições oficiais."""
    oficial = _particoes(tmp_path / "oficial", "sha-do-test-cientifico")
    monkeypatch.setattr(caminhos, "PARTICOES_V2", oficial)
    monkeypatch.setattr(caminhos, "PARTICOES_V1", tmp_path / "inexistente")
    return oficial


def test_particoes_tecnicas_nao_sao_o_test_cientifico(tmp_path, cientificas):
    assert cg.test_cientifico(cientificas)
    assert cg.test_cientifico(_particoes(tmp_path / "copia", "sha-do-test-cientifico"))   # cópia (ex.: Colab)
    assert not cg.test_cientifico(_particoes(tmp_path / "subconjunto", "outro-sha"))
    (tmp_path / "sem_manifesto").mkdir()
    assert cg.test_cientifico(tmp_path / "sem_manifesto")                                 # na dúvida, vale a trava
    assert cg.sha_de_test_cientifico(None) and cg.sha_de_test_cientifico("sha-do-test-cientifico")
    assert not cg.sha_de_test_cientifico("outro-sha")


def test_sem_congelamento_o_test_cientifico_e_recusado(tmp_path, cientificas):
    ex = _execucao(tmp_path / "exec")
    with pytest.raises(cg.ErroCongelamento, match="não há configuração final congelada"):
        cg.exigir_liberacao(ex, cientificas, congelamento=tmp_path / "c" / cg.ARQUIVO, liberacao="qualquer")
    # nas partições técnicas não há trava (nem registro a devolver)
    assert cg.exigir_liberacao(ex, _particoes(tmp_path / "tecnicas", "outro-sha"),
                               congelamento=tmp_path / "c" / cg.ARQUIVO) is None


def test_congelar_liberar_e_avaliar_uma_vez(tmp_path, cientificas):
    ex, outra = _execucao(tmp_path / "final"), _execucao(tmp_path / "outra")
    destino = tmp_path / "congelamento" / cg.ARQUIVO
    registro = cg.criar(ex, responsavel="Autor", justificativa="melhor macro-F1 no DEV", pasta_particoes=cientificas)
    cg.gravar(registro, destino)
    codigo = cg.codigo_de_liberacao(registro)
    assert json.loads(destino.read_text("utf-8"))["liberacao"]["codigo"] == codigo and len(codigo) == 12
    with pytest.raises(cg.ErroCongelamento, match="já existe um congelamento"):
        cg.gravar(registro, destino)

    kw = {"congelamento": destino}
    for liberacao in (None, "", "000000000000"):
        with pytest.raises(cg.ErroCongelamento, match="código de liberação"):
            cg.exigir_liberacao(ex, cientificas, liberacao=liberacao, **kw)
    with pytest.raises(cg.ErroCongelamento, match="a configuração congelada é 'final'"):
        cg.exigir_liberacao(outra, cientificas, liberacao=codigo, **kw)

    liberado = cg.exigir_liberacao(ex, cientificas, liberacao=codigo, **kw)
    assert liberado["execucao"]["pasta"] == "final"
    assert cg.avaliacoes_registradas(destino) == []
    linha = cg.registrar_avaliacao(liberado, macro_f1=0.5, negation_of_f1=0.6, n_test=10)
    assert linha["indice"] == 1 and cg.avaliacoes_registradas(destino)[0]["execucao"] == "final"
    with pytest.raises(cg.ErroCongelamento, match="já foi avaliada"):
        cg.exigir_liberacao(ex, cientificas, liberacao=codigo, **kw)
    assert cg.exigir_liberacao(ex, cientificas, liberacao=codigo, reavaliar=True, **kw) is not None
    assert cg.registrar_avaliacao(None, macro_f1=0, negation_of_f1=0, n_test=0) is None


def test_execucao_ou_registro_alterados_sao_recusados(tmp_path, cientificas):
    ex = _execucao(tmp_path / "final")
    destino = tmp_path / "c" / cg.ARQUIVO
    registro = cg.criar(ex, responsavel="Autor", justificativa="j", pasta_particoes=cientificas)
    cg.gravar(registro, destino)
    codigo = cg.codigo_de_liberacao(registro)
    (ex / "predicoes_dev.json").write_text(json.dumps({"split": "dev", "y_true": [0, 1], "y_pred": [0, 1]}), "utf-8")
    with pytest.raises(cg.ErroCongelamento, match="mudou depois do congelamento"):
        cg.exigir_liberacao(ex, cientificas, congelamento=destino, liberacao=codigo)
    adulterado = json.loads(destino.read_text("utf-8"))
    adulterado["execucao"] = cg.impressoes_da_execucao(ex)
    destino.write_text(json.dumps(adulterado), "utf-8")
    with pytest.raises(cg.ErroCongelamento, match="foi alterado depois de gravado"):
        cg.exigir_liberacao(ex, cientificas, congelamento=destino, liberacao=codigo)


def test_o_que_nao_pode_ser_congelado(tmp_path, cientificas):
    with pytest.raises(cg.ErroCongelamento, match="responsável e a justificativa"):
        cg.criar(_execucao(tmp_path / "a"), responsavel=" ", justificativa="j", pasta_particoes=cientificas)
    with pytest.raises(cg.ErroCongelamento, match="treino não concluído"):
        cg.criar(_execucao(tmp_path / "b", concluido=False), responsavel="A", justificativa="j")
    avaliada = _execucao(tmp_path / "c")
    (avaliada / "predicoes_test.json").write_text("{}", "utf-8")
    with pytest.raises(cg.ErroCongelamento, match="já foi avaliado"):
        cg.criar(avaliada, responsavel="A", justificativa="j")
    sem_dev = _execucao(tmp_path / "d")
    (sem_dev / "predicoes_dev.json").unlink()
    with pytest.raises(cg.ErroCongelamento, match="não tem predicoes_dev.json"):
        cg.criar(sem_dev, responsavel="A", justificativa="j")


@pytest.mark.parametrize("arquivo,funcao,leituras", [
    ("reclin/treino/classificador.py", "avaliar_test", ("ler_particao(", "_conferir_particoes(")),
    ("reclin/estrategias/two_stage/composicao.py", "avaliar_test", ('_particao(pasta_particoes, "test"',)),
    ("reclin/estrategias/baseline_calibrado.py", "aplicar_test", ("predicoes.ler(origem)",)),
    ("reclin/estrategias/pair_aware_limiar.py", "aplicar_test", ("predicoes.ler(origem)",)),
])
def test_a_trava_vem_antes_de_toda_leitura_do_test(arquivo, funcao, leituras):
    """Nas funções que leem o TEST, `exigir_liberacao` aparece antes da primeira leitura."""
    texto = (CODIGO / arquivo).read_text(encoding="utf-8")
    corpo = re.search(rf"^def {funcao}\(.*?(?=^def |\Z)", texto, re.S | re.M).group(0)
    trava = corpo.index("congelamento_final.exigir_liberacao(")
    for leitura in leituras:
        assert leitura in corpo and trava < corpo.index(leitura), (arquivo, leitura)
    assert "congelamento_final.registrar_avaliacao(" in corpo


def test_modulo_nao_importa_torch_nem_le_particoes():
    texto = (CODIGO / "reclin/execucao/congelamento.py").read_text(encoding="utf-8")
    assert "import torch" not in texto and "ler_particao" not in texto and ".jsonl\"" not in texto.replace(
        'TRILHA = "avaliacoes_test_final.jsonl"', "")


def test_os_pesos_do_melhor_modelo_entram_no_registro(tmp_path, cientificas):
    ex = _execucao(tmp_path / "final")
    destino = tmp_path / "congelamento" / cg.ARQUIVO
    registro = cg.criar(ex, responsavel="Autor", justificativa="DEV", pasta_particoes=cientificas)
    assert len(registro["execucao"]["sha256_melhor_modelo"]) == 64          # calculado da pasta, não lido
    cg.gravar(registro, destino)
    codigo = cg.codigo_de_liberacao(registro)
    assert cg.exigir_liberacao(ex, cientificas, congelamento=destino, liberacao=codigo)
    (ex / cg.PASTA_CHECKPOINTS / cg.MELHOR_MODELO / "model.safetensors").write_bytes(b"outros pesos")
    with pytest.raises(cg.ErroCongelamento, match="mudou depois do congelamento"):
        cg.exigir_liberacao(ex, cientificas, congelamento=destino, liberacao=codigo)

    sem_pesos = _execucao(tmp_path / "sem_pesos")
    (sem_pesos / cg.PASTA_CHECKPOINTS / cg.MELHOR_MODELO / "model.safetensors").unlink()
    (sem_pesos / cg.PASTA_CHECKPOINTS / cg.MELHOR_MODELO).rmdir()
    with pytest.raises(cg.ErroCongelamento, match="congele na máquina em que os pesos"):
        cg.criar(sem_pesos, responsavel="Autor", justificativa="DEV", pasta_particoes=cientificas)
    (sem_pesos / "treino.json").unlink()                                     # derivada (sem treino): não tem pesos
    assert cg.criar(sem_pesos, responsavel="Autor", justificativa="DEV",
                    pasta_particoes=cientificas)["execucao"]["sha256_melhor_modelo"] is None


def test_nomes_das_pastas_de_pesos_sao_os_do_treino():
    pytest.importorskip("torch")
    from reclin.treino import checkpoint
    assert checkpoint.pasta_checkpoints(Path("x")) / checkpoint.MELHOR == Path("x") / cg.PASTA_CHECKPOINTS / cg.MELHOR_MODELO


# --------------------------------------------------------------------------- #
# Configuração derivada (limiar sobre as probabilidades de uma base treinada)   #
# --------------------------------------------------------------------------- #
def _derivada(raiz: Path, base: Path, nome: str = "derivada") -> Path:
    pasta = raiz / nome
    pasta.mkdir(parents=True)
    (pasta / "config.json").write_text(json.dumps({"nome": nome, "estrategia": "pair_aware_limiar",
                                                   "config": {"estrategia": "pair_aware_limiar",
                                                              "base": {"nome": base.name, "pasta": "/outro/lugar"}}}),
                                       "utf-8")
    (pasta / "predicoes_dev.json").write_text(json.dumps({"split": "dev", "y_true": [0, 1], "y_pred": [2, 2]}), "utf-8")
    (pasta / "limiar.json").write_text(json.dumps({"limiar": 0.8}), "utf-8")
    return pasta


def test_derivada_congela_os_pesos_da_base_e_libera_base_e_derivada(tmp_path, cientificas):
    raiz = tmp_path / "execucoes"
    base = _execucao(raiz / "base")
    der = _derivada(raiz, base)
    destino = tmp_path / "c" / cg.ARQUIVO
    registro = cg.criar(der, responsavel="Autor", justificativa="emenda", pasta_particoes=cientificas)
    assert registro["execucao"]["sha256_melhor_modelo"] is None
    assert registro["execucao"]["base"]["pasta"] == "base" and len(registro["execucao"]["base"]["sha256_melhor_modelo"]) == 64
    assert "limiar.json" in registro["execucao"]["arquivos_sha256"]
    cg.gravar(registro, destino)
    codigo = cg.codigo_de_liberacao(registro)
    kw = {"congelamento": destino, "liberacao": codigo}

    # a derivada só depois da base; outra execução, nunca
    with pytest.raises(cg.ErroCongelamento, match="avalie a base primeiro"):
        cg.exigir_liberacao(der, cientificas, **kw)
    with pytest.raises(cg.ErroCongelamento, match="com a base 'base'"):
        cg.exigir_liberacao(_execucao(raiz / "outra"), cientificas, **kw)

    liberado = cg.exigir_liberacao(base, cientificas, **kw)
    assert liberado["_papel"] == cg.PAPEL_BASE
    linha = cg.registrar_avaliacao(liberado, macro_f1=0.5, negation_of_f1=0.6, n_test=10)
    assert linha["execucao"] == "base" and linha["papel"] == cg.PAPEL_BASE
    with pytest.raises(cg.ErroCongelamento, match="já foi avaliada"):
        cg.exigir_liberacao(base, cientificas, **kw)

    liberado = cg.exigir_liberacao(der, cientificas, **kw)
    assert liberado["_papel"] == cg.PAPEL_FINAL
    cg.registrar_avaliacao(liberado, macro_f1=0.5, negation_of_f1=0.7, n_test=10)
    assert [a["papel"] for a in cg.avaliacoes_registradas(destino)] == [cg.PAPEL_BASE, cg.PAPEL_FINAL]
    with pytest.raises(cg.ErroCongelamento, match="já foi avaliada"):
        cg.exigir_liberacao(der, cientificas, **kw)


def test_derivada_recusada_se_a_base_mudar_ou_faltar(tmp_path, cientificas):
    raiz = tmp_path / "execucoes"
    base = _execucao(raiz / "base")
    der = _derivada(raiz, base)
    destino = tmp_path / "c" / cg.ARQUIVO
    registro = cg.criar(der, responsavel="Autor", justificativa="emenda", pasta_particoes=cientificas)
    cg.gravar(registro, destino)
    codigo = cg.codigo_de_liberacao(registro)
    (base / cg.PASTA_CHECKPOINTS / cg.MELHOR_MODELO / "model.safetensors").write_bytes(b"outros pesos")
    with pytest.raises(cg.ErroCongelamento, match="mudou depois do congelamento"):
        cg.exigir_liberacao(base, cientificas, congelamento=destino, liberacao=codigo)

    sem_pesos = _execucao(tmp_path / "r2" / "base")
    (sem_pesos / cg.PASTA_CHECKPOINTS / cg.MELHOR_MODELO / "model.safetensors").unlink()
    (sem_pesos / cg.PASTA_CHECKPOINTS / cg.MELHOR_MODELO).rmdir()
    with pytest.raises(cg.ErroCongelamento, match="pesos do melhor modelo da base"):
        cg.criar(_derivada(tmp_path / "r2", sem_pesos), responsavel="A", justificativa="j", pasta_particoes=cientificas)

    avaliada = _execucao(tmp_path / "r3" / "base")
    (avaliada / "predicoes_test.json").write_text("{}", "utf-8")
    with pytest.raises(cg.ErroCongelamento, match="TEST da base base já foi avaliado"):
        cg.criar(_derivada(tmp_path / "r3", avaliada), responsavel="A", justificativa="j", pasta_particoes=cientificas)

    orfa = _derivada(tmp_path / "r4", tmp_path / "inexistente")
    with pytest.raises(cg.ErroCongelamento, match="não foi encontrada"):
        cg.criar(orfa, responsavel="A", justificativa="j", pasta_particoes=cientificas)


def test_config_conta_pela_configuracao_nao_pelos_conjuntos_avaliados(tmp_path, cientificas):
    ex = _execucao(tmp_path / "final")
    antes = cg.impressoes_da_execucao(ex)
    cfg = json.loads((ex / "config.json").read_text("utf-8"))
    (ex / "config.json").write_text(json.dumps({**cfg, "conjuntos": {"test": {"particao": "test"}}}), "utf-8")
    assert cg.impressoes_da_execucao(ex) == antes                 # a avaliação do TEST acrescenta isto
    (ex / "config.json").write_text(json.dumps({**cfg, "config": {"seed": 43}}), "utf-8")
    assert cg.impressoes_da_execucao(ex) != antes                 # outra configuração
