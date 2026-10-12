"""Versões dos dados: identificação e a guarda contra misturá-las, sem ler o
corpus (as partições reais são conferidas em `test_dados_v2.py`)."""
from __future__ import annotations

import shutil

import pytest

from reclin import tarefa, versao_dados as vd
from reclin.execucao import diretorio, inventario as inv, predicoes
from reclin.util import caminhos
from reclin.util.io import gravar_json, ler_json, sha256_json

V1_TRAIN = vd.REGISTRO[vd.V1]["particoes"]["train"]
V1_TEST = vd.REGISTRO[vd.V1]["particoes"]["test"]
V2_TEST = vd.REGISTRO[vd.V2]["particoes"]["test"]


def test_registro_e_fase():
    assert vd.VERSOES == (vd.V1, vd.V2) and vd.VERSAO_DA_FASE == vd.V2 == "v2-texto-fonte"
    assert vd.PASTAS == {vd.V1: caminhos.PARTICOES_V1, vd.V2: caminhos.PARTICOES_V2}
    assert caminhos.PARTICOES == vd.PASTAS[vd.VERSAO_DA_FASE]
    todos = [s for r in vd.REGISTRO.values() for s in [r["manifest_sha256"], *r["particoes"].values()]]
    assert len(todos) == len(set(todos)) == 8                          # nenhum hash compartilhado
    assert vd.versao_da_particao(V1_TRAIN) == vd.V1 and vd.versao_da_particao(V2_TEST) == vd.V2
    assert vd.versao_da_particao("0" * 64) is None and vd.versao_da_particao(None) is None


def test_versoes_do_registro_e_mistura():
    config = {"config": {"particoes": {"train": V1_TRAIN}},
              "conjuntos": {"dev": {"particao_sha256": "x"}, "test": {"particao_sha256": V1_TEST}}}
    versoes = vd.versoes_do_registro(config)
    assert versoes == {"treino.train": vd.V1, "conjunto.dev": None, "conjunto.test": vd.V1}
    assert vd.conferir_uma_versao(versoes) == [] and vd.versao_unica(versoes) == vd.V1
    config["conjuntos"]["test"]["particao_sha256"] = V2_TEST
    versoes = vd.versoes_do_registro(config)
    assert vd.conferir_uma_versao(versoes) and vd.versao_unica(versoes) is None


def test_pasta_registrada_por_manifesto(tmp_path):
    assert vd.versao_da_pasta(tmp_path) is None and not vd.congelada(tmp_path)
    (tmp_path / "MANIFEST.json").write_text("{}", "utf-8")
    assert vd.versao_da_pasta(tmp_path) is None


def test_exigir_mesma_versao(tmp_path, monkeypatch):
    pasta_v2 = tmp_path / "v2"
    pasta_v2.mkdir()
    (pasta_v2 / "MANIFEST.json").write_text("{}", "utf-8")
    monkeypatch.setattr(vd, "versao_da_pasta", lambda p: vd.V2 if p == pasta_v2 else None)
    treinado_v1 = {"config": {"particoes": {"train": V1_TRAIN}}, "conjuntos": {}}
    with pytest.raises(vd.ErroVersaoDados, match="não se combinam"):
        vd.exigir_mesma_versao(treinado_v1, pasta_v2, "a avaliação do TEST")
    vd.exigir_mesma_versao(treinado_v1, tmp_path / "subconjunto", "x")      # pasta fora do registro: sem versão
    vd.exigir_mesma_versao({"config": {"particoes": {"train": "s"}}}, pasta_v2, "x")   # treino fora do registro


def test_diretorio_recusa_gravar_predicoes_de_outra_versao(tmp_path):
    pasta = diretorio.criar(tmp_path, "baseline_x_seed42", estrategia="baseline",
                            config={"particoes": {"train": V1_TRAIN}})
    p = predicoes.montar(model="m", seed=42, y_true=[0, 1, 2], y_pred=[0, 1, 2])
    conjunto = tarefa.conjunto_referencia("test", [], particao_sha256=V2_TEST, conferir_tamanho=False)
    conjunto.update(n_candidatos=3, y_true_sha256=sha256_json([0, 1, 2]))
    with pytest.raises(vd.ErroVersaoDados, match="versões diferentes"):
        diretorio.gravar_predicoes(pasta, "test", p, conjunto)
    assert not (pasta / "predicoes_test.json").exists()
    conjunto["particao_sha256"] = V1_TEST
    diretorio.gravar_predicoes(pasta, "test", p, conjunto)
    assert diretorio.ler_config(pasta)["versoes_dados"] == {"test": vd.V1}


def test_inventario_acusa_mistura_e_registro_inconsistente(fabrica_execucoes, particoes_sinteticas, ferramentas):
    pasta = fabrica_execucoes("baseline_biobertpt_seed42", "baseline", ferramentas.registro_treinado("baseline"))
    config = ler_json(pasta / diretorio.CONFIG)
    assert config["versoes_dados"] == {"dev": None, "test": None}         # partições sintéticas: fora do registro
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter(pasta.name)
    assert ex.valida and ex.versao_dados is None
    config["config"]["particoes"] = {"train": V1_TRAIN}
    config["conjuntos"]["test"]["particao_sha256"] = V2_TEST
    gravar_json(pasta / diretorio.CONFIG, config)
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter(pasta.name)
    assert any("versões diferentes dos dados" in p for p in ex.problemas)
    assert any("versoes_dados[test]" in p for p in ex.problemas)


def test_inventario_derivada_de_outra_versao(fabrica_execucoes, particoes_sinteticas, ferramentas):
    fabrica_execucoes("baseline_biobertpt_seed42", "baseline", ferramentas.registro_treinado("baseline"))
    calibrado = fabrica_execucoes("calibrado", "baseline_calibrado",
                                  {"base": {"nome": "baseline_biobertpt_seed42"}, "seed": 42})
    base = fabrica_execucoes.raiz / "baseline_biobertpt_seed42" / diretorio.CONFIG
    c = ler_json(base)
    c["config"]["particoes"] = {"train": V1_TRAIN}
    gravar_json(base, c)
    c = ler_json(calibrado / diretorio.CONFIG)
    c["config"]["particoes"] = {"train": vd.REGISTRO[vd.V2]["particoes"]["train"]}
    gravar_json(calibrado / diretorio.CONFIG, c)
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    assert any("versão dos dados" in p for p in i.obter("calibrado").problemas)
    with pytest.raises(inv.ErroInventario):
        i.selecionar(["baseline_biobertpt_seed42", "calibrado"], natureza="fora_da_fase")


def test_inventario_seleciona_so_uma_versao(fabrica_execucoes, particoes_sinteticas, ferramentas):
    for nome, sha in (("a", V1_TRAIN), ("b", vd.REGISTRO[vd.V2]["particoes"]["train"])):
        registro = ferramentas.registro_treinado("baseline")
        registro["particoes"] = {"train": sha}
        fabrica_execucoes(nome, "baseline", registro)
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    assert (i.obter("a").versao_dados, i.obter("b").versao_dados) == (vd.V1, vd.V2)
    assert "dados da versão v1-conversao-legado (a fase usa v2-texto-fonte)" in i.obter("a").motivos_fora_da_fase
    # (o treino aponta partições reais, que não são as sintéticas do inventário: isso já é um problema;
    # sem ele, a seleção ainda recusa juntar as duas versões)
    assert all(any("não é a do MANIFEST" in p for p in i.obter(n).problemas) for n in "ab")
    for n in "ab":
        i.obter(n).problemas.clear()
    with pytest.raises(inv.ErroInventario, match="versões diferentes dos dados"):
        i.selecionar(["a", "b"], natureza="fora_da_fase")


def test_copia_do_manifesto_da_fase_e_congelada(tmp_path):
    if not (caminhos.PARTICOES_V2 / "MANIFEST.json").is_file():
        pytest.skip("partições da v2 ausentes nesta cópia")
    shutil.copy(caminhos.PARTICOES_V2 / "MANIFEST.json", tmp_path / "MANIFEST.json")
    assert vd.versao_da_pasta(tmp_path) == vd.V2 and vd.descrever(tmp_path)["da_fase"]


def test_preparar_dados_nao_regrava_a_versao_gerada_dos_xml(tmp_path):
    """`manifesto` e `particionar` recusam uma pasta cujo MANIFEST tem a
    proveniência da geração (chave `versao`): essa pasta só muda pelo
    `reconstruir_dados.py`."""
    import subprocess
    import sys
    gravar_json(tmp_path / "MANIFEST.json", {"versao": vd.V2, "splits": {}, "seed": 42, "source": {}})
    for comando in ("manifesto", "particionar"):
        r = subprocess.run([sys.executable, str(caminhos.CODIGO / "scripts" / "preparar_dados.py"),
                            "--pasta", str(tmp_path), comando], capture_output=True, text=True)
        assert r.returncode == 2, r.stderr
    assert ler_json(tmp_path / "MANIFEST.json")["versao"] == vd.V2


def test_protocolo_registra_a_versao_dos_dados_da_fase():
    """O bloco JSON do protocolo da fase aponta a versão, a pasta e o MANIFEST
    registrados em `versao_dados` (sem depender do torch, ao contrário de
    `test_protocolo_fase.py`)."""
    import json
    import re
    texto = (caminhos.CODIGO / "docs" / "protocolo" / "fase-biobertpt-seed42.md").read_text(encoding="utf-8")
    dados = json.loads(re.findall(r"```json\n(.*?)\n```", texto, flags=re.S)[0])["dados"]
    assert dados["versao"] == vd.VERSAO_DA_FASE
    assert caminhos.CODIGO / dados["pasta"] == vd.PASTAS[vd.VERSAO_DA_FASE]
    assert dados["manifest_sha256"] == vd.REGISTRO[vd.VERSAO_DA_FASE]["manifest_sha256"]
