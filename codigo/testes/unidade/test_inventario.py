"""Inventário das execuções e verificação de proveniência (etapa 7), sobre
partições e execuções sintéticas: nada do legado nem do TEST real é lido."""
from __future__ import annotations

import json

import pytest

from reclin.execucao import diretorio, inventario as inv
from reclin.util import caminhos
from reclin.util.io import gravar_json, ler_json


def test_execucao_valida_e_natureza(fabrica_execucoes, particoes_sinteticas, ferramentas):
    fabrica_execucoes("baseline_biobertpt_seed42", "baseline", ferramentas.registro_treinado("baseline"))
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    ex = i.obter("baseline_biobertpt_seed42")
    assert ex.valida and ex.tem("dev") and ex.tem("test") and ex.test_avaliacoes == 1
    # partições sintéticas: válida, mas fora da fase
    assert ex.natureza == "fora_da_fase" and ex.motivos_fora_da_fase == ["partições diferentes das congeladas"]
    assert not i.particoes_congeladas


def test_motivos_fora_da_fase(fabrica_execucoes, particoes_sinteticas, ferramentas):
    fabrica_execucoes("a", "baseline", ferramentas.registro_treinado("baseline", seed=43, encoder="bertimbau",
                                                                     proveniencia=False))
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter("a")
    assert ex.natureza == "fora_da_fase"
    assert ex.motivos_fora_da_fase[:3] == ["semente 43 (a fase usa 42)", "encoder bertimbau (a fase usa biobertpt)",
                                           "sem proveniência dos pesos registrada (C9)"]


def test_itens_recusados(fabrica_execucoes, particoes_sinteticas, ferramentas):
    fabrica_execucoes("b", "baseline", ferramentas.registro_treinado("baseline"))
    raiz = fabrica_execucoes.raiz
    for nome in ("x.preds.json", "x.dev_preds.json", "x.test_evals.jsonl", "significance_a_vs_b.json",
                 "CALIBRACAO_filtro.json", "FASE2_test_summary.json", "summary_by_seed.json", "solto.json"):
        (raiz / nome).write_text("{}", "utf-8")
    (raiz / "sem_config").mkdir()
    i = inv.inventariar(raiz, pasta_particoes=particoes_sinteticas)
    assert set(i.execucoes) == {"b"}
    motivos = {r["caminho"]: r["motivo"] for r in i.recusados}
    assert "formato ou o nome dos resultados do legado" in motivos["x.preds.json"]
    assert "formato ou o nome dos resultados do legado" in motivos["CALIBRACAO_filtro.json"]
    assert "arquivo solto" in motivos["solto.json"]
    assert "sem config.json" in motivos["sem_config"]


def test_referencias_do_legado_sao_recusadas():
    with pytest.raises(inv.ErroInventario, match="referências do legado"):
        inv.inventariar(caminhos.CODIGO / "testes" / "referencia" / "resultados_legado")
    with pytest.raises(inv.ErroInventario, match="referências do legado"):
        inv.inspecionar(caminhos.CODIGO / "testes" / "referencia" / "treino")
    assert inv.recusar_caminho(caminhos.CODIGO / "testes" / "referencia") is not None


def test_sidecar_de_outro_conjunto(fabrica_execucoes, particoes_sinteticas, ferramentas):
    pasta = fabrica_execucoes("c", "baseline", ferramentas.registro_treinado("baseline"))
    config = ler_json(pasta / "config.json")
    config["conjuntos"]["dev"]["particao_sha256"] = "0" * 64
    gravar_json(pasta / "config.json", config)
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter("c")
    assert not ex.valida and any("não é a do MANIFEST" in p for p in ex.problemas)
    assert not ex.tem("dev")


def test_y_true_adulterado(fabrica_execucoes, particoes_sinteticas, ferramentas):
    pasta = fabrica_execucoes("d", "baseline", ferramentas.registro_treinado("baseline"), metricas_=False)
    sidecar = json.loads((pasta / "predicoes_dev.json").read_text("utf-8"))
    sidecar["y_true"][0] = (sidecar["y_true"][0] + 1) % 3
    (pasta / "predicoes_dev.json").write_text(json.dumps(sidecar), "utf-8")
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter("d")
    assert any("y_true diferente" in p for p in ex.problemas)


def test_metricas_gravadas_diferentes_das_recalculadas(fabrica_execucoes, particoes_sinteticas, ferramentas):
    pasta = fabrica_execucoes("e", "baseline", ferramentas.registro_treinado("baseline"))
    m = diretorio.ler_metricas(pasta)
    m["dev"]["macro_f1"] += 0.01
    gravar_json(pasta / "metricas.json", m)
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter("e")
    assert any("metricas.json[dev]" in p for p in ex.problemas)


def test_test_sem_trilha_nao_e_usado(fabrica_execucoes, particoes_sinteticas, ferramentas):
    fabrica_execucoes("f", "baseline", ferramentas.registro_treinado("baseline"), avaliar_test=False)
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter("f")
    assert ex.valida and ex.tem("dev") and not ex.tem("test")
    assert any("sem trilha" in a for a in ex.avisos)
    with pytest.raises(inv.ErroInventario):
        ex.predicoes("test")


def test_reavaliacao_vira_aviso(fabrica_execucoes, particoes_sinteticas, ferramentas):
    pasta = fabrica_execucoes("g", "baseline", ferramentas.registro_treinado("baseline"))
    linha = (pasta / diretorio.TRILHA).read_text("utf-8")
    (pasta / diretorio.TRILHA).write_text(linha + linha, "utf-8")
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter("g")
    assert ex.valida and ex.test_avaliacoes == 2 and any("2 vezes" in a for a in ex.avisos)


def test_treino_nao_concluido(fabrica_execucoes, particoes_sinteticas, ferramentas):
    pasta = fabrica_execucoes("h", "baseline", ferramentas.registro_treinado("baseline"))
    ferramentas.gravar_treino(pasta, [], None, concluido=False)
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter("h")
    assert "treino não concluído" in ex.problemas


def _origem_da_base(pasta_base):
    from reclin.util.io import sha256_arquivo
    return {p: {"base_sha256": sha256_arquivo(pasta_base / f"predicoes_{p}.json")} for p in ("dev", "test")}


def test_derivadas_herdam_da_base(fabrica_execucoes, particoes_sinteticas, ferramentas):
    base = fabrica_execucoes("base", "baseline", ferramentas.registro_treinado("baseline"))
    fabrica_execucoes("filtro", "filtro_pistas", {"base": "base", "min_freq": 3}, extras=_origem_da_base(base))
    fabrica_execucoes("orfao", "filtro_pistas", {"base": "nao_existe", "min_freq": 3})
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    filtro = i.obter("filtro")
    assert filtro.valida and filtro.seed == 42 and filtro.encoder == "biobertpt"
    assert filtro.proveniencia["commit"] == "a" * 40
    assert any("nao_existe" in p for p in i.obter("orfao").problemas)


def test_filtro_sem_vinculo_com_a_base_ou_com_calibracao_externa(fabrica_execucoes, particoes_sinteticas,
                                                                 ferramentas):
    """D2/D3: um filtro que não aponta as predições atuais da base (por exemplo,
    feito sobre sidecars do legado) ou que traz um arquivo de calibração
    externo não é aceito."""
    base = fabrica_execucoes("base", "baseline", ferramentas.registro_treinado("baseline"))
    origem = _origem_da_base(base)
    fabrica_execucoes("f_sem", "filtro_pistas", {"base": "base"})
    fabrica_execucoes("f_outra", "filtro_pistas", {"base": "base"},
                      extras={"dev": {"base_sha256": "0" * 64}, "test": origem["test"]})
    fabrica_execucoes("f_cal", "filtro_pistas", {"base": "base", "calibracao": {"min_freq": 3}}, extras=origem)
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    assert any("base_sha256" in p for p in i.obter("f_sem").problemas)
    assert any("não veio das predições atuais" in p for p in i.obter("f_outra").problemas)
    assert any("calibração externo" in p for p in i.obter("f_cal").problemas)
    assert i.obter("base").valida


def test_marca_de_validacao_herdada_e_mistura(fabrica_execucoes, particoes_sinteticas, ferramentas):
    marca = {"epocas": 1, "epocas_protocolo": 3, "rotulo": "validação técnica, 1 época"}
    fabrica_execucoes("bv", "baseline", ferramentas.registro_treinado("baseline", validacao=marca))
    fabrica_execucoes("fv", "filtro_pistas", {"base": "bv"})
    fabrica_execucoes("cal", "baseline_calibrado", {"base": {"nome": "bv"}, "seed": 42,
                                                    "proveniencia_pesos": ferramentas.registro_treinado(
                                                        "x")["proveniencia_pesos"]})
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    assert i.obter("bv").natureza == "validacao" and i.obter("fv").natureza == "validacao"
    # o calibrado declara-se da fase científica, mas a base é de validação: recusado
    assert any("mistura" in p for p in i.obter("cal").problemas)


def test_selecionar_recusa_misturas(fabrica_execucoes, particoes_sinteticas, ferramentas):
    marca = {"epocas": 1, "epocas_protocolo": 3, "rotulo": "validação técnica, 1 época"}
    fabrica_execucoes("x1", "baseline", ferramentas.registro_treinado("baseline"))
    fabrica_execucoes("x2", "baseline", ferramentas.registro_treinado("baseline", validacao=marca))
    fabrica_execucoes("x3", "baseline", ferramentas.registro_treinado("baseline", seed=43))
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    assert [e.nome for e in i.selecionar(["x1", "nao_existe"], natureza="fora_da_fase")] == ["x1"]
    with pytest.raises(inv.ErroInventario, match="naturezas diferentes"):
        i.selecionar(["x1", "x2"], natureza="fora_da_fase")
    with pytest.raises(inv.ErroInventario, match="sementes diferentes"):
        i.selecionar(["x1", "x3"], natureza="fora_da_fase")
    with pytest.raises(inv.ErroInventario):
        i.selecionar(["nao_existe"], natureza="fora_da_fase", ausentes="falhar")


def test_origens_detectam_mudanca(fabrica_execucoes, particoes_sinteticas, ferramentas):
    pasta = fabrica_execucoes("o", "baseline", ferramentas.registro_treinado("baseline"))
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    origens = inv.origens([i.obter("o")])
    assert inv.conferir_origens(origens, i) == []
    sidecar = json.loads((pasta / "predicoes_dev.json").read_text("utf-8"))
    (pasta / "predicoes_dev.json").write_text(json.dumps(sidecar, indent=1), "utf-8")
    i2 = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    assert any("sidecars mudou" in p for p in inv.conferir_origens(origens, i2))
    # o sidecar lido depois do inventário também é conferido pelo hash
    (pasta / "predicoes_dev.json").write_text(json.dumps(sidecar), "utf-8")
    with pytest.raises(inv.ErroInventario, match="mudou depois do inventário"):
        i2.obter("o").predicoes("dev")


def test_inventario_nao_escreve_nada(fabrica_execucoes, particoes_sinteticas, ferramentas):
    fabrica_execucoes("w", "baseline", ferramentas.registro_treinado("baseline"))
    antes = sorted((p.relative_to(fabrica_execucoes.raiz), p.stat().st_mtime_ns)
                   for p in fabrica_execucoes.raiz.rglob("*"))
    inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).resumo()
    depois = sorted((p.relative_to(fabrica_execucoes.raiz), p.stat().st_mtime_ns)
                    for p in fabrica_execucoes.raiz.rglob("*"))
    assert antes == depois


def test_execucao_treinada_sem_treino_json(fabrica_execucoes, particoes_sinteticas, ferramentas):
    """Uma pasta montada à mão com predições e um config.json de estratégia
    treinada, mas sem o treino.json, não é aceita."""
    pasta = fabrica_execucoes("m", "baseline", ferramentas.registro_treinado("baseline"))
    (pasta / "treino.json").unlink()
    ex = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas).obter("m")
    assert "execução treinada sem treino.json" in ex.problemas
