"""Identificação, nomes oficiais, planejamento de execuções e identidade da infraestrutura.

Só infraestrutura local: nada de rede, GPU, pesos, corpus, treino ou TEST. Os
dados e os pesos são entregues já prontos, ou lidos de arquivos mínimos em
`tmp_path` com as consultas de versão trocadas por valores fixos."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from reclin.experimento import identidade, ids, persistencia, plano, proveniencia
from reclin.util import caminhos, identidade_codigo as ic
from reclin.util.io import gravar_json

PESOS = {"repositorio": "pucpr/biobertpt-all", "commit": "b" * 40, "hash_manifesto": "m" * 64}
DADOS = {"versao": "v2-texto-fonte", "particoes": {"train": "t" * 64, "dev": "d" * 64}}
CODIGO = {"escopo": "global-v1", "algoritmo": "sha256-manifesto-v1", "identidade": "i" * 64}
CFG = {"tipo": "Config", "seed": 42, "encoder": "biobertpt", "epochs": 3, "lr": 2e-5}


class _Config:
    """Imita `reclin.config.Config`: `como_dict()` inclui o caminho/identificador do modelo."""
    def __init__(self, **extra):
        self.extra = extra

    def como_dict(self):
        return {**CFG, "modelo": "/content/pesos/biobertpt", **self.extra}


def _chave(estrategia="baseline", config=None, **mudar):
    return plano.chave_da_execucao(estrategia=estrategia, config=config or _Config(), estrategia_config=None,
                                   pesos=PESOS, dados=DADOS, identidade_codigo=CODIGO, **mudar)


def _nova(raiz, chave, **kw):
    p = plano.planejar(chave, raiz, **kw)
    assert p["acao"] == "nova_execucao"
    return p, plano.preparar(p, chave, raiz, codigo={"origem": "local"})


def _como(pasta: Path, estado: str):
    """Deixa na pasta os arquivos que evidenciam o estado (como o treino os deixaria)."""
    if estado == "interrompida":
        (pasta / "checkpoints").mkdir(exist_ok=True)
        (pasta / "checkpoints" / "ultimo.pt").write_bytes(b"x")
    elif estado == "concluida":
        gravar_json(pasta / "treino.json", {"concluido": True, "sessoes": []})
    elif estado == "falhou":
        proveniencia.mudar_estado(pasta, "falhou", "teste")


# ------------------------------------------------------------------ chave
def test_chave_vem_do_config_efetivo_sem_o_caminho_do_modelo():
    a = _chave()
    assert "modelo" not in a["config"] and a["config"]["seed"] == 42
    assert ids.config_id(a) == ids.config_id(_chave(config={**CFG, "modelo": "/outro/lugar"}))
    assert ids.config_id(a) != ids.config_id(_chave(config=_Config(lr=3e-5)))
    assert ids.config_id(a) != ids.config_id(_chave(estrategia="restrito"))


def test_chave_muda_com_dados_pesos_e_codigo():
    base = ids.config_id(_chave())
    outros = [dict(dados={**DADOS, "particoes": {**DADOS["particoes"], "dev": "e" * 64}}),
              dict(pesos={**PESOS, "commit": "c" * 40}), dict(identidade_codigo={**CODIGO, "identidade": "j" * 64})]
    for mudanca in outros:
        argumentos = dict(estrategia="baseline", config=_Config(), estrategia_config=None, pesos=PESOS,
                          dados=DADOS, identidade_codigo=CODIGO)
        argumentos.update(mudanca)
        assert ids.config_id(plano.chave_da_execucao(**argumentos)) != base


def test_chave_exige_as_partes_e_recusa_caminho_absoluto():
    with pytest.raises(plano.ErroPlano, match="pasta_pesos"):
        plano.chave_da_execucao(estrategia="baseline", config=_Config(), estrategia_config=None, dados=DADOS,
                                identidade_codigo=CODIGO)
    with pytest.raises(plano.ErroPlano, match="caminhos absolutos"):
        _chave(config={**CFG, "saida": "/content/x"})


def test_dados_e_pesos_lidos_de_pastas(tmp_path, monkeypatch):
    import reclin.pretreino as pretreino
    import reclin.versao_dados as versao_dados
    gravar_json(tmp_path / "MANIFEST.json", {"splits": {"train": {"sha256": "t" * 64}, "dev": {"sha256": "d" * 64},
                                                        "test": {"sha256": "x" * 64}}})
    monkeypatch.setattr(versao_dados, "versao_da_pasta", lambda pasta: "v2-texto-fonte")
    assert plano.dados_da_pasta(tmp_path) == DADOS                      # o TEST não entra na chave
    monkeypatch.setattr(versao_dados, "versao_da_pasta", lambda pasta: None)
    with pytest.raises(plano.ErroPlano, match="versão registrada"):
        plano.dados_da_pasta(tmp_path)
    monkeypatch.setattr(pretreino, "registro_de", lambda pasta: dict(PESOS))
    assert plano.pesos_da_pasta(tmp_path) == PESOS
    monkeypatch.setattr(pretreino, "registro_de", lambda pasta: None)
    with pytest.raises(plano.ErroPlano, match="PROVENIENCIA"):
        plano.pesos_da_pasta(tmp_path)


# ------------------------------------------------------------------ nomes oficiais
def test_nomes_oficiais_constam_do_protocolo():
    texto = (caminhos.CODIGO / "docs" / "protocolo" / "fase-biobertpt-seed42.md").read_text("utf-8")
    for nome in plano.NOMES_OFICIAIS:
        assert nome in texto, nome
    assert set(plano.NOMES_OFICIAIS.values()) == {"baseline", "baseline_calibrado", "restrito", "pair_aware",
                                                  "two_stage_e1", "two_stage_e2"}


def test_alias_so_para_nome_oficial_da_estrategia_certa():
    chave = _chave()
    plano.conferir_alias("baseline_biobertpt_seed42", chave)
    with pytest.raises(plano.ErroPlano, match="não é um nome oficial"):
        plano.conferir_alias("meu_experimento", chave)
    with pytest.raises(plano.ErroPlano, match="nome oficial da estratégia"):
        plano.conferir_alias("restrito_biobertpt_seed42", chave)
    with pytest.raises(plano.ErroPlano, match="semente"):
        plano.conferir_alias("baseline_biobertpt_seed42", _chave(config={**CFG, "seed": 43}))
    with pytest.raises(ids.ErroId):
        plano.conferir_alias("../baseline_biobertpt_seed42", chave)


# ------------------------------------------------------------------ plano
def test_primeira_execucao_oficial_e_depois_retomada_e_conclusao(tmp_path):
    chave, raiz = _chave(), tmp_path / "execucoes"
    p, pasta = _nova(raiz, chave, alias_oficial="baseline_biobertpt_seed42")
    assert p["nome"] == "baseline_biobertpt_seed42" and p["alias_oficial"] == p["nome"] and p["tentativa"] == 1
    assert p["run_id"] == ids.run_id("baseline", ids.config_id(chave), 1)
    assert proveniencia.ler(pasta)["alias_oficial"] == "baseline_biobertpt_seed42"
    assert plano.planejar(chave, raiz, alias_oficial="baseline_biobertpt_seed42")["acao"] == "iniciar_planejada"
    _como(pasta, "interrompida")
    r = plano.planejar(chave, raiz)
    assert r["acao"] == "retomar" and r["run_id"] == p["run_id"] and r["nome"] == "baseline_biobertpt_seed42"
    _como(pasta, "concluida")
    c = plano.planejar(chave, raiz)
    assert c["acao"] == "ja_concluida" and c["run_id"] == p["run_id"]


def test_nova_tentativa_nao_reaproveita_o_nome_oficial(tmp_path):
    chave, raiz = _chave(), tmp_path / "execucoes"
    p1, pasta = _nova(raiz, chave, alias_oficial="baseline_biobertpt_seed42")
    _como(pasta, "concluida")
    p2 = plano.planejar(chave, raiz, alias_oficial="baseline_biobertpt_seed42", nova_tentativa=True)
    assert p2["acao"] == "nova_execucao" and p2["tentativa"] == 2 and p2["alias_oficial"] is None
    assert p2["nome"] == p2["run_id"] != p1["run_id"] and any("run_id como pasta" in a for a in p2["avisos"])
    pasta2 = plano.preparar(p2, chave, raiz, codigo={"origem": "local"})
    assert pasta2.name == p2["run_id"] and (raiz / "baseline_biobertpt_seed42").is_dir()


def test_execucao_que_falhou_gera_a_proxima_tentativa(tmp_path):
    chave, raiz = _chave(), tmp_path / "execucoes"
    p1, pasta = _nova(raiz, chave)
    _como(pasta, "falhou")
    p2 = plano.planejar(chave, raiz)
    assert p2["acao"] == "nova_execucao" and p2["tentativa"] == 2 and p2["nome"] == p2["run_id"]


def test_nome_oficial_de_outra_configuracao_nunca_e_sobrescrito(tmp_path):
    raiz = tmp_path / "execucoes"
    _nova(raiz, _chave(), alias_oficial="baseline_biobertpt_seed42")
    outra = _chave(config=_Config(lr=3e-5))
    with pytest.raises(plano.ErroPlano, match="outra configuração"):
        plano.planejar(outra, raiz, alias_oficial="baseline_biobertpt_seed42")
    p = plano.planejar(outra, raiz)                                     # sem alias: run_id, tentativa 1
    assert p["tentativa"] == 1 and p["nome"] == p["run_id"]


def test_compatibilidade_nao_depende_de_caminhos_de_sessao(tmp_path):
    """A execução interrompida existe só no Drive, sob outra raiz local: é achada pela chave."""
    chave = _chave()
    sessao1, drive = tmp_path / "sessao1" / "content" / "execucoes", tmp_path / "MyDrive" / "RECLin-PT"
    p, pasta = _nova(sessao1, chave)
    _como(pasta, "interrompida")
    shutil.copytree(pasta, persistencia.raiz_das_execucoes(drive) / p["nome"])
    shutil.rmtree(sessao1)                                              # a sessão acabou
    sessao2 = tmp_path / "sessao2" / "outro_caminho" / "execucoes"
    r = plano.planejar(chave, sessao2, drive)
    assert r["acao"] == "retomar" and r["run_id"] == p["run_id"] and r["onde"] == "drive"
    assert [e["onde"] for e in plano.compativeis(chave, sessao2, drive)] == ["drive"]
    assert plano.compativeis(_chave(config=_Config(lr=3e-5)), sessao2, drive) == []


def test_o_config_id_vem_da_chave_gravada_e_nao_do_nome_da_pasta(tmp_path):
    chave, raiz = _chave(), tmp_path / "execucoes"
    p, pasta = _nova(raiz, chave)
    _como(pasta, "interrompida")
    pasta.rename(raiz / "restrito__000000000000__t1")                  # nome enganoso, registro intacto
    achadas = plano.compativeis(chave, raiz)
    assert [e["run_id"] for e in achadas] == [p["run_id"]] and achadas[0]["nome"] == "restrito__000000000000__t1"
    (raiz / "restrito__000000000000__t1" / "experimento.json").write_text("{", encoding="utf-8")
    assert plano.compativeis(chave, raiz) == []                         # registro ilegível é ignorado, não confiado


def test_varias_tentativas_ativas_escolhem_a_mais_recente_com_aviso(tmp_path):
    chave, raiz = _chave(), tmp_path / "execucoes"
    p1, a = _nova(raiz, chave)
    _como(a, "interrompida")
    p2 = plano.planejar(chave, raiz, nova_tentativa=True)
    b = plano.preparar(p2, chave, raiz, codigo={"origem": "local"})
    _como(b, "interrompida")
    r = plano.planejar(chave, raiz)
    assert r["acao"] == "retomar" and r["tentativa"] == 2 and any("mais de uma tentativa" in x for x in r["avisos"])


def test_plano_nao_escreve_e_preparar_nao_sobrescreve(tmp_path):
    chave, raiz = _chave(), tmp_path / "execucoes"
    p = plano.planejar(chave, raiz)
    assert not raiz.exists()                                            # planejar não cria nada
    pasta = plano.preparar(p, chave, raiz, codigo={"origem": "local"})
    assert sorted(x.name for x in pasta.iterdir()) == ["experimento.json"]
    with pytest.raises(plano.ErroPlano, match="nada é sobrescrito"):
        plano.preparar(p, chave, raiz, codigo={"origem": "local"})
    assert plano.preparar({**p, "acao": "retomar"}, chave, raiz, codigo={}) == pasta


# ------------------------------------------------------------------ identidade da infraestrutura
def test_infraestrutura_fora_da_identidade_global_v1():
    assert not ic.incluido("reclin/experimento/plano.py") and not ic.incluido("scripts/experimento.py")
    assert ic.incluido("reclin/treino/laco.py") and ic.incluido("scripts/treinar.py")
    reais = ic.arquivos(caminhos.CODIGO)
    assert not any(r.startswith("reclin/experimento/") or r == "scripts/experimento.py" for r in reais)


def test_ninguem_fora_da_infraestrutura_a_importa():
    """A infraestrutura só aponta para o núcleo, nunca o contrário: não entra no fecho de componente algum."""
    donos = [p for p in list((caminhos.CODIGO / "reclin").rglob("*.py")) + list((caminhos.CODIGO / "scripts").glob("*.py"))
             if "experimento" not in p.relative_to(caminhos.CODIGO).parts[:2] and p.name != "experimento.py"]
    for arquivo in donos:
        for no in ast.walk(ast.parse(arquivo.read_text("utf-8"))):
            nomes = ([a.name for a in no.names] if isinstance(no, ast.Import) else
                     [no.module or ""] + [f"{no.module}.{a.name}" for a in no.names] if isinstance(no, ast.ImportFrom) else [])
            assert not any(n.startswith("reclin.experimento") for n in nomes), arquivo


def test_identidade_propria_da_infraestrutura(tmp_path):
    for relativo, texto in {"reclin/experimento/a.py": "A = 1\n", "reclin/experimento/__pycache__/a.cpython-310.pyc": "x",
                            "scripts/experimento.py": "print(1)\n", "scripts/treinar.py": "print(2)\n",
                            "reclin/tarefa.py": "T = 1\n"}.items():
        (tmp_path / relativo).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relativo).write_bytes(texto.encode())
    antes = identidade.calcular(tmp_path)
    assert sorted(antes["arquivos"]) == ["reclin/experimento/a.py", "scripts/experimento.py"]
    assert antes["componente"] == "infraestrutura"
    ic_antes = ic.calcular(tmp_path)["identidade"]
    (tmp_path / "reclin/experimento/a.py").write_text("A = 2\n")
    editada = identidade.calcular(tmp_path)["identidade"]
    assert editada != antes["identidade"]                               # editar a infraestrutura muda a identidade dela
    assert ic.calcular(tmp_path)["identidade"] == ic_antes              # ...e não move a identidade global
    (tmp_path / "reclin/tarefa.py").write_text("T = 2\n")
    assert identidade.calcular(tmp_path)["identidade"] == editada       # editar o núcleo não move a da infraestrutura
    assert ic.calcular(tmp_path)["identidade"] != ic_antes


def test_proveniencia_grava_a_identidade_da_infraestrutura_sem_a_lista_de_arquivos():
    bloco = proveniencia.codigo_do_ambiente(caminhos.CODIGO)
    infra = bloco["infraestrutura"]
    assert set(infra) == {"componente", "algoritmo", "identidade", "n_arquivos"} and infra["n_arquivos"] >= 10
    assert infra["identidade"] == identidade.calcular(caminhos.CODIGO)["identidade"]
