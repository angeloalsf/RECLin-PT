"""Infraestrutura compartilhada dos notebooks (`reclin.experimento`, `scripts/experimento.py`).

Tudo em pastas temporárias e com objetos falsos: nenhum teste fala com a rede,
com o Colab, com o Drive ou com o Hugging Face, e nenhum treina ou baixa pesos.
Escritos na etapa 8 (Prompt 2) e **ainda não executados**: o autor os roda quando quiser.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

from reclin.experimento import (ambiente, codigo_colab, ids, persistencia, proveniencia, publicacao_hf, raiz,
                                segredos)
from reclin.util import caminhos
from reclin.util.io import gravar_json, ler_json

SHA = "a" * 40
COMMIT_PESOS = "b" * 40
CFG = {"seed": 42, "encoder": "biobertpt", "epochs": 3, "lr": 2e-5}


# ------------------------------------------------------------------ código por tarball

def _tarball(destino: Path, topo: str, extras=()):
    with tarfile.open(destino, "w:gz") as tar:
        def add(nome, dados: bytes):
            info = tarfile.TarInfo(f"{topo}/{nome}")
            info.size = len(dados)
            tar.addfile(info, io.BytesIO(dados))
        add("codigo/pyproject.toml", b'[project]\nname = "reclin"\n')
        add("codigo/reclin/__init__.py", b"")
        add("codigo/reclin/experimento/codigo_colab.py", b"# proprio\n")
        for nome, dados in extras:
            add(nome, dados)


@pytest.mark.parametrize("ruim", ["main", "v1.0", "abc1234", "A" * 40, "", None, "a" * 41])
def test_so_aceita_sha_completo(ruim):
    with pytest.raises(codigo_colab.ErroCodigoColab):
        codigo_colab.validar_sha(ruim)


def test_extracao_confere_o_topo_com_o_sha(tmp_path):
    _tarball(tmp_path / "ok.tar.gz", "dono-repo-aaaaaaa")
    info = codigo_colab.extrair_seguro(tmp_path / "ok.tar.gz", tmp_path / "saida", sha=SHA)
    assert info["n_arquivos"] == 3 and (tmp_path / "saida" / "codigo" / "pyproject.toml").is_file()
    _tarball(tmp_path / "outro.tar.gz", "dono-repo-bbbbbbb")
    with pytest.raises(codigo_colab.ErroCodigoColab, match="não corresponde"):
        codigo_colab.extrair_seguro(tmp_path / "outro.tar.gz", tmp_path / "saida2", sha=SHA)


def test_extracao_recusa_caminho_inseguro_e_ignora_links(tmp_path):
    with tarfile.open(tmp_path / "ruim.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo("dono-repo-aaaaaaa/../fora.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(codigo_colab.ErroCodigoColab, match="inseguro"):
        codigo_colab.extrair_seguro(tmp_path / "ruim.tar.gz", tmp_path / "s", sha=SHA)
    with tarfile.open(tmp_path / "link.tar.gz", "w:gz") as tar:
        elo = tarfile.TarInfo("dono-repo-aaaaaaa/elo")
        elo.type, elo.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(elo)
        info = tarfile.TarInfo("dono-repo-aaaaaaa/a.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    res = codigo_colab.extrair_seguro(tmp_path / "link.tar.gz", tmp_path / "s2", sha=SHA)
    assert res["ignorados"] == 1 and not (tmp_path / "s2" / "elo").exists()


def test_preparar_codigo_reaproveita_a_extracao(tmp_path, monkeypatch):
    chamadas = []

    def falso(repositorio, sha, destino, *, token=None, timeout=0):
        chamadas.append(token)
        _tarball(Path(destino), "dono-repo-aaaaaaa")
        return {"sha256": "0" * 64, "bytes": 1}

    monkeypatch.setattr(codigo_colab, "baixar_tarball", falso)
    primeiro = codigo_colab.preparar_codigo("dono/repo", SHA, destino=tmp_path, token="segredo")
    segundo = codigo_colab.preparar_codigo("dono/repo", SHA, destino=tmp_path, token="segredo")
    assert len(chamadas) == 1 and primeiro["reaproveitado"] is False and segundo["reaproveitado"] is True
    registro = (Path(primeiro["raiz"]) / codigo_colab.ARQUIVO_REVISAO).read_text(encoding="utf-8")
    assert "segredo" not in registro and SHA in registro


def test_bootstrap_diferente_da_arvore_extraida_e_recusado(tmp_path, monkeypatch):
    monkeypatch.setattr(codigo_colab, "baixar_tarball",
                        lambda r, s, d, **k: (_tarball(Path(d), "dono-repo-aaaaaaa"), {"sha256": "0" * 64, "bytes": 1})[1])
    baixado = tmp_path / "baixado.py"
    baixado.write_bytes(b"# outro\n")
    with pytest.raises(codigo_colab.ErroCodigoColab, match="difere"):
        codigo_colab.preparar_codigo("dono/repo", SHA, destino=tmp_path / "d", arquivo_bootstrap=baixado)


# ------------------------------------------------------------------ raiz

def test_localizar_e_ativar_o_codigo(tmp_path, monkeypatch):
    codigo = tmp_path / "x" / "codigo"
    (codigo / "reclin").mkdir(parents=True)
    (codigo / "reclin" / "__init__.py").write_text("")
    (codigo / "pyproject.toml").write_text('[project]\nname = "reclin"\n')
    monkeypatch.chdir(tmp_path)
    assert raiz.localizar_codigo(tmp_path / "x") == codigo.resolve()
    monkeypatch.setattr(sys, "path", list(sys.path))
    ambiente_sub = raiz.ativar(codigo)
    assert sys.path[0] == str(codigo.resolve()) and ambiente_sub["PYTHONPATH"].split(":")[0] == str(codigo.resolve())
    with pytest.raises(raiz.ErroRaiz):
        raiz.ativar(tmp_path)


# ------------------------------------------------------------------ segredos

def test_credencial_nao_vaza_por_repr_json_ou_pickle():
    c = segredos.Credencial("HF_TOKEN", "hf_" + "x" * 30, "teste")
    assert "xxx" not in repr(c) and "xxx" not in str(c) and "xxx" not in f"{c}"
    with pytest.raises(TypeError):
        json.dumps({"c": c})
    import pickle
    with pytest.raises(TypeError):
        pickle.dumps(c)


def test_limpar_e_contem_segredo():
    token = "ghp_" + "a" * 36
    assert token not in segredos.limpar(f"erro com {token}")
    assert segredos.contem_segredo({"a": ["hf_" + "z" * 25]})
    assert segredos.contem_segredo({"a": "valor-secreto-simples"}, "valor-secreto-simples")
    assert not segredos.contem_segredo({"a": 1, "b": "texto normal"})
    with pytest.raises(segredos.ErroSegredo):
        segredos.exigir_sem_segredo({"x": token})


def test_so_os_segredos_da_operacao(monkeypatch):
    monkeypatch.setattr(segredos.colab, "em_colab", lambda: False)
    assert segredos.para_operacao("treinar", ambiente={}) == {}
    assert segredos.para_operacao("baixar_codigo", ambiente={}) == {segredos.GITHUB: None}
    with pytest.raises(segredos.ErroSegredo, match="Secrets do Colab"):
        segredos.para_operacao("publicar_hf", ambiente={})
    cred = segredos.para_operacao("publicar_hf", ambiente={"HF_TOKEN": "hf_" + "k" * 30})[segredos.HUGGING_FACE]
    assert cred.valor().startswith("hf_") and "kkk" not in repr(cred)
    with pytest.raises(segredos.ErroSegredo):
        segredos.obter("OUTRO_TOKEN")


# ------------------------------------------------------------------ ambiente

def test_requisitos_do_projeto_real():
    req = ambiente.requisitos_do_projeto(caminhos.CODIGO)
    assert {"torch", "transformers", "numpy", "scipy"} <= set(req["minimos"])
    assert req["fixados"]["torch"] == "2.11.0" and req["python"]


def test_plano_minimo_nao_reinstala_o_que_satisfaz_o_minimo():
    req = {"python": "3.10", "minimos": {"torch": "2.11", "numpy": "1.24"},
           "fixados": {"torch": "2.11.0", "numpy": "2.1.3", "huggingface-hub": "1.29.0"}}
    instalado = {"torch": "2.12.0", "numpy": "2.0.2", "huggingface-hub": None}
    minimo = ambiente.plano_instalacao(req, modo="minimo", instalado=instalado, modulos_carregados={})
    assert minimo["instalar"] == ["huggingface-hub==1.29.0"] and len(minimo["avisos"]) == 2
    fixado = ambiente.plano_instalacao(req, modo="fixado", instalado=instalado, modulos_carregados={"torch": 1})
    assert "torch==2.11.0" in fixado["instalar"] and fixado["reiniciar"] is True
    assert ambiente.instalar(minimo)["executado"] is False          # dry-run por padrão


def test_gpu_referencia_e_permitidas():
    assert ambiente.classificar_gpu("Tesla T4")["classe"] == "referencia"
    assert ambiente.classificar_gpu("NVIDIA L4")["classe"] == "permitida"
    assert "variar" in ambiente.classificar_gpu("NVIDIA A100-SXM4-40GB")["aviso"]
    assert ambiente.classificar_gpu("RTX 4090")["classe"] == "nao_prevista"
    assert ambiente.classificar_gpu(None)["classe"] == "sem_gpu"


def test_retomada_bloqueia_por_torch_e_gpu_mas_nao_por_scipy():
    base = {"python": "3.12.1", "versoes": {"torch": "2.11.0", "scipy": "1.16.3", "numpy": "2.1.3"}, "gpu": "Tesla T4"}
    igual = ambiente.comparar_para_retomada(base, base)
    assert igual["pode_retomar"] and not igual["avisos"]
    outro = {**base, "versoes": {**base["versoes"], "scipy": "1.17.0"}}
    assert ambiente.comparar_para_retomada(base, outro)["pode_retomar"]
    assert ambiente.comparar_para_retomada(base, outro)["avisos"]
    assert not ambiente.comparar_para_retomada(base, {**base, "versoes": {**base["versoes"], "torch": "2.12.0"}})["pode_retomar"]
    assert not ambiente.comparar_para_retomada(base, {**base, "gpu": "NVIDIA L4"})["pode_retomar"]
    assert not ambiente.comparar_para_retomada(None, base)["pode_retomar"]
    assert ambiente.comparar_para_reaproveitar(base, {**base, "gpu": "NVIDIA L4"})["avisos"]


# ------------------------------------------------------------------ ids

def _chave(**mudar):
    padrao = dict(estrategia="baseline", config=dict(CFG), estrategia_config=None,
                  pesos={"repositorio": "pucpr/biobertpt-all", "commit": COMMIT_PESOS, "hash_manifesto": "m" * 64},
                  dados={"versao": "v2-texto-fonte", "particoes": {"train": "t" * 64, "dev": "d" * 64}},
                  codigo={"escopo": "global-v1", "algoritmo": "sha256-manifesto-v1", "identidade": "i" * 64})
    padrao.update(mudar)
    return ids.chave_configuracao(**padrao)


def test_config_id_estavel_sem_caminhos():
    a = _chave()
    b = _chave(config={"lr": 2e-5, "epochs": 3, "encoder": "biobertpt", "seed": 42, "modelo": "/qualquer/lugar"})
    assert ids.config_id(a) == ids.config_id(b) and len(ids.config_id(a)) == 12
    assert ids.config_id(a) != ids.config_id(_chave(config={**CFG, "lr": 3e-5}))
    with pytest.raises(ids.ErroId, match="caminhos absolutos"):
        _chave(estrategia_config={"x": "/content/modelo"})
    with pytest.raises(ids.ErroId):
        _chave(pesos={"repositorio": "r", "commit": "", "hash_manifesto": "m"})


def test_run_id_e_tentativas():
    cid = ids.config_id(_chave())
    run = ids.run_id("baseline", cid, 2)
    assert run == f"baseline__{cid}__t2" and ids.decompor_run_id(run)["tentativa"] == 2
    assert ids.proxima_tentativa([run, f"baseline__{cid}__t1", "baseline_biobertpt_seed42"], "baseline", cid) == 3
    assert ids.proxima_tentativa([], "baseline", cid) == 1
    for ruim in ("../x", "a b", "", "x" * 200):
        with pytest.raises(ids.ErroId):
            ids.validar_nome(ruim)


# ------------------------------------------------------------------ proveniência

def _execucao(raiz_local: Path, tentativa=1, concluida=True, com_ultimo=False):
    chave = _chave()
    cid = ids.config_id(chave)
    run = ids.run_id("baseline", cid, tentativa)
    pasta = raiz_local / run
    gravar_json(pasta / "config.json", {
        "nome": run, "estrategia": "baseline", "conjuntos": {},
        "codigo": {"algoritmo": "sha256-manifesto-v1", "identidade": "i" * 64, "arquivos": {}},
        "config": {"estrategia": "baseline", "config": dict(CFG), "modelo": "/content/pesos", "particoes": {},
                   "proveniencia_pesos": {"commit": COMMIT_PESOS, "hash_manifesto": "m" * 64}}})
    (pasta / "checkpoints" / "melhor_modelo").mkdir(parents=True)
    (pasta / "checkpoints" / "melhor_modelo" / "model.safetensors").write_bytes(b"pesos" * 10)
    if com_ultimo:
        (pasta / "checkpoints" / "ultimo.pt").write_bytes(b"estado" * 10)
        gravar_json(pasta / "checkpoints" / "ultimo.json", {"posicao": {"passo_global": 10}})
    if concluida:
        gravar_json(pasta / "treino.json", {"concluido": True, "melhor_epoca": 2, "dev_history": [{}, {}], "sessoes": [
            {"ambiente": {"python": "3.12.1", "versoes": {"torch": "2.11.0"}, "gpu": "Tesla T4"}}]})
        gravar_json(pasta / "metricas.json", {"dev": {"macro_f1": 0.5}})
    return pasta, run, chave


def test_registro_de_proveniencia_ciclo_completo(tmp_path):
    pasta, run, chave = _execucao(tmp_path)
    proveniencia.criar(pasta, run=run, chave=chave, codigo={"origem": "local", "revisao": None},
                       ambiente={"python": "3.12.1"}, alias_oficial="baseline_biobertpt_seed42")
    with pytest.raises(proveniencia.ErroProveniencia, match="já existe"):
        proveniencia.criar(pasta, run=run, chave=chave, codigo={})
    reg = proveniencia.vincular_config(pasta)
    assert reg["sha256_config"] and reg["codigo"]["identidade_global"]["identidade"] == "i" * 64
    proveniencia.mudar_estado(pasta, "iniciada")
    proveniencia.mudar_estado(pasta, "concluida", "fim")
    with pytest.raises(proveniencia.ErroProveniencia, match="não prevista"):
        proveniencia.mudar_estado(pasta, "iniciada")
    assert proveniencia.registrar_metricas(pasta)["metricas"]["treino"]["melhor_epoca"] == 2
    assert proveniencia.conferir(pasta) == []
    gravar_json(pasta / "config.json", {**ler_json(pasta / "config.json"), "extra": 1})
    assert any("config.json mudou" in p for p in proveniencia.conferir(pasta))


def test_registro_nao_aceita_segredo_nem_run_id_incoerente(tmp_path):
    pasta, run, chave = _execucao(tmp_path)
    with pytest.raises(segredos.ErroSegredo):
        proveniencia.criar(pasta, run=run, chave=chave, codigo={"origem": "x", "nota": "ghp_" + "a" * 36})
    with pytest.raises(proveniencia.ErroProveniencia, match="não corresponde"):
        proveniencia.criar(pasta, run=ids.run_id("baseline", "0" * 12, 1), chave=chave, codigo={})


def test_estado_derivado_dos_arquivos(tmp_path):
    vazio = tmp_path / "v"
    assert proveniencia.derivar_estado(vazio)["estado"] == "planejada"
    pasta, _, _ = _execucao(tmp_path, concluida=False, com_ultimo=True)
    d = proveniencia.derivar_estado(pasta)
    assert d["estado"] == "interrompida" and d["ambiguo"] is True
    pasta2, _, _ = _execucao(tmp_path / "b", concluida=True)
    assert proveniencia.derivar_estado(pasta2)["estado"] == "concluida"


# ------------------------------------------------------------------ persistência

def test_sincronizar_e_restaurar_com_verificacao(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    pasta, run, _ = _execucao(local, concluida=False, com_ultimo=True)
    r1 = persistencia.sincronizar(pasta, drive)
    assert r1["copiados"] == r1["n_arquivos"] and (drive / "execucoes" / run / persistencia.MANIFESTO).is_file()
    assert not list((drive / "execucoes" / run).rglob("*" + persistencia.SUFIXO_TEMPORARIO))
    r2 = persistencia.sincronizar(pasta, drive)
    assert r2["copiados"] == 0 and r2["inalterados"] == r1["n_arquivos"]
    assert persistencia.verificar_espelho(drive / "execucoes" / run) == []
    (drive / "execucoes" / run / "checkpoints" / "ultimo.pt").write_bytes(b"corrompido")
    assert persistencia.verificar_espelho(drive / "execucoes" / run)
    novo = tmp_path / "novo_local"
    with pytest.raises(persistencia.ErroSincronizacao):
        persistencia.restaurar(drive, run, novo)                   # o Drive foi corrompido? a restauração confere o tamanho
    persistencia.sincronizar(pasta, drive)
    r3 = persistencia.restaurar(drive, run, novo)
    assert r3["copiados"] == r1["n_arquivos"] and (novo / run / "checkpoints" / "ultimo.pt").read_bytes() == b"estado" * 10


def test_restauracao_recusa_conflito_e_espelho_sem_manifesto(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    pasta, run, _ = _execucao(local)
    persistencia.sincronizar(pasta, drive)
    (pasta / "metricas.json").write_text("{}")
    with pytest.raises(persistencia.ErroSincronizacao, match="diferentes"):
        persistencia.restaurar(drive, run, local)
    assert persistencia.restaurar(drive, run, local, substituir=True)["substituidos"] == ["metricas.json"]
    (drive / "execucoes" / run / persistencia.MANIFESTO).unlink()
    with pytest.raises(persistencia.ErroSincronizacao, match="SYNC.json"):
        persistencia.restaurar(drive, run, tmp_path / "outro")


def test_nao_regrava_o_drive_com_estado_mais_antigo(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    pasta, run, _ = _execucao(local, concluida=False, com_ultimo=True)
    persistencia.sincronizar(pasta, drive)
    gravar_json(drive / "execucoes" / run / "checkpoints" / "ultimo.json", {"posicao": {"passo_global": 99}})
    with pytest.raises(persistencia.ErroSincronizacao, match="mais avançado"):
        persistencia.sincronizar(pasta, drive)


def test_listar_e_diagnosticar_retomada(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    pasta, run, _ = _execucao(local, concluida=False, com_ultimo=True)
    persistencia.sincronizar(pasta, drive)
    [linha] = persistencia.listar(local, drive)
    assert linha["nome"] == run and linha["situacao"] == "em_dia" and linha["retomavel"] and linha["espelho_completo"]
    amb = {"python": "3.12.1", "versoes": {"torch": "2.11.0"}, "gpu": "Tesla T4"}
    treino = {"concluido": False, "sessoes": [{"ambiente": amb}]}
    gravar_json(pasta / "treino.json", treino)
    ok = persistencia.diagnosticar_retomada(run, local, drive, ambiente_atual=amb, identidade_atual="i" * 64)
    assert ok["pode_retomar"] and ok["fonte"] == "local"
    errado = persistencia.diagnosticar_retomada(run, local, drive, ambiente_atual={**amb, "gpu": "NVIDIA L4"},
                                                identidade_atual="j" * 64)
    assert not errado["pode_retomar"] and len(errado["bloqueios"]) == 2
    so_drive = persistencia.diagnosticar_retomada(run, tmp_path / "vazio", drive, ambiente_atual=amb)
    assert so_drive["fonte"] == "drive"


# ------------------------------------------------------------------ publicação no Hugging Face

class _Op:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo, self.fonte = path_in_repo, path_or_fileobj


class _HfFalso:
    """Imita a parte usada de `huggingface_hub`, com um 'remoto' em memória."""

    def __init__(self, pasta: Path, privado=True, existentes=(), existe=True):
        self.pasta, self.privado, self.existentes, self.operacoes, self.ordens = pasta, privado, list(existentes), [], []
        self.existe = existe
        self.CommitOperationAdd = _Op
        self.HfApi = lambda: self

    def create_repo(self, repo, **k):
        self.existe = True
        self.ordens.append(("create_repo", k))

    def model_info(self, repo, **k):
        if not self.existe:
            class RepositoryNotFoundError(Exception):
                pass
            raise RepositoryNotFoundError("não existe")
        return types.SimpleNamespace(private=self.privado)
    def list_repo_files(self, repo, **k): return self.existentes

    def create_commit(self, repo, operacoes, **k):
        self.operacoes = operacoes
        self.ordens.append(("create_commit", k))
        return types.SimpleNamespace(oid="c" * 40)

    def get_paths_info(self, repo, caminhos, **k):
        tamanhos = {o.path_in_repo: (len(o.fonte.getvalue()) if isinstance(o.fonte, io.BytesIO)
                                     else Path(o.fonte).stat().st_size) for o in self.operacoes}
        return [types.SimpleNamespace(path=c, size=tamanhos[c], lfs=None) for c in caminhos if c in tamanhos]

    def hf_hub_download(self, repo, filename, **k):
        return str(self.pasta / filename.split("/", 2)[2])


def _pronta_para_publicar(tmp_path):
    pasta, run, chave = _execucao(tmp_path)
    (pasta / "checkpoints" / "ultimo.pt").write_bytes(b"u")
    (pasta / "predicoes_test.json").write_text("{}")
    proveniencia.criar(pasta, run=run, chave=chave, codigo={"origem": "local", "revisao": None}, estado="planejada")
    proveniencia.vincular_config(pasta)
    proveniencia.mudar_estado(pasta, "iniciada")
    proveniencia.mudar_estado(pasta, "concluida")
    return pasta, run


def test_plano_exclui_ultimo_e_test_e_nao_usa_a_rede(tmp_path, monkeypatch):
    pasta, run = _pronta_para_publicar(tmp_path)
    monkeypatch.setattr(publicacao_hf, "_importar_hf", lambda: pytest.fail("o plano não deve importar o Hugging Face"))
    plano = publicacao_hf.publicar(pasta, "usuario/artefatos")            # sem confirmar
    assert plano["executado"] is False and plano["pode_publicar"], plano["problemas"]
    assert "checkpoints/ultimo.pt" not in plano["arquivos"] and "predicoes_test.json" not in plano["arquivos"]
    assert {e["arquivo"] for e in plano["excluidos"]} == {"checkpoints/ultimo.pt", "predicoes_test.json"}
    assert plano["prefixo"] == f"runs/{run}"
    com_tudo = publicacao_hf.planejar(pasta, "usuario/artefatos", incluir_ultimo=True, incluir_test=True)
    assert "checkpoints/ultimo.pt" in com_tudo["arquivos"] and "predicoes_test.json" in com_tudo["arquivos"]


def test_plano_recusa_execucao_nao_concluida(tmp_path):
    pasta, run, chave = _execucao(tmp_path, concluida=False)
    proveniencia.criar(pasta, run=run, chave=chave, codigo={})
    plano = publicacao_hf.planejar(pasta, "usuario/artefatos")
    assert not plano["pode_publicar"] and any("concluídas" in p for p in plano["problemas"])


def test_publicar_exige_confirmacao_repositorio_privado_e_prefixo_novo(tmp_path, monkeypatch):
    pasta, run = _pronta_para_publicar(tmp_path)
    cred = segredos.Credencial("HF_TOKEN", "hf_" + "t" * 30, "teste")
    monkeypatch.setattr(segredos, "verificar_escrita_hf", lambda c: {"usuario": "u", "papel": "write"})
    publico = _HfFalso(pasta, privado=False)
    monkeypatch.setattr(publicacao_hf, "_importar_hf", lambda: publico)
    with pytest.raises(publicacao_hf.ErroPublicacao, match="não é privado"):
        publicacao_hf.publicar(pasta, "usuario/artefatos", credencial=cred, confirmar=True)
    ocupado = _HfFalso(pasta, existentes=[f"runs/{run}/config.json"])
    monkeypatch.setattr(publicacao_hf, "_importar_hf", lambda: ocupado)
    with pytest.raises(publicacao_hf.ErroPublicacao, match="nunca são sobrescritas"):
        publicacao_hf.publicar(pasta, "usuario/artefatos", credencial=cred, confirmar=True)
    assert not ocupado.operacoes                                        # nada foi enviado


def test_publicacao_completa_registra_referencia_sem_token(tmp_path, monkeypatch):
    pasta, run = _pronta_para_publicar(tmp_path)
    cred = segredos.Credencial("HF_TOKEN", "hf_" + "t" * 30, "teste")
    monkeypatch.setattr(segredos, "verificar_escrita_hf", lambda c: {"usuario": "u", "papel": "write"})
    falso = _HfFalso(pasta, existe=False)                              # o repositório ainda não existe: é criado privado
    monkeypatch.setattr(publicacao_hf, "_importar_hf", lambda: falso)
    resultado = publicacao_hf.publicar(pasta, "usuario/artefatos", credencial=cred, confirmar=True)
    assert resultado["executado"] and resultado["referencia"]["verificada"] is True
    assert [o for o, _ in falso.ordens] == ["create_repo", "create_commit"]
    assert falso.ordens[0][1]["private"] is True
    destinos = [o.path_in_repo for o in falso.operacoes]
    assert f"runs/{run}/{publicacao_hf.MANIFESTO_PUBLICACAO}" in destinos
    assert not any(d.endswith("ultimo.pt") or d.endswith("predicoes_test.json") for d in destinos)
    registro = proveniencia.ler(pasta)
    assert registro["publicacoes"][0]["commit"] == "c" * 40 and registro["checkpoints"]["hf"]["prefixo"] == f"runs/{run}"
    assert "hf_" + "t" * 30 not in (pasta / proveniencia.ARQUIVO).read_text(encoding="utf-8")


def test_repositorio_existente_nao_e_recriado(tmp_path, monkeypatch):
    pasta, run = _pronta_para_publicar(tmp_path)
    cred = segredos.Credencial("HF_TOKEN", "hf_" + "t" * 30, "teste")
    monkeypatch.setattr(segredos, "verificar_escrita_hf", lambda c: {"usuario": "u", "papel": "write"})
    falso = _HfFalso(pasta, existe=True)
    monkeypatch.setattr(publicacao_hf, "_importar_hf", lambda: falso)
    publicacao_hf.publicar(pasta, "usuario/artefatos", credencial=cred, confirmar=True)
    assert [o for o, _ in falso.ordens] == ["create_commit"]
