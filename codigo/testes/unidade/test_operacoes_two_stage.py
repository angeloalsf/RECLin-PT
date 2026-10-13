"""Notebook `reclin_two_stage.ipynb` e a extensão de `operacoes` para estágios (Two-Stage).

Estáticos ou com diretórios temporários: nada de rede, GPU, pesos, corpus, treino, TEST nem Git. Escritos na
etapa 8.4 e **não executados** pelo autor da etapa (a suíte só roda por autorização)."""
from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

from reclin.estrategias import two_stage
from reclin.experimento import operacoes as ops
from reclin.experimento import plano, publicacao_hf, segredos, versionamento
from reclin.util import caminhos

NOTEBOOK = caminhos.CODIGO / "notebooks" / "reclin_two_stage.ipynb"


def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _celulas() -> dict[str, str]:
    return {c["id"]: "".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"}


def _codigo() -> str:
    return "\n".join(_celulas().values())


def _espec(estagio: int) -> ops.Especificacao:
    return ops.Especificacao(estrategia=two_stage.ESTRATEGIA_ESTAGIO[estagio], script="two_stage.py",
                             config_cls=two_stage.CONFIGS[estagio], estagio=estagio)


def _ctx(tmp_path, estagio=1, **kw):
    return ops.Contexto(codigo=Path("/c"), raiz_local=tmp_path / "local", raiz_drive=tmp_path / "drive",
                        espec=_espec(estagio), pasta_pesos=Path("/p"), pasta_particoes=Path("/d"), **kw)


# ------------------------------------------------------------------ notebook
def test_notebook_valido_compila_sem_saidas_e_com_ids_unicos():
    nb = _nb()
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            fonte = "\n".join("pass" if l.lstrip().startswith("!") else l for l in "".join(c["source"]).split("\n"))
            ast.parse(fonte, c["id"])


def test_secoes_na_ordem():
    titulos = [re.match(r"## (\d+)\.", "".join(c["source"])) for c in _nb()["cells"] if c["cell_type"] == "markdown"]
    assert [int(t.group(1)) for t in titulos if t] == list(range(1, 13))


def test_estagios_treinam_com_os_valores_do_protocolo_e_so_a_celula_4_instala():
    c = _celulas()
    assert "'epochs': 3" in c["c7"] and "'epochs': 10" in c["c8"]
    assert "_treinar(1," in c["c7"] and "_treinar(2," in c["c8"]
    assert [k for k, v in c.items() if "pip install" in v] == ["c4"]
    assert "NOVA_EXECUCAO = False" in c["c7"] and "NOVA_EXECUCAO_E2 = False" in c["c8"]
    assert 'estagio=estagio' in c["c5a"] and "registro_estagio(estagio)" in c["c5a"]


def test_composicao_exige_confirmacao_e_estagios_do_protocolo():
    c9 = _celulas()["c9"]
    assert "CONFIRMO_COMPOR = False" in c9
    assert "!= 'PROTOCOLO'" in c9 and "operacoes.OP_COMPOR" in c9


def test_nenhuma_celula_avalia_ou_le_o_test():
    codigo = _codigo()
    for proibido in ("avaliar-test", "avaliar_test", "predicoes_test", "avaliacoes_test", "--particao test", "OP_TEST"):
        assert proibido not in codigo, proibido


def test_publicacao_so_com_confirmacao_explicita():
    c = _celulas()
    assert "PUBLICAR_NO_GITHUB = False" in c["c11"] and "BRANCH_RESULTADOS = 'resultados-colab'" in c["c11"]
    assert "perfil=perfil" in c["c11"] and "'estagio'" in c["c11"] and "'composicao'" in c["c11"]
    assert "PUBLICAR_ETAPA1 = False" in c["c12"] and "PUBLICAR_ETAPA2 = False" in c["c12"]
    assert "frase_de_publicacao" in c["c12"] and "etapa{_n}" in c["c12"]
    assert "OP_PUBLICAR" not in c["c11"] and "composi" not in c["c12"].lower()      # a composição não vai ao HF


def test_tokens_so_onde_precisam():
    codigo = _codigo()
    assert not re.search(r"(ghp_|github_pat_|hf_[A-Za-z0-9]{20})", codigo)
    assert [k for k, v in _celulas().items() if "GITHUB_PAT" in v] == ["c1", "c11"]
    assert "os.environ[" not in codigo


# ------------------------------------------------------------------ operações
def test_argumentos_de_estagio_usam_o_subcomando_treinar_e_flags_reais(tmp_path):
    two = (caminhos.CODIGO / "scripts" / "two_stage.py").read_text(encoding="utf-8")
    for estagio in (1, 2):
        ctx = _ctx(tmp_path, estagio)
        cfg = ops.config_da_operacao(ctx.espec, {})
        args = ops.argumentos_de_treino(ctx, cfg, "n", retomar=True)
        assert args[:3] == ["treinar", "--estagio", str(estagio)] and "--estrategia" not in args
        assert args[args.index("--epochs") + 1] == str(two_stage.CONFIGS[estagio]().epochs)
        por_campo = {"--" + f.name.replace("_", "-") for f in dataclasses.fields(two_stage.CONFIGS[estagio])}
        assert por_campo <= set(args) and "--retomar" in args
        for flag in ("--estagio", "--modelo", "--nome", "--saida", "--particoes", "--checkpoint-a-cada", "--retomar"):
            assert f'"{flag}"' in two


def test_baseline_continua_com_estrategia_e_sem_subcomando(tmp_path):
    from reclin.config import Config
    espec = ops.Especificacao(estrategia="baseline", script="treinar.py", config_cls=Config)
    ctx = ops.Contexto(codigo=Path("/c"), raiz_local=tmp_path, raiz_drive=None, espec=espec,
                       pasta_pesos=Path("/p"), pasta_particoes=Path("/d"))
    args = ops.argumentos_de_treino(ctx, ops.config_da_operacao(espec, {}), "n", retomar=False)
    assert args[:2] == ["--estrategia", "baseline"] and "treinar" not in args


def test_epocas_do_protocolo_por_estagio():
    assert two_stage.ConfigEstagio1().epochs == 3 and two_stage.ConfigEstagio2().epochs == 10
    assert _espec(1).alias_oficial == "twostage_e1_biobertpt_seed42"
    assert _espec(2).alias_oficial == "twostage_e2_biobertpt_seed42"
    assert two_stage.nome_execucao(1, two_stage.ConfigEstagio1(encoder="biobertpt", seed=42)) == _espec(1).alias_oficial
    assert two_stage.nome_composicao(two_stage.ConfigEstagio1(encoder="biobertpt", seed=42)) == "twostage_biobertpt_seed42"


def test_nomes_oficiais_dos_estagios_no_protocolo_e_conferidos():
    texto = (caminhos.CODIGO / "docs" / "protocolo" / "fase-biobertpt-seed42.md").read_text("utf-8")
    for nome in ("twostage_e1_biobertpt_seed42", "twostage_e2_biobertpt_seed42", "twostage_biobertpt_seed42"):
        assert nome in texto
    chave = {"estrategia": "two_stage_e1", "config": {"encoder": "biobertpt", "seed": 42}}
    plano.conferir_alias("twostage_e1_biobertpt_seed42", chave)
    with pytest.raises(plano.ErroPlano, match="estratégia"):
        plano.conferir_alias("twostage_e2_biobertpt_seed42", chave)
    with pytest.raises(plano.ErroPlano, match="semente"):
        plano.conferir_alias("twostage_e1_biobertpt_seed42", {**chave, "config": {"encoder": "biobertpt", "seed": 43}})


def test_protocolo_nao_da_mais_2_epocas_ao_two_stage():
    texto = (caminhos.CODIGO / "docs" / "protocolo" / "fase-biobertpt-seed42.md").read_text("utf-8")
    assert "o limite do Two-Stage é dado como 2" not in texto and "padronizar o orçamento" not in texto
    assert "A D5 não se aplica ao Two-Stage" in texto


def test_compor_esta_no_despacho_e_recusa_sem_estagios_ou_sem_drive(tmp_path, capsys):
    assert ops.OP_COMPOR in ops.OPERACOES and ops.OP_COMPOR in ops._DESPACHO
    assert ops.executar(ops.OP_COMPOR, _ctx(tmp_path)) is None
    assert "ESTAGIO1" in capsys.readouterr().out
    sem_drive = ops.Contexto(codigo=Path("/c"), raiz_local=tmp_path, raiz_drive=None, espec=_espec(1),
                             pasta_pesos=Path("/p"), pasta_particoes=Path("/d"), estagio1="a", estagio2="b")
    assert ops.executar(ops.OP_COMPOR, sem_drive) is None
    assert "Drive" in capsys.readouterr().out


def test_compor_recusa_estagio_ausente_do_disco_local(tmp_path, capsys):
    ctx = _ctx(tmp_path, estagio1="a", estagio2="b")
    assert ops.executar(ops.OP_COMPOR, ctx) is None
    assert "não está no disco local" in capsys.readouterr().out


# ------------------------------------------------------------------ GitHub: perfis estágio e composição
def _escrever(pasta: Path, **arquivos):
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, conteudo in arquivos.items():
        (pasta / nome).write_text(json.dumps(conteudo), encoding="utf-8")
    return pasta


SIDECAR = {"model": "two_stage(a + b)", "seed": 42, "labels": ["negation_of", "associated_with", "no_relation"],
           "y_true": [0, 1, 2], "y_pred": [0, 2, 2], "estagios": ["twostage_e1_biobertpt_seed42", "twostage_e2_biobertpt_seed42"],
           "limiar_tau": 0.5, "indices_estagio2": [0], "preds_estagio2": [0], "probs_estagio1": [[0.9, 0.1], [0.2, 0.8]]}


def _composicao(tmp_path, **troca):
    base = {"limiar.json": {"limiar": 0.5, "tabela": [{"limiar": 0.5, "f1_negation_of": 0.4, "macro_f1": 0.5}]},
            "metricas.json": {"dev": {"macro_f1": 0.5}, "dev_diagnostico_estagios": {"n_roteados_ao_estagio2": 1}},
            "predicoes_dev.json": SIDECAR}
    base.update(troca)
    return _escrever(tmp_path / "twostage_biobertpt_seed42", **base)


def test_composicao_salva_so_limiar_metricas_e_predicoes_do_dev(tmp_path):
    pasta = _composicao(tmp_path)
    (pasta / "checkpoints").mkdir()
    (pasta / "checkpoints" / "ultimo.pt").write_bytes(b"x")
    p = versionamento.plano_de_salvamento(pasta, "composicao")
    assert p["problemas"] == [] and sorted(p["arquivos"]) == ["limiar.json", "metricas.json", "predicoes_dev.json"]
    assert any(e["arquivo"] == "checkpoints/ultimo.pt" for e in p["excluidos"])


@pytest.mark.parametrize("troca,trecho", [
    ({"metricas.json": {"dev": {}, "test": {}}}, "chave 'test'"),
    ({"limiar.json": {"limiar": 0.5, "tabela": [], "validacao": {"epocas": 1}}}, "validação técnica"),
    ({"limiar.json": {"tabela": []}}, "τ não está congelado"),
    ({"predicoes_dev.json": {**SIDECAR, "texto": "x" * 300}}, "texto"),
    ({"predicoes_dev.json": {**SIDECAR, "frases": ["a", "b"]}}, "texto"),
    ({"predicoes_dev.json": {**SIDECAR, "estagios": ["texto com espaços e frase"]}}, "estagios"),
])
def test_composicao_recusa_test_validacao_e_texto(tmp_path, troca, trecho):
    p = versionamento.plano_de_salvamento(_composicao(tmp_path, **troca), "composicao")
    assert any(trecho in m for m in p["problemas"]), p["problemas"]


def test_composicao_recusa_artefato_do_test_e_segredo(tmp_path):
    pasta = _composicao(tmp_path)
    (pasta / "predicoes_test.json").write_text("{}", encoding="utf-8")
    assert any("TEST" in m for m in versionamento.plano_de_salvamento(pasta, "composicao")["problemas"])
    pasta2 = _composicao(tmp_path / "b", **{"metricas.json": {"dev": {}, "nota": "ghp_" + "a" * 36}})
    assert any("segredo" in m for m in versionamento.plano_de_salvamento(pasta2, "composicao")["problemas"])


def test_estagio_salva_config_e_treino_e_recusa_nao_concluido(tmp_path, monkeypatch):
    monkeypatch.setattr(versionamento.proveniencia, "ler", lambda pasta: {"estado": "concluida", "run_id": "r", "config_id": "c"})
    pasta = _escrever(tmp_path / "twostage_e1_biobertpt_seed42",
                      **{"config.json": {"config": {"config": {"epochs": 3}}}, "treino.json": {"concluido": True, "dev_history": []}})
    p = versionamento.plano_de_salvamento(pasta, "estagio")
    assert p["problemas"] == [] and sorted(p["arquivos"]) == ["config.json", "treino.json"]
    (pasta / "treino.json").write_text(json.dumps({"concluido": False}), encoding="utf-8")
    assert any("não foi concluído" in m for m in versionamento.plano_de_salvamento(pasta, "estagio")["problemas"])
    (pasta / "treino.json").write_text(json.dumps({"concluido": True, "nota": "y" * 300}), encoding="utf-8")
    assert any("texto" in m for m in versionamento.plano_de_salvamento(pasta, "estagio")["problemas"])
    (pasta / "treino.json").write_text(json.dumps({"concluido": True}), encoding="utf-8")
    (pasta / "config.json").write_text(json.dumps({"config": {"validacao": {"epocas": 1}}}), encoding="utf-8")
    assert any("validação técnica" in m for m in versionamento.plano_de_salvamento(pasta, "estagio")["problemas"])


def test_estagio_aceita_aviso_de_biblioteca_mas_recusa_outro_texto_longo(tmp_path, monkeypatch):
    monkeypatch.setattr(versionamento.proveniencia, "ler", lambda pasta: {"estado": "concluida", "run_id": "r", "config_id": "c"})
    aviso = ("UserWarning: Memory Efficient attention defaults to a non-deterministic algorithm. To explicitly enable "
             "determinism call torch.use_deterministic_algorithms(True, warn_only=False). (Triggered internally at "
             "/pytorch/aten/src/ATen/native/transformers/cuda/attention_backward.cu:900.)")
    assert len(aviso) > versionamento.LIMITE_DE_TEXTO
    pasta = _escrever(tmp_path / "twostage_e1_biobertpt_seed42", **{"config.json": {"config": {"config": {}}},
                      "treino.json": {"concluido": True, "sessoes": [{"avisos": [aviso]}]}})
    assert versionamento.plano_de_salvamento(pasta, "estagio")["problemas"] == []

    def problemas(treino):
        (pasta / "treino.json").write_text(json.dumps(treino), encoding="utf-8")
        return versionamento.plano_de_salvamento(pasta, "estagio")["problemas"]

    texto = "Paciente com " + "x" * 300
    assert any("texto" in m for m in problemas({"concluido": True, "sessoes": [{"avisos": [texto]}]}))
    assert any("texto" in m for m in problemas({"concluido": True, "sessoes": [{"avisos": ["UserWarning: " + "x" * 700]}]}))
    assert any("texto" in m for m in problemas({"concluido": True, "sessoes": [{"avisos": [aviso], "nota": aviso}]}))
    assert any("texto" in m for m in problemas({"concluido": True, "avisos": [aviso]}))


def test_perfil_desconhecido_e_branch_principal_recusados(tmp_path):
    with pytest.raises(versionamento.ErroVersionamento, match="perfil"):
        versionamento.plano_de_salvamento(tmp_path, "outro")
    with pytest.raises(versionamento.ErroVersionamento, match="branch"):
        versionamento.conferir(_composicao(tmp_path), repositorio="u/r", branch="main", sha_codigo="a" * 40,
                               perfil="composicao", imprimir=lambda *_: None)


def test_perfil_padrao_continua_o_do_baseline():
    assert versionamento.PERFIS["execucao"] == versionamento.ARQUIVOS_DA_EXECUCAO


# ------------------------------------------------------------------ HF: etapa1/ e etapa2/
class _FakeOp:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj


class _FakeApi:
    def __init__(self, existentes, envios):
        self.existentes, self.envios = existentes, envios

    def model_info(self, repo, token=None):
        return type("I", (), {"private": True})()

    def list_repo_files(self, repo, repo_type=None, token=None):
        return list(self.existentes)

    def create_commit(self, repo, operacoes, commit_message=None, repo_type=None, token=None):
        self.envios.append([o.path_in_repo for o in operacoes])
        return type("C", (), {"oid": "c" * 40})()


def _hf_falso(monkeypatch, existentes, envios):
    hf = type("HF", (), {"CommitOperationAdd": _FakeOp, "HfApi": staticmethod(lambda: _FakeApi(existentes, envios))})
    monkeypatch.setattr(publicacao_hf, "_importar_hf", lambda: hf)
    monkeypatch.setattr(segredos, "verificar_escrita_hf", lambda cred: {"usuario": "u"})
    monkeypatch.setattr(publicacao_hf, "_verificar_remoto", lambda *a, **k: {"verificada": True, "falhas": []})
    monkeypatch.setattr(publicacao_hf.proveniencia, "registrar_publicacao", lambda pasta, ref: None)


def _estagio_para_hf(tmp_path, monkeypatch, nome="twostage_e1_biobertpt_seed42"):
    pasta = tmp_path / nome
    (pasta / "checkpoints" / "melhor_modelo").mkdir(parents=True)
    (pasta / "checkpoints" / "melhor_modelo" / "m.bin").write_bytes(b"pesos")
    (pasta / "config.json").write_text("{}", encoding="utf-8")
    registro = {"estado": "concluida", "run_id": "two_stage_e1__" + "a" * 12 + "__t1", "config_id": "a" * 12,
                "estrategia": "two_stage_e1", "seed": 42, "codigo": {}, "pesos_pretreinados": {}, "ambiente": {},
                "alias_oficial": nome}
    pp = publicacao_hf.proveniencia
    monkeypatch.setattr(pp, "conferir", lambda p: [])
    monkeypatch.setattr(pp, "ler", lambda p: registro)
    monkeypatch.setattr(pp, "derivar_estado", lambda p: {"estado": "concluida", "evidencia": ""})
    return pasta


def test_prefixo_alternativo_e_validado():
    assert publicacao_hf.validar_prefixo("etapa1") == "etapa1"
    for ruim in ("../x", "etapa 1", "/etapa1", "a/b/c", ""):
        with pytest.raises(publicacao_hf.ErroPublicacao):
            publicacao_hf.validar_prefixo(ruim)


def test_cada_etapa_vai_para_a_sua_pasta_sem_tocar_na_outra(tmp_path, monkeypatch):
    pasta = _estagio_para_hf(tmp_path, monkeypatch)
    plano_hf = publicacao_hf.planejar(pasta, "u/r", prefixo="etapa1")
    assert plano_hf["pode_publicar"] and plano_hf["prefixo"] == "etapa1" and plano_hf["manifesto"]["prefixo"] == "etapa1"
    envios: list = []
    _hf_falso(monkeypatch, ["etapa2/m.bin", "etapa2/MANIFESTO_PUBLICACAO.json"], envios)       # a etapa 2 já está lá
    cred = segredos.Credencial("HF_TOKEN", "hf_" + "a" * 30, "teste")
    res = publicacao_hf.publicar(pasta, "u/r", credencial=cred, confirmar=True, prefixo="etapa1")
    assert res["executado"] and res["referencia"]["prefixo"] == "etapa1" and res["referencia"]["nome"] == pasta.name
    assert envios and all(c.startswith("etapa1/") for c in envios[0])


def test_etapa_ja_publicada_nao_e_sobrescrita(tmp_path, monkeypatch):
    pasta = _estagio_para_hf(tmp_path, monkeypatch)
    envios: list = []
    _hf_falso(monkeypatch, ["etapa1/m.bin"], envios)
    cred = segredos.Credencial("HF_TOKEN", "hf_" + "a" * 30, "teste")
    with pytest.raises(publicacao_hf.ErroPublicacao, match="já existe"):
        publicacao_hf.publicar(pasta, "u/r", credencial=cred, confirmar=True, prefixo="etapa1")
    assert envios == []


def test_operacoes_fixam_etapa_do_estagio_e_recusam_prefixo_trocado(tmp_path, monkeypatch):
    pasta = _estagio_para_hf(tmp_path, monkeypatch)
    ctx = ops.Contexto(codigo=Path("/c"), raiz_local=tmp_path, raiz_drive=None, espec=_espec(1),
                       pasta_pesos=Path("/p"), pasta_particoes=Path("/d"))
    assert ops._prefixo_hf(ctx, pasta) == "etapa1"
    assert ops._prefixo_hf(dataclasses.replace(ctx, espec=_espec(2), prefixo_hf="etapa2")) == "etapa2"
    with pytest.raises(ops.ErroOperacao, match="etapa2"):
        ops._prefixo_hf(dataclasses.replace(ctx, espec=_espec(2), prefixo_hf="etapa1"))
    with pytest.raises(ops.ErroOperacao, match="não é uma execução de"):
        ops._prefixo_hf(dataclasses.replace(ctx, espec=_espec(2)), pasta)          # a pasta é do estágio 1
