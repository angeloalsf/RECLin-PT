"""Notebook `reclin_baseline.ipynb` (simplificado), `operacoes`, `apresentacao`, `codigo_colab` e `versionamento`.

Estáticos ou com diretórios temporários: nada de rede, GPU, pesos, corpus, treino, TEST nem Git de verdade
(o Git é simulado). Escritos nas etapas 8.3, 8.3.1 e 8.3.2; a parte do notebook foi atualizada para o
notebook no formato do legado (11 seções) e **não foi executada**."""
from __future__ import annotations

import ast
import dataclasses
import json
import os
import re
import stat
from pathlib import Path

import pytest

from reclin.config import Config
from reclin.experimento import apresentacao, codigo_colab, plano, proveniencia, raiz, segredos, versionamento
from reclin.experimento import operacoes as ops
from reclin.util import caminhos

NOTEBOOK = caminhos.CODIGO / "notebooks" / "reclin_baseline.ipynb"
PESOS = {"repositorio": "pucpr/biobertpt-all", "commit": "b" * 40, "hash_manifesto": "m" * 64}
DADOS = {"versao": "v2-texto-fonte", "particoes": {"train": "t" * 64, "dev": "d" * 64}}
CODIGO_ID = {"escopo": "global-v1", "algoritmo": "sha256-manifesto-v1", "identidade": "i" * 64}
ESPEC = ops.Especificacao(estrategia="baseline", script="treinar.py", config_cls=Config)


def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _codigo_do_notebook() -> str:
    return "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code")


def _chave(**config):
    efetiva = {**ESPEC.config_protocolo().como_dict(), **config}
    return plano.chave_da_execucao(estrategia="baseline", config=efetiva, estrategia_config=None, pesos=PESOS,
                                   dados=DADOS, identidade_codigo=CODIGO_ID)


def _ctx(tmp_path, **kw):
    (tmp_path / "drive").mkdir(parents=True, exist_ok=True)
    return ops.Contexto(codigo=tmp_path / "codigo", raiz_local=tmp_path / "local", raiz_drive=tmp_path / "drive",
                        espec=ESPEC, pasta_pesos=tmp_path / "pesos", pasta_particoes=tmp_path / "particoes", **kw)


# ------------------------------------------------------------------ notebook
GIT = re.compile(r"(^|[\s;&|(\"'])git(\s|$)", re.M)
IDS_DE_CODIGO = ["c1", "c2", "c3", "c4", "c5a", "c5b", "c6", "c7", "c8", "c9", "c10", "c11"]
SECAO_DE_CADA_CELULA = {"c1": "1", "c2": "2", "c3": "3", "c4": "4", "c5a": "5", "c5b": "5", "c6": "6", "c7": "7",
                        "c8": "8", "c9": "9", "c10": "10", "c11": "11"}
REMOVIDOS = ("REPOSITORIO_HF", "PASTA_NO_DRIVE", r"^SHA\s*=", r"\bEXECUCAO\b", "AUTORIZACAO_TEST", "CONFIRMACAO_GITHUB",
             "CONFIRMAR_INSTALACAO", "OP_TEST", "OP_INSTALAR", "instalar_se_faltar", r"\binput\(")


def _celulas_de_codigo() -> dict[str, str]:
    """Código de cada célula, pelo id (c1, c2, ..., c5a, c5b, ..., c11)."""
    return {c["id"]: "".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"}


def _atribuicoes(fonte: str) -> dict:
    valores = {}
    for no in ast.parse(fonte).body:
        if isinstance(no, ast.Assign) and len(no.targets) == 1 and isinstance(no.targets[0], ast.Name):
            try:
                valores[no.targets[0].id] = ast.literal_eval(no.value)
            except ValueError:
                pass
    return valores


def test_notebook_valido_com_celulas_que_compilam_e_sem_saidas():
    nb = _nb()
    assert nb["nbformat"] == 4 and len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            fonte = "\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%")))
            ast.parse(fonte)
            assert c["outputs"] == [] and c["execution_count"] is None


def test_secoes_do_legado_na_ordem():
    assert list(_celulas_de_codigo()) == IDS_DE_CODIGO
    atual, vistas = None, []
    for c in _nb()["cells"]:
        fonte = "".join(c["source"])
        if c["cell_type"] == "markdown" and (m := re.match(r"## (\d+)\. ", fonte)):
            atual = m.group(1)
            vistas.append(atual)
        elif c["cell_type"] == "code":
            assert atual == SECAO_DE_CADA_CELULA[c["id"]], c["id"]
    assert vistas == [str(n) for n in range(1, 12)]


def test_execucao_controlada_instalar_e_baixar_pesos_so_nas_proprias_celulas():
    c = _celulas_de_codigo()
    assert [n for n, f in c.items() if "pip install" in f] == ["c4"]
    assert [n for n, f in c.items() if "operacoes.OP_PESOS" in f] == ["c5b"]
    assert not [n for n, f in c.items() if re.search(r"from_pretrained|snapshot_download|hf_hub_download|pretreino\.py", f)]
    assert _atribuicoes(c["c5a"])["REVISAO_PESOS"] == "36b11cfdfa49876b108fdbe88eaeeb9be3b03f1c"
    chamadas = {n: re.findall(r"operacoes\.executar\((\w+(?:\.\w+)?)", f) for n, f in c.items() if "operacoes.executar(" in f}
    assert chamadas == {"c5b": ["operacoes.OP_PESOS"], "c7": ["OPERACAO"], "c8": ["operacoes.OP_DEV"],
                        "c11": ["operacoes.OP_PUBLICACAO", "operacoes.OP_PUBLICAR"]}


def test_retomar_nao_e_nova_tentativa_e_parametros_sao_os_do_protocolo():
    c7 = _celulas_de_codigo()["c7"]
    v = _atribuicoes(c7)
    protocolo = {k: val for k, val in ESPEC.config_protocolo().como_dict().items() if k in ESPEC.permitidos}
    assert v["PARAMETROS"] == protocolo and set(v["PARAMETROS"]) == set(ESPEC.permitidos)
    assert v["NOVA_EXECUCAO"] is False and v["MOTIVO"] == ""
    assert "recomendar_ramo" in c7 and "rec['recomendada']" in c7 and "operacoes.OP_RETOMAR" in c7
    assert "NOVA_EXECUCAO and rec['recomendada'] == operacoes.OP_RETOMAR and not MOTIVO.strip()" in c7
    assert "EXPLORATÓRIA" in c7 and 'MOTIVO.strip() or' in c7
    assert c7.index("exigir_quadro") < c7.index("operacoes.executar(OPERACAO")
    assert ops.config_da_operacao(ESPEC, v["PARAMETROS"]) == ESPEC.config_protocolo()


def test_execucao_selecionada_na_secao_8_e_so_ela_vai_para_10_e_11():
    c = _celulas_de_codigo()
    assert "EXECUCAO_SELECIONADA = CTX.execucao" in c["c8"] and "NATUREZA" in c["c8"] and "OP_DEV" in c["c8"]
    for n in ("c10", "c11"):
        assert "'EXECUCAO_SELECIONADA' not in globals()" in c[n] and "execucao=EXECUCAO_SELECIONADA" in c[n]
        assert "_ctx_exec" not in c[n]                     # nunca "a mais recente": só a selecionada
    assert "versionamento" in c["c10"] and "versionamento" not in "".join(f for n, f in c.items() if n != "c10")


def test_publicacao_so_com_confirmacao_explicita_e_so_na_branch_de_resultados():
    c = _celulas_de_codigo()
    assert _atribuicoes(c["c10"])["PUBLICAR_NO_GITHUB"] is False and _atribuicoes(c["c11"])["PUBLICAR_NO_HF"] is False
    assert _atribuicoes(c["c10"])["BRANCH_RESULTADOS"] == "resultados-colab"
    assert c["c10"].index("versionamento.conferir") < c["c10"].index("if not PUBLICAR_NO_GITHUB") < c["c10"].index("versionamento.salvar")
    assert "branch=BRANCH_RESULTADOS" in c["c10"] and not re.search(r"branch=['\"](main|master)", c["c10"])
    assert "frase_de_publicacao" in c["c11"] and "if PUBLICAR_NO_HF else ''" in c["c11"]
    assert c["c11"].index("operacoes.OP_PUBLICACAO") < c["c11"].index("if not PUBLICAR_NO_HF") < c["c11"].index("operacoes.OP_PUBLICAR")
    assert "segredo_no_ambiente" in c["c11"] and "segredos.obter" not in c["c11"]


def test_segredos_e_git_so_onde_precisam():
    c = _celulas_de_codigo()
    assert [n for n, f in c.items() if "GITHUB_PAT" in f] == ["c1", "c10"]
    assert [n for n, f in c.items() if "segredos.HUGGING_FACE" in f] == ["c6", "c11"]
    assert "os.environ[" not in _codigo_do_notebook() and "'HF_TOKEN'" not in c["c11"]
    assert [n for n, f in c.items() if "subprocess" in f] == ["c1"] and [n for n, f in c.items() if "git(" in f] == ["c1"]
    assert "GIT_ASKPASS" in c["c1"] and "userdata.get('GITHUB_PAT')" in c["c1"]
    assert "url = f'https://github.com/angeloalsf/{REPO_NAME}.git'" in c["c1"] and "{TOKEN}" not in c["c1"] and "@github" not in c["c1"]
    for n, f in c.items():
        assert not GIT.search(f.replace("git(", "")) or n == "c1", n
    assert not re.search(r"(ghp_|github_pat_|hf_[A-Za-z0-9]{20})", _codigo_do_notebook())


def test_campos_removidos_nao_voltam_e_nenhuma_celula_avalia_ou_le_o_test():
    codigo = _codigo_do_notebook()
    for padrao in REMOVIDOS:
        assert not re.search(padrao, codigo, re.M), padrao
    for proibido in ("avaliar_test", "predicoes_test", "avaliacoes_test", "--particao test", "metricas.json", "OP_TEST"):
        assert proibido not in codigo, proibido
    assert _atribuicoes(_celulas_de_codigo()["c5a"])["need"] == ["train.jsonl", "dev.jsonl", "test.jsonl", "MANIFEST.json"]


def test_flags_usadas_existem_nos_scripts_reais():
    cfg = ops.config_da_operacao(ESPEC, {"lr": 3e-5})
    ctx = ops.Contexto(codigo=Path("/c"), raiz_local=Path("/r"), raiz_drive=None, espec=ESPEC,
                       pasta_pesos=Path("/p"), pasta_particoes=Path("/d"))
    argumentos = ops.argumentos_de_treino(ctx, cfg, "n", retomar=True)
    flags = [a for a in argumentos if a.startswith("--")]
    treinar = (caminhos.CODIGO / "scripts" / "treinar.py").read_text(encoding="utf-8")
    declaradas = set(re.findall(r'add_argument\("(--[a-z-]+)"', treinar))
    por_campo = {"--" + f.name.replace("_", "-") for f in dataclasses.fields(Config)}
    assert all(f in declaradas or f in por_campo for f in flags)
    assert por_campo <= set(flags) and "--retomar" in flags
    assert argumentos[argumentos.index("--lr") + 1] == "3e-05"


# ------------------------------------------------------------------ configuração
def test_fase_fixa_encoder_e_seed_e_recusa_campo_desconhecido():
    for ruim in ({"seed": 1}, {"encoder": "bertimbau"}, {"foo": 1}, {"epochs": 0}):
        with pytest.raises(ops.ErroOperacao):
            ops.config_da_operacao(ESPEC, ruim)
    assert ops.config_da_operacao(ESPEC, {}) == ESPEC.config_protocolo()
    assert ESPEC.config_protocolo().encoder == "biobertpt" and ESPEC.config_protocolo().seed == 42
    assert ESPEC.alias_oficial == "baseline_biobertpt_seed42"


def test_frases_de_autorizacao_e_confirmacao():
    run = "baseline__0123456789ab__t1"
    assert ops.autorizacao_test_valida(run, f"  {ops.frase_de_autorizacao_test(run)} ")
    assert not ops.autorizacao_test_valida(run, "sim") and not ops.autorizacao_test_valida(run, "")
    assert not ops.autorizacao_test_valida("baseline__0123456789ab__t2", ops.frase_de_autorizacao_test(run))
    assert ops.frase_de_publicacao("n", "u/r") == "PUBLICAR n EM u/r"


def test_diferencas_de_chave_apontam_o_campo():
    d = ops.diferencas_de_chave(_chave(), _chave(lr=3e-5))
    assert d == ["config.lr: 2e-05 → 3e-05"]


# ------------------------------------------------------------------ recusas
def test_novo_treinamento_recusa_parametros_fora_do_protocolo_e_nova_exige_motivo(tmp_path):
    with pytest.raises(ops.ErroOperacao, match="PARAMETROS"):
        ops.treinar_novo(_ctx(tmp_path, parametros={"lr": 3e-5}))
    with pytest.raises(ops.ErroOperacao, match="MOTIVO"):
        ops.treinar_nova_execucao(_ctx(tmp_path, parametros={"lr": 3e-5}))


def test_treinar_exige_drive(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.raiz_drive = None
    with pytest.raises(ops.ErroOperacao, match="Drive"):
        ops.treinar_novo(ctx)


def test_retomar_recusa_checkpoint_de_outra_configuracao_e_explica(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(ops, "_exigir_gpu", lambda c: None)
    antiga, atual = _chave(lr=3e-5), _chave()
    p = plano.planejar(antiga, ctx.raiz_local)
    pasta = plano.preparar(p, antiga, ctx.raiz_local, codigo={"origem": "local"})
    (pasta / "checkpoints").mkdir()
    (pasta / "checkpoints" / "ultimo.pt").write_bytes(b"x")
    monkeypatch.setattr(ops, "_chave", lambda c, cfg: atual)
    with pytest.raises(ops.ErroOperacao, match="COMPATÍVEL") as erro:
        ops.retomar_treino(ctx)
    assert "config.lr" in str(erro.value)
    assert ops.executar(ops.OP_RETOMAR, ctx) is None
    assert "ERRO:" in capsys.readouterr().out
    assert (pasta / "checkpoints" / "ultimo.pt").read_bytes() == b"x"        # nada foi tocado


def test_operacoes_que_precisam_da_estrategia_param_sem_ela(tmp_path, capsys):
    ctx = _ctx(tmp_path)
    ctx.espec = None
    assert ops.executar(ops.OP_NOVO, ctx) is None and "ERRO:" in capsys.readouterr().out


def test_metricas_dev_nao_le_o_test(tmp_path):
    ctx = _ctx(tmp_path)
    chave = _chave()
    p = plano.planejar(chave, ctx.raiz_local)
    pasta = plano.preparar(p, chave, ctx.raiz_local, codigo={"origem": "local"})
    (pasta / "treino.json").write_text(json.dumps({"concluido": True, "dev_history": [
        {"epoch": 1, "train_loss": 1.0, "dev_loss": 0.9, "dev_macro_f1": 0.5, "dev_negation_of_f1": 0.4,
         "duration_s": 10.0}], "melhor_epoca": 1, "melhor_dev_macro_f1": 0.5, "posicao": {}}), encoding="utf-8")
    (pasta / "metricas.json").write_text(json.dumps({"dev": {"macro_f1": 0.5}, "test": {"macro_f1": 0.99}}),
                                         encoding="utf-8")
    resultado = ops.metricas_dev(ctx)
    assert resultado["dev"] == {"macro_f1": 0.5} and "test" not in json.dumps(resultado)


# ------------------------------------------------------------------ rodar_script com log
def test_rodar_script_registra_log_sem_segredos_e_chama_periodico(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "eco.py").write_text(
        "import time\nprint('linha um', flush=True)\nprint('hf_' + 'a' * 30, flush=True)\ntime.sleep(2.5)\n",
        encoding="utf-8")
    chamadas = []
    codigo = raiz.rodar_script(tmp_path, "eco.py", log=tmp_path / "logs" / "x.log",
                               periodico=(1.0, lambda: chamadas.append(1)))
    texto = (tmp_path / "logs" / "x.log").read_text(encoding="utf-8")
    assert codigo == 0 and "linha um" in texto and "hf_" + "a" * 30 not in texto and chamadas


# ------------------------------------------------------------------ segredos (etapa 8.3.1)
def test_segredo_cai_na_variavel_de_ambiente_quando_o_userdata_nao_funciona(monkeypatch):
    monkeypatch.setattr(segredos.colab, "em_colab", lambda: True)
    monkeypatch.setattr(segredos.colab, "ler_secret", lambda nome: (None, "falha ao ler (subprocesso)"))
    cred = segredos.obter("HF_TOKEN", ambiente={"HF_TOKEN": "hf_" + "k" * 30})
    assert cred is not None and cred.origem == "variável de ambiente" and "k" * 30 not in repr(cred)
    assert segredos.obter("HF_TOKEN", ambiente={}) is None
    with pytest.raises(segredos.ErroSegredo, match="Secrets do Colab"):
        segredos.obter("HF_TOKEN", obrigatorio=True, ambiente={})


def test_token_de_escrita_do_github_e_um_secret_separado():
    assert segredos.GITHUB_ESCRITA == "GITHUB_TOKEN_ESCRITA" != segredos.GITHUB
    assert segredos.OPERACOES["salvar_github"] == ((segredos.GITHUB_ESCRITA,), ())
    assert "Read and write" in segredos.CATALOGO[segredos.GITHUB_ESCRITA].permissao_minima
    assert "Read-only" in segredos.CATALOGO[segredos.GITHUB].permissao_minima


def test_segredo_no_ambiente_so_vale_dentro_do_bloco(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    valor = "hf_" + "z" * 30
    monkeypatch.setattr(segredos, "obter", lambda nome, obrigatorio=False, ambiente=None: segredos.Credencial(nome, valor, "t"))
    with apresentacao.segredo_no_ambiente("HF_TOKEN"):
        assert os.environ["HF_TOKEN"] == valor
    assert "HF_TOKEN" not in os.environ
    with pytest.raises(RuntimeError):
        with apresentacao.segredo_no_ambiente("HF_TOKEN"):
            raise RuntimeError("falha no envio")
    assert "HF_TOKEN" not in os.environ
    monkeypatch.setenv("HF_TOKEN", "anterior")
    with apresentacao.segredo_no_ambiente("HF_TOKEN"):
        pass
    assert os.environ["HF_TOKEN"] == "anterior"


def test_segredo_ausente_levanta_com_instrucao(monkeypatch):
    monkeypatch.setattr(segredos.colab, "em_colab", lambda: False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(segredos.ErroSegredo, match="Secrets do Colab"):
        with apresentacao.segredo_no_ambiente("HF_TOKEN"):
            pass
    texto = apresentacao.instrucoes_secret("GITHUB_TOKEN_ESCRITA")
    assert "Acesso pelo notebook" in texto and "GITHUB_TOKEN_ESCRITA" in texto and "Read and write" in texto
    assert not re.search(r"(ghp_|github_pat_|hf_)[A-Za-z0-9]{10}", texto)


# ------------------------------------------------------------------ apresentacao (etapa 8.3.1)
def test_semente_da_fase_vem_da_constante_e_confere_com_a_pasta_oficial():
    assert apresentacao.semente_da_fase() == (42, "baseline_biobertpt_seed42")


def _ctx_com_chave(tmp_path, monkeypatch, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ctx = _ctx(tmp_path, **kw)
    monkeypatch.setattr(ops, "_chave", lambda c, cfg: _chave(**{k: getattr(cfg, k) for k in ("lr",)}))
    monkeypatch.setattr(apresentacao.ambiente, "coletar", lambda: {"gpu": "Tesla T4", "versoes": {}})
    return ctx


def test_conferir_particoes_diz_que_o_sha_nao_tem_as_particoes(tmp_path, capsys):
    ctx = _ctx(tmp_path)
    assert apresentacao.conferir_particoes(ctx, sha_codigo="a" * 40) is None
    saida = capsys.readouterr().out
    assert "ERRO:" in saida and "não contém as partições" in saida and "aaaaaaaaaaaa" in saida
    assert "Drive" not in saida.split("ERRO:")[1]          # não manda copiar partições para o Drive


def test_quadro_pre_treino_mostra_tudo_e_nao_cria_nada(tmp_path, monkeypatch, capsys):
    ctx = _ctx_com_chave(tmp_path, monkeypatch, revisao_pesos="b" * 40)
    quadro = apresentacao.quadro_pre_treino(ctx, ops.OP_NOVO, sha_codigo="c" * 40, repositorio="u/r")
    saida = capsys.readouterr().out
    assert quadro["pronto"] and quadro["impedimentos"] == [] and quadro["modo"] == "novo"
    for esperado in ("PROTOCOLO (oficial: baseline_biobertpt_seed42)", "u/r@" + "c" * 40, "b" * 40, "Tesla T4",
                     "config_id", "Destino no Drive", "a cada 500 passos", "a cada 10 min", "= protocolo", "Pronto"):
        assert esperado in saida, esperado
    assert not ctx.raiz_local.exists()                       # só leitura


def test_quadro_marca_execucao_exploratoria_e_impedimentos(tmp_path, monkeypatch, capsys):
    ctx = _ctx_com_chave(tmp_path, monkeypatch, parametros={"lr": 3e-5})
    quadro = apresentacao.quadro_pre_treino(ctx, ops.OP_NOVO)
    saida = capsys.readouterr().out
    assert not quadro["pronto"] and "EXPLORATÓRIA" in saida and "≠ protocolo" in saida and "IMPEDIMENTOS" in saida
    ctx.motivo = ""
    quadro = apresentacao.quadro_pre_treino(ctx, ops.OP_NOVA)
    assert any("MOTIVO" in i for i in quadro["impedimentos"])
    quadro = apresentacao.quadro_pre_treino(_ctx_com_chave(tmp_path, monkeypatch), ops.OP_RETOMAR)
    assert any("COMPATÍVEL" in i for i in quadro["impedimentos"])


def test_exigir_quadro_recusa_ausente_desatualizado_ou_com_impedimento(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    quadro = apresentacao.quadro_pre_treino(ctx, ops.OP_NOVO)
    apresentacao.exigir_quadro(quadro, ctx, ops.OP_NOVO)
    with pytest.raises(ops.ErroOperacao, match="6a"):
        apresentacao.exigir_quadro(None, ctx, ops.OP_NOVO)
    ctx.parametros = {"lr": 3e-5}
    with pytest.raises(ops.ErroOperacao, match="mudaram"):
        apresentacao.exigir_quadro(quadro, ctx, ops.OP_NOVO)
    ctx.parametros = {}
    with pytest.raises(ops.ErroOperacao, match="impedimentos"):
        apresentacao.exigir_quadro({**quadro, "pronto": False, "impedimentos": ["x"]}, ctx, ops.OP_NOVO)


def _execucao(ctx, chave, estados=()):
    p = plano.planejar(chave, ctx.raiz_local)
    pasta = plano.preparar(p, chave, ctx.raiz_local, codigo={"origem": "local"})
    for estado in estados:
        proveniencia.mudar_estado(pasta, estado, "teste")
    # o estado que vale é o derivado dos arquivos (plano.execucoes): grava a evidência do último estado
    if estados:
        (pasta / "config.json").write_text("{}", encoding="utf-8")
    if estados and estados[-1] == "interrompida":
        (pasta / "checkpoints").mkdir(exist_ok=True)
        (pasta / "checkpoints" / "ultimo.pt").write_bytes(b"x")
    if estados and estados[-1] == "concluida":
        (pasta / "treino.json").write_text(json.dumps({"concluido": True}), encoding="utf-8")
    return pasta


def test_recomenda_4_5_ou_6_conforme_as_execucoes(tmp_path, monkeypatch, capsys):
    ctx = _ctx_com_chave(tmp_path / "a", monkeypatch)
    assert apresentacao.recomendar_ramo(ctx)["recomendada"] == ops.OP_NOVO
    _execucao(ctx, _chave(), ("iniciada", "interrompida"))
    assert apresentacao.recomendar_ramo(ctx)["recomendada"] == ops.OP_RETOMAR
    ctx = _ctx_com_chave(tmp_path / "b", monkeypatch)
    _execucao(ctx, _chave(), ("iniciada", "concluida"))
    assert apresentacao.recomendar_ramo(ctx)["recomendada"] == ops.OP_NOVA
    ctx = _ctx_com_chave(tmp_path / "c", monkeypatch)
    _execucao(ctx, _chave(lr=3e-5), ("iniciada", "interrompida"))
    r = apresentacao.recomendar_ramo(ctx)
    assert r["recomendada"] == ops.OP_NOVO and r["outras"]


def test_curvas_sem_treino_json_nao_quebra_nem_abre_o_test(tmp_path, monkeypatch, capsys):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    pasta = _execucao(ctx, _chave(), ("iniciada",))
    (pasta / "predicoes_test.json").write_text("NAO LER", encoding="utf-8")
    assert apresentacao.curvas_dev(ctx) is None
    assert "treino.json ainda não existe" in capsys.readouterr().out


# ------------------------------------------------------------------ versionamento (etapa 8.3.1)
URL = "https://github.com/u/r.git"
BRANCH = "resultados-colab"
SHA = "d" * 40
TOKEN = "ghp_" + "e" * 36


def _concluida(ctx, texto_nas_predicoes=False, test_nas_metricas=False):
    pasta = _execucao(ctx, _chave(), ("iniciada", "concluida"))
    predicoes = {"model": "m", "seed": 42, "labels": ["a", "b", "c"], "y_true": [0, 1], "y_pred": [0, 2],
                 "probs": [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], "split": "dev"}
    if texto_nas_predicoes:
        predicoes["texto"] = ["paciente nega dor"]
    metricas = {"dev": {"macro_f1": 0.5}}
    if test_nas_metricas:
        metricas["test"] = {"macro_f1": 0.9}
    for nome, conteudo in (("config.json", {"encoder": "biobertpt"}), ("treino.json", {"concluido": True, "dev_history": []}),
                           ("predicoes_dev.json", predicoes), ("metricas.json", metricas)):
        (pasta / nome).write_text(json.dumps(conteudo), encoding="utf-8")
    (pasta / "checkpoints").mkdir()
    (pasta / "checkpoints" / "ultimo.pt").write_bytes(b"pesos")
    (pasta / "predicoes_test.json").write_text("{}", encoding="utf-8")
    (pasta / "logs").mkdir()
    (pasta / "logs" / "treino.log").write_text("log", encoding="utf-8")
    (pasta / "SYNC.json").write_text("{}", encoding="utf-8")
    return pasta


def test_plano_versiona_so_os_cinco_arquivos_e_exclui_o_resto(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    p = versionamento.plano_de_salvamento(_concluida(ctx))
    assert p["problemas"] == [] and tuple(p["arquivos"]) == versionamento.ARQUIVOS_DA_EXECUCAO
    motivos = {e["arquivo"]: e["motivo"] for e in p["excluidos"]}
    assert "pesos" in motivos["checkpoints/ultimo.pt"] and "TEST" in motivos["predicoes_test.json"]
    assert "logs" in motivos["logs/treino.log"] and "SYNC.json" in motivos
    assert not set(motivos) & set(p["arquivos"])


def test_recusa_test_nas_metricas_texto_nas_predicoes_e_execucao_nao_concluida(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path / "a", monkeypatch)
    assert any("'test'" in m for m in versionamento.plano_de_salvamento(_concluida(ctx, test_nas_metricas=True))["problemas"])
    ctx = _ctx_com_chave(tmp_path / "b", monkeypatch)
    assert any("texto" in m for m in versionamento.plano_de_salvamento(_concluida(ctx, texto_nas_predicoes=True))["problemas"])
    ctx = _ctx_com_chave(tmp_path / "c", monkeypatch)
    pasta = _execucao(ctx, _chave(), ("iniciada",))
    assert any("concluída" in m for m in versionamento.plano_de_salvamento(pasta)["problemas"])
    assert versionamento.sem_texto_clinico({"y_true": [0, 1], "probs": [[0.1, 0.9]], "labels": ["a"], "x": "curto"}) == []
    assert versionamento.sem_texto_clinico({"y": "x" * 500})


def test_recusa_branch_principal_e_destino_invalido(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    pasta = _concluida(ctx)
    for ruim in ("main", "master"):
        with pytest.raises(versionamento.ErroVersionamento, match="branch própria"):
            versionamento.conferir(pasta, repositorio="u/r", branch=ruim, sha_codigo=SHA, imprimir=lambda _: None)
    with pytest.raises(versionamento.ErroVersionamento):
        versionamento.conferir(pasta, repositorio="sem-barra", branch=BRANCH, sha_codigo=SHA, imprimir=lambda _: None)


def test_conferir_mostra_destino_arquivos_e_exclusoes(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    pasta = _concluida(ctx)
    linhas = []
    p = versionamento.conferir(pasta, repositorio="u/r", branch=BRANCH, sha_codigo=SHA, imprimir=linhas.append)
    texto = "\n".join(linhas)
    assert p["pode_salvar"] and f"codigo/resultados/execucoes/{pasta.name}/" in texto
    assert "ultimo.pt" in texto and "SALVAR" not in texto           # não pede frase de confirmação


class GitFalso:
    """Simula `git`: registra cada chamada com o ambiente e responde como um remoto qualquer."""

    def __init__(self, remoto=False, falha_em=None):
        self.chamadas, self.verificacoes = [], []
        self.remoto, self.falha_em = remoto, falha_em
        self.preparados, self.commitados = [], set()
        self.askpass = None

    def __call__(self, argumentos, cwd, env):
        args = list(argumentos)
        while args[0] == "-c":
            args = args[2:]
        self.chamadas.append((args, cwd, dict(env), list(argumentos)))
        script = Path(env["GIT_ASKPASS"])
        self.askpass = script.parent
        self.verificacoes.append((stat.S_IMODE(script.stat().st_mode), stat.S_IMODE(script.parent.stat().st_mode),
                                  TOKEN in script.read_text(encoding="utf-8")))
        verbo = args[0]
        if self.falha_em == verbo:
            return 1, f"falha simulada com {TOKEN}"
        if verbo == "ls-remote":
            return (0, "abc\trefs/heads/" + BRANCH) if self.remoto else (2, "")
        if verbo == "clone":
            (Path(args[-1]) / ".git").mkdir(parents=True)
        if verbo == "remote":
            return 0, URL
        if verbo == "rev-parse":
            return 0, BRANCH if "--abbrev-ref" in args else "c0ffee"
        if verbo == "add":
            self.preparados = args[3:]
        if verbo == "diff":
            return 0, "\n".join(a for a in self.preparados if a not in self.commitados)
        if verbo == "commit":
            self.commitados |= set(self.preparados)
        if verbo == "push":
            self.remoto = True
        return 0, ""


def _salvar(pasta, clone, git, **kw):
    cred = segredos.Credencial("GITHUB_TOKEN_ESCRITA", TOKEN, "teste")
    argumentos = dict(repositorio="u/r", branch=BRANCH, sha_codigo=SHA, clone=clone, nome_git="Fulano",
                      email_git="fulano@exemplo.org", credencial=cred, confirmacao=f"SALVAR {pasta.name} EM u/r@{BRANCH}",
                      executor=git, imprimir=lambda _: None)
    argumentos.update(kw)
    return versionamento.salvar(pasta, **argumentos)


def test_salvar_segue_a_ordem_e_nao_deixa_o_token_em_lugar_nenhum(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    pasta, clone = _concluida(ctx), tmp_path / "clone"
    monkeypatch.delenv("RECLIN_GIT_TOKEN", raising=False)
    git = GitFalso()
    resultado = _salvar(pasta, clone, git)
    verbos = [c[0][0] for c in git.chamadas]
    assert verbos == ["ls-remote", "clone", "fetch", "checkout", "add", "diff", "commit", "push", "rev-parse"]
    add = next(c[0] for c in git.chamadas if c[0][0] == "add")
    assert add[:3] == ["add", "-f", "--"] and add[3:] == [f"codigo/resultados/execucoes/{pasta.name}/{a}" for a in versionamento.ARQUIVOS_DA_EXECUCAO]
    assert not any(re.search(r"checkpoints|\.pt|_test|logs", a) for a in add[3:])
    assert resultado["novo_commit"] and resultado["commit"] == "c0ffee"
    # token: só no ambiente do subprocesso; nunca em argv, em os.environ, no script ou em .git/config
    assert all(TOKEN not in " ".join(c[3]) for c in git.chamadas) and all(TOKEN not in k for c in git.chamadas for k in c[3])
    assert all(c[2]["RECLIN_GIT_TOKEN"] == TOKEN for c in git.chamadas)
    assert "RECLIN_GIT_TOKEN" not in os.environ and TOKEN not in os.environ.values()
    assert all(modo == 0o700 and pasta_modo == 0o700 and not token_no_script for modo, pasta_modo, token_no_script in git.verificacoes)
    assert not git.askpass.exists()                                   # apagado ao terminar
    assert all("credential.helper=" in c[3] for c in git.chamadas)
    for arquivo in versionamento.ARQUIVOS_DA_EXECUCAO:
        assert (clone / "codigo" / "resultados" / "execucoes" / pasta.name / arquivo).is_file()
    assert not list(clone.rglob("*.pt")) and not list(clone.rglob("*_test*"))


def test_reexecutar_sem_mudanca_nao_cria_commit(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    pasta, clone = _concluida(ctx), tmp_path / "clone"
    git = GitFalso()
    _salvar(pasta, clone, git)
    git.chamadas.clear()
    resultado = _salvar(pasta, clone, git)
    assert [c[0][0] for c in git.chamadas] == ["ls-remote", "remote", "rev-parse", "add", "diff", "pull", "push", "rev-parse"]
    assert not resultado["novo_commit"]
    pull = next(c[0] for c in git.chamadas if c[0][0] == "pull")
    assert pull[:3] == ["pull", "--rebase", "--autostash"]


def test_salvar_recusa_frase_errada_se_dada_com_problemas_e_arquivo_diferente(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path / "a", monkeypatch)
    pasta, clone = _concluida(ctx), tmp_path / "clone"
    git = GitFalso()
    with pytest.raises(versionamento.ErroVersionamento, match="confirmação diferente"):
        _salvar(pasta, clone, git, confirmacao="sim")
    assert git.chamadas == []                                          # frase errada, se dada, barra antes do Git
    ctx = _ctx_com_chave(tmp_path / "b", monkeypatch)
    ruim = _concluida(ctx, test_nas_metricas=True)
    with pytest.raises(versionamento.ErroVersionamento, match="nada foi salvo"):
        _salvar(ruim, clone, git)
    assert git.chamadas == []
    destino = clone / "codigo" / "resultados" / "execucoes" / pasta.name
    destino.mkdir(parents=True)
    (clone / ".git").mkdir()
    (destino / "metricas.json").write_text('{"dev": {"macro_f1": 0.99}}', encoding="utf-8")
    with pytest.raises(versionamento.ErroVersionamento, match="diferente"):
        _salvar(pasta, clone, git)
    assert json.loads((destino / "metricas.json").read_text(encoding="utf-8"))["dev"]["macro_f1"] == 0.99


def test_askpass_e_apagado_mesmo_quando_o_push_falha_e_o_token_nao_vaza_na_mensagem(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    pasta, clone = _concluida(ctx), tmp_path / "clone"
    git = GitFalso(falha_em="push")
    with pytest.raises(versionamento.ErroVersionamento) as erro:
        _salvar(pasta, clone, git)
    assert TOKEN not in str(erro.value) and not git.askpass.exists()


def test_identidade_do_git_e_parametro_e_nao_vai_para_o_clone(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    pasta, clone = _concluida(ctx), tmp_path / "clone"
    for nome, email in (("", "a@b.c"), ("Fulano", "sem-arroba")):
        with pytest.raises(versionamento.ErroVersionamento, match="GIT_"):
            _salvar(pasta, clone, GitFalso(), nome_git=nome, email_git=email)
    git = GitFalso()
    _salvar(pasta, clone, git)
    assert any("user.name=Fulano" in c[3] for c in git.chamadas) and not (clone / ".git" / "config").exists()


def test_salvar_sem_confirmacao_funciona_e_so_vai_para_a_branch_de_resultados(tmp_path, monkeypatch):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    pasta, clone = _concluida(ctx), tmp_path / "clone"
    git = GitFalso()
    resultado = _salvar(pasta, clone, git, confirmacao=None)
    assert resultado["novo_commit"]
    push = next(c[0] for c in git.chamadas if c[0][0] == "push")
    assert push[-1].endswith(BRANCH) or BRANCH in push
    assert not any(c[0][0] == "push" and any(b in c[0] for b in ("main", "master")) for c in git.chamadas)
    for ruim in ("main", "master"):
        with pytest.raises(versionamento.ErroVersionamento, match="branch própria"):
            _salvar(pasta, tmp_path / "outro", GitFalso(), branch=ruim, confirmacao=None)


# ------------------------------------------------------------------ clonar o código pela branch (etapa 8.3.2)
SHA_CLONE = "9" * 40


class GitDoClone:
    """Simula o `git` de `clonar_codigo`: registra as chamadas e cria uma árvore `codigo/` mínima no clone."""

    def __init__(self, branch="main", sha=SHA_CLONE, url="https://github.com/u/r.git", sujo=False):
        self.chamadas, self.branch, self.sha, self.url, self.sujo, self.envs = [], branch, sha, url, sujo, []

    def __call__(self, argumentos, cwd, env, token):
        self.chamadas.append(list(argumentos))
        self.envs.append(dict(env))
        verbo = argumentos[0]
        if verbo == "clone":
            destino = Path(argumentos[-1])
            (destino / ".git").mkdir(parents=True)
            (destino / "codigo" / "reclin").mkdir(parents=True)
            (destino / "codigo" / "pyproject.toml").write_text("", encoding="utf-8")
            (destino / "codigo" / "reclin" / "__init__.py").write_text("", encoding="utf-8")
            return ""
        if verbo == "remote":
            return self.url
        if verbo == "rev-parse":
            return self.branch if "--abbrev-ref" in argumentos else self.sha
        if verbo == "status":
            return " M codigo/x.py" if self.sujo else ""
        return ""


def test_clonar_codigo_clona_a_branch_e_grava_o_commit_clonado(tmp_path):
    git, destino = GitDoClone(), tmp_path / "RECLin-PT"
    info = codigo_colab.clonar_codigo("u/r", "main", destino, git=git)
    assert info["sha"] == SHA_CLONE and info["origem"] == "git" and info["reaproveitado"] is False
    assert info["codigo"] == str(destino / "codigo") and info["tarball_sha256"] is None
    clone = next(c for c in git.chamadas if c[0] == "clone")
    assert clone[1:6] == ["--depth", "1", "--branch", "main", "https://github.com/u/r.git"]
    gravado = json.loads((destino / codigo_colab.ARQUIVO_REVISAO).read_text(encoding="utf-8"))
    assert gravado["sha"] == SHA_CLONE and gravado["origem"] == "git" and gravado["branch"] == "main"
    assert gravado["repositorio"] == "u/r"


def test_clonar_codigo_reaproveita_sem_atualizar_e_recusa_clone_estranho(tmp_path):
    destino = tmp_path / "RECLin-PT"
    codigo_colab.clonar_codigo("u/r", "main", destino, git=GitDoClone())
    git = GitDoClone()
    info = codigo_colab.clonar_codigo("u/r", "main", destino, git=git)
    assert info["reaproveitado"] is True and info["sha"] == SHA_CLONE
    assert not any(c[0] in ("clone", "fetch", "pull", "checkout") for c in git.chamadas)       # não atualiza
    for ruim, trecho in ((GitDoClone(url="https://github.com/outro/repo.git"), "outro repositório"),
                         (GitDoClone(branch="dev"), "outra branch"), (GitDoClone(sujo=True), "alterados")):
        with pytest.raises(codigo_colab.ErroCodigoColab, match=trecho):
            codigo_colab.clonar_codigo("u/r", "main", destino, git=ruim)


def test_clonar_codigo_recusa_tag_pasta_ocupada_e_nomes_invalidos(tmp_path):
    with pytest.raises(codigo_colab.ErroCodigoColab, match="não é uma branch"):
        codigo_colab.clonar_codigo("u/r", "v1.0", tmp_path / "a", git=GitDoClone(branch="HEAD"))
    ocupada = tmp_path / "b"
    ocupada.mkdir()
    (ocupada / "algo").write_text("x", encoding="utf-8")
    with pytest.raises(codigo_colab.ErroCodigoColab, match="não está vazio"):
        codigo_colab.clonar_codigo("u/r", "main", ocupada, git=GitDoClone())
    for repositorio, branch in (("sem-barra", "main"), ("u/r", "../x"), ("u/r", "a b"), ("u/r", "x.lock")):
        with pytest.raises(codigo_colab.ErroCodigoColab):
            codigo_colab.clonar_codigo(repositorio, branch, tmp_path / "c", git=GitDoClone())


def test_clonar_codigo_exige_a_pasta_codigo_na_branch(tmp_path):
    class SemCodigo(GitDoClone):
        def __call__(self, argumentos, cwd, env, token):
            saida = super().__call__(argumentos, cwd, env, token)
            if argumentos[0] == "clone":
                (Path(argumentos[-1]) / "codigo" / "pyproject.toml").unlink()
            return saida

    with pytest.raises(codigo_colab.ErroCodigoColab, match="codigo/"):
        codigo_colab.clonar_codigo("u/r", "main", tmp_path / "x", git=SemCodigo())


def test_clonar_codigo_com_token_usa_askpass_e_o_apaga(tmp_path):
    git = GitDoClone()
    token = "ghp_" + "q" * 36
    codigo_colab.clonar_codigo("u/r", "main", tmp_path / "x", token=token, git=git)
    envs = [e for e in git.envs if e]
    assert envs and all(e["RECLIN_GIT_TOKEN"] == token and Path(e["GIT_ASKPASS"]).name == "askpass.sh" for e in envs)
    assert all(token not in " ".join(c) for c in git.chamadas) and token not in os.environ.values()
    assert not Path(envs[0]["GIT_ASKPASS"]).exists()                           # apagado ao sair
    with pytest.raises(RuntimeError):
        with codigo_colab.askpass(token) as env:
            script = Path(env["GIT_ASKPASS"])
            modo = (stat.S_IMODE(script.stat().st_mode), stat.S_IMODE(script.parent.stat().st_mode))
            assert modo == (0o700, 0o700) and token not in script.read_text(encoding="utf-8")
            raise RuntimeError("falha")
    assert not script.exists()


def test_proveniencia_registra_o_commit_clonado_e_a_origem_git(tmp_path, monkeypatch):
    destino = tmp_path / "RECLin-PT"
    info = codigo_colab.clonar_codigo("u/r", "main", destino, git=GitDoClone())
    monkeypatch.setattr(proveniencia, "identidade_do_codigo", lambda *a, **k: {"identidade": "x"}, raising=False)
    codigo = proveniencia.codigo_do_ambiente(Path(info["codigo"]))
    assert codigo["origem"] == "git" and codigo["revisao"] == SHA_CLONE and codigo["repositorio"] == "u/r"
    assert codigo["branch"] == "main" and codigo["tarball_sha256"] is None


# ------------------------------------------------------------------ apresentação do notebook simples (etapa 8.3.2)
def test_execucao_atual_pega_a_mais_recente_e_restaura_do_drive(tmp_path, monkeypatch, capsys):
    ctx = _ctx_com_chave(tmp_path, monkeypatch)
    with pytest.raises(ops.ErroOperacao, match="nenhuma execução"):
        apresentacao.execucao_atual(ctx)
    antiga = _execucao(ctx, _chave(lr=3e-5), ("iniciada", "concluida"))
    nova = _execucao(ctx, _chave(), ("iniciada", "interrompida"))
    for pasta, quando in ((antiga, "2026-01-01T00:00:00+00:00"), (nova, "2026-02-01T00:00:00+00:00")):
        dados = proveniencia.ler(pasta)
        dados["atualizado_em"] = quando
        (pasta / proveniencia.ARQUIVO).write_text(json.dumps(dados), encoding="utf-8")
    assert apresentacao.execucao_atual(ctx) == nova.name and nova.name in capsys.readouterr().out
    restaurado = []
    monkeypatch.setattr(apresentacao.persistencia, "restaurar", lambda *a, **k: restaurado.append(a[1]))
    monkeypatch.setattr(apresentacao.plano, "execucoes", lambda *a: [
        {"nome": "so_no_drive", "run_id": nova.name, "onde": "drive", "config_id": "c", "estado": "concluida"}])
    monkeypatch.setattr(apresentacao.persistencia, "raiz_das_execucoes", lambda r: Path(r))
    monkeypatch.setattr(apresentacao.proveniencia, "ler", lambda p: {"atualizado_em": "2026-03-01T00:00:00+00:00"})
    assert apresentacao.execucao_atual(ctx, restaurar=True) == "so_no_drive" and restaurado == ["so_no_drive"]


def test_recomendacao_com_parametros_do_notebook(tmp_path, monkeypatch):
    protocolo = {k: v for k, v in ESPEC.config_protocolo().como_dict().items() if k in ESPEC.permitidos}
    ctx = _ctx_com_chave(tmp_path / "a", monkeypatch, parametros=dict(protocolo))
    assert apresentacao.recomendar_ramo(ctx)["recomendada"] == ops.OP_NOVO
    ctx = _ctx_com_chave(tmp_path / "b", monkeypatch, parametros={**protocolo, "lr": 3e-5})
    r = apresentacao.recomendar_ramo(ctx)
    assert r["recomendada"] == ops.OP_NOVA and "EXPLORATÓRIA" in r["motivo"]
    _execucao(ctx, _chave(lr=3e-5), ("iniciada", "interrompida"))
    assert apresentacao.recomendar_ramo(ctx)["recomendada"] == ops.OP_RETOMAR      # compatível com o parâmetro novo
    ctx = _ctx_com_chave(tmp_path / "c", monkeypatch, parametros=dict(protocolo))
    _execucao(ctx, _chave(), ("iniciada", "concluida"))
    assert apresentacao.recomendar_ramo(ctx)["recomendada"] == ops.OP_NOVA


def test_operacao_4_com_parametros_alterados_exige_a_6_e_avisa_exploratoria(tmp_path, monkeypatch, capsys):
    ctx = _ctx_com_chave(tmp_path, monkeypatch, parametros={"lr": 3e-5}, motivo="sem motivo informado")
    quadro = apresentacao.quadro_pre_treino(ctx, ops.OP_NOVO)
    assert not quadro["pronto"] and any(ops.OP_NOVA in i and "EXPLORATÓRIA" in i for i in quadro["impedimentos"])
    quadro = apresentacao.quadro_pre_treino(ctx, ops.OP_NOVA)
    assert "EXPLORATÓRIA" in quadro["natureza"] and not any("MOTIVO" in i for i in quadro["impedimentos"])
    saida = capsys.readouterr().out
    assert "não elegível ao TEST" in saida and "≠ protocolo" in saida and "6b" in saida and "2.2" not in saida
    with pytest.raises(ops.ErroOperacao, match="PARAMETROS"):
        ops.treinar_novo(ctx)


def test_instalar_se_faltar_instala_sem_confirmacao_e_avisa_reinicio(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path)
    instalou = []
    monkeypatch.setattr(ops, "instalar_dependencias", lambda c: instalou.append(1) or {"ok": True})
    relatorio = {"ok": False, "avisos": [], "erros": [], "plano": {"instalar": True, "reiniciar": True, "motivos": ["falta x"]}}
    monkeypatch.setattr(apresentacao.ambiente, "verificar", lambda *a, **k: relatorio)
    assert apresentacao.instalar_se_faltar(ctx) == {"ok": True, "reiniciar": True} and instalou == [1]
    assert "falta x" in capsys.readouterr().out
    relatorio["plano"] = {"instalar": False, "reiniciar": False, "motivos": []}
    relatorio["ok"] = True
    instalou.clear()
    assert apresentacao.instalar_se_faltar(ctx) == {"ok": True, "reiniciar": False} and not instalou


def test_avisar_revisao_pesos_compara_com_o_hub_sem_baixar(tmp_path, monkeypatch, capsys):
    import sys, types

    atual = {"sha": "a" * 40}
    modulo = types.ModuleType("huggingface_hub")
    modulo.HfApi = lambda: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha=atual["sha"]))
    monkeypatch.setitem(sys.modules, "huggingface_hub", modulo)
    ctx = _ctx(tmp_path, revisao_pesos="a" * 40)
    assert apresentacao.avisar_revisao_pesos(ctx) == "a" * 40 and "AVISO" not in capsys.readouterr().out
    atual["sha"] = "c" * 40
    apresentacao.avisar_revisao_pesos(ctx)
    saida = capsys.readouterr().out
    assert "AVISO" in saida and "congelado" in saida and "continua usando REVISAO_PESOS" in saida
    modulo.HfApi = lambda: (_ for _ in ()).throw(OSError("sem rede"))
    assert apresentacao.avisar_revisao_pesos(ctx) is None and "não consegui consultar" in capsys.readouterr().out


def test_verificar_conexoes_diz_o_que_falta_sem_bloquear_o_treino(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(apresentacao.colab, "montar_drive", lambda: tmp_path)
    monkeypatch.setattr(apresentacao.segredos, "obter", lambda *a, **k: None)
    r = apresentacao.verificar_conexoes(repositorio="u/r", branch_resultados="resultados-colab", git_nome="<PREENCHER>",
                                        git_email="<PREENCHER>", repositorio_hf="<PREENCHER usuario/nome>",
                                        pasta_no_drive="RECLin-PT")
    saida = capsys.readouterr().out
    assert r["bloqueios"] == [] and r["raiz_drive"] == tmp_path / "RECLin-PT" and len(r["pendencias"]) == 5
    for esperado in ("GITHUB_TOKEN_ESCRITA", "HF_TOKEN", "GIT_NOME", "GIT_EMAIL", "REPOSITORIO_HF", "Acesso pelo notebook"):
        assert esperado in saida, esperado
    r = apresentacao.verificar_conexoes(repositorio="u/r", branch_resultados="main", git_nome="A", git_email="a@b.c",
                                        repositorio_hf="u/m", pasta_no_drive="RECLin-PT")
    assert any("main" in b for b in r["bloqueios"])
    assert "PARE" in capsys.readouterr().out
    def sem_drive():
        raise apresentacao.colab.ErroColab("não montado")
    monkeypatch.setattr(apresentacao.colab, "montar_drive", sem_drive)
    r = apresentacao.verificar_conexoes(repositorio="u/r", branch_resultados="resultados-colab", git_nome="A",
                                        git_email="a@b.c", repositorio_hf="u/m", pasta_no_drive="RECLin-PT")
    assert r["raiz_drive"] is None and any("Drive" in b for b in r["bloqueios"])
