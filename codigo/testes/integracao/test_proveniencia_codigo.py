"""D6 de ponta a ponta em CPU: uma execução criada por outra versão do código
(ou sem a identidade) não é retomada, não tem o TEST avaliado e não serve de
base para execuções derivadas. BERT sintético criado no teste; partições
pequenas com um TEST sintético (linhas do DEV): o TEST real não é lido."""
from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from reclin import particoes  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.estrategias import baseline, baseline_calibrado  # noqa: E402
from reclin.execucao import diretorio  # noqa: E402
from reclin.treino import classificador  # noqa: E402
from reclin.util import identidade_codigo  # noqa: E402
from reclin.util.caminhos import CODIGO, PARTICOES  # noqa: E402
from reclin.util.io import gravar_json, ler_json  # noqa: E402

PEQ = {"batch_size": 16, "max_length": 64, "lr": 1e-3, "epochs": 1}


@pytest.fixture(scope="module")
def part(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("particoes_d6")
    train = (PARTICOES / "train.jsonl").read_bytes().splitlines(keepends=True)
    dev = (PARTICOES / "dev.jsonl").read_bytes().splitlines(keepends=True)
    (pasta / "train.jsonl").write_bytes(b"".join(train[:6]))
    (pasta / "dev.jsonl").write_bytes(b"".join(dev[:4]))
    (pasta / "test.jsonl").write_bytes(b"".join(dev[4:8]))           # TEST sintético
    particoes.gravar_manifesto(pasta, particoes.calcular_manifesto(
        pasta, seed=42, origem={"path": "teste D6 (test sintético)", "sha256": None}))
    return pasta


def _outra_versao(pasta):
    """Simula uma execução criada por outra versão do código: um registro
    consistente, com um arquivo do manifesto diferente."""
    conteudo = ler_json(pasta / "config.json")
    manifesto = {**conteudo["codigo"]["arquivos"], "reclin/tarefa.py": "0" * 64}
    conteudo["codigo"] = {**conteudo["codigo"], "arquivos": manifesto,
                          "identidade": identidade_codigo.identidade_de(manifesto)}
    gravar_json(pasta / "config.json", conteudo)


def test_retomada_com_outra_versao_do_codigo(tmp_path, part, modelo_sintetico):
    cfg = Config(**PEQ)
    r = classificador.treinar_execucao(tmp_path, "b", cfg, modelo=modelo_sintetico, pasta_particoes=part,
                                       montagem=baseline.montagem(cfg), parar_apos_passo=1)
    assert not r["concluido"] and r["sessoes"][0]["codigo"] == identidade_codigo.atual()["identidade"]
    _outra_versao(tmp_path / "b")
    with pytest.raises(classificador.ErroExecucao, match="outra versão do código"):
        classificador.treinar_execucao(tmp_path, "b", cfg, modelo=modelo_sintetico, pasta_particoes=part,
                                       montagem=baseline.montagem(cfg), retomar=True)


def test_test_e_derivadas_com_outra_versao_do_codigo(tmp_path, part, modelo_sintetico):
    cfg = Config(**PEQ)
    classificador.treinar_execucao(tmp_path, "b", cfg, modelo=modelo_sintetico, pasta_particoes=part,
                                   montagem=baseline.montagem(cfg))
    copia = tmp_path / "c"
    shutil.copytree(tmp_path / "b", copia)
    gravar_json(copia / "config.json", {**ler_json(copia / "config.json"), "nome": "c"})
    _outra_versao(copia)
    with pytest.raises(classificador.ErroExecucao, match="outra versão do código"):
        classificador.avaliar_test(copia, pasta_particoes=part, montagem=baseline.montagem(cfg))
    assert not (copia / "predicoes_test.json").exists()
    with pytest.raises(baseline_calibrado.ErroCalibracao, match="outra versão do código"):
        baseline_calibrado.calibrar_dev(copia, tmp_path, "cal")
    r = subprocess.run([sys.executable, str(CODIGO / "scripts" / "filtro_pistas.py"), "--base-execucao", copia,
                        "--particao", "dev", "--particoes", part, "--saida", tmp_path / "f"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "outra versão do código" in r.stdout
    # sem a identidade (execução anterior à D6): também recusada, sem completar os metadados
    conteudo = ler_json(copia / "config.json")
    conteudo.pop("codigo")
    gravar_json(copia / "config.json", conteudo)
    with pytest.raises(classificador.ErroExecucao, match="anterior à D6"):
        classificador.avaliar_test(copia, pasta_particoes=part, montagem=baseline.montagem(cfg))
    assert "codigo" not in diretorio.ler_config(copia)
    # a original, do código atual, segue normalmente
    classificador.avaliar_test(tmp_path / "b", pasta_particoes=part, montagem=baseline.montagem(cfg))
