"""Portão da etapa 4: o filtro de pistas, a regra pura e a calibração reproduzem,
byte a byte, os arquivos do legado.

Fonte: `referencia/resultados_legado/` (cópia sem alteração de `results/` do
legado, commit a5f055c). Entradas, as mesmas do legado: as partições
congeladas, as predições do DEV (calibração) e do TEST (filtro) dos quatro
baselines e, para a regra e o filtro, a calibração registrada.

* `CALIBRACAO_filtro.json` — `scripts/calibrar.py`, com uma pasta de partições
  SEM `test.jsonl`: a calibração não depende do TEST;
* a varredura da calibração — conferida, número a número, com as tabelas de
  `CALIBRACAO_filtro.md` (o legado só registrou a varredura nesse relatório,
  com 4 casas decimais);
* `filtro_<enc>_seed<N>.preds.json` (4) — `scripts/filtro_pistas.py` e
  `filtro_pistas.montar_predicoes`;
* `regra_pura.preds.json` — `scripts/regra_pura.py` e
  `regra_pura.montar_predicoes`.

As comparações são de bytes, sem tolerância.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from reclin import particoes, tarefa
from reclin.estrategias import filtro_pistas, regra_pura
from reclin.execucao import predicoes
from reclin.negacao import lexico
from reclin.util.caminhos import CODIGO, PARTICOES

RESULTADOS_LEGADO = CODIGO / "testes" / "referencia" / "resultados_legado"
SCRIPTS = CODIGO / "scripts"
CALIBRACAO = RESULTADOS_LEGADO / "CALIBRACAO_filtro.json"
BASES = ["baseline_biobertpt_seed42", "baseline_bertimbau_seed42",
         "baseline_biobertpt_seed43", "baseline_bertimbau_seed43"]


def rodar(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True)


def ok(r):
    assert r.returncode == 0, r.stdout + r.stderr
    return r


@pytest.fixture(scope="module")
def sem_test(tmp_path_factory):
    """Pasta com train, dev e o MANIFEST, mas sem o arquivo do TEST."""
    pasta = tmp_path_factory.mktemp("particoes_sem_test")
    for nome in ("train.jsonl", "dev.jsonl", "MANIFEST.json"):
        shutil.copyfile(PARTICOES / nome, pasta / nome)
    return pasta


@pytest.fixture(scope="module")
def calibracao(tmp_path_factory, sem_test):
    saida = tmp_path_factory.mktemp("calibracao")
    r = ok(rodar("calibrar.py", "--resultados", RESULTADOS_LEGADO, "--particoes", sem_test,
                 "--saida", saida / "CALIBRACAO_filtro.json", "--detalhes", saida / "detalhes.json",
                 "--conferir", CALIBRACAO))
    return {"arquivo": saida / "CALIBRACAO_filtro.json", "log": r.stdout,
            "detalhes": json.loads((saida / "detalhes.json").read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def particoes_lidas():
    assert particoes.conferir_particoes(PARTICOES) == []
    return particoes.ler_particoes(PARTICOES)


# --------------------------------------------------------------------------- #
# CALIBRACAO_filtro.json                                                      #
# --------------------------------------------------------------------------- #
def test_calibracao_identica_byte_a_byte(calibracao):
    assert calibracao["arquivo"].read_bytes() == CALIBRACAO.read_bytes()
    assert "idêntica, byte a byte" in calibracao["log"]


def test_calibracao_nao_le_o_test(sem_test):
    assert not (sem_test / "test.jsonl").exists()
    assert particoes.conferir_particoes(sem_test, ("train", "dev")) == []
    assert particoes.conferir_particoes(sem_test) == ["test.jsonl: arquivo ausente"]


def test_calibracao_escolhe_o_lexico_congelado(calibracao):
    lex = calibracao["detalhes"]["lexico"]
    assert lex["congelado"] is True
    assert lex["min_freq"] == lexico.CONGELADO["min_freq"]
    assert lex["lexico_sha1"] == lexico.CONGELADO["lexico_sha1"]


def _virgula(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def _tabela(md: str, titulo: str) -> list[list[str]]:
    """Linhas de dados da primeira tabela depois do título, sem os negritos."""
    trecho = md[md.index(titulo):]
    linhas = []
    for linha in trecho.splitlines():
        if linha.startswith("|"):
            if not set(linha) <= set("|-"):
                linhas.append([c.strip().strip("*") for c in linha.strip("|").split("|")])
        elif linhas:
            break
    return linhas[1:]                                     # sem o cabeçalho


def test_varredura_igual_ao_relatorio_do_legado(calibracao):
    md = (RESULTADOS_LEGADO / "CALIBRACAO_filtro.md").read_text(encoding="utf-8")
    det = calibracao["detalhes"]
    assert det["bases"] == BASES                          # a ordem das colunas do relatório
    assert f"Sao 19.064 candidatos e {det['n_negation_of']} pares" in md.replace("\n", " ")

    linhas = _tabela(md, "## 1. Varredura de `min_freq`")
    assert len(linhas) == len(det["min_freq"]) == 5
    for linha, r in zip(linhas, det["min_freq"]):
        f1 = [r["por_execucao"][b]["f1"] for b in det["bases"]]
        assert linha == [str(r["min_freq"]), str(r["n_formas"]), _virgula(r["cobertura"]),
                         *map(_virgula, f1), _virgula(r["f1_medio"])]

    linhas = _tabela(md, "## 2. Regra pura (R1-R4)")
    assert len(linhas) == len(det["regra"]) == 16
    for linha, r in zip(linhas, det["regra"]):
        assert linha == [r["regra"], str(r["gap"]), str(r["tp"]), str(r["fp"]), str(r["fn"]),
                         _virgula(r["precision"]), _virgula(r["recall"]), _virgula(r["f1"])]

    linhas = _tabela(md, "## 3. Sistema combinado")
    assert len(linhas) == len(det["porta_gap"]) == 7
    for linha, r in zip(linhas, det["porta_gap"]):
        f1 = [r["por_execucao"][b]["f1"] for b in det["bases"]]
        assert linha == [str(r["gap"]), *map(_virgula, f1), _virgula(r["f1_medio"])]

    formas = re.search(r"Lexico resultante.*?```\n(.*?)```", md, re.S).group(1).splitlines()
    assert formas == [f"{n:>6}  {f}" for f, n in det["lexico"]["formas"]]


def test_as_escolhas_nao_dependem_de_empate(calibracao):
    """Margem entre o escolhido e o segundo colocado: a escolha não muda com
    a última casa decimal da média (o legado usava `sum()` no Python 3.13;
    aqui, `math.fsum`)."""
    det = calibracao["detalhes"]
    for chave, campo in (("min_freq", "f1_medio"), ("porta_gap", "f1_medio"), ("regra", "f1")):
        valores = sorted((r[campo] for r in det[chave]), reverse=True)
        assert valores[0] - valores[1] > 1e-3, chave


def test_conferir_acusa_diferenca(tmp_path, sem_test):
    alterada = tmp_path / "ref.json"
    alterada.write_bytes(CALIBRACAO.read_bytes() + b"\n")
    r = rodar("calibrar.py", "--resultados", RESULTADOS_LEGADO, "--particoes", sem_test,
              "--saida", tmp_path / "c.json", "--conferir", alterada)
    assert r.returncode == 1 and "DIFERE" in r.stdout
    r = rodar("calibrar.py", "--resultados", RESULTADOS_LEGADO, "--particoes", sem_test,
              "--saida", tmp_path / "c.json")
    assert r.returncode == 1 and "--sobrescrever" in r.stdout


# --------------------------------------------------------------------------- #
# filtro_*.preds.json e regra_pura.preds.json                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def execucoes(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("execucoes")
    filtro = ok(rodar("filtro_pistas.py", "--resultados", RESULTADOS_LEGADO, "--calibracao", CALIBRACAO,
                      "--saida", raiz, "--conferir", RESULTADOS_LEGADO))
    regra = ok(rodar("regra_pura.py", "--calibracao", CALIBRACAO, "--saida", raiz,
                     "--conferir", RESULTADOS_LEGADO))
    return {"raiz": raiz, "log": filtro.stdout + regra.stdout}


@pytest.mark.parametrize("base", BASES)
def test_script_do_filtro_byte_a_byte(base, execucoes):
    nome = filtro_pistas.nome_execucao(base)
    gravado = execucoes["raiz"] / nome / "predicoes_test.json"
    assert gravado.read_bytes() == (RESULTADOS_LEGADO / f"{nome}.preds.json").read_bytes()
    assert (execucoes["raiz"] / nome / "predicoes_dev.json").exists()


def test_script_da_regra_byte_a_byte(execucoes):
    gravado = execucoes["raiz"] / "regra_pura" / "predicoes_test.json"
    assert gravado.read_bytes() == (RESULTADOS_LEGADO / "regra_pura.preds.json").read_bytes()
    assert "Conferência: 4 sidecars comparados, 4 idênticos" in execucoes["log"]
    assert "Conferência: 1 sidecars comparados, 1 idênticos" in execucoes["log"]


@pytest.mark.parametrize("base", BASES)
def test_modulo_do_filtro_byte_a_byte(base, particoes_lidas, tmp_path):
    lex = lexico.carregar_congelado(particoes_lidas["train"])
    p = filtro_pistas.montar_predicoes(
        predicoes.ler(RESULTADOS_LEGADO / f"{base}.preds.json"),
        tarefa.candidatos(particoes_lidas["test"]), lex,
        base_preds=f"{base}.preds.json", min_freq=3)
    predicoes.gravar(tmp_path / "f.json", p)
    nome = filtro_pistas.nome_execucao(base)
    assert (tmp_path / "f.json").read_bytes() == (RESULTADOS_LEGADO / f"{nome}.preds.json").read_bytes()


def test_modulo_da_regra_byte_a_byte(particoes_lidas, tmp_path):
    lex = lexico.carregar_congelado(particoes_lidas["train"])
    p = regra_pura.montar_predicoes(tarefa.candidatos(particoes_lidas["test"]), lex,
                                    regra="R3", max_target_gap=1, min_freq=3)
    predicoes.gravar(tmp_path / "r.json", p)
    assert (tmp_path / "r.json").read_bytes() == (RESULTADOS_LEGADO / "regra_pura.preds.json").read_bytes()


def test_execucoes_avaliaveis(execucoes):
    """As execuções novas são lidas pela avaliação como as do legado."""
    ok(rodar("avaliar.py", "--execucao", execucoes["raiz"] / "regra_pura"))
    metricas = json.loads((execucoes["raiz"] / "regra_pura" / "metricas.json").read_text(encoding="utf-8"))
    assert set(metricas) == {"dev", "test"}
    neg = metricas["dev"]["por_classe"]["negation_of"]
    assert (neg["tp"], neg["fp"], neg["fn"]) == (112, 61, 38)          # R3, gap<=1 no DEV


@pytest.mark.parametrize("script,args", [("filtro_pistas.py", ("--resultados", RESULTADOS_LEGADO)),
                                         ("regra_pura.py", ())])
def test_scripts_exigem_a_calibracao_do_lexico_congelado(script, args, tmp_path):
    """Uma calibração com outro `min_freq` não é aplicada com o léxico congelado."""
    outra = json.loads(CALIBRACAO.read_text(encoding="utf-8")) | {"min_freq": 2, "lexicon_size": 17}
    (tmp_path / "c.json").write_text(json.dumps(outra), encoding="utf-8")
    r = rodar(script, *args, "--calibracao", tmp_path / "c.json", "--saida", tmp_path / "x",
              "--particao", "dev")
    assert r.returncode == 1 and "CONGELADO" in r.stdout
    assert not (tmp_path / "x").exists()
