"""Os scripts `comparar.py` e `avaliar.py` de ponta a ponta, sobre sidecars pequenos."""
from __future__ import annotations

import json
import subprocess
import sys

from reclin.avaliacao import significancia
from reclin.execucao import diretorio, predicoes
from reclin.util import caminhos
from reclin.util.io import sha256_json

SCRIPTS = caminhos.CODIGO / "scripts"
Y_TRUE = [0, 0, 0, 1, 1, 2, 2, 2] * 3
Y_A = [0, 0, 2, 1, 0, 2, 2, 1] * 3
Y_B = [0, 2, 2, 1, 1, 2, 2, 2] * 3


def rodar(script, *args):
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_comparar_par_usa_a_semente_de_b_quando_a_nao_tem(tmp_path):
    a = predicoes.montar(model="regra", seed=None, y_true=Y_TRUE, y_pred=Y_A)
    b = predicoes.montar(model="modelo", seed=43, y_true=Y_TRUE, y_pred=Y_B)
    predicoes.gravar(tmp_path / "a.preds.json", a)
    predicoes.gravar(tmp_path / "b.preds.json", b)
    rodar("comparar.py", "par", "--a", tmp_path / "a.preds.json", "--b", tmp_path / "b.preds.json",
          "--saida", tmp_path / "s.json", "--n-boot", 30)
    esperado = significancia.relatorio_json(significancia.comparar(a, b, seed=43, n_boot=30))
    assert (tmp_path / "s.json").read_text(encoding="utf-8") == esperado


def test_avaliar_arquivos_e_execucao(tmp_path):
    p = predicoes.montar(model="m", seed=1, y_true=Y_TRUE, y_pred=Y_A)
    predicoes.gravar(tmp_path / "a.preds.json", p)
    rodar("avaliar.py", tmp_path / "a.preds.json", "--saida", tmp_path / "m.json")
    gravado = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert list(gravado.values())[0]["n"] == len(Y_TRUE)

    pasta = diretorio.criar(tmp_path, "exec", estrategia="baseline", config={})
    conjunto = {"particao": "dev", "n_candidatos": len(Y_TRUE), "y_true_sha256": sha256_json(Y_TRUE)}
    diretorio.gravar_predicoes(pasta, "dev", p, conjunto)
    rodar("avaliar.py", "--execucao", pasta)
    assert set(diretorio.ler_metricas(pasta)) == {"dev"}
