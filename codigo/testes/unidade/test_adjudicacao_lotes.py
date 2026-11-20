"""Lotes da adjudicação cega: o lote não pode revelar o grupo do caso."""
import csv

import pytest

from reclin.analises.adjudicacao_lotes import gravar_cego, montar
from reclin.analises.adjudicacao_negation_of import CONTROLE_FN, CONTROLE_VP, FALSO_POSITIVO
from reclin.tarefa import NEG, NOREL


def _casos():
    casos, y_true = [], []
    for i in range(30):                                   # 10 FP, 10 controles VP, 10 controles FN
        grupo = (FALSO_POSITIVO, CONTROLE_VP, CONTROLE_FN)[i % 3]
        casos.append({"caso": f"C{i:02d}", "documento_neutro": f"D{i % 4}", "grupo": grupo, "indice_candidato": i})
        y_true.append(NOREL if grupo == FALSO_POSITIVO else NEG)
    return casos, y_true


def test_lote_1_tem_os_fp_comuns_e_controles_na_mesma_proporcao():
    casos, y_true = _casos()
    comuns = {0, 3, 6, 9, 12, 15}                         # 6 dos 10 FP também são FP da outra execução
    y_outra = [NEG if (i in comuns or t == NEG) else NOREL for i, t in enumerate(y_true)]
    r = montar(casos, y_true, y_outra)
    assert r["fp_comuns_a_outra_execucao"] == 6 and r["fp_da_outra_execucao_fora_da_planilha"] == 0
    assert r["lotes"]["1"]["por_grupo"] == {FALSO_POSITIVO: 6, CONTROLE_VP: 6, CONTROLE_FN: 6}
    assert r["lotes"]["2"]["por_grupo"] == {FALSO_POSITIVO: 4, CONTROLE_VP: 4, CONTROLE_FN: 4}
    assert {c for c, comum in r["fp_comum_por_caso"].items() if comum} <= {c for c, n in r["lote_por_caso"].items() if n == 1}
    assert montar(casos, y_true, y_outra) == r            # determinístico


def test_arquivo_cego_so_tem_codigos(tmp_path):
    casos, y_true = _casos()
    y_outra = [NEG if (i in {0, 3} or t == NEG) else NOREL for i, t in enumerate(y_true)]
    r = montar(casos, y_true, y_outra)
    gravar_cego(tmp_path / "ordem.csv", casos, r["lote_por_caso"])
    texto = (tmp_path / "ordem.csv").read_text(encoding="utf-8-sig")
    linhas = list(csv.reader(texto.splitlines(), delimiter=";"))
    assert linhas[0] == ["lote", "ordem_no_lote", "caso", "documento"] and len(linhas) == 31
    for proibido in ("falso", "positivo", "controle", "negation", "relation", "indice"):
        assert proibido not in texto.lower()


def test_sem_dois_lotes_a_formar_e_recusado():
    casos, y_true = _casos()
    with pytest.raises(ValueError, match="não há dois lotes"):
        montar(casos, y_true, [NEG] * len(y_true))        # todos os FP são comuns
