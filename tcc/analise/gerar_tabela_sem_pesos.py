"""Tabela do experimento sem ponderação de classes (Pair-Aware Linear, DEV).

Lê apenas as predições do DEV gravadas pelas duas execuções da Pair-Aware Linear
com o BioBERTpt-all (com e sem pesos de classe), recalcula as métricas com
`reclin.avaliacao.metricas.avaliar` e aplica o bootstrap pareado por candidato
(10.000 reamostras, semente 42), no mesmo esquema de `reclin.avaliacao.significancia`
(um sorteio `integers(0, n, n)` por reamostra). Não treina, não executa modelos e
não lê o conjunto de teste.

Uso (a partir da raiz do repositório, com a branch de resultados disponível):

    python tcc/analise/gerar_tabela_sem_pesos.py \
        --execucoes codigo/resultados/execucoes \
        --saida tcc/tcc/src/tabelas/dev_sem_pesos.tex

O bootstrap por documento não é calculado aqui, porque exige a partição do DEV;
para incluí-lo, registre a execução em `codigo/diagnosticos/auditar_execucoes.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reclin.avaliacao.metricas import avaliar
from reclin.avaliacao.significancia import f1_classe
from reclin.tarefa import ASSOC, NEG, NOREL

COM_PESOS = "pairaware_biobertpt_seed42"
SEM_PESOS = "pair_aware__a3ac671a77e2__t2"
N_BOOT, SEED = 10_000, 42


def _carregar(pasta: Path, nome: str) -> dict:
    pred = json.loads((pasta / nome / "predicoes_dev.json").read_text(encoding="utf-8"))
    if pred.get("split") != "dev":
        raise ValueError(f"{nome}: predições de {pred.get('split')!r}; só o DEV é aceito")
    return pred


def _macro(y: np.ndarray, p: np.ndarray) -> float:
    return (f1_classe(y, p, NEG) + f1_classe(y, p, ASSOC) + f1_classe(y, p, NOREL)) / 3.0


def _bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Diferença A − B observada e IC95% (percentis) de três métricas, nas mesmas reamostras."""
    metricas = {"f1_neg": lambda t, p: f1_classe(t, p, NEG),
                "f1_assoc": lambda t, p: f1_classe(t, p, ASSOC), "macro": _macro}
    rng = np.random.default_rng(SEED)
    n = len(y)
    difs = {k: np.empty(N_BOOT) for k in metricas}
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for k, f in metricas.items():
            difs[k][i] = f(y[idx], a[idx]) - f(y[idx], b[idx])
    saida = {}
    for k, f in metricas.items():
        baixo, alto = np.percentile(difs[k], [2.5, 97.5])
        saida[k] = (f(y, a) - f(y, b), float(baixo), float(alto))
    return saida


def _br(x: float, casas: int = 3, sinal: bool = False) -> str:
    texto = f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}"
    return texto.replace(".", "{,}").replace("-", "$-$") if sinal else texto.replace(".", "{,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execucoes", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    args = ap.parse_args()

    com, sem = _carregar(args.execucoes, COM_PESOS), _carregar(args.execucoes, SEM_PESOS)
    if com["y_true"] != sem["y_true"]:
        raise ValueError("as duas execuções não têm o mesmo y_true")
    y = np.array(com["y_true"])
    linhas = []
    for rotulo, pred in (("ponderada (\\texttt{balanced})", com), ("sem pesos de classe", sem)):
        m = avaliar(pred["y_true"], pred["y_pred"])
        neg, assoc = m["por_classe"]["negation_of"], m["por_classe"]["associated_with"]
        linhas.append(
            f"{rotulo} & {_br(neg['precision'])} & {_br(neg['recall'])} & {_br(neg['f1'])} & "
            f"{neg['tp']}/{neg['fp']}/{neg['fn']} & {_br(assoc['precision'])} & {_br(assoc['recall'])} & "
            f"{_br(assoc['f1'])} & {_br(m['macro_f1'])} & {_br(m['mcc'])} \\\\")
    b = _bootstrap(y, np.array(sem["y_pred"]), np.array(com["y_pred"]))

    def dif(k: str) -> str:
        d, lo, hi = b[k]
        return f"{_br(d, sinal=True)} [{_br(lo, sinal=True)}; {_br(hi, sinal=True)}]"

    tex = f"""%% GERADO por analise/gerar_tabela_sem_pesos.py -- nao edite a mao.
\\begin{{table}}[htbp]
  \\centering
  \\caption{{Pair-Aware Linear com o BioBERTpt-all no DEV, com e sem ponderação de classes na perda.}}
  \\label{{tab:dev_sem_pesos}}
  \\small\\setlength{{\\tabcolsep}}{{4pt}}
\\adjustbox{{max width=\\textwidth}}{{%
  \\begin{{tabular}}{{lrrrrrrrrr}}
  \\toprule
  Perda & P neg & R neg & F1 neg & VP/FP/FN neg & P assoc & R assoc & F1 assoc & Macro-F1 & MCC \\\\
  \\midrule
{linhas[0]}
{linhas[1]}
  \\bottomrule
  \\end{{tabular}}}}\\\\[2pt]
  {{\\footnotesize DEV, $n=19\\,064$ pares candidatos, semente 42, 3 épocas. Diferença (sem pesos $-$ ponderada) e IC95\\% do \\emph{{bootstrap}} pareado por candidato: F1 de \\texttt{{negation\\_of}} {dif('f1_neg')}; F1 de \\texttt{{associated\\_with}} {dif('f1_assoc')}; macro-F1 {dif('macro')}. Experimento exploratório; não é resultado de teste.}}\\\\[4pt]
  {{\\footnotesize Fonte: elaborado pelo autor.}}
\\end{{table}}
"""
    args.saida.write_text(tex, encoding="utf-8", newline="\n")
    print(f"gravado: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
