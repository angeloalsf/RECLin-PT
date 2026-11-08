#!/usr/bin/env python3
"""Gera as tabelas LaTeX do TCC que vêm dos diagnósticos da Etapa 2, a partir de:

* <repo>/codigo/resultados/diagnosticos/controles_tipo_semantico.json
* <repo>/codigo/resultados/diagnosticos/sensibilidade_semente.json
* <repo>/codigo/resultados/diagnosticos/quase_duplicatas.json

(gerados por `python codigo/diagnosticos/diagnosticar.py tudo`). Nada é digitado à
mão: todo número das tabelas sai desses arquivos, que contêm só contagens,
métricas e ids de documentos (nenhum texto clínico).

Uso: python3 gerar_tabelas_diagnosticos.py <repo> <tcc/src>
"""
import json
import sys
from pathlib import Path

FONTE = r"{\footnotesize Fonte: elaborado pelo autor.}"
CABECALHO = "%% GERADO por analise/gerar_tabelas_diagnosticos.py -- nao edite a mao.\n"


def f(x, n=3):
    return f"{x:.{n}f}".replace(".", "{,}")


def milhar(x):
    return f"{x:,}".replace(",", "\\,")


def tt(s):
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def tabela(label, caption, colspec, cabecalho, linhas, resize=False, nota=None):
    corpo = "\n".join(r"\midrule" if l == ["\\midrule"] else " & ".join(l) + r" \\" for l in linhas)
    tab = (f"  \\begin{{tabular}}{{{colspec}}}\n  \\toprule\n  {cabecalho} \\\\\n  \\midrule\n{corpo}\n"
           f"  \\bottomrule\n  \\end{{tabular}}")
    if resize:
        tab = "\\resizebox{\\textwidth}{!}{%\n" + tab + "}"
    extra = f"\\\\[2pt]\n  {{\\footnotesize {nota}}}" if nota else ""
    return (f"{CABECALHO}\\begin{{table}}[htbp]\n  \\centering\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n"
            f"{tab}{extra}\\\\[4pt]\n  {FONTE}\n\\end{{table}}\n")


def controles(d: dict) -> str:
    nomes = {"train": "Treino", "dev": "DEV"}
    linhas = []
    for chave, rotulo in (("ent", "Tipo \\texttt{Negation} $\\rightarrow$ pista (entidades)"),
                          ("par", "Tipo \\texttt{Negation} $\\rightarrow$ \\texttt{negation\\_of} (pares)"),
                          ("pos", "Tipo \\texttt{Negation} + posição (pares)")):
        for i, p in enumerate(("train", "dev")):
            v = d["particoes"][p]
            m = {"ent": v["controle_tipo"]["nivel_entidade"], "par": v["controle_tipo"]["nivel_par"],
                 "pos": v["controle_tipo_posicao"]}[chave]
            linhas.append([rotulo if i == 0 else "", nomes[p], milhar(m["tp"]), milhar(m["fp"]), milhar(m["fn"]),
                           f(m["precision"]), f(m["recall"]), f(m["f1"])])
    nota = ("Controles diagnósticos informados pelas anotações, sem treino; não são configurações do RECLin-PT. "
            "Pares: os candidatos da tarefa (\\texttt{max\\_gap}$=25$). Teste não avaliado.")
    return tabela("tab:controles_tipo",
                  "Controles diagnósticos com o tipo semântico \\texttt{Negation} anotado, no treino e no DEV.",
                  "llrrrrrr", "Controle & Partição & TP & FP & FN & P & R & F1", linhas, resize=True, nota=nota)


def semente(d: dict) -> str:
    nomes = {"dev": "DEV", "test": "Teste"}
    linhas = []
    for t in ("negation_of", "associated_with"):
        for p in ("dev", "test"):
            r = d["resumo"][p][t]
            linhas.append([tt(t), nomes[p], milhar(r["min"]), milhar(r["p5"]), milhar(r["mediana"]),
                           milhar(r["p95"]), milhar(r["max"]), milhar(r["semente_de_referencia"]),
                           f(100 * r["percentil_da_referencia"], 1) + "\\%"])
    nota = (f"Mesmo algoritmo de partição, sementes {d['sementes'][0]} a {d['sementes'][1]} "
            f"({d['n_sementes']} sementes); a partição congelada não foi alterada.")
    return tabela("tab:sensibilidade_semente",
                  "Relações anotadas por partição com outras sementes do particionamento e posição da semente 42.",
                  "llrrrrrrr", "Relação & Partição & mín. & p5 & mediana & p95 & máx. & semente 42 & percentil",
                  linhas, resize=True, nota=nota)


def duplicatas(d: dict) -> str:
    nomes = {"train": "treino", "dev": "DEV", "test": "teste"}
    linhas = []
    for p in sorted(d["pares"], key=lambda x: (int(x["doc_a"]), int(x["doc_b"]))):
        a = p["anotacao"]
        neg = a["relacoes_por_tipo"].get("negation_of")
        esp = p["diferencas"]["caracteres_por_classe"].get("espaco", 0)
        linhas.append([f"{p['doc_a']} / {p['doc_b']}", f"{nomes[p['particao_a']]} / {nomes[p['particao_b']]}",
                       str(esp),
                       f"{a['entidades']['a']} / {a['entidades']['b']} / {a['entidades']['em_comum']}",
                       f"{a['relacoes']['a']} / {a['relacoes']['b']} / {a['relacoes']['em_comum']}",
                       f"{neg['a']} / {neg['b']} / {neg['em_comum']}" if neg else "--"])
    par = d["parametros"]
    nota = (f"Jaccard de \\emph{{shingles}} de {par['k']} caracteres do texto sem espaços em branco, limiar "
            f"{f(par['limiar_jaccard'], 1)}. Todos os pares ficam idênticos sem os espaços. ``Em comum'': texto e tipo "
            "das entidades e dos argumentos das relações, sem \\emph{offsets}.")
    return tabela("tab:quase_duplicatas",
                  "Documentos quase duplicados identificados no corpus (as partições não foram alteradas).",
                  "llrlll", "Documentos & Partições & Espaços diferentes & Entidades (a/b/comum) & Relações (a/b/comum) "
                  "& \\texttt{negation\\_of} (a/b/comum)", linhas, resize=True, nota=nota)


def main(repo: Path, src: Path) -> None:
    diag = repo / "codigo" / "resultados" / "diagnosticos"
    tab_dir = src / "tabelas"
    tab_dir.mkdir(exist_ok=True)
    saidas = {
        "controles_tipo": controles(json.loads((diag / "controles_tipo_semantico.json").read_text(encoding="utf-8"))),
        "sensibilidade_semente": semente(json.loads((diag / "sensibilidade_semente.json").read_text(encoding="utf-8"))),
        "quase_duplicatas": duplicatas(json.loads((diag / "quase_duplicatas.json").read_text(encoding="utf-8"))),
    }
    for nome, texto in saidas.items():
        (tab_dir / f"{nome}.tex").write_text(texto, encoding="utf-8")
    print("tabelas:", sorted(saidas))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
