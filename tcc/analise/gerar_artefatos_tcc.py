#!/usr/bin/env python3
"""Gera as tabelas LaTeX e as figuras do TCC a partir de:

* analise/resultados_dev.json      (analise_dev.py: métricas de DEV e comparações descritivas)
* analise/eda_v2/                  (scripts/gerar_tcc.py eda do repositório)
* analise/eda_complementar/        (eda_complementar.py)
* <repo>/codigo/dados/particoes-v2/MANIFEST.json e docs/entregas/d4/selecao_min_freq.json

Nada é digitado à mão: todo número das tabelas sai desses arquivos.
Uso: python3 gerar_artefatos_tcc.py <repo> <tcc/src>
"""
import hashlib
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROT = ["negation_of", "associated_with", "no_relation"]
FONTE_DEV = r"{\footnotesize Fonte: elaborado pelo autor.}"


def f(x, n=3):
    return f"{x:.{n}f}".replace(".", "{,}")


def milhar(x):
    return f"{x:,}".replace(",", "\\,")


def tt(s):
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def tabela(label, caption, colspec, cabecalho, linhas, fonte, resize=False, nota=None):
    corpo = "\n".join(" & ".join(l) + r" \\" if not l == ["\\midrule"] else r"\midrule" for l in linhas)
    tab = (f"  \\begin{{tabular}}{{{colspec}}}\n  \\toprule\n  {cabecalho} \\\\\n  \\midrule\n{corpo}\n"
           f"  \\bottomrule\n  \\end{{tabular}}")
    if resize:
        tab = "\\resizebox{\\textwidth}{!}{%\n" + tab + "}"
    extra = f"\\\\[2pt]\n  {{\\footnotesize {nota}}}" if nota else ""
    return (f"%% GERADO por analise/gerar_artefatos_tcc.py -- nao edite a mao.\n\\begin{{table}}[htbp]\n  \\centering\n"
            f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n{tab}{extra}\\\\[4pt]\n  {fonte}\n\\end{{table}}\n")


def main(repo: Path, src: Path) -> None:
    tab_dir, img_dir = src / "tabelas", src / "imagens" / "resultados"
    tab_dir.mkdir(exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    R = json.loads((AQUI / "resultados_dev.json").read_text())
    S = R["sistemas"]
    eda = json.loads((AQUI / "eda_v2" / "analises" / "corpus.json").read_text())["estatisticas"]
    mg = json.loads((AQUI / "eda_v2" / "analises" / "max_gap.json").read_text())
    comp = json.loads((AQUI / "eda_complementar" / "eda_complementar.json").read_text())
    fonte_eda = r"{\footnotesize Fonte: elaborado pelo autor.}"
    out = {}

    # ---------------- corpus ----------------
    linhas = []
    tot = {"doc": 0, "ent": 0, "neg": 0, "ass": 0, "cand": 0, "nr": 0}
    for p, nome in (("train", "Treino"), ("dev", "Desenvolvimento"), ("test", "Teste")):
        e = eda[p]
        r = e["relacoes_por_tipo"]; c = e["candidatos_por_rotulo"]
        linhas.append([nome, milhar(e["n_documentos"]), milhar(e["n_entidades"]), milhar(r["negation_of"]),
                       milhar(r["associated_with"]), milhar(e["n_candidatos"]), milhar(c["no_relation"])])
        for k, v in (("doc", e["n_documentos"]), ("ent", e["n_entidades"]), ("neg", r["negation_of"]),
                     ("ass", r["associated_with"]), ("cand", e["n_candidatos"]), ("nr", c["no_relation"])):
            tot[k] += v
    linhas.append(["\\midrule"])
    linhas.append(["Total"] + [milhar(tot[k]) for k in ("doc", "ent", "neg", "ass", "cand", "nr")])
    out["corpus"] = tabela("tab:corpus", "Partições congeladas: documentos, entidades, relações anotadas e pares candidatos (\\texttt{max\\_gap}$=25$).",
                              "lrrrrrr", "Partição & Documentos & Entidades & \\texttt{negation\\_of} & \\texttt{associated\\_with} & Candidatos & \\texttt{no\\_relation}",
                              linhas, fonte_eda, resize=True)

    # ---------------- distância ----------------
    dr = comp["distancia_relacoes"]
    linhas = []
    for t in ("negation_of", "associated_with"):
        d = dr[t]
        linhas.append([tt(t), milhar(d["n"]), f(d["media"], 1), f(d["mediana"], 0), f(d["p75"], 0), f(d["p95"], 0),
                       milhar(d["max"]), f(100 * d["frac_acima_max_gap"], 2) + "\\%"])
    out["distancia"] = tabela("tab:distancia", "Distância em caracteres entre as entidades de cada relação anotada (corpus completo).",
                                 "lrrrrrrr", "Tipo & $n$ & Média & Mediana & p75 & p95 & Máx. & $>25$",
                                 linhas, r"{\footnotesize Fonte: elaborado pelo autor.}")

    # ---------------- cobertura (teto de recall) ----------------
    linhas = []
    for p, nome in (("train", "Treino"), ("dev", "Desenvolvimento"), ("test", "Teste")):
        cr = eda[p]["cobertura_relacoes"]
        linhas.append([nome] + [f"{milhar(cr[t]['cobertas'])}/{milhar(cr[t]['total'])} ({f(cr[t]['fracao'], 4)})" for t in ("negation_of", "associated_with")])
    out["cobertura"] = tabela("tab:teto_recall", "Relações anotadas que se tornam pares candidatos com \\texttt{max\\_gap}$=25$ (teto de \\emph{recall}), por partição.",
                                 "lrr", "Partição & \\texttt{negation\\_of} & \\texttt{associated\\_with}", linhas, fonte_eda)

    # ---------------- sensibilidade ao max_gap ----------------
    md = (AQUI / "eda_v2" / "tabelas" / "eda_max_gap_train_dev.csv").read_text().strip().splitlines()
    linhas = []
    for l in md[1:]:
        c = l.split(",")
        if c[0] not in ("5", "10", "15", "20", "25 (tarefa)", "30", "50", "100", "200"):
            continue
        linhas.append([c[0].replace(" (tarefa)", "$^{\\ast}$"), milhar(int(c[1])), f(float(c[2]), 4), f(float(c[3]), 4),
                       c[4].replace(".", "{,}")])
    out["max_gap"] = tabela("tab:max_gap", "Sensibilidade ao \\texttt{max\\_gap} (treino e desenvolvimento): pares candidatos, cobertura das relações e pares \\texttt{no\\_relation} por relação coberta. $^{\\ast}$ valor adotado.",
                               "lrrrr", "\\texttt{max\\_gap} & Candidatos & Cob. \\texttt{negation\\_of} & Cob. \\texttt{associated\\_with} & \\texttt{no\\_relation}/relação",
                               linhas, fonte_eda, resize=True)

    # ---------------- resultados de DEV: protocolo ----------------
    proto = ["baseline_bio", "calibrado_bio", "filtro_bio", "pairaware_bio", "twostage_bio"]
    nomes = {"baseline_bio": "Baseline", "calibrado_bio": "Baseline calibrado ($t=0{,}90$)", "filtro_bio": "Filtro léxico",
             "pairaware_bio": "Pair-Aware", "twostage_bio": "Two-Stage ($\\tau=0{,}95$)", "pamlp_bio": "Pair-Aware MLP",
             "baseline_bert": "Baseline (BERTimbau)"}

    def linha_neg(k, nome=None):
        s = S[k]; c = s["por_classe"]["negation_of"]
        return [nome or nomes[k], f(c["precision"]), f(c["recall"]), f(c["f1"]), f"{c['tp']}/{c['fp']}/{c['fn']}",
                f(s["macro_f1"]), f(s["mcc"])]

    def melhor(linhas, cols):
        for j in cols:
            vals = [float(l[j].replace("{,}", ".")) for l in linhas]
            m = max(vals)
            for l, v in zip(linhas, vals):
                if v == m:
                    l[j] = "\\textbf{" + l[j] + "}"
        return linhas

    cab_neg = "Sistema & P & R & F1 & VP/FP/FN & Macro-F1 & MCC"
    out["dev_negacao"] = tabela("tab:dev_negacao",
                                "Resultados no conjunto de desenvolvimento (DEV) dos sistemas do protocolo, BioBERTpt-all, seed 42: precisão (P), \\emph{recall} (R) e F1 de \\texttt{negation\\_of}, macro-F1 e MCC. Melhor valor por coluna em negrito.",
                                "lrrrrrr", cab_neg, melhor([linha_neg(k) for k in proto], [1, 2, 3, 5, 6]), FONTE_DEV, resize=True,
                                nota="DEV, $n=19\\,064$ pares (150 \\texttt{negation\\_of}). Não são resultados de teste. $\\tau$ e $t$ foram escolhidos neste mesmo DEV.")

    def linha_outras(k, nome=None):
        s = S[k]; pc = s["por_classe"]
        return [nome or nomes[k]] + [f(pc[c][m]) for c in ("associated_with", "no_relation") for m in ("precision", "recall", "f1")]

    out["dev_outras_classes"] = tabela("tab:dev_outras",
                                       "Precisão, \\emph{recall} e F1 de \\texttt{associated\\_with} e \\texttt{no\\_relation} no DEV, sistemas do protocolo (BioBERTpt-all, seed 42).",
                                       "lrrrrrr", "& \\multicolumn{3}{c}{\\texttt{associated\\_with}} & \\multicolumn{3}{c}{\\texttt{no\\_relation}} \\\\\n  \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n  Sistema & P & R & F1 & P & R & F1",
                                       melhor([linha_outras(k) for k in proto], [1, 2, 3, 4, 5, 6]), FONTE_DEV)

    # ---------------- encoders ----------------
    def linha_enc(k, nome, params):
        s = S[k]; pc = s["por_classe"]
        return [nome, params] + [f(pc[c]["f1"]) for c in ROT] + [f(pc["negation_of"]["precision"]), f(pc["negation_of"]["recall"]), f(s["macro_f1"]), f(s["mcc"])]

    H = R["historico_treino"]
    out["dev_encoders"] = tabela("tab:dev_encoders",
                                 "Comparação de \\emph{encoders} na estratégia baseline, no DEV (seed 42, mesma configuração; só o \\emph{checkpoint} muda): F1 por classe, P e R de \\texttt{negation\\_of}, macro-F1 e MCC.",
                                 "llrrrrrrr", "\\emph{Encoder} & Parâm. & F1 neg & F1 assoc & F1 no\\_rel & P neg & R neg & Macro-F1 & MCC",
                                 [linha_enc("baseline_bio", "BioBERTpt-all", f"{H['baseline_biobertpt_seed42']['n_parametros']/1e6:.1f}".replace(".", "{,}") + " M"),
                                  linha_enc("baseline_bert", "BERTimbau-base", f"{H['baseline_BERTimbau']['n_parametros']/1e6:.1f}".replace(".", "{,}") + " M")],
                                 FONTE_DEV, resize=True,
                                 nota="A execução BERTimbau-base está fora do escopo pré-registrado (que fixa o BioBERTpt-all) e não tem nome oficial. BioBERTpt-clin: sem resultado.")

    # ---------------- exploratória ----------------
    out["dev_exploratoria"] = tabela("tab:dev_exploratoria",
                                     "Resultado exploratório no DEV (fora do protocolo pré-registrado): Pair-Aware MLP ao lado da Pair-Aware do protocolo, BioBERTpt-all, seed 42.",
                                     "lrrrrrr", cab_neg, [linha_neg("pairaware_bio", "Pair-Aware (protocolo, D10)"), linha_neg("pamlp_bio", "Pair-Aware MLP (exploratória)")],
                                     FONTE_DEV, resize=True)

    # ---------------- comparações descritivas ----------------
    nm = {"twostage_bio": "Two-Stage", "baseline_bio": "Baseline", "calibrado_bio": "Baseline calibrado", "filtro_bio": "Filtro léxico",
          "pairaware_bio": "Pair-Aware", "pamlp_bio": "Pair-Aware MLP", "baseline_bert": "BERTimbau-base (baseline)"}
    linhas = []
    for c in R["comparacoes_dev_descritivas"]:
        b, m = c["bootstrap"], c["mcnemar"]
        nome_b = "BioBERTpt-all (baseline)" if c["a"] == "baseline_bert" else nm[c["b"]]
        linhas.append([f"{nm[c['a']]} $-$ {nome_b}", ("$" + f"{c['diferenca']:+.4f}" + "$").replace(".", "{,}"),
                       f"$[{b['ci95_low']:+.4f};\\,{b['ci95_high']:+.4f}]$".replace(".", "{,}"), f(b["p_value"], 3),
                       f"{m['b_only_a_correct']}/{m['c_only_b_correct']}", ("$<0{,}001$" if m["p_value"] < 0.001 else f(m["p_value"], 3))])
    out["dev_comparacoes"] = tabela("tab:dev_comparacoes",
                                    "Comparações pareadas \\textbf{descritivas} no DEV: diferença no F1 de \\texttt{negation\\_of}, IC95\\% e valor-$p$ do \\emph{bootstrap} pareado (10.000 reamostras, semente 42) e McNemar exato sobre os acertos globais ($b/c$: só A acerta / só B acerta).",
                                    "lrrrrr", "Comparação (A $-$ B) & $\\Delta$F1 neg & IC95\\% & $p$ & $b/c$ & $p$ McNemar",
                                    linhas, r"{\footnotesize Fonte: elaborado pelo autor.}",
                                    resize=True,
                                    nota="Descritivas: $\\tau$ (Two-Stage) e $t$ (calibrado) foram escolhidos neste DEV, e a época de cada modelo também; não substituem as comparações pré-registradas no TEST.")

    # ---------------- histórico de treino ----------------
    def retomada(k):
        """Primeira sessão registrada: (época retomada, passos executados nela, passos por época) ou None."""
        t = json.loads((repo / "codigo" / "resultados" / "execucoes" / k / "treino.json").read_text())
        a = t["sessoes"][0].get("a_partir_de")
        return None if not a else (a["epoca"], a["passos_por_epoca"] - a["passo_na_epoca"], a["passos_por_epoca"])

    def dur(k, e):
        r = retomada(k)
        m = "" if not r else ("$^{\\dagger}$" if e["epoch"] == r[0] else ("$^{\\ddagger}$" if e["epoch"] < r[0] else ""))
        return f(e["duration_s"] / 60, 1) + m

    def nota_dur(ks, nomes):
        partes = []
        for k, n in zip(ks, nomes):
            ep, feitos, total = retomada(k)
            partes.append(f"{n}, época {ep}: {milhar(feitos)} dos {milhar(total)} passos")
        return ("$^{\\dagger}$ Época interrompida e retomada a partir de um \\emph{checkpoint}; a duração cobre apenas a sessão final (" + "; ".join(partes) + "). "
                "$^{\\ddagger}$ Época concluída em sessão anterior, sem registro de sessão preservado; a duração consta do histórico de treino, mas não pode ser conferida com o registro da sessão. "
                "As durações não são comparáveis entre execuções como medida de custo.")

    rotulo_exec = {"baseline_biobertpt_seed42": "Baseline (BioBERTpt-all)", "baseline_BERTimbau": "Baseline (BERTimbau-base)",
                   "pairaware_biobertpt_seed42": "Pair-Aware", "pair_aware_mlp_BioBERTpt-all": "Pair-Aware MLP"}
    linhas = []
    for k, nome in rotulo_exec.items():
        h = H[k]
        for i, e in enumerate(h["dev_history"]):
            marca = "$\\ast$" if e["epoch"] == h["melhor_epoca"] else ""
            linhas.append([nome if i == 0 else "", f"{e['epoch']}{marca}", f(e["train_loss"], 4), f(e["dev_loss"], 4),
                           f(e["dev_macro_f1"]), f(e["dev_negation_of_f1"]), dur(k, e)])
        linhas.append(["\\midrule"])
    linhas.pop()
    out["dev_history"] = tabela("tab:dev_history", "Evolução por época no DEV das execuções de três rótulos. $\\ast$ marca a época selecionada (maior macro-F1 no DEV).",
                                "llrrrrr", "Execução & Época & Perda (treino) & Perda (DEV) & Macro-F1 & F1 neg & Duração (min)",
                                linhas, r"{\footnotesize Fonte: elaborado pelo autor.}", resize=True,
                                nota=nota_dur(["baseline_BERTimbau", "pair_aware_mlp_BioBERTpt-all"], ["BERTimbau-base", "Pair-Aware MLP"]))
    e1, e2 = H["twostage_e1_biobertpt_seed42"], H["twostage_e2_biobertpt_seed42"]
    linhas = []
    for e in e1["dev_history"]:
        linhas.append([f"{e['epoch']}" + ("$\\ast$" if e["epoch"] == e1["melhor_epoca"] else ""), f(e["train_loss"], 4), f(e["dev_loss"], 4),
                       f(e["dev_ap_relacao"]), f(e["dev_precision_relacao"]), f(e["dev_recall_relacao"]),
                       f(e["dev_recall_relacao_nos_pares_negation_of"]), dur("twostage_e1_biobertpt_seed42", e)])
    out["dev_estagio1"] = tabela("tab:dev_estagio1", "Two-Stage, estágio 1 (relação $\\times$ \\texttt{no\\_relation}) no DEV completo: perdas, AP, precisão e \\emph{recall} de \\texttt{relacao}, \\emph{recall} nos pares \\texttt{negation\\_of} e duração. $\\ast$ época selecionada (maior AP).",
                                 "lrrrrrrr", "Época & Perda (treino) & Perda (DEV) & AP & P & R & R nos pares neg & Duração (min)", linhas,
                                 r"{\footnotesize Fonte: elaborado pelo autor.}", resize=True,
                                 nota=nota_dur(["twostage_e1_biobertpt_seed42"], ["estágio 1"]))
    linhas = []
    for e in e2["dev_history"]:
        linhas.append([f"{e['epoch']}" + ("$\\ast$" if e["epoch"] == e2["melhor_epoca"] else ""), f(e["train_loss"], 4), f(e["dev_loss"], 4),
                       f(e["dev_precision_negation_of_relacoes_anotadas"]), f(e["dev_recall_negation_of_relacoes_anotadas"]),
                       f(e["dev_negation_of_f1_relacoes_anotadas"]), f(e["duration_s"] / 60, 1)])
    out["dev_estagio2"] = tabela("tab:dev_estagio2", "Two-Stage, estágio 2 (\\texttt{negation\\_of} $\\times$ \\texttt{associated\\_with}) nos 913 pares do DEV com relação anotada: perdas, precisão, \\emph{recall} e F1 de \\texttt{negation\\_of} e duração. $\\ast$ época selecionada (maior F1 de \\texttt{negation\\_of}).",
                                 "lrrrrrr", "Época & Perda (treino) & Perda (DEV) & P neg & R neg & F1 neg & Duração (min)", linhas,
                                 r"{\footnotesize Fonte: elaborado pelo autor.}")

    # ---------------- tau (apêndice) ----------------
    linhas = []
    for t in R["tau_two_stage"]["tabela"]:
        c = t["contagens"]["negation_of"]
        linhas.append([f(t["tau"], 2) + ("$^{\\ast}$" if t["tau"] == R["tau_two_stage"]["tau"] else ""), milhar(t["n_roteados"]),
                       f"{c['tp']}/{c['fp']}/{c['fn']}", f(t["f1_negation_of"], 4), f(t["macro_f1"], 4)])
    out["tau_completo"] = tabela("tab:tau", "Busca de $\\tau$ do Two-Stage no DEV (pipeline completo, fluxo real). $^{\\ast}$ valor congelado.",
                                 "lrrrr", "$\\tau$ & Pares roteados & VP/FP/FN neg & F1 neg & Macro-F1", linhas,
                                 r"{\footnotesize Fonte: elaborado pelo autor.}")

    # ---------------- apêndice A: SHA-256 ----------------
    part = repo / "codigo" / "dados" / "particoes-v2"
    man = json.loads((part / "MANIFEST.json").read_text())
    linhas = []
    for n, nome in (("train", "Treino"), ("dev", "Desenvolvimento"), ("test", "Teste")):
        h = hashlib.sha256((part / f"{n}.jsonl").read_bytes()).hexdigest()
        linhas.append([nome, f"{n}.jsonl", "\\texttt{\\small " + h[:32] + "}\\\\\\texttt{\\small " + h[32:] + "}"])
    hm = hashlib.sha256((part / "MANIFEST.json").read_bytes()).hexdigest()
    linhas.append(["Manifesto", "MANIFEST.json", "\\texttt{\\small " + hm[:32] + "}\\\\\\texttt{\\small " + hm[32:] + "}"])
    out["sha256_particoes"] = ("%% GERADO por analise/gerar_artefatos_tcc.py\n\\begin{quadro}[htbp]\n  \\centering\n  \\caption{SHA-256 das partições congeladas e do manifesto.}\n  \\label{quad:sha256}\n"
                        "  \\begin{tabular}{llc}\n  \\toprule\n  Partição & Arquivo & SHA-256 \\\\\n  \\midrule\n"
                        + "\n".join(f"  {a} & \\texttt{{{b.replace('_', chr(92) + '_')}}} & \\begin{{tabular}}[c]{{@{{}}c@{{}}}}{c}\\end{{tabular}} \\\\" for a, b, c in linhas)
                        + "\n  \\bottomrule\n  \\end{tabular}\\\\[4pt]\n  {\\footnotesize Fonte: elaborado pelo autor.}\n\\end{quadro}\n")

    # ---------------- apêndice B: léxico D4 ----------------
    d4 = json.loads((repo / "codigo" / "docs" / "entregas" / "d4" / "selecao_min_freq.json").read_text())
    lx = d4["lexico_final"]
    linhas = [[tt(k), milhar(v)] for k, v in sorted(lx["formas"].items(), key=lambda kv: -kv[1])]
    out["lexico_d4"] = ("%% GERADO por analise/gerar_artefatos_tcc.py\n\\begin{quadro}[htbp]\n  \\centering\n"
                        f"  \\caption{{Léxico de pistas congelado (D4): \\texttt{{min\\_freq}}$={lx['min_freq']}$, {lx['n_formas']} formas, \\texttt{{lexico\\_sha1}} \\texttt{{{lx['lexico_sha1']}}}.}}\n  \\label{{quad:lexico}}\n"
                        "  \\begin{tabular}{lr}\n  \\toprule\n  Forma normalizada & Ocorrências como $e_1$ (treino) \\\\\n  \\midrule\n"
                        + "\n".join("  " + " & ".join(l) + r" \\" for l in linhas)
                        + "\n  \\bottomrule\n  \\end{tabular}\\\\[4pt]\n  {\\footnotesize Fonte: elaborado pelo autor.}\n\\end{quadro}\n")

    for nome, txt in out.items():
        (tab_dir / f"{nome}.tex").write_text(txt, encoding="utf-8")

    # ---------------- figuras ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    AZUL, LARANJA, TINTA, MUTED = "#2a78d6", "#eb6834", "#2b2b2b", "#6b6b6b"

    def virg(ax):
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))

    # τ
    tb = R["tau_two_stage"]["tabela"]
    x = [t["tau"] for t in tb]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x, [t["f1_negation_of"] for t in tb], color=AZUL, lw=2, marker="o", ms=5, label="F1 de negation_of")
    ax.plot(x, [t["macro_f1"] for t in tb], color=LARANJA, lw=2, marker="s", ms=5, label="Macro-F1")
    ax.axvline(R["tau_two_stage"]["tau"], color=MUTED, ls="--", lw=1)
    ax.text(0.935, 0.665, "τ congelado = 0,95", color=TINTA, fontsize=10, ha="right")
    ax.text(x[-1] + 0.012, tb[-1]["f1_negation_of"], "F1 neg", color=TINTA, va="center", fontsize=10)
    ax.text(x[-1] + 0.012, tb[-1]["macro_f1"], "Macro-F1", color=TINTA, va="center", fontsize=10)
    ax.set_xlabel("τ (limiar de P(relação) do estágio 1)")
    ax.set_ylabel("métrica no DEV")
    ax.set_xlim(0, 1.06)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85])
    ax.set_ylim(0.64, 0.86)
    virg(ax)
    fig.tight_layout()
    fig.savefig(img_dir / "tau_two_stage.png", dpi=200)
    plt.close(fig)

    # matrizes de confusão
    sis = [("baseline_bio", "Baseline"), ("pairaware_bio", "Pair-Aware"), ("twostage_bio", "Two-Stage")]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("az", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    for ax, (k, titulo) in zip(axs, sis):
        M = np.array(S[k]["matriz_confusao"]["matriz"], dtype=float)
        N = M / M.sum(axis=1, keepdims=True)
        ax.imshow(N, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{N[i, j]:.2f}".replace(".", ",") + f"\n({int(M[i, j])})", ha="center", va="center",
                        fontsize=9, color="white" if N[i, j] > 0.55 else TINTA)
        ax.set_xticks(range(3), ["neg", "assoc", "no_rel"])
        ax.set_yticks(range(3), ["neg", "assoc", "no_rel"])
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("predita")
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("verdadeira")
    fig.tight_layout()
    fig.savefig(img_dir / "matrizes_confusao_dev.png", dpi=200)
    plt.close(fig)

    # distância (EDA)
    import shutil
    shutil.copy(AQUI / "eda_complementar" / "eda_distancia_v2.png", img_dir / "eda_distancia.png")
    print("tabelas:", sorted(out), "| figuras em", img_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
