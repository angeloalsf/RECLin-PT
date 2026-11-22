#!/usr/bin/env python3
"""Gera as tabelas do TCC acrescentadas na Etapa 5, a partir de arquivos já gravados:

* <repo>/codigo/resultados/auditoria_execucoes/metricas_dev.json e comparacoes_dev.json
  (`python codigo/diagnosticos/auditar_execucoes.py`);
* <repo>/codigo/resultados/diagnosticos/precisao_recall_sem_pesos_x_ponderada.json
  (`python codigo/diagnosticos/comparar_precisao_recall.py`);
* <repo>/codigo/resultados/diagnosticos/associated_with_sem_direcao.json
  (`python codigo/diagnosticos/avaliar_assoc_sem_direcao.py`).

Nada é digitado à mão nem recalculado aqui: todo número sai desses arquivos, que
só têm contagens, métricas e ids de execução (nenhum texto clínico). Todos os
números são do DEV; nenhum vem do TEST.

Tabelas: `dev_sem_pesos` (substitui a gerada por `gerar_tabela_sem_pesos.py`, agora
com o bootstrap por documento), `dev_sem_pesos_comparacoes`, `dev_controles_encoders`
`dev_assoc_sem_direcao` e, depois da ablação do pooler (M3),
`dev_ablacao_pooler_sistemas` e `dev_ablacao_pooler`; depois da decisão da configuração final
(`codigo/resultados/diagnosticos/configuracao_final_dev.json`), `dev_limiar_negation_of` e
`dev_configuracao_final`. As tabelas de `gerar_tabelas_etapa3.py` não são regeradas:
as que estão em `tcc/src/tabelas/` receberam ajustes de redação depois de geradas.

Uso: python3 gerar_tabelas_etapa5.py <repo> <tcc/src>
"""
import json
import sys
from pathlib import Path

FONTE = r"{\footnotesize Fonte: elaborado pelo autor.}"
CABECALHO = "%% GERADO por analise/gerar_tabelas_etapa5.py -- nao edite a mao.\n"
NEG, ASSOC = "negation_of", "associated_with"
ALL, CLIN, BERT = "BioBERTpt-all", "BioBERTpt-clin", "BERTimbau"
NOME_ENC = {ALL: "BioBERTpt-all", CLIN: "BioBERTpt-clin", BERT: "BERTimbau-base"}
NOME_EST = {"pair_aware": "Pair-Aware Linear", "pair_aware_sem_pesos": "Pair-Aware Linear, sem pesos",
            "baseline": "Baseline", "two_stage": "Two-Stage", "pair_aware_mlp": "Pair-Aware MLP",
            "baseline_calibrado": "Baseline calibrado", "filtro_pistas": "Filtro léxico"}
METRICAS = (("f1_negation_of", "F1 de \\texttt{negation\\_of}"),
            ("f1_associated_with", "F1 de \\texttt{associated\\_with}"), ("macro_f1", "Macro-F1"))


def f(x, n=3):
    return f"{x:.{n}f}".replace(".", "{,}")


def s(x, n=3):
    return f"${'+' if x >= 0 else '-'}{f(abs(x), n)}$"


def limite(x, n=3):
    """Limite de IC com sinal; se arredondaria a zero sem ser zero, usa mais casas."""
    if x == 0:
        return f(0.0, n)
    d = n
    while round(abs(x), d) == 0 and d < 6:
        d += 1
    return ("+" if x > 0 else "-") + f(abs(x), d)


def ic(r, n=3):
    """IC95% como gravado; * quando o intervalo exclui o zero (sem arredondamento)."""
    a, b = r["ic95"]
    return f"$[{limite(a, n)};\\,{limite(b, n)}]$" + (r"$^{\ast}$" if (a > 0 or b < 0) else "")


def ic_texto(r, n=3):
    a, b = r["ic95"]
    return f"[${limite(a, n)}$; ${limite(b, n)}$]"


def tabela(label, caption, colspec, cabecalho, linhas, nota):
    corpo = "\n".join(l if isinstance(l, str) else " & ".join(l) + r" \\" for l in linhas)
    return (f"{CABECALHO}\\begin{{table}}[htbp]\n  \\centering\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n"
            f"  \\small\\setlength{{\\tabcolsep}}{{4pt}}\n\\adjustbox{{max width=\\textwidth}}{{%\n"
            f"  \\begin{{tabular}}{{{colspec}}}\n  \\toprule\n  {cabecalho} \\\\\n  \\midrule\n{corpo}\n"
            f"  \\bottomrule\n  \\end{{tabular}}}}\\\\[2pt]\n  {{\\footnotesize {nota}}}\\\\[4pt]\n  {FONTE}\n\\end{{table}}\n")


def comparacao(comp, a, b):
    achadas = [c for c in comp["comparacoes"] if c["a"] == a and c["b"] == b]
    if len(achadas) != 1:
        raise SystemExit(f"comparação {a} × {b}: {len(achadas)} registros em comparacoes_dev.json")
    return achadas[0]


# --------------------------------------------------------------------------- #
def t_sem_pesos(met, comp, pr):
    sis = met["sistemas"]
    linhas = []
    for rot, k in (("ponderada (\\texttt{balanced})", f"pair_aware|{ALL}"), ("sem pesos de classe", f"pair_aware_sem_pesos|{ALL}")):
        m = sis[k]
        n, a = m["por_classe"][NEG], m["por_classe"][ASSOC]
        linhas.append([rot, f(n["precision"]), f(n["recall"]), f(n["f1"]), f"{n['tp']}/{n['fp']}/{n['fn']}",
                       f(a["precision"]), f(a["recall"]), f(a["f1"]), f(m["macro_f1"]), f(m["mcc"])])
    if pr["a_execucao"] != sis[f"pair_aware_sem_pesos|{ALL}"]["pasta"] or pr["b_execucao"] != sis[f"pair_aware|{ALL}"]["pasta"]:
        raise SystemExit("o arquivo de precisão e recall não é o das duas execuções da tabela")
    d = pr["por_documento"]
    nota = ("DEV, $n=19\\,064$ pares candidatos, semente 42, 3 épocas. Diferença (sem pesos $-$ ponderada) e IC95\\% do "
            "\\emph{bootstrap} pareado por documento ($98$ notas): precisão de \\texttt{negation\\_of} "
            f"{s(d['precision']['diferenca'])} {ic_texto(d['precision'])}; \\emph{{recall}} "
            f"{s(d['recall']['diferenca'])} {ic_texto(d['recall'])}; F1 {s(d['f1']['diferenca'])} {ic_texto(d['f1'])}. "
            "As diferenças de F1 das outras classes e do macro-F1 estão na Tabela~\\ref{tab:dev_sem_pesos_comparacoes}. "
            "Experimento exploratório; não é resultado de teste.")
    return tabela("tab:dev_sem_pesos",
                  "Pair-Aware Linear com o BioBERTpt-all no DEV, com e sem ponderação de classes na perda.",
                  "lrrrrrrrrr", "Perda & P neg & R neg & F1 neg & VP/FP/FN neg & P assoc & R assoc & F1 assoc & Macro-F1 & MCC",
                  linhas, nota)


def t_sem_pesos_comparacoes(met, comp):
    a = f"pair_aware_sem_pesos|{ALL}"
    def com_limiar(nome, k):                    # o limiar do rótulo é o gravado na execução
        m = met["sistemas"][k]
        return nome + "".join(f" (${simbolo}={f(m[campo], 2)}$)" for campo, simbolo in (("tau", "\\tau"), ("t", "t")) if campo in m)
    outros = [("Pair-Aware Linear ponderada", f"pair_aware|{ALL}"), ("Baseline", f"baseline|{ALL}"),
              (com_limiar("Two-Stage", f"two_stage|{ALL}"), f"two_stage|{ALL}"),
              (com_limiar("Baseline calibrado", f"baseline_calibrado|{ALL}"), f"baseline_calibrado|{ALL}")]
    linhas = []
    for i, (chave, titulo) in enumerate(METRICAS):
        if i:
            linhas.append("\\midrule")
        linhas.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Métrica:}} {titulo}}} \\\\")
        for rot, b in outros:
            c = comparacao(comp, a, b)
            linhas.append([f"\\quad {rot}", s(c["por_candidato"][chave]["diferenca"], 4),
                           ic(c["por_candidato"][chave]), ic(c["por_documento"][chave])])
    nota = ("A = Pair-Aware Linear treinada sem pesos de classe (BioBERTpt-all). \\emph{Bootstrap} pareado, $10\\,000$ "
            "reamostras, semente 42, IC95\\% por percentis; $^{\\ast}$ o intervalo exclui o zero. Descritivas, no DEV, com "
            "uma única semente: as épocas, $\\tau$ e $t$ foram escolhidos neste mesmo conjunto. Não são resultados de teste.")
    return tabela("tab:dev_sem_pesos_comparacoes",
                  "Pair-Aware Linear sem pesos de classe frente à versão ponderada, ao baseline, ao Two-Stage e ao "
                  "baseline calibrado, no DEV (A $-$ B).",
                  "lrcc", "B & $\\Delta$ & IC95\\% (candidato) & IC95\\% (documento)", linhas, nota)


def t_controles_encoders(met, comp):
    sis = met["sistemas"]
    linhas = []
    for i, enc in enumerate((ALL, CLIN, BERT)):
        if i:
            linhas.append("\\midrule")
        for j, est in enumerate(("pair_aware", "baseline_calibrado", "filtro_pistas")):
            m = sis[f"{est}|{enc}"]
            n = m["por_classe"][NEG]
            nome = NOME_EST[est] + (f" ($t={f(m['t'], 2)}$)" if "t" in m else "")
            if est == "pair_aware":
                deltas = ["--", "--", "--", "--"]
            else:
                c = comparacao(comp, f"pair_aware|{enc}", f"{est}|{enc}")["por_documento"]
                deltas = [s(c["f1_negation_of"]["diferenca"]), ic(c["f1_negation_of"]),
                          s(c["macro_f1"]["diferenca"]), ic(c["macro_f1"])]
            linhas.append([NOME_ENC[enc] if j == 0 else "", nome, f(n["precision"]), f(n["recall"]), f(n["f1"]),
                           f"{n['tp']}/{n['fp']}/{n['fn']}", f(m["macro_f1"]), *deltas])
    nota = ("DEV, $n=19\\,064$ pares candidatos, semente 42. $\\Delta$ = Pair-Aware Linear $-$ controle, com o mesmo "
            "\\emph{encoder}; IC95\\% do \\emph{bootstrap} pareado por documento ($10\\,000$ reamostras); $^{\\ast}$ o "
            "intervalo exclui o zero. O limiar $t$ de cada baseline calibrado foi escolhido neste mesmo DEV, na grade "
            "de $0{,}05$ a $0{,}95$; o léxico do filtro é o congelado antes da fase. Os controles do BioBERTpt-clin e "
            "do BERTimbau-base foram calculados depois da fase, a partir das predições já gravadas dos baselines. Não "
            "são resultados de teste.")
    return tabela("tab:dev_controles_encoders",
                  "Controles (baseline calibrado e filtro léxico) nos três \\emph{encoders}, no DEV, e diferença da "
                  "Pair-Aware Linear para cada um.",
                  "llrrrrrrcrc", "\\emph{Encoder} & Sistema & P neg & R neg & F1 neg & VP/FP/FN & Macro-F1 & "
                  "$\\Delta$ F1 neg & IC95\\% & $\\Delta$ macro-F1 & IC95\\%", linhas, nota)


def t_assoc_sem_direcao(sd):
    sis = sd["sistemas"]
    linhas, anterior = [], None
    for est in ("pair_aware", "pair_aware_sem_pesos", "baseline", "two_stage", "pair_aware_mlp"):
        for enc in (ALL, CLIN, BERT):
            m = sis.get(f"{est}|{enc}")
            if m is None:
                continue
            if anterior is not None and anterior != est:
                linhas.append("\\midrule")
            c, d = m["com_direcao"], m["sem_direcao"]
            linhas.append([NOME_EST[est] if anterior != est else "", NOME_ENC[enc], f(c["precision"]), f(c["recall"]),
                           f(c["f1"]), f(d["precision"]), f(d["recall"]), f(d["f1"]), s(m["diferenca_f1"])])
            anterior = est
    um = next(iter(sis.values()))
    if any(m["sem_direcao"]["pares_sem_o_sentido_inverso"] for m in sis.values()):
        raise SystemExit("há pares sem o sentido inverso entre os candidatos: revise a nota da tabela")
    nota = (f"DEV. Com direção: os ${um['com_direcao']['n_pares_ordenados']:,}$ pares ordenados da tarefa "
            f"(${um['com_direcao']['suporte']}$ \\texttt{{associated\\_with}}). Sem direção: os "
            f"${um['sem_direcao']['n_pares_nao_ordenados']:,}$ pares não ordenados; um par é positivo na referência se "
            f"algum dos sentidos está anotado como \\texttt{{associated\\_with}} (${um['sem_direcao']['suporte']}$ pares) e "
            "é positivo na predição se o modelo previu a classe em algum dos sentidos. Análise secundária e descritiva: "
            "não altera a tarefa nem as métricas principais. Não são resultados de teste.").replace(",", "\\,", 2)
    return tabela("tab:dev_assoc_sem_direcao",
                  "Classe \\texttt{associated\\_with} avaliada com e sem a direção do par, no DEV.",
                  "llrrrrrrr", "Estratégia & \\emph{Encoder} & P & R & F1 & P s/ dir. & R s/ dir. & F1 s/ dir. & $\\Delta$ F1",
                  linhas, nota)


ABLACAO = (("pair_aware|BioBERTpt-all", "cls_direto|BioBERTpt-all", "Ler os marcadores, sem \\emph{pooler}"),
           ("pair_aware_pooler|BioBERTpt-all", "baseline|BioBERTpt-all", "Ler os marcadores, com \\emph{pooler}"),
           ("cls_direto|BioBERTpt-all", "baseline|BioBERTpt-all", "Retirar o \\emph{pooler}, só \\texttt{[CLS]}"),
           ("pair_aware|BioBERTpt-all", "pair_aware_pooler|BioBERTpt-all",
            "Retirar o \\emph{pooler}, com marcadores"))
CELULAS = (("baseline", "só \\texttt{[CLS]}", "com"), ("cls_direto", "só \\texttt{[CLS]}", "sem"),
           ("pair_aware_pooler", "\\texttt{[CLS]}+\\texttt{[E1]}+\\texttt{[E2]}", "com"),
           ("pair_aware", "\\texttt{[CLS]}+\\texttt{[E1]}+\\texttt{[E2]}", "sem"))


def t_ablacao_pooler_sistemas(met):
    linhas = []
    for est, rep, pooler in CELULAS:
        m = met["sistemas"][f"{est}|{ALL}"]
        n, a = m["por_classe"][NEG], m["por_classe"][ASSOC]
        nome = {"cls_direto": "\\texttt{cls\\_direto}", "pair_aware_pooler": "\\texttt{pair\\_aware\\_pooler}",
                "baseline": "Baseline", "pair_aware": "Pair-Aware Linear"}[est]
        linhas.append([nome, rep, pooler, f(n["precision"]), f(n["recall"]), f(n["f1"]), f(a["f1"]), f(m["macro_f1"])])
    nota = ("BioBERTpt-all, semente 42, mesmos hiperparâmetros e mesma entrada (o texto leva os marcadores nas quatro "
            "células); muda só o que a cabeça lê e se o \\texttt{[CLS]} passa pelo \\emph{pooler}. "
            "\\texttt{cls\\_direto} e \\texttt{pair\\_aware\\_pooler} são exploratórias, posteriores ao protocolo e não "
            "elegíveis ao teste. Não são resultados de teste.")
    return tabela("tab:dev_ablacao_pooler_sistemas",
                  "Fatorial da ablação do \\emph{pooler}: métricas das quatro cabeças no DEV.",
                  "lllrrrrr", "Cabeça & Representação & \\emph{Pooler} & P neg & R neg & F1 neg & F1 assoc & Macro-F1",
                  linhas, nota)


def t_ablacao_pooler(comp):
    linhas = []
    for chave, rotulo in METRICAS:
        linhas.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Métrica:}} {rotulo}}} \\\\")
        for a, b, efeito in ABLACAO:
            c = comparacao(comp, a, b)
            linhas.append([f"\\quad {efeito}", s(c["por_candidato"][chave]["diferenca"], 4),
                           ic(c["por_candidato"][chave]), ic(c["por_documento"][chave])])
        linhas.append("\\midrule")
    linhas.pop()
    nota = ("Cada linha compara duas cabeças que diferem num único fator (A $-$ B): ler os marcadores é "
            "Pair-Aware Linear $-$ \\texttt{cls\\_direto} e \\texttt{pair\\_aware\\_pooler} $-$ baseline; retirar o "
            "\\emph{pooler} é \\texttt{cls\\_direto} $-$ baseline e Pair-Aware Linear $-$ \\texttt{pair\\_aware\\_pooler}. "
            "\\emph{Bootstrap} pareado, $10\\,000$ reamostras, semente 42, IC95\\% por percentis; $^{\\ast}$ o intervalo "
            "exclui o zero. Descritivas, no DEV, com uma única semente. Não são resultados de teste.")
    return tabela("tab:dev_ablacao_pooler",
                  "Ablação do \\emph{pooler}: efeitos simples de cada fator no DEV (BioBERTpt-all).",
                  "lrcc", "Efeito & $\\Delta$ & IC95\\% (candidato) & IC95\\% (documento)", linhas, nota)


def t_limiar_negation_of(cf):
    """A busca de t da configuração final: a grade inteira, com o ponto escolhido em negrito."""
    linhas = []
    for linha in cf["tabela"]:
        c = linha["contagens"][NEG]
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"])
        celulas = [f(linha["limiar"], 2), str(c["tp"]), str(c["fp"]), str(c["fn"]), f(p), f(r),
                   f(linha["f1_negation_of"]), f(linha["macro_f1"]), "sim" if r >= 0.8 else "não"]
        if linha["limiar"] == cf["t"]:
            celulas = [f"\\textbf{{{x}}}" for x in celulas]
        linhas.append(celulas)
    nota = ("DEV, Pair-Aware Linear sem pesos de classe (BioBERTpt-all, semente 42). Regra: \\texttt{negation\\_of} "
            "se P(\\texttt{negation\\_of}) $\\geq t$; senão, o argmax entre \\texttt{associated\\_with} e "
            "\\texttt{no\\_relation}. Critério: maior precisão de \\texttt{negation\\_of} entre os limiares com "
            "\\emph{recall} de pelo menos 80\\%; desempates: maior F1 de \\texttt{negation\\_of}, maior macro-F1, menor "
            "limiar. Em negrito, o limiar escolhido. Critério formulado depois de conhecida a curva (pós-DEV). "
            "Não são resultados de teste.")
    return tabela("tab:dev_limiar_negation_of",
                  "Escolha do limiar da configuração final: \\texttt{negation\\_of} em função de $t$ no DEV.",
                  "rrrrrrrrc", "$t$ & VP & FP & FN & P & R & F1 & Macro-F1 & R $\\geq 80\\%$", linhas, nota)


def t_configuracao_final(cf):
    """A configuração final (A) contra as alternativas (B), no DEV."""
    metricas_ic = (("precision", "Precisão de \\texttt{negation\\_of}", "precisao_recall_negation_of"),
                   ("recall", "\\emph{Recall} de \\texttt{negation\\_of}", "precisao_recall_negation_of"),
                   ("f1_negation_of", "F1 de \\texttt{negation\\_of}", "f1_e_macro"),
                   ("macro_f1", "Macro-F1", "f1_e_macro"))
    linhas = []
    for chave, rotulo, fonte in metricas_ic:
        linhas.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Métrica:}} {rotulo}}} \\\\")
        for nome, comp in cf["comparacoes"].items():
            r = comp[fonte]
            cand, doc = r["por_candidato"][chave], r["por_documento"][chave]
            rotulo_b = comp["rotulo"].replace("τ", "$\\tau$")
            linhas.append(["\\quad " + rotulo_b, s(cand["diferenca"]), ic(cand), ic(doc)])
        linhas.append("\\midrule")
    linhas.pop()
    m = cf["metricas_dev"]
    neg = m["por_classe"][NEG]
    nota = (f"A = configuração final (Pair-Aware Linear sem pesos, $t={f(cf['t'], 2)}$): P ${f(neg['precision'])}$, "
            f"R ${f(neg['recall'])}$, F1 ${f(neg['f1'])}$, F1 de \\texttt{{associated\\_with}} "
            f"${f(m['f1_per_class'][ASSOC])}$, macro-F1 ${f(m['macro_f1'])}$. \\emph{{Bootstrap}} pareado, "
            "$10\\,000$ reamostras, semente 42, IC95\\% por percentis; $^{\\ast}$ o intervalo exclui o zero. "
            "Descritivas, no DEV, com uma única semente; $t$ e $\\tau$ foram escolhidos neste mesmo conjunto. "
            "Não são resultados de teste.")
    return tabela("tab:dev_configuracao_final",
                  "Configuração final frente às alternativas no DEV (A $-$ B).",
                  "lrcc", "B & $\\Delta$ & IC95\\% (candidato) & IC95\\% (documento)", linhas, nota)


def main(repo: Path, src: Path) -> None:
    ler = lambda *p: json.loads(repo.joinpath("codigo", "resultados", *p).read_text(encoding="utf-8"))  # noqa: E731
    met, comp = ler("auditoria_execucoes", "metricas_dev.json"), ler("auditoria_execucoes", "comparacoes_dev.json")
    pr = ler("diagnosticos", "precisao_recall_sem_pesos_x_ponderada.json")
    sd = ler("diagnosticos", "associated_with_sem_direcao.json")
    cf = ler("diagnosticos", "configuracao_final_dev.json")
    if cf.get("test_lido") is not False:
        raise SystemExit("configuracao_final_dev.json não declara que o TEST não foi lido")
    if pr.get("test_lido") is not False or sd["proveniencia"].get("test_lido") is not False:
        raise SystemExit("os arquivos de diagnóstico não declaram que o TEST não foi lido")
    saidas = {"dev_sem_pesos": t_sem_pesos(met, comp, pr), "dev_sem_pesos_comparacoes": t_sem_pesos_comparacoes(met, comp),
              "dev_controles_encoders": t_controles_encoders(met, comp), "dev_assoc_sem_direcao": t_assoc_sem_direcao(sd),
              "dev_ablacao_pooler_sistemas": t_ablacao_pooler_sistemas(met), "dev_ablacao_pooler": t_ablacao_pooler(comp),
              "dev_limiar_negation_of": t_limiar_negation_of(cf), "dev_configuracao_final": t_configuracao_final(cf)}
    for nome, texto in saidas.items():
        (src / "tabelas" / f"{nome}.tex").write_text(texto, encoding="utf-8", newline="\n")
    print("tabelas:", sorted(saidas))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
