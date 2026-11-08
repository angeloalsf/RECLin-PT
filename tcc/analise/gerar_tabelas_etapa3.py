#!/usr/bin/env python3
"""Gera as tabelas e a figura do TCC que vêm da auditoria experimental da Etapa 3,
a partir de:

* <repo>/codigo/resultados/auditoria_execucoes/inventario.json
* <repo>/codigo/resultados/auditoria_execucoes/metricas_dev.json
* <repo>/codigo/resultados/auditoria_execucoes/comparacoes_dev.json
* <repo>/codigo/resultados/auditoria_execucoes/two_stage_tau.json
* <repo>/codigo/resultados/auditoria_execucoes/associated_with_dev.json

(gerados por `python codigo/diagnosticos/auditar_execucoes.py`). Nada é digitado
à mão: todo número sai desses arquivos, que contêm só contagens, métricas, ids
de execução e ids de documento (nenhum texto clínico). Todos os números são do
DEV; nenhum vem do TEST.

As tabelas antigas geradas por `gerar_artefatos_tcc.py` (dev_negacao,
dev_outras_classes, dev_comparacoes, dev_encoders, dev_history, tau_completo,
dev_exploratoria) ficam preservadas, mas deixam de ser usadas pelo texto, que
passa a usar as desta etapa (nomes novos, sem sobrescrever as antigas).

Revisão de 04/11/2026 (apresentação; nenhum número é recalculado aqui):
* comparações pareadas (dev_comparacoes_pa, _ts, _encoders) em retrato, em blocos
  verticais por métrica, com os IC95% por candidato e por documento das três
  métricas, como gravados em comparacoes_dev.json; o McNemar passa ao quadro
  complementar mcnemar_comparacoes;
* limites de IC que arredondariam a zero saem com mais casas;
* busca de τ (tau_encoders) em três blocos, um por encoder, em longtable;
* dev_exploratoria_mlp substitui, no Apêndice E, a tabela antiga dev_exploratoria
  (gerada por gerar_artefatos_tcc.py, preservada);
* \resizebox foi trocado por \adjustbox{max width}: tabelas só são reduzidas,
  nunca ampliadas.

Uso: python3 gerar_tabelas_etapa3.py <repo> <tcc/src>
"""
import json
import sys
from pathlib import Path

FONTE = r"{\footnotesize Fonte: elaborado pelo autor.}"
CABECALHO = "%% GERADO por analise/gerar_tabelas_etapa3.py -- nao edite a mao.\n"
ENCODERS = ("BioBERTpt-all", "BioBERTpt-clin", "BERTimbau")
NOME_ENC = {"BioBERTpt-all": "BioBERTpt-all", "BioBERTpt-clin": "BioBERTpt-clin", "BERTimbau": "BERTimbau-base"}
NOME_EST = {"pair_aware": "Pair-Aware linear", "baseline": "Baseline", "two_stage": "Two-Stage",
            "pair_aware_mlp": "Pair-Aware MLP", "baseline_calibrado": "Baseline calibrado",
            "filtro_pistas": "Filtro léxico", "two_stage_e1": "Two-Stage, estágio 1",
            "two_stage_e2": "Two-Stage, estágio 2"}
NEG, ASSOC, NOREL = "negation_of", "associated_with", "no_relation"


def f(x, n=3):
    return f"{x:.{n}f}".replace(".", "{,}")


def s(x, n=4):
    return f"${'+' if x >= 0 else '-'}{f(abs(x), n)}$"


def milhar(x):
    return f"{x:,}".replace(",", "\\,")


def dl(a, b, alinhamento="c"):
    """Cabeçalho em duas linhas, para reduzir a largura sem diminuir a fonte."""
    return f"\\begin{{tabular}}[b]{{@{{}}{alinhamento}@{{}}}}{a}\\\\{b}\\end{{tabular}}"


def tt(texto):
    return r"\texttt{" + texto.replace("_", r"\_") + "}"


def limite(x, n=3):
    """Limite de IC com sinal. Se o valor não é zero mas arredondaria a zero com n
    casas, usa mais casas (até 6), para que o arredondamento não contradiga a marcação
    de detecção. Zero exato sai sem sinal."""
    if x == 0:
        return f(0.0, n)
    d = n
    while round(abs(x), d) == 0 and d < 6:
        d += 1
    return ("+" if x > 0 else "-") + f(abs(x), d)


def ic(r, n=3):
    """IC95% como gravado; * quando o intervalo exclui o zero (limite inferior > 0 ou
    superior < 0, sem arredondamento)."""
    a, b = r["ic95"]
    marca = r"$^{\ast}$" if (a > 0 or b < 0) else ""
    return f"$[{limite(a, n)};\\,{limite(b, n)}]${marca}"


def ic_ou_traco(bloco, metrica):
    r = bloco.get(metrica) or {}
    return ic(r) if r.get("ic95") is not None else "--"


def tabela(label, caption, colspec, cabecalho, linhas, resize=False, nota=None, posicao="htbp", girada=False):
    corpo = "\n".join(r"\midrule" if l == ["\\midrule"] else " & ".join(l) + r" \\" for l in linhas)
    tab = (f"  \\begin{{tabular}}{{{colspec}}}\n  \\toprule\n  {cabecalho} \\\\\n  \\midrule\n{corpo}\n"
           f"  \\bottomrule\n  \\end{{tabular}}")
    if resize:
        # só reduz quando a tabela é mais larga que o texto; nunca amplia
        tab = "\\adjustbox{max width=\\textwidth}{%\n" + tab + "}"
    extra = f"\\\\[2pt]\n  {{\\footnotesize {nota}}}" if nota else ""
    ambiente = "sidewaystable" if girada else "table"
    pos = "" if girada else f"[{posicao}]"
    ajuste = "  \\small\\setlength{\\tabcolsep}{4pt}\n" if resize else ""
    return (f"{CABECALHO}\\begin{{{ambiente}}}{pos}\n  \\centering\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n{ajuste}"
            f"{tab}{extra}\\\\[4pt]\n  {FONTE}\n\\end{{{ambiente}}}\n")


def _linha_bloco(l, ncols):
    if l == ["\\midrule"]:
        return "\\midrule"
    if len(l) == 1 and l[0].startswith("BLOCO:"):
        return f"\\multicolumn{{{ncols}}}{{l}}{{{l[0][6:]}}} \\\\"
    if len(l) == 1 and l[0].startswith("SUB:"):
        return f"\\addlinespace[2pt]\\multicolumn{{{ncols}}}{{l}}{{\\quad\\textit{{{l[0][4:]}}}}} \\\\"
    return " & ".join(l) + r" \\"


def tabela_blocos(label, caption, colspec, cabecalho, linhas, ncols, nota=None):
    """Mesma apresentação em blocos da tabela_longa, mas como flutuante (cabe numa
    página): nota e fonte ficam sempre junto da tabela."""
    corpo = "\n".join(_linha_bloco(l, ncols) for l in linhas)
    extra = f"  \\parbox{{\\textwidth}}{{\\footnotesize {nota}}}\\\\[4pt]\n" if nota else ""
    return (f"{CABECALHO}\\begin{{table}}[htbp]\n  \\centering\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n"
            f"  \\small\\renewcommand{{\\arraystretch}}{{1.2}}\\setlength{{\\tabcolsep}}{{3.5pt}}\n"
            f"  \\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}\n  \\toprule\n"
            f"  {cabecalho} \\\\\n  \\midrule\n{corpo}\n  \\bottomrule\n  \\end{{tabular*}}\\\\[2pt]\n"
            f"{extra}  {FONTE}\n\\end{{table}}\n")


def tabela_longa(label, caption, colspec, cabecalho, linhas, ncols, nota=None, fonte="\\small", sep="4pt"):
    """Tabela multipágina (longtable) em retrato, com cabeçalho repetido. O longtable
    nunca divide uma linha; '\\\\*' depois do título de bloco impede quebra logo
    após ele."""
    def linha(l):
        if l == ["\\midrule"]:
            return "\\midrule"
        if len(l) == 1 and l[0].startswith("BLOCO:"):
            return f"\\multicolumn{{{ncols}}}{{l}}{{{l[0][6:]}}} \\\\*"
        if len(l) == 1 and l[0].startswith("SUB:"):
            return f"\\addlinespace[2pt]\\multicolumn{{{ncols}}}{{l}}{{\\quad\\textit{{{l[0][4:]}}}}} \\\\*"
        return " & ".join(l) + r" \\"
    corpo = "\n".join(linha(l) for l in linhas)
    # nota e fonte numa minipage: não se separam da última linha da tabela por quebra
    # de página dentro delas
    rodape = "\\nopagebreak\\noindent\\begin{minipage}{\\linewidth}\n"
    if nota:
        rodape += f"{{\\footnotesize {nota}\\par}}\\vspace{{4pt}}\n"
    rodape += f"{FONTE}\n\\end{{minipage}}\\par\n"
    return (f"{CABECALHO}\\begingroup{fonte}\\renewcommand{{\\arraystretch}}{{1.2}}\\setlength{{\\tabcolsep}}{{{sep}}}\n"
            f"\\setlength{{\\LTleft}}{{0pt}}\\setlength{{\\LTright}}{{0pt}}\\setlength{{\\LTcapwidth}}{{\\textwidth}}\n"
            f"\\Needspace*{{0.3\\textheight}}\n"
            f"\\begin{{longtable}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}} \\\\\n\\toprule\n{cabecalho} \\\\\n\\midrule\n\\endfirsthead\n"
            f"\\multicolumn{{{ncols}}}{{l}}{{\\footnotesize Tabela~\\thetable{{}} -- continuação}} \\\\\n"
            f"\\toprule\n{cabecalho} \\\\\n\\midrule\n\\endhead\n"
            f"\\midrule\n\\multicolumn{{{ncols}}}{{r}}{{\\footnotesize continua}} \\\\\n\\endfoot\n"
            f"\\bottomrule\n\\endlastfoot\n{corpo}\n\\end{{longtable}}\n"
            f"\\vspace{{-\\baselineskip}}\n{rodape}\\endgroup\n")


def ordem_sistemas(sistemas):
    """Pair-Aware linear, baseline, Two-Stage, MLP e controles; encoders na ordem fixa."""
    chaves = []
    for est in ("pair_aware", "baseline", "two_stage", "pair_aware_mlp", "baseline_calibrado", "filtro_pistas"):
        grupo = [f"{est}|{e}" for e in ENCODERS if f"{est}|{e}" in sistemas]
        if grupo:
            chaves.append(grupo)
    return chaves


def rotulo(m, curto=False):
    est = NOME_EST[m["estrategia"]]
    extra = ""
    if "tau" in m:
        extra = f" ($\\tau={f(m['tau'], 2)}$)"
    if "t" in m:
        extra = f" ($t={f(m['t'], 2)}$)"
    if curto:
        return NOME_ENC[m["encoder"]] + extra
    return f"{est} ({NOME_ENC[m['encoder']]}){extra}"


# --------------------------------------------------------------------------- #
def t_sistemas(met):
    sis = met["sistemas"]
    linhas = []
    for grupo in ordem_sistemas(sis):
        if linhas:
            linhas.append(["\\midrule"])
        for i, k in enumerate(grupo):
            m = sis[k]
            n, a, r = (m["por_classe"][c] for c in (NEG, ASSOC, NOREL))
            nome = NOME_EST[m["estrategia"]] if i == 0 else ""
            linhas.append([nome, rotulo(m, curto=True), f(n["precision"]), f(n["recall"]), f(n["f1"]),
                           f"{n['tp']}/{n['fp']}/{n['fn']}", f(a["f1"]), f(r["f1"]), f(m["macro_f1"]),
                           f(m["mcc"])])
    nota = ("DEV, $n=19\\,064$ pares candidatos (150 \\texttt{negation\\_of}, 763 \\texttt{associated\\_with}), "
            "semente 42. Não são resultados de teste. A época de cada modelo, $\\tau$ (Two-Stage) e $t$ "
            "(baseline calibrado) foram escolhidos neste mesmo DEV. O protocolo pré-registrado fixa o "
            "BioBERTpt-all; as execuções com o BioBERTpt-clin e o BERTimbau-base são variantes de "
            "\\emph{encoder} fora do pré-registro, e a Pair-Aware MLP é exploratória.")
    return tabela("tab:dev_sistemas",
                  "Resultados no DEV das três estratégias com os três \\emph{encoders}, da variante MLP e dos "
                  "controles: precisão (P), \\emph{recall} (R) e F1 de \\texttt{negation\\_of}, VP/FP/FN, F1 das "
                  "outras classes, macro-F1 e MCC.",
                  "llrrrrrrrr", "Estratégia & \\emph{Encoder} & " + " & ".join([
                      dl("P", "neg"), dl("R", "neg"), dl("F1", "neg"), dl("VP/FP/FN", "neg"), dl("F1", "assoc"),
                      dl("F1", "no\\_rel"), "Macro-F1", "MCC"]), linhas, resize=True, nota=nota)


def t_classes(met):
    sis = met["sistemas"]
    linhas = []
    for grupo in ordem_sistemas(sis):
        if linhas:
            linhas.append(["\\midrule"])
        for i, k in enumerate(grupo):
            m = sis[k]
            a, r = m["por_classe"][ASSOC], m["por_classe"][NOREL]
            ap = m.get("ap", {}).get(ASSOC)
            linhas.append([NOME_EST[m["estrategia"]] if i == 0 else "", rotulo(m, curto=True),
                           f(a["precision"]), f(a["recall"]), f(a["f1"]), f"{a['tp']}/{a['fp']}/{a['fn']}",
                           f(ap) if ap is not None else "--",
                           f(r["precision"]), f(r["recall"]), f(r["f1"])])
    nota = ("AP: precisão média de \\texttt{associated\\_with} calculada das probabilidades gravadas, sem "
            "limiar; não se aplica ao Two-Stage (as probabilidades do estágio 2 não são gravadas) nem ao "
            "filtro léxico (pós-processamento sem probabilidades próprias). O baseline calibrado usa as "
            "probabilidades do baseline, por isso a mesma AP.")
    return tabela("tab:dev_classes",
                  "\\texttt{associated\\_with} e \\texttt{no\\_relation} no DEV: precisão, \\emph{recall}, F1, "
                  "VP/FP/FN e AP de \\texttt{associated\\_with}.",
                  "llrrrrrrrr", "& & \\multicolumn{5}{c}{\\texttt{associated\\_with}} & "
                  "\\multicolumn{3}{c}{\\texttt{no\\_relation}} \\\\\n  \\cmidrule(lr){3-7}\\cmidrule(lr){8-10}\n"
                  "  Estratégia & \\emph{Encoder} & P & R & F1 & VP/FP/FN & AP & P & R & F1",
                  linhas, resize=True, nota=nota)


METRICAS_COMP = (("f1_negation_of", "\\textit{Métrica:} F1 de \\texttt{negation\\_of}"),
                 ("f1_associated_with", "\\textit{Métrica:} F1 de \\texttt{associated\\_with}"),
                 ("macro_f1", "\\textit{Métrica:} Macro-F1"))


def _linhas_blocos(itens, ncols, agrupar=False):
    """itens: lista de (rotulo1, rotulo2, comparacao). Gera um bloco vertical por
    métrica, com a diferença e os intervalos efetivamente gravados nos dois níveis."""
    linhas = []
    for metrica, nome in METRICAS_COMP:
        if linhas:
            linhas.append(["\\midrule"])
        linhas.append([f"BLOCO:{nome}"])
        anterior = None
        for r1, r2, c in itens:
            pc, pd = c["por_candidato"], c["por_documento"]
            rotulo1 = r1 if (r1 != anterior or not agrupar) else ""
            anterior = r1
            linhas.append([rotulo1, r2, s(pc[metrica]["diferenca"]), ic_ou_traco(pc, metrica),
                           ic_ou_traco(pd, metrica)])
    return linhas


def _itens_estrategias(comps, met, filtro, so_b=False):
    itens = []
    for c in comps:
        if not filtro(c):
            continue
        ma, mb = met["sistemas"][c["a"]], met["sistemas"][c["b"]]
        nome = NOME_EST[mb["estrategia"]] if so_b else f"{NOME_EST[ma['estrategia']]} $-$ {NOME_EST[mb['estrategia']]}"
        itens.append((nome, NOME_ENC[ma["encoder"]], c))
    return itens


def _faltam_ic_candidato(itens):
    return any((c["por_candidato"].get(m) or {}).get("ic95") is None for _, _, c in itens for m, _ in METRICAS_COMP)


CAB_COMP = ("Comparação (A $-$ B) & \\emph{Encoder} & $\\Delta$ & IC95\\% (candidato) & "
            "IC95\\% (documento)")
CAB_PA = ("B (A = Pair-Aware linear) & \\emph{Encoder} & $\\Delta$ & IC95\\% (candidato) & IC95\\% (documento)")
NOTA_COMP = ("Descritivas, no DEV: as épocas, $\\tau$ e $t$ foram escolhidos neste mesmo conjunto; não substituem "
             "as comparações pré-registradas no teste. A e B usam o mesmo \\emph{encoder}. $\\Delta$: diferença "
             "observada A $-$ B no DEV completo. IC95\\% (candidato): \\emph{bootstrap} pareado que reamostra com reposição os "
             "{ncand} pares candidatos, o do protocolo ({nboot} reamostras, semente {seed}). IC95\\% (documento): "
             "\\emph{bootstrap} pareado que reamostra com reposição os {ndoc} documentos do DEV que têm pares candidatos, "
             "cada um com todos os seus pares ({nboot} reamostras, semente {seed}). As três métricas são calculadas "
             "sobre as mesmas reamostras de cada nível. Intervalos: percentis 2,5 e 97,5 das diferenças reamostradas. $^{\\ast}$: intervalo que exclui o zero. Limites que "
             "arredondariam a zero com três casas aparecem com mais casas. O teste de McNemar dessas comparações "
             "está no Quadro~\\ref{quad:mcnemar}.")


def parametros_bootstrap(comp):
    """Parâmetros efetivamente gravados em comparacoes_dev.json (iguais em todas as
    comparações; a função falha se não forem)."""
    vals = {(c["n"], c["n_documentos"], c["n_boot"], c["seed"]) for c in comp["comparacoes"]}
    if len(vals) != 1:
        raise ValueError(f"parâmetros de bootstrap diferentes entre comparações: {vals}")
    ncand, ndoc, nboot, seed = vals.pop()
    return {"ncand": milhar(ncand), "ndoc": ndoc, "nboot": f"{nboot:,}".replace(",", "."), "seed": seed}


def _nota(itens, nota=NOTA_COMP):
    for chave, valor in PARAMS.items():
        nota = nota.replace("{" + chave + "}", str(valor))
    if _faltam_ic_candidato(itens):
        nota += " --: intervalo não calculado."
    return nota


PARAMS = {}


def t_comparacoes_pa(comp, met):
    PARAMS.update(parametros_bootstrap(comp))
    itens = _itens_estrategias(comp["comparacoes"], met, lambda c: c["grupo"].startswith("Pair-Aware linear ×"),
                               so_b=True)
    return tabela_longa("tab:dev_comparacoes_pa",
                        "Comparações pareadas descritivas da Pair-Aware linear, no DEV, por métrica: diferenças e "
                        "intervalos de confiança de F1 de \\texttt{negation\\_of}, de F1 de \\texttt{associated\\_with} "
                        "e de macro-F1.",
                        "llrcc", CAB_PA, _linhas_blocos(itens, 5), 5,
                        nota=_nota(itens, NOTA_COMP.replace("A e B usam o mesmo", "A é a Pair-Aware linear; A e "
                                                            "B usam o mesmo")))


FILTRO_PROTOCOLO = ("Two-Stage × Baseline (mesmo encoder)", "Two-Stage × controles", "Baseline calibrado × Baseline")


def t_comparacoes_protocolo(comp, met):
    PARAMS.update(parametros_bootstrap(comp))
    itens = _itens_estrategias(comp["comparacoes"], met, lambda c: c["grupo"] in FILTRO_PROTOCOLO)
    return tabela_blocos("tab:dev_comparacoes_ts",
                        "Comparações pareadas descritivas do Two-Stage e do baseline calibrado, no DEV, por métrica "
                        "(as comparações que o protocolo prevê para o teste, aqui aplicadas ao DEV e estendidas aos "
                        "três \\emph{encoders}).",
                        ">{\\raggedright\\arraybackslash}p{3.85cm}lrcc", CAB_COMP, _linhas_blocos(itens, 5), 5,
                        nota=_nota(itens))


def _linhas_blocos_encoders(itens):
    linhas = []
    for metrica, nome in METRICAS_COMP:
        if linhas:
            linhas.append(["\\midrule"])
        linhas.append([f"BLOCO:{nome}"])
        anterior = None
        for est, r2, c in itens:
            if est != anterior:
                linhas.append([f"SUB:{est}"])
                anterior = est
            pc, pd = c["por_candidato"], c["por_documento"]
            linhas.append(["\\quad " + r2, s(pc[metrica]["diferenca"]), ic_ou_traco(pc, metrica),
                           ic_ou_traco(pd, metrica)])
    return linhas


def t_comparacoes_encoders(comp, met):
    PARAMS.update(parametros_bootstrap(comp))
    itens = []
    for c in comp["comparacoes"]:
        if c["grupo"] != "Encoders (mesma estratégia)":
            continue
        est = c["a"].split("|")[0]
        ma, mb = met["sistemas"][c["a"]], met["sistemas"][c["b"]]
        itens.append((NOME_EST[est], f"{NOME_ENC[ma['encoder']]} $-$ {NOME_ENC[mb['encoder']]}", c))
    cab = ("Estratégia e comparação (A $-$ B) & $\\Delta$ & IC95\\% (candidato) & IC95\\% (documento)")
    nota = NOTA_COMP.replace("A e B usam o mesmo \\emph{encoder}.", "A e B usam a mesma estratégia e a mesma "
                             "configuração; só o \\emph{checkpoint} de pré-treino muda.")
    nota = nota.split(" O teste de McNemar")[0]
    return tabela_longa("tab:dev_comparacoes_encoders",
                        "Comparação entre \\emph{encoders} dentro de cada estratégia, no DEV, por métrica (mesma "
                        "configuração; só o \\emph{checkpoint} de pré-treino muda).",
                        "lrcc", cab, _linhas_blocos_encoders(itens), 4, nota=_nota(itens, nota))


def q_mcnemar(comp, met):
    """Quadro complementar: McNemar exato sobre o acerto global das comparações das
    Tabelas de comparação entre estratégias."""
    linhas = []
    for titulo, filtro in (("Pair-Aware linear frente às demais estratégias (Tabela~\\ref{tab:dev_comparacoes_pa})",
                            lambda c: c["grupo"].startswith("Pair-Aware linear ×")),
                           ("Two-Stage e baseline calibrado (Tabela~\\ref{tab:dev_comparacoes_ts})",
                            lambda c: c["grupo"] in FILTRO_PROTOCOLO)):
        itens = _itens_estrategias(comp["comparacoes"], met, filtro)
        if linhas:
            linhas.append("  \\midrule")
        linhas.append(f"  \\multicolumn{{5}}{{l}}{{\\textit{{{titulo}}}}} \\\\")
        for r1, r2, c in itens:
            m = c["mcnemar"]
            p = next((m[k] for k in ("p_value", "p", "pvalue", "p_valor") if k in m), None)
            linhas.append(f"  {r1} & {r2} & {milhar(m['b_only_a_correct'])} & {milhar(m['c_only_b_correct'])} & "
                          + (pvalor(p) if p is not None else "--") + " \\\\")
    corpo = "\n".join(linhas)
    nota = ("Teste de McNemar exato sobre o acerto global de cada par candidato do DEV, dominado pela classe "
            "\\texttt{no\\_relation}; não é específico de nenhuma classe. $b$: pares em que só A acerta; $c$: pares "
            "em que só B acerta. $p$: valor-$p$ bilateral do teste binomial exato, $b \\sim "
            "\\mathrm{Binomial}(b+c,\\,0{,}5)$. Descritivo, no DEV.")
    return (f"{CABECALHO}\\begin{{quadro}}[htbp]\n  \\centering\n"
            f"  \\caption{{Teste de McNemar das comparações pareadas entre estratégias, no DEV.}}\n"
            f"  \\label{{quad:mcnemar}}\n  \\small\n"
            f"  \\begin{{tabular}}{{llrrr}}\n  \\toprule\n"
            f"  Comparação (A $-$ B) & \\emph{{Encoder}} & $b$ & $c$ & $p$ \\\\\n  \\midrule\n{corpo}\n"
            f"  \\bottomrule\n  \\end{{tabular}}\\\\[2pt]\n"
            f"  \\parbox{{\\textwidth}}{{\\footnotesize {nota}}}\\\\[4pt]\n  {FONTE}\n\\end{{quadro}}\n")


def pvalor(p):
    if p < 1e-4:
        return "$<0{,}0001$"
    return f"${f(p, 4)}$"


def t_assoc(assoc, met):
    sis = assoc["sistemas"]
    linhas = []
    faixas = ("0-1", "2-5", "6-10", "11-25")
    for grupo in ordem_sistemas(met["sistemas"]):
        grupo = [k for k in grupo if not k.startswith(("baseline_calibrado", "filtro_pistas"))]
        if not grupo:
            continue
        if linhas:
            linhas.append(["\\midrule"])
        for i, k in enumerate(grupo):
            v, m = sis[k], met["sistemas"][k]
            o, tot = v["fp_por_origem"], v["fp_total"]
            pct = lambda x: f(100 * x / tot, 1) + "\\%" if tot else "--"  # noqa: E731
            linhas.append([NOME_EST[m["estrategia"]] if i == 0 else "", rotulo(m, curto=True), milhar(tot),
                           pct(o["inverso_associated_with"]), pct(o["sem_relacao_nas_duas_direcoes"]),
                           pct(o["verdadeiro_negation_of"])]
                          + [f(v["por_distancia"][fx]["precision"], 2) for fx in faixas])
    ref = sis["_referencia"]
    nota = (f"Direção invertida: o par $(e_2, e_1)$ tem \\texttt{{associated\\_with}} anotada, e o par $(e_1, e_2)$, "
            f"não. Das {ref['associated_with_candidatas_no_dev']} relações \\texttt{{associated\\_with}} candidatas "
            f"do DEV, só {ref['com_o_par_inverso_tambem_associated_with']} têm o par inverso também anotado. "
            "Distância: caracteres entre os dois \\emph{spans} (a mesma do \\texttt{max\\_gap}).")
    cab = ("& & & \\multicolumn{3}{c}{Origem dos FP} & \\multicolumn{4}{c}{Precisão por distância} \\\\\n"
           "  \\cmidrule(lr){4-6}\\cmidrule(lr){7-10}\n"
           "  Estratégia & \\emph{Encoder} & FP & " + dl("direção", "invertida") + " & " + dl("sem", "relação")
           + " & " + dl("é", "\\texttt{negation\\_of}") + " & "
           "0--1 & 2--5 & 6--10 & 11--25")
    return tabela("tab:dev_assoc_fp",
                  "Falsos positivos de \\texttt{associated\\_with} no DEV: origem e precisão por faixa de distância "
                  "entre os argumentos.", "llrrrrrrrr", cab, linhas, resize=True, nota=nota)


def t_tau(tau):
    """Busca de τ em três blocos consecutivos, um por encoder, com a grade completa."""
    linhas = []
    for e in ENCODERS:
        congelado = tau["two_stage"][e]["tau"]
        if linhas:
            linhas.append(["\\midrule"])
        linhas.append([f"BLOCO:\\textbf{{{NOME_ENC[e]}}} (valor congelado: $\\tau={f(congelado, 2)}$)"])
        for l in sorted(tau["two_stage"][e]["grade"], key=lambda x: x["tau"]):
            marca = "$^{\\ast}$" if l["tau"] == congelado else ""
            linhas.append([f(l["tau"], 2) + marca, milhar(l["n_roteados"]), f(l["f1_negation_of"], 4),
                           f(l["macro_f1"], 4), f(l["assoc_recall"], 3)])
    cab = ("$\\tau$ & Pares roteados & F1 \\texttt{negation\\_of} & Macro-F1 & \\emph{Recall} "
           "\\texttt{associated\\_with}")
    return tabela_longa("tab:tau_encoders",
                        "Busca de $\\tau$ do Two-Stage no DEV, por \\emph{encoder} (\\emph{pipeline} completo, fluxo "
                        "real). $^{\\ast}$ valor congelado (maior F1 de \\texttt{negation\\_of}).",
                        "rrrrr", cab, linhas, 5,
                        nota="Pares roteados: pares que o estágio 1 encaminha ao estágio 2 com o $\\tau$ da linha. "
                             "Grade ($0{,}05$ a $0{,}95$, passo $0{,}05$) e critério do protocolo, iguais para os três "
                             "\\emph{encoders}. Valores escolhidos no DEV e, por isso, otimistas como estimativa de "
                             "desempenho.")


def t_exploratoria(met):
    """Pair-Aware MLP (exploratória) ao lado da Pair-Aware linear, BioBERTpt-all."""
    linhas = []
    for k, nome in (("pair_aware|BioBERTpt-all", "Pair-Aware linear (protocolo)"),
                    ("pair_aware_mlp|BioBERTpt-all", "Pair-Aware MLP (exploratória)")):
        m = met["sistemas"][k]
        n = m["por_classe"][NEG]
        linhas.append([nome, f(n["precision"]), f(n["recall"]), f(n["f1"]), f"{n['tp']}/{n['fp']}/{n['fn']}",
                       f(m["macro_f1"]), f(m["mcc"])])
    return tabela("tab:dev_exploratoria",
                  "Resultado exploratório no DEV (fora do protocolo pré-registrado): Pair-Aware MLP ao lado da "
                  "Pair-Aware linear, BioBERTpt-all, semente 42.",
                  "lrrrrrr", "Sistema & P neg & R neg & F1 neg & VP/FP/FN & Macro-F1 & MCC", linhas,
                  resize=True, nota="P, R e F1 neg: precisão, \\emph{recall} e F1 de \\texttt{negation\\_of}; VP/FP/FN da mesma "
                       "classe.")


def t_two_stage_diag(met):
    linhas = []
    for e in ENCODERS:
        m = met["sistemas"][f"two_stage|{e}"]
        d = m["diagnostico_estagios"]
        linhas.append([NOME_ENC[e], f(m["tau"], 2), milhar(d["n_roteados_ao_estagio2"]),
                       milhar(d["roteados_sem_relacao_anotada"]),
                       f(100 * d["recall_estagio1_nos_pares_negation_of"], 1) + "\\%",
                       f(100 * d["recall_estagio1_nos_pares_com_relacao"], 1) + "\\%",
                       f(100 * d["acuracia_do_tipo_nos_roteados_com_relacao"], 1) + "\\%"])
    return tabela("tab:two_stage_diag",
                  "Two-Stage no DEV, por \\emph{encoder}: o que o estágio 1 encaminha com o $\\tau$ congelado e quanto "
                  "o estágio 2 acerta do tipo.",
                  "lrrrrrr", "\\emph{Encoder} & $\\tau$ & Roteados & " + " & ".join([
                      dl("Sem relação", "anotada"), dl("\\emph{Recall} est.\\,1", "(neg)"),
                      dl("\\emph{Recall} est.\\,1", "(relação)"), dl("Acerto", "do tipo")]), linhas,
                  resize=True, nota="Diagnóstico do fluxo real, gravado na composição (\\texttt{metricas.json}); não é "
                                    "resultado da arquitetura. Acerto do tipo: entre os pares roteados que têm relação "
                                    "anotada.")


def _sessoes_marcas(t):
    """Épocas concluídas antes da sessão registrada (‡) e a época retomada no meio (†)."""
    sess = t.get("sessoes") or []
    ppe = t.get("passos_por_epoca")
    if not sess or sess[0]["retomada_do_passo"] is None or not ppe:
        return {}, None
    passo = sess[0]["retomada_do_passo"]
    epoca, resto = passo // ppe + 1, passo % ppe
    marcas = {e: "$^{\\ddagger}$" for e in range(1, epoca)}
    if resto:
        marcas[epoca] = "$^{\\dagger}$"
    else:
        marcas.update({epoca: ""})
    return marcas, (epoca, ppe - resto, ppe) if resto else None


def t_historico(inv):
    linhas, notas = [], []
    registros = {r["pasta"]: r for r in inv["execucoes"]}
    ordem = [r for est in ("pair_aware", "baseline", "pair_aware_mlp") for e in ENCODERS
             for r in inv["execucoes"] if r.get("estrategia") == est and r.get("encoder") == e and r["tipo"] == "execucao"]
    for r in ordem:
        t = r["treino"]
        marcas, parcial = _sessoes_marcas(t)
        if linhas:
            linhas.append(["\\midrule"])
        for h in t["dev_history"]:
            ep = h["epoch"]
            sel = "$\\ast$" if ep == t["melhor_epoca"] else ""
            linhas.append([f"{NOME_EST[r['estrategia']]} ({NOME_ENC[r['encoder']]})" if ep == 1 else "",
                           f"{ep}{sel}", f(h["train_loss"], 4), f(h["dev_loss"], 4), f(h["dev_macro_f1"]),
                           f(h["dev_negation_of_f1"]), f(h["duration_s"] / 60, 1) + marcas.get(ep, "")])
        if parcial:
            notas.append(f"{NOME_EST[r['estrategia']]} ({NOME_ENC[r['encoder']]}), época {parcial[0]}: "
                         f"{milhar(parcial[1])} dos {milhar(parcial[2])} passos")
    del registros
    nota = ("$^{\\dagger}$ Época interrompida e retomada a partir de um \\emph{checkpoint}; a duração cobre apenas a "
            "sessão final (" + "; ".join(notas) + "). $^{\\ddagger}$ Época concluída em sessão anterior, sem registro "
            "de sessão preservado. As durações não são comparáveis entre execuções como medida de custo.")
    return tabela("tab:dev_historico_execucoes",
                  "Evolução por época no DEV das execuções de três rótulos. $\\ast$ marca a época selecionada (maior "
                  "macro-F1 no DEV).", "llrrrrr",
                  "Execução & Época & Perda (treino) & Perda (DEV) & Macro-F1 & F1 neg & Duração (min)",
                  linhas, resize=True, nota=nota)


def t_estagios(inv):
    linhas = []
    for estagio, campo, n in (("two_stage_e1", "dev_ap_relacao", 3),
                              ("two_stage_e2", "dev_negation_of_f1_relacoes_anotadas", 10)):
        if linhas:
            linhas.append(["\\midrule"])
        for e in ENCODERS:
            r = next(x for x in inv["execucoes"] if x.get("estrategia") == estagio and x.get("encoder") == e)
            t = r["treino"]
            valores = []
            for h in t["dev_history"]:
                v = f(h[campo])
                valores.append(f"\\textbf{{{v}}}" if h["epoch"] == t["melhor_epoca"] else v)
            valores += ["--"] * (n - len(valores))
            linhas.append([(dl("Estágio 1", "(AP de relação)", "l") if estagio.endswith("1")
                            else dl("Estágio 2", "(F1 neg)", "l"))
                           if e == ENCODERS[0] else "", NOME_ENC[e]] + valores + ["--"] * (10 - n))
    return tabela("tab:dev_estagios",
                  "Two-Stage: valor do critério de seleção de cada estágio por época, no DEV, para os três "
                  "\\emph{encoders}. Em negrito, a época selecionada.",
                  "ll" + "r" * 10, "Estágio & \\emph{Encoder} & " + " & ".join(str(i) for i in range(1, 11)),
                  linhas, resize=True,
                  nota="Estágio 1: AP da classe \\texttt{relacao} no DEV completo (3 épocas). Estágio 2: F1 de "
                       "\\texttt{negation\\_of} nos 913 pares do DEV com relação anotada (10 épocas). Valores de cada "
                       "estágio isolado, não da arquitetura.")


def t_inventario(inv):
    linhas = []
    tipo_nome = {"execucao": "treinada", "estagio_two_stage": "estágio", "composicao_two_stage": "composição",
                 "derivada_dev": "derivada"}
    chave = lambda r: (["execucao", "estagio_two_stage", "composicao_two_stage", "derivada_dev"].index(r["tipo"]),  # noqa: E731
                       r.get("estrategia") or "", ENCODERS.index(r["encoder"]) if r.get("encoder") in ENCODERS else 9)
    for r in sorted(inv["execucoes"], key=chave):
        t = r.get("treino") or {}
        e = r.get("experimento") or {}
        sess = t.get("sessoes") or []
        if sess:
            s0 = sess[0]
            sessao = "nova" if s0["tipo"] == "nova" else f"retomada ({milhar(s0['retomada_do_passo'])})"
        else:
            sessao = "--"
        if r["tipo"] == "composicao_two_stage":
            melhor = f"$\\tau={f(r['limiar']['tau'], 2)}$"
        elif r["tipo"] == "derivada_dev":
            melhor = "--"
        else:
            melhor = f"{t.get('melhor_epoca')}/{t.get('epocas_registradas')}"
        ident = (r.get("identidade_codigo") or "")[:6] or "--"
        linhas.append([tt(r["pasta"]), tipo_nome[r["tipo"]], NOME_EST.get(r.get("estrategia"), r.get("estrategia") or ""),
                       NOME_ENC.get(r.get("encoder"), "--"), tt((e.get("commit") or "")[:7]) if e.get("commit") else "--",
                       tt(ident) if ident != "--" else "--", sessao, melhor,
                       "não" if t.get("test_avaliado") is False else ("--" if not t else "não registrado")])
    nota = ("Commit: revisão do código registrada em \\texttt{experimento.json} (os estágios do Two-Stage e as "
            "composições não têm esse arquivo). Identidade: os seis primeiros dígitos do SHA-256 do manifesto do "
            "código (\\texttt{config.json}). Sessões: só a sessão final de cada treino fica registrada; "
            "\"retomada (passo)\" indica que o treino começou em sessão anterior, sem registro preservado. "
            "Melhor época: selecionada/registradas. TEST: o campo \\texttt{test\\_avaliado} do \\texttt{treino.json}; "
            "os estágios do Two-Stage não têm o campo. Checkpoints e \\emph{logs} de console ficam fora do "
            "repositório (Google Drive) e não foram verificados.")
    cab = ("Pasta & Tipo & Estratégia & \\emph{Encoder} & Commit & " + dl("Identi-", "dade") + " & "
           + dl("Sessão", "registrada") + " & Melhor & TEST")
    colspec = (">{\\raggedright\\arraybackslash}p{2.9cm} l >{\\raggedright\\arraybackslash}p{1.8cm} "
               "l l l >{\\raggedright\\arraybackslash}p{1.6cm} l "
               ">{\\raggedright\\arraybackslash\\hyphenpenalty=10000}p{1.55cm}")
    return tabela_longa("tab:inventario_execucoes",
                        "Inventário das execuções da branch \\texttt{resultados-colab} e das execuções derivadas só "
                        "do DEV.", colspec, cab, linhas, 9, nota=nota, fonte="\\footnotesize", sep="2.5pt")


def figura_tau(tau, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    cores = {"BioBERTpt-all": "#2a78d6", "BioBERTpt-clin": "#eb6834", "BERTimbau": "#1baf7a"}
    marcadores = {"BioBERTpt-all": "o", "BioBERTpt-clin": "s", "BERTimbau": "^"}
    tinta, tinta2, grade = "#1f1f1e", "#5c5b55", "#e4e3de"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": tinta2, "axes.labelcolor": tinta,
                         "xtick.color": tinta2, "ytick.color": tinta2})
    fig, eixos = plt.subplots(1, 2, figsize=(8.2, 3.3), sharex=True)
    for ax, campo, titulo in ((eixos[0], "f1_negation_of", "F1 de negation_of"), (eixos[1], "macro_f1", "Macro-F1")):
        for e in ENCODERS:
            g = tau["two_stage"][e]["grade"]
            x = [l["tau"] for l in g]
            y = [l[campo] for l in g]
            ax.plot(x, y, color=cores[e], lw=2, marker=marcadores[e], ms=4.5, label=NOME_ENC[e], zorder=3)
            esc = tau["two_stage"][e]["tau"]
            ye = next(l[campo] for l in g if l["tau"] == esc)
            ax.plot([esc], [ye], marker=marcadores[e], ms=13 if e == ENCODERS[0] else 9, mfc="none",
                    mec=cores[e], mew=2, zorder=4 if e == ENCODERS[0] else 5)
        ax.set_title(titulo, color=tinta, fontsize=10, loc="left")
        ax.set_xlabel("limiar τ")
        ax.grid(True, color=grade, lw=0.8)
        ax.set_axisbelow(True)
        virgula = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
        ax.yaxis.set_major_formatter(virgula)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    eixos[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.text(0.01, 0.01, "Marcador vazado: τ congelado (maior F1 de negation_of). DEV, semente 42.",
             color=tinta2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def main(repo: Path, src: Path) -> None:
    pasta = repo / "codigo" / "resultados" / "auditoria_execucoes"
    ler = lambda n: json.loads((pasta / n).read_text(encoding="utf-8"))  # noqa: E731
    inv, met, comp = ler("inventario.json"), ler("metricas_dev.json"), ler("comparacoes_dev.json")
    tau, assoc = ler("two_stage_tau.json"), ler("associated_with_dev.json")
    saidas = {
        "dev_sistemas": t_sistemas(met),
        "dev_classes": t_classes(met),
        "dev_comparacoes_pa": t_comparacoes_pa(comp, met),
        "dev_comparacoes_ts": t_comparacoes_protocolo(comp, met),
        "dev_comparacoes_encoders": t_comparacoes_encoders(comp, met),
        "dev_assoc_fp": t_assoc(assoc, met),
        "tau_encoders": t_tau(tau),
        "two_stage_diag": t_two_stage_diag(met),
        "dev_historico_execucoes": t_historico(inv),
        "dev_estagios": t_estagios(inv),
        "inventario_execucoes": t_inventario(inv),
        "mcnemar_comparacoes": q_mcnemar(comp, met),
        "dev_exploratoria_mlp": t_exploratoria(met),
    }
    (src / "tabelas").mkdir(exist_ok=True)
    for nome, texto in saidas.items():
        (src / "tabelas" / f"{nome}.tex").write_text(texto, encoding="utf-8")
    figura_tau(tau, src / "imagens" / "resultados" / "tau_two_stage_encoders.png")
    print("tabelas:", sorted(saidas), "| figura: tau_two_stage_encoders.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
