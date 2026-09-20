#!/usr/bin/env python3
"""
Auditoria dos falsos positivos de `negation_of` que SOBRAM depois do filtro.

Produto: um relatorio. Nada aqui altera predicao, gold, lexico ou configuracao,
e nenhum numero do Cap. 6 depende deste arquivo. O objetivo e estabelecer o TETO
antes de gastar GPU perseguindo-o, e o enquadramento tem de ser prospectivo, nao
uma desculpa para desempenho ruim.

SORTEIO DA EXECUCAO
-------------------
A execucao auditada e sorteada, nao escolhida. `random.Random(DRAW_SEED)` com
`DRAW_SEED` fixado no codigo ANTES do primeiro sorteio, e o resultado e
registrado no relatorio. Nao ha segunda tentativa: se o sorteio fosse repetido
ate cair numa execucao conveniente, a amostra deixaria de ser amostra.

CRITERIO DE CLASSIFICACAO, FIXADO ANTES DE OLHAR OS CASOS
---------------------------------------------------------
Cada FP remanescente e um par (pista `e1`, alvo `e2`) num documento `d`. Seja
`G` o conjunto de alvos que o gold liga a ESSA pista NESSE documento.

  1. `G` vazio e `gap(e1,e2) <= 1`
     -> PROVAVEL ERRO DE ANOTACAO. Ha uma pista explicita colada ao alvo e o
        anotador nao registrou negacao nenhuma para ela no documento inteiro.
  2. `G` vazio e `gap(e1,e2) > 1`
     -> ERRO DO MODELO. A pista nao nega nada anotado e o alvo escolhido ainda
        por cima esta distante.
  3. `G` nao vazio e o texto entre `e2` e o alvo gold mais proximo contem apenas
     separadores de coordenacao (`,` `;` `/` `+` `-` e as conjuncoes `e`, `ou`,
     `nem`)
     -> AMBIGUIDADE GENUINA. `e2` e outro item da mesma lista negada, e a
        divergencia e a convencao de fan-out 1 do SemClinBr, nao leitura clinica.
  4. `G` nao vazio e `gap(e1,e2) <= min gap sobre G`
     -> AMBIGUIDADE GENUINA. O modelo escolheu um alvo tao proximo quanto o
        anotado, ou mais.
  5. resto
     -> ERRO DO MODELO. Escopo estendido para alem do alvo anotado sem
        coordenacao que o justifique.

A ordem importa e e a de cima para baixo. O criterio e mecanico de proposito: ele
roda igual para quem repetir, e nao depende de eu ter lido os casos antes.

CORRECAO DE IMPLEMENTACAO (mesmo dia, criterio inalterado)
-----------------------------------------------------------
A primeira versao da regra 3 lia o texto entre os spans direto de `doc["text"]`
com os offsets de `doc["entities"]`, e isso esta ERRADO no SemClinBr: so 64,4%
das entidades do test satisfazem `text[start:end] == text_da_entidade`. Os
offsets sofrem uma deriva cumulativa negativa dentro do documento, compativel
com uma normalizacao de espaco em branco aplicada ao texto depois que os offsets
foram calculados. Com isso a regra 3 nunca disparava, e listas coordenadas
obvias (`NEGA CIRURGIAS E TRAUMAS OCULARES`) caiam na regra 5.

A correcao reancora cada entidade em `doc["text"]` procurando o proprio texto da
entidade a partir da posicao esperada, acompanhando a deriva (97,1% das
entidades do test sao reancoradas). Isso vale SO para ler o trecho entre as
entidades e para imprimir o contexto. O `gap` continua vindo dos offsets
originais, que sao mutuamente consistentes e sao os que `src/candidates.py` usa,
para a auditoria falar do mesmo objeto que o sistema auditado.

O criterio nao mudou, so a leitura do texto que ele consulta.

Uso:
    python scripts/audit_residual_fp.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from candidates import entity_gap, iter_candidate_pairs  # noqa: E402
from negation_lexicon import (LABELS, apply_cue_filter,  # noqa: E402
                              induce_lexicon, normalize_surface, read_jsonl)
from utils.logger import get_logger  # noqa: E402

log = get_logger("audit_fp")

NEG = LABELS.index("negation_of")

# Fixado antes do primeiro sorteio. A data e a da decisao que tornou a rodada de
# 17-20/09 a fonte oficial do Cap. 6.
DRAW_SEED = 20260920

CANDIDATE_RUNS = [
    "filtro_biobertpt_seed42",
    "filtro_bertimbau_seed42",
    "filtro_biobertpt_seed43",
    "filtro_bertimbau_seed43",
]

COORD = re.compile(r"^[\s,;/+\-]*(?:(?:e|ou|nem)[\s,;/+\-]*)*$", re.IGNORECASE)

ERRO_MODELO = "erro do modelo"
AMBIGUIDADE = "ambiguidade genuina"
ERRO_GOLD = "provavel erro de anotacao do gold"


def align_entities(doc, win=80):
    """Reancora as entidades em `doc["text"]`, seguindo a deriva de offsets.

    Devolve id -> (start, end) corrigido, ou None quando a forma de superficie da
    entidade nao e encontrada. Ver a nota de correcao no docstring do modulo.
    """
    text = doc["text"]
    shift = 0
    out: dict[str, tuple[int, int] | None] = {}
    for e in sorted(doc["entities"], key=lambda e: (e["start"], e["end"], e["id"])):
        s0, t = e["start"], e["text"]
        if text[s0:e["end"]] == t:
            out[e["id"]] = (s0, e["end"])
            shift = 0
            continue
        target = s0 + shift
        # Ocorrencia MAIS PROXIMA da posicao esperada, nao a primeira da janela:
        # formas como `NEGA` se repetem no mesmo documento, e a primeira
        # ocorrencia da janela pode ser outra mencao.
        pos, best = -1, None
        for lo, hi in ((max(0, target - win), target + win),
                       (max(0, s0 - 400), s0 + 400)):
            j = text.find(t, lo, hi + len(t))
            while j >= 0:
                if best is None or abs(j - target) < best:
                    best, pos = abs(j - target), j
                j = text.find(t, j + 1, hi + len(t))
            if pos >= 0:
                break
        out[e["id"]] = (pos, pos + len(t)) if pos >= 0 else None
        if pos >= 0:
            shift = pos - s0
    return out


def between(text, a, b):
    """Trecho entre dois spans JA REANCORADOS. None quando indisponivel."""
    if a is None or b is None:
        return None
    if a[1] <= b[0]:
        return text[a[1]:b[0]]
    if b[1] <= a[0]:
        return text[b[1]:a[0]]
    return ""


def classify(text, e1, e2, gold_targets, aligned):
    g = entity_gap(e1, e2)
    if not gold_targets:
        if g <= 1:
            return ERRO_GOLD, "1: pista sem alvo anotado no documento, adjacente ao alvo"
        return ERRO_MODELO, "2: pista sem alvo anotado no documento, alvo distante"
    nearest = min(gold_targets, key=lambda t: entity_gap(e1, t))
    span = between(text, aligned.get(nearest["id"]), aligned.get(e2["id"]))
    if span is not None and COORD.match(span):
        return AMBIGUIDADE, "3: item coordenado do mesmo alvo anotado"
    if g <= min(entity_gap(e1, t) for t in gold_targets):
        return AMBIGUIDADE, "4: alvo tao proximo quanto o anotado"
    return ERRO_MODELO, "5: escopo estendido sem coordenacao"


def snippet(text, e1, e2, aligned, pad=45):
    a, b = aligned.get(e1["id"]), aligned.get(e2["id"])
    if a is None or b is None:
        return "(offsets nao reancorados)"
    lo = max(0, min(a[0], b[0]) - pad)
    hi = min(len(text), max(a[1], b[1]) + pad)
    s = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return ("..." if lo else "") + s + ("..." if hi < len(text) else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gap", type=int, default=25)
    ap.add_argument("--calib", default="results/CALIBRACAO_filtro.json")
    ap.add_argument("--out", default="results/AUDITORIA_fp_negation_of.md")
    args = ap.parse_args()

    calib = json.loads((REPO / args.calib).read_text(encoding="utf-8"))
    drawn = random.Random(DRAW_SEED).choice(CANDIDATE_RUNS)
    log.info("Sorteio (seed=%d) entre %s -> %s", DRAW_SEED,
             CANDIDATE_RUNS, drawn)

    docs = list(read_jsonl(REPO / "data/splits/test.jsonl"))
    texts = {d["doc_id"]: d["text"] for d in docs}
    ents = {d["doc_id"]: {e["id"]: e for e in d["entities"]} for d in docs}
    aligns = {d["doc_id"]: align_entities(d) for d in docs}
    gold_neg: dict[tuple, list] = {}
    for d in docs:
        for r in d["relations"]:
            if r["type"] == "negation_of":
                gold_neg.setdefault((d["doc_id"], r["e1_id"]), []).append(
                    ents[d["doc_id"]][r["e2_id"]])

    cands = [c for d in docs for c in iter_candidate_pairs(d, max_gap=args.max_gap)]
    y_true = [LABELS.index(c["label"]) for c in cands]
    d = json.loads((REPO / f"results/{drawn}.preds.json").read_text(encoding="utf-8"))
    if d["y_true"] != y_true:
        log.error("Sidecar %s fora do espaco de candidatos", drawn)
        return 2
    y_pred = d["y_pred"]

    fps = []
    for c, t, p in zip(cands, y_true, y_pred):
        if p == NEG and t != NEG:
            text = texts[c["doc_id"]]
            aligned = aligns[c["doc_id"]]
            targets = gold_neg.get((c["doc_id"], c["e1"]["id"]), [])
            kind, why = classify(text, c["e1"], c["e2"], targets, aligned)
            fps.append({
                "doc_id": c["doc_id"], "e1": c["e1"]["text"],
                "e2": c["e2"]["text"], "gap": entity_gap(c["e1"], c["e2"]),
                "gold_label": LABELS[t],
                "gold_targets": [g["text"] for g in targets],
                "kind": kind, "why": why,
                "snippet": snippet(text, c["e1"], c["e2"], aligned),
            })
    counts = Counter(f["kind"] for f in fps)
    log.info("FP remanescentes em %s: %d | %s", drawn, len(fps), dict(counts))

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == NEG and p == NEG)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == NEG and p != NEG)

    L = []
    A = L.append
    A("# Auditoria dos falsos positivos remanescentes de `negation_of`")
    A("")
    A(f"`{datetime.now(timezone.utc).strftime('%d/%m/%Y')}` · "
      f"`split = test` · `execucao sorteada = {drawn}` · "
      f"**relatorio, nao decisao**")
    A("")
    A("Nada foi filtrado, editado ou reclassificado com base nesta leitura. Nenhum")
    A("numero do Cap. 6 depende deste arquivo. O produto e uma fracao, e ela serve para")
    A("saber se o teto que resta perseguir e 0,95 ou 0,88 antes de gastar GPU atras")
    A("dele.")
    A("")
    A("## Sorteio")
    A("")
    A(f"`random.Random({DRAW_SEED}).choice(...)` sobre as quatro execucoes filtradas,")
    A(f"com a semente fixada no codigo antes do primeiro sorteio. Saiu **{drawn}**.")
    A("Nao houve segunda tentativa.")
    A("")
    A("## Criterio, fixado antes de olhar os casos")
    A("")
    A("Seja `G` o conjunto de alvos que o gold liga a essa pista nesse documento.")
    A("")
    A("| # | condicao | classe |")
    A("|---|---|---|")
    A("| 1 | `G` vazio e `gap <= 1` | provavel erro de anotacao |")
    A("| 2 | `G` vazio e `gap > 1` | erro do modelo |")
    A("| 3 | `G` nao vazio, so separadores de coordenacao entre o alvo anotado e `e2` | ambiguidade genuina |")
    A("| 4 | `G` nao vazio e `gap` menor ou igual ao do alvo anotado mais proximo | ambiguidade genuina |")
    A("| 5 | resto | erro do modelo |")
    A("")
    A("As regras sao aplicadas de cima para baixo. O criterio e mecanico de proposito,")
    A("para que rode igual para quem repetir e nao dependa de os casos terem sido lidos")
    A("antes de ele ser escrito.")
    A("")
    A("### Correcao de implementacao, no mesmo dia")
    A("")
    A("A primeira execucao deste script tinha um defeito na regra 3, e o defeito")
    A("aparece ao ler a saida. Ela pegava o trecho entre as entidades fatiando")
    A("`doc[\"text\"]` com os offsets de `doc[\"entities\"]`, e no SemClinBr esses offsets")
    A("nao indexam o texto: so 64,4% das entidades do test satisfazem")
    A("`text[start:end] == texto_da_entidade`. A deriva e cumulativa e negativa dentro")
    A("do documento, o que e a assinatura de uma normalizacao de espaco em branco feita")
    A("depois do calculo dos offsets. Com isso a regra 3 nunca disparava, e listas")
    A("coordenadas obvias como `NEGA CIRURGIAS E TRAUMAS OCULARES` caiam na regra 5.")
    A("")
    A("A correcao reancora cada entidade procurando a propria forma de superficie a")
    A("partir da posicao esperada, o que recupera 97,1% das entidades do test. Ela vale")
    A("so para LER o trecho entre as entidades e imprimir o contexto. O `gap` continua")
    A("saindo dos offsets originais, que sao mutuamente consistentes e sao os que")
    A("`src/candidates.py` usa, para a auditoria falar do mesmo objeto que o sistema")
    A("auditado. O criterio nao mudou, so a leitura do texto que ele consulta.")
    A("")
    A("## Resultado")
    A("")
    A(f"A execucao sorteada acerta {tp} das 152 negacoes do teste, perde {fn} e")
    A(f"produz {len(fps)} falsos positivos.")
    A("")
    A("| classe | casos | fracao dos FP |")
    A("|---|---|---|")
    for k in (ERRO_MODELO, AMBIGUIDADE, ERRO_GOLD):
        n = counts.get(k, 0)
        A(f"| {k} | {n} | {n/len(fps):.1%}".replace(".", ",") + " |")
    A(f"| **total** | **{len(fps)}** | |")
    A("")
    nao_modelo = counts.get(AMBIGUIDADE, 0) + counts.get(ERRO_GOLD, 0)
    A(f"Pelo criterio acima, **{nao_modelo} dos {len(fps)} FP ({nao_modelo/len(fps):.1%}"
      .replace(".", ",") + ") nao sao erro de leitura clinica**.")
    A("Somando os como acertos, a precisao da execucao sorteada iria de "
      f"{tp/(tp+len(fps)):.4f} para {tp/(tp+counts.get(ERRO_MODELO,0)):.4f}"
      .replace(".", ",") + ", e o F1 de ")
    A(f"{2*tp/(2*tp+len(fps)+fn):.4f} para {2*tp/(2*tp+counts.get(ERRO_MODELO,0)+fn):.4f}"
      .replace(".", ",") + ".")
    A("Esse segundo numero **nao e um resultado**. Ele e o teto que sobra se toda a")
    A("divergencia de anotacao fosse resolvida a favor do modelo, e serve so para")
    A("dimensionar quanto ainda ha para ganhar em precisao.")
    A("")
    A("## Casos, um a um")
    A("")
    for k in (ERRO_MODELO, AMBIGUIDADE, ERRO_GOLD):
        sel = [f for f in fps if f["kind"] == k]
        A(f"### {k} ({len(sel)})")
        A("")
        if not sel:
            A("Nenhum caso.")
            A("")
            continue
        A("| doc | pista `e1` | alvo `e2` | gap | gold do par | alvos anotados da pista | regra | trecho |")
        A("|---|---|---|---|---|---|---|---|")
        for f in sel:
            alvos = ", ".join(f["gold_targets"]) if f["gold_targets"] else "(nenhum)"
            trecho = f["snippet"].replace("|", "\\|")
            A(f"| {f['doc_id']} | {f['e1']} | {f['e2']} | {f['gap']} | {f['gold_label']} | "
              f"{alvos} | {f['why'].split(':')[0]} | {trecho} |")
        A("")
    A("## Limite desta auditoria")
    A("")
    A("O criterio decide por evidencia estrutural do gold, nao por leitura clinica caso")
    A("a caso. Ele acerta o padrao dominante, que e a lista coordenada, e vai errar em")
    A("casos que exigem saber o que a frase quer dizer. Uma revisao humana das linhas")
    A("acima pode mover casos entre as tres classes, e as colunas `trecho` e `alvos")
    A("anotados da pista` estao no relatorio exatamente para permitir isso.")
    A("")
    (REPO / args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    (REPO / "results/AUDITORIA_fp_negation_of.json").write_text(
        json.dumps({"drawn_run": drawn, "draw_seed": DRAW_SEED,
                    "n_fp": len(fps), "counts": dict(counts), "cases": fps},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Auditoria salva em %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
