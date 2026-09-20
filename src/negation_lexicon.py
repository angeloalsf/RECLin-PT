#!/usr/bin/env python3
"""
Lexico de gatilhos de negacao induzido do TRAIN, e filtro de pista a posteriori.

MOTIVACAO
---------
O gargalo de `negation_of` e PRECISAO, nao recall. Nas execucoes oficiais o
recall fica em 0,88-0,93 enquanto a precisao fica em 0,55-0,66, e ~96% dos
falsos positivos vem de `no_relation`, nao de confusao com `associated_with`.
O modelo acha o contexto de negacao e atribui a relacao ao par errado. Restringir
ONDE a classe pode ser predita ataca esse erro; mexer na fronteira entre
`negation_of` e `associated_with` nao ataca quase nada.

Toda negacao anotada no SemClinBr tem uma pista lexical como PRIMEIRO argumento
(`SEM`, `NEGA`, `NAO`, `AUSENCIA`, e negacoes morfologicas como `INDOLOR`).
Este modulo induz essas formas de superficie do TRAIN e usa a pertinencia ao
lexico como restricao sobre predicoes ja salvas.

ENQUADRAMENTO (importa para o texto do TCC)
-------------------------------------------
O filtro NAO e um retorno ao NegEx. Ele nao decide a relacao; ele restringe o
espaco onde o modelo pode decidi-la. Quem escolhe o ALVO da negacao dentro desse
espaco continua sendo o classificador contextual, que e exatamente a capacidade
que uma regra fixa nao tem (ver `scripts/make_rule_baseline.py`, que mede o que
a regra sozinha alcanca).

DECISOES DE IMPLEMENTACAO
-------------------------
1) Inducao SO do TRAIN. Induzir do corpus inteiro seria vazamento.
2) Inducao restrita ao ESPACO DE CANDIDATOS (`max_gap`), nao ao conjunto de
   relacoes anotadas. Os dois diferem: em `max_gap=25` ha 1.299 relacoes
   `negation_of` anotadas no train mas apenas 1.255 pares candidatos com esse
   rotulo, porque 44 relacoes ficam fora da janela. O filtro opera sobre o
   espaco de candidatos, entao o lexico e induzido do mesmo espaco. Por isso
   `max_gap` e parametro de `induce_lexicon`.
3) Normalizacao = NFKD + remocao de diacriticos + colapso de espaco em branco +
   `casefold`. A remocao de diacriticos NAO e cosmetica: no corpus a mesma forma
   aparece acentuada e nao acentuada (`NAO` 128x contra `NÃO`, `AUSENCIA` contra
   `AUSÊNCIA`, `EVACUACAO` contra `EVACUAÇÃO`). Sem remover diacriticos, `ausencia`
   se parte em duas formas e cai abaixo do limiar de frequencia por um artefato
   de digitacao. Com a normalizacao aqui, `min_freq` 1/2/3/5/10 produz lexicos de
   51/17/11/9/8 formas.
4) O destino do rebaixamento e FIXO (`no_relation`). Isso torna o macro-F1
   determinado a partir dos sidecars, sem reexecutar inferencia: a regra so toca
   predicoes `negation_of`, entao o F1 de `associated_with` nao muda.

O modulo nao treina nada, nao usa GPU e nao le o TEST.

Uso como biblioteca:
    from negation_lexicon import induce_lexicon, apply_cue_filter
    lex = induce_lexicon(read_jsonl("data/splits/train.jsonl"), 25, min_freq=2)
    y_filt = apply_cue_filter(candidates, y_pred, lex)

Uso como script (inspecao do lexico):
    python src/negation_lexicon.py --min-freq 2 --max-gap 25
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import iter_candidate_pairs  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("negation_lexicon")

# Mesma ordem de LABELS em src/relation_extraction.py. Os sidecars .preds.json
# guardam ids inteiros nessa ordem.
LABELS = ["negation_of", "associated_with", "no_relation"]

_WS = re.compile(r"\s+")


def normalize_surface(text: str) -> str:
    """Forma normalizada de comparacao: sem diacriticos, sem caixa, espaco unico."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _WS.sub(" ", stripped).strip().casefold()


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_cue_forms(train_docs: Iterable[dict], max_gap: int) -> Counter:
    """Frequencia das formas de superficie de e1 nos pares gold `negation_of`.

    Conta pares CANDIDATOS rotulados `negation_of` (nao relacoes anotadas), pelo
    motivo registrado na decisao (2) do docstring do modulo.
    """
    counts: Counter = Counter()
    for doc in train_docs:
        for cand in iter_candidate_pairs(doc, max_gap=max_gap):
            if cand["label"] == "negation_of":
                counts[normalize_surface(cand["e1"]["text"])] += 1
    return counts


def induce_lexicon(train_docs: Iterable[dict], max_gap: int,
                   min_freq: int) -> dict[str, int]:
    """Lexico de gatilhos: forma normalizada -> frequencia no TRAIN.

    `train_docs` DEVE ser o train e so o train. O dicionario devolvido serve como
    conjunto (`forma in lexico`) e carrega a frequencia para o relatorio de
    calibracao.
    """
    counts = count_cue_forms(train_docs, max_gap)
    lex = {form: n for form, n in counts.items() if n >= min_freq}
    log.info("Lexico induzido do train (max_gap=%d, min_freq=%d): %d formas, "
             "%d ocorrencias cobertas de %d",
             max_gap, min_freq, len(lex), sum(lex.values()), sum(counts.values()))
    return dict(sorted(lex.items(), key=lambda kv: (-kv[1], kv[0])))


def is_cue(entity: dict, lexicon: dict[str, int] | set[str]) -> bool:
    """A entidade e uma pista de negacao segundo o lexico?"""
    return normalize_surface(entity["text"]) in lexicon


def apply_cue_filter(candidates: Sequence[dict[str, Any]],
                     y_pred: Sequence[Any],
                     lexicon: dict[str, int] | set[str],
                     *, labels: Sequence[str] = LABELS) -> list[Any]:
    """Rebaixa para `no_relation` toda predicao `negation_of` sem pista em e1.

    `candidates` e `y_pred` precisam estar na MESMA ordem (a de
    `iter_candidate_pairs` sobre os documentos na ordem do arquivo, que e a ordem
    em que `relation_extraction.build_dataset` monta o dataset). Aceita `y_pred`
    como ids inteiros ou como strings e devolve no mesmo tipo. Nao modifica a
    lista recebida.
    """
    if len(candidates) != len(y_pred):
        raise ValueError(
            f"candidates ({len(candidates)}) e y_pred ({len(y_pred)}) tem "
            "tamanhos diferentes: os sidecars nao pertencem a este espaco de "
            "candidatos.")

    as_int = not isinstance(y_pred[0], str) if len(y_pred) else True
    neg = labels.index("negation_of") if as_int else "negation_of"
    nor = labels.index("no_relation") if as_int else "no_relation"

    out = list(y_pred)
    demoted = 0
    for i, (cand, pred) in enumerate(zip(candidates, y_pred)):
        if pred == neg and not is_cue(cand["e1"], lexicon):
            out[i] = nor
            demoted += 1
    log.info("Filtro de pista: %d predicoes `negation_of` rebaixadas para "
             "`no_relation` (de %d preditas)",
             demoted, sum(1 for p in y_pred if p == neg))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--train", default="data/splits/train.jsonl")
    ap.add_argument("--max-gap", type=int, default=25)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--out", default=None, help="grava o lexico em JSON")
    args = ap.parse_args()

    lex = induce_lexicon(read_jsonl(args.train), args.max_gap, args.min_freq)
    for form, n in lex.items():
        log.info("  %6d  %s", n, form)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"max_gap": args.max_gap, "min_freq": args.min_freq,
                        "lexicon": lex}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        log.info("Lexico salvo em %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
