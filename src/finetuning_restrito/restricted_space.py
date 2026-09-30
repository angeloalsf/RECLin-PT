"""
Espaco restrito de candidatos (e1 no lexico de pistas) e remapeamento ao espaco
completo.

O QUE E O ESPACO RESTRITO
-------------------------
O espaco completo e o de `src/candidates.iter_candidate_pairs(max_gap=25)`, o
mesmo dos quatro baselines: 152.686 / 19.064 / 19.210 pares em
train / dev / test. O espaco restrito guarda so os pares cujo e1 e uma pista de
negacao segundo o lexico induzido do TRAIN por `negation_lexicon.induce_lexicon`
com `min_freq=3`, a mesma calibracao do filtro da fase 2. Nada e recalibrado
aqui: reusar o mesmo lexico e o que torna as duas frentes comparaveis.

O modelo desta frente e treinado, selecionado e avaliado SO sobre esse espaco.

REMAPEAMENTO (existe desde o primeiro commit)
---------------------------------------------
Para comparar com os baselines via `src/significance.py` sem alterar uma linha
dele, as predicoes precisam voltar ao espaco completo, na mesma ordem e com o
mesmo `y_true`. A regra e:

* `y_true` NUNCA e remapeado. Ele e o gold do espaco completo, reconstruido pela
  mesma iteracao que `relation_extraction.build_dataset` usa. As relacoes
  `negation_of` cujo e1 ficou fora do lexico continuam no `y_true` e viram falso
  negativo: e isso que impoe o teto de recall.
* `y_pred` de todo par descartado pelo filtro e `no_relation`. Dentro do espaco
  restrito vale a predicao do modelo.

Consequencia que precisa ser lida junto com qualquer numero desta frente: quase
todo `associated_with` do gold fica fora do espaco restrito (17 de 7.096 pares
no train), entao o F1 de `associated_with` remapeado e praticamente zero, o
macro-F1 remapeado NAO e comparavel ao dos baselines, e o McNemar de
`significance.py` (que mede acerto global, nas tres classes) tambem nao. A
comparacao valida e o bootstrap no F1 de `negation_of`, a metrica-alvo.

Este modulo nao importa torch: o remapeamento e testavel sem GPU e sem pesos.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import _isolamento  # noqa: F401  (tem de vir antes dos imports do nucleo)

from candidates import iter_candidate_pairs  # noqa: E402
from negation_lexicon import induce_lexicon, is_cue  # noqa: E402
from relation_extraction import (LABEL2ID, LABELS,  # noqa: E402
                                 build_marked_window, read_jsonl)

log = _isolamento.get()

NEG = LABEL2ID["negation_of"]
ASSOC = LABEL2ID["associated_with"]
NOREL = LABEL2ID["no_relation"]

# Configuracao congelada do pipeline dos baselines (results/baseline_*.json,
# chave `config`). Defaults daqui para frente sao estes, nao os do argparse de
# relation_extraction (75 / 192 / 32), que nao sao os usados nas rodadas.
MAX_GAP = 25
CTX_CHARS = 128
MIN_FREQ = 3

# Tamanhos do espaco completo, verificados contra os sidecars oficiais. Servem
# de guarda: se algum dia o espaco mudar (max_gap, splits), o remapeamento para
# de ser comparavel com os baselines e isto precisa falhar alto.
N_FULL_ESPERADO = {"train": 152686, "dev": 19064, "test": 19210}


# --------------------------------------------------------------------------- #
# Lexico                                                                       #
# --------------------------------------------------------------------------- #
def lexico_sha1(lexico: dict[str, int]) -> str:
    blob = json.dumps(sorted(lexico.items()), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def carregar_lexico(train_docs: Iterable[dict], *, max_gap: int = MAX_GAP,
                    min_freq: int = MIN_FREQ,
                    guarda: str | Path | None = None) -> dict[str, int]:
    """Induz o lexico do TRAIN com `negation_lexicon.induce_lexicon`.

    `guarda` aponta para `results/CALIBRACAO_filtro.json` (a calibracao da
    fase 2). Se existir, `min_freq` e o tamanho do lexico precisam bater com o
    que ficou congelado la; senao aborta. Se nao existir (por exemplo num clone
    sem os resultados), segue com um aviso.
    """
    lex = induce_lexicon(train_docs, max_gap, min_freq)
    if guarda is not None:
        g = Path(guarda)
        if g.is_file():
            cal = json.loads(g.read_text(encoding="utf-8"))
            if int(cal.get("min_freq", -1)) != int(min_freq):
                raise SystemExit(
                    f"min_freq={min_freq} difere da calibracao congelada em {g} "
                    f"(min_freq={cal.get('min_freq')}). Esta frente reusa a "
                    f"calibracao da fase 2 e nao recalibra.")
            if int(cal.get("lexicon_size", -1)) != len(lex):
                raise SystemExit(
                    f"Lexico induzido tem {len(lex)} formas, a calibracao em {g} "
                    f"registra {cal.get('lexicon_size')}. O train ou o max_gap "
                    f"mudaram; os resultados deixariam de ser comparaveis.")
            log.info("Lexico confere com %s (min_freq=%d, %d formas).",
                     g, min_freq, len(lex))
        else:
            log.warning("Guarda de lexico %s ausente: nao foi possivel conferir "
                        "contra a calibracao da fase 2.", g)
    return lex


# --------------------------------------------------------------------------- #
# Espaco restrito de um split                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class EspacoRestrito:
    split: str
    n_full: int = 0
    indices: list[int] = field(default_factory=list)   # posicoes no espaco completo
    y_true_full: list[int] = field(default_factory=list)
    textos: list[str] = field(default_factory=list)    # janelas marcadas, so restritos
    y_restrito: list[int] = field(default_factory=list)

    @property
    def n_restrito(self) -> int:
        return len(self.indices)

    def contagens(self) -> dict:
        full = {l: 0 for l in LABELS}
        rest = {l: 0 for l in LABELS}
        for y in self.y_true_full:
            full[LABELS[y]] += 1
        for y in self.y_restrito:
            rest[LABELS[y]] += 1
        return {"completo": full, "restrito": rest}

    def teto_recall(self, label: str = "negation_of") -> float | None:
        """Fracao do gold `label` que cabe no espaco restrito (teto de recall)."""
        c = self.contagens()
        tot = c["completo"][label]
        return (c["restrito"][label] / tot) if tot else None

    def remapear(self, y_pred_restrito: Sequence[int],
                 probs_restrito: Sequence[Sequence[float]] | None = None):
        return remapear(self.indices, self.n_full, y_pred_restrito,
                        probs_restrito)

    def resumo(self) -> dict:
        c = self.contagens()
        return {
            "n_completo": self.n_full,
            "n_restrito": self.n_restrito,
            "fracao": round(self.n_restrito / self.n_full, 6) if self.n_full else None,
            "contagens_completo": c["completo"],
            "contagens_restrito": c["restrito"],
            "teto_recall_negation_of": self.teto_recall("negation_of"),
            "teto_recall_associated_with": self.teto_recall("associated_with"),
            "razao_no_relation_por_negation_of": (
                round(c["restrito"]["no_relation"] / c["restrito"]["negation_of"], 4)
                if c["restrito"]["negation_of"] else None),
        }


def construir_espaco(docs: Iterable[dict], split: str, lexico: dict[str, int],
                     *, max_gap: int = MAX_GAP, ctx_chars: int = CTX_CHARS,
                     checar_tamanho: bool = True) -> EspacoRestrito:
    """Percorre o espaco completo na MESMA ordem de `build_dataset` e separa os
    candidatos cujo e1 e pista.

    A janela marcada de cada candidato restrito sai de
    `relation_extraction.build_marked_window` (importada, nao copiada), com os
    mesmos `max_gap`/`ctx_chars` dos baselines. Assim cada exemplo de treino
    desta frente e, byte a byte, o mesmo exemplo que o baseline viu para aquele
    par; o que muda e so quais pares entram.
    """
    esp = EspacoRestrito(split=split)
    pos = 0
    for doc in docs:
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            y = LABEL2ID[c["label"]]
            esp.y_true_full.append(y)
            if is_cue(c["e1"], lexico):
                esp.indices.append(pos)
                esp.textos.append(build_marked_window(doc["text"], c["e1"], c["e2"],
                                                      ctx_chars))
                esp.y_restrito.append(y)
            pos += 1
    esp.n_full = pos
    if checar_tamanho and max_gap == MAX_GAP and split in N_FULL_ESPERADO:
        if esp.n_full != N_FULL_ESPERADO[split]:
            raise SystemExit(
                f"Espaco completo do {split} tem {esp.n_full} candidatos, "
                f"esperado {N_FULL_ESPERADO[split]} (o dos baselines). O "
                f"remapeamento deixaria de ser comparavel.")
    log.info("Espaco restrito %-5s: %6d de %6d candidatos (%.2f%%) | teto de "
             "recall negation_of=%.4f", split, esp.n_restrito, esp.n_full,
             100.0 * esp.n_restrito / max(1, esp.n_full),
             esp.teto_recall() or float("nan"))
    return esp


def carregar_espacos(splits_dir: str | Path, lexico: dict[str, int], *,
                     max_gap: int = MAX_GAP, ctx_chars: int = CTX_CHARS,
                     splits: Sequence[str] = ("train", "dev", "test"),
                     checar_tamanho: bool = True) -> dict[str, EspacoRestrito]:
    sp = Path(splits_dir)
    return {s: construir_espaco(read_jsonl(sp / f"{s}.jsonl"), s, lexico,
                                max_gap=max_gap, ctx_chars=ctx_chars,
                                checar_tamanho=checar_tamanho)
            for s in splits}


# --------------------------------------------------------------------------- #
# Remapeamento                                                                 #
# --------------------------------------------------------------------------- #
PROBS_FORA = [0.0, 0.0, 1.0]   # descartado pelo filtro = no_relation com certeza


def remapear(indices: Sequence[int], n_full: int,
             y_pred_restrito: Sequence[int],
             probs_restrito: Sequence[Sequence[float]] | None = None):
    """Leva predicoes do espaco restrito de volta ao completo.

    Devolve `(y_pred_full, probs_full)`; `probs_full` e None se
    `probs_restrito` for None. Fora do espaco: `no_relation` e probabilidade
    [0, 0, 1] (a decisao e da regra, nao do modelo).
    """
    if len(indices) != len(y_pred_restrito):
        raise ValueError(f"{len(y_pred_restrito)} predicoes para "
                         f"{len(indices)} posicoes do espaco restrito.")
    if probs_restrito is not None and len(probs_restrito) != len(indices):
        raise ValueError("probs_restrito com tamanho diferente de indices.")
    if len(set(indices)) != len(indices) or (indices and (
            min(indices) < 0 or max(indices) >= n_full)):
        raise ValueError("indices do espaco restrito invalidos para n_full="
                         f"{n_full}.")
    y_full = [NOREL] * n_full
    for i, y in zip(indices, y_pred_restrito):
        y_full[i] = int(y)
    probs_full = None
    if probs_restrito is not None:
        probs_full = [list(PROBS_FORA) for _ in range(n_full)]
        for i, p in zip(indices, probs_restrito):
            probs_full[i] = [float(x) for x in p]
    return y_full, probs_full


# --------------------------------------------------------------------------- #
# Pesos de classe                                                              #
# --------------------------------------------------------------------------- #
def pesos_balanced(y: Sequence[int], n_classes: int = len(LABELS)) -> list[float]:
    """Mesma formula de `relation_extraction.run` (class_weight=balanced),
    aplicada ao train RESTRITO: w_c = N / (C * n_c), com n_c=0 tratado como 1.

    A formula esta inline dentro de `run()` e nao e uma funcao importavel; por
    isso e reescrita aqui. Sao tres linhas e o teste compara o resultado contra
    o valor que `run()` registrou para o espaco completo (40,55 para
    `negation_of`).
    """
    counts = [0] * n_classes
    for v in y:
        counts[int(v)] += 1
    counts = [c if c > 0 else 1 for c in counts]
    tot = float(sum(counts))
    return [tot / (n_classes * c) for c in counts]


def conferir_y_true(espaco: EspacoRestrito, sidecar: str | Path) -> bool:
    """O `y_true` reconstruido bate com o de um sidecar oficial do mesmo split?"""
    d = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    if d.get("labels") != LABELS:
        raise SystemExit(f"{sidecar}: labels {d.get('labels')} != {LABELS}")
    return list(d["y_true"]) == list(espaco.y_true_full)
