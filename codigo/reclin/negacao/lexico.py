"""Léxico de pistas de negação, induzido do TRAIN.

Toda relação `negation_of` anotada no SemClinBr tem como primeiro argumento uma
expressão que marca a negação ("sem", "nega", "não", "ausência", negações
morfológicas como "indolor"). O léxico guarda essas formas de superfície e o
predicado `e_pista` diz se uma entidade é uma delas.

DECISÕES (as do legado, mantidas)
---------------------------------
1. Indução só do TRAIN. Induzir do corpus inteiro seria vazamento. As funções
   recebem os documentos; quem chama passa o TRAIN.
2. Indução sobre o ESPAÇO DE CANDIDATOS (`max_gap`), não sobre as relações
   anotadas. Com `max_gap=25` há 1.299 relações `negation_of` anotadas no
   TRAIN, mas só 1.255 pares candidatos com esse rótulo: 44 ficam fora da
   janela. As estratégias operam sobre candidatos, então o léxico também.
3. Normalização = NFKD + remoção de diacríticos + colapso de espaços +
   `casefold`. A remoção de diacríticos não é cosmética: no corpus a mesma
   forma aparece com e sem acento ("NÃO" e "NAO", "AUSÊNCIA" e "AUSENCIA"), e
   sem ela uma forma se partiria em duas, cada uma abaixo do limiar. Com esta
   normalização, `min_freq` 1/2/3/5/10 dá léxicos de 51/17/11/9/8 formas.
4. A identidade de um léxico é `lexico_sha1`: SHA-1 (12 caracteres) das
   formas e contagens ordenadas. Dois léxicos com o mesmo hash são iguais,
   inclusive nas frequências.

COBERTURA
---------
`cobertura` é a fração dos pares candidatos `negation_of` de uma partição cujo
`e1` é pista. Como o filtro de pistas e o fine-tuning restrito só predizem
`negation_of` onde `e1` é pista, ela é o teto de recall dessas estratégias na
classe-alvo. Com o léxico congelado, no DEV: 142 / 150 = 0,9467.

LÉXICO CONGELADO
----------------
`CONGELADO` registra o léxico que todas as estratégias usam para serem
comparáveis: `min_freq=3`, escolhido pela calibração do filtro no DEV dos
baselines do legado (`results/CALIBRACAO_filtro.json`), 11 formas,
`lexico_sha1` `70c93fa807de`. `carregar_congelado` induz o léxico do TRAIN com
esses parâmetros e falha se o resultado não for exatamente esse — se as
partições, a enumeração de candidatos ou a normalização mudarem, nenhum
resultado anterior seria comparável. Quando o filtro for recalibrado sobre os
baselines novos (etapa 8), é este registro que muda, com a nova origem.

Origem no legado: `normalize_surface`, `count_cue_forms`, `induce_lexicon` e
`is_cue` de `src/negation_lexicon.py`; `lexico_sha1` e a guarda de
`carregar_lexico` de `src/finetuning_restrito/restricted_space.py`. O filtro
(`apply_cue_filter`) é estratégia e não está aqui.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from typing import Any, Collection, Iterable, Mapping

from reclin.tarefa import MAX_GAP, Documento, Entidade, iter_candidate_pairs

log = logging.getLogger(__name__)

_ESPACOS = re.compile(r"\s+")

# O léxico que as estratégias usam (ver o docstring do módulo).
CONGELADO: dict[str, Any] = {
    "min_freq": 3,
    "max_gap": MAX_GAP,
    "n_formas": 11,
    "lexico_sha1": "70c93fa807de",
    "origem": "calibração do filtro de pistas no DEV dos baselines do legado "
              "(results/CALIBRACAO_filtro.json, commit a5f055c)",
}


def normalizar(texto: str) -> str:
    """Forma de comparação: sem diacríticos, sem caixa, com espaço único."""
    decomposto = unicodedata.normalize("NFKD", texto)
    sem_diacriticos = "".join(c for c in decomposto if not unicodedata.combining(c))
    return _ESPACOS.sub(" ", sem_diacriticos).strip().casefold()


def contar_formas(documentos: Iterable[Documento], max_gap: int = MAX_GAP) -> Counter[str]:
    """Frequência das formas normalizadas de `e1` nos pares candidatos
    rotulados `negation_of`."""
    contagem: Counter[str] = Counter()
    for doc in documentos:
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            if c["label"] == "negation_of":
                contagem[normalizar(c["e1"]["text"])] += 1
    return contagem


def induzir(documentos_train: Iterable[Documento], *, min_freq: int,
            max_gap: int = MAX_GAP) -> dict[str, int]:
    """Léxico: forma normalizada → frequência, para as formas com frequência
    de pelo menos `min_freq`, da mais frequente para a menos (empate em ordem
    alfabética). Serve como conjunto (`forma in lexico`)."""
    contagem = contar_formas(documentos_train, max_gap)
    lexico = {forma: n for forma, n in contagem.items() if n >= min_freq}
    log.info("Léxico induzido (max_gap=%d, min_freq=%d): %d formas, %d de %d ocorrências cobertas",
             max_gap, min_freq, len(lexico), sum(lexico.values()), sum(contagem.values()))
    return dict(sorted(lexico.items(), key=lambda kv: (-kv[1], kv[0])))


def e_pista(entidade: Entidade, lexico: Collection[str]) -> bool:
    """A entidade é uma pista de negação segundo o léxico?"""
    return normalizar(entidade["text"]) in lexico


def lexico_sha1(lexico: Mapping[str, int]) -> str:
    """Identidade do léxico: SHA-1 das formas e contagens ordenadas (12 caracteres)."""
    blob = json.dumps(sorted(lexico.items()), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def cobertura(documentos: Iterable[Documento], lexico: Collection[str],
              max_gap: int = MAX_GAP) -> dict[str, Any]:
    """Quantos pares candidatos `negation_of` têm `e1` no léxico, e a fração."""
    total = cobertos = 0
    for doc in documentos:
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            if c["label"] == "negation_of":
                total += 1
                cobertos += e_pista(c["e1"], lexico)
    return {"negation_of": total, "cobertos": cobertos,
            "cobertura": cobertos / total if total else None}


def carregar_congelado(documentos_train: Iterable[Documento],
                       registro: Mapping[str, Any] = CONGELADO) -> dict[str, int]:
    """Induz do TRAIN o léxico congelado e confere que é exatamente o registrado."""
    lexico = induzir(documentos_train, min_freq=registro["min_freq"], max_gap=registro["max_gap"])
    sha = lexico_sha1(lexico)
    if len(lexico) != registro["n_formas"] or sha != registro["lexico_sha1"]:
        raise ValueError(
            f"o léxico induzido do TRAIN (min_freq={registro['min_freq']}, "
            f"max_gap={registro['max_gap']}) tem {len(lexico)} formas e lexico_sha1 {sha}; "
            f"o congelado tem {registro['n_formas']} formas e {registro['lexico_sha1']}. "
            "As partições, os candidatos ou a normalização mudaram, e os resultados "
            "deixariam de ser comparáveis.")
    return lexico
