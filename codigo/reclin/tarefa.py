"""A tarefa: rótulos, pares candidatos e a identidade do conjunto de referência.

É a definição do problema que toda estratégia resolve e em que toda estratégia é
avaliada. Classificar relações entre entidades de uma nota clínica significa
classificar cada par ORDENADO de entidades próximas em um de três rótulos:

* `negation_of` — e1 nega e2 (a classe-alvo do trabalho);
* `associated_with` — e1 e e2 estão associadas;
* `no_relation` — nenhuma relação anotada entre os dois.

O SemClinBr só anota as relações positivas. Os pares `no_relation` são criados
aqui, pela enumeração de candidatos:

1. pares ordenados (e1, e2): a direção importa para `negation_of`;
2. só pares cuja distância entre os spans é de no máximo `MAX_GAP` caracteres,
   para não explodir o número de negativos;
3. nunca entre documentos; sem auto-pares nem pares com span idêntico;
4. ordem determinística: entidades ordenadas por (start, end, id).

Com `MAX_GAP = 25`, as partições congeladas têm 152.686 / 19.064 / 19.210 pares
em train / dev / test. Esse conjunto, nessa ordem, é o CONJUNTO DE REFERÊNCIA:
o `y_true` de qualquer execução é o dele, e duas execuções só são comparáveis se
se referem ao mesmo conjunto (`conjunto_referencia`).

Origem no legado (repositório original, commit a5f055c): `src/candidates.py` (`iter_candidate_pairs`, `entity_gap`),
`LABELS` de `src/relation_extraction.py` e `NEG`/`ASSOC`/`NOREL` e
`N_FULL_ESPERADO` de `src/finetuning_restrito/restricted_space.py`. Os nomes
`iter_candidate_pairs` e `entity_gap` foram mantidos porque o texto do TCC os
cita. Diferença deliberada: o default de `max_gap` passa de 75 (que nenhum
experimento usava) para 25, o valor de todos os experimentos.
"""
from __future__ import annotations

from typing import Iterable, Iterator, TypedDict

from reclin.util.io import sha256_json

# --------------------------------------------------------------------------- #
# Rótulos                                                                     #
# --------------------------------------------------------------------------- #
# A ordem é parte do contrato: define os ids, a ordem das colunas de `probs`
# nos sidecars e a ordem das linhas da matriz de confusão.
LABELS: tuple[str, ...] = ("negation_of", "associated_with", "no_relation")
LABEL2ID: dict[str, int] = {rotulo: i for i, rotulo in enumerate(LABELS)}
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))
NEG = LABEL2ID["negation_of"]
ASSOC = LABEL2ID["associated_with"]
NOREL = LABEL2ID["no_relation"]

# --------------------------------------------------------------------------- #
# Conjunto de referência                                                      #
# --------------------------------------------------------------------------- #
# Janela máxima, em caracteres, entre os spans de um par candidato. Era 20 até
# a análise de sensibilidade (analysis/max_gap/ do legado), que mostrou que 20
# cortava 11,6% das relações anotadas, quase todas `associated_with`.
MAX_GAP = 25

# Número de candidatos por partição com MAX_GAP. Uma divergência aqui significa
# partição ou enumeração diferente, e nenhuma comparação com execuções
# anteriores seria válida.
TAMANHOS_ESPERADOS: dict[str, int] = {"train": 152_686, "dev": 19_064, "test": 19_210}


# --------------------------------------------------------------------------- #
# Tipos dos registros (documentam os dicionários; não mudam o comportamento)  #
# --------------------------------------------------------------------------- #
class Entidade(TypedDict):
    id: str
    start: int
    end: int
    text: str
    type: str


class Relacao(TypedDict):
    e1_id: str
    e2_id: str
    type: str


class Documento(TypedDict):
    doc_id: str
    text: str
    entities: list[Entidade]
    relations: list[Relacao]


class Candidato(TypedDict):
    doc_id: str
    e1: Entidade
    e2: Entidade
    label: str


class ConjuntoReferencia(TypedDict):
    """Identidade do conjunto em que uma execução é avaliada."""
    particao: str
    max_gap: int
    particao_sha256: str
    n_candidatos: int
    candidatos_sha256: str
    y_true_sha256: str


# --------------------------------------------------------------------------- #
# Candidatos                                                                  #
# --------------------------------------------------------------------------- #
def entity_gap(e1: Entidade, e2: Entidade) -> int:
    """Caracteres entre os dois spans; 0 se eles se tocam ou se sobrepõem."""
    if e1["start"] <= e2["start"]:
        return max(0, e2["start"] - e1["end"])
    return max(0, e1["start"] - e2["end"])


def iter_candidate_pairs(doc: Documento, max_gap: int = MAX_GAP) -> Iterator[Candidato]:
    """Enumera os pares candidatos de um documento, com o rótulo anotado.

    O rótulo é o da relação anotada de e1 para e2, ou `no_relation`. Uma
    relação anotada de e2 para e1 não rotula o par (e1, e2).
    """
    anotadas = {(r["e1_id"], r["e2_id"]): r["type"] for r in doc["relations"]}
    entidades = sorted(doc["entities"], key=lambda e: (e["start"], e["end"], e["id"]))

    for i, e1 in enumerate(entidades):
        for j, e2 in enumerate(entidades):
            if i == j:
                continue
            if (e1["start"], e1["end"]) == (e2["start"], e2["end"]):
                continue
            if entity_gap(e1, e2) > max_gap:
                continue
            yield {
                "doc_id": doc["doc_id"],
                "e1": e1,
                "e2": e2,
                "label": anotadas.get((e1["id"], e2["id"]), "no_relation"),
            }


def candidatos(documentos: Iterable[Documento], max_gap: int = MAX_GAP) -> list[Candidato]:
    """Todos os candidatos dos documentos, na ordem do conjunto de referência."""
    return [c for doc in documentos for c in iter_candidate_pairs(doc, max_gap=max_gap)]


def y_true(cands: Iterable[Candidato]) -> list[int]:
    """Ids dos rótulos anotados, na ordem dos candidatos."""
    return [LABEL2ID[c["label"]] for c in cands]


def contagem_por_rotulo(cands: Iterable[Candidato]) -> dict[str, int]:
    """Número de candidatos por rótulo, com todos os rótulos (inclusive zeros)."""
    contagem = dict.fromkeys(LABELS, 0)
    for c in cands:
        contagem[c["label"]] += 1
    return contagem


def chave_candidato(c: Candidato) -> list[str]:
    """O que identifica um candidato no conjunto: documento, e1, e2 e rótulo."""
    return [c["doc_id"], c["e1"]["id"], c["e2"]["id"], c["label"]]


def conjunto_referencia(particao: str, documentos: Iterable[Documento], *,
                        particao_sha256: str, max_gap: int = MAX_GAP) -> ConjuntoReferencia:
    """Identidade do conjunto de referência de uma partição.

    `particao_sha256` é o hash do arquivo da partição (o do MANIFEST). Os outros
    dois hashes cobrem a sequência de candidatos e a de rótulos, na ordem de
    enumeração: duas execuções com a mesma identidade têm o mesmo `y_true`,
    elemento a elemento, e podem ser pareadas.

    Com `max_gap == MAX_GAP`, confere o número de candidatos das três partições
    congeladas e falha se ele divergir.
    """
    cands = candidatos(documentos, max_gap=max_gap)
    esperado = TAMANHOS_ESPERADOS.get(particao) if max_gap == MAX_GAP else None
    if esperado is not None and len(cands) != esperado:
        raise ValueError(f"partição {particao!r} com max_gap={max_gap} tem {len(cands)} "
                         f"candidatos; o conjunto de referência tem {esperado}")
    return {
        "particao": particao,
        "max_gap": max_gap,
        "particao_sha256": particao_sha256,
        "n_candidatos": len(cands),
        "candidatos_sha256": sha256_json([chave_candidato(c) for c in cands]),
        "y_true_sha256": sha256_json(y_true(cands)),
    }
