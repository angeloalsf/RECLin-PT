"""A entrada dos modelos: janela marcada de cada candidato e o loader tokenizado.

Cada par candidato (na ordem de `tarefa.iter_candidate_pairs`) vira um texto: a
janela do documento com `ctx_chars` caracteres de contexto de cada lado do par,
com os marcadores de entidade inseridos em volta de e1 e de e2:

    "... paciente [E1] nega [/E1] [E2] febre [/E2] e tosse ..."

Os marcadores são tokens especiais acrescentados ao tokenizer
(`modelos.carregar_tokenizer`), sem tipo semântico (Soares et al., 2019,
"Matching the Blanks"). Os offsets das entidades são usados como estão: quando
uma entidade se sobrepõe à outra, os marcadores saem na ordem definida pela
prioridade (fechamentos antes de aberturas na mesma posição) — o mesmo texto
que o legado produzia, inclusive nos casos degenerados.

O loader tokeniza cada lote no momento de montá-lo (`padding` até o maior
exemplo do lote, `truncation` em `max_length`) e embaralha com um
`torch.Generator` próprio, semeado com a semente da execução. O estado desse
gerador é o que define a ordem dos exemplos de cada época; por isso o
checkpoint o guarda (`treino.checkpoint`).

Exemplos e rótulos andam juntos: `exemplos` devolve as duas listas alinhadas
com os candidatos, e o loader devolve, em cada lote, os rótulos dos mesmos
exemplos tokenizados.

Origem no legado: `E1_OPEN`..., `MARKER_TOKENS`, `build_marked_window`,
`build_dataset` e `make_loader` de `src/relation_extraction.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from reclin.tarefa import LABEL2ID, MAX_GAP, Documento, Entidade, iter_candidate_pairs

E1_ABRE, E1_FECHA, E2_ABRE, E2_FECHA = "[E1]", "[/E1]", "[E2]", "[/E2]"
MARCADORES = (E1_ABRE, E1_FECHA, E2_ABRE, E2_FECHA)
CTX_CHARS = 128


def janela_marcada(texto: str, e1: Entidade, e2: Entidade, ctx_chars: int = CTX_CHARS) -> str:
    """Recorte do texto em torno do par, com os marcadores inseridos."""
    inicio_par, fim_par = min(e1["start"], e2["start"]), max(e1["end"], e2["end"])
    ini, fim = max(0, inicio_par - ctx_chars), min(len(texto), fim_par + ctx_chars)
    janela = texto[ini:fim]

    # posição na janela -> [(prioridade, marcador)]; na mesma posição, os de
    # prioridade 0 (fechamentos) vêm antes dos de prioridade 1 (aberturas)
    insercoes: dict[int, list[tuple[int, str]]] = {}
    for pos, marcador, prioridade in ((e1["start"] - ini, E1_ABRE + " ", 1),
                                      (e1["end"] - ini, " " + E1_FECHA, 0),
                                      (e2["start"] - ini, E2_ABRE + " ", 1),
                                      (e2["end"] - ini, " " + E2_FECHA, 0)):
        insercoes.setdefault(pos, []).append((prioridade, marcador))

    partes = []
    for i in range(len(janela) + 1):
        if i in insercoes:
            partes.extend(m for _, m in sorted(insercoes[i]))
        if i < len(janela):
            partes.append(janela[i])
    return "".join(partes)


@dataclass(frozen=True)
class Exemplos:
    """Textos marcados e rótulos (ids), alinhados com os candidatos da partição."""

    textos: list[str]
    rotulos: list[int]

    def __post_init__(self) -> None:
        if len(self.textos) != len(self.rotulos):
            raise ValueError(f"{len(self.textos)} textos e {len(self.rotulos)} rótulos")

    def __len__(self) -> int:
        return len(self.rotulos)

    def subconjunto(self, indices: Sequence[int]) -> "Exemplos":
        """Os exemplos nas posições `indices`, na ordem dada (a mesma janela e
        o mesmo rótulo de cada candidato; só muda quais entram)."""
        return Exemplos([self.textos[i] for i in indices], [self.rotulos[i] for i in indices])


def exemplos(documentos: Iterable[Documento], *, max_gap: int = MAX_GAP,
             ctx_chars: int = CTX_CHARS) -> Exemplos:
    """Um exemplo por candidato, na ordem do conjunto de referência."""
    textos, rotulos = [], []
    for doc in documentos:
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            textos.append(janela_marcada(doc["text"], c["e1"], c["e2"], ctx_chars))
            rotulos.append(LABEL2ID[c["label"]])
    return Exemplos(textos, rotulos)


def criar_loader(tokenizer: Any, dados: Exemplos, *, max_length: int, batch_size: int,
                 embaralhar: bool, seed: int) -> Any:
    """DataLoader que tokeniza cada lote: devolve `(input_ids, attention_mask,
    rotulos)`. O embaralhamento usa um gerador próprio (`loader.generator`),
    semeado com `seed`, e não o RNG global."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    textos, rotulos = dados.textos, dados.rotulos

    class _Exemplos(Dataset):
        def __len__(self) -> int:
            return len(textos)

        def __getitem__(self, i: int) -> tuple[str, int]:
            return textos[i], rotulos[i]

    def montar_lote(lote: Sequence[tuple[str, int]]):
        lote_textos, lote_rotulos = zip(*lote)
        enc = tokenizer(list(lote_textos), truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(lote_rotulos)

    gerador = torch.Generator().manual_seed(seed)
    return DataLoader(_Exemplos(), batch_size=batch_size, shuffle=embaralhar,
                      generator=gerador, collate_fn=montar_lote)
