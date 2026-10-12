"""Os marcadores de entidade nas janelas e depois da tokenização.

Duas verificações, por partição e por rótulo:

1. **Na janela** (`entrada.janela_marcada`, sem tokenizer): cada um dos
   quatro marcadores aparece exatamente uma vez, `[E1]` antes de `[/E1]` e
   `[E2]` antes de `[/E2]`. Entidades sobrepostas produzem janelas em que os
   pares de marcadores se entrelaçam; são contadas à parte.
2. **Depois da tokenização** (com um tokenizer que tenha os marcadores como
   tokens especiais, `modelos.carregar_tokenizer`): quantos exemplos perdem
   `[E1]` ou `[E2]` no truncamento em `max_length`. A Pair-Aware lê o estado
   desses dois tokens e, sem eles, cai no `[CLS]`; o baseline e o restrito
   perdem a marcação da entidade.

Os offsets das entidades são usados como estão (`analises.alinhamento`
descreve os desalinhados). Nada aqui depende de um modelo treinado.

Origem no legado (só a finalidade): `auditoria_marcadores.py`.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from reclin.entrada import E1_ABRE, E1_FECHA, E2_ABRE, E2_FECHA, MARCADORES, janela_marcada
from reclin.tarefa import ID2LABEL, LABELS, MAX_GAP, Documento, iter_candidate_pairs

LOTE = 256


def situacao_janela(janela: str) -> str:
    """'ok', 'entrelacados' (pares de marcadores se cruzam) ou 'invalida'."""
    if any(janela.count(m) != 1 for m in MARCADORES):
        return "invalida"
    a1, f1, a2, f2 = (janela.index(m) for m in (E1_ABRE, E1_FECHA, E2_ABRE, E2_FECHA))
    if not (a1 < f1 and a2 < f2):
        return "invalida"
    disjuntos = f1 < a2 or f2 < a1
    aninhados = (a1 < a2 and f2 < f1) or (a2 < a1 and f1 < f2)
    return "ok" if disjuntos or aninhados else "entrelacados"


def auditar_janelas(documentos: Iterable[Documento], *, max_gap: int = MAX_GAP,
                    ctx_chars: int = 128) -> dict[str, Any]:
    por_situacao: dict[str, Counter[str]] = {}
    for doc in documentos:
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            s = situacao_janela(janela_marcada(doc["text"], c["e1"], c["e2"], ctx_chars))
            por_situacao.setdefault(s, Counter())[c["label"]] += 1
    return {s: {r: cont.get(r, 0) for r in LABELS} for s, cont in sorted(por_situacao.items())}


def auditar_tokenizacao(tokenizer: Any, textos: Sequence[str], rotulos: Sequence[int], *,
                        max_length: int, lote: int = LOTE) -> dict[str, Any]:
    """Exemplos sem `[E1]`, sem `[E2]` ou sem os dois depois do truncamento,
    por rótulo, e o tamanho (em tokens, sem truncar) das janelas."""
    ids = {m: tokenizer.convert_tokens_to_ids(m) for m in (E1_ABRE, E2_ABRE)}
    if any(i is None or i == tokenizer.unk_token_id for i in ids.values()):
        raise ValueError("o tokenizer não tem os marcadores como tokens: use modelos.carregar_tokenizer")
    contagem: dict[str, Counter[str]] = {k: Counter() for k in ("sem_e1", "sem_e2", "sem_nenhum", "total")}
    tamanhos = []
    for i in range(0, len(textos), lote):
        bloco = list(textos[i:i + lote])
        truncados = tokenizer(bloco, truncation=True, max_length=max_length)["input_ids"]
        inteiros = tokenizer(bloco, truncation=False)["input_ids"]
        for seq, inteira, y in zip(truncados, inteiros, rotulos[i:i + lote]):
            rotulo = ID2LABEL[int(y)]
            tem1, tem2 = ids[E1_ABRE] in seq, ids[E2_ABRE] in seq
            contagem["total"][rotulo] += 1
            contagem["sem_e1"][rotulo] += not tem1
            contagem["sem_e2"][rotulo] += not tem2
            contagem["sem_nenhum"][rotulo] += not (tem1 or tem2)
            tamanhos.append(len(inteira))
    tamanhos.sort()
    n = len(tamanhos)
    return {"max_length": max_length,
            **{k: {r: c.get(r, 0) for r in LABELS} for k, c in contagem.items()},
            "acima_de_max_length": sum(1 for t in tamanhos if t > max_length),
            "tokens": {"n": n, "mediana": tamanhos[n // 2] if n else None,
                       "p95": tamanhos[min(n - 1, int(0.95 * n))] if n else None,
                       "max": tamanhos[-1] if n else None}}
