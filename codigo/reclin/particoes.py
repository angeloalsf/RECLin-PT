"""Partições train/dev/test por documento e o MANIFEST que as congela.

POR QUE POR DOCUMENTO
---------------------
Relações do mesmo prontuário compartilham vocabulário; se caíssem em partições
diferentes haveria vazamento e a métrica do TEST inflaria. Por isso a unidade da
partição é o documento, nunca a relação.

COMO
----
Estratificação leve pela PRESENÇA de `negation_of`: os documentos com ao menos
uma relação de negação e os sem nenhuma são embaralhados separadamente (mesma
semente, nessa ordem) e cortados 80/10/10 dentro de cada grupo — primeiro o
TEST, depois o DEV, o resto vai para o TRAIN. Isso garante negação nas três
partições sem pacote externo de estratificação. Cada partição sai ordenada por
`doc_id`. Semente fixa: 42.

MANIFEST
--------
`MANIFEST.json` registra, para cada partição, o SHA-256, o número de documentos
(`n_records`), o tamanho em bytes e a contagem de relações por tipo, além da
semente e do dataset de origem. Sem timestamp, de propósito: a mesma entrada
produz o mesmo arquivo byte a byte, então um diff no git significa que o
conteúdo mudou. As chaves são as do MANIFEST do legado, citado no apêndice A
do TCC.

Origem no legado: `src/make_splits.py`. O algoritmo é o mesmo, e
`testes/equivalencia/test_particoes.py` confere que ele reproduz as partições
congeladas bit a bit a partir dos próprios documentos. Diferença deliberada:
a ordenação inicial usa a mesma chave da final, que não falha com `doc_id`
não numérico (para os ids do SemClinBr, todos numéricos, a ordem é a mesma).
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from reclin.tarefa import Documento
from reclin.util.io import gravar_json, gravar_jsonl, ler_json, ler_jsonl, sha256_arquivo

log = logging.getLogger(__name__)

PARTICOES = ("train", "dev", "test")
ARQUIVO_MANIFESTO = "MANIFEST.json"
SEED = 42
PROPORCAO_DEV = 0.10
PROPORCAO_TEST = 0.10
GERADOR = "reclin/particoes.py"


# --------------------------------------------------------------------------- #
# Particionamento                                                             #
# --------------------------------------------------------------------------- #
def _chave_doc(doc: Documento) -> tuple[int, str]:
    doc_id = doc["doc_id"]
    return (int(doc_id) if doc_id.isdigit() else 0, doc_id)


def _tem_negacao(doc: Documento) -> bool:
    return any(r["type"] == "negation_of" for r in doc["relations"])


def particionar(documentos: Iterable[Documento], seed: int = SEED) -> dict[str, list[Documento]]:
    """Divide os documentos em train/dev/test (80/10/10), estratificando pela
    presença de `negation_of`. Determinístico para a mesma entrada e semente."""
    docs = sorted(documentos, key=_chave_doc)
    repetidos = [k for k, n in Counter(d["doc_id"] for d in docs).items() if n > 1]
    if repetidos:
        raise ValueError(f"doc_id repetido no dataset: {repetidos[:5]}")

    com_negacao = [d for d in docs if _tem_negacao(d)]
    sem_negacao = [d for d in docs if not _tem_negacao(d)]
    log.info("Estratificação por negação: %d documentos com negation_of, %d sem",
             len(com_negacao), len(sem_negacao))

    rng = random.Random(seed)
    rng.shuffle(com_negacao)
    rng.shuffle(sem_negacao)

    saida: dict[str, list[Documento]] = {nome: [] for nome in PARTICOES}
    for grupo in (com_negacao, sem_negacao):
        n_test = round(len(grupo) * PROPORCAO_TEST)
        n_dev = round(len(grupo) * PROPORCAO_DEV)
        saida["test"] += grupo[:n_test]
        saida["dev"] += grupo[n_test:n_test + n_dev]
        saida["train"] += grupo[n_test + n_dev:]
    for nome in PARTICOES:
        saida[nome].sort(key=_chave_doc)
    return saida


def arquivo_particao(pasta: str | Path, nome: str) -> Path:
    if nome not in PARTICOES:
        raise ValueError(f"partição desconhecida: {nome!r}; opções: {PARTICOES}")
    return Path(pasta) / f"{nome}.jsonl"


def gravar_particoes(particoes: dict[str, list[Documento]], pasta: str | Path) -> dict[str, int]:
    """Grava `<nome>.jsonl` para cada partição e devolve o número de documentos."""
    return {nome: gravar_jsonl(arquivo_particao(pasta, nome), particoes[nome])
            for nome in PARTICOES}


def ler_particao(pasta: str | Path, nome: str) -> list[Documento]:
    return list(ler_jsonl(arquivo_particao(pasta, nome)))


def ler_particoes(pasta: str | Path) -> dict[str, list[Documento]]:
    return {nome: ler_particao(pasta, nome) for nome in PARTICOES}


# --------------------------------------------------------------------------- #
# MANIFEST                                                                    #
# --------------------------------------------------------------------------- #
def descrever_particao(arquivo: str | Path) -> dict[str, Any]:
    """Entrada do MANIFEST para um arquivo de partição já gravado."""
    relacoes: Counter[str] = Counter()
    n_documentos = 0
    for doc in ler_jsonl(arquivo):
        n_documentos += 1
        relacoes.update(r["type"] for r in doc["relations"])
    return {
        "sha256": sha256_arquivo(arquivo),
        "n_records": n_documentos,
        "n_bytes": Path(arquivo).stat().st_size,
        "n_relations": dict(sorted(relacoes.items())),
    }


def calcular_manifesto(pasta: str | Path, *, seed: int, origem: dict[str, Any]) -> dict[str, Any]:
    """MANIFEST das partições presentes em `pasta`.

    `origem` descreve o dataset de onde as partições vieram:
    `{"path": ..., "sha256": ...}`. Quem chama decide como obtê-la (ver
    `scripts/preparar_dados.py`), porque o dataset contém o texto do corpus e
    nem sempre está disponível.
    """
    return {
        "generator": GERADOR,
        "seed": seed,
        "source": dict(origem),
        "splits": {nome: descrever_particao(arquivo_particao(pasta, nome)) for nome in PARTICOES},
    }


def gravar_manifesto(pasta: str | Path, manifesto: dict[str, Any]) -> Path:
    caminho = Path(pasta) / ARQUIVO_MANIFESTO
    gravar_json(caminho, manifesto)
    return caminho


def ler_manifesto(pasta: str | Path) -> dict[str, Any]:
    return ler_json(Path(pasta) / ARQUIVO_MANIFESTO)


def conferir_particoes(pasta: str | Path, nomes: Iterable[str] = PARTICOES) -> list[str]:
    """Compara os arquivos das partições com o MANIFEST da pasta.

    Devolve a lista de divergências; vazia quando tudo confere. Deve ser
    chamada antes de qualquer uso das partições que gere números para o TCC.
    `nomes` restringe a conferência às partições que serão lidas (a calibração
    no DEV, por exemplo, não abre o arquivo do TEST).
    """
    manifesto = ler_manifesto(pasta)
    divergencias = []
    for nome in nomes:
        arquivo = arquivo_particao(pasta, nome)
        if not arquivo.exists():
            divergencias.append(f"{arquivo.name}: arquivo ausente")
            continue
        registrado = manifesto["splits"].get(nome)
        atual = descrever_particao(arquivo)
        if registrado is None:
            divergencias.append(f"{nome}: ausente do {ARQUIVO_MANIFESTO}")
            continue
        for campo in ("sha256", "n_records", "n_bytes", "n_relations"):
            if atual[campo] != registrado.get(campo):
                divergencias.append(f"{nome}.{campo}: arquivo tem {atual[campo]!r}, "
                                    f"{ARQUIVO_MANIFESTO} registra {registrado.get(campo)!r}")
    return divergencias
