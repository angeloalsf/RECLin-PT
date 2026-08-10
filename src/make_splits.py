#!/usr/bin/env python3
"""
Splits 80/10/10 em NIVEL DE DOCUMENTO, com manifesto SHA-256.

Por que nivel de documento e nao de relacao: relacoes do mesmo prontuario
compartilham vocabulario; se cairem em splits diferentes ha vazamento e a
metrica de teste infla. E o "leakage canonico" do NLP clinico.

Estratificacao leve por PRESENCA de `negation_of`: separamos os documentos que
tem ao menos uma relacao de negacao dos que nao tem, embaralhamos cada grupo
com seed 42 e cortamos 80/10/10 dentro de cada grupo. Isso garante negacao em
train/dev/test sem depender de pacote externo de estratificacao multi-rotulo.

Seed FIXO = 42 (reprodutibilidade).

MANIFESTO
---------
Alem dos tres `.jsonl`, grava `data/splits/MANIFEST.json` com o SHA-256, a
contagem de registros (documentos) e a contagem de relacoes por tipo de cada
particao, mais a seed e o arquivo de entrada usados. Motivo: os splits sao
versionados e rodar este script por engano (com outra seed, ou com um
`dataset.jsonl` diferente) os sobrescreve em silencio -- e as quatro execucoes
de baseline deixariam de ser comparaveis sem que nada acusasse. Com o
manifesto, o `git diff` mostra o hash mudando.

O manifesto NAO tem timestamp de propria proposito: rodar duas vezes com a
mesma entrada e a mesma seed produz um arquivo byte-a-byte identico, entao
qualquer diff no git significa que o conteudo mudou de verdade.

    python src/make_splits.py                   # gera splits + MANIFEST.json
    python src/make_splits.py --manifest-only   # so recalcula o manifesto dos
                                                # splits que ja estao no disco
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.logger import get_logger  # noqa: E402

log = get_logger("make_splits")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n")
            n += 1
    return n


def split_indices(n, seed):
    """Indices 80/10/10 de uma lista ja embaralhada de tamanho n."""
    n_test = round(n * 0.10)
    n_dev = round(n * 0.10)
    test = list(range(0, n_test))
    dev = list(range(n_test, n_test + n_dev))
    train = list(range(n_test + n_dev, n))
    return train, dev, test


SPLIT_NAMES = ("train", "dev", "test")
MANIFEST_NAME = "MANIFEST.json"


def sha256_of(path):
    """SHA-256 do arquivo, lido em blocos (os splits chegam a 4,6 MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_split(path):
    """Entrada do manifesto para um `.jsonl` de split ja gravado no disco."""
    rels = Counter()
    n_records = 0
    for doc in read_jsonl(path):
        n_records += 1
        for r in doc["relations"]:
            rels[r["type"]] += 1
    return {
        "sha256": sha256_of(path),
        "n_records": n_records,
        "n_bytes": Path(path).stat().st_size,
        "n_relations": dict(sorted(rels.items())),
    }


def write_manifest(out_dir, seed, input_path):
    """Grava `MANIFEST.json` descrevendo os splits presentes em `out_dir`."""
    out_dir = Path(out_dir)
    manifest = {
        "generator": "src/make_splits.py",
        "seed": seed,
        "source": {
            "path": Path(input_path).as_posix(),
            "sha256": sha256_of(input_path) if Path(input_path).exists() else None,
        },
        "splits": {
            name: describe_split(out_dir / f"{name}.jsonl") for name in SPLIT_NAMES
        },
    }
    path = out_dir / MANIFEST_NAME
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return manifest, path


def summarize(name, docs):
    rels = Counter()
    with_neg = 0
    for d in docs:
        types = {r["type"] for r in d["relations"]}
        if "negation_of" in types:
            with_neg += 1
        for r in d["relations"]:
            rels[r["type"]] += 1
    return name, len(docs), with_neg, dict(rels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/dataset.jsonl")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest-only", action="store_true",
                    help="nao regera os splits: so recalcula MANIFEST.json a "
                         "partir dos .jsonl que ja existem em --out-dir")
    args = ap.parse_args()

    if args.manifest_only:
        out = Path(args.out_dir)
        faltando = [n for n in SPLIT_NAMES if not (out / f"{n}.jsonl").exists()]
        if faltando:
            log.error("splits ausentes em %s: %s", out,
                      ", ".join(f"{n}.jsonl" for n in faltando))
            return 2
        manifest, path = write_manifest(out, args.seed, args.input)
        log.info("Manifesto recalculado (sem regerar splits): %s", path)
        for name in SPLIT_NAMES:
            entry = manifest["splits"][name]
            log.info("  %-6s sha256=%s n_records=%d",
                     name, entry["sha256"], entry["n_records"])
        return 0

    log.info("Iniciando splits | input=%s | out-dir=%s | seed=%d",
             args.input, args.out_dir, args.seed)

    docs = sorted(read_jsonl(args.input), key=lambda d: int(d["doc_id"])
                  if d["doc_id"].isdigit() else d["doc_id"])
    if not docs:
        log.error("Dataset vazio: %s", args.input)
        return 2
    log.info("%d documentos carregados", len(docs))

    has_neg = [d for d in docs if any(r["type"] == "negation_of"
                                      for r in d["relations"])]
    no_neg = [d for d in docs if d not in has_neg]
    log.info("Estratificacao por negacao: %d docs com negation_of, %d sem",
             len(has_neg), len(no_neg))

    rng = random.Random(args.seed)
    rng.shuffle(has_neg)
    rng.shuffle(no_neg)

    train, dev, test = [], [], []
    for group in (has_neg, no_neg):
        tr, dv, te = split_indices(len(group), args.seed)
        train += [group[i] for i in tr]
        dev += [group[i] for i in dv]
        test += [group[i] for i in te]

    # ordem deterministica final por doc_id
    keyf = lambda d: (int(d["doc_id"]) if d["doc_id"].isdigit() else 0, d["doc_id"])
    train.sort(key=keyf)
    dev.sort(key=keyf)
    test.sort(key=keyf)

    out = Path(args.out_dir)
    n_tr = write_jsonl(out / "train.jsonl", train)
    n_dv = write_jsonl(out / "dev.jsonl", dev)
    n_te = write_jsonl(out / "test.jsonl", test)

    n = len(docs)
    log.info("Splits 80/10/10 (doc-level, seed=%d) gravados em %s", args.seed, out)
    log.info("  train: %d (%.1f%%) | dev: %d (%.1f%%) | test: %d (%.1f%%)",
             n_tr, 100 * n_tr / n, n_dv, 100 * n_dv / n, n_te, 100 * n_te / n)
    for name, docs_ in (("train", train), ("dev", dev), ("test", test)):
        _, nd, wn, rels = summarize(name, docs_)
        log.info("  %-6s docs=%-5d docs_com_negacao=%-4d | negation_of=%-5d associated_with=%d",
                 name, nd, wn, rels.get("negation_of", 0), rels.get("associated_with", 0))

    manifest, manifest_path = write_manifest(out, args.seed, args.input)
    log.info("Manifesto SHA-256 gravado em %s", manifest_path)
    for name in SPLIT_NAMES:
        log.info("  %-6s sha256=%s", name, manifest["splits"][name]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
