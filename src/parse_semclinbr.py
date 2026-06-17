#!/usr/bin/env python3
"""
Parser SemClinBr XML -> JSONL canonico (versao minima).

Le SemClinBr-xml-public-v1/*.xml e gera um JSON por linha:

    {"doc_id": "9234", "text": "...",
     "entities": [{"id": "15264", "type": "Patient or Disabled Group",
                   "start": 3, "end": 11, "text": "PACIENTE"}],
     "relations": [{"e1_id": "15270", "e2_id": "15271", "type": "associated_with"}]}

Decisoes (fiel ao XML):
  - IDs como string.
  - `tag`/`reltype` copiados sem normalizacao (lowercase, split de "|" etc sao
    responsabilidade das etapas seguintes).
  - Quebras de linha colapsadas para 1 espaco para nao desalinhar os offsets
    das anotacoes (\\r\\n -> ' ' antes de \\r/\\n isolados, preservando o
    comprimento). Nenhuma outra normalizacao.
  - Inconsistencias (offset que nao bate com o texto da anotacao) sao contadas
    e reportadas no final, nunca abortam o parse (SemClinBr tem erros conhecidos).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.logger import get_logger  # noqa: E402

log = get_logger("parse_semclinbr")


def _natural(p: Path):
    try:
        return (int(p.stem), p.stem)
    except ValueError:
        return (10**9, p.stem)


def parse_one(xml_path: Path):
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    root = etree.parse(str(xml_path), parser=parser).getroot()
    if root is None or root.tag != "ANNOTATIONS":
        raise ValueError(f"{xml_path.name}: raiz invalida")

    text_el = root.find("TEXT")
    if text_el is None or text_el.text is None:
        raise ValueError(f"{xml_path.name}: <TEXT> ausente")
    raw = text_el.text
    text = raw.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

    entities, n_mismatch = [], 0
    tags_el = root.find("TAGS")
    if tags_el is not None:
        for ann in tags_el.findall("annotation"):
            try:
                start = int(ann.get("start"))
                end = int(ann.get("end"))
            except (TypeError, ValueError):
                continue
            surf = ann.get("text") or ""
            # checagem leve de alinhamento (apenas para contagem)
            if " ".join(text[start:end].split()) != " ".join(surf.split()):
                n_mismatch += 1
            entities.append({
                "id": str(ann.get("id")),
                "type": ann.get("tag") or "",
                "start": start,
                "end": end,
                "text": surf,
            })

    ent_ids = {e["id"] for e in entities}
    relations, n_orphan = [], 0
    rels_el = root.find("RELATIONS")
    if rels_el is not None:
        for rel in rels_el.findall("rel"):
            a, b = str(rel.get("annotation1")), str(rel.get("annotation2"))
            if a not in ent_ids or b not in ent_ids:
                n_orphan += 1
                continue
            relations.append({"e1_id": a, "e2_id": b,
                              "type": rel.get("reltype") or ""})

    rec = {"doc_id": xml_path.stem, "text": text,
           "entities": entities, "relations": relations}
    return rec, n_mismatch, n_orphan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", default="SemClinBr-xml-public-v1")
    ap.add_argument("--out", default="data/processed/dataset.jsonl")
    args = ap.parse_args()

    log.info("Iniciando parse | xml-dir=%s | out=%s", args.xml_dir, args.out)

    files = sorted(Path(args.xml_dir).glob("*.xml"), key=_natural)
    if not files:
        log.error("Nenhum .xml encontrado em %s", args.xml_dir)
        return 2
    log.info("%d arquivos XML encontrados", len(files))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_docs = n_rel = tot_mismatch = tot_orphan = n_skip = 0
    rel_by_type: dict[str, int] = {}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for p in files:
            try:
                rec, mm, orph = parse_one(p)
            except Exception as e:  # noqa: BLE001
                log.warning("Documento ignorado %s: %s", p.name, e)
                n_skip += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n")
            n_docs += 1
            n_rel += len(rec["relations"])
            tot_mismatch += mm
            tot_orphan += orph
            for r in rec["relations"]:
                rel_by_type[r["type"]] = rel_by_type.get(r["type"], 0) + 1

    log.info("Parse concluido: %d docs gravados em %s (%d ignorados)",
             n_docs, out, n_skip)
    log.info("Total de relacoes: %d", n_rel)
    if tot_mismatch:
        log.warning("Offsets divergentes (anotacao != texto): %d", tot_mismatch)
    if tot_orphan:
        log.warning("Relacoes orfas descartadas (entidade inexistente): %d", tot_orphan)
    for t, c in sorted(rel_by_type.items(), key=lambda kv: -kv[1]):
        log.info("  relacao %-18s %d", t, c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
