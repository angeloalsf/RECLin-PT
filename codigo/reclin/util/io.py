"""Leitura e gravação de JSON e JSONL, e hashes SHA-256.

Todo arquivo de texto gravado pelo pacote usa UTF-8, sem escapar acentos, e
quebra de linha LF em qualquer sistema operacional: os hashes das partições e
das execuções dependem dos bytes exatos. A gravação é atômica (arquivo
temporário + `os.replace`), para que uma sessão interrompida não deixe um
arquivo pela metade com o nome do definitivo.

Origem no legado: `read_jsonl`, `write_jsonl` e `sha256_of`, repetidos em cinco
arquivos (`relation_extraction`, `make_splits`, `_artifacts`, ...).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def ler_jsonl(caminho: str | Path) -> Iterator[dict[str, Any]]:
    """Itera sobre os registros de um JSONL, ignorando linhas em branco."""
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                yield json.loads(linha)


def linha_jsonl(registro: Any) -> str:
    """Forma canônica de um registro: chaves ordenadas, sem espaços, sem escapar
    acentos. É a forma das partições congeladas."""
    return json.dumps(registro, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def gravar_jsonl(caminho: str | Path, registros: Iterable[Any]) -> int:
    """Grava um registro por linha na forma canônica e devolve quantos gravou."""
    n = 0

    def linhas():
        nonlocal n
        for registro in registros:
            n += 1
            yield linha_jsonl(registro) + "\n"

    _gravar_atomico(caminho, linhas())
    return n


def ler_json(caminho: str | Path) -> Any:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def gravar_json(caminho: str | Path, obj: Any) -> None:
    """Grava JSON legível e estável: indentação 2, chaves ordenadas, LF final.

    Gravar duas vezes o mesmo objeto produz arquivos idênticos byte a byte, de
    modo que um diff no git só aparece quando o conteúdo muda."""
    texto = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _gravar_atomico(caminho, [texto])


def gravar_texto(caminho: str | Path, texto: str) -> None:
    """Grava `texto` exatamente como está (UTF-8, LF, sem acrescentar quebra
    final). Para formatos cujo conteúdo byte a byte faz parte do contrato,
    como os relatórios de comparação herdados do legado."""
    _gravar_atomico(caminho, [texto])


def sha256_arquivo(caminho: str | Path) -> str:
    """SHA-256 do conteúdo do arquivo, lido em blocos de 1 MiB."""
    digest = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 de um objeto serializado em JSON compacto, na ordem em que está.

    É o hash usado para identificar sequências (candidatos, `y_true`): a ordem
    dos elementos faz parte do que é identificado, então as chaves de
    dicionários NÃO são ordenadas aqui."""
    texto = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _gravar_atomico(caminho: str | Path, partes: Iterable[str]) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    try:
        with open(temporario, "w", encoding="utf-8", newline="\n") as f:
            for parte in partes:
                f.write(parte)
        os.replace(temporario, caminho)
    finally:
        if temporario.exists():
            temporario.unlink()
