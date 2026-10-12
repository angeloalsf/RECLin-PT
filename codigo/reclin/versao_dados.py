"""Versões dos dados (partições congeladas) e a guarda contra misturá-las.

Duas versões convivem no projeto, as duas congeladas e versionadas:

| Versão | Pasta | Origem | Uso |
| --- | --- | --- | --- |
| `v1-conversao-legado` | `dados/particoes/` | conversão do legado (`parse_semclinbr.py`) | regressão técnica e histórico; **não** é entrada dos experimentos novos |
| `v2-texto-fonte` | `dados/particoes-v2/` | `reclin/corpus` (regra C1 do D9) a partir dos XML | **a fase científica atual** (`VERSAO_DA_FASE`) |

As duas têm os mesmos documentos por partição, as mesmas entidades e
relações, os mesmos candidatos e o mesmo `y_true`; só o `text` de 421
documentos difere (o MANIFEST da v2 registra a comparação). Por isso o
conjunto de referência (`tarefa.conjunto_referencia`) das duas versões difere
**só** no `particao_sha256` — e é por ele que a versão de cada artefato é
reconhecida.

IDENTIFICAÇÃO
-------------
A versão de uma partição é a do registro `REGISTRO` cujo SHA-256 coincide
com o do arquivo; a de uma pasta, pelo SHA-256 do MANIFEST (e, na v2, pela
chave `versao` do próprio MANIFEST, que precisa concordar). Uma partição que
não está no registro (um subconjunto usado em testes, por exemplo) não tem
versão (`None`): não é "congelada" e nunca é aceita como científica.

GUARDA
------
`versoes_do_registro` reúne as versões de tudo o que uma execução gravou
(`config.particoes` do treino e o `particao_sha256` de cada conjunto de
referência); `conferir_uma_versao` falha se houver mais de uma versão
registrada entre elas — por exemplo, um modelo treinado na v1 avaliado no TEST
da v2, ou uma execução derivada (filtro, calibrado, composição) sobre uma
base de outra versão. As operações que acrescentam artefatos a uma execução
existente chamam `exigir_mesma_versao` antes de agir; o inventário marca como
problema a execução que mistura versões e como fora da fase a que não é da
`VERSAO_DA_FASE`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from reclin.util import caminhos

V1 = "v1-conversao-legado"
V2 = "v2-texto-fonte"
VERSOES = (V1, V2)
VERSAO_DA_FASE = V2

PASTAS: dict[str, Path] = {V1: caminhos.PARTICOES_V1, V2: caminhos.PARTICOES_V2}

# SHA-256 congelados de cada versão (MANIFEST e partições). Mudar um valor
# aqui é declarar uma nova versão dos dados.
REGISTRO: dict[str, dict[str, Any]] = {
    V1: {
        "manifest_sha256": "a73f291a18a1b72e1e84e1841b6e0b46a30dd2b4ddc194ee4f8ae9f0896f8ae1",
        "particoes": {
            "train": "36014b2adc8308bce94aeecf066c38f69e1f53dda4517fa424527554b31a437a",
            "dev": "a9ea327684165c5053f8f3db5468a27f71c1d9e30d2859e343b6d275839c8aeb",
            "test": "86304d8d507c6707695e2b86c3533b763a96247f06326e5980b4567385231e62",
        },
        "descricao": "partições do legado (conversão com o texto normalizado e os offsets da fonte; "
                     "desalinhamento documentado no D9)",
    },
    V2: {
        "manifest_sha256": "18264860a3d8fff7bcc335b8f5703a55d60e08893a09be2afa20b1ea446d3b70",
        "particoes": {
            "train": "b8aaafce423afb9cda49fa5d116c14b14d6b9c2d09a0521bff4ef8733305d034",
            "dev": "19feb024ca9d60f0016491b2e9bc35ab7de9b5cc384e0d22ba34b21a85a9f99c",
            "test": "47d40d0b06f23714a8f332e3f8455bff0471bb68fad7b4a835b2aa72812e9666",
        },
        "descricao": "partições reconstruídas dos XML com a regra C1 (reclin/corpus); mesmos documentos, "
                     "entidades, relações, candidatos e y_true da v1; texto alinhado aos offsets",
    },
}

_POR_PARTICAO = {sha: versao for versao, r in REGISTRO.items() for sha in r["particoes"].values()}
_POR_MANIFESTO = {r["manifest_sha256"]: versao for versao, r in REGISTRO.items()}


class ErroVersaoDados(ValueError):
    """Artefatos de versões diferentes dos dados combinados numa operação."""


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def versao_da_particao(sha256: str | None) -> str | None:
    """A versão registrada de um arquivo de partição pelo seu SHA-256."""
    return _POR_PARTICAO.get(sha256) if sha256 else None


def versao_da_pasta(pasta: str | Path) -> str | None:
    """A versão das partições da pasta, pelo SHA-256 do MANIFEST (None se não
    for uma versão registrada). Uma chave `versao` no MANIFEST que discorde do
    registro é erro."""
    manifesto = Path(pasta) / "MANIFEST.json"
    if not manifesto.is_file():
        return None
    versao = _POR_MANIFESTO.get(_sha256(manifesto))
    if versao is not None:
        declarada = json.loads(manifesto.read_text(encoding="utf-8")).get("versao")
        if declarada is not None and declarada != versao:
            raise ErroVersaoDados(f"{manifesto}: declara a versão {declarada!r}, mas o SHA-256 é o da {versao!r}")
    return versao


def congelada(pasta: str | Path) -> bool:
    """A pasta tem o MANIFEST de uma versão registrada (byte a byte)."""
    return versao_da_pasta(pasta) is not None


def versoes_do_registro(conteudo_config: Mapping[str, Any]) -> dict[str, str | None]:
    """Versão de cada partição que uma execução gravou: as do treino
    (`config.particoes`) e as dos conjuntos de referência (`conjuntos`).
    Chave: `treino.<partição>` ou `conjunto.<partição>`."""
    saida: dict[str, str | None] = {}
    registro = conteudo_config.get("config") or {}
    for nome, sha in (registro.get("particoes") or {}).items():
        saida[f"treino.{nome}"] = versao_da_particao(sha)
    for nome, conjunto in (conteudo_config.get("conjuntos") or {}).items():
        saida[f"conjunto.{nome}"] = versao_da_particao((conjunto or {}).get("particao_sha256"))
    return saida


def registradas(versoes: Iterable[str | None]) -> set[str]:
    return {v for v in versoes if v is not None}


def conferir_uma_versao(versoes: Mapping[str, str | None]) -> list[str]:
    """Problemas se as partições usadas forem de mais de uma versão registrada."""
    distintas = registradas(versoes.values())
    if len(distintas) <= 1:
        return []
    detalhe = ", ".join(f"{k}={v}" for k, v in sorted(versoes.items()) if v is not None)
    return [f"artefatos de versões diferentes dos dados ({detalhe})"]


def versao_unica(versoes: Mapping[str, str | None]) -> str | None:
    """A versão comum (None se nenhuma partição registrada ou se houver mistura)."""
    distintas = registradas(versoes.values())
    return next(iter(distintas)) if len(distintas) == 1 else None


def exigir_mesma_versao(conteudo_config: Mapping[str, Any], pasta_particoes: str | Path, operacao: str) -> None:
    """Falha se a pasta de partições de uma operação for de outra versão
    registrada que a dos artefatos já gravados na execução."""
    gravadas = registradas(versoes_do_registro(conteudo_config).values())
    atual = versao_da_pasta(pasta_particoes)
    if len(gravadas) > 1:
        raise ErroVersaoDados(f"{operacao}: a execução já mistura versões dos dados ({sorted(gravadas)})")
    if atual is not None and gravadas and atual not in gravadas:
        raise ErroVersaoDados(f"{operacao}: a execução usa a versão {sorted(gravadas)[0]!r} dos dados e as "
                              f"partições informadas são da {atual!r}; versões não se combinam")


def descrever(pasta: str | Path) -> dict[str, Any]:
    """Identificação legível da versão de uma pasta de partições."""
    pasta = Path(pasta)
    versao = versao_da_pasta(pasta)
    manifesto = pasta / "MANIFEST.json"
    return {"versao": versao, "da_fase": versao == VERSAO_DA_FASE,
            "manifest_sha256": _sha256(manifesto) if manifesto.is_file() else None}
