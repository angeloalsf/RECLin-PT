"""Execuções de validação técnica: marca, nome e recusa.

A rodada de validação técnica (passo 1V do protocolo da fase) roda o pipeline
com poucas épocas, só até o DEV, para verificar que formatos, sidecars,
vínculos e recusas funcionam com o modelo real. Os números dela não avaliam
desempenho e nunca entram na fase científica.

Uma execução de validação tem, no `config.json` (em `config`), o bloco

    "validacao": {"epocas": N, "epocas_protocolo": P, "rotulo": "validação técnica, N época(s)"}

em que `epocas` é o número de épocas efetivamente usado (o parâmetro dedicado
`--validacao-epocas`) e `epocas_protocolo` o do protocolo, que não muda. O
nome padrão da execução termina em `_validacao_<N>ep`, para não colidir com
os nomes da fase científica (`*_biobertpt_seed42`).

Sem o parâmetro, o bloco não existe e nada muda no `config.json` de uma
execução. Com ele, só as épocas e a marca mudam.

`recusar_execucao` e `recusar_sidecar` são o que as operações da fase
científica chamam antes de agir: a avaliação do TEST, a aplicação de um limiar
ao TEST, a checagem final de proveniência e as comparações recusam qualquer
execução marcada. Uma composição ou calibração feita sobre execuções de
validação também é marcada (`herdar`), e misturar execuções de validação com
execuções da fase científica é recusado (`conferir_mesma_natureza`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from reclin.util.io import ler_json

CHAVE = "validacao"
CONFIG = "config.json"


class ExecucaoDeValidacao(ValueError):
    """A operação não aceita execuções de validação técnica."""


def rotulo(epocas: int) -> str:
    return f"validação técnica, {epocas} época" + ("" if epocas == 1 else "s")


def bloco(epocas: int, epocas_protocolo: int) -> dict[str, Any]:
    """A marca gravada no `config.json` de uma execução de validação."""
    if not isinstance(epocas, int) or epocas <= 0:
        raise ValueError(f"o número de épocas da validação precisa ser positivo: {epocas!r}")
    return {"epocas": epocas, "epocas_protocolo": int(epocas_protocolo), "rotulo": rotulo(epocas)}


def nome(prefixo: str, epocas: int) -> str:
    """`<prefixo>_validacao_<N>ep` (por exemplo, `baseline_validacao_1ep`)."""
    return f"{prefixo}_validacao_{epocas}ep"


def da_config(registro: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A marca de um registro (o `config` de um `config.json`), ou None."""
    if not registro:
        return None
    marca = registro.get(CHAVE)
    return dict(marca) if marca else None


def da_execucao(pasta: str | Path) -> dict[str, Any] | None:
    """A marca de uma pasta de execução, ou None (sem marca ou sem config.json)."""
    caminho = Path(pasta) / CONFIG
    if not caminho.is_file():
        return None
    return da_config(ler_json(caminho).get("config"))


def recusar_execucao(pasta: str | Path, operacao: str) -> None:
    """Falha se a execução for de validação técnica."""
    marca = da_execucao(pasta)
    if marca:
        raise ExecucaoDeValidacao(
            f"{Path(pasta).name} é uma execução de validação técnica ({marca.get('rotulo')}) e não pode "
            f"ser usada em: {operacao}. Execuções de validação servem só para verificar o pipeline.")


def recusar_sidecar(caminho: str | Path, operacao: str) -> None:
    """Falha se o sidecar pertencer a uma execução de validação técnica (a
    pasta que o contém tem um `config.json` marcado). Sidecars soltos, como os
    do legado, não têm marca e passam."""
    recusar_execucao(Path(caminho).parent, operacao)


def herdar(marcas: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any] | None:
    """A marca de uma operação feita sobre outras execuções (composição,
    calibração): None se nenhuma for de validação; recusa misturar."""
    conferir_mesma_natureza(marcas)
    presentes = {n: dict(m) for n, m in marcas.items() if m}
    if not presentes:
        return None
    return {"origens": presentes, "rotulo": "derivada de execuções de validação técnica"}


def conferir_mesma_natureza(marcas: Mapping[str, Mapping[str, Any] | None]) -> None:
    """Falha se umas execuções forem de validação e outras não."""
    com = sorted(n for n, m in marcas.items() if m)
    sem = sorted(n for n, m in marcas.items() if not m)
    if com and sem:
        raise ExecucaoDeValidacao("não é permitido misturar execuções de validação técnica ("
                                  + ", ".join(com) + ") com execuções da fase científica ("
                                  + ", ".join(sem) + ")")
