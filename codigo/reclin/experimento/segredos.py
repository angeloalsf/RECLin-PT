"""Credenciais: separadas por finalidade, pedidas só para a operação que as usa.

Dois Secrets, com finalidades distintas (nunca um serve ao outro):

* `GITHUB_TOKEN` — só para **baixar o código** de um repositório privado;
  precisa apenas de leitura de conteúdo (token de granularidade fina, só este
  repositório, `Contents: Read-only`). Repositório público não precisa dele;
* `HF_TOKEN` — Hugging Face. Para **publicar** artefatos exige escrita (token
  de granularidade fina, só o repositório de artefatos, escrita). Para baixar
  pesos de um repositório público (`pucpr/biobertpt-all`) não é necessário.

`OPERACOES` diz quais Secrets cada operação pede. Treinar, retomar, sincronizar
com o Drive e consultar o DEV não pedem nenhum.

Garantias:

* o valor fica num `Credencial`, cujo `repr`/`str` nunca o mostram; ele não é
  serializável (JSON e pickle falham) e só sai por `.valor()`, no ponto de uso;
* `limpar` troca o valor por `***` em qualquer texto de erro ou log;
* `exigir_sem_segredo` recusa gravar um registro que contenha algo com a forma
  de token (`ghp_…`, `github_pat_…`, `hf_…`) ou o valor de uma credencial
  conhecida — é o que `proveniencia` chama antes de gravar.

Fora do Colab, a credencial pode vir da variável de ambiente de mesmo nome
(para uso local); dentro dele, vem dos Secrets.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from reclin.experimento import colab

GITHUB = "GITHUB_TOKEN"
HUGGING_FACE = "HF_TOKEN"

PADROES_TOKEN = (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), re.compile(r"\bgho_[A-Za-z0-9]{20,}"),
                 re.compile(r"\bghs_[A-Za-z0-9]{20,}"), re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
                 re.compile(r"\bhf_[A-Za-z0-9]{20,}"))


@dataclass(frozen=True)
class Segredo:
    nome: str
    finalidade: str
    permissao_minima: str


CATALOGO = {
    GITHUB: Segredo(GITHUB, "baixar o código de um repositório privado do GitHub",
                    "token de granularidade fina, só este repositório, Contents: Read-only"),
    HUGGING_FACE: Segredo(HUGGING_FACE, "publicar artefatos no repositório privado do Hugging Face",
                          "token de granularidade fina, só o repositório de artefatos, com escrita"),
}

# operação -> (segredos necessários, segredos opcionais)
OPERACOES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "baixar_codigo": ((), (GITHUB,)),            # opcional: só se o repositório for privado
    "baixar_pesos": ((), ()),                    # BioBERTpt é público; ver docs se usar repositório privado
    "treinar": ((), ()),
    "retomar": ((), ()),
    "sincronizar_drive": ((), ()),
    "consultar_dev": ((), ()),
    "planejar_publicacao": ((), ()),             # só lista arquivos; não fala com o Hub
    "publicar_hf": ((HUGGING_FACE,), ()),
}


class ErroSegredo(RuntimeError):
    """Credencial ausente, com permissão insuficiente ou vazando para um registro."""


class Credencial:
    """Guarda um valor secreto sem deixá-lo escapar por repr, str, JSON ou pickle."""

    __slots__ = ("nome", "origem", "_valor")

    def __init__(self, nome: str, valor: str, origem: str):
        object.__setattr__(self, "nome", nome)
        object.__setattr__(self, "origem", origem)
        object.__setattr__(self, "_valor", valor)

    def valor(self) -> str:
        return self._valor

    def __setattr__(self, nome: str, valor: Any) -> None:
        raise AttributeError("Credencial é imutável")

    def __repr__(self) -> str:
        return f"Credencial({self.nome}, origem={self.origem!r}, valor=***)"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("uma Credencial não pode ser serializada")


def obter(nome: str, *, obrigatorio: bool = False, ambiente: dict[str, str] | None = None) -> Credencial | None:
    """A credencial `nome` dos Secrets do Colab (ou, fora do Colab, da variável
    de ambiente). `obrigatorio=True` falha com instrução de configuração."""
    if nome not in CATALOGO:
        raise ErroSegredo(f"Secret {nome!r} desconhecido; os previstos são {sorted(CATALOGO)}")
    if colab.em_colab():
        valor, motivo = colab.ler_secret(nome)
    else:
        valor = (os.environ if ambiente is None else ambiente).get(nome) or None
        motivo = "variável de ambiente" if valor else f"variável de ambiente {nome} ausente (fora do Colab)"
    if valor:
        return Credencial(nome, valor, motivo)
    if obrigatorio:
        s = CATALOGO[nome]
        raise ErroSegredo(f"{motivo}. Necessário para: {s.finalidade}. Configure em Secrets do Colab "
                          f"({s.permissao_minima}) e ative o acesso do notebook.")
    return None


def para_operacao(operacao: str, *, ambiente: dict[str, str] | None = None) -> dict[str, Credencial | None]:
    """Só os Secrets que a operação pede: os necessários (falha se faltar) e os
    opcionais (None se faltar). Operações sem Secret devolvem `{}` sem consultar nada."""
    if operacao not in OPERACOES:
        raise ErroSegredo(f"operação {operacao!r} desconhecida; opções: {sorted(OPERACOES)}")
    necessarios, opcionais = OPERACOES[operacao]
    saida: dict[str, Credencial | None] = {}
    for nome in necessarios:
        saida[nome] = obter(nome, obrigatorio=True, ambiente=ambiente)
    for nome in opcionais:
        saida[nome] = obter(nome, ambiente=ambiente)
    return saida


def limpar(texto: str, *credenciais: Credencial | str | None) -> str:
    """`texto` sem os valores das credenciais e sem nada com forma de token."""
    for c in credenciais:
        valor = c.valor() if isinstance(c, Credencial) else c
        if valor:
            texto = texto.replace(valor, "***")
    for padrao in PADROES_TOKEN:
        texto = padrao.sub("***", texto)
    return texto


def contem_segredo(objeto: Any, *credenciais: Credencial | str | None) -> list[str]:
    """Motivos pelos quais `objeto` (serializável em JSON) parece conter um segredo."""
    texto = json.dumps(objeto, ensure_ascii=False, default=str)
    motivos = [f"forma de token ({p.pattern})" for p in PADROES_TOKEN if p.search(texto)]
    for c in credenciais:
        valor = c.valor() if isinstance(c, Credencial) else c
        if valor and valor in texto:
            motivos.append("valor de uma credencial conhecida")
    return motivos


def exigir_sem_segredo(objeto: Any, *credenciais: Credencial | str | None) -> None:
    motivos = contem_segredo(objeto, *credenciais)
    if motivos:
        raise ErroSegredo("o registro não foi gravado porque parece conter um segredo: " + "; ".join(motivos))


def verificar_escrita_hf(credencial: Credencial) -> dict[str, str | None]:
    """Consulta o Hugging Face (`whoami`) e confirma que o token pode escrever.
    Devolve `{usuario, papel}`; nunca o token. Falha se o papel for só leitura."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    try:
        info = HfApi().whoami(token=credencial.valor())
    except Exception as erro:  # noqa: BLE001
        raise ErroSegredo("o Hugging Face recusou o token: " + limpar(type(erro).__name__, credencial)) from None
    acesso = (info.get("auth") or {}).get("accessToken") or {}
    papel = acesso.get("role")
    if papel == "read":
        raise ErroSegredo("o token do Hugging Face é só de leitura; publicar exige escrita")
    return {"usuario": info.get("name"), "papel": papel}


def exemplos_de_configuracao() -> Iterable[str]:
    """Texto curto para o usuário configurar os Secrets (sem pedir valores)."""
    for s in CATALOGO.values():
        yield f"{s.nome}: {s.finalidade}. Permissão mínima: {s.permissao_minima}."
