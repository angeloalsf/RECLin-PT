"""Conferência do texto do TCC contra o código novo e o registro de valores.

Duas verificações sobre os `.tex` de uma pasta do TCC:

1. **Caminhos antigos** — o texto não pode citar caminhos do repositório
   original (`CAMINHOS_ANTIGOS`, a lista da seção 5.3 do mapa de migração).
   Cada ocorrência é reportada com arquivo e linha.
2. **Afirmações numéricas** — cada afirmação de uma lista (JSON) diz em que
   arquivo um trecho deve aparecer com um valor do registro gerado pelas
   tabelas (`valores.json`): `{"id", "arquivo", "trecho", "chave"}`, em que
   `trecho` contém `{valor}`, substituído pelo texto ABNT do valor. A
   afirmação passa se o trecho, com o valor atual, está no arquivo.

A lista de afirmações é escrita junto com o texto (etapa 9): esta etapa só
fornece o mecanismo. Nenhuma afirmação, valor ou trecho do texto antigo é
incorporado aqui. Sem a pasta do TCC, a conferência fica pendente.

Origem no legado (só a finalidade): `check_tcc_numbers.py` e a sua lista
`GHOST_PATHS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# Caminhos do repositório original citados no texto do TCC (mapa de migração,
# seção 5.3). São padrões de texto; nenhum arquivo deles é lido.
CAMINHOS_ANTIGOS: tuple[str, ...] = (
    "scripts/make_tcc_eda.py", "src/relation_extraction.py", "results/baseline_", "results/significance_",
    "scripts/make_tcc_artifacts.py", "src/candidates.py", "src/negation_lexicon.py",
    "scripts/make_rule_baseline.py", "scripts/make_tcc_curves.py", "scripts/_artifacts.py",
    "src/make_splits.py", "data/splits/MANIFEST.json",
)

CAMPOS_AFIRMACAO = ("id", "arquivo", "trecho", "chave")


@dataclass
class Resultado:
    pasta_tcc: str
    status: str                       # "ok", "divergente" ou "pendente"
    caminhos_antigos: list[dict[str, Any]] = field(default_factory=list)
    afirmacoes: list[dict[str, Any]] = field(default_factory=list)
    pendencias: list[str] = field(default_factory=list)

    def como_dict(self) -> dict[str, Any]:
        return {"pasta_tcc": self.pasta_tcc, "status": self.status, "caminhos_antigos": self.caminhos_antigos,
                "afirmacoes": self.afirmacoes, "pendencias": self.pendencias}


def arquivos_tex(pasta: str | Path) -> list[Path]:
    return sorted(p for p in Path(pasta).rglob("*.tex") if p.is_file())


def caminhos_antigos(pasta: str | Path, padroes: Iterable[str] = CAMINHOS_ANTIGOS) -> list[dict[str, Any]]:
    pasta = Path(pasta)
    achados = []
    for arquivo in arquivos_tex(pasta):
        for n, linha in enumerate(arquivo.read_text(encoding="utf-8").splitlines(), start=1):
            if linha.lstrip().startswith("%"):
                continue
            for padrao in padroes:
                if padrao in linha:
                    achados.append({"arquivo": arquivo.relative_to(pasta).as_posix(), "linha": n,
                                    "caminho": padrao})
    return achados


def conferir_afirmacoes(pasta: str | Path, afirmacoes: Iterable[Mapping[str, Any]],
                        valores: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    pasta = Path(pasta)
    saida = []
    for a in afirmacoes:
        faltando = [c for c in CAMPOS_AFIRMACAO if c not in a]
        if faltando:
            saida.append({"id": a.get("id"), "ok": False, "motivo": f"afirmação sem os campos {faltando}"})
            continue
        if "{valor}" not in a["trecho"]:
            saida.append({"id": a["id"], "ok": False, "motivo": "o trecho não tem {valor}"})
            continue
        if a["chave"] not in valores:
            saida.append({"id": a["id"], "ok": False,
                          "motivo": f"a chave {a['chave']} não está no registro de valores (resultado pendente?)"})
            continue
        arquivo = pasta / a["arquivo"]
        if not arquivo.is_file():
            saida.append({"id": a["id"], "ok": False, "motivo": f"arquivo {a['arquivo']} não existe"})
            continue
        esperado = a["trecho"].replace("{valor}", valores[a["chave"]]["texto"])
        texto = " ".join(arquivo.read_text(encoding="utf-8").split())
        ok = " ".join(esperado.split()) in texto
        saida.append({"id": a["id"], "ok": ok, "esperado": esperado,
                      **({} if ok else {"motivo": "trecho com o valor atual não encontrado"})})
    return saida


def conferir(pasta_tcc: str | Path, *, afirmacoes: Iterable[Mapping[str, Any]] | None = None,
             valores: Mapping[str, Mapping[str, Any]] | None = None) -> Resultado:
    """Caminhos antigos e afirmações. Pendente sem a pasta, sem `.tex`, sem a
    lista de afirmações ou sem o registro de valores; divergente se algum
    caminho antigo ou afirmação falhar."""
    pasta = Path(pasta_tcc)
    r = Resultado(pasta_tcc=pasta.as_posix(), status="ok")
    if not pasta.is_dir() or not arquivos_tex(pasta):
        r.status = "pendente"
        r.pendencias.append(f"sem os arquivos .tex do TCC em {pasta.as_posix()}")
        return r
    r.caminhos_antigos = caminhos_antigos(pasta)
    if afirmacoes is None:
        r.pendencias.append("sem a lista de afirmações (escrita junto com o texto, na etapa 9)")
    elif valores is None:
        r.pendencias.append("sem o registro de valores (gerar_tcc.py tabelas)")
    else:
        r.afirmacoes = conferir_afirmacoes(pasta, afirmacoes, valores)
    if r.caminhos_antigos or any(not a["ok"] for a in r.afirmacoes):
        r.status = "divergente"
    elif r.pendencias:
        r.status = "pendente"
    return r
