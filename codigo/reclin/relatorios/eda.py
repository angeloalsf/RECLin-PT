"""Tabelas da análise exploratória do corpus (as partições congeladas).

Recebem os dicionários de `analises.corpus` e `analises.max_gap` e devolvem
`tabelas.Tabela` de natureza `dados` (não dependem de execução nenhuma); a
origem registrada é o SHA-256 do MANIFEST das partições.

Origem no legado (só a finalidade): as tabelas de `make_tcc_eda.py`.
"""
from __future__ import annotations

from typing import Any, Mapping

from reclin.analises.corpus import rotulos_faixas
from reclin.relatorios.formatacao import Numero
from reclin.relatorios.tabelas import Tabela
from reclin.tarefa import LABELS

NATUREZA = "dados"


def _origem(manifest_sha256: str) -> list[dict[str, Any]]:
    return [{"execucao": "particoes congeladas", "estrategia": "dados", "natureza": NATUREZA,
             "manifest_sha256": manifest_sha256}]


def corpus(estat: Mapping[str, Any], manifest_sha256: str) -> Tabela:
    """Documentos, entidades, relações anotadas e pares candidatos por partição."""
    linhas = []
    for nome in [p for p in estat if not p.startswith("_")] + ["_total"]:
        s = estat[nome]
        rotulo = "Total" if nome == "_total" else nome.upper()
        k = f"corpus.{nome.strip('_')}"
        rel, cand = s["relacoes_por_tipo"], s["candidatos_por_rotulo"]
        linhas.append([rotulo, Numero(s["n_documentos"], inteiro=True, chave=f"{k}.documentos"),
                       Numero(s["n_entidades"], inteiro=True, chave=f"{k}.entidades"),
                       Numero(rel.get("negation_of", 0), inteiro=True, chave=f"{k}.relacoes_negation_of"),
                       Numero(rel.get("associated_with", 0), inteiro=True, chave=f"{k}.relacoes_associated_with"),
                       Numero(s["n_candidatos"], inteiro=True, chave=f"{k}.candidatos"),
                       Numero(cand["no_relation"], inteiro=True, chave=f"{k}.candidatos_no_relation")])
    return Tabela("eda_corpus", "Corpus por partição", ["Partição", "Documentos", "Entidades", "negation_of",
                                                        "associated_with", "Pares candidatos", "no_relation"],
                  linhas, NATUREZA, origens=_origem(manifest_sha256),
                  notas=["Pares candidatos com max_gap = 25 caracteres (tarefa.MAX_GAP)."])


def cobertura(estat: Mapping[str, Any], manifest_sha256: str) -> Tabela:
    """Fração das relações anotadas que viram pares candidatos."""
    linhas = []
    for nome in [p for p in estat if not p.startswith("_")]:
        c = estat[nome]["cobertura_relacoes"]
        linha = [nome.upper()]
        for tipo in ("negation_of", "associated_with"):
            item = c.get(tipo)
            linha.append(Numero(item["fracao"], chave=f"cobertura.{nome}.{tipo}") if item and item["fracao"] is not None
                         else "—")
        linhas.append(linha)
    return Tabela("eda_cobertura", "Relações anotadas cobertas pelos pares candidatos",
                  ["Partição", "negation_of", "associated_with"], linhas, NATUREZA, origens=_origem(manifest_sha256))


def distancias(estat: Mapping[str, Any], manifest_sha256: str, particao: str = "train") -> Tabela:
    """Relações anotadas por faixa de distância (caracteres entre as entidades)."""
    gaps = estat[particao]["gap_relacoes"]
    linhas = []
    for f in rotulos_faixas():
        linhas.append([f] + [Numero(gaps[t]["faixas"][f], inteiro=True, chave=f"gap.{particao}.{t}.{f}")
                             if t in gaps else "—" for t in ("negation_of", "associated_with")])
    return Tabela(f"eda_distancias_{particao}", f"Distância entre as entidades das relações ({particao.upper()})",
                  ["Faixa (caracteres)", "negation_of", "associated_with"], linhas, NATUREZA,
                  origens=_origem(manifest_sha256))


def entidades(estat: Mapping[str, Any], manifest_sha256: str, particao: str = "train", top: int = 10) -> Tabela:
    tipos = list(estat[particao]["entidades_por_tipo"].items())[:top]
    linhas = [[t, Numero(n, inteiro=True, chave=f"entidades.{particao}.{t}")] for t, n in tipos]
    return Tabela(f"eda_entidades_{particao}", f"Tipos de entidade mais frequentes ({particao.upper()})",
                  ["Tipo", "Entidades"], linhas, NATUREZA, origens=_origem(manifest_sha256))


def max_gap(sens: Mapping[str, Any], manifest_sha256: str, rotulo: str) -> Tabela:
    """Sensibilidade ao limite de distância (`analises.max_gap`)."""
    linhas = []
    for linha in sens["linhas"]:
        g = linha["max_gap"]
        cob = linha["cobertura"]
        linhas.append([f"{g}{' (tarefa)' if linha['da_tarefa'] else ''}",
                       Numero(linha["n_candidatos"], inteiro=True, chave=f"max_gap.{rotulo}.{g}.candidatos"),
                       Numero(cob["negation_of"]["fracao"], chave=f"max_gap.{rotulo}.{g}.cobertura_negation_of")
                       if "negation_of" in cob else "—",
                       Numero(cob["associated_with"]["fracao"], chave=f"max_gap.{rotulo}.{g}.cobertura_associated_with")
                       if "associated_with" in cob else "—",
                       Numero(linha["no_relation_por_relacao"], casas=2,
                              chave=f"max_gap.{rotulo}.{g}.no_relation_por_relacao")
                       if linha["no_relation_por_relacao"] is not None else "—"])
    return Tabela(f"eda_max_gap_{rotulo}", f"Sensibilidade ao max_gap ({rotulo})",
                  ["max_gap", "Pares candidatos", "Cobertura negation_of", "Cobertura associated_with",
                   "no_relation por relação"], linhas, NATUREZA, origens=_origem(manifest_sha256))


def candidatos_por_rotulo(estat: Mapping[str, Any], manifest_sha256: str) -> Tabela:
    linhas = []
    for nome in [p for p in estat if not p.startswith("_")]:
        c = estat[nome]["candidatos_por_rotulo"]
        total = sum(c.values())
        linhas.append([nome.upper()] + [Numero(c[r] / total if total else 0.0, chave=f"proporcao.{nome}.{r}")
                                        for r in LABELS])
    return Tabela("eda_proporcao_rotulos", "Proporção de cada rótulo entre os pares candidatos",
                  ["Partição"] + list(LABELS), linhas, NATUREZA, origens=_origem(manifest_sha256))
