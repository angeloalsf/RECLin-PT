"""O relatório da fase em Markdown: escopo, inventário, tabelas, pendências e
proveniência, num só arquivo legível.

Recebe o que as outras partes já calcularam (inventário e tabelas); não lê
execução nenhuma por conta própria. As tabelas entram com o status
(`completa`, `parcial`, `pendente`) e as pendências explícitas, para que um
resultado ausente nunca pareça um resultado.

Origem no legado (só a finalidade): `make_fase2_report.py` e o relatório
Markdown da calibração.
"""
from __future__ import annotations

from typing import Any, Sequence

from reclin.execucao import inventario as inv
from reclin.relatorios import fase
from reclin.relatorios.tabelas import Tabela
from reclin.util import caminhos


def _inventario(inventario: inv.Inventario) -> list[str]:
    linhas = ["## Inventário das execuções", "",
              f"Raiz: `{caminhos.relativo_ao_codigo(inventario.raiz)}`. Partições: "
              + ("congeladas (MANIFEST idêntico)." if inventario.particoes_congeladas
                 else "**diferentes das congeladas**."), ""]
    if not inventario.execucoes:
        linhas += ["Nenhuma execução do código novo encontrada.", ""]
    else:
        linhas += ["| Execução | Estratégia | Natureza | Partições utilizáveis | Situação |",
                   "| --- | --- | --- | --- | --- |"]
        for nome, ex in sorted(inventario.execucoes.items()):
            situacao = "ok" if ex.valida else "**problemas**: " + "; ".join(ex.problemas)
            if ex.avisos:
                situacao += " (avisos: " + "; ".join(ex.avisos) + ")"
            if ex.natureza == "fora_da_fase":
                situacao += " — fora da fase: " + "; ".join(ex.motivos_fora_da_fase)
            particoes = ", ".join(p.upper() for p in sorted(ex.sidecars) if ex.tem(p)) or "—"
            linhas.append(f"| `{nome}` | {ex.estrategia} | {ex.natureza} | {particoes} | {situacao} |")
        linhas.append("")
    if inventario.recusados:
        linhas += ["Itens recusados na raiz (não são lidos):", ""]
        linhas += [f"- `{r['caminho']}`: {r['motivo']}" for r in inventario.recusados]
        linhas.append("")
    return linhas


def _tabela(t: Tabela) -> list[str]:
    linhas = [f"### {t.titulo}", "", f"Status: **{t.status}**.", ""]
    if t.linhas:
        linhas += [t.markdown()]
    for nota in t.notas:
        linhas.append(f"- {nota}")
    for p in t.pendencias:
        linhas.append(f"- Pendente: {p}")
    if t.origens:
        linhas.append("- Origem: " + ", ".join(f"`{o['execucao']}`" for o in t.origens))
    linhas.append("")
    return linhas


def relatorio(inventario: inv.Inventario, tabelas: Sequence[Tabela], *, natureza: str,
              titulo: str | None = None, extras: Sequence[str] = ()) -> str:
    """O texto Markdown do relatório."""
    rotulo = fase.ROTULO_NATUREZA.get(natureza, natureza)
    partes = [f"# {titulo or 'Relatório da fase'} — {rotulo}", "",
              "Gerado por `codigo/scripts/gerar_tcc.py relatorio` a partir das execuções do código novo "
              "verificadas pelo inventário. Nenhum resultado do legado é lido. Valores ausentes aparecem "
              "como *pendente*, nunca preenchidos.", "",
              "## Escopo", "",
              f"- Fase: {fase.ROTULO_FASE} (BioBERTpt, `pucpr/biobertpt-all`).",
              "- Protocolo: `codigo/docs/protocolo/fase-biobertpt-seed42.md`.",
              f"- Natureza das execuções deste relatório: **{natureza}** ({rotulo}).", ""]
    partes += _inventario(inventario)
    partes += ["## Tabelas", ""]
    for t in tabelas:
        partes += _tabela(t)
    pendentes = [(t.nome, p) for t in tabelas for p in t.pendencias]
    partes += ["## Pendências", ""]
    partes += [f"- `{nome}`: {p}" for nome, p in pendentes] or ["- Nenhuma."]
    partes.append("")
    partes += list(extras)
    return "\n".join(partes).rstrip("\n") + "\n"


def resumo_status(tabelas: Sequence[Tabela]) -> dict[str, Any]:
    return {t.nome: t.status for t in tabelas}
