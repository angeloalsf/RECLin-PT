"""As 26 comparações de significância do TCC e a regra de semente do bootstrap.

O protocolo foi fixado no legado antes de os resultados serem lidos e é
aplicado igual às execuções novas (etapa 8). As execuções são identificadas
pelo nome (`baseline_biobertpt_seed42`, `filtro_bertimbau_seed43`,
`regra_pura`, ...); este módulo não conhece estratégias, só nomes.

| Grupo | Comparações | Semente do bootstrap |
| --- | --- | --- |
| BioBERTpt × BERTimbau (fase 1) | 2, uma por semente (42, 43) | a da execução |
| Filtro × baseline (fase 2) | 16 (4 filtros × 4 baselines) | a do lado A |
| Regra pura × baseline (fase 2) | 4 | a do lado B (a regra não tem semente) |
| Filtro × regra pura (fase 2) | 4 | a do lado A |

Regra de semente (`semente_bootstrap`): a da execução do lado A; quando A não
tem semente (a regra pura), a do lado B. Assim a mesma comparação sempre
reproduz o mesmo intervalo.

Nome do relatório: o do legado. Na fase 2, `significance_<A>_vs_<B>.json`
(`nome_relatorio`); na fase 1, `significance_biobertpt_vs_bertimbau_seed<N>.json`.

Origem no legado: a regra `SIGNIF_RULE` do `Makefile` (fase 1) e
`SYSTEMS`, `BASELINES`, `EXTRA` e `jobs()` de
`scripts/run_fase2_significance.py` (fase 2), na mesma ordem.
"""
from __future__ import annotations

from typing import NamedTuple

ENCODERS_FASE1 = ("biobertpt", "bertimbau")
SEMENTES = (42, 43)


class Comparacao(NamedTuple):
    a: str                 # execução do lado A
    b: str                 # execução do lado B
    seed_a: int | None     # semente das execuções (None para a regra pura)
    seed_b: int | None
    arquivo: str           # nome do relatório

    @property
    def seed(self) -> int:
        return semente_bootstrap(self.seed_a, self.seed_b)


def semente_bootstrap(seed_a: int | None, seed_b: int | None) -> int:
    """A semente da execução A; se A não tem semente, a de B."""
    if seed_a is not None:
        return seed_a
    if seed_b is not None:
        return seed_b
    raise ValueError("nenhuma das duas execuções tem semente")


def nome_relatorio(a: str, b: str) -> str:
    return f"significance_{a}_vs_{b}.json"


def comparacoes_tcc() -> list[Comparacao]:
    """As 26 comparações, na ordem do legado: fase 1 e depois fase 2."""
    fase1 = [Comparacao(f"baseline_biobertpt_seed{s}", f"baseline_bertimbau_seed{s}", s, s,
                        f"significance_biobertpt_vs_bertimbau_seed{s}.json")
             for s in SEMENTES]
    sistemas = [(f"filtro_{e}_seed{s}", s) for s in SEMENTES for e in ENCODERS_FASE1]
    sistemas.append(("regra_pura", None))
    baselines = [(f"baseline_{e}_seed{s}", s) for s in SEMENTES for e in ENCODERS_FASE1]
    fase2 = [Comparacao(a, b, sa, sb, nome_relatorio(a, b)) for a, sa in sistemas for b, sb in baselines]
    fase2 += [Comparacao(a, "regra_pura", sa, None, nome_relatorio(a, "regra_pura"))
              for a, sa in sistemas[:4]]
    return fase1 + fase2
