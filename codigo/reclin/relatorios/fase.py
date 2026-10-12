"""Os sistemas e as comparações da fase atual, como no protocolo pré-registrado
(`docs/protocolo/fase-biobertpt-seed42.md`, seções 1, 3 e 7).

Os nomes das execuções seguem as convenções das estratégias (conferidas pelos
testes contra as funções `nome_execucao` de cada uma); o encoder e a semente
são os da configuração padrão (`Config`), que são os da fase.

Nada aqui escolhe um vencedor: a regra do modelo final fica para uma etapa
posterior (C13), e a leitura das comparações é a do protocolo (IC95% do
bootstrap acima de 0, abaixo de 0 ou contendo o 0).
"""
from __future__ import annotations

from dataclasses import dataclass

from reclin.config import Config

ROTULO_FASE = "validação inicial, seed 42"
ROTULO_NATUREZA = {
    "cientifica": ROTULO_FASE,
    "validacao": "validação técnica — não avalia desempenho",
    "fora_da_fase": "execução fora da fase (técnica) — não é resultado do TCC",
}

# só o F1 de negation_of é comparável com o baseline no restrito e na Pair-Aware
COMPLETA, SO_NEG = "completa", "so_f1_negation_of"


@dataclass(frozen=True)
class Sistema:
    chave: str
    rotulo: str
    execucao: str
    comparabilidade: str = COMPLETA
    nota_dev: str | None = None


def sistemas(encoder: str | None = None, seed: int | None = None) -> tuple[Sistema, ...]:
    c = Config()
    enc, s = encoder or c.encoder, c.seed if seed is None else seed
    sufixo = f"{enc}_seed{s}"
    otimista = "otimista: o limiar foi escolhido neste DEV"
    return (
        Sistema("baseline", "Baseline", f"baseline_{sufixo}"),
        Sistema("baseline_calibrado", "Baseline calibrado", f"baseline_calibrado_{sufixo}", nota_dev=otimista),
        Sistema("filtro", "Filtro léxico", f"filtro_{sufixo}"),
        Sistema("restrito", "Restrito", f"restrito_{sufixo}", SO_NEG),
        Sistema("pair_aware", "Pair-Aware", f"pairaware_{sufixo}", SO_NEG),
        Sistema("two_stage", "Two-Stage", f"twostage_{sufixo}", nota_dev=otimista),
    )


def treinadas(encoder: str | None = None, seed: int | None = None) -> tuple[tuple[str, str], ...]:
    """(rótulo, execução) das execuções com treino (curvas por época)."""
    c = Config()
    sufixo = f"{encoder or c.encoder}_seed{c.seed if seed is None else seed}"
    return (("Baseline", f"baseline_{sufixo}"), ("Restrito", f"restrito_{sufixo}"),
            ("Pair-Aware", f"pairaware_{sufixo}"), ("Two-Stage, estágio 1", f"twostage_e1_{sufixo}"),
            ("Two-Stage, estágio 2", f"twostage_e2_{sufixo}"))


def sistemas_validacao() -> tuple[Sistema, ...]:
    """Os sistemas da rodada de validação técnica (passo 1V, seção 5 do
    protocolo): só até o DEV, com os nomes registrados no protocolo."""
    return (Sistema("baseline", "Baseline", "baseline_validacao_1ep"),
            Sistema("baseline_calibrado", "Baseline calibrado", "baseline_calibrado_validacao_1ep"),
            Sistema("two_stage", "Two-Stage", "twostage_validacao"))


def treinadas_validacao() -> tuple[tuple[str, str], ...]:
    return (("Baseline", "baseline_validacao_1ep"), ("Two-Stage, estágio 1", "twostage_e1_validacao_1ep"),
            ("Two-Stage, estágio 2", "twostage_e2_validacao_2ep"))


PRINCIPAL, EXPLORATORIA = "principal", "exploratória"

# (A, B, tipo): a diferença é sempre A − B (seção 7 do protocolo)
COMPARACOES: tuple[tuple[str, str, str], ...] = (
    ("two_stage", "baseline", PRINCIPAL),
    ("two_stage", "baseline_calibrado", PRINCIPAL),
    ("baseline_calibrado", "baseline", EXPLORATORIA),
    ("two_stage", "filtro", EXPLORATORIA),
    ("two_stage", "restrito", EXPLORATORIA),
    ("two_stage", "pair_aware", EXPLORATORIA),
)


def leitura(ci_baixo: float, ci_alto: float) -> str:
    """A redação do protocolo para o IC95% de A − B."""
    if ci_baixo > 0:
        return "diferença positiva"
    if ci_alto < 0:
        return "diferença negativa"
    return "diferença não detectada"
