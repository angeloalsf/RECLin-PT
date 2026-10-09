"""Configuração base comum às estratégias treinadas.

Uma única `dataclass` congelada com os hiperparâmetros que todas as estratégias
treinadas compartilham. Cada estratégia declara a sua configuração herdando
desta e mudando só o que difere — por exemplo, em `estrategias/restrito.py`:

    @dataclass(frozen=True)
    class ConfigRestrito(Config):
        epochs: int = 10

O que a estratégia não declara, herda. Por isso a paridade entre estratégias
fica visível no próprio código: lado a lado, duas configurações diferem só nos
campos que cada uma redeclara. Campos de uma estratégia específica (seleção de
candidatos, cabeça, léxico) não entram aqui.

Os defaults são os valores usados em todos os experimentos — os registrados no
`config` dos quatro baselines do legado (conferido em
`testes/equivalencia/test_config.py`). Os nomes dos hiperparâmetros também são
os registrados nas execuções e citados no TCC (`max_gap`, `ctx_chars`,
`max_length`, `class_weight`, ...).

Origem no legado: `build_arg_parser` de `src/relation_extraction.py` (cujos
defaults — `max_gap` 75, `max_length` 192, `batch_size` 32 — não eram os usados;
os valores reais vinham do Makefile, do `run.sh` e dos notebooks), `ENCODERS`
de `src/finetuning_restrito/train_restrito.py` e as constantes de
`restricted_space.py`. O `max_grad_norm` era fixo no laço de treino.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from reclin.tarefa import MAX_GAP

# Encoders comparados. Só o checkpoint de pré-treino muda entre eles: o
# caminho de código é o mesmo, o que garante a paridade por construção.
ENCODERS: dict[str, str] = {
    "biobertpt": "pucpr/biobertpt-all",                      # clínico
    "bertimbau": "neuralmind/bert-base-portuguese-cased",    # geral
}

CLASS_WEIGHTS = ("balanced", "none")


@dataclass(frozen=True)
class Config:
    """Hiperparâmetros comuns a todas as estratégias treinadas."""

    encoder: str = "biobertpt"     # chave de ENCODERS
    seed: int = 42

    # Candidatos e entrada do modelo
    max_gap: int = MAX_GAP         # caracteres entre os spans de um par
    ctx_chars: int = 128           # contexto de cada lado da janela marcada
    max_length: int = 128          # tokens após a tokenização

    # Otimização
    epochs: int = 3
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1      # fração dos passos com aquecimento linear
    max_grad_norm: float = 1.0     # recorte do gradiente
    class_weight: str = "balanced"  # "balanced" (inverso da frequência no TRAIN) | "none"

    def __post_init__(self) -> None:
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder {self.encoder!r} desconhecido; opções: {sorted(ENCODERS)}")
        if self.class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"class_weight {self.class_weight!r} inválido; opções: {CLASS_WEIGHTS}")
        positivos = ("max_gap", "ctx_chars", "max_length", "epochs", "batch_size", "lr")
        for campo in positivos:
            if getattr(self, campo) <= 0:
                raise ValueError(f"{campo} precisa ser positivo: {getattr(self, campo)}")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(f"warmup_ratio fora de [0, 1): {self.warmup_ratio}")

    @property
    def modelo(self) -> str:
        """Identificador do checkpoint de pré-treino no Hugging Face Hub."""
        return ENCODERS[self.encoder]

    def como_dict(self) -> dict[str, Any]:
        """Todos os campos, mais o nome da configuração e o checkpoint: é o que
        uma execução grava para registrar com que configuração foi produzida."""
        return {"tipo": type(self).__name__, "modelo": self.modelo, **asdict(self)}
