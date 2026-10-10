"""Pair-Aware: o fine-tuning restrito com uma cabeça que lê o par de entidades.

A hipótese (uma só): uma cabeça que lê explicitamente o par,
`[h_cls ; h_[E1] ; h_[E2]] → MLP` (`cabeca.PairAwareClassifier`), em vez do
`[CLS]` do classificador de sequência, reduz os falsos positivos
`no_relation → negation_of` o bastante para superar o filtro de pistas no F1
de `negation_of` do DEV.

O que muda em relação ao fine-tuning restrito (e só isto):

1. A cabeça: `PairAwareClassifier` no lugar de
   `AutoModelForSequenceClassification` (encoder sem pooler; MLP com a largura
   e o dropout do encoder, salvo `mlp_hidden`/`head_dropout`; `lr` da cabeça =
   a do encoder);
2. A recarga do melhor modelo, pela classe da cabeça.

Todo o resto é o do restrito: os mesmos exemplos (e1 no léxico congelado, na
ordem do conjunto), os mesmos pesos `balanced` do TRAIN restrito, a mesma
escolha de época (macro-F1 do DEV remapeado), o mesmo remapeamento e os
mesmos hiperparâmetros (`ConfigPairAware` difere de `ConfigRestrito` só nos
campos da cabeça). A seleção dos pares é repetida aqui (duas linhas), para que
uma estratégia não importe a outra.

Antes do treino (e antes do TEST, na avaliação) é medido quantos exemplos
ficam sem `[E1]` ou `[E2]` depois da tokenização e caem no fallback para
`h_cls` (`descricao.marcadores_ausentes` em `treino.json`). No legado essa
contagem incluía o TEST já no treino; aqui a do TEST só é feita na avaliação
do TEST.

Não é um filtro: o filtro de pistas (etapa 4) pós-processa predições de outro
modelo; a Pair-Aware é um modelo treinado no espaço restrito.

Origem no legado: `src/pair_aware/train_pair_aware.py` e `model.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from reclin.config import Config
from reclin.estrategias.pair_aware.cabeca import ARQUITETURA, PairAwareClassifier, contar_marcadores_ausentes
from reclin.negacao.lexico import CONGELADO, e_pista, lexico_sha1
from reclin.tarefa import ID2LABEL, LABEL2ID, LABELS
from reclin.treino.montagem import Montagem

NOME = "pair_aware"
ESPACO = "restrito: e1 no lexico induzido do train"


@dataclass(frozen=True)
class ConfigPairAware(Config):
    epochs: int = 10
    mlp_hidden: int | None = None       # None = hidden_size do encoder
    head_dropout: float | None = None   # None = hidden_dropout_prob do encoder

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mlp_hidden is not None and self.mlp_hidden <= 0:
            raise ValueError(f"mlp_hidden precisa ser positivo: {self.mlp_hidden}")
        if self.head_dropout is not None and not 0 <= self.head_dropout < 1:
            raise ValueError(f"head_dropout fora de [0, 1): {self.head_dropout}")


def nome_execucao(config: Config) -> str:
    return f"pairaware_{config.encoder}_seed{config.seed}"


def selecionar(candidatos: Sequence[dict], lexico: Mapping[str, int]) -> list[int]:
    """Posições dos candidatos cujo e1 é pista, na ordem do conjunto."""
    return [i for i, c in enumerate(candidatos) if e_pista(c["e1"], lexico)]


def criar_modelo(modelo_id: str, tokenizer: Any, config: ConfigPairAware) -> PairAwareClassifier:
    return PairAwareClassifier.from_encoder_pretrained(
        modelo_id, tokenizer=tokenizer, num_labels=len(LABELS), mlp_hidden=config.mlp_hidden,
        dropout=config.head_dropout, id2label=dict(ID2LABEL), label2id=dict(LABEL2ID))


def montagem(config: ConfigPairAware, *, lexico: Mapping[str, int]) -> Montagem:
    lexico = dict(lexico)

    def descrever(tokenizer: Any, modelo: PairAwareClassifier, exemplos: Mapping[str, Any]) -> dict:
        return {"cabeca": modelo.pair_aware_config, "n_params_cabeca": modelo.n_parametros_cabeca(),
                "marcadores_ausentes": {p: contar_marcadores_ausentes(tokenizer, ex.textos, config.max_length)
                                        for p, ex in exemplos.items()}}

    return Montagem(
        estrategia=NOME, classe_config=ConfigPairAware, criar_modelo=criar_modelo,
        classe_modelo=PairAwareClassifier,
        selecionar=lambda candidatos: selecionar(candidatos, lexico),
        registro={"espaco": ESPACO,
                  "lexico": {"min_freq": CONGELADO["min_freq"], "max_gap": CONGELADO["max_gap"],
                             "n_formas": len(lexico), "lexico_sha1": lexico_sha1(lexico), "formas": lexico},
                  "cabeca": {"arquitetura": ARQUITETURA, "fallback_marcador_ausente": "h_cls",
                             "lr_cabeca": "a mesma do encoder"},
                  "selecao_melhor_epoca": "dev_macro_f1 remapeado ao espaco completo",
                  "remapeamento": "fora do espaco restrito -> no_relation"},
        extras_sidecar={"espaco": "restrito remapeado ao completo", "cabeca": "pair-aware",
                        "encoder": config.encoder},
        descrever=descrever)


def montagem_de_registro(registro: Mapping[str, Any]) -> Montagem:
    gravado = registro["estrategia_config"]["lexico"]
    lexico = dict(gravado["formas"])
    if lexico_sha1(lexico) != gravado["lexico_sha1"]:
        raise ValueError("o léxico gravado na execução não confere com o seu lexico_sha1")
    return montagem(ConfigPairAware(**{k: v for k, v in registro["config"].items()
                                       if k not in ("tipo", "modelo")}), lexico=lexico)
