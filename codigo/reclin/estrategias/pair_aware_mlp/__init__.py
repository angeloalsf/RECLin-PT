"""Pair-Aware com MLP: o baseline com uma cabeça MLP que lê o par de entidades, sem léxico.

Estratégia **independente** da `pair_aware` (cabeça linear, D10), que não é alterada: nome, configuração,
registro de proveniência, nome de execução, arquivos do modelo e checkpoints são próprios.

A hipótese (uma só): combinar `[h_cls ; h_[E1] ; h_[E2]]` com uma MLP (`cabeca.PairAwareMLPClassifier`) antes
de classificar melhora a extração de relações como um todo em relação ao `[CLS]` do baseline.

O que muda em relação ao baseline (e só isto): a cabeça (encoder sem pooler + MLP sobre o par) e a recarga do
melhor modelo pela classe dela. O resto é o do baseline: **todos** os pares candidatos elegíveis (nenhum filtro
lexical, nenhum remapeamento), a mesma geração de candidatos e janelas, os pesos `balanced` do TRAIN completo, a
escolha da época pelo macro-F1 do DEV completo (primeiro máximo) e os hiperparâmetros de `config.Config`
(3 épocas, lote 64). **O léxico não é usado de nenhuma forma** (nem filtro, nem feature, nem indicador).

Fora do protocolo pré-registrado da fase (`docs/protocolo/fase-biobertpt-seed42.md`): não tem nome oficial nem
configuração de protocolo em `scripts/pretreino.py`; a conferência final da fase não a inclui. É uma estratégia
exploratória até que o autor a registre numa emenda ao protocolo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from reclin.config import Config
from reclin.estrategias.pair_aware_mlp.cabeca import (ARQUITETURA, PairAwareMLPClassifier,
                                                      contar_marcadores_ausentes)
from reclin.tarefa import ID2LABEL, LABEL2ID, LABELS
from reclin.treino.montagem import Montagem

NOME = "pair_aware_mlp"
ESPACO = "completo: todos os pares candidatos elegiveis, sem filtro lexical"
VERSAO = "v1: cabeca MLP, sem filtro lexical, sem lexico"


@dataclass(frozen=True)
class ConfigPairAwareMLP(Config):
    mlp_hidden: int | None = None       # None = hidden_size do encoder
    head_dropout: float | None = None   # None = hidden_dropout_prob do encoder

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mlp_hidden is not None and self.mlp_hidden <= 0:
            raise ValueError(f"mlp_hidden precisa ser positivo: {self.mlp_hidden}")
        if self.head_dropout is not None and not 0 <= self.head_dropout < 1:
            raise ValueError(f"head_dropout fora de [0, 1): {self.head_dropout}")


def nome_execucao(config: Config) -> str:
    return f"pairawaremlp_{config.encoder}_seed{config.seed}"


def criar_modelo(modelo_id: str, tokenizer: Any, config: ConfigPairAwareMLP) -> PairAwareMLPClassifier:
    return PairAwareMLPClassifier.from_encoder_pretrained(
        modelo_id, tokenizer=tokenizer, num_labels=len(LABELS), mlp_hidden=config.mlp_hidden,
        dropout=config.head_dropout, id2label=dict(ID2LABEL), label2id=dict(LABEL2ID))


def montagem(config: ConfigPairAwareMLP | None = None, **_: Any) -> Montagem:
    config = config or ConfigPairAwareMLP()

    def descrever(tokenizer: Any, modelo: PairAwareMLPClassifier, exemplos: Mapping[str, Any]) -> dict:
        return {"cabeca": modelo.pair_aware_mlp_config, "n_params_cabeca": modelo.n_parametros_cabeca(),
                "marcadores_ausentes": {p: contar_marcadores_ausentes(tokenizer, ex.textos, config.max_length)
                                        for p, ex in exemplos.items()}}

    return Montagem(
        estrategia=NOME, classe_config=ConfigPairAwareMLP, criar_modelo=criar_modelo,
        classe_modelo=PairAwareMLPClassifier, selecionar=None,
        registro={"versao": VERSAO, "espaco": ESPACO, "filtro_lexical": "desativado",
                  "lexico": "nao utilizado (nem como filtro, nem como feature)",
                  "cabeca": {"arquitetura": ARQUITETURA, "mlp": True, "fallback_marcador_ausente": "h_cls",
                             "lr_cabeca": "a mesma do encoder"},
                  "selecao_melhor_epoca": "dev_macro_f1 no DEV completo"},
        descrever=descrever)


def montagem_de_registro(registro: Mapping[str, Any]) -> Montagem:
    """A montagem de uma execução gravada; recusa registros de outra versão desta estratégia."""
    if registro.get("estrategia_config", {}).get("versao") != VERSAO:
        raise ValueError("execução pair_aware_mlp de outra versão da configuração: ela não é retomada nem "
                         "avaliada por este código")
    return montagem(ConfigPairAwareMLP(**{k: v for k, v in registro["config"].items()
                                          if k not in ("tipo", "modelo")}))
