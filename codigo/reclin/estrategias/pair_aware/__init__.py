"""Pair-Aware: o baseline com uma cabeça que lê o par de entidades, sem MLP e sem léxico.

A hipótese (uma só): ler explicitamente o par, `[h_cls ; h_[E1] ; h_[E2]] → Dropout →
Linear(3d, 3)` (`cabeca.PairAwareClassifier`), em vez do `[CLS]` do classificador de sequência,
melhora a extração de relações como um todo.

O que muda em relação ao baseline (e só isto):

1. A cabeça: `PairAwareClassifier` no lugar de `AutoModelForSequenceClassification` (encoder sem
   pooler; camada linear única sobre a concatenação do par — **sem MLP de fusão**; dropout do
   encoder, salvo `head_dropout`; `lr` da cabeça = a do encoder);
2. A recarga do melhor modelo, pela classe da cabeça.

Todo o resto é o do baseline: **todos** os pares candidatos elegíveis (nenhum filtro lexical, nenhum
remapeamento), a mesma geração de candidatos e janelas, os pesos `balanced` do TRAIN completo, a
escolha da época pelo macro-F1 do DEV completo (empate → a primeira) e os mesmos hiperparâmetros
(`ConfigPairAware` é a `Config` com `epochs=3` e `batch_size=64`, os do baseline, mais `head_dropout`).
**O léxico não é usado de nenhuma forma**: nem para selecionar pares, nem como feature, indicador ou
atributo derivado dele. Por isso a avaliação é sobre o mesmo universo do baseline e do Two-Stage, e as
métricas de todas as classes são comparáveis com as deles.

Antes do treino (e antes do TEST, na avaliação) é medido quantos exemplos ficam sem `[E1]` ou `[E2]`
depois da tokenização e caem no fallback para `h_cls` (`descricao.marcadores_ausentes` em
`treino.json`).

Histórico: a versão anterior (legado e etapa 6) era o fine-tuning restrito com cabeça MLP: treinava só
nos pares com `e1` no léxico congelado, avaliava remapeando o resto para `no_relation` e tinha teto de
2 épocas (D5). Essa configuração foi substituída por esta (autorizado pelo autor em 13/10/2026); as
referências do legado em `testes/referencia/` e `resultados_legado/` ficam preservadas como histórico e
não são mais reproduzidas por este código. Execuções gravadas com a configuração antiga não são
retomadas nem avaliadas aqui (`montagem_de_registro` as recusa).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from reclin.config import Config
from reclin.estrategias.pair_aware.cabeca import ARQUITETURA, PairAwareClassifier, contar_marcadores_ausentes
from reclin.tarefa import ID2LABEL, LABEL2ID, LABELS
from reclin.treino.montagem import Montagem

NOME = "pair_aware"
ESPACO = "completo: todos os pares candidatos elegiveis, sem filtro lexical"
VERSAO = "v2: cabeca linear sem MLP, sem filtro lexical, sem lexico"


@dataclass(frozen=True)
class ConfigPairAware(Config):
    epochs: int = 3                     # o mesmo teto do baseline
    head_dropout: float | None = None   # None = hidden_dropout_prob do encoder

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.head_dropout is not None and not 0 <= self.head_dropout < 1:
            raise ValueError(f"head_dropout fora de [0, 1): {self.head_dropout}")


def nome_execucao(config: Config) -> str:
    return f"pairaware_{config.encoder}_seed{config.seed}"


def criar_modelo(modelo_id: str, tokenizer: Any, config: ConfigPairAware) -> PairAwareClassifier:
    return PairAwareClassifier.from_encoder_pretrained(
        modelo_id, tokenizer=tokenizer, num_labels=len(LABELS), dropout=config.head_dropout,
        id2label=dict(ID2LABEL), label2id=dict(LABEL2ID))


def montagem(config: ConfigPairAware | None = None, **_: Any) -> Montagem:
    config = config or ConfigPairAware()

    def descrever(tokenizer: Any, modelo: PairAwareClassifier, exemplos: Mapping[str, Any]) -> dict:
        return {"cabeca": modelo.pair_aware_config, "n_params_cabeca": modelo.n_parametros_cabeca(),
                "marcadores_ausentes": {p: contar_marcadores_ausentes(tokenizer, ex.textos, config.max_length)
                                        for p, ex in exemplos.items()}}

    return Montagem(
        estrategia=NOME, classe_config=ConfigPairAware, criar_modelo=criar_modelo,
        classe_modelo=PairAwareClassifier, selecionar=None,
        registro={"versao": VERSAO, "espaco": ESPACO, "filtro_lexical": "desativado",
                  "lexico": "nao utilizado (nem como filtro, nem como feature)",
                  "cabeca": {"arquitetura": ARQUITETURA, "mlp": False, "fallback_marcador_ausente": "h_cls",
                             "lr_cabeca": "a mesma do encoder"},
                  "selecao_melhor_epoca": "dev_macro_f1 no DEV completo"},
        extras_sidecar={},                 # sem seleção, o orquestrador não grava extras (como o baseline)
        descrever=descrever)


def montagem_de_registro(registro: Mapping[str, Any]) -> Montagem:
    """A montagem de uma execução gravada; recusa as da configuração antiga (restrita, com MLP)."""
    if registro.get("estrategia_config", {}).get("versao") != VERSAO:
        raise ValueError("execução Pair-Aware de outra configuração (a anterior era restrita ao léxico e com "
                         "MLP): ela não é retomada nem avaliada por este código")
    return montagem(ConfigPairAware(**{k: v for k, v in registro["config"].items()
                                       if k not in ("tipo", "modelo")}))
