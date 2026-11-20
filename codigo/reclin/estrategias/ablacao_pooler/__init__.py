"""Ablação do pooler: separa o efeito de ler os marcadores do efeito de retirar o pooler.

O baseline e a Pair-Aware linear diferem em **duas** coisas ao mesmo tempo:

                      | com pooler (dense + tanh)      | sem pooler (h_cls direto)
    ------------------+--------------------------------+------------------------------
    só o [CLS]        | `baseline`                     | **`cls_direto`** (nova)
    [CLS] + [E1]+[E2] | **`pair_aware_pooler`** (nova) | `pair_aware`

As duas estratégias novas completam o fatorial 2×2. Com as quatro, mede-se o
efeito dos marcadores com e sem pooler e o efeito do pooler com e sem marcadores.
Tudo o mais é o do baseline e da Pair-Aware linear: todos os pares candidatos,
sem filtro lexical e sem léxico, a mesma entrada com marcadores, os mesmos
hiperparâmetros de `config.Config` (3 épocas, lote 64, lr 2e-5, pesos
`balanced`), a mesma semente e a seleção da época pelo macro-F1 do DEV completo.

* `cls_direto`: encoder sem pooler; `h_cls` (última camada) → Dropout → Linear(d, C).
* `pair_aware_pooler`: encoder com pooler; `[pooler(h_cls) ; h_[E1] ; h_[E2]]` →
  Dropout → Linear(3d, C). O pooler é o do encoder pré-treinado, como no
  `BertForSequenceClassification` do baseline; se o checkpoint não trouxer os
  pesos do pooler, eles são inicializados aleatoriamente (como no baseline) e
  isso fica registrado na execução.

São estratégias **exploratórias**, posteriores ao protocolo da fase: não têm
nome oficial, não entram na conferência final e não são elegíveis ao TEST. A
cabeça (`cabeca.CabecaAblacao`) é própria; `baseline` e `pair_aware` não são
alterados. Este módulo não importa torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from reclin.config import Config
from reclin.tarefa import ID2LABEL, LABEL2ID, LABELS
from reclin.treino.montagem import Montagem

ESPACO = "completo: todos os pares candidatos elegiveis, sem filtro lexical"


def arquitetura(ler_marcadores: bool, usar_pooler: bool) -> str:
    cls = "pooler(h_cls) [dense + tanh do encoder]" if usar_pooler else "h_cls (ultima camada, sem pooler)"
    entrada = f"concat[{cls} ; h_[E1] ; h_[E2]]" if ler_marcadores else cls
    return f"{entrada} -> Dropout(p) -> Linear({'3d' if ler_marcadores else 'd'}, C)"


@dataclass(frozen=True)
class ConfigAblacao(Config):
    epochs: int = 3                     # o mesmo teto do baseline e da Pair-Aware linear
    head_dropout: float | None = None   # None = dropout do encoder (o da cabeça do baseline)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.head_dropout is not None and not 0 <= self.head_dropout < 1:
            raise ValueError(f"head_dropout fora de [0, 1): {self.head_dropout}")


@dataclass(frozen=True)
class Variante:
    """Uma célula do fatorial; tem a interface que os scripts esperam de uma estratégia."""
    NOME: str
    VERSAO: str
    ler_marcadores: bool
    usar_pooler: bool
    prefixo_execucao: str
    Config = ConfigAblacao

    def nome_execucao(self, config: Config) -> str:
        return f"{self.prefixo_execucao}_{config.encoder}_seed{config.seed}"

    def criar_modelo(self, modelo_id: str, tokenizer: Any, config: ConfigAblacao) -> Any:
        from reclin.estrategias.ablacao_pooler.cabeca import CabecaAblacao
        return CabecaAblacao.from_encoder_pretrained(
            modelo_id, tokenizer=tokenizer, ler_marcadores=self.ler_marcadores, usar_pooler=self.usar_pooler,
            num_labels=len(LABELS), dropout=config.head_dropout, id2label=dict(ID2LABEL), label2id=dict(LABEL2ID))

    def registro(self) -> dict[str, Any]:
        return {"versao": self.VERSAO, "espaco": ESPACO, "filtro_lexical": "desativado",
                "lexico": "nao utilizado (nem como filtro, nem como feature)",
                "cabeca": {"arquitetura": arquitetura(self.ler_marcadores, self.usar_pooler), "mlp": False,
                           "ler_marcadores": self.ler_marcadores, "usar_pooler": self.usar_pooler,
                           "fallback_marcador_ausente": "h_cls" if self.ler_marcadores else "nao se aplica",
                           "lr_cabeca": "a mesma do encoder"},
                "ablacao": "fatorial 2x2 marcadores x pooler (com `baseline` e `pair_aware`)",
                "selecao_melhor_epoca": "dev_macro_f1 no DEV completo"}

    def montagem(self, config: ConfigAblacao | None = None, **_: Any) -> Montagem:
        from reclin.estrategias.ablacao_pooler.cabeca import CabecaAblacao
        from reclin.estrategias.pair_aware.cabeca import contar_marcadores_ausentes
        config = config or ConfigAblacao()

        def descrever(tokenizer: Any, modelo: Any, exemplos: Mapping[str, Any]) -> dict:
            return {"cabeca": modelo.ablacao_config, "n_params_cabeca": modelo.n_parametros_cabeca(),
                    "marcadores_ausentes": {p: contar_marcadores_ausentes(tokenizer, ex.textos, config.max_length)
                                            for p, ex in exemplos.items()}}

        return Montagem(estrategia=self.NOME, classe_config=ConfigAblacao, criar_modelo=self.criar_modelo,
                        classe_modelo=CabecaAblacao, selecionar=None, registro=self.registro(),
                        extras_sidecar={}, descrever=descrever)

    def montagem_de_registro(self, registro: Mapping[str, Any]) -> Montagem:
        """A montagem de uma execução gravada; recusa outra versão da variante."""
        if registro.get("estrategia_config", {}).get("versao") != self.VERSAO:
            raise ValueError(f"execução {self.NOME} de outra versão: não é retomada nem avaliada por este código")
        return self.montagem(ConfigAblacao(**{k: v for k, v in registro["config"].items()
                                              if k not in ("tipo", "modelo")}))


cls_direto = Variante(NOME="cls_direto", VERSAO="v1: h_cls direto (sem pooler), cabeca linear, sem lexico",
                      ler_marcadores=False, usar_pooler=False, prefixo_execucao="clsdireto")
pair_aware_pooler = Variante(NOME="pair_aware_pooler",
                             VERSAO="v1: pooler(h_cls) + marcadores, cabeca linear, sem lexico",
                             ler_marcadores=True, usar_pooler=True, prefixo_execucao="pairawarepooler")
VARIANTES = (cls_direto, pair_aware_pooler)
