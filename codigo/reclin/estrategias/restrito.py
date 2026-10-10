"""Fine-tuning restrito: treinar só nos pares cujo e1 é pista de negação.

O que muda em relação ao baseline (e só isto):

1. Os exemplos de TRAIN, DEV e TEST são só os candidatos cujo e1 é pista do
   léxico congelado (`negacao.lexico`, `min_freq=3`, induzido do TRAIN),
   na ordem do conjunto completo. A janela de cada um é, byte a byte, a que o
   baseline vê para o mesmo par.
2. `epochs` padrão 10 (`ConfigRestrito`): cada época tem ~107 passos, não 2.386.
3. Os pesos `balanced` são recalculados no TRAIN restrito (a mesma fórmula do
   baseline aplicada ao espaço em que se treina).
4. A melhor época é escolhida pelo macro-F1 do DEV **remapeado** ao conjunto
   completo (`execucao.subconjunto`: fora do espaço, `no_relation` com
   probabilidades [0, 0, 1]); as métricas do DEV restrito também entram no
   histórico (`dev_restrito_*`), só como informação.

As predições gravadas cobrem o conjunto completo (com o `y_true` dele) e
levam os índices do espaço restrito (`restricted_indices`). Como quase todo
`associated_with` fica fora do espaço, só o F1 de `negation_of` é comparável
com o baseline; macro-F1, MCC e o McNemar não são.

O léxico é dado de fora (o congelado) e gravado na execução; a avaliação do
TEST reconstrói a seleção a partir dele, sem ler o TRAIN.

Origem no legado: `src/finetuning_restrito/train_restrito.py` e a construção
do espaço em `restricted_space.py` (`construir_espaco`, `pesos_balanced`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from reclin.config import Config
from reclin.negacao.lexico import CONGELADO, e_pista, lexico_sha1
from reclin.treino.montagem import Montagem

NOME = "restrito"
ESPACO = "restrito: e1 no lexico induzido do train"


@dataclass(frozen=True)
class ConfigRestrito(Config):
    epochs: int = 10


def nome_execucao(config: Config) -> str:
    return f"restrito_{config.encoder}_seed{config.seed}"


def selecionar(candidatos: Sequence[dict], lexico: Mapping[str, int]) -> list[int]:
    """Posições dos candidatos cujo e1 é pista, na ordem do conjunto."""
    return [i for i, c in enumerate(candidatos) if e_pista(c["e1"], lexico)]


def registro_lexico(lexico: Mapping[str, int]) -> dict[str, Any]:
    return {"min_freq": CONGELADO["min_freq"], "max_gap": CONGELADO["max_gap"], "n_formas": len(lexico),
            "lexico_sha1": lexico_sha1(lexico), "formas": dict(lexico)}


def lexico_de_registro(registro: Mapping[str, Any]) -> dict[str, int]:
    """O léxico gravado numa execução, conferido pelo seu hash."""
    lexico = dict(registro["lexico"]["formas"])
    if lexico_sha1(lexico) != registro["lexico"]["lexico_sha1"]:
        raise ValueError("o léxico gravado na execução não confere com o seu lexico_sha1")
    return lexico


def montagem(config: Config, *, lexico: Mapping[str, int]) -> Montagem:
    lexico = dict(lexico)
    return Montagem(
        estrategia=NOME, classe_config=ConfigRestrito,
        selecionar=lambda candidatos: selecionar(candidatos, lexico),
        registro={"espaco": ESPACO, "lexico": registro_lexico(lexico),
                  "selecao_melhor_epoca": "dev_macro_f1 remapeado ao espaco completo",
                  "remapeamento": "fora do espaco restrito -> no_relation"},
        extras_sidecar={"espaco": "restrito remapeado ao completo", "encoder": config.encoder})


def montagem_de_registro(registro: Mapping[str, Any]) -> Montagem:
    return montagem(ConfigRestrito(**{k: v for k, v in registro["config"].items()
                                      if k not in ("tipo", "modelo")}),
                    lexico=lexico_de_registro(registro["estrategia_config"]))
