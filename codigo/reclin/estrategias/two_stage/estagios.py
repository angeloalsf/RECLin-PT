"""Exemplos, modelo e pontuação do DEV de cada estágio.

* `exemplos_do_estagio` — a partir dos exemplos completos da partição (as
  janelas do baseline, na ordem do conjunto), os do estágio: no estágio 1,
  todos, com o rótulo binário `relacao`/`no_relation`; no estágio 2, só os
  pares com relação anotada (as mesmas janelas, nos mesmos índices), com
  `negation_of`/`associated_with`. A seleção do estágio 2 usa o rótulo
  ANOTADO e serve só ao treino e à escolha do checkpoint; na inferência
  (`composicao`) quem decide o que chega ao estágio 2 é o estágio 1.
* `criar_modelo` — o encoder com a cabeça de classificação de sequência de 2
  saídas, na mesma ordem do classificador do baseline (`from_pretrained`,
  depois `resize_token_embeddings`), com o relatório de carga.
* `pontuador` — o que o laço chama ao fim de cada época: no estágio 1, o AP de
  `relacao` sobre as probabilidades do DEV completo (uma passada de inferência
  a mais, sem consumir geradores aleatórios), mais P/R/F1 de `relacao` e o
  recall dos pares `negation_of` (diagnóstico); no estágio 2, o F1 de
  `negation_of` nos pares do DEV com relação anotada.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from reclin import entrada
from reclin.estrategias.two_stage import (CAMPO_SELECAO, NEG_E2, RELACAO, ROTULOS, rotulo_estagio1,
                                         rotulo_estagio2)
from reclin.estrategias.two_stage import metricas as metricas_estagio
from reclin.tarefa import NEG, NOREL


def indices_com_relacao(rotulos: Sequence[int]) -> list[int]:
    """Posições dos pares com relação anotada (negation_of ou associated_with)."""
    return [i for i, r in enumerate(rotulos) if r != NOREL]


def exemplos_do_estagio(estagio: int, completos: entrada.Exemplos
                        ) -> tuple[list[int] | None, entrada.Exemplos]:
    """(índices no conjunto completo, ou None = todos; exemplos com o rótulo do estágio)."""
    if estagio == 1:
        return None, entrada.Exemplos(list(completos.textos), [rotulo_estagio1(r) for r in completos.rotulos])
    if estagio == 2:
        indices = indices_com_relacao(completos.rotulos)
        sub = completos.subconjunto(indices)
        return indices, entrada.Exemplos(sub.textos, [rotulo_estagio2(r) for r in sub.rotulos])
    raise ValueError(f"estágio desconhecido: {estagio!r}")


def criar_modelo(modelo_id: str, tokenizer: Any, estagio: int) -> tuple[Any, dict[str, Any]]:
    """Encoder + cabeça [CLS] de 2 saídas, a partir dos pesos de `modelo_id`."""
    from transformers import AutoModelForSequenceClassification
    rotulos = ROTULOS[estagio]
    rede, info = AutoModelForSequenceClassification.from_pretrained(
        str(modelo_id), num_labels=len(rotulos), id2label=dict(enumerate(rotulos)),
        label2id={r: i for i, r in enumerate(rotulos)}, output_loading_info=True)
    rede.resize_token_embeddings(len(tokenizer))
    return rede, dict(info)


def pontuador(estagio: int, *, rede: Any, loader_dev: Any, dispositivo: Any, y_dev: Sequence[int],
              rotulos_tarefa_dev: Sequence[int]) -> Callable[[list[int]], dict[str, float]]:
    """A função `pontuar_dev(preds)` do laço para o estágio."""
    from reclin.treino import laco
    campo = CAMPO_SELECAO[estagio]

    if estagio == 1:
        def pontuar(preds: list[int]) -> dict[str, float]:
            _, probs = laco.prever(rede, loader_dev, dispositivo, com_probs=True)
            positivo = [1 if y == RELACAO else 0 for y in y_dev]
            m = metricas_estagio.binarias(y_dev, preds, positivo=RELACAO)
            negacoes = [i for i, r in enumerate(rotulos_tarefa_dev) if r == NEG]
            recall_neg = (sum(1 for i in negacoes if preds[i] == RELACAO) / len(negacoes)) if negacoes else 0.0
            return {campo: metricas_estagio.ap(positivo, [p[RELACAO] for p in probs]),
                    "dev_f1_relacao": m["f1"], "dev_precision_relacao": m["precision"],
                    "dev_recall_relacao": m["recall"], "dev_macro_f1_estagio": m["macro_f1"],
                    "dev_recall_relacao_nos_pares_negation_of": recall_neg}
        return pontuar

    if estagio == 2:
        def pontuar(preds: list[int]) -> dict[str, float]:
            m = metricas_estagio.binarias(y_dev, preds, positivo=NEG_E2)
            return {campo: m["f1"], "dev_precision_negation_of_relacoes_anotadas": m["precision"],
                    "dev_recall_negation_of_relacoes_anotadas": m["recall"],
                    "dev_macro_f1_estagio": m["macro_f1"]}
        return pontuar
    raise ValueError(f"estágio desconhecido: {estagio!r}")
