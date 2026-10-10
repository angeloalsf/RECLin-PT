"""O laço de treino, com avaliação no DEV, checkpoints e retomada exata.

Cada época: passa pelos lotes de treino (zero_grad → forward → loss →
backward → recorte do gradiente → passo do otimizador → passo do agendador),
calcula a loss média e as predições no DEV, pontua o DEV (`pontuar_dev`, que a
estratégia fornece) e guarda a melhor época pelo `dev_macro_f1` — estritamente
maior, de modo que no empate vence a primeira. Grava um checkpoint completo ao
fim de cada época e, com `checkpoint_a_cada`, a cada N passos.

Retomada exata
--------------
Com `retomar=True`, o laço carrega o último checkpoint e continua do ponto em
que ele foi gravado:

* fim de época — começa a época seguinte com o gerador do DataLoader no
  estado em que a época anterior o deixou;
* meio de época (passo k) — volta o gerador ao estado do início daquela
  época, recria a mesma ordem de lotes, avança os k lotes já feitos (só
  tokeniza; não toca no modelo nem nos geradores globais) e continua no lote
  k + 1 com a soma da loss da época até ali.

Em seguida restaura os geradores globais (python, numpy, torch, CUDA), de que
dependem o dropout e qualquer sorteio, no estado do momento do checkpoint. Com
isso, no mesmo ambiente, uma execução interrompida e retomada produz os mesmos
pesos, o mesmo otimizador, o mesmo agendador, o mesmo histórico e as mesmas
predições que a execução contínua (conferido bit a bit nos testes em CPU).

O laço percorre o iterador do DataLoader até o fim em cada época (o
`RandomSampler` faz um último sorteio ao se esgotar): é o que mantém o gerador
igual entre a execução contínua e a retomada.

`parar_apos_passo` encerra o treino logo depois de gravar o checkpoint daquele
passo global; serve para testar e demonstrar a retomada.

Origem no legado: o laço de `run()` de `src/relation_extraction.py` (copiado
em `train_restrito.py` e `train_pair_aware.py`), `predict`, `evaluate`,
`best_epoch_from_history` e os pesos `balanced` (inline em `run()` e
`restricted_space.pesos_balanced`). O avanço do gerador na retomada, que no
legado só a Pair-Aware fazia (por épocas inteiras), vale aqui para qualquer
treino e também no meio de uma época.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from reclin.tarefa import LABELS
from reclin.treino import checkpoint, reprodutibilidade

log = logging.getLogger(__name__)

CAMPO_SELECAO = "dev_macro_f1"
LOG_A_CADA = 20


def pesos_balanced(rotulos: Sequence[int], n_rotulos: int = len(LABELS)) -> list[float]:
    """n / (n_rotulos · contagem) por rótulo; um rótulo ausente conta como 1."""
    contagens = np.bincount(np.asarray(rotulos, dtype=np.int64), minlength=n_rotulos).astype(float)
    contagens[contagens == 0] = 1.0
    return [float(p) for p in contagens.sum() / (n_rotulos * contagens)]


def prever(modelo: Any, loader: Any, dispositivo: Any, *,
           com_probs: bool = False) -> list[int] | tuple[list[int], list[list[float]]]:
    """Predições (argmax) e, com `com_probs`, a softmax por exemplo, na ordem do
    loader. Em `eval()` e sem gradiente: não consome os geradores aleatórios."""
    modelo.eval()
    preds: list[int] = []
    probs: list[list[float]] = []
    with torch.no_grad():
        for ids, mascara, _ in loader:
            logits = modelo(input_ids=ids.to(dispositivo), attention_mask=mascara.to(dispositivo)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            if com_probs:
                probs.extend(torch.softmax(logits.float(), dim=-1).cpu().tolist())
    return (preds, probs) if com_probs else preds


def avaliar_loss(modelo: Any, loader: Any, dispositivo: Any, perda: Any) -> tuple[list[int], float]:
    """Predições e a loss média por lote (a curva de validação)."""
    modelo.eval()
    preds: list[int] = []
    total, n_lotes = 0.0, 0
    with torch.no_grad():
        for ids, mascara, rotulos in loader:
            logits = modelo(input_ids=ids.to(dispositivo), attention_mask=mascara.to(dispositivo)).logits
            total += perda(logits, rotulos.to(dispositivo)).item()
            n_lotes += 1
            preds.extend(logits.argmax(-1).cpu().tolist())
    return preds, total / max(1, n_lotes)


def melhor_epoca(historico: Sequence[dict[str, Any]], campo: str = CAMPO_SELECAO) -> tuple[int | None, float | None]:
    """A época escolhida pelo laço (o primeiro máximo estrito de `campo`),
    reconstruída do histórico; (None, None) sem épocas avaliadas."""
    epoca, melhor = None, -1.0
    for entrada in historico:
        valor = entrada.get(campo)
        if valor is not None and valor > melhor:
            epoca, melhor = int(entrada["epoch"]), valor
    return epoca, (melhor if epoca is not None else None)


@dataclass
class ResultadoTreino:
    concluido: bool
    historico: list[dict[str, Any]]
    melhor_f1: float
    melhor_estado: dict[str, Any] | None   # pesos da melhor época, se ela ocorreu nesta sessão
    posicao: dict[str, Any]                # onde o treino parou
    retomado_de: dict[str, Any] | None = None
    checkpoints: list[dict[str, Any]] = field(default_factory=list)


def treinar(*, modelo: Any, tokenizer: Any, loader_treino: Any, loader_dev: Any, perda: Any,
            otimizador: Any, agendador: Any, epocas: int, max_grad_norm: float, dispositivo: Any,
            pontuar_dev: Callable[[list[int]], dict[str, float]], pasta_execucao: str | Path,
            identidade: dict[str, Any], retomar: bool = False,
            checkpoint_a_cada: int | None = None,
            parar_apos_passo: int | None = None) -> ResultadoTreino:
    """Treina `modelo` até `epocas` (ou até `parar_apos_passo`) gravando
    checkpoints em `pasta_execucao`. `pontuar_dev(preds)` devolve as métricas
    do DEV da época, com `dev_macro_f1` (o critério de seleção) primeiro."""
    if checkpoint_a_cada is not None and checkpoint_a_cada <= 0:
        raise ValueError(f"checkpoint_a_cada precisa ser positivo: {checkpoint_a_cada}")
    if parar_apos_passo is not None and parar_apos_passo <= 0:
        raise ValueError(f"parar_apos_passo precisa ser positivo: {parar_apos_passo}")
    n_passos = len(loader_treino)
    gerador = loader_treino.generator
    historico: list[dict[str, Any]] = []
    melhor_f1, melhor_estado = -1.0, None
    epoca_inicial, pular, soma_retomada, passo_global = 1, 0, 0.0, 0
    rng_pendente, retomado_de = None, None

    if retomar:
        estado = checkpoint.carregar_estado(pasta_execucao)
        if estado is None:
            raise checkpoint.ErroRetomada(f"não há checkpoint em {checkpoint.pasta_checkpoints(pasta_execucao)}")
        checkpoint.conferir_identidade(estado, identidade)
        modelo.load_state_dict(estado["modelo"])
        otimizador.load_state_dict(estado["otimizador"])
        agendador.load_state_dict(estado["agendador"])
        historico, melhor_f1 = [dict(h) for h in estado["historico"]], estado["melhor_f1"]
        posicao = retomado_de = dict(estado["posicao"])
        passo_global = posicao["passo_global"]
        if posicao["epoca_concluida"]:
            epoca_inicial = posicao["epoca"] + 1
            gerador.set_state(estado["gerador_atual"])
        else:
            epoca_inicial, pular = posicao["epoca"], posicao["passo_na_epoca"]
            soma_retomada = estado["soma_loss_epoca"]
            gerador.set_state(estado["gerador_inicio_epoca"])
        rng_pendente = estado["rng"]
        log.info("Retomando: época %d, passo %d de %d (passo global %d); melhor dev_macro_f1=%.4f",
                 posicao["epoca"], posicao["passo_na_epoca"], n_passos, passo_global, melhor_f1)
    elif checkpoint.carregar_estado(pasta_execucao) is not None:
        raise checkpoint.ErroRetomada(f"já há checkpoint em {checkpoint.pasta_checkpoints(pasta_execucao)}: "
                                      "para continuar o treino, retome; para recomeçar, use outra pasta")

    resultado = ResultadoTreino(concluido=False, historico=historico, melhor_f1=melhor_f1,
                                melhor_estado=None, posicao={}, retomado_de=retomado_de)

    def gravar(epoca: int, passo: int, concluida: bool, inicio_epoca: Any, soma: float) -> None:
        posicao = {"epoca": epoca, "passo_na_epoca": passo, "passo_global": passo_global,
                   "epoca_concluida": concluida, "passos_por_epoca": n_passos, "epocas": epocas}
        estado = checkpoint.montar_estado(
            posicao=posicao, modelo=modelo, otimizador=otimizador, agendador=agendador,
            rng=reprodutibilidade.estados_rng(), gerador_inicio_epoca=inicio_epoca,
            gerador_atual=gerador.get_state(), soma_loss_epoca=soma, historico=historico,
            melhor_f1=melhor_f1, identidade=identidade, ambiente=reprodutibilidade.ambiente())
        checkpoint.salvar_estado(pasta_execucao, estado)
        resultado.posicao = posicao
        resultado.checkpoints.append(posicao)
        log.info("Checkpoint gravado: época %d, passo %d/%d (global %d)%s", epoca, passo, n_passos,
                 passo_global, ", fim da época" if concluida else "")

    if epoca_inicial > epocas:
        log.info("As %d épocas já foram concluídas: nada a treinar", epocas)
        resultado.concluido, resultado.posicao = True, dict(retomado_de or {})
        return resultado

    for epoca in range(epoca_inicial, epocas + 1):
        inicio = time.time()
        modelo.train()
        inicio_epoca = gerador.get_state()
        iterador = iter(loader_treino)
        primeiro = pular if epoca == epoca_inicial else 0
        soma = soma_retomada if primeiro else 0.0
        for _ in range(primeiro):            # lotes já feitos antes da interrupção
            next(iterador)
        if rng_pendente is not None:
            reprodutibilidade.restaurar_rng(rng_pendente)
            rng_pendente = None
        log.info("Época %d/%d: início%s", epoca, epocas,
                 f" (retomada no passo {primeiro + 1})" if primeiro else "")

        for passo, (ids, mascara, rotulos) in enumerate(iterador, primeiro + 1):
            otimizador.zero_grad()
            logits = modelo(input_ids=ids.to(dispositivo), attention_mask=mascara.to(dispositivo)).logits
            loss = perda(logits, rotulos.to(dispositivo))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_grad_norm)
            otimizador.step()
            agendador.step()
            soma += loss.item()
            passo_global += 1
            if passo % LOG_A_CADA == 0:
                log.info("  época %d | passo %d/%d | loss média=%.4f", epoca, passo, n_passos, soma / passo)
            if passo < n_passos:
                parar = passo_global == parar_apos_passo
                if parar or (checkpoint_a_cada and passo_global % checkpoint_a_cada == 0):
                    gravar(epoca, passo, False, inicio_epoca, soma)
                if parar:
                    log.info("Treino interrompido a pedido depois do passo global %d", passo_global)
                    resultado.historico, resultado.melhor_f1 = historico, melhor_f1
                    return resultado

        train_loss = soma / max(1, n_passos)
        preds_dev, dev_loss = avaliar_loss(modelo, loader_dev, dispositivo, perda)
        pontos = pontuar_dev(preds_dev)
        if CAMPO_SELECAO not in pontos:
            raise ValueError(f"pontuar_dev precisa devolver {CAMPO_SELECAO}")
        historico.append({"epoch": epoca, "train_loss": train_loss, "dev_loss": dev_loss, **pontos,
                          "duration_s": round(time.time() - inicio, 1)})
        log.info("Época %d: train_loss=%.4f | dev_loss=%.4f | %s", epoca, train_loss, dev_loss,
                 " | ".join(f"{k}={v:.4f}" for k, v in pontos.items()))
        if pontos[CAMPO_SELECAO] > melhor_f1:
            melhor_f1 = pontos[CAMPO_SELECAO]
            melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            checkpoint.salvar_melhor(pasta_execucao, modelo, tokenizer)
            log.info("Época %d: nova melhor (%s=%.4f)", epoca, CAMPO_SELECAO, melhor_f1)
        resultado.melhor_estado = melhor_estado
        gravar(epoca, n_passos, True, inicio_epoca, soma)
        if passo_global == parar_apos_passo and epoca < epocas:
            log.info("Treino interrompido a pedido depois do passo global %d (fim da época %d)",
                     passo_global, epoca)
            resultado.historico, resultado.melhor_f1 = historico, melhor_f1
            return resultado

    resultado.concluido = True
    resultado.historico, resultado.melhor_f1 = historico, melhor_f1
    return resultado
