"""Checkpoints: o estado completo de retomada e o melhor modelo.

Dentro do diretório da execução:

    checkpoints/
        ultimo.pt          estado completo de retomada (ver ESTADO), regravado a
                           cada checkpoint
        ultimo.json        resumo legível do mesmo checkpoint (posição, data)
        melhor_modelo/     pesos e tokenizer da melhor época (`save_pretrained`)

`ultimo.pt` é a fonte da verdade: é gravado num arquivo temporário e renomeado
(`os.replace`), de modo que uma interrupção durante a gravação deixa o
checkpoint anterior intacto. `ultimo.json` é gravado depois e só informa.

O estado de retomada (`montar_estado`) guarda tudo o que o laço precisa para
continuar exatamente de onde parou, não só os pesos:

* pesos do modelo, estado do otimizador e do agendador de lr;
* estados dos geradores globais (python, numpy, torch, CUDA) e do gerador do
  DataLoader de treino — o do início da época corrente, que define a ordem dos
  exemplos dela, e o atual;
* posição: época, passo dentro da época, passo global e se a época terminou
  (com avaliação no DEV e seleção da melhor época feitas);
* soma da loss da época até o passo, histórico do DEV e o melhor macro-F1;
* identidade da execução (configuração, modelo, tokenizer, partições) e o
  ambiente que gravou o checkpoint.

`conferir_identidade` recusa retomar com outra configuração: o legado ignorava
o checkpoint em silêncio e recomeçava do zero.

Origem no legado: `save_last_checkpoint`, `load_last_checkpoint`,
`save_best_model`, `load_best_state` e `_config_guard` de
`src/relation_extraction.py`; `checar_ckpt_dir` dos treinadores do restrito e
da Pair-Aware.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from reclin import modelos

PASTA = "checkpoints"
ULTIMO = "ultimo.pt"
RESUMO = "ultimo.json"
MELHOR = "melhor_modelo"
FORMATO = 1


class ErroRetomada(RuntimeError):
    """O checkpoint não pode ser retomado por esta execução."""


def pasta_checkpoints(pasta_execucao: str | Path) -> Path:
    return Path(pasta_execucao) / PASTA


def montar_estado(*, posicao: dict[str, Any], modelo: Any, otimizador: Any, agendador: Any,
                  rng: dict[str, Any], gerador_inicio_epoca: Any, gerador_atual: Any,
                  soma_loss_epoca: float, historico: list[dict[str, Any]], melhor_f1: float,
                  identidade: dict[str, Any], ambiente: dict[str, Any]) -> dict[str, Any]:
    return {
        "formato": FORMATO,
        "posicao": dict(posicao),
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "agendador": agendador.state_dict(),
        "rng": rng,
        "gerador_inicio_epoca": gerador_inicio_epoca,
        "gerador_atual": gerador_atual,
        "soma_loss_epoca": soma_loss_epoca,
        "historico": [dict(h) for h in historico],
        "melhor_f1": melhor_f1,
        "identidade": identidade,
        "ambiente": ambiente,
    }


def salvar_estado(pasta_execucao: str | Path, estado: dict[str, Any]) -> Path:
    pasta = pasta_checkpoints(pasta_execucao)
    pasta.mkdir(parents=True, exist_ok=True)
    final, tmp = pasta / ULTIMO, pasta / (ULTIMO + ".tmp")
    torch.save(estado, tmp)
    os.replace(tmp, final)
    resumo = {"posicao": estado["posicao"], "gravado_em": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "melhor_f1": estado["melhor_f1"], "epocas_no_historico": len(estado["historico"])}
    tmp_json = pasta / (RESUMO + ".tmp")
    tmp_json.write_text(json.dumps(resumo, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    os.replace(tmp_json, pasta / RESUMO)
    return final


def carregar_estado(pasta_execucao: str | Path) -> dict[str, Any] | None:
    caminho = pasta_checkpoints(pasta_execucao) / ULTIMO
    if not caminho.is_file():
        return None
    # weights_only=False: o estado tem os geradores do python e do numpy, que
    # não são tensores. O arquivo é gravado por esta mesma execução.
    estado = torch.load(caminho, map_location="cpu", weights_only=False)
    if estado.get("formato") != FORMATO:
        raise ErroRetomada(f"{caminho} tem formato {estado.get('formato')}; este código lê {FORMATO}")
    return estado


def conferir_identidade(estado: dict[str, Any], identidade: dict[str, Any]) -> None:
    """Falha se o checkpoint foi gravado por outra configuração, outro modelo
    ou outras partições."""
    gravada = estado["identidade"]
    divergentes = sorted(k for k in set(gravada) | set(identidade) if gravada.get(k) != identidade.get(k))
    if divergentes:
        detalhes = "; ".join(f"{k}: checkpoint={gravada.get(k)!r}, agora={identidade.get(k)!r}"
                             for k in divergentes)
        raise ErroRetomada("o checkpoint foi gravado por outra execução (" + detalhes + "). "
                           "Retome com a mesma configuração ou use outra pasta.")


def salvar_melhor(pasta_execucao: str | Path, modelo: Any, tokenizer: Any) -> Path:
    return modelos.salvar(pasta_checkpoints(pasta_execucao) / MELHOR, modelo, tokenizer)


def tem_melhor(pasta_execucao: str | Path) -> bool:
    return (pasta_checkpoints(pasta_execucao) / MELHOR / "config.json").is_file()


def estado_do_melhor(pasta_execucao: str | Path, modelo: Any) -> dict[str, Any] | None:
    """Os pesos de `melhor_modelo/` como state_dict, recarregados pela classe do
    próprio modelo. None se ainda não houver melhor modelo."""
    if not tem_melhor(pasta_execucao):
        return None
    _, melhor = modelos.recarregar(pasta_checkpoints(pasta_execucao) / MELHOR, type(modelo))
    return {k: v.detach().cpu().clone() for k, v in melhor.state_dict().items()}
