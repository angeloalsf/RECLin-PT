"""Inicialização independente de cada estágio a partir dos pesos originais.

Antes da carga (`recusar_origem`), o ponto de partida precisa ser a pasta do
pré-treino baixada por `reclin.pretreino` (com `PROVENIENCIA.json`, conferida)
e é recusado se tiver sinais de um checkpoint ajustado:

* o caminho está dentro de uma execução (uma pasta com o `config.json` de uma
  execução, ou `checkpoints/`, ou `melhor_modelo/`) — o `melhor_modelo/` do
  baseline, do outro estágio, do restrito ou da Pair-Aware;
* o tokenizer já tem os marcadores de entidade (`[E1]`…`[/E2]`), que só os
  checkpoints gravados pelo RECLin têm;
* a configuração já tem rótulos da tarefa ou de um estágio, ou uma
  arquitetura de classificação (`*ForSequenceClassification`, Pair-Aware).

Depois da carga (`verificar_carga`), três verificações separadas:

1. **Encoder** — cada tensor carregado é igual, elemento a elemento, ao do
   arquivo de pesos do commit; as linhas `[0, V)` da matriz de embeddings
   também. As chaves que faltam no checkpoint precisam ser exatamente as
   esperadas: a cabeça (`classifier.*`) e, se o checkpoint não tiver pooler,
   o pooler (registrado e tratado como parte nova, como no baseline).
2. **Cabeça** — `classifier.*` é nova (não veio do checkpoint), com formato
   [2, hidden]; foi inicializada pelo `from_pretrained` depois de
   `fixar_sementes(seed)`. Os dois estágios têm o mesmo formato e a mesma
   semente, então começam com os mesmos valores de cabeça: são instâncias
   separadas, sem estado compartilhado.
3. **Marcadores** — os quatro marcadores têm os ids V a V+3, os mesmos do
   baseline (o mesmo `modelos.carregar_tokenizer`); as quatro linhas novas
   vêm de `resize_token_embeddings` na versão fixada da biblioteca. Como a
   cabeça tem outro formato que a do baseline, o consumo do gerador muda e os
   valores dessas linhas diferem dos do baseline — esperado.

O relatório (hash de cada parte) vai para o `treino.json` de cada estágio e é
o que `scripts/two_stage.py verificar-inicializacao` imprime antes do treino.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from reclin import pretreino
from reclin.entrada import MARCADORES
from reclin.tarefa import LABELS

ARQUITETURAS_RECUSADAS = ("ForSequenceClassification", "PairAware")
CABECA = ("classifier.weight", "classifier.bias")


class ErroInicializacao(ValueError):
    """O ponto de partida não é o pré-treino original, ou a carga não confere."""


def _rotulos_proibidos() -> set[str]:
    from reclin.estrategias.two_stage import ROTULOS
    return set(LABELS) | {r for rs in ROTULOS.values() for r in rs}


def _dentro_de_execucao(pasta: Path) -> Path | None:
    for atual in (pasta, *pasta.parents):
        if atual.name in ("checkpoints", "melhor_modelo"):
            return atual
        config = atual / "config.json"
        if config.is_file():
            try:
                conteudo = json.loads(config.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(conteudo, dict) and {"estrategia", "conjuntos"} <= set(conteudo):
                return atual
    return None


def recusar_origem(modelo: str | Path) -> dict[str, str]:
    """Confere o ponto de partida antes de qualquer carga; devolve a
    proveniência (`pretreino.conferir`)."""
    pasta = Path(modelo)
    if not pasta.is_dir():
        raise ErroInicializacao(f"o Two-Stage parte de uma pasta local com os pesos do pré-treino "
                                f"(baixada por scripts/pretreino.py); {modelo!r} não é uma pasta")
    if not pretreino.tem_manifesto(pasta):
        raise ErroInicializacao(f"{pasta} não tem {pretreino.ARQUIVO}: baixe os pesos num commit exato "
                                "com scripts/pretreino.py baixar")
    execucao = _dentro_de_execucao(pasta.resolve())
    if execucao is not None:
        raise ErroInicializacao(f"{pasta} está dentro de uma execução ({execucao}): o ponto de partida "
                                "é o pré-treino original, nunca um checkpoint ajustado")
    proveniencia = pretreino.conferir(pasta)
    config = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    rotulos = {str(v) for v in (config.get("id2label") or {}).values()}
    if rotulos & _rotulos_proibidos():
        raise ErroInicializacao(f"a configuração de {pasta} já tem rótulos da tarefa "
                                f"({sorted(rotulos & _rotulos_proibidos())}): é um checkpoint ajustado")
    arquiteturas = [a for a in (config.get("architectures") or [])
                    if any(r in a for r in ARQUITETURAS_RECUSADAS)]
    if arquiteturas:
        raise ErroInicializacao(f"a configuração de {pasta} tem uma cabeça de classificação "
                                f"({arquiteturas}): é um checkpoint ajustado")
    from transformers import AutoTokenizer
    vocabulario = AutoTokenizer.from_pretrained(str(pasta)).get_vocab()
    presentes = [m for m in MARCADORES if m in vocabulario]
    if presentes:
        raise ErroInicializacao(f"o tokenizer de {pasta} já tem os marcadores {presentes}: "
                                "é um checkpoint gravado pelo RECLin")
    return proveniencia


def _tensores_do_checkpoint(pasta: Path) -> dict[str, Any]:
    """Os tensores dos arquivos de pesos da pasta (safetensors ou .bin, com ou
    sem índice de shards)."""
    import torch
    arquivos: list[Path] = []
    for indice in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (pasta / indice).is_file():
            mapa = json.loads((pasta / indice).read_text(encoding="utf-8"))["weight_map"]
            arquivos = [pasta / n for n in sorted(set(mapa.values()))]
            break
    else:
        for nome in ("model.safetensors", "pytorch_model.bin"):
            if (pasta / nome).is_file():
                arquivos = [pasta / nome]
                break
    if not arquivos:
        raise ErroInicializacao(f"nenhum arquivo de pesos em {pasta}")
    tensores: dict[str, Any] = {}
    for arquivo in arquivos:
        if arquivo.suffix == ".safetensors":
            from safetensors.torch import load_file
            tensores.update(load_file(str(arquivo)))
        else:
            tensores.update(torch.load(arquivo, map_location="cpu", weights_only=True))
    return tensores


def _nome_no_modelo(chave: str, prefixo: str) -> list[str]:
    normal = chave.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias")
    return [normal, f"{prefixo}.{normal}"]


def _hash_tensores(tensores: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for nome in sorted(tensores):
        t = tensores[nome].detach().cpu().contiguous()
        h.update(nome.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def verificar_carga(modelo: str | Path, rede: Any, info: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    """As três verificações (encoder, cabeça, marcadores); devolve o relatório."""
    import torch
    pasta = Path(modelo)
    prefixo = rede.base_model_prefix
    estado = rede.state_dict()
    checkpoint: dict[str, Any] = {}
    for chave, tensor in _tensores_do_checkpoint(pasta).items():
        for candidato in _nome_no_modelo(chave, prefixo):
            if candidato in estado:
                checkpoint[candidato] = tensor
                break
    chave_embeddings = f"{prefixo}.embeddings.word_embeddings.weight"
    if chave_embeddings not in checkpoint:
        raise ErroInicializacao("o checkpoint não tem a matriz de embeddings de palavras")
    v0 = int(checkpoint[chave_embeddings].shape[0])

    faltantes = sorted(info.get("missing_keys", []))
    pooler = sorted(k for k in estado if k.startswith(f"{prefixo}.pooler."))
    pooler_novo = bool(pooler) and not any(k in checkpoint for k in pooler)
    esperadas = sorted(set(CABECA) | (set(pooler) if pooler_novo else set()))
    if faltantes != esperadas:
        raise ErroInicializacao(f"chaves novas inesperadas na carga: {faltantes} (esperadas: {esperadas})")
    if info.get("mismatched_keys"):
        raise ErroInicializacao(f"tensores com formato divergente do checkpoint: {info['mismatched_keys']}")

    divergentes, encoder = [], {}
    for nome, tensor in estado.items():
        if nome in esperadas:
            continue
        if nome not in checkpoint:
            divergentes.append(f"{nome}: sem correspondente no checkpoint")
            continue
        carregado = tensor[:v0] if nome == chave_embeddings else tensor
        original = checkpoint[nome].to(carregado.dtype)
        if carregado.shape != original.shape or not torch.equal(carregado, original):
            divergentes.append(f"{nome}: difere do arquivo de pesos")
        encoder[nome] = carregado
    if divergentes:
        raise ErroInicializacao("o encoder carregado não é o do pré-treino: " + "; ".join(divergentes[:10]))

    cabeca = {n: estado[n] for n in CABECA}
    if tuple(cabeca["classifier.weight"].shape) != (2, rede.config.hidden_size):
        raise ErroInicializacao(f"cabeça com formato {tuple(cabeca['classifier.weight'].shape)}; "
                                f"esperado (2, {rede.config.hidden_size})")
    ids = tokenizer.convert_tokens_to_ids(list(MARCADORES))
    if ids != list(range(v0, v0 + len(MARCADORES))):
        raise ErroInicializacao(f"marcadores com ids {ids}; esperados {list(range(v0, v0 + len(MARCADORES)))}")
    linhas = estado[chave_embeddings]
    if int(linhas.shape[0]) != v0 + len(MARCADORES):
        raise ErroInicializacao(f"matriz de embeddings com {int(linhas.shape[0])} linhas; esperadas "
                                f"{v0 + len(MARCADORES)} (vocabulário + marcadores)")
    return {
        "encoder": {"origem": "arquivo de pesos do commit", "n_tensores": len(encoder),
                    "iguais_ao_checkpoint": True, "sha256": _hash_tensores(encoder)},
        "cabeca": {"chaves": list(CABECA), "nova": True, "formato": list(cabeca["classifier.weight"].shape),
                   "sha256": _hash_tensores(cabeca)},
        "pooler": {"novo": pooler_novo, "chaves": pooler},
        "marcadores": {"tokens": list(MARCADORES), "ids": ids, "vocabulario_original": v0,
                       "sha256_linhas_novas": _hash_tensores({"marcadores": linhas[v0:]})},
        "chaves_nao_usadas_do_checkpoint": sorted(info.get("unexpected_keys", [])),
    }


def hash_encoder(rede: Any, vocabulario_original: int) -> str:
    """Hash do encoder (sem cabeça, sem as linhas dos marcadores) de um modelo
    já carregado: compara os estágios depois do treino."""
    prefixo = rede.base_model_prefix
    chave_embeddings = f"{prefixo}.embeddings.word_embeddings.weight"
    estado = {n: t for n, t in rede.state_dict().items() if n not in CABECA}
    estado[chave_embeddings] = estado[chave_embeddings][:vocabulario_original]
    return _hash_tensores(estado)
