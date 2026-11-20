"""Cabeça da ablação do pooler (`reclin.estrategias.ablacao_pooler`).

Uma classe, duas chaves:

* `usar_pooler`: o vetor do `[CLS]` é `pooler_output` (dense + tanh do encoder, o
  que o `BertForSequenceClassification` do baseline usa) ou `h_cls` direto da
  última camada (o que a Pair-Aware linear usa);
* `ler_marcadores`: a representação é só esse vetor ou a concatenação dele com
  os estados de `[E1]` e `[E2]`.

`(usar_pooler=True, ler_marcadores=False)` é a conta do baseline e
`(False, True)` é a da Pair-Aware linear; os testes conferem as duas
equivalências com os mesmos pesos. As estratégias novas são as outras duas
combinações. A cabeça é `Dropout(p) -> Linear(k·d, C)`, com a inicialização das
cabeças do `transformers` (normal(0, initializer_range), viés zero).

Arquivos no `melhor_modelo/`: `ablacao_pooler_config.json` e
`ablacao_pooler_head.safetensors`, além do encoder no formato HF (com o pooler,
quando usado).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from reclin.estrategias.ablacao_pooler import arquitetura
from reclin.estrategias.pair_aware.cabeca import ids_dos_marcadores, posicoes_dos_marcadores

FORMATO = 1
CONFIG_FILE = "ablacao_pooler_config.json"
HEAD_FILE = "ablacao_pooler_head.safetensors"


def _carregar_encoder(nome_ou_pasta: str, usar_pooler: bool):
    """(encoder, pooler_veio_do_checkpoint). Sem pooler, o encoder é carregado sem a camada."""
    from transformers import AutoModel
    if not usar_pooler:
        return AutoModel.from_pretrained(nome_ou_pasta, add_pooling_layer=False), None
    encoder, info = AutoModel.from_pretrained(nome_ou_pasta, output_loading_info=True)
    if getattr(encoder, "pooler", None) is None:
        raise ValueError(f"{nome_ou_pasta}: o encoder não tem pooler; a variante com pooler não se aplica")
    ausentes = [k for k in info.get("missing_keys", []) if "pooler" in k]
    return encoder, not ausentes


class CabecaAblacao(nn.Module):
    """Encoder HF + Dropout + Linear sobre o `[CLS]` (com ou sem pooler), com ou sem os marcadores."""

    def __init__(self, encoder: nn.Module, *, ler_marcadores: bool, usar_pooler: bool, e1_token_id: int,
                 e2_token_id: int, num_labels: int = 3, dropout: float | None = None, id2label: dict | None = None,
                 label2id: dict | None = None, pooler_do_checkpoint: bool | None = None):
        super().__init__()
        if usar_pooler and getattr(encoder, "pooler", None) is None:
            raise ValueError("usar_pooler=True exige um encoder carregado com o pooler")
        self.encoder = encoder
        self.ler_marcadores, self.usar_pooler = bool(ler_marcadores), bool(usar_pooler)
        cfg = encoder.config
        d = int(cfg.hidden_size)
        p = float(dropout) if dropout is not None else float(
            getattr(cfg, "classifier_dropout", None) or getattr(cfg, "hidden_dropout_prob", 0.1))
        self.e1_token_id, self.e2_token_id = int(e1_token_id), int(e2_token_id)
        self.num_labels = int(num_labels)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear((3 if self.ler_marcadores else 1) * d, self.num_labels))
        std = float(getattr(cfg, "initializer_range", 0.02))
        nn.init.normal_(self.head[1].weight, mean=0.0, std=std)
        nn.init.zeros_(self.head[1].bias)
        self.ablacao_config = {
            "formato": FORMATO, "arquitetura": arquitetura(self.ler_marcadores, self.usar_pooler),
            "ler_marcadores": self.ler_marcadores, "usar_pooler": self.usar_pooler,
            "pooler_do_checkpoint": pooler_do_checkpoint, "hidden_size": d, "dropout": p,
            "num_labels": self.num_labels, "e1_token_id": self.e1_token_id, "e2_token_id": self.e2_token_id,
            "fallback_marcador_ausente": "posicao 0 (estado de [CLS] da ultima camada)" if self.ler_marcadores else None,
            "id2label": {str(k): v for k, v in (id2label or {}).items()}, "label2id": dict(label2id or {}),
            "encoder_class": type(encoder).__name__, "initializer_range": std}

    @classmethod
    def from_encoder_pretrained(cls, nome_ou_pasta: str, *, tokenizer, ler_marcadores: bool, usar_pooler: bool,
                                num_labels: int = 3, dropout: float | None = None, id2label: dict | None = None,
                                label2id: dict | None = None):
        """Encoder pré-treinado (Hub ou pasta) + cabeça nova; redimensiona os embeddings ao tokenizer."""
        encoder, do_checkpoint = _carregar_encoder(nome_ou_pasta, usar_pooler)
        encoder.resize_token_embeddings(len(tokenizer))
        e1, e2 = ids_dos_marcadores(tokenizer)
        return cls(encoder, ler_marcadores=ler_marcadores, usar_pooler=usar_pooler, e1_token_id=e1, e2_token_id=e2,
                   num_labels=num_labels, dropout=dropout, id2label=id2label, label2id=label2id,
                   pooler_do_checkpoint=do_checkpoint)

    @classmethod
    def from_pretrained(cls, pasta: str | Path, **_ignorado):
        from safetensors.torch import load_file
        pasta = Path(pasta)
        cfg = json.loads((pasta / CONFIG_FILE).read_text(encoding="utf-8"))
        if int(cfg.get("formato", -1)) != FORMATO:
            raise ValueError(f"{pasta / CONFIG_FILE}: formato {cfg.get('formato')} != {FORMATO}")
        encoder, do_checkpoint = _carregar_encoder(str(pasta), cfg["usar_pooler"])
        if cfg["usar_pooler"] and not do_checkpoint:
            raise ValueError(f"{pasta}: o modelo gravado não tem os pesos do pooler")
        modelo = cls(encoder, ler_marcadores=cfg["ler_marcadores"], usar_pooler=cfg["usar_pooler"],
                     e1_token_id=cfg["e1_token_id"], e2_token_id=cfg["e2_token_id"], num_labels=cfg["num_labels"],
                     dropout=cfg["dropout"], id2label={int(k): v for k, v in cfg.get("id2label", {}).items()},
                     label2id=cfg.get("label2id"), pooler_do_checkpoint=cfg.get("pooler_do_checkpoint"))
        modelo.head.load_state_dict(load_file(str(pasta / HEAD_FILE)))
        return modelo

    def save_pretrained(self, pasta: str | Path, **_ignorado) -> None:
        from safetensors.torch import save_file
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(pasta)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  str(pasta / HEAD_FILE))
        (pasta / CONFIG_FILE).write_text(json.dumps(self.ablacao_config, ensure_ascii=False, indent=2, sort_keys=True),
                                         encoding="utf-8", newline="\n")

    @property
    def config(self):
        return self.encoder.config

    def n_parametros_cabeca(self) -> int:
        return sum(p.numel() for p in self.head.parameters())

    def representacao(self, input_ids, attention_mask=None, token_type_ids=None):
        saida = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h = saida.last_hidden_state
        cls = saida.pooler_output if self.usar_pooler else h[:, 0]
        if not self.ler_marcadores:
            return cls
        pos1, _ = posicoes_dos_marcadores(input_ids, attention_mask, self.e1_token_id)
        pos2, _ = posicoes_dos_marcadores(input_ids, attention_mask, self.e2_token_id)
        linhas = torch.arange(h.size(0), device=h.device)
        return torch.cat([cls, h[linhas, pos1], h[linhas, pos2]], dim=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **_ignorado):
        """`SequenceClassifierOutput` só com `logits` (a perda fica no laço de treino)."""
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(logits=self.head(self.representacao(input_ids, attention_mask, token_type_ids)))
