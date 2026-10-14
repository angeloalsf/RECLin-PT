"""Cabeça Pair-Aware com MLP: classificação a partir do par de entidades.

Lê o estado oculto da última camada em três posições, `h_cls = H[0]`, `h_e1 = H[pos([E1])]` e
`h_e2 = H[pos([E2])]`, concatena os três e os combina com uma MLP antes da classificação:

    [h_cls ; h_e1 ; h_e2]  (3d)  ->  Linear(3d, m)  ->  GELU  ->  Dropout(p)  ->  Linear(m, 3)

`m` é, por padrão, o `hidden_size` do encoder e `p`, o dropout do encoder (`hidden_dropout_prob`). Esta é a
cabeça independente da estratégia `pair_aware_mlp`; a `pair_aware` (cabeça linear, sem MLP) fica em
`reclin/estrategias/pair_aware/` e não é tocada. O que não depende da cabeça é reutilizado de lá, sem cópia:
a leitura das posições dos marcadores, o carregamento do encoder sem pooler e a contagem de marcadores ausentes.

Arquivos gravados no `melhor_modelo/` (nomes próprios, para que nunca sejam confundidos com os da `pair_aware`):
`pair_aware_mlp_config.json` e `pair_aware_mlp_head.safetensors`, além do encoder no formato HF.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from reclin.estrategias.pair_aware.cabeca import (_carregar_encoder, contar_marcadores_ausentes,  # noqa: F401
                                                  ids_dos_marcadores, posicoes_dos_marcadores)

FORMATO = 1
CONFIG_FILE = "pair_aware_mlp_config.json"
HEAD_FILE = "pair_aware_mlp_head.safetensors"
ARQUITETURA = ("concat[h_cls ; h_[E1] ; h_[E2]] (ultima camada) -> Linear(3d, m) -> GELU -> Dropout(p) "
               "-> Linear(m, C)")


class PairAwareMLPClassifier(nn.Module):
    """Encoder HF + cabeça MLP sobre [h_cls ; h_e1 ; h_e2] -> C classes."""

    def __init__(self, encoder: nn.Module, *, e1_token_id: int, e2_token_id: int, num_labels: int = 3,
                 mlp_hidden: int | None = None, dropout: float | None = None, id2label: dict | None = None,
                 label2id: dict | None = None, tem_pooler: bool = False):
        super().__init__()
        self.encoder = encoder
        cfg = encoder.config
        d = int(cfg.hidden_size)
        m = int(mlp_hidden) if mlp_hidden else d
        p = float(dropout) if dropout is not None else float(
            getattr(cfg, "classifier_dropout", None) or getattr(cfg, "hidden_dropout_prob", 0.1))
        self.e1_token_id, self.e2_token_id = int(e1_token_id), int(e2_token_id)
        self.num_labels = int(num_labels)
        self.head = nn.Sequential(nn.Linear(3 * d, m), nn.GELU(), nn.Dropout(p), nn.Linear(m, self.num_labels))
        # inicialização das cabeças do `transformers`: normal(0, initializer_range), viés zero
        std = float(getattr(cfg, "initializer_range", 0.02))
        for mod in self.head:
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                nn.init.zeros_(mod.bias)
        self.pair_aware_mlp_config = {
            "formato": FORMATO, "arquitetura": ARQUITETURA, "hidden_size": d, "mlp_hidden": m, "dropout": p,
            "num_labels": self.num_labels, "e1_token_id": self.e1_token_id, "e2_token_id": self.e2_token_id,
            "fallback_marcador_ausente": "posicao 0 (h_cls)",
            "id2label": {str(k): v for k, v in (id2label or {}).items()}, "label2id": dict(label2id or {}),
            "encoder_class": type(encoder).__name__, "tem_pooler": bool(tem_pooler), "initializer_range": std}

    @classmethod
    def from_encoder_pretrained(cls, nome_ou_pasta: str, *, tokenizer, num_labels: int = 3,
                                mlp_hidden: int | None = None, dropout: float | None = None,
                                id2label: dict | None = None, label2id: dict | None = None):
        """Encoder pré-treinado (Hub ou pasta) + cabeça nova; redimensiona os embeddings ao tokenizer."""
        encoder, tem_pooler = _carregar_encoder(nome_ou_pasta)
        encoder.resize_token_embeddings(len(tokenizer))
        e1, e2 = ids_dos_marcadores(tokenizer)
        return cls(encoder, e1_token_id=e1, e2_token_id=e2, num_labels=num_labels, mlp_hidden=mlp_hidden,
                   dropout=dropout, id2label=id2label, label2id=label2id, tem_pooler=tem_pooler)

    @classmethod
    def from_pretrained(cls, pasta: str | Path, **_ignorado):
        from safetensors.torch import load_file
        pasta = Path(pasta)
        cfg = json.loads((pasta / CONFIG_FILE).read_text(encoding="utf-8"))
        if int(cfg.get("formato", -1)) != FORMATO:
            raise ValueError(f"{pasta / CONFIG_FILE}: formato {cfg.get('formato')} != {FORMATO}")
        encoder, tem_pooler = _carregar_encoder(str(pasta), add_pooling_layer=cfg.get("tem_pooler", False))
        modelo = cls(encoder, e1_token_id=cfg["e1_token_id"], e2_token_id=cfg["e2_token_id"],
                     num_labels=cfg["num_labels"], mlp_hidden=cfg["mlp_hidden"], dropout=cfg["dropout"],
                     id2label={int(k): v for k, v in cfg.get("id2label", {}).items()},
                     label2id=cfg.get("label2id"), tem_pooler=tem_pooler)
        modelo.head.load_state_dict(load_file(str(pasta / HEAD_FILE)))
        return modelo

    def save_pretrained(self, pasta: str | Path, **_ignorado) -> None:
        from safetensors.torch import save_file
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(pasta)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  str(pasta / HEAD_FILE))
        (pasta / CONFIG_FILE).write_text(
            json.dumps(self.pair_aware_mlp_config, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8", newline="\n")

    @property
    def config(self):
        return self.encoder.config

    def n_parametros_cabeca(self) -> int:
        return sum(p.numel() for p in self.head.parameters())

    def representacao_do_par(self, input_ids, attention_mask=None, token_type_ids=None):
        """(concatenação [h_cls ; h_e1 ; h_e2], achou_e1, achou_e2); marcador ausente -> h_cls."""
        saida = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h = saida.last_hidden_state
        pos1, achou1 = posicoes_dos_marcadores(input_ids, attention_mask, self.e1_token_id)
        pos2, achou2 = posicoes_dos_marcadores(input_ids, attention_mask, self.e2_token_id)
        linhas = torch.arange(h.size(0), device=h.device)
        return torch.cat([h[:, 0], h[linhas, pos1], h[linhas, pos2]], dim=-1), achou1, achou2

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **_ignorado):
        """`SequenceClassifierOutput` só com `logits` (a perda fica no laço de treino)."""
        from transformers.modeling_outputs import SequenceClassifierOutput
        par, _, _ = self.representacao_do_par(input_ids, attention_mask, token_type_ids)
        return SequenceClassifierOutput(logits=self.head(par))
