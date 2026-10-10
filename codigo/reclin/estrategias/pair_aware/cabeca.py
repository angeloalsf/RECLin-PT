"""Cabeça Pair-Aware: classificação a partir do par de entidades.

O baseline e o restrito usam `AutoModelForSequenceClassification`: a predição
sai só da representação agregada da sequência (o `[CLS]`, pelo pooler), e os
marcadores influenciam a decisão só pela atenção. Esta cabeça lê
explicitamente o estado oculto da última camada em três posições,

    h_cls = H[0]        h_e1 = H[pos([E1])]        h_e2 = H[pos([E2])]

concatena os três e classifica com uma MLP pequena:

    [h_cls ; h_e1 ; h_e2]  (3d)  ->  Linear(3d, m)  ->  GELU  ->  Dropout(p)
                                 ->  Linear(m, 3)

É a variante "entity start" de Baldini Soares et al. (2019), com o `[CLS]`
mantido na concatenação. `m` é, por padrão, o `hidden_size` do encoder e `p`,
o `hidden_dropout_prob` (o dropout do classificador dos baselines).

Posição dos marcadores: `[E1]` e `[E2]` são tokens especiais atômicos; a
posição de cada um é a primeira ocorrência do id em `input_ids`, só entre as
posições com `attention_mask = 1`. Se o marcador não está na sequência
(truncado em `max_length`, ou não escrito na janela porque o offset da
entidade passa do fim do texto), a cabeça usa a posição 0 (`h_cls`).
`contar_marcadores_ausentes` mede quantos exemplos caem nesse caso, com a
mesma tokenização do loader.

O encoder é carregado com `AutoModel` sem o pooler (`add_pooling_layer=False`):
`h_cls` vem direto da última camada. A cabeça é inicializada como as cabeças do
`transformers` (normal(0, `initializer_range`), viés zero), consumindo o
gerador global do torch depois do encoder e do redimensionamento dos
embeddings — a ordem do legado.

`PairAwareClassifier` é um `nn.Module` com `save_pretrained`/`from_pretrained`
(encoder no formato HF + cabeça em safetensors + JSON de configuração): é o
que `modelos.salvar` e `modelos.recarregar` usam, sem conhecer a cabeça.

Origem no legado: `src/pair_aware/model.py`, sem mudança de comportamento.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

FORMATO = 1
CONFIG_FILE = "pair_aware_config.json"
HEAD_FILE = "pair_aware_head.safetensors"
ARQUITETURA = ("concat[h_cls ; h_[E1] ; h_[E2]] (ultima camada) -> Linear(3d, m) "
               "-> GELU -> Dropout(p) -> Linear(m, C)")


def _carregar_encoder(nome_ou_pasta: str, add_pooling_layer: bool = False):
    """`AutoModel` sem o pooler quando a classe aceita (BERT aceita).

    O pooler (dense + tanh sobre o [CLS]) e o que `BertForSequenceClassification`
    usa; aqui o `h_cls` vem direto da ultima camada, como pedido, e um pooler sem
    uso so carregaria parametros que nao recebem gradiente. Devolve
    (encoder, tem_pooler).
    """
    from transformers import AutoModel
    if not add_pooling_layer:
        try:
            return AutoModel.from_pretrained(nome_ou_pasta, add_pooling_layer=False), False
        except TypeError:
            pass
    return AutoModel.from_pretrained(nome_ou_pasta), True


def ids_dos_marcadores(tokenizer, e1: str = "[E1]", e2: str = "[E2]") -> tuple[int, int]:
    """Ids de [E1] e [E2] no tokenizer (que ja tem de te-los como especiais)."""
    ids = []
    for m in (e1, e2):
        i = tokenizer.convert_tokens_to_ids(m)
        if i is None or i == tokenizer.unk_token_id:
            raise ValueError(f"{m} nao e um token do tokenizer: chame "
                             f"add_special_tokens antes de construir a cabeca.")
        ids.append(int(i))
    return ids[0], ids[1]


def posicoes_dos_marcadores(input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                            token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Primeira posicao de `token_id` em cada linha e mascara de "achou".

    Linha sem o token devolve posicao 0 (fallback para `h_cls`). Opera so com
    comparacao, `any` e `argmax` sobre inteiros, todos deterministicos.
    """
    alvo = input_ids == token_id
    if attention_mask is not None:
        alvo = alvo & attention_mask.bool()
    achou = alvo.any(dim=1)
    # argmax devolve o PRIMEIRO maximo (documentado no torch.argmax).
    primeira = alvo.long().argmax(dim=1)
    return torch.where(achou, primeira, torch.zeros_like(primeira)), achou


class PairAwareClassifier(nn.Module):
    """Encoder HF + cabeca [h_cls ; h_e1 ; h_e2] -> MLP -> C classes."""

    def __init__(self, encoder: nn.Module, *, e1_token_id: int, e2_token_id: int,
                 num_labels: int = 3, mlp_hidden: int | None = None,
                 dropout: float | None = None, id2label: dict | None = None,
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
        self.head = nn.Sequential(
            nn.Linear(3 * d, m),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(m, self.num_labels),
        )
        # Mesma inicializacao das cabecas do `transformers` (BertForSequence-
        # Classification): normal(0, initializer_range), vies zero. Consome o RNG
        # global do torch DEPOIS do encoder, na mesma posicao em que o
        # AutoModelForSequenceClassification inicializa o classificador.
        std = float(getattr(cfg, "initializer_range", 0.02))
        for mod in self.head:
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                nn.init.zeros_(mod.bias)
        self.pair_aware_config = {
            "formato": FORMATO, "arquitetura": ARQUITETURA,
            "hidden_size": d, "mlp_hidden": m, "dropout": p,
            "num_labels": self.num_labels,
            "e1_token_id": self.e1_token_id, "e2_token_id": self.e2_token_id,
            "fallback_marcador_ausente": "posicao 0 (h_cls)",
            "id2label": {str(k): v for k, v in (id2label or {}).items()},
            "label2id": dict(label2id or {}),
            "encoder_class": type(encoder).__name__,
            "tem_pooler": bool(tem_pooler),
            "initializer_range": std,
        }

    # ------------------------------------------------------------------ #
    # Construcao                                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_encoder_pretrained(cls, nome_ou_pasta: str, *, tokenizer, num_labels: int = 3,
                                mlp_hidden: int | None = None, dropout: float | None = None,
                                id2label: dict | None = None, label2id: dict | None = None):
        """Encoder pre-treinado (Hub ou pasta) + cabeca nova.

        Redimensiona os embeddings para o tamanho do tokenizer, como `run()` faz
        nos baselines, porque o tokenizer ja recebeu os 4 marcadores.
        """
        encoder, tem_pooler = _carregar_encoder(nome_ou_pasta)
        encoder.resize_token_embeddings(len(tokenizer))
        e1, e2 = ids_dos_marcadores(tokenizer)
        return cls(encoder, e1_token_id=e1, e2_token_id=e2, num_labels=num_labels,
                   mlp_hidden=mlp_hidden, dropout=dropout, id2label=id2label,
                   label2id=label2id, tem_pooler=tem_pooler)

    @classmethod
    def from_pretrained(cls, pasta: str | Path, **_ignorado):
        """Recarrega o que `save_pretrained` gravou (encoder + cabeca)."""
        from safetensors.torch import load_file
        pasta = Path(pasta)
        cfg = json.loads((pasta / CONFIG_FILE).read_text(encoding="utf-8"))
        if int(cfg.get("formato", -1)) != FORMATO:
            raise ValueError(f"{pasta / CONFIG_FILE}: formato {cfg.get('formato')} != {FORMATO}")
        encoder, tem_pooler = _carregar_encoder(str(pasta),
                                                add_pooling_layer=cfg.get("tem_pooler", False))
        modelo = cls(encoder, e1_token_id=cfg["e1_token_id"], e2_token_id=cfg["e2_token_id"],
                     num_labels=cfg["num_labels"], mlp_hidden=cfg["mlp_hidden"],
                     dropout=cfg["dropout"],
                     id2label={int(k): v for k, v in cfg.get("id2label", {}).items()},
                     label2id=cfg.get("label2id"), tem_pooler=tem_pooler)
        modelo.head.load_state_dict(load_file(str(pasta / HEAD_FILE)))
        return modelo

    def save_pretrained(self, pasta: str | Path, **_ignorado) -> None:
        """Encoder no formato HF (`config.json` + pesos) + cabeca + configuracao.

        Assinatura compatível com o que `modelos.salvar` chama.
        """
        from safetensors.torch import save_file
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(pasta)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  str(pasta / HEAD_FILE))
        (pasta / CONFIG_FILE).write_text(
            json.dumps(self.pair_aware_config, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8", newline="\n")

    # ------------------------------------------------------------------ #
    # Uso                                                                 #
    # ------------------------------------------------------------------ #
    @property
    def config(self):
        """Config do encoder (o que codigo do `transformers` costuma consultar)."""
        return self.encoder.config

    def n_parametros_cabeca(self) -> int:
        return sum(p.numel() for p in self.head.parameters())

    def representacao_do_par(self, input_ids, attention_mask=None, token_type_ids=None):
        """Devolve (concatenacao [h_cls ; h_e1 ; h_e2], achou_e1, achou_e2)."""
        saida = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        h = saida.last_hidden_state                              # [B, T, d]
        pos1, achou1 = posicoes_dos_marcadores(input_ids, attention_mask, self.e1_token_id)
        pos2, achou2 = posicoes_dos_marcadores(input_ids, attention_mask, self.e2_token_id)
        linhas = torch.arange(h.size(0), device=h.device)
        par = torch.cat([h[:, 0], h[linhas, pos1], h[linhas, pos2]], dim=-1)
        return par, achou1, achou2

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **_ignorado):
        """Devolve um `SequenceClassifierOutput` so com `logits`.

        Mesmo contrato que `treino.laco.prever`/`avaliar_loss` usam
        (`model(input_ids=..., attention_mask=...).logits`). A perda fica fora,
        no laco de treino, com os pesos de classe do experimento.
        """
        from transformers.modeling_outputs import SequenceClassifierOutput
        par, _, _ = self.representacao_do_par(input_ids, attention_mask, token_type_ids)
        return SequenceClassifierOutput(logits=self.head(par))


def contar_marcadores_ausentes(tokenizer, textos: Sequence[str], max_length: int,
                               e1: str = "[E1]", e2: str = "[E2]",
                               lote: int = 512) -> dict:
    """Quantos textos ficam sem [E1]/[E2] depois da MESMA tokenizacao do loader
    (`truncation=True, max_length=...`, ver `entrada.criar_loader`).

    Separa os dois motivos: marcador que nem esta no texto (nao foi escrito na
    janela) e marcador cortado pelo truncamento.
    """
    id1, id2 = ids_dos_marcadores(tokenizer, e1, e2)
    out = {"n": len(textos), "max_length": int(max_length),
           "E1_ausente": 0, "E2_ausente": 0, "algum_ausente": 0,
           "E1_fora_do_texto": 0, "E2_fora_do_texto": 0,
           "E1_truncado": 0, "E2_truncado": 0}
    for i in range(0, len(textos), lote):
        parte = list(textos[i:i + lote])
        enc = tokenizer(parte, truncation=True, max_length=max_length)["input_ids"]
        for t, ids in zip(parte, enc):
            f1, f2 = id1 not in ids, id2 not in ids
            out["E1_ausente"] += f1
            out["E2_ausente"] += f2
            out["algum_ausente"] += (f1 or f2)
            for marc, falta, chave in ((e1, f1, "E1"), (e2, f2, "E2")):
                if falta:
                    out[f"{chave}_fora_do_texto" if marc not in t else f"{chave}_truncado"] += 1
    return out
