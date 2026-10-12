"""Fixtures da integração que não dependem de nenhum artefato do legado.

`modelo_sintetico` cria, dentro do próprio teste, um BERT aleatório minúsculo
e um tokenizer WordPiece de vocabulário sintético (caracteres), sem rede e sem
ler `testes/referencia/`. Serve aos testes que verificam propriedades do
código novo por si mesmas (critérios de seleção, relatórios), sem comparar
com resultados históricos.
"""
from __future__ import annotations

import string

import pytest

TOKENS_ESPECIAIS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def criar_modelo_sintetico(pasta, *, semente: int = 0) -> None:
    """Grava em `pasta` um `BertModel` aleatório (16 dimensões, 2 camadas) e
    um tokenizer WordPiece sobre letras, dígitos e pontuação."""
    import torch
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    simbolos = string.ascii_lowercase + string.digits + string.punctuation
    vocab = TOKENS_ESPECIAIS + list(simbolos) + [f"##{c}" for c in simbolos]
    tok = Tokenizer(models.WordPiece({t: i for i, t in enumerate(vocab)}, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    cls, sep = vocab.index("[CLS]"), vocab.index("[SEP]")
    from tokenizers import processors
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]", special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
    rapido = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                     cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                     model_max_length=128)
    torch.manual_seed(semente)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=128)
    BertModel(config).save_pretrained(pasta)
    rapido.save_pretrained(pasta)


@pytest.fixture(scope="session")
def modelo_sintetico(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    pasta = tmp_path_factory.mktemp("modelo_sintetico")
    criar_modelo_sintetico(pasta)
    return pasta
