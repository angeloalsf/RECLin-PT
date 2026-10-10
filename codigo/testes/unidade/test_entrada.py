"""Entrada dos modelos (`reclin.entrada`): janela marcada, exemplos e loader."""
from __future__ import annotations

import pytest

from reclin import entrada
from reclin.tarefa import LABEL2ID, candidatos
from reclin.util.caminhos import CODIGO

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"


def ent(id_, start, end, texto="x"):
    return {"id": id_, "start": start, "end": end, "text": texto, "type": "t"}


TEXTO = "0123456789abcdefghij"


def test_janela_marcada_simples():
    e1, e2 = ent("1", 2, 4), ent("2", 6, 9)
    assert entrada.janela_marcada(TEXTO, e1, e2, ctx_chars=1) == "1[E1] 23 [/E1]45[E2] 678 [/E2]9"


def test_janela_recorta_nos_limites_do_texto():
    e1, e2 = ent("1", 0, 2), ent("2", 18, 20)
    assert entrada.janela_marcada(TEXTO, e1, e2, ctx_chars=100) == \
        "[E1] 01 [/E1]23456789abcdefgh[E2] ij [/E2]"


def test_ordem_dos_marcadores_na_mesma_posicao():
    """Fechamentos antes de aberturas; empate entre fechamentos ou entre
    aberturas pela ordem do texto do marcador."""
    e1, e2 = ent("1", 2, 4), ent("2", 4, 6)            # e1 termina onde e2 começa
    assert entrada.janela_marcada(TEXTO, e1, e2, ctx_chars=0) == "[E1] 23 [/E1][E2] 45 [/E2]"
    e1, e2 = ent("1", 2, 6), ent("2", 2, 4)            # spans aninhados com o mesmo início
    assert entrada.janela_marcada(TEXTO, e1, e2, ctx_chars=0) == "[E1] [E2] 23 [/E2]45 [/E1]"
    e2, e1 = ent("2", 2, 6), ent("1", 4, 6)            # mesmo fim
    assert entrada.janela_marcada(TEXTO, e1, e2, ctx_chars=0) == "[E2] 23[E1] 45 [/E1] [/E2]"


def doc():
    entidades = [ent("1", 0, 4, "NEGA"), ent("2", 5, 10, "febre"), ent("3", 11, 14, "dor")]
    return {"doc_id": "d", "text": "NEGA febre dor e tosse", "entities": entidades,
            "relations": [{"e1_id": "1", "e2_id": "2", "type": "negation_of"}]}


def test_exemplos_alinhados_com_os_candidatos():
    d = doc()
    ex = entrada.exemplos([d], ctx_chars=3)
    cands = candidatos([d])
    assert len(ex) == len(cands) == 6
    assert ex.rotulos == [LABEL2ID[c["label"]] for c in cands]
    for texto, c in zip(ex.textos, cands):
        assert texto == entrada.janela_marcada(d["text"], c["e1"], c["e2"], 3)
    assert ex.textos[0].startswith("[E1] NEGA [/E1] [E2] febre [/E2]")


def test_exemplos_exigem_listas_alinhadas():
    with pytest.raises(ValueError, match="2 textos e 1 rótulos"):
        entrada.Exemplos(["a", "b"], [0])


def test_marcadores():
    assert entrada.MARCADORES == ("[E1]", "[/E1]", "[E2]", "[/E2]")


def test_loader_tokeniza_cada_lote_e_embaralha_com_gerador_proprio():
    torch = pytest.importorskip("torch")
    from reclin import modelos
    tok = modelos.carregar_tokenizer(MODELO_MINUSCULO)
    ex = entrada.exemplos([doc()], ctx_chars=3)
    estado_global = torch.get_rng_state()
    lotes = list(entrada.criar_loader(tok, ex, max_length=8, batch_size=4, embaralhar=True, seed=7))
    assert torch.equal(torch.get_rng_state(), estado_global)       # o RNG global não é usado
    assert [len(l[2]) for l in lotes] == [4, 2]
    assert all(l[0].shape[1] <= 8 and l[0].shape == l[1].shape for l in lotes)

    def pares(loader):          # (tokens sem padding, rótulo) de cada exemplo
        return sorted((tuple(t for t, m in zip(ids.tolist(), msk.tolist()) if m), int(r))
                      for ids_lote, msk_lote, rot in loader
                      for ids, msk, r in zip(ids_lote, msk_lote, rot))
    em_ordem = entrada.criar_loader(tok, ex, max_length=8, batch_size=4, embaralhar=False, seed=7)
    assert pares(lotes) == pares(em_ordem)       # embaralhar não separa exemplo e rótulo
    repetido = list(entrada.criar_loader(tok, ex, max_length=8, batch_size=4, embaralhar=True, seed=7))
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[2], b[2]) for a, b in zip(lotes, repetido))
    outra = list(entrada.criar_loader(tok, ex, max_length=8, batch_size=6, embaralhar=True, seed=8))
    mesma = list(entrada.criar_loader(tok, ex, max_length=8, batch_size=6, embaralhar=True, seed=7))
    assert not torch.equal(outra[0][0], mesma[0][0])


def test_loader_sem_embaralhar_mantem_a_ordem():
    pytest.importorskip("torch")
    from reclin import modelos
    tok = modelos.carregar_tokenizer(MODELO_MINUSCULO)
    ex = entrada.exemplos([doc()], ctx_chars=3)
    rotulos = [int(r) for _, _, lote in entrada.criar_loader(tok, ex, max_length=16, batch_size=4,
                                                            embaralhar=False, seed=1) for r in lote]
    assert rotulos == ex.rotulos
