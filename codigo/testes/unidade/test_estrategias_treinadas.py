"""Estratégias treinadas (`baseline`, `restrito`, `pair_aware`) sobre exemplos
pequenos: configurações, seleção dos pares, montagem e a cabeça Pair-Aware."""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from reclin import entrada, modelos  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.estrategias import baseline, pair_aware, restrito  # noqa: E402
from reclin.estrategias.pair_aware import cabeca  # noqa: E402
from reclin.negacao.lexico import lexico_sha1  # noqa: E402
from reclin.tarefa import candidatos  # noqa: E402
from reclin.treino import reprodutibilidade  # noqa: E402
from reclin.treino.montagem import CLASSIFICADOR  # noqa: E402
from reclin.util.caminhos import CODIGO  # noqa: E402

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"
LEX = {"nega": 3, "sem": 2}


def ent(id_, start, end, texto):
    return {"id": id_, "start": start, "end": end, "text": texto, "type": "t"}


DOC = {"doc_id": "d", "text": "NEGA febre sem dor e tosse",
       "entities": [ent("1", 0, 4, "NEGA"), ent("2", 5, 10, "febre"), ent("3", 11, 14, "sem"),
                    ent("4", 15, 18, "dor")],
       "relations": [{"e1_id": "1", "e2_id": "2", "type": "negation_of"}]}
CANDS = candidatos([DOC])


# --------------------------------------------------------------------------- #
# Configurações                                                               #
# --------------------------------------------------------------------------- #
def campos(classe):
    return {f.name: f.default for f in dataclasses.fields(classe)}


def test_configuracoes_diferem_so_no_que_declaram():
    base, rest, pa = campos(Config), campos(restrito.ConfigRestrito), campos(pair_aware.ConfigPairAware)
    assert {k for k in base if base[k] != rest[k]} == {"epochs"} and rest["epochs"] == 10
    # a Pair-Aware difere do restrito só nos campos da cabeça
    assert {k: v for k, v in pa.items() if k not in rest} == {"mlp_hidden": None, "head_dropout": None}
    assert all(pa[k] == rest[k] for k in rest)


@pytest.mark.parametrize("kw,mensagem", [({"mlp_hidden": 0}, "mlp_hidden"),
                                         ({"head_dropout": 1.0}, "head_dropout"),
                                         ({"lr": 0}, "lr")])
def test_config_pair_aware_valida(kw, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        pair_aware.ConfigPairAware(**kw)


def test_nomes_das_execucoes_seguem_o_legado():
    c = Config(encoder="bertimbau", seed=43)
    assert baseline.nome_execucao(c) == "baseline_bertimbau_seed43"
    assert restrito.nome_execucao(c) == "restrito_bertimbau_seed43"
    assert pair_aware.nome_execucao(c) == "pairaware_bertimbau_seed43"


# --------------------------------------------------------------------------- #
# Montagens                                                                   #
# --------------------------------------------------------------------------- #
def test_baseline_e_o_classificador_com_outro_nome():
    m = baseline.montagem(Config())
    assert m.estrategia == "baseline" and dataclasses.replace(m, estrategia="classificador") == CLASSIFICADOR
    assert m.selecionar is None and m.registro == {} and m.extras_sidecar == {}


@pytest.mark.parametrize("modulo", [restrito, pair_aware])
def test_selecao_pelo_e1_em_ordem(modulo):
    selecionados = modulo.selecionar(CANDS, LEX)
    assert selecionados == sorted(selecionados)
    assert {(CANDS[i]["e1"]["text"]) for i in selecionados} == {"NEGA", "sem"}
    assert len(selecionados) == sum(1 for c in CANDS if c["e1"]["text"] in ("NEGA", "sem"))
    # a pista em e2 não basta
    assert all(CANDS[i]["e1"]["text"] in ("NEGA", "sem") for i in selecionados)


@pytest.mark.parametrize("modulo,classe", [(restrito, restrito.ConfigRestrito),
                                           (pair_aware, pair_aware.ConfigPairAware)])
def test_montagem_reconstruida_do_registro(modulo, classe):
    config = classe(encoder="bertimbau", seed=7)
    m = modulo.montagem(config, lexico=LEX)
    registro = {"estrategia": m.estrategia, "config": config.como_dict(), "estrategia_config": m.registro}
    r = modulo.montagem_de_registro(registro)
    assert r.registro == m.registro and r.extras_sidecar == m.extras_sidecar
    assert r.extras_sidecar["encoder"] == "bertimbau"
    assert r.selecionar(CANDS) == m.selecionar(CANDS)
    assert m.registro["lexico"]["lexico_sha1"] == lexico_sha1(LEX)
    adulterado = {**registro, "estrategia_config": {**m.registro, "lexico": {**m.registro["lexico"],
                                                                            "formas": {"nega": 3}}}}
    with pytest.raises(ValueError, match="lexico_sha1"):
        modulo.montagem_de_registro(adulterado)


def test_extras_dos_sidecars():
    assert restrito.montagem(restrito.ConfigRestrito(), lexico=LEX).extras_sidecar == {
        "espaco": "restrito remapeado ao completo", "encoder": "biobertpt"}
    assert pair_aware.montagem(pair_aware.ConfigPairAware(), lexico=LEX).extras_sidecar == {
        "espaco": "restrito remapeado ao completo", "cabeca": "pair-aware", "encoder": "biobertpt"}


# --------------------------------------------------------------------------- #
# Cabeça Pair-Aware                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tokenizer():
    return modelos.carregar_tokenizer(MODELO_MINUSCULO)


def pa(tokenizer, seed=42, **kw):
    reprodutibilidade.fixar_sementes(seed)
    return pair_aware.criar_modelo(str(MODELO_MINUSCULO), tokenizer, pair_aware.ConfigPairAware(**kw))


def test_posicoes_dos_marcadores():
    ids = torch.tensor([[5, 9, 7, 9, 0], [5, 6, 6, 0, 0], [5, 6, 0, 9, 0]])
    mascara = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    pos, achou = cabeca.posicoes_dos_marcadores(ids, mascara, 9)
    assert pos.tolist() == [1, 0, 0]               # primeira ocorrência; ausente -> 0
    assert achou.tolist() == [True, False, False]  # o 9 da terceira linha está no padding


def test_cabeca_le_cls_e1_e2(tokenizer):
    rede = pa(tokenizer)
    rede.eval()
    lote = tokenizer(["[E1] nega [/E1] [E2] febre [/E2]", "nega febre"], padding=True, return_tensors="pt")
    with torch.no_grad():
        par, achou1, achou2 = rede.representacao_do_par(lote["input_ids"], lote["attention_mask"])
        h = rede.encoder(**lote).last_hidden_state
    e1, e2 = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])
    p1 = lote["input_ids"][0].tolist().index(e1)
    p2 = lote["input_ids"][0].tolist().index(e2)
    assert torch.equal(par[0], torch.cat([h[0, 0], h[0, p1], h[0, p2]]))
    assert torch.equal(par[1], torch.cat([h[1, 0], h[1, 0], h[1, 0]]))       # fallback h_cls
    assert achou1.tolist() == [True, False] and achou2.tolist() == [True, False]


def test_arquitetura_e_configuracao(tokenizer):
    rede = pa(tokenizer)
    camadas = list(rede.head)
    assert [type(c).__name__ for c in camadas] == ["Linear", "GELU", "Dropout", "Linear"]
    assert (camadas[0].in_features, camadas[0].out_features, camadas[3].out_features) == (48, 16, 3)
    assert camadas[2].p == 0.1 and not rede.pair_aware_config["tem_pooler"]
    assert not hasattr(rede.encoder, "pooler") or rede.encoder.pooler is None
    assert pa(tokenizer, mlp_hidden=8, head_dropout=0.3).pair_aware_config["mlp_hidden"] == 8


def test_inicializacao_deterministica(tokenizer):
    a, b, c = pa(tokenizer, 1), pa(tokenizer, 1), pa(tokenizer, 2)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not torch.equal(a.head[0].weight, c.head[0].weight)


def test_salvar_e_recarregar_mesmos_logits(tokenizer, tmp_path):
    rede = pa(tokenizer)
    modelos.salvar(tmp_path / "m", rede, tokenizer)
    tok2, rede2 = modelos.recarregar(tmp_path / "m", cabeca.PairAwareClassifier)
    lote = tok2(["[E1] sem [/E1] [E2] dor [/E2]"], return_tensors="pt")
    rede.eval(); rede2.eval()
    with torch.no_grad():
        assert torch.equal(rede(**lote).logits, rede2(**lote).logits)
    assert rede2.pair_aware_config == rede.pair_aware_config


def test_marcador_que_nao_e_token(tokenizer):
    from transformers import AutoTokenizer
    sem_marcadores = AutoTokenizer.from_pretrained(str(MODELO_MINUSCULO))
    with pytest.raises(ValueError, match="add_special_tokens"):
        cabeca.ids_dos_marcadores(sem_marcadores)


def test_contar_marcadores_ausentes(tokenizer):
    textos = ["[E1] nega [/E1] [E2] dor [/E2]",            # completo
              "[E1] nega [/E1] febre",                       # E2 fora do texto
              "[E1] nega " + "a " * 40 + "[/E1] [E2] dor [/E2]"]   # E2 truncado
    r = cabeca.contar_marcadores_ausentes(tokenizer, textos, max_length=16)
    assert (r["n"], r["E1_ausente"], r["E2_ausente"], r["algum_ausente"]) == (3, 0, 2, 2)
    assert (r["E2_fora_do_texto"], r["E2_truncado"]) == (1, 1)


def test_descrever_mede_os_marcadores(tokenizer):
    config = pair_aware.ConfigPairAware(max_length=16)
    m = pair_aware.montagem(config, lexico=LEX)
    rede = pa(tokenizer)
    ex = entrada.Exemplos(["[E1] nega [/E1] febre"], [0])
    d = m.descrever(tokenizer, rede, {"dev": ex})
    assert d["marcadores_ausentes"]["dev"]["E2_fora_do_texto"] == 1
    assert d["n_params_cabeca"] == 48 * 16 + 16 + 16 * 3 + 3 and d["cabeca"] == rede.pair_aware_config
