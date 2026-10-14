"""Estratégia `pair_aware_mlp` (cabeça MLP, todos os pares, sem léxico) e seu isolamento da `pair_aware`.

Unidade: configuração, montagem, cabeça sobre o modelo minúsculo (sem treino, sem rede, sem pesos reais, sem TEST,
sem Git) e o notebook `reclin_pair_aware_mlp.ipynb`. Escrito junto da estratégia e **não executado** pelo autor."""
from __future__ import annotations

import ast
import dataclasses
import json
import re

import pytest

torch = pytest.importorskip("torch")

from reclin import modelos  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.estrategias import pair_aware, pair_aware_mlp  # noqa: E402
from reclin.estrategias.pair_aware_mlp import cabeca  # noqa: E402
from reclin.treino import reprodutibilidade  # noqa: E402
from reclin.util.caminhos import CODIGO  # noqa: E402

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"
NOTEBOOK = CODIGO / "notebooks" / "reclin_pair_aware_mlp.ipynb"


def campos(classe):
    return {f.name: f.default for f in dataclasses.fields(classe)}


def test_configuracao_e_a_do_baseline_mais_a_cabeca():
    base, mlp = campos(Config), campos(pair_aware_mlp.ConfigPairAwareMLP)
    assert {k: v for k, v in mlp.items() if k not in base} == {"mlp_hidden": None, "head_dropout": None}
    assert all(mlp[k] == base[k] for k in base) and mlp["epochs"] == 3 and mlp["batch_size"] == 64
    c = pair_aware_mlp.ConfigPairAwareMLP()
    assert (c.modelo, c.encoder, c.seed) == (Config().modelo, "biobertpt", 42)
    for kw, msg in (({"mlp_hidden": 0}, "mlp_hidden"), ({"head_dropout": 1.0}, "head_dropout")):
        with pytest.raises(ValueError, match=msg):
            pair_aware_mlp.ConfigPairAwareMLP(**kw)


def test_montagem_sem_filtro_nem_lexico():
    m = pair_aware_mlp.montagem()
    assert m.estrategia == "pair_aware_mlp" and m.selecionar is None and m.extras_sidecar == {}
    assert m.classe_config is pair_aware_mlp.ConfigPairAwareMLP and m.classe_modelo is cabeca.PairAwareMLPClassifier
    r = m.registro
    assert r["filtro_lexical"] == "desativado" and r["cabeca"]["mlp"] is True
    assert r["selecao_melhor_epoca"] == "dev_macro_f1 no DEV completo"
    assert "formas" not in str(r) and "lexico_sha1" not in str(r)
    assert not hasattr(pair_aware_mlp, "selecionar") and not hasattr(pair_aware_mlp, "registro_do_lexico")


def test_montagem_reconstruida_do_registro_e_recusa_de_outra_versao():
    config = pair_aware_mlp.ConfigPairAwareMLP(encoder="bertimbau", seed=7, mlp_hidden=32)
    m = pair_aware_mlp.montagem(config)
    registro = {"estrategia": m.estrategia, "config": config.como_dict(), "estrategia_config": m.registro}
    assert pair_aware_mlp.montagem_de_registro(registro).registro == m.registro
    with pytest.raises(ValueError, match="outra versão"):
        pair_aware_mlp.montagem_de_registro({**registro, "estrategia_config": pair_aware.montagem().registro})


def test_isolamento_da_pair_aware_original():
    pa, mlp = pair_aware.montagem(), pair_aware_mlp.montagem()
    assert pa.estrategia != mlp.estrategia and pa.registro != mlp.registro      # outra chave de configuração
    assert pair_aware.nome_execucao(Config()) == "pairaware_biobertpt_seed42"
    assert pair_aware_mlp.nome_execucao(Config()) == "pairawaremlp_biobertpt_seed42"
    assert cabeca.CONFIG_FILE != pair_aware.cabeca.CONFIG_FILE and cabeca.HEAD_FILE != pair_aware.cabeca.HEAD_FILE
    assert "mlp_hidden" not in campos(pair_aware.ConfigPairAware)               # a original continua sem MLP
    assert pair_aware.montagem().registro["cabeca"]["mlp"] is False
    with pytest.raises(ValueError, match="outra configuração"):                 # e não aceita o registro da MLP
        pair_aware.montagem_de_registro({"estrategia_config": mlp.registro, "config": {}})


@pytest.fixture(scope="module")
def tokenizer():
    return modelos.carregar_tokenizer(MODELO_MINUSCULO)


def mlp(tokenizer, seed=42, **kw):
    reprodutibilidade.fixar_sementes(seed)
    return pair_aware_mlp.criar_modelo(str(MODELO_MINUSCULO), tokenizer, pair_aware_mlp.ConfigPairAwareMLP(**kw))


def test_arquitetura_com_mlp(tokenizer):
    rede = mlp(tokenizer)
    assert [type(c).__name__ for c in rede.head] == ["Linear", "GELU", "Dropout", "Linear"]
    assert (rede.head[0].in_features, rede.head[0].out_features, rede.head[3].out_features) == (48, 16, 3)
    assert rede.n_parametros_cabeca() == 48 * 16 + 16 + 16 * 3 + 3 and rede.head[2].p == 0.1
    assert mlp(tokenizer, mlp_hidden=8, head_dropout=0.3).pair_aware_mlp_config["mlp_hidden"] == 8


def test_cabeca_le_cls_e1_e2_e_faz_fallback(tokenizer):
    rede = mlp(tokenizer)
    rede.eval()
    lote = tokenizer(["[E1] nega [/E1] [E2] febre [/E2]", "nega febre"], padding=True, return_tensors="pt")
    with torch.no_grad():
        par, a1, a2 = rede.representacao_do_par(lote["input_ids"], lote["attention_mask"])
        h = rede.encoder(**lote).last_hidden_state
    e1, e2 = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])
    p1, p2 = (lote["input_ids"][0].tolist().index(i) for i in (e1, e2))
    assert torch.equal(par[0], torch.cat([h[0, 0], h[0, p1], h[0, p2]]))
    assert torch.equal(par[1], torch.cat([h[1, 0]] * 3)) and a1.tolist() == [True, False]


def test_salvar_e_recarregar_mesmos_logits_com_arquivos_proprios(tokenizer, tmp_path):
    rede = mlp(tokenizer)
    modelos.salvar(tmp_path / "m", rede, tokenizer)
    assert (tmp_path / "m" / cabeca.HEAD_FILE).is_file() and not (tmp_path / "m" / pair_aware.cabeca.HEAD_FILE).exists()
    tok2, rede2 = modelos.recarregar(tmp_path / "m", cabeca.PairAwareMLPClassifier)
    lote = tok2(["[E1] sem [/E1] [E2] dor [/E2]"], return_tensors="pt")
    rede.eval(); rede2.eval()
    with torch.no_grad():
        assert torch.equal(rede(**lote).logits, rede2(**lote).logits)
    assert rede2.pair_aware_mlp_config == rede.pair_aware_mlp_config


def test_notebook_da_estrategia_independente():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    celulas = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse("\n".join("pass" if l.lstrip().startswith("!") else l for l in "".join(c["source"]).splitlines()))
    codigo = "\n".join(celulas.values())
    assert "pair_aware_mlp.NOME" in celulas["c5a"] and "ConfigPairAwareMLP" in celulas["c5a"]
    assert "carregar_congelado" not in codigo and "pair_aware." not in codigo
    assert "'epochs': 3" in celulas["c7"] and "'batch_size': 64" in celulas["c7"] and "'mlp_hidden': None" in celulas["c7"]
    assert "reclin-pt-pairawaremlp-" in celulas["c6"] and "NATUREZA = 'EXPLORATÓRIA'" in celulas["c8"]
    for proibido in ("avaliar_test", "predicoes_test", "--particao test", "OP_TEST", "os.environ["):
        assert proibido not in codigo, proibido
    assert not re.search(r"(ghp_|github_pat_|hf_[A-Za-z0-9]{20})", codigo)
    assert "PUBLICAR_NO_GITHUB = False" in codigo and "PUBLICAR_NO_HF = False" in codigo
