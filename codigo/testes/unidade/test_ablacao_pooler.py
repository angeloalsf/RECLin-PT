"""Ablação do pooler: as duas células novas do fatorial marcadores × pooler (só software, modelo minúsculo)."""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from reclin import modelos  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.estrategias import ablacao_pooler, baseline, pair_aware  # noqa: E402
from reclin.estrategias.ablacao_pooler import cabeca  # noqa: E402
from reclin.estrategias.ablacao_pooler import cls_direto, pair_aware_pooler  # noqa: E402
from reclin.treino import reprodutibilidade  # noqa: E402
from reclin.util.caminhos import CODIGO  # noqa: E402

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"
FRASES = ["[E1] nega [/E1] [E2] febre [/E2]", "paciente [E2] dor [/E2] , [E1] sem [/E1] queixas", "nega febre"]


@pytest.fixture(scope="module")
def tokenizer():
    return modelos.carregar_tokenizer(MODELO_MINUSCULO)


def rede(variante, tokenizer, seed=42, **kw):
    reprodutibilidade.fixar_sementes(seed)
    return variante.criar_modelo(str(MODELO_MINUSCULO), tokenizer, ablacao_pooler.ConfigAblacao(**kw))


def test_configuracao_e_a_da_pair_aware_linear_e_nomes_sao_proprios():
    campos = lambda c: {f.name: f.default for f in dataclasses.fields(c)}                       # noqa: E731
    assert campos(ablacao_pooler.ConfigAblacao) == campos(pair_aware.ConfigPairAware)
    assert [v.NOME for v in ablacao_pooler.VARIANTES] == ["cls_direto", "pair_aware_pooler"]
    nomes = {cls_direto.nome_execucao(Config()), pair_aware_pooler.nome_execucao(Config()),
             pair_aware.nome_execucao(Config()), baseline.nome_execucao(Config())}
    assert nomes == {"clsdireto_biobertpt_seed42", "pairawarepooler_biobertpt_seed42",
                     "pairaware_biobertpt_seed42", "baseline_biobertpt_seed42"}
    assert cabeca.CONFIG_FILE != pair_aware.cabeca.CONFIG_FILE and cabeca.HEAD_FILE != pair_aware.cabeca.HEAD_FILE


def test_montagem_sem_filtro_nem_lexico_e_registro_distingue_as_celulas():
    m1, m2 = cls_direto.montagem(), pair_aware_pooler.montagem()
    assert (m1.estrategia, m2.estrategia) == ("cls_direto", "pair_aware_pooler")
    assert m1.selecionar is None and m2.selecionar is None and m1.registro != m2.registro
    assert m1.registro["filtro_lexical"] == "desativado" and "nao utilizado" in m1.registro["lexico"]
    assert (m1.registro["cabeca"]["ler_marcadores"], m1.registro["cabeca"]["usar_pooler"]) == (False, False)
    assert (m2.registro["cabeca"]["ler_marcadores"], m2.registro["cabeca"]["usar_pooler"]) == (True, True)
    assert m1.registro != pair_aware.montagem().registro
    with pytest.raises(ValueError, match="outra versão"):
        cls_direto.montagem_de_registro({"estrategia_config": m2.registro, "config": {}})
    with pytest.raises(ValueError, match="outra configuração"):                     # a Pair-Aware não aceita a ablação
        pair_aware.montagem_de_registro({"estrategia_config": m2.registro, "config": {}})
    cfg = dataclasses.asdict(ablacao_pooler.ConfigAblacao())
    assert pair_aware_pooler.montagem_de_registro({"estrategia_config": m2.registro, "config": cfg}).registro == m2.registro


def test_arquitetura_das_duas_celulas(tokenizer):
    a, b = rede(cls_direto, tokenizer), rede(pair_aware_pooler, tokenizer)
    d = a.config.hidden_size
    assert a.encoder.pooler is None and b.encoder.pooler is not None
    assert [type(c).__name__ for c in a.head] == [type(c).__name__ for c in b.head] == ["Dropout", "Linear"]
    assert (a.head[1].in_features, b.head[1].in_features) == (d, 3 * d)
    assert a.n_parametros_cabeca() == d * 3 + 3 and b.n_parametros_cabeca() == 3 * d * 3 + 3
    assert a.ablacao_config["pooler_do_checkpoint"] is None and isinstance(b.ablacao_config["pooler_do_checkpoint"], bool)
    assert "sem pooler" in a.ablacao_config["arquitetura"] and "pooler(h_cls)" in b.ablacao_config["arquitetura"]


def test_o_que_cada_celula_le(tokenizer):
    a, b = rede(cls_direto, tokenizer), rede(pair_aware_pooler, tokenizer)
    a.eval(); b.eval()
    lote = tokenizer(FRASES, padding=True, return_tensors="pt")
    e1, e2 = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])
    with torch.no_grad():
        ra = a.representacao(lote["input_ids"], lote["attention_mask"])
        assert torch.equal(ra, a.encoder(**lote).last_hidden_state[:, 0])           # só o h_cls, sem pooler
        rb = b.representacao(lote["input_ids"], lote["attention_mask"])
        saida = b.encoder(**lote)
    h, d = saida.last_hidden_state, b.config.hidden_size
    assert torch.equal(rb[:, :d], saida.pooler_output)                               # o [CLS] passa pelo pooler
    for i in (0, 1):                                                                 # os marcadores não passam
        p1, p2 = (lote["input_ids"][i].tolist().index(t) for t in (e1, e2))
        assert torch.equal(rb[i, d:], torch.cat([h[i, p1], h[i, p2]]))
    assert torch.equal(rb[2, d:], torch.cat([h[2, 0], h[2, 0]]))                     # marcador ausente -> h_cls


def test_as_outras_duas_combinacoes_sao_o_baseline_e_a_pair_aware(tokenizer):
    """Com os mesmos pesos, a cabeça da ablação reproduz as duas estratégias existentes."""
    lote = tokenizer(FRASES, padding=True, return_tensors="pt")
    e1, e2 = tokenizer.convert_tokens_to_ids(["[E1]", "[E2]"])

    reprodutibilidade.fixar_sementes(42)
    base = modelos.carregar_classificador(MODELO_MINUSCULO, tokenizer)
    como_base = cabeca.CabecaAblacao(base.bert, ler_marcadores=False, usar_pooler=True, e1_token_id=e1, e2_token_id=e2)
    como_base.head[1].load_state_dict(base.classifier.state_dict())
    base.eval(); como_base.eval()
    with torch.no_grad():
        assert torch.allclose(como_base(**lote).logits, base(**lote).logits, atol=1e-6)

    reprodutibilidade.fixar_sementes(42)
    pa = pair_aware.criar_modelo(str(MODELO_MINUSCULO), tokenizer, pair_aware.ConfigPairAware())
    como_pa = cabeca.CabecaAblacao(pa.encoder, ler_marcadores=True, usar_pooler=False, e1_token_id=e1, e2_token_id=e2)
    como_pa.head.load_state_dict(pa.head.state_dict())
    pa.eval(); como_pa.eval()
    with torch.no_grad():
        assert torch.equal(como_pa(**lote).logits, pa(**lote).logits)


@pytest.mark.parametrize("variante", ablacao_pooler.VARIANTES, ids=lambda v: v.NOME)
def test_salvar_e_recarregar_mesmos_logits(variante, tokenizer, tmp_path):
    r = rede(variante, tokenizer)
    modelos.salvar(tmp_path / "m", r, tokenizer)
    assert (tmp_path / "m" / cabeca.HEAD_FILE).is_file() and not (tmp_path / "m" / pair_aware.cabeca.HEAD_FILE).exists()
    tok2, r2 = modelos.recarregar(tmp_path / "m", cabeca.CabecaAblacao)
    lote = tok2(FRASES, padding=True, return_tensors="pt")
    r.eval(); r2.eval()
    with torch.no_grad():
        assert torch.equal(r(**lote).logits, r2(**lote).logits)
    assert (r2.ler_marcadores, r2.usar_pooler) == (variante.ler_marcadores, variante.usar_pooler)


def test_o_pacote_nao_importa_torch_e_os_scripts_conhecem_as_variantes():
    import subprocess
    import sys
    codigo = ("import sys; from reclin.estrategias import ablacao_pooler as a; a.cls_direto.registro(); "
              "assert 'torch' not in sys.modules, 'torch importado'")
    assert subprocess.run([sys.executable, "-c", codigo], cwd=CODIGO, env={"PYTHONPATH": str(CODIGO), "PATH": ""}).returncode == 0
    for script in ("treinar.py", "avaliar_test.py"):
        assert "ablacao_pooler.VARIANTES" in (CODIGO / "scripts" / script).read_text(encoding="utf-8")


@pytest.mark.parametrize("variante", ablacao_pooler.VARIANTES, ids=lambda v: v.NOME)
def test_notebook_da_ablacao(variante):
    import ast
    import json
    nb = json.loads((CODIGO / "notebooks" / "ablacao" / f"reclin_{variante.NOME}.ipynb").read_text(encoding="utf-8"))
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    celulas = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse("\n".join("pass" if l.lstrip().startswith("!") else l for l in "".join(c["source"]).splitlines()))
    codigo = "\n".join(celulas.values())
    assert f"{variante.NOME}.NOME" in celulas["c5a"] and "ConfigAblacao" in celulas["c5a"]
    assert "pair_aware_mlp" not in codigo and "mlp_hidden" not in codigo and "carregar_congelado" not in codigo
    assert "'epochs': 3" in celulas["c7"] and "'batch_size': 64" in celulas["c7"] and "'class_weight': 'balanced'" in celulas["c7"]
    assert "assert not DIFERENCAS" in celulas["c7"]                       # mesmos hiperparâmetros da Pair-Aware linear
    assert f"== '{variante.NOME}'" in celulas["c8"] and "NATUREZA = 'EXPLORATÓRIA'" in celulas["c8"]
    assert "comparar_pareado" in celulas["c8d"] and "dev.jsonl" in celulas["c8d"]
    for proibido in ("OP_TEST", "avaliar_test", "predicoes_test", "ler_particao"):   # o notebook não avalia nem carrega o TEST
        assert proibido not in codigo
    assert "PUBLICAR_NO_GITHUB = False" in celulas["c10"] and "PUBLICAR_NO_HF = False" in celulas["c11"]
