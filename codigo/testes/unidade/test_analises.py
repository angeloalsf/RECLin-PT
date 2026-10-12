"""Análises descritivas (etapa 7): corpus, max_gap, alinhamento, marcadores e
falsos positivos, sobre documentos sintéticos (contas conferidas à mão) e, onde
indicado, sobre TRAIN e DEV congelados (nunca o TEST)."""
from __future__ import annotations

import pytest

from reclin import entrada, particoes, tarefa
from reclin.analises import alinhamento, corpus, falsos_positivos, marcadores, max_gap
from reclin.tarefa import ASSOC, NEG, NOREL
from reclin.util import caminhos

TEXTO = "paciente nega febre e tosse"
#        0        9    14    20 22


def doc(entidades, relacoes, texto=TEXTO, doc_id="d"):
    return {"doc_id": doc_id, "text": texto, "relations": relacoes,
            "entities": [{"id": i, "start": s, "end": e, "text": texto[s:e] if t is None else t, "type": tipo}
                         for i, s, e, t, tipo in entidades]}


DOC = doc([("1", 9, 13, None, "Neg"), ("2", 14, 19, None, "Sign"), ("3", 22, 27, None, "Sign")],
          [{"e1_id": "1", "e2_id": "2", "type": "negation_of"}, {"e1_id": "1", "e2_id": "3", "type": "negation_of"},
           {"e1_id": "2", "e2_id": "3", "type": "associated_with"}])


def test_faixas():
    assert [corpus.faixa(g) for g in (0, 1, 5, 6, 25, 26, 100, 101)] == \
        ["0", "1-5", "1-5", "6-10", "11-25", "26-50", "51-100", ">100"]
    assert corpus.rotulos_faixas() == ["0", "1-5", "6-10", "11-25", "26-50", "51-100", ">100"]


def test_estatisticas_de_um_documento():
    e = corpus.estatisticas_particao([DOC], max_gap=5)
    assert e["n_documentos"] == 1 and e["n_entidades"] == 3
    assert e["relacoes_por_tipo"] == {"associated_with": 1, "negation_of": 2}
    # pares com gap <= 5: (1,2),(2,1),(2,3),(3,2); (1,3) tem gap 9
    assert e["candidatos_por_rotulo"] == {"negation_of": 1, "associated_with": 1, "no_relation": 2}
    assert e["cobertura_relacoes"]["negation_of"] == {"cobertas": 1, "total": 2, "fracao": 0.5}
    assert e["relacoes_fora_dos_candidatos"]["acima_de_max_gap"] == 1
    assert e["gap_relacoes"]["negation_of"]["faixas"]["1-5"] == 1
    assert e["gap_relacoes"]["negation_of"]["faixas"]["6-10"] == 1


def test_estatisticas_batem_com_a_tarefa(documentos):
    """No TRAIN e no DEV congelados, os candidatos por rótulo são os de
    `tarefa.candidatos`, e as relações por tipo são as do MANIFEST."""
    e = corpus.estatisticas({p: documentos[p] for p in ("train", "dev")})
    manifesto = particoes.ler_manifesto(caminhos.PARTICOES_V1)      # a fixture `documentos` é a v1
    for p in ("train", "dev"):
        assert e[p]["candidatos_por_rotulo"] == tarefa.contagem_por_rotulo(tarefa.candidatos(documentos[p]))
        assert e[p]["n_candidatos"] == tarefa.TAMANHOS_ESPERADOS[p]
        assert e[p]["relacoes_por_tipo"] == manifesto["splits"][p]["n_relations"]


def test_max_gap_usa_a_enumeracao_da_tarefa():
    s = max_gap.sensibilidade([DOC], [0, 5, 10])
    linhas = {l["max_gap"]: l for l in s["linhas"]}
    for g, linha in linhas.items():
        assert linha["candidatos_por_rotulo"] == tarefa.contagem_por_rotulo(tarefa.candidatos([DOC], max_gap=g))
    assert linhas[10]["cobertura"]["negation_of"]["fracao"] == 1.0
    assert linhas[5]["cobertura"]["negation_of"]["fracao"] == 0.5
    with pytest.raises(ValueError):
        max_gap.sensibilidade([DOC], [-1])


def test_max_gap_da_tarefa_no_dev(documentos):
    s = max_gap.sensibilidade(documentos["dev"], [tarefa.MAX_GAP])
    assert s["linhas"][0]["n_candidatos"] == tarefa.TAMANHOS_ESPERADOS["dev"] and s["linhas"][0]["da_tarefa"]


def test_alinhamento():
    texto = "a  febre alta &gt; 38"
    assert alinhamento.alinhar_entidade(texto, {"start": 3, "end": 8, "text": "febre"})["status"] == "exato"
    assert alinhamento.alinhar_entidade(texto, {"start": 1, "end": 6, "text": "febre"}) == \
        {"status": "deslocado", "delta": 2}
    assert alinhamento.alinhar_entidade(texto, {"start": 9, "end": 21, "text": "alta > 38"})["status"] == \
        "equivalente"
    assert alinhamento.alinhar_entidade(texto, {"start": 0, "end": 1, "text": "tosse"})["status"] == "ausente"
    r = alinhamento.auditar([doc([("1", 9, 13, None, "x"), ("2", 10, 14, "nega", "x")], [])])
    assert r["status"] == {"exato": 1, "equivalente": 0, "deslocado": 1, "ausente": 0}
    assert r["deslocamentos"] == {"-1": 1}


def test_situacao_das_janelas():
    e1, e2 = {"start": 9, "end": 13}, {"start": 14, "end": 19}
    assert marcadores.situacao_janela(entrada.janela_marcada(TEXTO, e1, e2)) == "ok"
    aninhados = entrada.janela_marcada(TEXTO, {"start": 9, "end": 27}, {"start": 14, "end": 19})
    assert marcadores.situacao_janela(aninhados) == "ok"
    # mesmo fim: os fechamentos saem na ordem [/E1] [/E2] (a convenção da janela), e os pares se cruzam
    mesmo_fim = entrada.janela_marcada(TEXTO, {"start": 9, "end": 19}, {"start": 14, "end": 19})
    assert marcadores.situacao_janela(mesmo_fim) == "entrelacados"
    cruzados = entrada.janela_marcada(TEXTO, {"start": 9, "end": 16}, {"start": 14, "end": 19})
    assert marcadores.situacao_janela(cruzados) == "entrelacados"
    assert marcadores.situacao_janela("sem marcadores") == "invalida"
    r = marcadores.auditar_janelas([DOC])
    assert sum(sum(v.values()) for v in r.values()) == len(tarefa.candidatos([DOC]))


def test_marcadores_depois_da_tokenizacao():
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + list("abcdefghijklmnopqrstuvwxyz"))}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    rapido = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    with pytest.raises(ValueError, match="marcadores"):
        marcadores.auditar_tokenizacao(rapido, ["a"], [0], max_length=8)
    rapido.add_special_tokens({"additional_special_tokens": list(entrada.MARCADORES)})
    textos = ["[E1] a [/E1] [E2] b [/E2]", "a b c d e f [E1] a [/E1] [E2] b [/E2]"]
    r = marcadores.auditar_tokenizacao(rapido, textos, [NEG, NOREL], max_length=6)
    assert r["sem_e1"] == {"negation_of": 0, "associated_with": 0, "no_relation": 1}
    assert r["sem_e2"]["no_relation"] == 1 and r["acima_de_max_length"] == 1


def test_falsos_positivos():
    cands = tarefa.candidatos([DOC])
    y = tarefa.y_true(cands)
    pred = [NEG] * len(y)                         # tudo negation_of
    sidecar = {"y_true": y, "y_pred": pred}
    r = falsos_positivos.auditar(cands, sidecar, {"d": DOC}, e_pista=lambda e: e["text"] == "nega")
    resumo = r["resumo"]
    assert resumo["tp"] == y.count(NEG) and resumo["fp"] == len(y) - y.count(NEG)
    assert resumo["por_origem"] == {"associated_with": y.count(ASSOC), "no_relation": y.count(NOREL)}
    assert resumo["e1_pista"]["sim"] + resumo["e1_pista"]["nao"] == resumo["fp"]
    assert all(f["alinhamento"] == ["exato", "exato"] for f in r["exemplos"])
    with pytest.raises(ValueError, match="outro conjunto"):
        falsos_positivos.auditar(cands, {"y_true": y[::-1], "y_pred": pred}, {"d": DOC})


def test_comandos_de_analise(particoes_sinteticas, tmp_path):
    import json
    import subprocess
    import sys

    def rodar(*args):
        return subprocess.run([sys.executable, str(caminhos.CODIGO / "scripts" / "analisar.py"), *map(str, args)],
                              capture_output=True, text=True)
    base = ["--particoes", particoes_sinteticas]
    for comando in ("corpus", "max-gap", "alinhamento", "marcadores"):
        r = rodar(comando, *base, "--saida", tmp_path / f"{comando}.json")
        assert r.returncode == 0, r.stdout + r.stderr
        gravado = json.loads((tmp_path / f"{comando}.json").read_text("utf-8"))
        assert gravado["origem"]["particoes_congeladas"] is False
    assert json.loads((tmp_path / "max-gap.json").read_text("utf-8"))["origem"]["particoes"] == ["train", "dev"]
    r = rodar("max-gap", *base, "--particao", "test")
    assert r.returncode != 0 and "--incluir-test" in r.stderr
    assert rodar("max-gap", *base, "--incluir-test").returncode == 0
