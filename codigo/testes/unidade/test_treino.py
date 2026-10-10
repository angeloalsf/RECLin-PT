"""Peças do treino em CPU com o modelo minúsculo: modelos, reprodutibilidade,
checkpoint e as funções do laço."""
from __future__ import annotations

import os
import random
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reclin import entrada, modelos  # noqa: E402
from reclin.tarefa import ID2LABEL, LABELS  # noqa: E402
from reclin.treino import checkpoint, laco, reprodutibilidade  # noqa: E402
from reclin.util.caminhos import CODIGO  # noqa: E402

MODELO_MINUSCULO = CODIGO / "testes" / "referencia" / "modelo_minusculo"


@pytest.fixture(scope="module")
def tokenizer():
    return modelos.carregar_tokenizer(MODELO_MINUSCULO)


def classificador(tokenizer, seed=0):
    reprodutibilidade.fixar_sementes(seed)
    return modelos.carregar_classificador(MODELO_MINUSCULO, tokenizer)


# --------------------------------------------------------------------------- #
# modelos                                                                     #
# --------------------------------------------------------------------------- #
def test_tokenizer_com_os_quatro_marcadores(tokenizer):
    assert len(tokenizer) == 118 + 4
    ids = tokenizer.convert_tokens_to_ids(list(entrada.MARCADORES))
    assert ids == [118, 119, 120, 121]
    codificado = tokenizer("[E1] nega [/E1] [E2] febre [/E2]")["input_ids"]
    assert all(i in codificado for i in ids)          # cada marcador é um token só


def test_classificador_de_tres_rotulos(tokenizer):
    rede = classificador(tokenizer)
    assert rede.config.num_labels == 3 and rede.config.id2label == ID2LABEL
    assert rede.get_input_embeddings().weight.shape[0] == len(tokenizer)
    assert modelos.contar_parametros(rede) == 14979


def test_inicializacao_depende_so_da_semente(tokenizer):
    a, b, c = classificador(tokenizer, 1), classificador(tokenizer, 1), classificador(tokenizer, 2)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not torch.equal(a.classifier.weight, c.classifier.weight)


def test_salvar_e_recarregar(tokenizer, tmp_path):
    rede = classificador(tokenizer)
    pasta = modelos.salvar(tmp_path / "m", rede, tokenizer)
    assert not (tmp_path / "m.tmp").exists()
    tok2, rede2 = modelos.recarregar(pasta)
    assert len(tok2) == len(tokenizer)
    lote = tokenizer(["[E1] nega [/E1] [E2] dor [/E2]"], return_tensors="pt")
    rede.eval(); rede2.eval()
    with torch.no_grad():
        assert torch.equal(rede(**lote).logits, rede2(**lote).logits)
    assert modelos.identidade("x", tok2, rede2)["vocabulario_sha256"] == \
        modelos.identidade("x", tokenizer, rede)["vocabulario_sha256"]
    _, mesma_classe = modelos.recarregar(pasta, type(rede))
    assert type(mesma_classe) is type(rede)


def test_identidade(tokenizer):
    ident = modelos.identidade("m", tokenizer, classificador(tokenizer))
    assert ident["vocabulario"] == 122 and ident["ids_marcadores"] == [118, 119, 120, 121]
    assert ident["n_parametros"] == 14979 and ident["classe"] == "BertForSequenceClassification"


# --------------------------------------------------------------------------- #
# reprodutibilidade                                                           #
# --------------------------------------------------------------------------- #
def test_configurar_ambiente_nao_sobrescreve(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    assert reprodutibilidade.configurar_ambiente() == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                                       "PYTHONHASHSEED": "7"}
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_estados_rng_restauram_os_quatro_geradores():
    reprodutibilidade.fixar_sementes(3)
    estados = reprodutibilidade.estados_rng()
    antes = (random.random(), np.random.rand(), torch.rand(1).item())
    reprodutibilidade.fixar_sementes(99)
    reprodutibilidade.restaurar_rng(estados)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == antes


def test_restaurar_rng_recusa_checkpoint_de_gpu_sem_gpu():
    estados = reprodutibilidade.estados_rng()
    if torch.cuda.is_available():
        pytest.skip("máquina com GPU")
    with pytest.raises(RuntimeError, match="GPU"):
        reprodutibilidade.restaurar_rng(dict(estados, cuda=[torch.zeros(1)]))


def test_coletar_avisos():
    with reprodutibilidade.coletar_avisos() as avisos:
        warnings.warn("op sem versão determinística", UserWarning)
        warnings.warn("op sem versão determinística", UserWarning)
    assert avisos == ["UserWarning: op sem versão determinística"]


def test_ambiente_registra_versoes():
    amb = reprodutibilidade.ambiente()
    assert amb["versoes"]["torch"] == torch.__version__ and "threads_torch" in amb


# --------------------------------------------------------------------------- #
# laço                                                                        #
# --------------------------------------------------------------------------- #
def test_pesos_balanced():
    # o rótulo ausente conta como 1, também na soma (como no legado)
    assert laco.pesos_balanced([0, 2, 2, 2]) == [5 / 3, 5 / 3, 5 / 9]
    assert laco.pesos_balanced([0, 1, 2]) == [1.0, 1.0, 1.0]


def test_melhor_epoca_primeiro_maximo_estrito():
    hist = [{"epoch": 1, "dev_macro_f1": 0.5}, {"epoch": 2, "dev_macro_f1": 0.7},
            {"epoch": 3, "dev_macro_f1": 0.7}]
    assert laco.melhor_epoca(hist) == (2, 0.7)
    assert laco.melhor_epoca([]) == (None, None)


def test_prever_e_avaliar_loss(tokenizer):
    rede = classificador(tokenizer)
    ex = entrada.Exemplos(["[E1] nega [/E1] [E2] dor [/E2]", "sem febre", "tosse"], [0, 1, 2])
    loader = entrada.criar_loader(tokenizer, ex, max_length=16, batch_size=2, embaralhar=False, seed=1)
    estado = torch.get_rng_state()
    preds, probs = laco.prever(rede, loader, "cpu", com_probs=True)
    assert len(preds) == 3 and all(abs(sum(p) - 1) < 1e-5 for p in probs)
    assert preds == [int(np.argmax(p)) for p in probs] == laco.prever(rede, loader, "cpu")
    preds2, loss = laco.avaliar_loss(rede, loader, "cpu", torch.nn.CrossEntropyLoss())
    assert preds2 == preds and loss > 0
    assert torch.equal(torch.get_rng_state(), estado)    # avaliar não consome o RNG global


# --------------------------------------------------------------------------- #
# checkpoint                                                                  #
# --------------------------------------------------------------------------- #
def estado_minimo(tokenizer):
    rede = classificador(tokenizer)
    opt = torch.optim.AdamW(rede.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    gerador = torch.Generator().manual_seed(1)
    return checkpoint.montar_estado(
        posicao={"epoca": 1, "passo_na_epoca": 3, "passo_global": 3, "epoca_concluida": False},
        modelo=rede, otimizador=opt, agendador=sched, rng=reprodutibilidade.estados_rng(),
        gerador_inicio_epoca=gerador.get_state(), gerador_atual=gerador.get_state(),
        soma_loss_epoca=1.5, historico=[], melhor_f1=-1.0, identidade={"config": {"seed": 1}},
        ambiente={})


def test_checkpoint_grava_e_le(tokenizer, tmp_path):
    assert checkpoint.carregar_estado(tmp_path) is None
    estado = estado_minimo(tokenizer)
    checkpoint.salvar_estado(tmp_path, estado)
    arquivos = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert arquivos == ["ultimo.json", "ultimo.pt"]               # sem .tmp sobrando
    lido = checkpoint.carregar_estado(tmp_path)
    assert lido["posicao"] == estado["posicao"] and lido["soma_loss_epoca"] == 1.5
    assert all(torch.equal(lido["modelo"][k], v) for k, v in estado["modelo"].items())
    assert lido["rng"]["python"] == estado["rng"]["python"]


def test_checkpoint_recusa_outra_identidade(tokenizer):
    estado = estado_minimo(tokenizer)
    checkpoint.conferir_identidade(estado, {"config": {"seed": 1}})
    with pytest.raises(checkpoint.ErroRetomada, match="config"):
        checkpoint.conferir_identidade(estado, {"config": {"seed": 2}})


def test_checkpoint_de_outro_formato(tokenizer, tmp_path):
    checkpoint.salvar_estado(tmp_path, dict(estado_minimo(tokenizer), formato=99))
    with pytest.raises(checkpoint.ErroRetomada, match="formato"):
        checkpoint.carregar_estado(tmp_path)


def test_melhor_modelo(tokenizer, tmp_path):
    rede = classificador(tokenizer)
    assert checkpoint.estado_do_melhor(tmp_path, rede) is None
    checkpoint.salvar_melhor(tmp_path, rede, tokenizer)
    estado = checkpoint.estado_do_melhor(tmp_path, rede)
    assert all(torch.equal(estado[k], v) for k, v in rede.state_dict().items())


def test_rotulos_na_ordem_da_tarefa():
    assert list(ID2LABEL.values()) == list(LABELS)
