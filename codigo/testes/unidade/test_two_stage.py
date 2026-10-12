"""Two-Stage e baseline calibrado: rótulos, dados de cada estágio, métricas,
configurações e regras (A2, A7, A14, A19).

Os testes que leem as partições usam só o TRAIN e o DEV congelados.
"""
from __future__ import annotations

import numpy as np
import pytest

from reclin import entrada, particoes
from reclin.avaliacao import limiar
from reclin.config import Config
from reclin.estrategias import baseline_calibrado, two_stage
from reclin.estrategias.two_stage import estagios
from reclin.estrategias.two_stage import metricas as m2
from reclin.tarefa import ASSOC, NEG, NOREL
from reclin.util import caminhos
from reclin.util.io import sha256_json


@pytest.fixture(scope="module")
def completos():
    # as janelas de referência (`referencia_dados`, do legado) são as da v1 dos dados
    return {p: entrada.exemplos(particoes.ler_particao(caminhos.PARTICOES_V1, p)) for p in ("train", "dev")}


def test_configuracoes_e_nomes():
    e1, e2 = two_stage.ConfigEstagio1(), two_stage.ConfigEstagio2()
    assert (e1.epochs, e2.epochs) == (3, 10)
    base = {k: v for k, v in Config().como_dict().items() if k not in ("tipo", "epochs")}
    for c in (e1, e2):       # só as épocas diferem da configuração base (semente 42, BioBERTpt)
        assert {k: v for k, v in c.como_dict().items() if k not in ("tipo", "epochs")} == base
    assert e1.seed == 42 and e1.encoder == "biobertpt" and e1.modelo == "pucpr/biobertpt-all"
    assert two_stage.nome_execucao(1, e1) == "twostage_e1_biobertpt_seed42"
    assert two_stage.nome_execucao(2, two_stage.ConfigEstagio2(seed=43)) == "twostage_e2_biobertpt_seed43"
    assert two_stage.nome_composicao(e1) == "twostage_biobertpt_seed42"
    assert two_stage.CAMPO_SELECAO == {1: "dev_ap_relacao", 2: "dev_negation_of_f1_relacoes_anotadas"}


def test_mapeamento_dos_rotulos():
    assert [two_stage.rotulo_estagio1(r) for r in (NEG, ASSOC, NOREL)] == [0, 0, 1]
    assert [two_stage.rotulo_estagio2(r) for r in (NEG, ASSOC)] == [0, 1]
    with pytest.raises(ValueError):
        two_stage.rotulo_estagio2(NOREL)
    assert two_stage.ROTULOS == {1: ("relacao", "no_relation"), 2: ("negation_of", "associated_with")}


@pytest.mark.parametrize("particao,esperado", [
    ("train", {1: (8351, 144335), 2: (1255, 7096)}),
    ("dev", {1: (913, 18151), 2: (150, 763)}),
])
def test_contagens_de_cada_estagio(completos, particao, esperado):
    for estagio in (1, 2):
        _, ex = estagios.exemplos_do_estagio(estagio, completos[particao])
        assert (ex.rotulos.count(0), ex.rotulos.count(1)) == esperado[estagio]


@pytest.mark.parametrize("particao", ["train", "dev"])
def test_janelas_sao_as_do_baseline(completos, referencia_dados, particao):
    indices1, e1 = estagios.exemplos_do_estagio(1, completos[particao])
    assert indices1 is None
    assert sha256_json(e1.textos) == referencia_dados["particoes"][particao]["janelas_sha256"]
    indices2, e2 = estagios.exemplos_do_estagio(2, completos[particao])
    assert e2.textos == [completos[particao].textos[i] for i in indices2]
    assert all(completos[particao].rotulos[i] != NOREL for i in indices2)
    assert indices2 == sorted(indices2)


def test_ap_como_no_sklearn():
    # exemplo da documentação do scikit-learn: AP = 0,8333...
    assert m2.ap([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.8333333333333333)
    assert m2.ap([1, 0, 1], [0.5, 0.5, 0.2]) == pytest.approx(0.5 * 0.5 + 0.5 * (2 / 3))   # empate agrupado
    assert m2.ap([0, 0], [0.3, 0.9]) == 0.0
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 200)
        s = np.round(rng.random(200), 2)          # com empates
        if y.sum():
            assert m2.ap(y, s) == pytest.approx(sklearn.average_precision_score(y, s))


def test_metricas_binarias():
    m = m2.binarias([0, 0, 1, 1], [0, 1, 1, 1], positivo=0)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)
    assert m["f1"] == pytest.approx(2 / 3) and m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_regra_do_baseline_calibrado():
    probs = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.4, 0.4], [0.45, 0.5, 0.05]]
    assert baseline_calibrado.prever(probs, 0.5) == [NEG, ASSOC, NOREL, ASSOC, ASSOC]   # empate -> associated_with
    assert baseline_calibrado.prever(probs, 0.45) == [NEG, ASSOC, NOREL, ASSOC, NEG]    # >= t
    assert baseline_calibrado.prever(probs, 0.05) == [NEG] * 5


def test_baseline_calibrado_usa_a_mesma_busca():
    y_true = [NEG, NOREL, NEG, ASSOC]
    probs = [[0.7, 0.2, 0.1], [0.35, 0.1, 0.55], [0.4, 0.3, 0.3], [0.1, 0.8, 0.1]]
    t, tabela = limiar.buscar(y_true, lambda t: baseline_calibrado.prever(probs, t))
    assert t == 0.4 and len(tabela) == len(limiar.GRADE)
    assert baseline_calibrado.nome_execucao("baseline_biobertpt_seed42") == "baseline_calibrado_biobertpt_seed42"
    assert limiar.hash_configuracao(baseline_calibrado.REGRA) != limiar.hash_configuracao("outra")


def test_laco_com_outro_campo_de_selecao(tmp_path):
    """`campo_selecao` escolhe a época pelo campo pedido (primeiro máximo
    estrito); sem ele, o padrão continua `dev_macro_f1`."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from reclin.treino import laco

    class Rede(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(2, 2)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.l(input_ids.float()))

        def save_pretrained(self, pasta):
            torch.save(self.state_dict(), f"{pasta}/pesos.pt")

    class Tok:
        def save_pretrained(self, pasta):
            pass

    gerador = torch.Generator().manual_seed(0)
    dados = [(torch.tensor([[1, 0], [0, 1]]), torch.ones(2, 2), torch.tensor([0, 1]))]

    class Loader(list):
        generator = gerador

    pontos = iter([{"x": 0.5, "dev_macro_f1": 0.9}, {"x": 0.7, "dev_macro_f1": 0.1},
                   {"x": 0.7, "dev_macro_f1": 0.2}])
    rede = Rede()
    estados = []                                  # os pesos ao fim de cada época

    def pontuar(_):
        estados.append({k: v.detach().clone() for k, v in rede.state_dict().items()})
        return next(pontos)

    otim = torch.optim.SGD(rede.parameters(), lr=0.1)
    agendador = torch.optim.lr_scheduler.LambdaLR(otim, lambda _: 1.0)
    r = laco.treinar(modelo=rede, tokenizer=Tok(), loader_treino=Loader(dados), loader_dev=Loader(dados),
                     perda=torch.nn.CrossEntropyLoss(), otimizador=otim, agendador=agendador, epocas=3,
                     max_grad_norm=1.0, dispositivo="cpu", pontuar_dev=pontuar,
                     pasta_execucao=tmp_path / "e", identidade={}, campo_selecao="x")
    assert laco.melhor_epoca(r.historico, "x") == (2, 0.7)       # empate entre 2 e 3 -> a primeira
    assert r.melhor_f1 == 0.7
    # os pesos guardados como melhores são os da época 2, não os da 3
    assert all(torch.equal(r.melhor_estado[k], estados[1][k]) for k in estados[1])
    assert not all(torch.equal(r.melhor_estado[k], estados[2][k]) for k in estados[2])
    salvos = torch.load(tmp_path / "e" / "checkpoints" / "melhor_modelo" / "pesos.pt")
    assert all(torch.equal(salvos[k], estados[1][k]) for k in estados[1])
    assert laco.melhor_epoca(r.historico) == (1, 0.9)            # o campo padrão não mudou


def test_pasta_so_com_train_dev_e_o_manifest_congelado(tmp_path):
    """A rodada de validação roda com uma pasta sem o TEST: com o MANIFEST
    congelado (byte a byte), o tamanho dos conjuntos continua conferido."""
    pytest.importorskip("torch")
    import shutil

    from reclin import tarefa
    from reclin.treino import classificador
    for nome in ("train.jsonl", "dev.jsonl", "MANIFEST.json"):
        shutil.copy(caminhos.PARTICOES / nome, tmp_path / nome)
    assert classificador.particoes_congeladas(caminhos.PARTICOES)
    assert classificador.particoes_congeladas(tmp_path)
    assert particoes.conferir_particoes(tmp_path, ("train", "dev")) == []
    docs = particoes.ler_particao(tmp_path, "dev")[:5]
    with pytest.raises(ValueError, match="19064"):       # o conjunto truncado é recusado, como na pasta congelada
        tarefa.conjunto_referencia("dev", docs, particao_sha256="x",
                                   conferir_tamanho=classificador.particoes_congeladas(tmp_path))
    (tmp_path / "MANIFEST.json").write_text((tmp_path / "MANIFEST.json").read_text() + " ")
    assert not classificador.particoes_congeladas(tmp_path)
