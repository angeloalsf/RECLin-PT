"""Portão da etapa 5: entrada e treino iguais aos do legado.

Referências (geradas pelo código do legado em `referencia/gerar_referencias.py`):

* `dados.json` — SHA-256 das janelas marcadas das três partições completas e os
  pesos `balanced` do TRAIN;
* `lotes.json` — SHA-256 de cada lote (`input_ids`, `attention_mask`, rótulos)
  na ordem do DataLoader do legado, com o tokenizer minúsculo: TRAIN
  embaralhado nas épocas 1 e 2, DEV e TEST em ordem;
* `treino/baseline_pequeno_seed42.*` — o baseline do legado sobre um
  subconjunto (as primeiras 30/10/10 linhas), 3 épocas, instrumentado: loss de
  cada passo e de cada lote do DEV, hash dos parâmetros iniciais, dos
  parâmetros e do estado do otimizador ao fim de cada época, lr, parâmetros da
  melhor época, histórico, predições e métricas;
* `treino/baseline_minusculo_seed42.*` — o mesmo baseline sobre as partições
  completas, 2 épocas (cerca de 11 minutos em CPU): só roda com
  `RECLIN_TREINO_COMPLETO=1`.

Janelas e lotes são conferidos em qualquer ambiente. A reprodução numérica do
treino (loss, pesos, predições) só é igual bit a bit no ambiente em que as
referências foram geradas (Linux x86_64, torch 2.11.0, transformers 5.16.1,
uma thread); em outro ambiente esses testes são pulados com o motivo, e podem
ser forçados com `RECLIN_FORCAR_EQUIVALENCIA=1`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from reclin import entrada, modelos, particoes  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.treino import classificador, laco, reprodutibilidade  # noqa: E402
from reclin.util.caminhos import CODIGO, PARTICOES  # noqa: E402
from reclin.util.io import ler_json, sha256_json  # noqa: E402

REFERENCIA = CODIGO / "testes" / "referencia"
MODELO_MINUSCULO = REFERENCIA / "modelo_minusculo"
TREINO = REFERENCIA / "treino"

ESPERADO = {"sistema": "Linux", "arquitetura": "x86_64", "torch": "2.11.0", "transformers": "5.16.1"}
ATUAL = {"sistema": platform.system(), "arquitetura": platform.machine(),
         "torch": torch.__version__.split("+")[0], "transformers": transformers.__version__}
EXATO = pytest.mark.skipif(
    ATUAL != ESPERADO and os.environ.get("RECLIN_FORCAR_EQUIVALENCIA") != "1",
    reason=f"reprodução numérica do legado: referências geradas em {ESPERADO}, este ambiente é "
           f"{ATUAL}; os números podem diferir nos últimos bits (RECLIN_FORCAR_EQUIVALENCIA=1 roda assim mesmo)")


def h(obj) -> str:
    """A receita dos hashes de lotes.json."""
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode("utf-8")).hexdigest()


def hash_tensores(tensores) -> str:
    """A receita dos hashes de parâmetros do registro do treino pequeno."""
    m = hashlib.sha256()
    for t in tensores:
        t = t.detach().cpu().contiguous()
        m.update(str(tuple(t.shape)).encode())
        m.update(str(t.dtype).encode())
        m.update(t.numpy().tobytes())
    return m.hexdigest()


def hash_otimizador(opt) -> str:
    tensores = []
    for grupo in opt.param_groups:
        for p in grupo["params"]:
            estado = opt.state.get(p, {})
            tensores += [estado[k] for k in ("step", "exp_avg", "exp_avg_sq") if k in estado]
    return hash_tensores(tensores)


@pytest.fixture(scope="module")
def uma_thread():
    antes = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(antes)


# --------------------------------------------------------------------------- #
# A. Janelas e lotes                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nome", ["train", "dev", "test"])
def test_janelas_identicas(nome, referencia_dados, documentos):
    ref = referencia_dados["particoes"][nome]
    ex = entrada.exemplos(documentos[nome], max_gap=referencia_dados["max_gap"],
                          ctx_chars=referencia_dados["ctx_chars"])
    assert len(ex) == ref["n_candidatos"]
    assert sha256_json(ex.textos) == ref["janelas_sha256"]
    assert sha256_json(ex.rotulos) == ref["y_true_sha256"]


@pytest.fixture(scope="module")
def lotes_ref():
    return ler_json(REFERENCIA / "lotes.json")


@pytest.fixture(scope="module")
def tokenizer_minusculo():
    return modelos.carregar_tokenizer(MODELO_MINUSCULO)


def test_tokenizer_com_marcadores(lotes_ref, tokenizer_minusculo):
    assert len(tokenizer_minusculo) == lotes_ref["tokenizer"]["tamanho"]
    assert tokenizer_minusculo.convert_tokens_to_ids(list(entrada.MARCADORES)) == \
        lotes_ref["tokenizer"]["ids_marcadores"]


def conferir_lotes(nome, lotes_ref, tokenizer, documentos):
    ref = lotes_ref["particoes"][nome]
    ex = entrada.exemplos(documentos[nome], max_gap=lotes_ref["max_gap"], ctx_chars=lotes_ref["ctx_chars"])
    assert len(ex) == ref["n_exemplos"]
    loader = entrada.criar_loader(tokenizer, ex, max_length=lotes_ref["max_length"],
                                  batch_size=lotes_ref["batch_size"], embaralhar=ref["embaralhado"],
                                  seed=lotes_ref["seed"])
    for epoca in ref["epocas"]:
        lotes, n_tokens, maior = [], 0, 0
        for ids, mascara, rotulos in loader:
            lotes.append(h([ids.tolist(), mascara.tolist(), rotulos.tolist()]))
            n_tokens += int(mascara.sum())
            maior = max(maior, int(ids.shape[1]))
        assert len(lotes) == epoca["n_lotes"]
        assert lotes[:3] == epoca["primeiros_lotes"]
        assert (n_tokens, maior) == (epoca["n_tokens_reais"], epoca["maior_comprimento"])
        assert h(lotes) == epoca["lotes_sha256"]


@pytest.mark.parametrize("nome", ["dev", "test"])
def test_lotes_identicos(nome, lotes_ref, tokenizer_minusculo, documentos):
    conferir_lotes(nome, lotes_ref, tokenizer_minusculo, documentos)


@pytest.mark.lento
def test_lotes_do_train_identicos_nas_duas_epocas(lotes_ref, tokenizer_minusculo, documentos):
    """Ordem do embaralhamento e conteúdo dos 2 x 2.386 lotes (cerca de 2 min)."""
    conferir_lotes("train", lotes_ref, tokenizer_minusculo, documentos)


def test_pesos_balanced(referencia_dados, documentos):
    ex = entrada.exemplos(documentos["train"])
    assert dict(zip(("negation_of", "associated_with", "no_relation"), laco.pesos_balanced(ex.rotulos))) == \
        referencia_dados["pesos_balanced"]["espaco_completo"]


# --------------------------------------------------------------------------- #
# B. Treino pequeno                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def registro_ref():
    return ler_json(TREINO / "baseline_pequeno_seed42.registro.json")


@EXATO
def test_parametros_iniciais(registro_ref, uma_thread):
    reprodutibilidade.fixar_sementes(42)
    tok = modelos.carregar_tokenizer(MODELO_MINUSCULO)
    rede = modelos.carregar_classificador(MODELO_MINUSCULO, tok)
    assert hash_tensores(list(rede.parameters())) == registro_ref["parametros_iniciais"]


@pytest.fixture(scope="module")
def treino_pequeno(tmp_path_factory, subconjunto, uma_thread):
    """O treino pequeno pelo código novo, observado como o driver do legado
    observou o dele (sem mudar nada no cálculo)."""
    subconj = ler_json(REFERENCIA / "referencias.json")["etapa5"]["baseline_pequeno_seed42"]["subconjunto"]
    pasta_particoes = subconjunto(*(subconj[n]["primeiras_linhas"] for n in ("train", "dev", "test")))
    for nome in ("train", "dev", "test"):
        assert hashlib.sha256((pasta_particoes / f"{nome}.jsonl").read_bytes()).hexdigest() == subconj[nome]["sha256"]

    reg = {"loss_treino": [], "loss_dev": [], "epocas": [], "parametros_iniciais": None,
           "parametros_finais_melhor_epoca": None}
    opt = {}
    mp = pytest.MonkeyPatch()
    original_ce, original_adamw = torch.nn.CrossEntropyLoss, torch.optim.AdamW
    original_avaliar, original_prever = laco.avaliar_loss, laco.prever

    class CE(original_ce):
        def forward(self, a, b):
            r = super().forward(a, b)
            (reg["loss_treino"] if torch.is_grad_enabled() else reg["loss_dev"]).append(r.item())
            return r

    class AdamW(original_adamw):
        def __init__(self, params, *a, **k):
            params = list(params)
            reg["parametros_iniciais"] = hash_tensores(params)
            super().__init__(params, *a, **k)
            opt["o"] = self

    def avaliar_loss(modelo, loader, dispositivo, perda):
        reg["epocas"].append({"parametros": hash_tensores(list(modelo.parameters())),
                              "otimizador": hash_otimizador(opt["o"]),
                              "lr": [g["lr"] for g in opt["o"].param_groups],
                              "passos_ate_aqui": len(reg["loss_treino"])})
        return original_avaliar(modelo, loader, dispositivo, perda)

    def prever(modelo, loader, dispositivo, **kw):
        if reg["parametros_finais_melhor_epoca"] is None:
            reg["parametros_finais_melhor_epoca"] = hash_tensores(list(modelo.parameters()))
        return original_prever(modelo, loader, dispositivo, **kw)

    mp.setattr(torch.nn, "CrossEntropyLoss", CE)
    mp.setattr(torch.optim, "AdamW", AdamW)
    mp.setattr(laco, "avaliar_loss", avaliar_loss)
    mp.setattr(laco, "prever", prever)
    raiz = tmp_path_factory.mktemp("pequeno")
    try:
        treino = classificador.treinar_execucao(
            raiz, "p", Config(epochs=3, batch_size=64, max_length=128, lr=1e-3, seed=42),
            modelo=MODELO_MINUSCULO, pasta_particoes=pasta_particoes, dispositivo="cpu")
    finally:
        mp.undo()
    classificador.avaliar_test(raiz / "p", pasta_particoes=pasta_particoes, dispositivo="cpu")
    return {"registro": reg, "treino": treino, "pasta": raiz / "p"}


@EXATO
@pytest.mark.parametrize("campo", ["parametros_iniciais", "loss_treino", "loss_dev", "epocas",
                                   "parametros_finais_melhor_epoca"])
def test_treino_pequeno_passo_a_passo(campo, treino_pequeno, registro_ref):
    """Mesmos parâmetros iniciais; mesma loss em cada um dos 249 passos e em
    cada lote do DEV; mesmos parâmetros, estado do otimizador e lr ao fim de
    cada época; mesmos parâmetros da melhor época. Igualdade exata."""
    assert treino_pequeno["registro"][campo] == registro_ref[campo]


def sem_duracao(historico):
    return [{k: v for k, v in h.items() if k != "duration_s"} for h in historico]


def conferir_com_o_legado(pasta, treino, nome_ref):
    ref = ler_json(TREINO / f"{nome_ref}.json")
    assert sem_duracao(treino["dev_history"]) == sem_duracao(ref["dev_history"])
    assert treino["n_params"] == ref["n_params"]
    assert treino["n_candidatos"] == {k: ref["n_candidates"][k] for k in ("train", "dev")}
    assert treino["melhor_epoca"] == ref["instrumentation"]["best_epoch"]
    for sufixo, arquivo in ((".dev_preds.json", "predicoes_dev.json"), (".preds.json", "predicoes_test.json")):
        legado, novo = ler_json(TREINO / f"{nome_ref}{sufixo}"), ler_json(pasta / arquivo)
        assert list(novo) == list(legado), sufixo                       # mesmas chaves, mesma ordem
        for chave in legado:
            if chave != "model":                                        # caminho do modelo na máquina
                assert novo[chave] == legado[chave], (sufixo, chave)
    m = ler_json(pasta / "metricas.json")["test"]
    assert m["n"] == ref["n_candidates"]["test"]
    assert (m["macro_f1"], m["micro_f1"], m["weighted_f1"], m["mcc"]) == \
        (ref["test_macro_f1"], ref["test_micro_f1"], ref["test_weighted_f1"], ref["test_mcc"])
    assert m["f1_per_class"] == ref["test_f1_per_class"]
    assert m["classification_report"] == ref["sklearn_report"]
    assert m["confusion_matrix"] == ref["confusion_matrix"]


@EXATO
def test_treino_pequeno_historico_predicoes_e_metricas(treino_pequeno):
    conferir_com_o_legado(treino_pequeno["pasta"], treino_pequeno["treino"], "baseline_pequeno_seed42")


def test_treino_pequeno_na_referencia_cobre_o_empate():
    """A referência pequena tem o mesmo macro-F1 nas três épocas: confere o
    desempate pela primeira época (a escolha é a época 1)."""
    ref = ler_json(TREINO / "baseline_pequeno_seed42.json")
    assert len({h["dev_macro_f1"] for h in ref["dev_history"]}) == 1
    assert ref["instrumentation"]["best_epoch"] == 1


@EXATO
@pytest.mark.skipif(os.environ.get("RECLIN_TREINO_COMPLETO") != "1",
                    reason="treino completo do modelo minúsculo (cerca de 11 min em CPU): "
                           "rode com RECLIN_TREINO_COMPLETO=1")
def test_baseline_minusculo_completo(tmp_path):
    """O baseline minúsculo sobre as partições completas, pelos scripts, numa
    thread: histórico, predições do DEV e do TEST e métricas iguais às do legado."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0",
               TOKENIZERS_PARALLELISM="false")
    cmd = [sys.executable, str(CODIGO / "scripts" / "treinar.py"), "--nome", "m", "--saida", str(tmp_path),
           "--particoes", str(PARTICOES), "--modelo", str(MODELO_MINUSCULO), "--epochs", "2",
           "--batch-size", "64", "--max-length", "128", "--lr", "1e-3", "--seed", "42",
           "--dispositivo", "cpu", "--threads", "1"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    r = subprocess.run([sys.executable, str(CODIGO / "scripts" / "avaliar_test.py"), "--execucao",
                        str(tmp_path / "m"), "--dispositivo", "cpu", "--threads", "1"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    conferir_com_o_legado(tmp_path / "m", ler_json(tmp_path / "m" / "treino.json"), "baseline_minusculo_seed42")


def test_particoes_congeladas_inalteradas():
    assert particoes.conferir_particoes(PARTICOES) == []
