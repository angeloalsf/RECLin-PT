"""Etapa 7 de ponta a ponta, com execuções produzidas pelo código novo em CPU:
inventário, tabelas, comparações, limiares, curvas, relatório e conferência.

Validação técnica, não avaliação científica: o modelo é um BERT sintético
aleatório criado no teste (`modelo_sintetico`), sem nenhum arquivo de
`testes/referencia/`. As partições são as primeiras linhas do TRAIN e do DEV
congelados e um TEST SINTÉTICO (linhas 11 a 20 do DEV), com MANIFEST próprio:
o TEST real não é lido. Os pesos levam um `PROVENIENCIA.json` com commit
fictício. Como as partições não são as congeladas, todas as execuções são de
natureza `fora_da_fase`, e é com essa natureza que as tabelas são geradas.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from reclin import particoes, pretreino  # noqa: E402
from reclin.avaliacao import limiar as busca_limiar  # noqa: E402
from reclin.avaliacao import metricas, significancia  # noqa: E402
from reclin.config import Config  # noqa: E402
from reclin.estrategias import (baseline, baseline_calibrado, pair_aware, restrito,  # noqa: E402
                                two_stage)
from reclin.estrategias.two_stage import composicao, treino  # noqa: E402
from reclin.execucao import diretorio, inventario  # noqa: E402
from reclin.negacao import lexico as modulo_lexico  # noqa: E402
from reclin.relatorios import fase  # noqa: E402
from reclin.treino import classificador  # noqa: E402
from reclin.util.caminhos import CODIGO, PARTICOES  # noqa: E402

SCRIPTS = CODIGO / "scripts"
PEQ = {"batch_size": 16, "max_length": 64, "lr": 1e-3, "epochs": 2}
N_BOOT = 200
SIS = {s.chave: s for s in fase.sistemas()}


def rodar(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def part(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("particoes_etapa7")
    train = (PARTICOES / "train.jsonl").read_bytes().splitlines(keepends=True)
    dev = (PARTICOES / "dev.jsonl").read_bytes().splitlines(keepends=True)
    (pasta / "train.jsonl").write_bytes(b"".join(train[:10]))
    (pasta / "dev.jsonl").write_bytes(b"".join(dev[:10]))
    (pasta / "test.jsonl").write_bytes(b"".join(dev[10:20]))          # TEST sintético
    particoes.gravar_manifesto(pasta, particoes.calcular_manifesto(
        pasta, seed=42, origem={"path": "TRAIN/DEV congelados; test sintético (linhas 11-20 do DEV)", "sha256": None}))
    return pasta


@pytest.fixture(scope="module")
def raiz(tmp_path_factory, part, modelo_sintetico):
    """As seis execuções da fase (com os nomes da fase) e os dois estágios,
    treinadas em CPU, e o DEV do filtro pelo script, no modo da fase."""
    pre = tmp_path_factory.mktemp("pre") / "sintetico"
    shutil.copytree(modelo_sintetico, pre)
    pretreino.gravar(pre, repositorio="teste/sintetico", commit="1" * 40)
    sem_test = tmp_path_factory.mktemp("sem_test")
    for nome in ("train.jsonl", "dev.jsonl", "MANIFEST.json"):
        shutil.copy(part / nome, sem_test / nome)
    raiz = tmp_path_factory.mktemp("execucoes")
    lex = modulo_lexico.carregar_congelado(particoes.ler_particao(PARTICOES, "train"))

    cfg = Config(**PEQ)
    classificador.treinar_execucao(raiz, SIS["baseline"].execucao, cfg, modelo=pre, pasta_particoes=sem_test,
                                   montagem=baseline.montagem(cfg))
    for modulo, classe, chave in ((restrito, restrito.ConfigRestrito, "restrito"),
                                  (pair_aware, pair_aware.ConfigPairAware, "pair_aware")):
        c = classe(**PEQ)
        classificador.treinar_execucao(raiz, SIS[chave].execucao, c, modelo=pre, pasta_particoes=sem_test,
                                       montagem=modulo.montagem(c, lexico=lex) if modulo is restrito else modulo.montagem(c))
    for estagio in (1, 2):
        treino.treinar_estagio(raiz, two_stage.nome_execucao(estagio, two_stage.CONFIGS[estagio]()), estagio,
                               two_stage.CONFIGS[estagio](**PEQ), modelo=pre, pasta_particoes=sem_test)
    e1, e2 = (raiz / two_stage.nome_execucao(e, two_stage.CONFIGS[e]()) for e in (1, 2))
    composicao.compor_dev(e1, e2, raiz, SIS["two_stage"].execucao, pasta_particoes=sem_test)
    baseline_calibrado.calibrar_dev(raiz / SIS["baseline"].execucao, raiz, SIS["baseline_calibrado"].execucao)

    # o filtro pelo script, no modo da fase: o DEV agora; o TEST antes do TEST da base é recusado
    filtro = ["--base-execucao", raiz / SIS["baseline"].execucao, "--saida", raiz]
    r = rodar("filtro_pistas.py", *filtro, "--particao", "dev", "--particoes", sem_test)
    assert r.returncode == 0, r.stdout + r.stderr
    cedo = rodar("filtro_pistas.py", *filtro, "--particao", "test", "--particoes", part)
    return raiz, filtro, cedo


@pytest.fixture(scope="module")
def avaliadas(raiz, part):
    raiz, filtro, cedo = raiz
    gerado_antes = rodar("gerar_tcc.py", "resultados", "--execucoes", raiz, "--particoes", part,
                         "--natureza", "fora_da_fase", "--destino", raiz.parent / "antes", "--n-boot", N_BOOT)
    for chave, modulo in (("baseline", baseline), ("restrito", restrito), ("pair_aware", pair_aware)):
        pasta = raiz / SIS[chave].execucao
        classificador.avaliar_test(pasta, pasta_particoes=part,
                                   montagem=modulo.montagem_de_registro(diretorio.ler_config(pasta)["config"])
                                   if chave != "baseline" else baseline.montagem(Config(**PEQ)))
    composicao.avaliar_test(raiz / SIS["two_stage"].execucao, pasta_particoes=part)
    baseline_calibrado.aplicar_test(raiz / SIS["baseline_calibrado"].execucao)
    r = rodar("filtro_pistas.py", *filtro, "--particao", "test", "--particoes", part)
    assert r.returncode == 0, r.stdout + r.stderr
    de_novo = rodar("filtro_pistas.py", *filtro, "--particao", "test", "--particoes", part)
    destino = raiz.parent / "relatorios"
    r = rodar("gerar_tcc.py", "tudo", "--execucoes", raiz, "--particoes", part, "--natureza", "fora_da_fase",
              "--destino", destino, "--n-boot", N_BOOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return raiz, destino, gerado_antes, {"cedo": cedo, "de_novo": de_novo}


def _csv(destino, nome):
    linhas = (destino / "tabelas" / f"{nome}.csv").read_text("utf-8").splitlines()
    return [linha.split(",") for linha in linhas]


def _indice(destino, grupo):
    return json.loads((destino / "indice" / f"{grupo}.json").read_text("utf-8"))


def test_inventario_das_execucoes(avaliadas, part):
    raiz, *_ = avaliadas
    inv = inventario.inventariar(raiz, pasta_particoes=part)
    assert len(inv.execucoes) == 8 and not inv.recusados
    for ex in inv.execucoes.values():
        assert ex.valida, (ex.nome, ex.problemas)
        assert ex.natureza == "fora_da_fase" and ex.motivos_fora_da_fase == ["partições diferentes das congeladas"]
        assert ex.seed == 42 and ex.encoder == "biobertpt" and ex.proveniencia["commit"] == "1" * 40
    assert inv.obter(SIS["filtro"].execucao).dependencias == [SIS["baseline"].execucao]
    assert inv.obter(SIS["filtro"].execucao).tem("test")             # TEST com trilha, pelo próprio script
    assert len({ex.codigo for ex in inv.execucoes.values()}) == 1 and inv.codigo_referencia in {
        ex.codigo for ex in inv.execucoes.values()}
    assert inv.obter(SIS["two_stage"].execucao).tem("test")


def test_as_tabelas_nao_avaliam_o_test_pela_primeira_vez(avaliadas):
    raiz, _, antes, _ = avaliadas
    assert antes.returncode == 0, antes.stderr
    linhas = _csv(raiz.parent / "antes", "resultados_test")
    assert all(c == "pendente" for linha in linhas[1:] for c in linha[1:])


def test_resultados_recalculados_das_predicoes(avaliadas):
    raiz, destino, *_ = avaliadas
    linhas = {linha[0]: linha[1:] for linha in _csv(destino, "resultados_test")[1:]}
    for s in fase.sistemas():
        p = diretorio.ler_predicoes(raiz / s.execucao, "test")
        m = metricas.avaliar(p["y_true"], p["y_pred"])
        alvo = m["por_classe"]["negation_of"]
        assert [float(x) for x in linhas[s.rotulo][:3]] == [alvo["precision"], alvo["recall"], alvo["f1"]]
        if s.comparabilidade == fase.SO_NEG:
            assert linhas[s.rotulo][3:] == ["n/c", "n/c"]
        else:
            assert [float(x) for x in linhas[s.rotulo][3:]] == [m["macro_f1"], m["mcc"]]
    indice = _indice(destino, "resultados")["tabelas"]
    assert indice["resultados_test"]["status"] == "completa" and not indice["resultados_test"]["pendencias"]
    assert indice["diagnostico_dev"]["status"] == "completa"


def test_comparacoes_iguais_as_da_significancia(avaliadas):
    raiz, destino, *_ = avaliadas
    linhas = {linha[0]: linha for linha in _csv(destino, "comparacoes_test")[1:]}
    feitas = 0
    for a, b, tipo in fase.COMPARACOES:
        rotulo = f"{SIS[a].rotulo} × {SIS[b].rotulo}"
        linha = linhas[rotulo]
        assert linha[1] == tipo
        pa = diretorio.ler_predicoes(raiz / SIS[a].execucao, "test")
        pb = diretorio.ler_predicoes(raiz / SIS[b].execucao, "test")
        r = significancia.comparar(pa, pb, seed=42, n_boot=N_BOOT)
        bs = r["paired_bootstrap"]
        assert [float(x) for x in linha[2:8]] == [r["target_f1"]["a"], r["target_f1"]["b"], bs["observed_diff"],
                                                  bs["ci95_low"], bs["ci95_high"], bs["p_value"]]
        assert linha[8] == ("n/c" if fase.SO_NEG in (SIS[a].comparabilidade, SIS[b].comparabilidade)
                            else repr(float(r["mcnemar"]["p_value"])))
        assert linha[9] == fase.leitura(bs["ci95_low"], bs["ci95_high"])
        gravado = json.loads((destino / "comparacoes" / f"{SIS[a].execucao}__vs__{SIS[b].execucao}.json")
                             .read_text("utf-8"))
        assert gravado["paired_bootstrap"] == bs
        feitas += 1
    assert feitas == 6


def test_limiares_e_curvas(avaliadas):
    raiz, destino, *_ = avaliadas
    linhas = {linha[0]: linha for linha in _csv(destino, "limiares_dev")[1:]}
    for chave in ("baseline_calibrado", "two_stage"):
        gravado = json.loads((raiz / SIS[chave].execucao / busca_limiar.ARQUIVO).read_text("utf-8"))
        assert float(linhas[SIS[chave].rotulo][1]) == gravado["limiar"]
    for _, nome in fase.treinadas():
        treino_json = json.loads((raiz / nome / "treino.json").read_text("utf-8"))
        linhas = _csv(destino, f"curva_{nome}")[1:]
        assert [int(linha[0]) for linha in linhas] == [h["epoch"] for h in treino_json["dev_history"]]
        escolhida = [int(linha[0]) for linha in linhas if linha[4] == "sim"]
        assert escolhida == [treino_json["melhor_epoca"]]


def test_relatorio_e_valores(avaliadas):
    _, destino, *_ = avaliadas
    texto = (destino / "relatorio.md").read_text("utf-8")
    assert "fora da fase" in texto and "Filtro léxico" in texto
    valores = json.loads((destino / "valores" / "resultados.json").read_text("utf-8"))
    chave = "test.two_stage.f1_negation_of"
    assert chave in valores and valores[chave]["natureza"] == "fora_da_fase"
    assert "test.filtro.f1_negation_of" in valores
    tex = (destino / "valores" / "resultados.tex").read_text("utf-8")
    assert r"\csname rv:" + chave + r"\endcsname{" + valores[chave]["texto"] + "}" in tex


def test_outra_natureza_nao_entra(avaliadas, part, tmp_path):
    raiz, *_ = avaliadas
    r = rodar("gerar_tcc.py", "resultados", "--execucoes", raiz, "--particoes", part, "--destino", tmp_path / "c",
              "--n-boot", N_BOOT)
    assert r.returncode == 0, r.stderr
    indice = json.loads((tmp_path / "c" / "indice" / "resultados.json").read_text("utf-8"))
    assert indice["tabelas"]["resultados_test"]["status"] == "pendente"
    assert all("natureza 'fora_da_fase'" in p for p in indice["tabelas"]["resultados_test"]["pendencias"])
    # um destino com resultados de uma natureza recusa os de outra
    r = rodar("gerar_tcc.py", "resultados", "--execucoes", raiz, "--particoes", part, "--destino", tmp_path / "c",
              "--natureza", "fora_da_fase", "--n-boot", N_BOOT)
    assert r.returncode == 1 and "outro destino" in r.stdout + r.stderr


def test_conferencia_do_tcc(avaliadas, part, tmp_path):
    raiz, destino, *_ = avaliadas
    valores = json.loads((destino / "valores" / "resultados.json").read_text("utf-8"))
    chave = "test.two_stage.f1_negation_of"
    tcc = tmp_path / "tcc"
    tcc.mkdir()
    (tcc / "cap.tex").write_text(f"O Two-Stage obteve F1 de {valores[chave]['texto']} no TEST.\n", "utf-8")
    afirmacoes = tmp_path / "afirmacoes.json"
    afirmacoes.write_text(json.dumps([{"id": "f1-ts", "arquivo": "cap.tex", "trecho": "F1 de {valor} no TEST",
                                       "chave": chave}]), "utf-8")
    base = ["--tcc", tcc, "--relatorios", destino, "--afirmacoes", afirmacoes, "--execucoes", raiz,
            "--particoes", part]
    assert rodar("conferir_tcc.py", *base).returncode == 0
    (tcc / "ap.tex").write_text("Ver src/relation_extraction.py.\n", "utf-8")
    assert rodar("conferir_tcc.py", *base).returncode == 1
    (tcc / "ap.tex").unlink()
    # uma execução que mudou depois da geração invalida os valores gerados
    sidecar = raiz / SIS["baseline"].execucao / "predicoes_test.json"
    original = sidecar.read_bytes()
    try:
        sidecar.write_bytes(json.dumps(json.loads(original), ensure_ascii=False, indent=1).encode("utf-8"))
        r = rodar("conferir_tcc.py", *base)
        assert r.returncode == 1 and "mudou desde a geração" in r.stdout
    finally:
        sidecar.write_bytes(original)
    assert rodar("conferir_tcc.py", *base).returncode == 0


def test_auditoria_de_falsos_positivos(avaliadas, part, tmp_path):
    raiz, *_ = avaliadas
    base = ["falsos-positivos", "--execucoes", raiz, "--particoes", part]
    r = rodar("analisar.py", *base, "--execucao", SIS["baseline"].execucao, "--saida", tmp_path / "fp.json")
    assert r.returncode == 0, r.stdout + r.stderr
    gravado = json.loads((tmp_path / "fp.json").read_text("utf-8"))
    assert gravado["particao"] == "dev" and gravado["origem"]["execucao"]["natureza"] == "fora_da_fase"
    p = diretorio.ler_predicoes(raiz / SIS["baseline"].execucao, "dev")
    fp = sum(1 for yt, yp in zip(p["y_true"], p["y_pred"]) if yp == 0 and yt != 0)
    assert gravado["falsos_positivos"]["resumo"]["fp"] == fp
    r = rodar("analisar.py", *base, "--execucao", SIS["filtro"].execucao, "--particao", "test")
    assert r.returncode == 0, r.stdout + r.stderr


def test_filtro_no_modo_da_fase(avaliadas):
    """D3: o filtro usa o léxico congelado sem arquivo de calibração; o TEST
    só depois do TEST da base, uma vez, com a trilha; o vínculo com as
    predições da base fica em cada sidecar."""
    raiz, _, _, logs = avaliadas
    assert logs["cedo"].returncode == 1 and "comando próprio" in logs["cedo"].stdout
    assert logs["de_novo"].returncode == 1 and "--reavaliar" in logs["de_novo"].stdout
    pasta = raiz / SIS["filtro"].execucao
    config = diretorio.ler_config(pasta)
    assert "calibracao" not in config["config"] and config["config"]["lexico_sha1"] == \
        modulo_lexico.CONGELADO["lexico_sha1"]
    linhas = [json.loads(l) for l in (pasta / "avaliacoes_test.jsonl").read_text("utf-8").splitlines()]
    assert len(linhas) == 1 and linhas[0]["codigo"] == config["codigo"]["identidade"]
    from reclin.util.io import sha256_arquivo
    for p in ("dev", "test"):
        assert diretorio.ler_predicoes(pasta, p)["base_sha256"] == \
            sha256_arquivo(raiz / SIS["baseline"].execucao / f"predicoes_{p}.json")
