"""Relatórios (etapa 7): formatação, sistemas da fase, tabelas a partir do
inventário, registro de valores, curvas, EDA, relatório Markdown e conferência
do TCC. Execuções e partições sintéticas; nada do legado nem do TEST real."""
from __future__ import annotations

import json
import math
import re

import pytest

from reclin.avaliacao import metricas
from reclin.config import Config
from reclin.estrategias import baseline, baseline_calibrado, filtro_pistas, pair_aware, restrito, two_stage
from reclin.execucao import inventario as inv
from reclin.relatorios import conferencia, curvas, eda, fase, figuras, markdown, tabelas
from reclin.relatorios.formatacao import NAO_COMPARAVEL, PENDENTE, Numero, csv_texto, escapar_latex, latex
from reclin.util import caminhos, formatacao
from reclin.util.io import gravar_json

PROTOCOLO = caminhos.CODIGO / "docs" / "protocolo" / "fase-biobertpt-seed42.md"


# --------------------------------------------------------------------------- #
# Formatação                                                                  #
# --------------------------------------------------------------------------- #
def test_numeros():
    assert formatacao.ptbr(0.81234) == "0,8123" and formatacao.fmt(0.81235, 2) == "0.81"
    assert formatacao.sinal(0.0123) == "+0,0123" and formatacao.sinal(-0.0123) == "−0,0123"
    assert formatacao.sinal(0.00001) == "0,0000"
    assert formatacao.porcentagem(0.9467) == "94,7%" and formatacao.inteiro(152686) == "152.686"
    assert formatacao.ptbr(None) == formatacao.TRACO and formatacao.ptbr(math.nan) == formatacao.TRACO


def test_latex_csv_e_marcadores():
    assert escapar_latex("negation_of & 50%") == r"negation\_of \& 50\%"
    texto = latex(["Sistema", "F1"], [["A_b", Numero(0.5, chave="k")], ["B", PENDENTE], ["C", NAO_COMPARAVEL],
                                      ["D", Numero(-0.25, com_sinal=True)]], comentarios=["origem"])
    assert texto.startswith("% origem\n\\begin{tabular}{lr}")
    assert r"A\_b & 0,5000 \\" in texto and r"\textit{pendente}" in texto and "n/c" in texto
    assert "$-$0,2500" in texto
    linhas = csv_texto(["S", "F1", "n"], [["A", Numero(1 / 3), Numero(1234, inteiro=True)], ["B", PENDENTE, "x"]])
    assert linhas.splitlines() == ["S,F1,n", f"A,{1 / 3!r},1234", "B,pendente,x"]


# --------------------------------------------------------------------------- #
# Sistemas e comparações da fase                                              #
# --------------------------------------------------------------------------- #
def test_nomes_das_execucoes_sao_os_das_estrategias():
    c = Config()
    nomes = {s.chave: s.execucao for s in fase.sistemas()}
    assert nomes["baseline"] == baseline.nome_execucao(c)
    assert nomes["baseline_calibrado"] == baseline_calibrado.nome_execucao(baseline.nome_execucao(c))
    assert nomes["filtro"] == filtro_pistas.nome_execucao(baseline.nome_execucao(c))
    assert nomes["restrito"] == restrito.nome_execucao(c)
    assert nomes["pair_aware"] == pair_aware.nome_execucao(c)
    assert nomes["two_stage"] == two_stage.nome_composicao(c)
    assert dict(fase.treinadas())["Two-Stage, estágio 1"] == two_stage.nome_execucao(1, c)
    assert [s.chave for s in fase.sistemas() if s.comparabilidade == fase.SO_NEG] == ["restrito", "pair_aware"]


def test_sistemas_da_validacao_sao_os_do_protocolo():
    registro = json.loads(re.findall(r"```json\n(.*?)\n```", PROTOCOLO.read_text("utf-8"), flags=re.S)[0])
    nomes = registro["validacao_tecnica"]["nomes"]
    assert {s.execucao for s in fase.sistemas_validacao()} == \
        {nomes["baseline"], nomes["baseline_calibrado"], nomes["two_stage"]}
    assert {n for _, n in fase.treinadas_validacao()} == \
        {nomes["baseline"], nomes["two_stage_e1"], nomes["two_stage_e2"]}


def test_comparacoes_sao_as_do_protocolo():
    """As linhas da tabela da seção 7 do protocolo, na mesma ordem e tipo."""
    texto = PROTOCOLO.read_text("utf-8")
    secao = texto[texto.index("## 7."):texto.index("## 8.")]
    chaves = {"two-stage": "two_stage", "baseline padrão": "baseline", "baseline calibrado": "baseline_calibrado",
              "filtro": "filtro", "restrito": "restrito", "pair-aware": "pair_aware"}
    esperadas = []
    for linha in secao.splitlines():
        m = re.match(r"\| (.+?) \| (principal|exploratória)", linha)
        if not m:
            continue
        a, *bs = [x.strip().lower() for x in m.group(1).replace("; ×", " ×").split(" × ")]
        esperadas += [(chaves[a], chaves[b], m.group(2)) for b in bs]
    assert list(fase.COMPARACOES) == esperadas


def test_leitura_do_intervalo():
    assert fase.leitura(0.01, 0.2) == "diferença positiva"
    assert fase.leitura(-0.2, -0.01) == "diferença negativa"
    assert fase.leitura(0.0, 0.2) == "diferença não detectada"


# --------------------------------------------------------------------------- #
# Tabelas a partir do inventário                                              #
# --------------------------------------------------------------------------- #
@pytest.fixture
def inventario_sintetico(fabrica_execucoes, particoes_sinteticas, ferramentas):
    sis = {s.chave: s.execucao for s in fase.sistemas()}
    fabrica_execucoes(sis["baseline"], "baseline", ferramentas.registro_treinado("baseline"), semente=1)
    fabrica_execucoes(sis["restrito"], "restrito", ferramentas.registro_treinado("restrito"), semente=2,
                      avaliar_test=False)
    fabrica_execucoes(sis["two_stage"], "two_stage", {"estagios": {"1": {"nome": "e1"}, "2": {"nome": "e2"}},
                                                     "seed": 42}, semente=3)
    for nome in ("e1", "e2"):
        pasta = fabrica_execucoes(nome, "two_stage_e" + nome[-1], ferramentas.registro_treinado("x"),
                                  particoes_=())
        ferramentas.gravar_treino(pasta, [{"epoch": 1, "train_loss": 1.0, "dev_loss": 0.9, "dev_ap_relacao": 0.3},
                                          {"epoch": 2, "train_loss": 0.8, "dev_loss": 0.85, "dev_ap_relacao": 0.3}],
                                  1, campo="dev_ap_relacao")
    return inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)


def test_resultados_com_pendencias(inventario_sintetico):
    t = tabelas.resultados(inventario_sintetico, "test", natureza="fora_da_fase")
    por_sistema = {linha[0]: linha[1:] for linha in t.linhas}
    assert t.status == "parcial"
    assert por_sistema["Baseline calibrado"] == [PENDENTE] * 5
    assert por_sistema["Restrito"] == [PENDENTE] * 5              # sem trilha do TEST
    ex = inventario_sintetico.obter(fase.sistemas()[0].execucao)
    p = ex.predicoes("test")
    m = metricas.avaliar(p["y_true"], p["y_pred"])
    assert por_sistema["Baseline"][2].valor == m["por_classe"]["negation_of"]["f1"]
    assert por_sistema["Baseline"][3].chave == "test.baseline.macro_f1"
    dev = {linha[0]: linha[1:] for linha in tabelas.resultados(inventario_sintetico, "dev",
                                                                  natureza="fora_da_fase").linhas}
    assert dev["Restrito"][3:] == [NAO_COMPARAVEL, NAO_COMPARAVEL]
    assert {o["execucao"] for o in t.origens} == {fase.sistemas()[0].execucao, fase.sistemas()[5].execucao}


def test_natureza_diferente_vira_pendencia(inventario_sintetico):
    t = tabelas.resultados(inventario_sintetico, "dev", natureza="cientifica")
    assert t.status == "pendente" and not t.valores()
    assert any("natureza 'fora_da_fase'" in p for p in t.pendencias)


def test_execucao_com_problemas_interrompe(inventario_sintetico):
    ex = inventario_sintetico.obter(fase.sistemas()[0].execucao)
    ex.problemas.append("adulterada")
    with pytest.raises(tabelas.ErroTabela, match="problemas de proveniência"):
        tabelas.resultados(inventario_sintetico, "dev", natureza="fora_da_fase")


def test_comparacoes_so_com_os_dois_lados(inventario_sintetico):
    t, relatorios = tabelas.comparacoes(inventario_sintetico, natureza="fora_da_fase", n_boot=50)
    feitas = [linha for linha in t.linhas if isinstance(linha[2], Numero)]
    assert len(feitas) == 1 and feitas[0][0] == "Two-Stage × Baseline" and len(relatorios) == 1


def test_registro_de_valores_e_gravacao(inventario_sintetico, tmp_path):
    t = tabelas.resultados(inventario_sintetico, "dev", natureza="fora_da_fase")
    indice = tabelas.gravar([t], tmp_path, grupo="resultados")
    assert indice["tabelas"]["diagnostico_dev"]["status"] == t.status
    valores = tabelas.ler_valores(tmp_path)
    assert valores["dev.baseline.mcc"]["natureza"] == "fora_da_fase"
    assert (tmp_path / "tabelas" / "diagnostico_dev.tex").read_text("utf-8").startswith("% Gerado")
    outra = tabelas.Tabela("x", "x", ["a"], [[Numero(0.1, chave="k")]], "validacao")
    with pytest.raises(tabelas.ErroTabela, match="outro destino"):
        tabelas.gravar([outra], tmp_path, grupo="outro")
    conflito = [tabelas.Tabela("a", "a", ["v"], [[Numero(0.1, chave="k")]], "dados"),
                tabelas.Tabela("b", "b", ["v"], [[Numero(0.2, chave="k")]], "dados")]
    with pytest.raises(tabelas.ErroTabela, match="valores diferentes"):
        tabelas.registro_valores(conflito)


def test_limiar_conferido(fabrica_execucoes, particoes_sinteticas, ferramentas):
    from reclin.avaliacao import limiar
    sis = {s.chave: s.execucao for s in fase.sistemas()}
    fabrica_execucoes(sis["baseline"], "baseline", ferramentas.registro_treinado("baseline"))
    pasta = fabrica_execucoes(sis["baseline_calibrado"], "baseline_calibrado",
                              {"base": {"nome": sis["baseline"]}, "seed": 42,
                               "proveniencia_pesos": ferramentas.registro_treinado("x")["proveniencia_pesos"]},
                              extras={"dev": {"limiar_t": 0.4}})
    y = [0, 1, 2, 0]
    _, tabela = limiar.buscar(y, lambda t: [0, 1, 2, 0] if t <= 0.4 else [2, 1, 2, 2])
    limiar.congelar(pasta / limiar.ARQUIVO, limiar=0.4, regra="r", vinculos={}, tabela=tabela)
    i = inv.inventariar(fabrica_execucoes.raiz, pasta_particoes=particoes_sinteticas)
    t = tabelas.limiares(i, natureza="fora_da_fase")
    assert t.linhas[0][1].valor == 0.4 and t.linhas[1][1] is PENDENTE
    dados = json.loads((pasta / limiar.ARQUIVO).read_text("utf-8"))
    dados["limiar"] = 0.45
    gravar_json(pasta / limiar.ARQUIVO, dados)
    with pytest.raises(tabelas.ErroTabela, match="não é o que a regra escolhe"):
        tabelas.limiares(i, natureza="fora_da_fase")


def test_particoes_e_proveniencia(inventario_sintetico, particoes_sinteticas):
    t = tabelas.particoes_tabela(particoes_sinteticas)
    assert [linha[0] for linha in t.linhas] == ["TRAIN", "DEV", "TEST"] and t.status == "completa"
    p = tabelas.proveniencia(inventario_sintetico.execucoes.values(), "fora_da_fase")
    linha = next(linha for linha in p.linhas if linha[0] == fase.sistemas()[0].execucao)
    assert linha[2] == "a" * 12 and linha[5].valor == 1


def test_curvas(inventario_sintetico):
    lista, historicos = curvas.tabelas(inventario_sintetico, natureza="fora_da_fase",
                                       treinadas=[("E1", "e1"), ("Ausente", "nao_existe")])
    assert lista[0].status == "completa" and [linha[4] for linha in lista[0].linhas] == ["sim", ""]
    assert historicos[0]["campo_selecao"] == "dev_ap_relacao"
    assert lista[1].status == "pendente" and "não encontrada" in lista[1].pendencias[0]


def test_eda(particoes_sinteticas):
    from reclin import particoes
    from reclin.analises import corpus, max_gap
    docs = particoes.ler_particoes(particoes_sinteticas)
    e = corpus.estatisticas(docs)
    t = eda.corpus(e, "sha")
    assert [linha[0] for linha in t.linhas] == ["TRAIN", "DEV", "TEST", "Total"]
    assert t.linhas[-1][5].valor == sum(e[p]["n_candidatos"] for p in ("train", "dev", "test"))
    s = eda.max_gap(max_gap.sensibilidade(docs["train"], [5, 25]), "sha", "train")
    assert s.linhas[1][0] == "25 (tarefa)"
    assert eda.cobertura(e, "sha").status == "completa" and eda.distancias(e, "sha").status == "completa"


def test_relatorio_markdown(inventario_sintetico):
    t = tabelas.resultados(inventario_sintetico, "test", natureza="fora_da_fase")
    texto = markdown.relatorio(inventario_sintetico, [t], natureza="fora_da_fase")
    assert "Status: **parcial**" in texto and "Pendente:" in texto and "## Pendências" in texto
    assert str(caminhos.CODIGO) not in texto or "codigo" in texto


def test_figuras_sem_matplotlib_ou_com():
    if not figuras.disponivel():
        pytest.skip("matplotlib não instalado: figuras não verificadas neste ambiente")
    assert figuras.estilo(11)["font.size"] == 11


# --------------------------------------------------------------------------- #
# Conferência do TCC                                                          #
# --------------------------------------------------------------------------- #
def test_conferencia(tmp_path):
    assert conferencia.conferir(tmp_path / "nao_existe").status == "pendente"
    (tmp_path / "cap").mkdir()
    (tmp_path / "cap" / "a.tex").write_text("F1 de 0,8123 no TEST.\n% scripts/_artifacts.py em comentário\n",
                                            "utf-8")
    valores = {"k": {"texto": "0,8123"}, "j": {"texto": "0,5000"}}
    afirmacoes = [{"id": 1, "arquivo": "cap/a.tex", "trecho": "F1 de {valor} no", "chave": "k"}]
    r = conferencia.conferir(tmp_path, afirmacoes=afirmacoes, valores=valores)
    assert r.status == "ok" and r.afirmacoes[0]["ok"]
    assert conferencia.conferir(tmp_path).status == "pendente"            # sem afirmações
    erradas = afirmacoes + [{"id": 2, "arquivo": "cap/a.tex", "trecho": "F1 de {valor}", "chave": "j"},
                            {"id": 3, "arquivo": "cap/a.tex", "trecho": "{valor}", "chave": "sem_valor"},
                            {"id": 4, "arquivo": "cap/a.tex", "trecho": "sem marcador", "chave": "k"}]
    r = conferencia.conferir(tmp_path, afirmacoes=erradas, valores=valores)
    assert r.status == "divergente" and [a["ok"] for a in r.afirmacoes] == [True, False, False, False]
    (tmp_path / "cap" / "b.tex").write_text("Veja data/splits/MANIFEST.json.\n", "utf-8")
    r = conferencia.conferir(tmp_path, afirmacoes=afirmacoes, valores=valores)
    assert r.status == "divergente" and r.caminhos_antigos == [
        {"arquivo": "cap/b.tex", "linha": 1, "caminho": "data/splits/MANIFEST.json"}]
