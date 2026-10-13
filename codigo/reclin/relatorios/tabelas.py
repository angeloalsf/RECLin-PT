"""Tabelas de resultados da fase, a partir de execuções verificadas, e o
registro de valores que o texto do TCC cita.

Toda tabela de resultados é montada a partir do inventário
(`execucao.inventario`) e de uma única natureza (`cientifica`, `validacao` ou
`fora_da_fase`), nunca misturadas; os números são recalculados dos sidecars
(`avaliacao.metricas`, `avaliacao.significancia`), não lidos de resumos. Um
sistema sem execução, sem a partição avaliada ou só com execução de outra
natureza aparece com as células `pendente` e a pendência descrita; uma
execução com problemas de proveniência interrompe a geração.

O TEST só é usado quando a execução tem a trilha de avaliação
(`avaliacoes_test.jsonl`), isto é, quando o TEST foi avaliado pelo comando
próprio; as tabelas não avaliam nada pela primeira vez.

Origem no legado (só a finalidade): `make_tcc_artifacts.py` e os resumos de
`make_fase2_report.py`. Nenhum número, lista de sistemas ou critério vem de lá.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reclin import particoes, versao_dados
from reclin.avaliacao import limiar as busca_limiar
from reclin.avaliacao import metricas, protocolo, significancia
from reclin.execucao import inventario as inv
from reclin.relatorios import fase
from reclin.relatorios.formatacao import NAO_COMPARAVEL, PENDENTE, Numero, csv_texto, latex, markdown
from reclin.util import caminhos
from reclin.util.io import gravar_json, gravar_texto, ler_json, sha256_arquivo

AVISO = "Gerado por codigo/scripts/gerar_tcc.py a partir de execuções verificadas; não edite à mão."
COLUNAS_METRICAS = ["Sistema", "P (negation_of)", "R (negation_of)", "F1 (negation_of)", "Macro-F1", "MCC"]


class ErroTabela(ValueError):
    pass


@dataclass
class Tabela:
    nome: str
    titulo: str
    colunas: list[str]
    linhas: list[list[Any]]
    natureza: str
    origens: list[dict[str, Any]] = field(default_factory=list)
    pendencias: list[str] = field(default_factory=list)
    notas: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        numeros = [c for linha in self.linhas for c in linha if isinstance(c, Numero)]
        if not numeros:
            return "pendente"
        return "parcial" if self.pendencias else "completa"

    def valores(self) -> dict[str, Numero]:
        return {c.chave: c for linha in self.linhas for c in linha if isinstance(c, Numero) and c.chave}

    def comentarios(self) -> list[str]:
        nomes = ", ".join(f"{o['execucao']}" for o in self.origens) or "nenhuma execução"
        return [AVISO, f"Tabela {self.nome}: {self.titulo}",
                f"Natureza: {fase.ROTULO_NATUREZA.get(self.natureza, self.natureza)}; status: {self.status}",
                f"Origem: {nomes}"] + [f"Pendente: {p}" for p in self.pendencias]

    def latex(self) -> str:
        return latex(self.colunas, self.linhas, comentarios=self.comentarios())

    def markdown(self) -> str:
        return markdown(self.colunas, self.linhas)

    def csv(self) -> str:
        return csv_texto(self.colunas, self.linhas)


# --------------------------------------------------------------------------- #
# Seleção dos sistemas                                                        #
# --------------------------------------------------------------------------- #
def _disponiveis(inventario: inv.Inventario, sistemas: Sequence[fase.Sistema], particao: str,
                 natureza: str) -> tuple[dict[str, inv.Execucao], list[str]]:
    """Execuções utilizáveis de cada sistema (chave → execução) e as pendências."""
    usadas, pendencias = {}, []
    for s in sistemas:
        ex = inventario.execucoes.get(s.execucao)
        if ex is None:
            pendencias.append(f"{s.rotulo}: execução {s.execucao} não encontrada")
        elif not ex.valida:
            raise ErroTabela(f"{s.execucao} tem problemas de proveniência: " + "; ".join(ex.problemas))
        elif ex.natureza != natureza:
            pendencias.append(f"{s.rotulo}: {s.execucao} é de natureza {ex.natureza!r} e não entra numa "
                              f"tabela {natureza!r}")
        elif not ex.tem(particao):
            pendencias.append(f"{s.rotulo}: {s.execucao} sem {particao.upper()} "
                              + ("avaliado (sem trilha de avaliação)" if particao == "test" else "gravado"))
        else:
            usadas[s.chave] = ex
    try:
        inventario.selecionar([ex.nome for ex in usadas.values()], natureza=natureza, ausentes="falhar")
    except inv.ErroInventario as erro:
        raise ErroTabela(str(erro)) from erro
    return usadas, pendencias


def _linha_metricas(s: fase.Sistema, m: Mapping[str, Any] | None, prefixo: str) -> list[Any]:
    if m is None:
        return [s.rotulo] + [PENDENTE] * 5
    alvo = m["por_classe"]["negation_of"]
    k = f"{prefixo}.{s.chave}"
    linha = [s.rotulo, Numero(alvo["precision"], chave=f"{k}.p_negation_of"),
             Numero(alvo["recall"], chave=f"{k}.r_negation_of"), Numero(alvo["f1"], chave=f"{k}.f1_negation_of")]
    if s.comparabilidade == fase.SO_NEG:
        return linha + [NAO_COMPARAVEL, NAO_COMPARAVEL]
    return linha + [Numero(m["macro_f1"], chave=f"{k}.macro_f1"), Numero(m["mcc"], chave=f"{k}.mcc")]


def resultados(inventario: inv.Inventario, particao: str, *, natureza: str = "cientifica",
               sistemas: Sequence[fase.Sistema] | None = None) -> Tabela:
    """Precisão, recall e F1 de `negation_of`, macro-F1 e MCC de cada sistema
    na partição (`test`: resultado; `dev`: diagnóstico)."""
    if particao not in ("dev", "test"):
        raise ValueError(f"partição {particao!r}")
    sistemas = list(sistemas or fase.sistemas())
    usadas, pendencias = _disponiveis(inventario, sistemas, particao, natureza)
    linhas = []
    for s in sistemas:
        ex = usadas.get(s.chave)
        m = None
        if ex is not None:
            sidecar = ex.predicoes(particao)
            m = metricas.avaliar(sidecar["y_true"], sidecar["y_pred"])
        linhas.append(_linha_metricas(s, m, particao))
    notas = ["n/c: não comparável — no restrito só o F1 de negation_of se compara "
             "com o baseline (os pares fora do espaço restrito são no_relation por construção)."]
    if particao == "dev":
        notas.append("DEV: diagnóstico. No baseline calibrado e no Two-Stage é otimista "
                     "(o limiar foi escolhido neste DEV).")
    nome = "resultados_test" if particao == "test" else "diagnostico_dev"
    titulo = ("Resultados no TEST" if particao == "test" else "Diagnóstico no DEV") + \
        f" ({fase.ROTULO_NATUREZA.get(natureza, natureza)})"
    return Tabela(nome, titulo, list(COLUNAS_METRICAS), linhas, natureza,
                  origens=inv.origens(usadas.values()), pendencias=pendencias, notas=notas)


# --------------------------------------------------------------------------- #
# Comparações                                                                 #
# --------------------------------------------------------------------------- #
def comparacoes(inventario: inv.Inventario, *, natureza: str = "cientifica",
                sistemas: Sequence[fase.Sistema] | None = None,
                pares: Sequence[tuple[str, str, str]] = fase.COMPARACOES,
                n_boot: int = significancia.N_BOOT) -> tuple[Tabela, dict[str, dict[str, Any]]]:
    """As comparações do protocolo no TEST (bootstrap pareado do F1 de
    `negation_of` e McNemar exato) e os relatórios completos de cada uma."""
    sistemas = list(sistemas or fase.sistemas())
    por_chave = {s.chave: s for s in sistemas}
    usadas, pendencias = _disponiveis(inventario, sistemas, "test", natureza)
    colunas = ["Comparação (A − B)", "Tipo", "F1 A", "F1 B", "Diferença", "IC95% inferior",
               "IC95% superior", "p (bootstrap)", "p (McNemar)", "Leitura"]
    linhas, relatorios, origens = [], {}, {}
    for a, b, tipo in pares:
        sa, sb = por_chave[a], por_chave[b]
        rotulo = f"{sa.rotulo} × {sb.rotulo}"
        if a not in usadas or b not in usadas:
            linhas.append([rotulo, tipo] + [PENDENTE] * 8)
            continue
        ea, eb = usadas[a], usadas[b]
        pa, pb = ea.predicoes("test"), eb.predicoes("test")
        seed = protocolo.semente_bootstrap(pa.get("seed"), pb.get("seed"))
        r = significancia.comparar(pa, pb, seed=seed, n_boot=n_boot)
        r["model_a"], r["model_b"] = ea.nome, eb.nome
        relatorios[f"{ea.nome}__vs__{eb.nome}"] = r
        origens[ea.nome], origens[eb.nome] = ea, eb
        bs, k = r["paired_bootstrap"], f"comparacao.{a}_vs_{b}"
        so_neg = fase.SO_NEG in (sa.comparabilidade, sb.comparabilidade)
        linhas.append([rotulo, tipo, Numero(r["target_f1"]["a"], chave=f"{k}.f1_a"),
                       Numero(r["target_f1"]["b"], chave=f"{k}.f1_b"),
                       Numero(bs["observed_diff"], chave=f"{k}.diferenca", com_sinal=True),
                       Numero(bs["ci95_low"], chave=f"{k}.ic95_inferior", com_sinal=True),
                       Numero(bs["ci95_high"], chave=f"{k}.ic95_superior", com_sinal=True),
                       Numero(bs["p_value"], chave=f"{k}.p_bootstrap"),
                       NAO_COMPARAVEL if so_neg else Numero(r["mcnemar"]["p_value"], chave=f"{k}.p_mcnemar"),
                       fase.leitura(bs["ci95_low"], bs["ci95_high"])])
    notas = [f"Bootstrap pareado do F1 de negation_of no TEST ({n_boot} reamostras, IC95% por percentis; "
             "semente pela regra de protocolo.semente_bootstrap) e McNemar exato.",
             "As duas comparações principais são reportadas sem correção de multiplicidade.",
             "Com uma semente, a incerteza é a da amostra do TEST, dados os modelos treinados; "
             "não mede a variação do treino.",
             "n/c: McNemar não comparável quando um dos lados é o restrito."]
    tabela = Tabela("comparacoes_test", f"Comparações no TEST ({fase.ROTULO_NATUREZA.get(natureza, natureza)})",
                    colunas, linhas, natureza, origens=inv.origens(origens.values()),
                    pendencias=pendencias, notas=notas)
    return tabela, relatorios


# --------------------------------------------------------------------------- #
# Limiares escolhidos no DEV                                                  #
# --------------------------------------------------------------------------- #
CAMPO_LIMIAR = {"baseline_calibrado": "limiar_t", "two_stage": "limiar_tau"}


def limiares(inventario: inv.Inventario, *, natureza: str = "cientifica",
             sistemas: Sequence[fase.Sistema] | None = None) -> Tabela:
    """τ (Two-Stage) e t (baseline calibrado) congelados no DEV, conferidos:
    o limiar gravado é o que a regra escolhe na tabela da busca e o mesmo do
    sidecar do DEV; F1 de `negation_of` e macro-F1 do DEV nesse limiar."""
    sistemas = [s for s in (sistemas or fase.sistemas()) if s.chave in CAMPO_LIMIAR]
    usadas, pendencias = _disponiveis(inventario, sistemas, "dev", natureza)
    linhas = []
    for s in sistemas:
        ex = usadas.get(s.chave)
        if ex is None:
            linhas.append([s.rotulo, PENDENTE, PENDENTE, PENDENTE])
            continue
        arquivo = ex.pasta / busca_limiar.ARQUIVO
        if not arquivo.is_file():
            raise ErroTabela(f"{ex.nome} não tem {busca_limiar.ARQUIVO}")
        gravado = ler_json(arquivo)
        valor = float(gravado["limiar"])
        if busca_limiar.escolhido_da_tabela(gravado["tabela"]) != valor:
            raise ErroTabela(f"{ex.nome}: o limiar gravado não é o que a regra escolhe na tabela da busca")
        sidecar = ex.predicoes("dev")
        if sidecar.get(CAMPO_LIMIAR[s.chave]) != valor:
            raise ErroTabela(f"{ex.nome}: o limiar do sidecar do DEV difere do limiar.json")
        linha = next(t for t in gravado["tabela"] if float(t["limiar"]) == valor)
        k = f"limiar.{s.chave}"
        linhas.append([s.rotulo, Numero(valor, casas=2, chave=f"{k}.valor"),
                       Numero(linha["f1_negation_of"], chave=f"{k}.dev_f1_negation_of"),
                       Numero(linha["macro_f1"], chave=f"{k}.dev_macro_f1")])
    return Tabela("limiares_dev", "Limiares escolhidos no DEV (τ e t)",
                  ["Sistema", "Limiar", "F1 (negation_of) no DEV", "Macro-F1 no DEV"], linhas, natureza,
                  origens=inv.origens(usadas.values()), pendencias=pendencias,
                  notas=["Valores do DEV no limiar escolhido: otimistas, só diagnóstico."])


# --------------------------------------------------------------------------- #
# Proveniência e partições                                                    #
# --------------------------------------------------------------------------- #
def proveniencia(execucoes: Iterable[inv.Execucao], natureza: str) -> Tabela:
    linhas = []
    execucoes = list(execucoes)
    for ex in execucoes:
        commit = (ex.proveniencia or {}).get("commit")
        dev, test = ex.sidecars.get("dev"), ex.sidecars.get("test") if ex.tem("test") else None
        linhas.append([ex.nome, ex.estrategia, commit[:12] if commit else "—",
                       dev["sha256"][:12] if dev else "—", test["sha256"][:12] if test else "—",
                       Numero(ex.test_avaliacoes, inteiro=True), ex.codigo[:12] if ex.codigo else "—"])
    return Tabela("proveniencia", "Proveniência das execuções usadas",
                  ["Execução", "Estratégia", "Commit dos pesos", "SHA-256 DEV", "SHA-256 TEST",
                   "Avaliações do TEST", "Código"], linhas, natureza, origens=inv.origens(execucoes),
                  notas=["SHA-256 dos sidecars (12 primeiros caracteres); commit do BioBERTpt (12); "
                         "identidade do código que produziu a execução (12; util.identidade_codigo)."])


def particoes_tabela(pasta_particoes: str | Path = caminhos.PARTICOES) -> Tabela:
    """Documentos, relações e SHA-256 de cada partição (do MANIFEST, depois de
    conferir as partições contra ele)."""
    divergencias = particoes.conferir_particoes(pasta_particoes)
    if divergencias:
        raise ErroTabela("; ".join(divergencias))
    manifesto = particoes.ler_manifesto(pasta_particoes)
    linhas = []
    for nome in particoes.PARTICOES:
        s = manifesto["splits"][nome]
        rel = s["n_relations"]
        linhas.append([nome.upper(), Numero(s["n_records"], inteiro=True, chave=f"particao.{nome}.documentos"),
                       Numero(rel.get("negation_of", 0), inteiro=True, chave=f"particao.{nome}.negation_of"),
                       Numero(rel.get("associated_with", 0), inteiro=True,
                              chave=f"particao.{nome}.associated_with"), s["sha256"]])
    origem = {"execucao": "particoes congeladas", "estrategia": "dados", "natureza": "dados",
              "manifest_sha256": sha256_arquivo(Path(pasta_particoes) / "MANIFEST.json"),
              "versao_dados": versao_dados.versao_da_pasta(pasta_particoes)}
    return Tabela("particoes", "Partições congeladas", ["Partição", "Documentos", "negation_of",
                                                        "associated_with", "SHA-256"],
                  linhas, "dados", origens=[origem])


# --------------------------------------------------------------------------- #
# Registro de valores e gravação                                              #
# --------------------------------------------------------------------------- #
def registro_valores(tabelas: Iterable[Tabela]) -> dict[str, dict[str, Any]]:
    """chave → valor, texto (ABNT), tabela e execuções de origem. A mesma chave
    com valores diferentes em duas tabelas é recusada."""
    registro: dict[str, dict[str, Any]] = {}
    for t in tabelas:
        for chave, numero in t.valores().items():
            item = {"valor": numero.valor, "texto": numero.texto(virgula=True), "tabela": t.nome,
                    "natureza": t.natureza, "origens": [o["execucao"] for o in t.origens]}
            if chave in registro and registro[chave]["valor"] != numero.valor:
                raise ErroTabela(f"a chave {chave} tem valores diferentes em {registro[chave]['tabela']} e {t.nome}")
            registro.setdefault(chave, item)
    return dict(sorted(registro.items()))


def valores_tex(registro: Mapping[str, Mapping[str, Any]]) -> str:
    """Macros: `\\rv{chave}` expande no texto ABNT do valor."""
    linhas = ["% " + AVISO, r"\providecommand{\rv}[1]{\csname rv:#1\endcsname}"]
    for chave, item in registro.items():
        texto = item["texto"].replace("−", "$-$")
        linhas.append(r"\expandafter\def\csname rv:" + chave + r"\endcsname{" + texto + "}")
    return "\n".join(linhas) + "\n"


def gravar(tabelas: Sequence[Tabela], destino: str | Path, *, grupo: str,
           extras: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Grava cada tabela (`tabelas/<nome>.tex|.csv|.md`) e, para o grupo
    (`resultados`, `curvas`, `eda`...), `valores/<grupo>.json`,
    `valores/<grupo>.tex` e `indice/<grupo>.json` com o status, as pendências
    e a proveniência de cada tabela. Devolve o conteúdo do índice.

    Um destino guarda uma natureza só: gravar um grupo de outra natureza num
    destino que já tem outra (fora os dados do corpus) é recusado."""
    destino = Path(destino)
    naturezas = {t.natureza for t in tabelas} - {"dados"}
    if len(naturezas) > 1:
        raise ErroTabela(f"tabelas de naturezas diferentes no mesmo grupo: {sorted(naturezas)}")
    natureza = next(iter(naturezas), "dados")
    if natureza != "dados":
        for outro in sorted((destino / "indice").glob("*.json")):
            anterior = ler_json(outro).get("natureza")
            if anterior not in (None, "dados", natureza):
                raise ErroTabela(f"{destino} já tem resultados de natureza {anterior!r} ({outro.name}); "
                                 f"use outro destino para {natureza!r}")
    indice = {}
    for t in tabelas:
        for sufixo, texto in (("tex", t.latex()), ("csv", t.csv()), ("md", t.markdown())):
            gravar_texto(destino / "tabelas" / f"{t.nome}.{sufixo}", texto)
        indice[t.nome] = {"titulo": t.titulo, "status": t.status, "natureza": t.natureza,
                          "pendencias": t.pendencias, "notas": t.notas, "origens": t.origens,
                          "sha256_tex": sha256_arquivo(destino / "tabelas" / f"{t.nome}.tex")}
    registro = registro_valores(tabelas)
    gravar_json(destino / "valores" / f"{grupo}.json", registro)
    gravar_texto(destino / "valores" / f"{grupo}.tex", valores_tex(registro))
    conteudo = {"aviso": AVISO, "grupo": grupo, "natureza": natureza, "fase": fase.ROTULO_FASE,
                "tabelas": indice, **(extras or {})}
    gravar_json(destino / "indice" / f"{grupo}.json", conteudo)
    return conteudo


def ler_valores(destino: str | Path) -> dict[str, dict[str, Any]]:
    """Todos os valores registrados num destino (todos os grupos). A mesma
    chave com valores diferentes em dois grupos é recusada."""
    registro: dict[str, dict[str, Any]] = {}
    for arquivo in sorted((Path(destino) / "valores").glob("*.json")):
        for chave, item in ler_json(arquivo).items():
            if chave in registro and registro[chave]["valor"] != item["valor"]:
                raise ErroTabela(f"a chave {chave} tem valores diferentes em dois grupos ({arquivo.name})")
            registro.setdefault(chave, item)
    return dict(sorted(registro.items()))


def ler_origens(destino: str | Path) -> list[dict[str, Any]]:
    """As origens (execuções) registradas nos índices de um destino."""
    vistas, saida = set(), []
    for arquivo in sorted((Path(destino) / "indice").glob("*.json")):
        for tabela in ler_json(arquivo)["tabelas"].values():
            for o in tabela["origens"]:
                if o.get("natureza") != "dados" and o["execucao"] not in vistas:
                    vistas.add(o["execucao"])
                    saida.append(o)
    return saida
