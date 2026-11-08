"""Adjudicação humana dos falsos positivos de `negation_of` (análise secundária).

Prepara para revisão humana os falsos positivos de `negation_of` de uma execução
do código novo no DEV (pares preditos `negation_of` cujo rótulo anotado é
outro) e, depois de congeladas as decisões humanas, reconcilia essas decisões
com as predições e os rótulos anotados. São três subcomandos:

    python -m reclin.analises.adjudicacao_negation_of preparar \\
        --execucao PASTA_DA_EXECUCAO --saida PASTA_DA_ANALISE \\
        [--controles-vp N] [--semente S] [--registro-origem TEXTO]
    python -m reclin.analises.adjudicacao_negation_of congelar \\
        --rastreabilidade PASTA_DA_ANALISE/restrito/rastreabilidade.json \\
        --planilha A.xlsx [--planilha B.xlsx] [--consenso C.xlsx] \\
        --saida PASTA_DA_ANALISE/restrito/decisoes_congeladas.json
    python -m reclin.analises.adjudicacao_negation_of reconciliar \\
        --rastreabilidade .../rastreabilidade.json --congeladas .../decisoes_congeladas.json \\
        --saida .../reconciliacao.json [--markdown .../reconciliacao.md]

Cegueira. A planilha de `cega/` não contém a relação predita, as
probabilidades, o rótulo anotado, a condição de falso positivo, os
identificadores do corpus nem classificações de análises anteriores. Como todo
falso positivo tem, por definição, um rótulo anotado diferente de
`negation_of`, uma planilha só com eles revelaria a anotação. Por isso eles são
misturados a controles: todos os falsos negativos e uma amostra, com semente
fixa, dos verdadeiros positivos de `negation_of`. Na planilha, nada distingue
um grupo do outro. Documentos e casos recebem códigos neutros (`D01`, `C001`),
e a ordem dos documentos é sorteada.

Rastreabilidade. `restrito/rastreabilidade.json` liga cada código neutro ao
documento, aos ids e offsets das entidades, ao índice no conjunto de
avaliação do DEV, ao grupo (falso positivo ou controle), ao rótulo anotado, ao
predito e às probabilidades, e registra a proveniência: SHA-256 dos arquivos da
execução, identidade do código, semente, partição, MANIFEST e métricas
oficiais. Esse arquivo não vai para quem adjudica.

Congelamento. `congelar` só aceita planilhas completas, com o texto dos casos
intacto (conferido pelo SHA-256 de cada caso) e decisões válidas, e grava as
decisões num arquivo que não pode ser sobrescrito. `reconciliar` só lê
decisões congeladas.

Nada aqui treina, faz inferência, lê o TEST nem altera partições, rótulos,
execuções ou métricas oficiais. Nenhuma categoria é atribuída
automaticamente. A precisão "secundária" do `reconciliar` é exploratória e
fica separada da oficial.
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reclin import particoes, tarefa
from reclin.analises.alinhamento import EXATO, alinhar_entidade
from reclin.execucao import diretorio, inventario
from reclin.tarefa import ID2LABEL, LABELS, NEG, Candidato, Documento
from reclin.util import caminhos
from reclin.util.io import gravar_json, ler_json, sha256_arquivo, sha256_json

log = logging.getLogger("reclin.analises.adjudicacao_negation_of")

ESQUEMA = "reclin-adjudicacao-negation-of/1"
PARTICAO = "dev"                      # o TEST nunca é usado aqui
SEMENTE = 42
CONTEXTO = 250                        # caracteres antes e depois do trecho com as entidades

FALSO_POSITIVO = "falso_positivo"
CONTROLE_VP = "controle_verdadeiro_positivo"
CONTROLE_FN = "controle_falso_negativo"
GRUPOS = (FALSO_POSITIVO, CONTROLE_VP, CONTROLE_FN)

MARCAS = {"e1": ("⟦E1⟧", "⟦/E1⟧"), "e2": ("⟦E2⟧", "⟦/E2⟧")}

# Decisão principal. A planilha usa a formulação neutra (sobre o par, sem dizer
# se ele foi anotado ou predito); `NOME_PARA_FALSO_POSITIVO` é a leitura de cada
# código quando o caso é um falso positivo (as categorias da análise).
CATEGORIAS = {
    "1": "Negação válida: E1 é pista de negação e E2 é o alvo negado por ela.",
    "2": "Alvo incorreto: há negação expressa por E1, mas o alvo é outra entidade, não E2.",
    "3": "Escopo incorreto: E1 é pista de negação, mas E2 está fora da sua abrangência "
         "(ou a abrangência foi estendida ou restringida indevidamente).",
    "4": "Sem relação de negação: não há evidência suficiente de que E1 negue E2.",
}
NAO_CONCLUSIVAS = {
    "inconclusivo": "Contexto insuficiente ou ambiguidade não resolvida.",
    "revisao_necessaria": "O caso exige discussão adicional (segunda revisão).",
}
DECISOES = (*CATEGORIAS, *NAO_CONCLUSIVAS)
NOME_PARA_FALSO_POSITIVO = {
    "1": "negacao_clinicamente_correta_ausente_na_anotacao",
    "2": "alvo_incorreto",
    "3": "escopo_incorreto",
    "4": "relacao_incorreta_ou_ausencia_de_negacao",
    "inconclusivo": "inconclusivo",
    "revisao_necessaria": "revisao_necessaria",
}

COLUNAS_CASO = ("caso", "documento", "e1_texto", "e1_tipo", "e2_texto", "e2_tipo", "ordem",
                "contexto_anterior", "trecho_marcado", "contexto_posterior", "observacao_texto")
COLUNAS_DECISAO = ("adjudicador", "data", "decisao_principal", "problemas_secundarios",
                   "justificativa", "observacoes")
TITULOS = {
    "caso": "caso", "documento": "documento",
    "e1_texto": "E1 — texto (possível pista)", "e1_tipo": "E1 — tipo",
    "e2_texto": "E2 — texto (possível alvo)", "e2_tipo": "E2 — tipo",
    "ordem": "ordem no texto", "contexto_anterior": "contexto anterior",
    "trecho_marcado": "trecho com as entidades marcadas", "contexto_posterior": "contexto posterior",
    "observacao_texto": "observação sobre o texto",
    "adjudicador": "adjudicador", "data": "data (AAAA-MM-DD)",
    "decisao_principal": "decisão principal", "problemas_secundarios": "problemas secundários",
    "justificativa": "justificativa", "observacoes": "observações",
}
ABA_CASOS, ABA_NOTAS, ABA_CATEGORIAS, ABA_INSTRUCOES = "Casos", "Notas", "Categorias", "Instruções"
ABAS = (ABA_INSTRUCOES, ABA_CASOS, ABA_NOTAS, ABA_CATEGORIAS)
PLANILHA = "planilha_adjudicacao.xlsx"
RASTREABILIDADE = "rastreabilidade.json"
RASTREABILIDADE_CSV = "rastreabilidade.csv"

# Termos que não podem aparecer nas células que não são texto clínico.
REVELADORES = re.compile(r"negation_of|associated_with|no_relation|falso|verdadeiro|positivo|negativo|"
                         r"previs|predi|probab|confian|modelo|score|acert|erro|controle",
                         re.IGNORECASE)
ROTULOS_DA_TAREFA = re.compile("|".join(LABELS))


class ErroAdjudicacao(ValueError):
    """Entrada recusada: execução não verificada, planilha alterada ou incompleta, saída já existente."""


# --------------------------------------------------------------------------- #
# Preparação                                                                  #
# --------------------------------------------------------------------------- #
def selecionar(y_true: Sequence[int], y_pred: Sequence[int], *, n_controles_vp: int | None,
               semente: int = SEMENTE) -> dict[int, str]:
    """Índice do candidato → grupo. Todos os falsos positivos e falsos
    negativos de `negation_of`; dos verdadeiros positivos, uma amostra de
    `n_controles_vp` (por padrão, tantos quantos são os falsos positivos)."""
    fp = [i for i, (a, b) in enumerate(zip(y_true, y_pred)) if b == NEG and a != NEG]
    vp = [i for i, (a, b) in enumerate(zip(y_true, y_pred)) if b == NEG and a == NEG]
    fn = [i for i, (a, b) in enumerate(zip(y_true, y_pred)) if b != NEG and a == NEG]
    n = len(fp) if n_controles_vp is None else n_controles_vp
    if not 0 <= n <= len(vp):
        raise ErroAdjudicacao(f"--controles-vp deve estar entre 0 e {len(vp)} (verdadeiros positivos)")
    amostra = random.Random(semente).sample(vp, n)
    grupos = {i: FALSO_POSITIVO for i in fp}
    grupos.update({i: CONTROLE_VP for i in amostra})
    grupos.update({i: CONTROLE_FN for i in fn})
    return grupos


def _marcar(texto: str, c: Candidato) -> tuple[int, int, str, str]:
    """(início, fim, trecho marcado, observação) do trecho que vai do início da
    primeira entidade ao fim da última. Os marcadores são inseridos pelos
    offsets; o texto entre eles fica como está no corpus."""
    e1, e2 = c["e1"], c["e2"]
    ini, fim = min(e1["start"], e2["start"]), max(e1["end"], e2["end"])
    sobrepostas = e1["start"] < e2["end"] and e2["start"] < e1["end"]
    if sobrepostas:
        return ini, fim, texto[ini:fim], "E1 e E2 se sobrepõem no texto; o trecho não tem marcadores."
    pontos = sorted([(e1["start"], 1, MARCAS["e1"][0]), (e1["end"], 0, MARCAS["e1"][1]),
                     (e2["start"], 1, MARCAS["e2"][0]), (e2["end"], 0, MARCAS["e2"][1])])
    partes, cursor = [], ini
    for pos, _, marca in pontos:
        partes.append(texto[cursor:pos])
        partes.append(marca)
        cursor = pos
    partes.append(texto[cursor:fim])
    return ini, fim, "".join(partes), ""


def _observacao_alinhamento(texto: str, c: Candidato) -> tuple[list[str], str]:
    status, notas = [], []
    for nome in ("e1", "e2"):
        r = alinhar_entidade(texto, c[nome])
        status.append(r["status"])
        if r["status"] != EXATO:
            ent = c[nome]
            notas.append(f"O texto anotado de {nome.upper()} difere do trecho nos seus offsets "
                         f"(trecho nos offsets: {texto[ent['start']:ent['end']]!r}).")
    return status, " ".join(notas)


def montar_casos(candidatos: Sequence[Candidato], documentos: Mapping[str, Documento],
                 grupos: Mapping[int, str], *, semente: int = SEMENTE,
                 contexto: int = CONTEXTO) -> tuple[list[dict], list[dict], list[dict]]:
    """(casos cegos, notas cegas, registros restritos), na ordem da planilha.

    A ordem dos documentos é sorteada (semente própria, derivada de `semente`);
    dentro de um documento, os casos seguem a posição no texto."""
    por_doc: dict[str, list[int]] = {}
    for i in sorted(grupos):
        por_doc.setdefault(candidatos[i]["doc_id"], []).append(i)
    ordem_docs = sorted(por_doc)
    random.Random(f"{semente}:documentos").shuffle(ordem_docs)
    largura_doc = max(2, len(str(len(ordem_docs))))
    largura_caso = max(3, len(str(len(grupos))))

    cegos, notas, restritos = [], [], []
    n = 0
    for k, doc_id in enumerate(ordem_docs, start=1):
        documento = f"D{k:0{largura_doc}d}"
        texto = documentos[doc_id]["text"]
        for marca in (m for par in MARCAS.values() for m in par):
            if marca in texto:
                raise ErroAdjudicacao(f"o marcador {marca!r} já ocorre no texto de um documento")
        indices = sorted(por_doc[doc_id], key=lambda i: (
            min(candidatos[i]["e1"]["start"], candidatos[i]["e2"]["start"]),
            max(candidatos[i]["e1"]["end"], candidatos[i]["e2"]["end"]),
            candidatos[i]["e1"]["start"], candidatos[i]["e2"]["start"]))
        casos_do_doc = []
        for i in indices:
            n += 1
            caso = f"C{n:0{largura_caso}d}"
            c = candidatos[i]
            ini, fim, trecho, obs_sobreposicao = _marcar(texto, c)
            status, obs_alinhamento = _observacao_alinhamento(texto, c)
            cego = {
                "caso": caso, "documento": documento,
                "e1_texto": c["e1"]["text"], "e1_tipo": c["e1"]["type"],
                "e2_texto": c["e2"]["text"], "e2_tipo": c["e2"]["type"],
                "ordem": "E1 antes de E2" if c["e1"]["start"] < c["e2"]["start"] else "E2 antes de E1",
                "contexto_anterior": texto[max(0, ini - contexto):ini],
                "trecho_marcado": trecho,
                "contexto_posterior": texto[fim:fim + contexto],
                "observacao_texto": " ".join(x for x in (obs_sobreposicao, obs_alinhamento) if x),
            }
            cegos.append(cego)
            casos_do_doc.append(caso)
            restritos.append({
                "caso": caso, "documento_neutro": documento, "doc_id": doc_id,
                "indice_candidato": i, "grupo": grupos[i],
                "e1": dict(c["e1"]), "e2": dict(c["e2"]),
                "distancia": tarefa.entity_gap(c["e1"], c["e2"]),
                "alinhamento": status,
                "sha256_caso_cego": sha256_json(cego),
                "trecho_marcado": trecho,
            })
        notas.append({"documento": documento, "casos": "; ".join(casos_do_doc), "texto": texto})
    return cegos, notas, restritos


def _sha_execucao(pasta: Path) -> dict[str, str]:
    return {p.name: sha256_arquivo(p) for p in sorted(pasta.iterdir()) if p.is_file()}


def preparar(pasta_execucao: str | Path, saida: str | Path, *,
             pasta_particoes: str | Path = caminhos.PARTICOES, n_controles_vp: int | None = None,
             semente: int = SEMENTE, registro_origem: str | None = None) -> dict[str, Any]:
    """Gera `saida/cega/planilha_adjudicacao.xlsx` e `saida/restrito/rastreabilidade.{json,csv}`.

    Recusa a execução se o inventário apontar qualquer problema, se o `y_true`
    do sidecar não for o do conjunto do DEV ou se as contagens não forem as do
    `metricas.json`. Não sobrescreve nada: `saida` precisa estar vazia ou não existir."""
    pasta, saida, pasta_particoes = Path(pasta_execucao), Path(saida), Path(pasta_particoes)
    if saida.exists() and any(saida.iterdir()):
        raise ErroAdjudicacao(f"{saida} já existe e não está vazia: nada é sobrescrito")

    ex = inventario.inspecionar(pasta, pasta_particoes=pasta_particoes)
    if ex.problemas:
        raise ErroAdjudicacao("execução recusada pelo inventário: " + "; ".join(ex.problemas))
    if PARTICAO not in ex.sidecars:
        raise ErroAdjudicacao(f"a execução não tem predições válidas do {PARTICAO.upper()}")
    divergencias = particoes.conferir_particoes(pasta_particoes, [PARTICAO])
    if divergencias:
        raise ErroAdjudicacao("partição divergente do MANIFEST: " + "; ".join(divergencias))

    config = diretorio.ler_config(pasta)
    conjunto = config["conjuntos"][PARTICAO]
    docs = particoes.ler_particao(pasta_particoes, PARTICAO)
    candidatos = tarefa.candidatos(docs, max_gap=conjunto["max_gap"])
    sidecar = diretorio.ler_predicoes(pasta, PARTICAO)
    if list(sidecar["labels"]) != list(LABELS):
        raise ErroAdjudicacao(f"rótulos do sidecar {sidecar['labels']} diferentes de {list(LABELS)}")
    if sidecar["y_true"] != tarefa.y_true(candidatos):
        raise ErroAdjudicacao("o y_true do sidecar não é o dos candidatos do DEV")

    y_true, y_pred = sidecar["y_true"], sidecar["y_pred"]
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == b == NEG)
    fp = sum(1 for a, b in zip(y_true, y_pred) if b == NEG and a != NEG)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == NEG and b != NEG)
    oficial = diretorio.ler_metricas(pasta)[PARTICAO]["por_classe"]["negation_of"]
    if (oficial["tp"], oficial["fp"], oficial["fn"]) != (tp, fp, fn):
        raise ErroAdjudicacao(f"contagens recalculadas (tp={tp}, fp={fp}, fn={fn}) diferentes das do metricas.json")

    grupos = selecionar(y_true, y_pred, n_controles_vp=n_controles_vp, semente=semente)
    por_id = {d["doc_id"]: d for d in docs}
    cegos, notas, restritos = montar_casos(candidatos, por_id, grupos, semente=semente)
    probs = sidecar.get("probs")
    for r in restritos:
        i = r["indice_candidato"]
        r["rotulo_anotado"] = ID2LABEL[y_true[i]]
        r["rotulo_predito"] = ID2LABEL[y_pred[i]]
        r["probabilidades"] = dict(zip(LABELS, probs[i])) if probs else None

    manifesto = pasta_particoes / particoes.ARQUIVO_MANIFESTO
    rastreabilidade = {
        "esquema": ESQUEMA,
        "natureza": "rastreabilidade restrita à análise; não entregar a quem adjudica",
        "origem": {
            "execucao": config["nome"], "estrategia": ex.estrategia, "seed": ex.seed, "encoder": ex.encoder,
            "identidade_codigo": ex.codigo, "versao_dados": ex.versao_dados,
            "arquivos_sha256": _sha_execucao(pasta),
            "registro_origem": registro_origem,
            "particao": PARTICAO, "particao_sha256": conjunto["particao_sha256"],
            "manifest_sha256": sha256_arquivo(manifesto),
            "conjunto_avaliacao": {k: conjunto[k] for k in ("n_candidatos", "candidatos_sha256",
                                                            "y_true_sha256", "max_gap")},
        },
        "contagens": {
            "tp": tp, "fp": fp, "fn": fn,
            "fp_por_rotulo_anotado": dict(sorted(Counter(r["rotulo_anotado"] for r in restritos
                                                         if r["grupo"] == FALSO_POSITIVO).items())),
            "documentos_com_fp": len({r["doc_id"] for r in restritos if r["grupo"] == FALSO_POSITIVO}),
            "casos_por_grupo": {g: sum(1 for r in restritos if r["grupo"] == g) for g in GRUPOS},
        },
        "metricas_oficiais_negation_of": oficial,
        "amostragem": {"semente": semente, "controles_vp": sum(1 for g in grupos.values() if g == CONTROLE_VP),
                       "vp_disponiveis": tp, "controles_fn": fn, "contexto_caracteres": CONTEXTO},
        "planilha": {"arquivo": f"cega/{PLANILHA}", "n_casos": len(cegos),
                     "sha256_casos_cegos": sha256_json(cegos), "sha256_notas_cegas": sha256_json(notas)},
        "casos": restritos,
    }

    (saida / "cega").mkdir(parents=True, exist_ok=True)
    (saida / "restrito").mkdir(parents=True, exist_ok=True)
    gravar_planilha(saida / "cega" / PLANILHA, cegos, notas)
    gravar_json(saida / "restrito" / RASTREABILIDADE, rastreabilidade)
    _gravar_csv(saida / "restrito" / RASTREABILIDADE_CSV, restritos)
    verificacao = verificar_planilha_cega(saida / "cega" / PLANILHA, cegos, notas, restritos)
    if not verificacao["ok"]:
        raise ErroAdjudicacao("verificação da planilha cega falhou: " + "; ".join(verificacao["falhas"]))
    return {"rastreabilidade": rastreabilidade, "verificacao": verificacao}


def _gravar_csv(caminho: Path, restritos: Sequence[Mapping[str, Any]]) -> None:
    import csv
    colunas = ["caso", "documento_neutro", "doc_id", "indice_candidato", "grupo", "rotulo_anotado",
               "rotulo_predito", "p_negation_of", "p_associated_with", "p_no_relation",
               "e1_id", "e1_start", "e1_end", "e1_texto", "e1_tipo",
               "e2_id", "e2_start", "e2_end", "e2_texto", "e2_tipo", "distancia", "alinhamento", "sha256_caso_cego"]
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(colunas)
        for r in restritos:
            p = r["probabilidades"] or {}
            w.writerow([r["caso"], r["documento_neutro"], r["doc_id"], r["indice_candidato"], r["grupo"],
                        r["rotulo_anotado"], r["rotulo_predito"],
                        p.get("negation_of"), p.get("associated_with"), p.get("no_relation"),
                        r["e1"]["id"], r["e1"]["start"], r["e1"]["end"], r["e1"]["text"], r["e1"]["type"],
                        r["e2"]["id"], r["e2"]["start"], r["e2"]["end"], r["e2"]["text"], r["e2"]["type"],
                        r["distancia"], "/".join(r["alinhamento"]), r["sha256_caso_cego"]])


# --------------------------------------------------------------------------- #
# Planilha cega                                                               #
# --------------------------------------------------------------------------- #
def instrucoes() -> list[str]:
    return [
        "Adjudicação de relações de negação — ficha de revisão",
        "",
        "Cada linha da aba Casos é um par de entidades (E1, E2) de uma nota clínica do SemClinBr. "
        "E1 é uma possível pista de negação; E2, um possível alvo.",
        "Decida, apenas pelo texto, se E1 nega E2. Até as suas decisões serem congeladas, não consulte "
        "outras fontes sobre estes pares (anotações do corpus, saídas de sistemas automáticos ou análises "
        "anteriores) nem as decisões de outro adjudicador.",
        "Casos da mesma nota têm o mesmo código de documento; a nota completa está na aba Notas. "
        "Analise-os em conjunto quando o contexto de um depender do outro.",
        "No trecho marcado, ⟦E1⟧…⟦/E1⟧ e ⟦E2⟧…⟦/E2⟧ delimitam as entidades pelos offsets do corpus; "
        "o restante do texto (acentos, pontuação, abreviações) está exatamente como no corpus. "
        "Os contextos anterior e posterior têm até 250 caracteres e não foram cortados por frase.",
        "",
        "Campos a preencher (colunas L a Q da aba Casos):",
        "• adjudicador — obrigatório; o mesmo identificador em todas as linhas da sua cópia.",
        "• data — opcional (AAAA-MM-DD).",
        "• decisão principal — obrigatória: 1, 2, 3, 4, inconclusivo ou revisao_necessaria (ver aba Categorias). "
        "Use 1 a 4 só quando o caso for conclusivo.",
        "• problemas secundários — opcional: outros códigos de 1 a 4 que também se apliquem, separados por ';', "
        "diferentes da decisão principal.",
        "• justificativa — texto livre (recomendado em todos os casos).",
        "• observações — texto livre.",
        "",
        "Não altere as colunas A a K, os códigos dos casos nem a ordem das linhas: a conferência recusa "
        "planilhas com o texto dos casos alterado.",
        "Com dois adjudicadores, cada um preenche a própria cópia desta planilha, sem ver a do outro. "
        "A decisão consensual, se houver, é registrada numa terceira cópia (adjudicador = consenso) "
        "só depois de as duas decisões individuais estarem fechadas.",
        "A decisão de um único adjudicador não é consenso nem validação clínica definitiva.",
    ]


def _celula(ws, linha: int, coluna: int, valor: Any):
    cel = ws.cell(row=linha, column=coluna, value=valor)
    if isinstance(valor, str):
        cel.data_type = "s"          # texto começado por '=' não vira fórmula
    return cel


def gravar_planilha(caminho: Path, cegos: Sequence[Mapping[str, str]], notas: Sequence[Mapping[str, str]]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    wb.properties.title = "Adjudicação de relações de negação"
    wb.properties.creator = "reclin"
    negrito, quebra = Font(bold=True), Alignment(wrap_text=True, vertical="top")

    ws = wb.active
    ws.title = ABA_INSTRUCOES
    for i, linha in enumerate(instrucoes(), start=1):
        _celula(ws, i, 1, linha or None).alignment = quebra
    ws["A1"].font = negrito
    ws.column_dimensions["A"].width = 140

    ws = wb.create_sheet(ABA_CASOS)
    colunas = (*COLUNAS_CASO, *COLUNAS_DECISAO)
    for j, col in enumerate(colunas, start=1):
        _celula(ws, 1, j, TITULOS[col]).font = negrito
    for i, caso in enumerate(cegos, start=2):
        for j, col in enumerate(COLUNAS_CASO, start=1):
            _celula(ws, i, j, caso[col] or None).alignment = quebra
    larguras = {"caso": 8, "documento": 11, "e1_texto": 18, "e1_tipo": 16, "e2_texto": 22, "e2_tipo": 18,
                "ordem": 14, "contexto_anterior": 45, "trecho_marcado": 45, "contexto_posterior": 45,
                "observacao_texto": 22, "adjudicador": 14, "data": 12, "decisao_principal": 18,
                "problemas_secundarios": 16, "justificativa": 50, "observacoes": 30}
    for j, col in enumerate(colunas, start=1):
        ws.column_dimensions[get_column_letter(j)].width = larguras[col]
    ws.freeze_panes = "C2"
    j_dec = colunas.index("decisao_principal") + 1
    letra = get_column_letter(j_dec)
    dv = DataValidation(type="list", formula1='"' + ",".join(DECISOES) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Decisão inválida",
                        error="Use 1, 2, 3, 4, inconclusivo ou revisao_necessaria.")
    ws.add_data_validation(dv)
    dv.add(f"{letra}2:{letra}{len(cegos) + 1}")

    ws = wb.create_sheet(ABA_NOTAS)
    for j, titulo in enumerate(("documento", "casos", "texto completo da nota"), start=1):
        _celula(ws, 1, j, titulo).font = negrito
    for i, nota in enumerate(notas, start=2):
        for j, col in enumerate(("documento", "casos", "texto"), start=1):
            _celula(ws, i, j, nota[col]).alignment = quebra
    for letra_col, largura in zip("ABC", (11, 24, 140)):
        ws.column_dimensions[letra_col].width = largura
    ws.freeze_panes = "B2"

    ws = wb.create_sheet(ABA_CATEGORIAS)
    for j, titulo in enumerate(("código", "definição", "conclusivo"), start=1):
        _celula(ws, 1, j, titulo).font = negrito
    linhas = [(k, v, "sim") for k, v in CATEGORIAS.items()] + [(k, v, "não") for k, v in NAO_CONCLUSIVAS.items()]
    for i, (k, v, conc) in enumerate(linhas, start=2):
        _celula(ws, i, 1, k)
        _celula(ws, i, 2, v).alignment = quebra
        _celula(ws, i, 3, conc)
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 110
    wb.save(caminho)


def _valor(v: Any) -> str:
    return "" if v is None else str(v)


def verificar_planilha_cega(caminho: Path, cegos: Sequence[Mapping[str, str]], notas: Sequence[Mapping[str, str]],
                            restritos: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Relê a planilha gravada e confere: abas e títulos esperados, nada oculto
    nem comentado, texto dos casos e das notas idêntico ao da partição, colunas
    de decisão vazias, códigos neutros, nenhum id do corpus, e nenhum termo
    revelador fora do texto clínico."""
    from openpyxl import load_workbook

    falhas: list[str] = []
    wb = load_workbook(caminho)
    if tuple(wb.sheetnames) != ABAS:
        falhas.append(f"abas {wb.sheetnames} diferentes de {list(ABAS)}")
    for ws in wb.worksheets:
        if ws.sheet_state != "visible":
            falhas.append(f"aba {ws.title} não está visível")
        for linha in ws.iter_rows():
            for cel in linha:
                if cel.comment is not None:
                    falhas.append(f"comentário em {ws.title}!{cel.coordinate}")
                if isinstance(cel.value, str) and ROTULOS_DA_TAREFA.search(cel.value):
                    falhas.append(f"rótulo da tarefa em {ws.title}!{cel.coordinate}")
    if wb.defined_names and len(list(wb.defined_names)) > 0:
        falhas.append("a planilha tem nomes definidos")

    casos = wb[ABA_CASOS]
    colunas = (*COLUNAS_CASO, *COLUNAS_DECISAO)
    cabecalho = [_valor(c.value) for c in casos[1]]
    if cabecalho != [TITULOS[c] for c in colunas]:
        falhas.append("cabeçalho da aba Casos diferente do esperado")
    if casos.max_row != len(cegos) + 1:
        falhas.append(f"a aba Casos tem {casos.max_row - 1} linhas; esperado {len(cegos)}")
    for i, caso in enumerate(cegos, start=2):
        lidos = [_valor(casos.cell(row=i, column=j).value) for j in range(1, len(colunas) + 1)]
        if lidos[:len(COLUNAS_CASO)] != [caso[c] for c in COLUNAS_CASO]:
            falhas.append(f"texto do caso {caso['caso']} diferente do original após a gravação")
        if any(lidos[len(COLUNAS_CASO):]):
            falhas.append(f"colunas de decisão do caso {caso['caso']} não estão vazias")
    notas_ws = wb[ABA_NOTAS]
    for i, nota in enumerate(notas, start=2):
        if [_valor(notas_ws.cell(row=i, column=j).value) for j in (1, 2, 3)] != \
                [nota["documento"], nota["casos"], nota["texto"]]:
            falhas.append(f"nota {nota['documento']} diferente do original após a gravação")

    ids_corpus = {r["doc_id"] for r in restritos} | {r["e1"]["id"] for r in restritos} | \
                 {r["e2"]["id"] for r in restritos}
    for caso in cegos:
        if not re.fullmatch(r"C\d{3,}", caso["caso"]) or not re.fullmatch(r"D\d{2,}", caso["documento"]):
            falhas.append(f"código não neutro: {caso['caso']}/{caso['documento']}")
        if caso["caso"] in ids_corpus or caso["documento"] in ids_corpus:
            falhas.append(f"código coincide com um id do corpus: {caso['caso']}")
    nao_clinicas = [(ABA_INSTRUCOES, c.coordinate, c.value) for ln in wb[ABA_INSTRUCOES].iter_rows() for c in ln]
    nao_clinicas += [(ABA_CATEGORIAS, c.coordinate, c.value) for ln in wb[ABA_CATEGORIAS].iter_rows() for c in ln]
    nao_clinicas += [(ABA_CASOS, c.coordinate, c.value) for c in casos[1]]
    nao_clinicas += [(ABA_NOTAS, c.coordinate, c.value) for c in notas_ws[1]]
    nao_clinicas += [(ABA_CASOS, f"G{i}", casos.cell(row=i, column=COLUNAS_CASO.index("ordem") + 1).value)
                     for i in range(2, len(cegos) + 2)]
    for aba, coord, valor in nao_clinicas:
        if isinstance(valor, str) and REVELADORES.search(valor):
            falhas.append(f"termo revelador em {aba}!{coord}: {REVELADORES.search(valor).group(0)!r}")
    for campo in ("title", "subject", "description", "keywords", "category"):
        valor = getattr(wb.properties, campo)
        if isinstance(valor, str) and REVELADORES.search(valor):
            falhas.append(f"termo revelador nas propriedades do arquivo ({campo})")
    return {"ok": not falhas, "falhas": falhas, "n_casos": len(cegos), "n_documentos": len(notas)}


# --------------------------------------------------------------------------- #
# Congelamento                                                                #
# --------------------------------------------------------------------------- #
def _codigo(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)) and float(v).is_integer():
        return str(int(v))
    texto = _valor(v).strip().lower()
    texto = texto.replace("ã", "a").replace("á", "a").replace("ç", "c").replace(" ", "_")
    return texto


def _data(v: Any) -> str | None:
    if v is None or v == "":
        return None
    if isinstance(v, (datetime, date)):
        return v.date().isoformat() if isinstance(v, datetime) else v.isoformat()
    return str(v).strip()


def ler_decisoes(caminho: str | Path, casos: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Lê uma planilha preenchida e confere: os casos são os da rastreabilidade,
    na mesma ordem e com o texto intacto; toda linha tem adjudicador (o mesmo
    em todas) e decisão principal válida; os problemas secundários são códigos
    de 1 a 4 diferentes da principal."""
    from openpyxl import load_workbook

    caminho = Path(caminho)
    wb = load_workbook(caminho, data_only=True)
    if ABA_CASOS not in wb.sheetnames:
        raise ErroAdjudicacao(f"{caminho.name}: sem a aba {ABA_CASOS}")
    ws = wb[ABA_CASOS]
    colunas = (*COLUNAS_CASO, *COLUNAS_DECISAO)
    if [_valor(c.value) for c in ws[1]][:len(colunas)] != [TITULOS[c] for c in colunas]:
        raise ErroAdjudicacao(f"{caminho.name}: cabeçalho da aba Casos alterado")
    erros: list[str] = []
    avisos: list[str] = []
    decisoes: dict[str, dict[str, Any]] = {}
    adjudicadores = set()
    linhas = list(ws.iter_rows(min_row=2, max_col=len(colunas), values_only=True))
    linhas = [ln for ln in linhas if any(v not in (None, "") for v in ln)]
    if len(linhas) != len(casos):
        raise ErroAdjudicacao(f"{caminho.name}: {len(linhas)} linhas preenchidas; esperado {len(casos)}")
    for ln, esperado in zip(linhas, casos):
        valores = dict(zip(colunas, ln))
        caso = _valor(valores["caso"])
        if caso != esperado["caso"]:
            raise ErroAdjudicacao(f"{caminho.name}: caso {caso!r} fora da ordem (esperado {esperado['caso']})")
        cego = {c: _valor(valores[c]) for c in COLUNAS_CASO}
        if sha256_json(cego) != esperado["sha256_caso_cego"]:
            erros.append(f"{caso}: texto do caso alterado")
        adjudicador = _valor(valores["adjudicador"]).strip()
        if not adjudicador:
            erros.append(f"{caso}: sem adjudicador")
        adjudicadores.add(adjudicador)
        principal = _codigo(valores["decisao_principal"])
        if not principal:
            erros.append(f"{caso}: sem decisão principal")
        elif principal not in DECISOES:
            erros.append(f"{caso}: decisão principal inválida {valores['decisao_principal']!r}")
        secundarios = [_codigo(x) for x in re.split(r"[;,\s]+", _valor(valores["problemas_secundarios"])) if x.strip()]
        invalidos = [s for s in secundarios if s not in CATEGORIAS]
        if invalidos:
            erros.append(f"{caso}: problemas secundários inválidos {invalidos}")
        if principal in secundarios:
            erros.append(f"{caso}: o problema secundário repete a decisão principal")
        justificativa = _valor(valores["justificativa"]).strip()
        if not justificativa:
            avisos.append(f"{caso}: sem justificativa")
        decisoes[caso] = {"principal": principal, "secundarios": sorted(set(secundarios)),
                          "justificativa": justificativa or None,
                          "observacoes": _valor(valores["observacoes"]).strip() or None,
                          "data": _data(valores["data"])}
    if len(adjudicadores - {""}) > 1:
        erros.append(f"mais de um adjudicador na mesma planilha: {sorted(adjudicadores - {''})}")
    if erros:
        raise ErroAdjudicacao(f"{caminho.name}: " + "; ".join(erros))
    return {"arquivo": caminho.name, "sha256": sha256_arquivo(caminho),
            "adjudicador": next(iter(adjudicadores)), "n_casos": len(decisoes),
            "decisoes": decisoes, "avisos": avisos}


def congelar(rastreabilidade: str | Path, planilhas: Sequence[str | Path], saida: str | Path, *,
             consenso: str | Path | None = None) -> dict[str, Any]:
    """Grava as decisões humanas, conferidas, num arquivo que não é
    sobrescrito. Não lê nem mostra predições ou rótulos."""
    saida, rastreabilidade = Path(saida), Path(rastreabilidade)
    if saida.exists():
        raise ErroAdjudicacao(f"{saida} já existe: decisões congeladas não são sobrescritas")
    if not planilhas:
        raise ErroAdjudicacao("nenhuma planilha individual")
    casos = ler_json(rastreabilidade)["casos"]
    individuais = [ler_decisoes(p, casos) for p in planilhas]
    nomes = [p["adjudicador"] for p in individuais]
    if len(set(nomes)) != len(nomes):
        raise ErroAdjudicacao(f"adjudicadores repetidos entre as planilhas: {nomes}")
    lido_consenso = ler_decisoes(consenso, casos) if consenso else None
    if lido_consenso and lido_consenso["adjudicador"] in nomes:
        raise ErroAdjudicacao("a planilha de consenso usa o identificador de um adjudicador individual")
    if lido_consenso and len(individuais) < 2:
        raise ErroAdjudicacao("consenso só com pelo menos duas decisões individuais")

    def _registro(p, papel):
        return {"papel": papel, "arquivo": p["arquivo"], "sha256": p["sha256"],
                "adjudicador": p["adjudicador"], "n_casos": p["n_casos"], "avisos": p["avisos"]}
    congeladas = {
        "esquema": ESQUEMA,
        "congelado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rastreabilidade_sha256": sha256_arquivo(rastreabilidade),
        "planilhas": [_registro(p, "individual") for p in individuais] +
                     ([_registro(lido_consenso, "consenso")] if lido_consenso else []),
        "decisoes": {"individuais": {p["adjudicador"]: p["decisoes"] for p in individuais},
                     "consenso": lido_consenso["decisoes"] if lido_consenso else None},
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    gravar_json(saida, congeladas)
    return congeladas


# --------------------------------------------------------------------------- #
# Reconciliação                                                               #
# --------------------------------------------------------------------------- #
def kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Kappa de Cohen entre duas sequências de decisões."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def resumir(casos: Sequence[Mapping[str, Any]], decisoes: Mapping[str, Mapping[str, Any]],
            contagens: Mapping[str, Any], *, exemplos: int = 3) -> dict[str, Any]:
    """Métricas secundárias de um conjunto de decisões (exploratórias)."""
    tp, fp = contagens["tp"], contagens["fp"]
    fps = [c for c in casos if c["grupo"] == FALSO_POSITIVO]
    revisados = [c for c in fps if c["caso"] in decisoes]
    principal = Counter(decisoes[c["caso"]]["principal"] for c in revisados)
    n = len(revisados)
    por_categoria = {k: {"categoria": NOME_PARA_FALSO_POSITIVO[k], "n": principal.get(k, 0),
                         "proporcao": principal.get(k, 0) / n if n else None} for k in DECISOES}
    alteradas = principal.get("1", 0)
    por_rotulo = {}
    for c in revisados:
        por_rotulo.setdefault(c["rotulo_anotado"], Counter())[decisoes[c["caso"]]["principal"]] += 1
    controles = {}
    for grupo in (CONTROLE_VP, CONTROLE_FN):
        do_grupo = [c for c in casos if c["grupo"] == grupo and c["caso"] in decisoes]
        cont = Counter(decisoes[c["caso"]]["principal"] for c in do_grupo)
        conclusivos = sum(cont.get(k, 0) for k in CATEGORIAS)
        controles[grupo] = {"n": len(do_grupo), "por_decisao": {k: cont.get(k, 0) for k in DECISOES},
                            "concordancia_com_a_anotacao": cont.get("1", 0) / conclusivos if conclusivos else None}
    return {
        "fp_total": fp, "fp_revisados": n,
        "por_categoria": por_categoria,
        "conclusivos": sum(principal.get(k, 0) for k in CATEGORIAS),
        "inconclusivos": principal.get("inconclusivo", 0),
        "revisao_necessaria": principal.get("revisao_necessaria", 0),
        "problemas_secundarios": dict(sorted(Counter(s for c in revisados
                                                     for s in decisoes[c["caso"]]["secundarios"]).items())),
        "por_rotulo_anotado": {r: {k: cnt.get(k, 0) for k in DECISOES} for r, cnt in sorted(por_rotulo.items())},
        "precisao_original": tp / (tp + fp) if tp + fp else None,
        "precisao_secundaria_exploratoria": (tp + alteradas) / (tp + fp) if tp + fp else None,
        "criterio_precisao_secundaria": "(VP + FP adjudicados como 1) / (VP + FP); inconclusivos, revisão "
                                        "necessária e categorias 2 a 4 continuam falsos positivos; os VP não "
                                        "são reavaliados (os controles só indicam a concordância com a anotação)",
        "previsoes_alteradas_na_interpretacao_secundaria": alteradas,
        "casos_alterados": sorted(c["caso"] for c in revisados if decisoes[c["caso"]]["principal"] == "1"),
        "controles": controles,
        "exemplos": {k: [{"caso": c["caso"], "trecho_marcado": c["trecho_marcado"],
                          "justificativa": decisoes[c["caso"]]["justificativa"]}
                         for c in revisados if decisoes[c["caso"]]["principal"] == k][:exemplos]
                     for k in DECISOES},
    }


def reconciliar(rastreabilidade: str | Path, congeladas: str | Path, *, exemplos: int = 3) -> dict[str, Any]:
    rastreabilidade, congeladas = Path(rastreabilidade), Path(congeladas)
    rast, cong = ler_json(rastreabilidade), ler_json(congeladas)
    if cong.get("rastreabilidade_sha256") != sha256_arquivo(rastreabilidade):
        raise ErroAdjudicacao("as decisões congeladas foram conferidas contra outra rastreabilidade")
    casos, contagens = rast["casos"], rast["contagens"]
    individuais = cong["decisoes"]["individuais"]
    consenso = cong["decisoes"]["consenso"]
    resultados = {nome: resumir(casos, d, contagens, exemplos=exemplos) for nome, d in individuais.items()}
    if consenso:
        fonte = "consenso"
        resultados["consenso"] = resumir(casos, consenso, contagens, exemplos=exemplos)
    elif len(individuais) == 1:
        fonte = "adjudicador_unico"
    else:
        fonte = "adjudicadores_individuais_sem_consenso"
    concordancia = None
    if len(individuais) >= 2:
        nomes = sorted(individuais)
        concordancia = {}
        for i, a in enumerate(nomes):
            for b in nomes[i + 1:]:
                comuns = [c["caso"] for c in casos if c["caso"] in individuais[a] and c["caso"] in individuais[b]]
                da = [individuais[a][c]["principal"] for c in comuns]
                db = [individuais[b][c]["principal"] for c in comuns]
                fps = {c["caso"] for c in casos if c["grupo"] == FALSO_POSITIVO}
                concordancia[f"{a} x {b}"] = {
                    "n": len(comuns), "concordancia_observada": sum(x == y for x, y in zip(da, db)) / len(comuns)
                    if comuns else None, "kappa_cohen": kappa(da, db),
                    "kappa_cohen_so_fp": kappa([individuais[a][c]["principal"] for c in comuns if c in fps],
                                               [individuais[b][c]["principal"] for c in comuns if c in fps])}
    oficial = rast["metricas_oficiais_negation_of"]
    return {
        "esquema": ESQUEMA,
        "natureza": "análise secundária exploratória: não altera rótulos, partições nem métricas oficiais",
        "origem": rast["origem"],
        "rastreabilidade_sha256": sha256_arquivo(rastreabilidade),
        "congeladas_sha256": sha256_arquivo(congeladas),
        "congelado_em": cong["congelado_em"],
        "fonte_das_decisoes": fonte,
        "aviso": None if fonte == "consenso" else
        "sem decisão consensual: os resultados são de cada adjudicador e não constituem consenso "
        "nem validação clínica definitiva",
        "oficial_negation_of": {k: oficial[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
        "resultados": resultados,
        "concordancia": concordancia,
        "limitacoes": [
            "a proporção por categoria vale só para os falsos positivos desta execução no DEV; "
            "não é extrapolável ao corpus nem a outras execuções sem amostragem apropriada",
            "a precisão secundária não reavalia os verdadeiros positivos nem os pares não preditos",
            "os exemplos contêm texto do SemClinBr: conferir a licença e a anonimização antes de publicar",
        ],
    }


def markdown(resultado: Mapping[str, Any]) -> str:
    """Resumo sem texto clínico (sem os exemplos), para documentação."""
    linhas = ["# Adjudicação dos falsos positivos de `negation_of` — reconciliação", "",
              f"Natureza: {resultado['natureza']}.", "",
              f"Execução: `{resultado['origem']['execucao']}` · partição {resultado['origem']['particao'].upper()} · "
              f"decisões congeladas em {resultado['congelado_em']} · fonte: {resultado['fonte_das_decisoes']}.", ""]
    if resultado["aviso"]:
        linhas += [f"> {resultado['aviso']}.", ""]
    of = resultado["oficial_negation_of"]
    linhas += [f"Oficial (inalterado): VP {of['tp']}, FP {of['fp']}, FN {of['fn']}, precisão {of['precision']:.4f}.", ""]
    for nome, r in resultado["resultados"].items():
        linhas += [f"## {nome}", "", "| decisão | categoria (falsos positivos) | n | proporção |", "| --- | --- | --- | --- |"]
        for k, v in r["por_categoria"].items():
            prop = "—" if v["proporcao"] is None else f"{v['proporcao']:.1%}"
            linhas.append(f"| {k} | {v['categoria']} | {v['n']} | {prop} |")
        sec = r["precisao_secundaria_exploratoria"]
        linhas += ["", f"- FP revisados: {r['fp_revisados']} de {r['fp_total']}; conclusivos {r['conclusivos']}, "
                       f"inconclusivos {r['inconclusivos']}, revisão necessária {r['revisao_necessaria']}.",
                   f"- Precisão original: {r['precisao_original']:.4f}. Precisão secundária (exploratória): "
                   f"{'—' if sec is None else f'{sec:.4f}'} ({r['previsoes_alteradas_na_interpretacao_secundaria']} "
                   "predições reinterpretadas).",
                   f"- Critério: {r['criterio_precisao_secundaria']}."]
        for g, cont in r["controles"].items():
            conc = cont["concordancia_com_a_anotacao"]
            linhas.append(f"- Controles {g}: n = {cont['n']}; concordância com a anotação "
                          f"{'—' if conc is None else f'{conc:.1%}'}.")
        linhas.append("")
    if resultado["concordancia"]:
        linhas += ["## Concordância entre adjudicadores", ""]
        for par, c in resultado["concordancia"].items():
            k = c["kappa_cohen"]
            linhas.append(f"- {par}: n = {c['n']}, concordância {c['concordancia_observada']:.1%}, "
                          f"kappa {'—' if k is None else f'{k:.3f}'}.")
        linhas.append("")
    linhas += ["## Limitações", ""] + [f"- {x}." for x in resultado["limitacoes"]]
    return "\n".join(linhas) + "\n"


# --------------------------------------------------------------------------- #
# Linha de comando                                                            #
# --------------------------------------------------------------------------- #
def _argumentos(argv: Iterable[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m reclin.analises.adjudicacao_negation_of",
                                description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True)
    a = sub.add_parser("preparar", help="gera a planilha cega e a rastreabilidade")
    a.add_argument("--execucao", required=True, type=Path, help="pasta da execução (config.json, predicoes_dev.json ...)")
    a.add_argument("--saida", required=True, type=Path, help="pasta da análise (vazia ou inexistente)")
    a.add_argument("--particoes", type=Path, default=caminhos.PARTICOES)
    a.add_argument("--controles-vp", type=int, default=None,
                   help="verdadeiros positivos sorteados como controles (padrão: o número de falsos positivos)")
    a.add_argument("--semente", type=int, default=SEMENTE)
    a.add_argument("--registro-origem", default=None,
                   help="texto livre registrado na rastreabilidade (de onde vieram os arquivos da execução)")
    c = sub.add_parser("congelar", help="confere e congela as decisões humanas")
    c.add_argument("--rastreabilidade", required=True, type=Path)
    c.add_argument("--planilha", required=True, type=Path, action="append", help="planilha individual (repetível)")
    c.add_argument("--consenso", type=Path, default=None)
    c.add_argument("--saida", required=True, type=Path)
    r = sub.add_parser("reconciliar", help="reconcilia as decisões congeladas com predições e rótulos")
    r.add_argument("--rastreabilidade", required=True, type=Path)
    r.add_argument("--congeladas", required=True, type=Path)
    r.add_argument("--saida", required=True, type=Path)
    r.add_argument("--markdown", type=Path, default=None, help="resumo sem texto clínico")
    r.add_argument("--exemplos", type=int, default=3)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    from reclin.util.log import configurar
    args = _argumentos(argv)
    configurar()
    try:
        if args.comando == "preparar":
            r = preparar(args.execucao, args.saida, pasta_particoes=args.particoes,
                         n_controles_vp=args.controles_vp, semente=args.semente,
                         registro_origem=args.registro_origem)
            rast = r["rastreabilidade"]
            log.info("Casos na planilha cega: %d (%s); documentos: %d", rast["planilha"]["n_casos"],
                     ", ".join(f"{g}={n}" for g, n in rast["contagens"]["casos_por_grupo"].items()),
                     r["verificacao"]["n_documentos"])
            log.info("Verificação da planilha cega: ok")
        elif args.comando == "congelar":
            cong = congelar(args.rastreabilidade, args.planilha, args.saida, consenso=args.consenso)
            log.info("Decisões congeladas: %s", ", ".join(f"{p['adjudicador']} ({p['papel']})" for p in cong["planilhas"]))
        else:
            if args.saida.exists() or (args.markdown and args.markdown.exists()):
                raise ErroAdjudicacao("a saída da reconciliação já existe: nada é sobrescrito")
            resultado = reconciliar(args.rastreabilidade, args.congeladas, exemplos=args.exemplos)
            args.saida.parent.mkdir(parents=True, exist_ok=True)
            gravar_json(args.saida, resultado)
            if args.markdown:
                args.markdown.write_text(markdown(resultado), encoding="utf-8", newline="\n")
            log.info("Reconciliação gravada em %s (fonte: %s)", args.saida, resultado["fonte_das_decisoes"])
    except ErroAdjudicacao as erro:
        log.error("%s", erro)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
