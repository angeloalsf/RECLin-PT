"""Busca de limiar no DEV, pré-registrada, e o congelamento do limiar escolhido.

Um só procedimento para os dois limiares escolhidos no DEV na fase atual — o τ
do Two-Stage (roteamento ao estágio 2) e o t do baseline calibrado — para que
os dois sejam idênticos por construção:

* grade fixa: 0,05; 0,10; …; 0,95 (19 valores);
* critério primário: o F1 de `negation_of` da saída final (3 rótulos) no DEV,
  calculado pelas contagens: 2·TP / (2·TP + FP + FN);
* desempates, nesta ordem: maior macro-F1 (média dos F1 por classe, cada um
  pelas contagens); menor |limiar − 0,5|; maior limiar;
* comparação exata, com frações das contagens (`fractions.Fraction`), sem
  tolerância de ponto flutuante.

Quem chama fornece a regra (`prever(limiar) -> y_pred`) e uma descrição dela;
este módulo não conhece estratégias.

O limiar escolhido é congelado num `limiar.json` junto com os vínculos que o
tornam válido (hashes dos checkpoints ou das predições de origem, identidade
do DEV, seed, proveniência), com o hash da configuração da busca e com um
hash de todo o registro (`hash_registro`: limiar, configuração, vínculos e
tabela). `conferir_congelado` recalcula os vínculos antes da aplicação ao
TEST, confere o hash do registro e refaz a escolha a partir das contagens da
tabela; qualquer divergência é recusada. Não existe forma de informar um
limiar à mão.
"""
from __future__ import annotations

import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from reclin.avaliacao import metricas
from reclin.tarefa import LABELS, NEG
from reclin.util.io import gravar_json, ler_json

ARQUIVO = "limiar.json"
VERSAO = 1
GRADE: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
CRITERIO = "maior F1 de negation_of (contagens) da saída final no DEV"
DESEMPATES = ("maior macro-F1 (contagens)", "menor |limiar - 0,5|", "maior limiar")


class ErroLimiar(ValueError):
    """O limiar congelado não vale para as entradas atuais."""


def _hash(obj: Any) -> str:
    texto = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def configuracao_busca(regra: str, grade: Sequence[float] = GRADE) -> dict[str, Any]:
    """A configuração da busca (o que o hash gravado no `limiar.json` cobre)."""
    return {"versao": VERSAO, "regra": regra, "grade": [float(g) for g in grade],
            "criterio": CRITERIO, "desempates": list(DESEMPATES),
            "comparacao": "exata, por frações das contagens"}


def hash_configuracao(regra: str, grade: Sequence[float] = GRADE) -> str:
    return _hash(configuracao_busca(regra, grade))


def _f1(tp: int, fp: int, fn: int) -> Fraction:
    denominador = 2 * tp + fp + fn
    return Fraction(2 * tp, denominador) if denominador else Fraction(0)


def pontuar(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Contagens por classe, F1 de `negation_of` e macro-F1 (exatos e em float)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true tem {len(y_true)} elementos e y_pred {len(y_pred)}")
    matriz = metricas.matriz_confusao(y_true, y_pred)
    contagens = {}
    f1s = []
    for i, rotulo in enumerate(LABELS):
        tp = matriz[i][i]
        fp = sum(matriz[j][i] for j in range(len(LABELS))) - tp
        fn = sum(matriz[i]) - tp
        contagens[rotulo] = {"tp": tp, "fp": fp, "fn": fn}
        f1s.append(_f1(tp, fp, fn))
    f1_neg = f1s[NEG]
    macro = sum(f1s, Fraction(0)) / len(f1s)
    return {"contagens": contagens, "f1_negation_of": float(f1_neg), "macro_f1": float(macro),
            "_f1_exato": f1_neg, "_macro_exato": macro}


def _chave(linha: Mapping[str, Any]) -> tuple:
    t = Fraction(str(linha["limiar"]))
    return (linha["_f1_exato"], linha["_macro_exato"], -abs(t - Fraction(1, 2)), t)


def escolher(tabela: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """A linha vencedora pelo critério e pelos desempates (ver o módulo)."""
    if not tabela:
        raise ValueError("tabela vazia")
    return max(tabela, key=_chave)


def escolhido_da_tabela(tabela: Sequence[Mapping[str, Any]]) -> float:
    """Refaz a escolha a partir das contagens gravadas numa tabela."""
    linhas = []
    for linha in tabela:
        f1s = [_f1(**{k: linha["contagens"][r][k] for k in ("tp", "fp", "fn")}) for r in LABELS]
        linhas.append({"limiar": linha["limiar"], "_f1_exato": f1s[NEG],
                       "_macro_exato": sum(f1s, Fraction(0)) / len(f1s)})
    return float(escolher(linhas)["limiar"])


def buscar(y_true: Sequence[int], prever: Callable[[float], Sequence[int]],
           grade: Sequence[float] = GRADE, extra: Callable[[float, Sequence[int]], dict] | None = None
           ) -> tuple[float, list[dict[str, Any]]]:
    """Avalia `prever(limiar)` em cada ponto da grade; devolve o limiar escolhido
    e a tabela (sem os campos exatos, pronta para gravar). `extra(limiar,
    y_pred)` acrescenta campos informativos a cada linha (pares roteados, por
    exemplo)."""
    linhas = []
    for limiar in grade:
        y_pred = list(prever(limiar))
        linha = {"limiar": float(limiar), **pontuar(y_true, y_pred)}
        if extra is not None:
            linha.update(extra(limiar, y_pred))
        linhas.append(linha)
    vencedora = escolher(linhas)
    tabela = [{k: v for k, v in linha.items() if not k.startswith("_")} for linha in linhas]
    return vencedora["limiar"], tabela


def congelar(caminho: str | Path, *, limiar: float, regra: str, vinculos: Mapping[str, Any],
             tabela: Sequence[Mapping[str, Any]], grade: Sequence[float] = GRADE,
             validacao: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Grava o `limiar.json`. Falha se já existir: um limiar congelado não é
    regravado."""
    caminho = Path(caminho)
    if caminho.exists():
        raise ErroLimiar(f"{caminho} já existe: o limiar já foi congelado")
    if float(limiar) not in [float(g) for g in grade]:
        raise ErroLimiar(f"limiar {limiar} fora da grade")
    registro = {
        "limiar": float(limiar),
        "configuracao_busca": configuracao_busca(regra, grade),
        "hash_configuracao_busca": hash_configuracao(regra, grade),
        "vinculos": dict(vinculos),
        "tabela": [dict(t) for t in tabela],
    }
    if escolhido_da_tabela(registro["tabela"]) != float(limiar):
        raise ErroLimiar(f"o limiar {limiar} não é o que a regra escolhe na tabela")
    registro["hash_registro"] = _hash(registro)
    registro["congelado_em"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if validacao:
        registro["validacao"] = dict(validacao)
    gravar_json(caminho, registro)
    return registro


def conferir_congelado(caminho: str | Path, *, regra: str, vinculos: Mapping[str, Any],
                       grade: Sequence[float] = GRADE) -> float:
    """O limiar congelado, se os vínculos atuais e a configuração da busca
    forem exatamente os gravados; caso contrário, `ErroLimiar` com as
    divergências."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroLimiar(f"não há limiar congelado em {caminho}: escolha-o no DEV antes do TEST")
    gravado = ler_json(caminho)
    problemas = []
    if gravado.get("hash_configuracao_busca") != hash_configuracao(regra, grade):
        problemas.append("a configuração da busca (grade, regra, critério ou desempates) mudou")
    antigos = gravado.get("vinculos", {})
    for chave in sorted(set(antigos) | set(vinculos)):
        if antigos.get(chave) != vinculos.get(chave):
            problemas.append(f"vínculo {chave!r}: congelado={antigos.get(chave)!r}, atual={vinculos.get(chave)!r}")
    nucleo = {k: gravado.get(k) for k in ("limiar", "configuracao_busca", "hash_configuracao_busca",
                                          "vinculos", "tabela")}
    if gravado.get("hash_registro") != _hash(nucleo):
        problemas.append("o limiar.json foi editado (hash_registro não confere)")
    elif escolhido_da_tabela(gravado["tabela"]) != float(gravado["limiar"]):
        problemas.append("o limiar gravado não é o que a regra escolhe na tabela gravada")
    if problemas:
        raise ErroLimiar("o limiar congelado não vale para estas entradas: " + "; ".join(problemas))
    return float(gravado["limiar"])
