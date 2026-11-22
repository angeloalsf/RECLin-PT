"""Pair-Aware com limiar em P(negation_of): a regra de decisão da configuração final.

A configuração final do RECLin-PT (emenda de 22/11/2026) é a Pair-Aware Linear
treinada sem pesos de classe, com uma regra de decisão diferente do argmax:

* regra: se P(negation_of) >= t, a predição é `negation_of`; senão, o argmax
  entre `associated_with` e `no_relation` (empate → `associated_with`, o
  primeiro na ordem de `LABELS`, a convenção do argmax). É a mesma regra do
  `baseline_calibrado`, aplicada a uma execução da Pair-Aware;
* grade: a de `avaliacao.limiar` (0,05; 0,10; …; 0,95);
* critério: `avaliacao.limiar.PrecisaoComPisoDeRecall` — a maior precisão de
  `negation_of` no DEV entre os limiares com recall de `negation_of` >= 4/5
  (`PISO_RECALL`); desempates: maior F1 de `negation_of`, maior macro-F1, menor
  limiar. O critério prioriza a precisão, que é o objetivo declarado do
  projeto, sem aceitar um ponto que perca mais de um quinto das negações;
* entradas: as probabilidades de 4 casas que a execução de base gravou em
  `predicoes_dev.json` (e, no TEST, em `predicoes_test.json`). Não há treino
  nem GPU: a regra é um pós-processamento das predições da base. O modelo
  entregue é a base (os pesos) mais o `limiar.json` desta execução.

O t escolhido é congelado em `limiar.json`, vinculado ao nome e ao hash do
`config.json` da base, ao SHA-256 do `predicoes_dev.json` dela, à proveniência
dos pesos e à identidade do DEV. A aplicação ao TEST confere esses vínculos,
exige a liberação da configuração final (`execucao.congelamento`) e que o TEST
da base já tenha sido avaliado sob essa liberação, e é feita uma única vez.

Só aceita como base uma pasta de execução `pair_aware` do código atual
(treino concluído); uma execução de outra versão do código precisa antes ser
reempacotada (`treino.reempacotamento`).
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from reclin.avaliacao import limiar as busca_limiar
from reclin.avaliacao import metricas
from reclin.execucao import congelamento as congelamento_final
from reclin.execucao import diretorio, predicoes, trilha, validacao
from reclin.tarefa import ASSOC, NEG, NOREL
from reclin.util import identidade_codigo
from reclin.util.io import ler_json, sha256_arquivo

NOME = "pair_aware_limiar"
ESTRATEGIA_BASE = "pair_aware"
PISO_RECALL = Fraction(4, 5)
REGRA = ("pair_aware_limiar: P(negation_of) >= t -> negation_of; senão argmax entre associated_with e "
         "no_relation (empate -> associated_with); probabilidades de 4 casas do sidecar da base")


class ErroLimiarPairAware(ValueError):
    """O limiar não pode ser escolhido (ou aplicado) com estas entradas."""


def criterio(piso_recall: Fraction = PISO_RECALL) -> busca_limiar.PrecisaoComPisoDeRecall:
    return busca_limiar.PrecisaoComPisoDeRecall(piso_recall)


def prever(probs: Sequence[Sequence[float]], t: float) -> list[int]:
    """A regra de decisão aplicada a cada linha de probabilidades."""
    saida = []
    for p in probs:
        if p[NEG] >= t:
            saida.append(NEG)
        else:
            saida.append(ASSOC if p[ASSOC] >= p[NOREL] else NOREL)
    return saida


def nome_execucao(nome_base: str) -> str:
    """`pair_aware_sem_pesos_biobertpt_seed42` → `pair_aware_limiar_sem_pesos_biobertpt_seed42`."""
    if nome_base.startswith(f"{ESTRATEGIA_BASE}_"):
        return f"{NOME}_" + nome_base[len(ESTRATEGIA_BASE) + 1:]
    return f"{NOME}_{nome_base}"


def _hash(obj: Any) -> str:
    texto = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _conferir_codigo(pasta: Path, operacao: str) -> None:
    try:
        diretorio.conferir_codigo(pasta, operacao)
    except identidade_codigo.ErroCodigo as erro:
        raise ErroLimiarPairAware(str(erro)) from erro


def ler_base(pasta: str | Path) -> dict[str, Any]:
    """A execução de base (Pair-Aware do código atual, treino concluído)."""
    pasta = Path(pasta)
    if not (pasta / diretorio.CONFIG).is_file() or not (pasta / "treino.json").is_file():
        raise ErroLimiarPairAware(f"{pasta} não é uma execução treinada do código novo (sem config.json ou "
                                  "treino.json)")
    registro = diretorio.ler_config(pasta)["config"]
    if registro.get("estrategia") != ESTRATEGIA_BASE:
        raise ErroLimiarPairAware(f"{pasta.name} é da estratégia {registro.get('estrategia')!r}; a base "
                                  f"precisa ser {ESTRATEGIA_BASE!r}")
    treino = ler_json(pasta / "treino.json")
    if not treino.get("concluido"):
        raise ErroLimiarPairAware(f"o treino de {pasta.name} não terminou")
    _conferir_codigo(pasta, "o uso como base da Pair-Aware com limiar")
    dev = pasta / diretorio.arquivo_predicoes("dev")
    if not dev.is_file():
        raise ErroLimiarPairAware(f"{pasta.name} não tem {dev.name}")
    return {"pasta": pasta, "nome": pasta.name, "registro": registro, "treino": treino,
            "sha256_config": _hash(registro), "sha256_predicoes_dev": sha256_arquivo(dev),
            "proveniencia": registro.get("proveniencia_pesos"), "validacao": validacao.da_config(registro),
            "conjunto_dev": diretorio.ler_config(pasta)["conjuntos"].get("dev")}


def _probs(sidecar: Mapping[str, Any], origem: str) -> list[list[float]]:
    probs = sidecar.get("probs")
    if not probs or len(probs) != len(sidecar["y_true"]):
        raise ErroLimiarPairAware(f"{origem} não tem as probabilidades de cada par (probs)")
    return probs


def vinculos(base: Mapping[str, Any]) -> dict[str, Any]:
    return {"base": {"nome": base["nome"], "sha256_config": base["sha256_config"],
                     "sha256_predicoes_dev": base["sha256_predicoes_dev"]},
            "seed": base["registro"]["config"]["seed"], "proveniencia_pesos": base["proveniencia"],
            "dev": base["conjunto_dev"]}


def calibrar_dev(base: str | Path, raiz: str | Path, nome: str, *,
                 piso_recall: Fraction = PISO_RECALL) -> dict[str, Any]:
    """Escolhe e congela t no DEV da execução de base e grava a execução derivada."""
    pasta = Path(raiz) / nome
    if (pasta / diretorio.CONFIG).exists():
        raise ErroLimiarPairAware(f"a execução {pasta} já existe: o t dela já foi escolhido")
    info = ler_base(base)
    if Path(raiz).resolve() != info["pasta"].parent.resolve():
        raise ErroLimiarPairAware("a execução derivada é gravada ao lado da base (mesma raiz de execuções), "
                                  "para que o congelamento encontre os pesos da base")
    sidecar = predicoes.ler(info["pasta"] / diretorio.arquivo_predicoes("dev"))
    probs = _probs(sidecar, f"{info['nome']}/predicoes_dev.json")
    y_true = sidecar["y_true"]
    regra_criterio = criterio(piso_recall)
    try:
        t, tabela = busca_limiar.buscar(y_true, lambda limiar: prever(probs, limiar), criterio=regra_criterio)
    except busca_limiar.ErroLimiar as erro:
        raise ErroLimiarPairAware(str(erro)) from erro
    y_pred = prever(probs, t)
    marca = validacao.herdar({info["nome"]: info["validacao"]})
    registro = {"estrategia": NOME, "regra": REGRA, "criterio": regra_criterio.registro(),
                "base": {"nome": info["nome"], "pasta": str(info["pasta"])},
                "seed": info["registro"]["config"]["seed"], "proveniencia_pesos": info["proveniencia"]}
    if marca:
        registro[validacao.CHAVE] = marca
    diretorio.criar(raiz, nome, estrategia=NOME, config=registro)
    busca_limiar.congelar(pasta / busca_limiar.ARQUIVO, limiar=t, regra=REGRA, vinculos=vinculos(info),
                          tabela=tabela, validacao=marca, criterio=regra_criterio)
    saida = predicoes.montar(model=f"limiar({sidecar['model']})", seed=sidecar["seed"], y_true=y_true,
                             y_pred=y_pred, probs=probs,
                             extra={"estrategia": NOME, "split": "dev", "limiar_t": t, "base": info["nome"]})
    _gravar(pasta, "dev", saida, info["conjunto_dev"])
    m = metricas.avaliar(y_true, y_pred)
    diretorio.gravar_metricas(pasta, "dev", m)
    return {"t": t, "tabela": tabela, "metricas": m, "validacao": marca}


def _gravar(pasta: Path, particao: str, saida: dict[str, Any], conjunto: Mapping[str, Any] | None,
            sobrescrever: bool = False) -> None:
    if conjunto is None:
        raise ErroLimiarPairAware(f"a execução de base não registrou o conjunto de referência do {particao}")
    diretorio.gravar_predicoes(pasta, particao, saida, dict(conjunto), sobrescrever=sobrescrever)


def _criterio_gravado(pasta: Path) -> busca_limiar.PrecisaoComPisoDeRecall:
    gravado = ler_json(pasta / busca_limiar.ARQUIVO) if (pasta / busca_limiar.ARQUIVO).is_file() else {}
    regra_criterio = busca_limiar.criterio_do_registro(gravado.get("configuracao_busca"))
    if regra_criterio is None:
        raise ErroLimiarPairAware(f"{pasta.name}: o limiar.json não registra o critério com piso de recall")
    return regra_criterio


def aplicar_test(pasta: str | Path, *, base: str | Path | None = None, reavaliar: bool = False,
                 congelamento: str | Path | None = None, liberacao: str | None = None) -> dict[str, Any]:
    """Aplica o t congelado às predições do TEST da base, uma vez. Quando a base
    foi avaliada no TEST científico, exige a configuração final congelada e o
    código de liberação (`reclin.execucao.congelamento`), antes de abrir as
    predições do TEST."""
    pasta = Path(pasta)
    if not (pasta / diretorio.CONFIG).is_file():
        raise ErroLimiarPairAware(f"{pasta} não é uma execução da Pair-Aware com limiar")

    validacao.recusar_execucao(pasta, "aplicação do limiar ao TEST")
    registro = diretorio.ler_config(pasta)["config"]
    if registro.get("estrategia") != NOME:
        raise ErroLimiarPairAware(f"{pasta.name} é da estratégia {registro.get('estrategia')!r}, não {NOME!r}")
    _conferir_codigo(pasta, "a aplicação do limiar ao TEST")
    if base is None:
        vizinha = pasta.parent / registro["base"]["nome"]
        base = vizinha if (vizinha / diretorio.CONFIG).is_file() else registro["base"]["pasta"]
    info = ler_base(base)
    validacao.recusar_execucao(info["pasta"], "aplicação do limiar ao TEST")
    try:
        t = busca_limiar.conferir_congelado(pasta / busca_limiar.ARQUIVO, regra=REGRA, vinculos=vinculos(info),
                                            criterio=_criterio_gravado(pasta))
    except busca_limiar.ErroLimiar as erro:
        raise ErroLimiarPairAware(str(erro)) from erro
    origem = info["pasta"] / diretorio.arquivo_predicoes("test")
    if not origem.is_file():
        raise ErroLimiarPairAware(f"o TEST da base {info['nome']} ainda não foi avaliado (sem {origem.name})")
    if (pasta / diretorio.arquivo_predicoes("test")).exists() and not reavaliar:
        raise ErroLimiarPairAware("o TEST desta execução já foi avaliado; uma nova avaliação exige reavaliar "
                                  "e fica registrada na trilha como reavaliação")
    conjunto_test = diretorio.ler_config(info["pasta"])["conjuntos"].get("test") or {}
    registro_final = congelamento_final.exigir_liberacao(
        pasta, None, congelamento=congelamento, liberacao=liberacao, reavaliar=reavaliar,
        sha_test=conjunto_test.get("particao_sha256"))
    sidecar = predicoes.ler(origem)
    probs = _probs(sidecar, f"{info['nome']}/predicoes_test.json")
    y_pred = prever(probs, t)
    saida = predicoes.montar(model=f"limiar({sidecar['model']})", seed=sidecar["seed"],
                             y_true=sidecar["y_true"], y_pred=y_pred, probs=probs,
                             extra={"estrategia": NOME, "split": "test", "limiar_t": t, "base": info["nome"]})
    _gravar(pasta, "test", saida, conjunto_test or None, sobrescrever=reavaliar)
    m = metricas.avaliar(sidecar["y_true"], y_pred)
    diretorio.gravar_metricas(pasta, "test", m)
    congelamento_final.registrar_avaliacao(registro_final, macro_f1=m["macro_f1"],
                                           negation_of_f1=m["f1_per_class"]["negation_of"], n_test=len(y_pred))
    linha = trilha.registrar(pasta / diretorio.TRILHA, execucao=pasta.name, model=saida["model"],
                             seed=saida["seed"], config_sha1=trilha.sha1_config(registro),
                             n_test=len(y_pred), macro_f1=m["macro_f1"],
                             negation_of_f1=m["f1_per_class"]["negation_of"],
                             extra={"estrategia": NOME, "limiar_t": t})
    return {"metricas": m, "trilha": linha, "t": t}
