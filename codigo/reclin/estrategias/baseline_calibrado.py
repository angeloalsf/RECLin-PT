"""Baseline calibrado: o controle de calibração do Two-Stage.

O Two-Stage recebe um ajuste escolhido no DEV (τ); para que a comparação com o
baseline isole o efeito da arquitetura, o baseline recebe uma calibração
equivalente, pré-registrada:

* regra: se P(negation_of) >= t, a predição é `negation_of`; senão, o argmax
  entre `associated_with` e `no_relation` — em empate exato entre os dois,
  `associated_with` (o primeiro na ordem de `LABELS`, a convenção do argmax);
* grade, critério e desempates: os de `avaliacao.limiar` (os mesmos do τ);
* entradas: as probabilidades de 4 casas que o baseline já grava em
  `predicoes_dev.json` (e, no TEST, em `predicoes_test.json`). Não há treino
  nem GPU: o baseline calibrado é um pós-processamento das predições de uma
  execução do baseline do código novo.

O t escolhido é congelado em `limiar.json`, vinculado ao nome e ao hash do
`config.json` da execução de base, ao SHA-256 do `predicoes_dev.json` dela, à
proveniência dos pesos e à identidade do DEV. A aplicação ao TEST confere
esses vínculos, exige que o TEST da base já tenha sido avaliado (a base
grava `predicoes_test.json` uma vez) e é feita uma única vez.

Uma execução de base de validação técnica produz uma calibração também
marcada como validação, que nunca é aplicada ao TEST.

Só aceita como base uma pasta de execução do baseline do código novo
(`config.json` com `estrategia: baseline`, treino concluído); sidecars soltos,
como os do legado, são recusados.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from reclin.avaliacao import limiar as busca_limiar
from reclin.avaliacao import metricas
from reclin.execucao import diretorio, predicoes, trilha, validacao
from reclin.tarefa import ASSOC, NEG, NOREL
from reclin.util.io import ler_json, sha256_arquivo

NOME = "baseline_calibrado"
ESTRATEGIA_BASE = "baseline"
REGRA = ("baseline_calibrado: P(negation_of) >= t -> negation_of; senão argmax entre associated_with e "
         "no_relation (empate -> associated_with); probabilidades de 4 casas do sidecar da base")


class ErroCalibracao(ValueError):
    """A calibração não pode ser feita (ou aplicada) com estas entradas."""


def prever(probs: Sequence[Sequence[float]], t: float) -> list[int]:
    """A regra de calibração aplicada a cada linha de probabilidades."""
    saida = []
    for p in probs:
        if p[NEG] >= t:
            saida.append(NEG)
        else:
            saida.append(ASSOC if p[ASSOC] >= p[NOREL] else NOREL)
    return saida


def nome_execucao(nome_base: str) -> str:
    """`baseline_biobertpt_seed42` → `baseline_calibrado_biobertpt_seed42`."""
    if nome_base.startswith(f"{ESTRATEGIA_BASE}_"):
        return f"{NOME}_" + nome_base[len(ESTRATEGIA_BASE) + 1:]
    return f"{NOME}_{nome_base}"


def _hash(obj: Any) -> str:
    texto = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def ler_base(pasta: str | Path) -> dict[str, Any]:
    """A execução de base (baseline do código novo, treino concluído)."""
    pasta = Path(pasta)
    if not (pasta / diretorio.CONFIG).is_file() or not (pasta / "treino.json").is_file():
        raise ErroCalibracao(f"{pasta} não é uma execução do baseline do código novo (sem config.json ou "
                             "treino.json); sidecars soltos, como os do legado, não são aceitos")
    registro = diretorio.ler_config(pasta)["config"]
    if registro.get("estrategia") != ESTRATEGIA_BASE:
        raise ErroCalibracao(f"{pasta.name} é da estratégia {registro.get('estrategia')!r}; a base precisa "
                             f"ser {ESTRATEGIA_BASE!r}")
    treino = ler_json(pasta / "treino.json")
    if not treino.get("concluido"):
        raise ErroCalibracao(f"o treino de {pasta.name} não terminou")
    dev = pasta / diretorio.arquivo_predicoes("dev")
    if not dev.is_file():
        raise ErroCalibracao(f"{pasta.name} não tem {dev.name}")
    return {"pasta": pasta, "nome": pasta.name, "registro": registro, "treino": treino,
            "sha256_config": _hash(registro), "sha256_predicoes_dev": sha256_arquivo(dev),
            "proveniencia": registro.get("proveniencia_pesos"), "validacao": validacao.da_config(registro),
            "conjunto_dev": diretorio.ler_config(pasta)["conjuntos"].get("dev")}


def _probs(sidecar: Mapping[str, Any], origem: str) -> list[list[float]]:
    probs = sidecar.get("probs")
    if not probs or len(probs) != len(sidecar["y_true"]):
        raise ErroCalibracao(f"{origem} não tem as probabilidades de cada par (probs)")
    return probs


def vinculos(base: Mapping[str, Any]) -> dict[str, Any]:
    return {"base": {"nome": base["nome"], "sha256_config": base["sha256_config"],
                     "sha256_predicoes_dev": base["sha256_predicoes_dev"]},
            "seed": base["registro"]["config"]["seed"], "proveniencia_pesos": base["proveniencia"],
            "dev": base["conjunto_dev"]}


def calibrar_dev(base: str | Path, raiz: str | Path, nome: str) -> dict[str, Any]:
    """Escolhe e congela t no DEV da execução de base e grava a execução."""
    pasta = Path(raiz) / nome
    if (pasta / diretorio.CONFIG).exists():
        raise ErroCalibracao(f"a execução {pasta} já existe: o t dela já foi escolhido")
    info = ler_base(base)
    sidecar = predicoes.ler(info["pasta"] / diretorio.arquivo_predicoes("dev"))
    probs = _probs(sidecar, f"{info['nome']}/predicoes_dev.json")
    y_true = sidecar["y_true"]
    t, tabela = busca_limiar.buscar(y_true, lambda limiar: prever(probs, limiar))
    y_pred = prever(probs, t)
    marca = validacao.herdar({info["nome"]: info["validacao"]})
    registro = {"estrategia": NOME, "regra": REGRA, "base": {"nome": info["nome"], "pasta": str(info["pasta"])},
                "seed": info["registro"]["config"]["seed"], "proveniencia_pesos": info["proveniencia"]}
    if marca:
        registro[validacao.CHAVE] = marca
    diretorio.criar(raiz, nome, estrategia=NOME, config=registro)
    busca_limiar.congelar(pasta / busca_limiar.ARQUIVO, limiar=t, regra=REGRA, vinculos=vinculos(info),
                          tabela=tabela, validacao=marca)
    saida = predicoes.montar(model=f"calibrado({sidecar['model']})", seed=sidecar["seed"], y_true=y_true,
                             y_pred=y_pred, probs=probs,
                             extra={"estrategia": NOME, "split": "dev", "limiar_t": t, "base": info["nome"]})
    _gravar(pasta, "dev", saida, info["conjunto_dev"])
    m = metricas.avaliar(y_true, y_pred)
    diretorio.gravar_metricas(pasta, "dev", m)
    return {"t": t, "tabela": tabela, "metricas": m, "validacao": marca}


def _gravar(pasta: Path, particao: str, saida: dict[str, Any], conjunto: Mapping[str, Any] | None,
            sobrescrever: bool = False) -> None:
    if conjunto is None:
        raise ErroCalibracao(f"a execução de base não registrou o conjunto de referência do {particao}")
    diretorio.gravar_predicoes(pasta, particao, saida, dict(conjunto), sobrescrever=sobrescrever)


def aplicar_test(pasta: str | Path, *, base: str | Path | None = None, reavaliar: bool = False) -> dict[str, Any]:
    """Aplica o t congelado às predições do TEST da base, uma vez."""
    pasta = Path(pasta)
    if not (pasta / diretorio.CONFIG).is_file():
        raise ErroCalibracao(f"{pasta} não é uma execução do baseline calibrado")
    validacao.recusar_execucao(pasta, "aplicação do limiar ao TEST")
    registro = diretorio.ler_config(pasta)["config"]
    if registro.get("estrategia") != NOME:
        raise ErroCalibracao(f"{pasta.name} é da estratégia {registro.get('estrategia')!r}, não {NOME!r}")
    info = ler_base(base or registro["base"]["pasta"])
    validacao.recusar_execucao(info["pasta"], "aplicação do limiar ao TEST")
    try:
        t = busca_limiar.conferir_congelado(pasta / busca_limiar.ARQUIVO, regra=REGRA, vinculos=vinculos(info))
    except busca_limiar.ErroLimiar as erro:
        raise ErroCalibracao(str(erro)) from erro
    origem = info["pasta"] / diretorio.arquivo_predicoes("test")
    if not origem.is_file():
        raise ErroCalibracao(f"o TEST da base {info['nome']} ainda não foi avaliado (sem {origem.name})")
    if (pasta / diretorio.arquivo_predicoes("test")).exists() and not reavaliar:
        raise ErroCalibracao("o TEST desta execução já foi avaliado; uma nova avaliação exige reavaliar "
                             "e fica registrada na trilha como reavaliação")
    sidecar = predicoes.ler(origem)
    probs = _probs(sidecar, f"{info['nome']}/predicoes_test.json")
    y_pred = prever(probs, t)
    saida = predicoes.montar(model=f"calibrado({sidecar['model']})", seed=sidecar["seed"],
                             y_true=sidecar["y_true"], y_pred=y_pred, probs=probs,
                             extra={"estrategia": NOME, "split": "test", "limiar_t": t, "base": info["nome"]})
    conjunto = diretorio.ler_config(info["pasta"])["conjuntos"].get("test")
    _gravar(pasta, "test", saida, conjunto, sobrescrever=reavaliar)
    m = metricas.avaliar(sidecar["y_true"], y_pred)
    diretorio.gravar_metricas(pasta, "test", m)
    linha = trilha.registrar(pasta / diretorio.TRILHA, execucao=pasta.name, model=saida["model"],
                             seed=saida["seed"], config_sha1=trilha.sha1_config(registro),
                             n_test=len(y_pred), macro_f1=m["macro_f1"],
                             negation_of_f1=m["f1_per_class"]["negation_of"],
                             extra={"estrategia": NOME, "limiar_t": t})
    return {"metricas": m, "trilha": linha, "t": t}
