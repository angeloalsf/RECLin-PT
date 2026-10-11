"""Inferência conjunta do Two-Stage: escolha de τ no DEV e avaliação do TEST.

Fluxo real (o mesmo no DEV e no TEST):

1. o estágio 1 calcula P(relacao) para todos os pares candidatos; o valor é
   arredondado a 4 casas, a convenção dos sidecars (`predicoes.arredondar_probs`);
2. os pares com P(relacao) >= τ seguem para o estágio 2, que roda SÓ sobre
   eles e decide o tipo pelo argmax (`negation_of` ou `associated_with`);
3. os demais pares ficam `no_relation`.

A saída tem os 3 rótulos da tarefa sobre o conjunto completo, no formato
padrão de sidecar, e é comparável com o baseline por todas as métricas.

`compor_dev` escolhe τ na grade pré-registrada (`avaliacao.limiar`): para cada
τ, o pipeline completo roda pelo fluxo real (o estágio 2 só sobre os pares
roteados por aquele τ) e é pontuado pelo F1 de `negation_of` da saída final no
DEV, com os desempates do protocolo. O τ escolhido é congelado em
`limiar.json`, vinculado aos hashes dos dois `melhor_modelo/`, às
configurações gravadas, à seed, à proveniência dos pesos e à identidade do
DEV. O DEV final é recalculado pelo mesmo fluxo e precisa coincidir com a
linha do τ escolhido.

`avaliar_test` recalcula todos os vínculos, recusa qualquer divergência e
aplica o τ congelado ao TEST, uma vez (uma segunda avaliação exige
`reavaliar` e fica na trilha). Não existe opção para informar τ.

Entradas só do código novo: cada estágio precisa ser uma pasta de execução
concluída (`config.json` com a estratégia do estágio, `treino.json`,
`melhor_modelo/`). Sidecars soltos, como os do legado, são recusados.

O diretório da composição:

    <raiz>/<nome>/
        config.json            estágios (nome, pasta, hashes), seed, proveniência, regra
        limiar.json            busca completa e τ congelado, com os vínculos
        predicoes_dev.json     DEV pelo fluxo real com τ (3 rótulos)
        metricas.json          DEV (e, depois de avaliar_test, TEST) + diagnósticos por estágio
        predicoes_test.json    só depois de avaliar_test
        avaliacoes_test.jsonl  trilha
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from reclin import entrada, modelos, particoes, pretreino, tarefa
from reclin.avaliacao import limiar as busca_limiar
from reclin.avaliacao import metricas
from reclin.estrategias.two_stage import (CASAS_ROTEAMENTO, CONFIGS, ESTRATEGIA_ESTAGIO, NEG_E2, NOME,
                                         RELACAO)
from reclin.execucao import diretorio, predicoes, trilha, validacao
from reclin.tarefa import ASSOC, NEG, NOREL
from reclin.treino import checkpoint, laco, reprodutibilidade
from reclin.treino.classificador import ErroExecucao, particoes_congeladas
from reclin.util import caminhos
from reclin.util.io import ler_json

log = logging.getLogger(__name__)

REGRA = ("two_stage: P(relacao) do estágio 1, arredondada a 4 casas, >= τ -> estágio 2 (argmax entre "
         "negation_of e associated_with); senão no_relation")
CAMPOS_BASE = ("encoder", "seed", "max_gap", "ctx_chars", "max_length", "batch_size", "lr", "weight_decay",
               "warmup_ratio", "max_grad_norm", "class_weight")


def _hash(obj: Any) -> str:
    texto = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def ler_estagio(pasta: str | Path, estagio: int) -> dict[str, Any]:
    """Uma execução de estágio concluída, com o hash do seu `melhor_modelo/`."""
    pasta = Path(pasta)
    if not (pasta / diretorio.CONFIG).is_file() or not (pasta / "treino.json").is_file():
        raise ErroExecucao(f"{pasta} não é uma execução de estágio do código novo (sem config.json ou "
                           "treino.json): o Two-Stage só aceita execuções produzidas por scripts/two_stage.py")
    registro = diretorio.ler_config(pasta)["config"]
    if registro.get("estrategia") != ESTRATEGIA_ESTAGIO[estagio]:
        raise ErroExecucao(f"{pasta.name} é da estratégia {registro.get('estrategia')!r}; "
                           f"esperado {ESTRATEGIA_ESTAGIO[estagio]!r}")
    treino = ler_json(pasta / "treino.json")
    if not treino.get("concluido"):
        raise ErroExecucao(f"o treino de {pasta.name} não terminou")
    melhor = checkpoint.pasta_checkpoints(pasta) / checkpoint.MELHOR
    if not checkpoint.tem_melhor(pasta):
        raise ErroExecucao(f"{pasta.name} não tem {checkpoint.MELHOR}/")
    hash_melhor = pretreino.hash_pasta(melhor)
    if hash_melhor != treino.get("final", {}).get("sha256_melhor_modelo"):
        raise ErroExecucao(f"o {checkpoint.MELHOR}/ de {pasta.name} mudou depois do treino")
    return {"pasta": pasta, "nome": pasta.name, "registro": registro, "treino": treino,
            "sha256_melhor_modelo": hash_melhor, "sha256_config": _hash(registro),
            "seed": registro["config"]["seed"], "proveniencia": registro.get("proveniencia_pesos"),
            "validacao": validacao.da_config(registro)}


def conferir_par(e1: Mapping[str, Any], e2: Mapping[str, Any]) -> dict[str, Any] | None:
    """Os dois estágios combinam: mesma seed e configuração base, mesmas
    partições, mesma proveniência, mesma natureza (validação ou não). Devolve a
    marca de validação herdada (ou None)."""
    c1, c2 = e1["registro"]["config"], e2["registro"]["config"]
    for estagio, c in ((1, c1), (2, c2)):
        if c.get("tipo") != CONFIGS[estagio].__name__:
            raise ErroExecucao(f"configuração do estágio {estagio} é {c.get('tipo')!r}")
    divergentes = [k for k in CAMPOS_BASE if c1.get(k) != c2.get(k)]
    if divergentes:
        raise ErroExecucao(f"os estágios divergem na configuração base: {divergentes} "
                           f"(semente do estágio 1: {c1.get('seed')}, do estágio 2: {c2.get('seed')})")
    if e1["registro"]["particoes"] != e2["registro"]["particoes"]:
        raise ErroExecucao("os estágios foram treinados com partições diferentes")
    try:
        pretreino.mesma_proveniencia({e1["nome"]: e1["proveniencia"], e2["nome"]: e2["proveniencia"]})
        return validacao.herdar({e1["nome"]: e1["validacao"], e2["nome"]: e2["validacao"]})
    except (pretreino.ErroProveniencia, validacao.ExecucaoDeValidacao) as erro:
        raise ErroExecucao(str(erro)) from erro


def _carregar(info: Mapping[str, Any], disp: Any) -> tuple[Any, Any]:
    tokenizer, rede = modelos.recarregar(checkpoint.pasta_checkpoints(info["pasta"]) / checkpoint.MELHOR)
    rede.to(disp)
    return tokenizer, rede


def _prever(tokenizer: Any, rede: Any, exemplos: entrada.Exemplos, config: Mapping[str, Any], disp: Any):
    loader = entrada.criar_loader(tokenizer, exemplos, max_length=config["max_length"],
                                  batch_size=config["batch_size"], embaralhar=False, seed=config["seed"])
    return laco.prever(rede, loader, disp, com_probs=True)


class Pipeline:
    """Os dois estágios carregados, prontos para o fluxo real numa partição."""

    def __init__(self, e1: Mapping[str, Any], e2: Mapping[str, Any], disp: Any):
        self.config = e1["registro"]["config"]
        self.disp = disp
        self.tok1, self.rede1 = _carregar(e1, disp)
        self.tok2, self.rede2 = _carregar(e2, disp)

    def probabilidades_estagio1(self, exemplos: entrada.Exemplos) -> list[list[float]]:
        """P(relacao), P(no_relation) de cada par, arredondadas a 4 casas."""
        _, probs = _prever(self.tok1, self.rede1, exemplos, self.config, self.disp)
        return predicoes.arredondar_probs(probs, CASAS_ROTEAMENTO)

    def fluxo(self, exemplos: entrada.Exemplos, probs1: list[list[float]], tau: float
              ) -> tuple[list[int], list[int], list[int]]:
        """(y_pred de 3 rótulos, índices roteados ao estágio 2, predições do estágio 2)."""
        roteados = [i for i, p in enumerate(probs1) if p[RELACAO] >= tau]
        y_pred = [NOREL] * len(exemplos)
        preds2: list[int] = []
        if roteados:
            preds2, _ = _prever(self.tok2, self.rede2, exemplos.subconjunto(roteados), self.config, self.disp)
            for i, p in zip(roteados, preds2):
                y_pred[i] = NEG if p == NEG_E2 else ASSOC
        return y_pred, roteados, list(preds2)


def _particao(pasta_particoes: Path, nome: str, config: Mapping[str, Any]):
    divergencias = particoes.conferir_particoes(pasta_particoes, (nome,))
    if divergencias:
        raise ErroExecucao("partições não conferem com o MANIFEST: " + "; ".join(divergencias))
    sha = particoes.ler_manifesto(pasta_particoes)["splits"][nome]["sha256"]
    docs = particoes.ler_particao(pasta_particoes, nome)
    exemplos = entrada.exemplos(docs, max_gap=config["max_gap"], ctx_chars=config["ctx_chars"])
    conjunto = tarefa.conjunto_referencia(nome, docs, particao_sha256=sha, max_gap=config["max_gap"],
                                          conferir_tamanho=particoes_congeladas(pasta_particoes))
    return exemplos, conjunto


def _vinculos(e1: Mapping[str, Any], e2: Mapping[str, Any], conjunto_dev: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "estagio1": {"nome": e1["nome"], "sha256_melhor_modelo": e1["sha256_melhor_modelo"],
                     "sha256_config": e1["sha256_config"], "melhor_epoca": e1["treino"].get("melhor_epoca")},
        "estagio2": {"nome": e2["nome"], "sha256_melhor_modelo": e2["sha256_melhor_modelo"],
                     "sha256_config": e2["sha256_config"], "melhor_epoca": e2["treino"].get("melhor_epoca")},
        "seed": e1["seed"], "proveniencia_pesos": e1["proveniencia"], "dev": dict(conjunto_dev),
    }


def _diagnostico(y_true: list[int], roteados: list[int], preds2: list[int]) -> dict[str, Any]:
    """Diagnóstico por estágio no fluxo real (rotulado; não é resultado da arquitetura)."""
    negacoes = [i for i, y in enumerate(y_true) if y == NEG]
    com_relacao = [i for i, y in enumerate(y_true) if y != NOREL]
    roteado = set(roteados)
    pares = [(y_true[i], p) for i, p in zip(roteados, preds2) if y_true[i] != NOREL]
    acertos_tipo = sum(1 for y, p in pares if (y == NEG) == (p == NEG_E2))
    return {"rotulo": "diagnóstico por estágio no fluxo real (não é resultado da arquitetura)",
            "n_roteados_ao_estagio2": len(roteados),
            "recall_estagio1_nos_pares_negation_of": (sum(1 for i in negacoes if i in roteado) / len(negacoes)
                                                      if negacoes else 0.0),
            "recall_estagio1_nos_pares_com_relacao": (sum(1 for i in com_relacao if i in roteado)
                                                      / len(com_relacao) if com_relacao else 0.0),
            "roteados_sem_relacao_anotada": sum(1 for i in roteados if y_true[i] == NOREL),
            "acuracia_do_tipo_nos_roteados_com_relacao": (acertos_tipo / len(pares)) if pares else 0.0}


def compor_dev(estagio1: str | Path, estagio2: str | Path, raiz: str | Path, nome: str, *,
               pasta_particoes: str | Path = caminhos.PARTICOES, dispositivo: Any = "cpu") -> dict[str, Any]:
    """Escolhe e congela τ no DEV e grava a execução da composição."""
    pasta = Path(raiz) / nome
    if (pasta / diretorio.CONFIG).exists():
        raise ErroExecucao(f"a execução {pasta} já existe: o τ dela já foi escolhido")
    e1, e2 = ler_estagio(estagio1, 1), ler_estagio(estagio2, 2)
    marca = conferir_par(e1, e2)
    config = e1["registro"]["config"]
    pasta_particoes = Path(pasta_particoes)
    if e1["registro"]["particoes"]["dev"] != particoes.ler_manifesto(pasta_particoes)["splits"]["dev"]["sha256"]:
        raise ErroExecucao("o DEV destas partições não é o DEV com que os estágios foram treinados")
    reprodutibilidade.fixar_sementes(config["seed"])
    dev, conjunto_dev = _particao(pasta_particoes, "dev", config)
    pipeline = Pipeline(e1, e2, dispositivo)
    probs1 = pipeline.probabilidades_estagio1(dev)

    def prever(tau: float) -> list[int]:
        return pipeline.fluxo(dev, probs1, tau)[0]

    def extra(tau: float, y_pred) -> dict[str, Any]:
        roteados = sum(1 for p in probs1 if p[RELACAO] >= tau)
        return {"n_roteados": roteados}

    tau, tabela = busca_limiar.buscar(dev.rotulos, prever, extra=extra)
    y_pred, roteados, preds2 = pipeline.fluxo(dev, probs1, tau)
    linha = next(t for t in tabela if t["limiar"] == tau)
    if busca_limiar.pontuar(dev.rotulos, y_pred)["contagens"] != linha["contagens"]:
        raise ErroExecucao("o DEV recalculado pelo fluxo real com o τ escolhido difere da linha da busca")
    log.info("τ escolhido no DEV: %.2f (F1 negation_of=%.4f, macro-F1=%.4f, %d pares roteados)",
             tau, linha["f1_negation_of"], linha["macro_f1"], len(roteados))

    registro = {"estrategia": NOME, "regra": REGRA,
                "estagios": {"1": {"nome": e1["nome"], "pasta": str(e1["pasta"])},
                             "2": {"nome": e2["nome"], "pasta": str(e2["pasta"])}},
                "seed": config["seed"], "config_base": {k: config[k] for k in CAMPOS_BASE},
                "proveniencia_pesos": e1["proveniencia"], "particoes": {"dev": conjunto_dev["particao_sha256"]}}
    if marca:
        registro[validacao.CHAVE] = marca
    diretorio.criar(raiz, nome, estrategia=NOME, config=registro)
    busca_limiar.congelar(pasta / busca_limiar.ARQUIVO, limiar=tau, regra=REGRA,
                          vinculos=_vinculos(e1, e2, conjunto_dev), tabela=tabela, validacao=marca)
    sidecar = predicoes.montar(
        model=f"two_stage({e1['nome']} + {e2['nome']})", seed=config["seed"], y_true=dev.rotulos, y_pred=y_pred,
        extra={"estrategia": NOME, "split": "dev", "limiar_tau": tau,
               "estagios": [e1["nome"], e2["nome"]], "indices_estagio2": roteados, "preds_estagio2": preds2,
               "probs_estagio1": probs1})
    diretorio.gravar_predicoes(pasta, "dev", sidecar, conjunto_dev)
    m = metricas.avaliar(dev.rotulos, y_pred)
    diretorio.gravar_metricas(pasta, "dev", m)
    diretorio.gravar_metricas(pasta, "dev_diagnostico_estagios", _diagnostico(dev.rotulos, roteados, preds2))
    return {"tau": tau, "tabela": tabela, "metricas": m, "validacao": marca}


def avaliar_test(pasta: str | Path, *, estagio1: str | Path | None = None, estagio2: str | Path | None = None,
                 pasta_particoes: str | Path = caminhos.PARTICOES, reavaliar: bool = False,
                 dispositivo: Any = "cpu") -> dict[str, Any]:
    """Aplica o τ congelado ao TEST, uma vez, depois de conferir os vínculos."""
    pasta, pasta_particoes = Path(pasta), Path(pasta_particoes)
    if not (pasta / diretorio.CONFIG).is_file():
        raise ErroExecucao(f"{pasta} não é uma execução do Two-Stage (sem config.json)")
    try:
        validacao.recusar_execucao(pasta, "avaliação do TEST")
    except validacao.ExecucaoDeValidacao as erro:
        raise ErroExecucao(str(erro)) from erro
    registro = diretorio.ler_config(pasta)["config"]
    if registro.get("estrategia") != NOME:
        raise ErroExecucao(f"{pasta.name} é da estratégia {registro.get('estrategia')!r}, não {NOME!r}")
    e1 = ler_estagio(estagio1 or registro["estagios"]["1"]["pasta"], 1)
    e2 = ler_estagio(estagio2 or registro["estagios"]["2"]["pasta"], 2)
    if conferir_par(e1, e2):
        raise ErroExecucao("os estágios são execuções de validação técnica: o TEST não é avaliado")
    config = e1["registro"]["config"]
    _, conjunto_dev = _particao(pasta_particoes, "dev", config)
    try:
        tau = busca_limiar.conferir_congelado(pasta / busca_limiar.ARQUIVO, regra=REGRA,
                                              vinculos=_vinculos(e1, e2, conjunto_dev))
    except busca_limiar.ErroLimiar as erro:
        raise ErroExecucao(str(erro)) from erro
    if (pasta / diretorio.arquivo_predicoes("test")).exists() and not reavaliar:
        raise ErroExecucao("o TEST desta execução já foi avaliado; uma nova avaliação exige reavaliar "
                           "e fica registrada na trilha como reavaliação")
    reprodutibilidade.fixar_sementes(config["seed"])
    test, conjunto_test = _particao(pasta_particoes, "test", config)
    pipeline = Pipeline(e1, e2, dispositivo)
    probs1 = pipeline.probabilidades_estagio1(test)
    y_pred, roteados, preds2 = pipeline.fluxo(test, probs1, tau)
    sidecar = predicoes.montar(
        model=f"two_stage({e1['nome']} + {e2['nome']})", seed=config["seed"], y_true=test.rotulos, y_pred=y_pred,
        extra={"estrategia": NOME, "split": "test", "limiar_tau": tau,
               "estagios": [e1["nome"], e2["nome"]], "indices_estagio2": roteados, "preds_estagio2": preds2,
               "probs_estagio1": probs1})
    diretorio.gravar_predicoes(pasta, "test", sidecar, conjunto_test, sobrescrever=reavaliar)
    m = metricas.avaliar(test.rotulos, y_pred)
    diretorio.gravar_metricas(pasta, "test", m)
    diretorio.gravar_metricas(pasta, "test_diagnostico_estagios", _diagnostico(test.rotulos, roteados, preds2))
    linha = trilha.registrar(pasta / diretorio.TRILHA, execucao=pasta.name, model=sidecar["model"],
                             seed=config["seed"], config_sha1=trilha.sha1_config(registro),
                             n_test=len(test), macro_f1=m["macro_f1"],
                             negation_of_f1=m["f1_per_class"]["negation_of"],
                             extra={"estrategia": NOME, "limiar_tau": tau})
    log.info("TEST avaliado (avaliação %d desta configuração): macro-F1=%.4f | F1 negation_of=%.4f",
             linha["eval_index_for_config"], m["macro_f1"], m["f1_per_class"]["negation_of"])
    return {"metricas": m, "trilha": linha, "tau": tau}
