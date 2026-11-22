"""Reempacotamento: uma execução treinada por outra versão do código, sob o código atual.

As operações que acrescentam artefatos a uma execução (avaliação do TEST,
execuções derivadas) exigem que o código atual seja o mesmo que a criou
(`execucao.diretorio.conferir_codigo`). Uma execução treinada antes de uma
mudança de código — por exemplo, a Pair-Aware sem pesos de classe, treinada
com o código `0526d912…` e escolhida como configuração final depois da Etapa 5
— não pode, então, ser avaliada no TEST pelo código que tem a trava do TEST.

O reempacotamento resolve isso sem afrouxar a regra e sem treinar de novo:

1. lê a execução de base (config, treino concluído, TEST nunca avaliado,
   `predicoes_dev.json` e `checkpoints/melhor_modelo/`) e o DEV (só a partição
   DEV, conferida contra o MANIFEST; o arquivo do TEST não é aberto);
2. recarrega o `melhor_modelo/` da base com o código atual e prevê o DEV
   inteiro;
3. compara com o `predicoes_dev.json` gravado pela base: mesmo `y_true`, a
   mesma classe prevista em todos os pares e as probabilidades (4 casas) com
   diferença absoluta máxima <= `TOLERANCIA_PROBS`. Qualquer divergência
   recusa a operação, e nada é gravado;
4. só então cria uma execução nova, do código atual, com a mesma
   configuração (`config.json` → `config` idêntico ao da base), uma cópia
   conferida (hash) do `melhor_modelo/`, as predições e métricas do DEV
   recalculadas e um `treino.json` que repete o da base com o bloco
   `reempacotamento` (de onde veio, com que código, o resultado da
   comparação). O `config.json` também ganha o bloco `reempacotamento`.

Os pesos não mudam: o reempacotamento prova, no DEV, que o código atual
reproduz as predições da base, e registra essa prova. A base fica intacta.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from reclin import entrada, modelos, particoes, pretreino
from reclin.execucao import diretorio, predicoes, validacao
from reclin.treino import checkpoint, classificador, laco, reprodutibilidade
from reclin.treino.montagem import Montagem
from reclin.util import identidade_codigo
from reclin.util.io import gravar_json, ler_json, sha256_arquivo

log = logging.getLogger(__name__)

TOLERANCIA_PROBS = 1e-3
BLOCO = "reempacotamento"


class ErroReempacotamento(RuntimeError):
    """A execução não pode ser reempacotada (ou o código atual não a reproduz)."""


def comparar_sidecars(gravado: Mapping[str, Any], novo: Mapping[str, Any],
                      tolerancia: float = TOLERANCIA_PROBS) -> dict[str, Any]:
    """Compara as predições do DEV gravadas pela base com as recalculadas."""
    if list(gravado["y_true"]) != list(novo["y_true"]):
        return {"equivalente": False, "motivo": "y_true diferente", "n": len(novo["y_true"])}
    divergentes = [i for i, (a, b) in enumerate(zip(gravado["y_pred"], novo["y_pred"])) if a != b]
    probs_a, probs_b = gravado.get("probs"), novo.get("probs")
    if not probs_a or not probs_b or len(probs_a) != len(probs_b):
        return {"equivalente": False, "motivo": "probabilidades ausentes ou de tamanhos diferentes",
                "n": len(novo["y_true"]), "classes_divergentes": len(divergentes)}
    max_dif = max((abs(x - y) for pa, pb in zip(probs_a, probs_b) for x, y in zip(pa, pb)), default=0.0)
    equivalente = not divergentes and max_dif <= tolerancia
    resultado = {"equivalente": equivalente, "n": len(novo["y_true"]), "classes_divergentes": len(divergentes),
                 "indices_divergentes": divergentes[:20], "max_dif_prob": round(max_dif, 6),
                 "tolerancia_prob": tolerancia}
    if not equivalente:
        resultado["motivo"] = (f"{len(divergentes)} pares com outra classe prevista" if divergentes
                               else f"diferença máxima de probabilidade {max_dif:.6f} > {tolerancia}")
    return resultado


def _ler_base(base: Path, montagem: Montagem) -> dict[str, Any]:
    if not (base / diretorio.CONFIG).is_file() or not (base / classificador.ARQUIVO_TREINO).is_file():
        raise ErroReempacotamento(f"{base} não é uma execução treinada (sem config.json ou treino.json)")
    config_json = diretorio.ler_config(base)
    registro = config_json["config"]
    if validacao.da_config(registro):
        raise ErroReempacotamento(f"{base.name} é uma execução de validação técnica: não é reempacotada")
    if registro.get("estrategia") != montagem.estrategia or registro.get("estrategia_config") != montagem.registro:
        raise ErroReempacotamento(f"{base.name} é da estratégia {registro.get('estrategia')!r} (com a configuração "
                                  f"gravada); a montagem pedida é {montagem.estrategia!r}")
    treino = ler_json(base / classificador.ARQUIVO_TREINO)
    if not treino.get("concluido"):
        raise ErroReempacotamento(f"o treino de {base.name} não terminou")
    if treino.get("test_avaliado") or (base / diretorio.arquivo_predicoes("test")).exists():
        raise ErroReempacotamento(f"o TEST de {base.name} já foi avaliado: a execução não é reempacotada")
    if not checkpoint.tem_melhor(base):
        raise ErroReempacotamento(f"{base.name} não tem {checkpoint.MELHOR}/ em {checkpoint.pasta_checkpoints(base)}")
    if not (base / diretorio.arquivo_predicoes("dev")).is_file():
        raise ErroReempacotamento(f"{base.name} não tem predicoes_dev.json")
    if not (config_json.get("conjuntos") or {}).get("dev"):
        raise ErroReempacotamento(f"{base.name} não registrou o conjunto de referência do DEV")
    return {"config_json": config_json, "registro": registro, "treino": treino}


def reempacotar(base: str | Path, raiz: str | Path, nome: str, *, pasta_particoes: str | Path,
                montagem: Montagem, dispositivo: str | None = None,
                tolerancia: float = TOLERANCIA_PROBS) -> dict[str, Any]:
    """Reempacota `base` em `<raiz>/<nome>` (ver o módulo). Lê só o DEV."""
    base, destino = Path(base), Path(raiz) / nome
    if (destino / diretorio.CONFIG).exists() or destino.exists():
        raise ErroReempacotamento(f"{destino} já existe")
    info = _ler_base(base, montagem)
    registro = info["registro"]
    config = classificador.config_de_registro(registro["config"], montagem.classe_config)
    pasta_particoes = Path(pasta_particoes)
    try:
        sha_dev = classificador._conferir_particoes(pasta_particoes, ("dev",))["dev"]
    except classificador.ErroExecucao as erro:
        raise ErroReempacotamento(str(erro)) from erro
    if (registro.get("particoes") or {}).get("dev") not in (None, sha_dev):
        raise ErroReempacotamento(f"o DEV de {pasta_particoes} não é o da base ({registro['particoes']['dev'][:12]}…)")

    melhor = checkpoint.pasta_checkpoints(base) / checkpoint.MELHOR
    sha_melhor = pretreino.hash_pasta(melhor)
    gravado = predicoes.ler(base / diretorio.arquivo_predicoes("dev"))

    disp = classificador._dispositivo(dispositivo)
    reprodutibilidade.fixar_sementes(config.seed)
    docs = particoes.ler_particao(pasta_particoes, "dev")
    dev = entrada.exemplos(docs, max_gap=config.max_gap, ctx_chars=config.ctx_chars)
    indices, usado = classificador._selecao(montagem, docs, config, dev)
    tokenizer, rede = modelos.recarregar(melhor, montagem.classe_modelo)
    rede.to(disp)
    loader = entrada.criar_loader(tokenizer, usado, max_length=config.max_length, batch_size=config.batch_size,
                                  embaralhar=False, seed=config.seed)
    preds, probs = laco.prever(rede, loader, disp, com_probs=True)
    m, m_sub, preds_completas = classificador._metricas_dev(dev.rotulos, indices, usado.rotulos, preds)
    extra = {"split": "dev", "best_epoch": gravado.get("best_epoch"),
             "best_dev_macro_f1_history": gravado.get("best_dev_macro_f1_history"),
             "dev_macro_f1_recomputed": m["macro_f1"], "restored_best_state": gravado.get("restored_best_state"),
             "reempacotado_de": base.name}
    if indices is not None:
        from reclin.execucao import subconjunto
        _, probs = subconjunto.remapear(indices, len(dev), preds, probs)
        extra.update({**montagem.extras_sidecar, "n_restrito": len(indices), "restricted_indices": list(indices)})
    novo = predicoes.montar(model=registro["modelo"], seed=config.seed, y_true=dev.rotulos, y_pred=preds_completas,
                            probs=probs, extra=extra)
    comparacao = comparar_sidecars(gravado, novo, tolerancia)
    conjunto = classificador._conjunto("dev", docs, sha_dev, config, pasta_particoes)
    if dict(conjunto) != dict(info["config_json"]["conjuntos"]["dev"]):
        comparacao.update({"equivalente": False, "motivo": "o conjunto de referência do DEV difere do da base"})
    if not comparacao["equivalente"]:
        raise ErroReempacotamento(f"o código atual não reproduz as predições do DEV de {base.name}: "
                                  f"{comparacao['motivo']}. Nada foi gravado.")

    atual = identidade_codigo.atual()
    bloco = {"base": base.name, "codigo_da_base": (info["config_json"].get("codigo") or {}).get("identidade"),
             "codigo_atual": atual["identidade"],
             "arquivos_de_codigo_diferentes": identidade_codigo.diferencas(info["config_json"]["codigo"], atual)
             if info["config_json"].get("codigo") else None,
             "sha256_config_base": sha256_arquivo(base / diretorio.CONFIG),
             "sha256_predicoes_dev_base": sha256_arquivo(base / diretorio.arquivo_predicoes("dev")),
             "sha256_melhor_modelo": sha_melhor, "comparacao_dev": comparacao,
             "em": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "natureza": "mesmos pesos e mesma configuração; o código atual reproduziu as predições do DEV"}

    diretorio.criar(raiz, nome, estrategia=montagem.estrategia, config=dict(registro))
    copia = checkpoint.pasta_checkpoints(destino) / checkpoint.MELHOR
    shutil.copytree(melhor, copia)
    if pretreino.hash_pasta(copia) != sha_melhor:
        raise ErroReempacotamento(f"a cópia de {melhor} não confere com o original")
    diretorio.gravar_predicoes(destino, "dev", novo, conjunto)
    diretorio.gravar_metricas(destino, "dev", m)
    if m_sub is not None:
        diretorio.gravar_metricas(destino, "dev_restrito", m_sub)
    config_json = diretorio.ler_config(destino)
    config_json[BLOCO] = bloco
    gravar_json(destino / diretorio.CONFIG, config_json)
    treino = {**info["treino"], "nome": nome, "test_avaliado": False, BLOCO: bloco}
    gravar_json(destino / classificador.ARQUIVO_TREINO, treino)
    log.info("Reempacotada: %s → %s (%d pares do DEV, mesma classe em todos; dif. máx. de probabilidade %.6f)",
             base.name, nome, comparacao["n"], comparacao["max_dif_prob"])
    return {"pasta": destino, "comparacao": comparacao, "metricas_dev": m, "sha256_melhor_modelo": sha_melhor}
