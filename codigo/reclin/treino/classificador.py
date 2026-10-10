"""Treino de uma estratégia treinada e avaliação do TEST, em operações separadas.

O padrão é o classificador da tarefa: todos os pares candidatos (`tarefa`), a
janela marcada (`entrada`), o encoder com a cabeça de classificação de
sequência ([CLS] → 3 rótulos, `modelos`) e a configuração base
(`config.Config`). Uma estratégia muda o que cabe numa `treino.montagem.Montagem`
— quais candidatos entram, qual modelo, o que vai para os sidecars — e o resto
é o mesmo código para todas (o baseline, o restrito e a Pair-Aware de
`reclin.estrategias`).

`treinar_execucao` — lê só TRAIN e DEV (o arquivo do TEST não é aberto),
treina com o laço de `treino.laco`, grava checkpoints e, ao concluir, as
predições do DEV na melhor época. A ordem das operações é a do legado, porque
decide o consumo dos geradores aleatórios: sementes → dados → tokenizer →
modelo → loaders → perda → otimizador → agendador.

`avaliar_test` — operação explícita, depois do treino: recarrega o
`melhor_modelo/` gravado, prevê o TEST, grava as predições e as métricas e
acrescenta uma linha à trilha de avaliações do TEST (`execucao.trilha`). Nada
do que ela faz volta para o treino ou para a escolha da época.

O diretório da execução (formato de `execucao.diretorio`):

    <raiz>/<nome>/
        config.json            configuração, modelo, partições usadas
        treino.json            histórico do DEV, melhor época, sessões (nova e
                               retomadas), ambiente e avisos
        checkpoints/           estado de retomada e melhor_modelo/ (`treino.checkpoint`)
        predicoes_dev.json     DEV na melhor época (ao concluir o treino), no conjunto
                               completo (com seleção, remapeado)
        metricas.json          métricas do DEV e, depois de avaliar_test, do TEST (com
                               seleção, também as do subconjunto: dev_restrito, test_restrito)
        predicoes_test.json    só depois de avaliar_test
        avaliacoes_test.jsonl  trilha das avaliações do TEST

Origem no legado: `run()` de `src/relation_extraction.py` e as cópias dele em
`train_restrito.py` e `train_pair_aware.py`, separados em duas operações. No
legado o TEST era previsto no fim do mesmo comando de treino (a Pair-Aware só
com `--avaliar-test`).
"""
from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

import torch
from transformers import get_linear_schedule_with_warmup

from reclin import entrada, modelos, particoes, tarefa
from reclin.avaliacao import metricas
from reclin.config import Config
from reclin.execucao import diretorio, predicoes, subconjunto, trilha
from reclin.treino import checkpoint, laco, reprodutibilidade
from reclin.treino.montagem import CLASSIFICADOR, Montagem
from reclin.util import caminhos
from reclin.util.io import gravar_json, ler_json

log = logging.getLogger(__name__)

ESTRATEGIA = CLASSIFICADOR.estrategia
ARQUIVO_TREINO = "treino.json"
PARTICOES_TREINO = ("train", "dev")


class ErroExecucao(RuntimeError):
    """A operação não pode ser feita nesta execução."""


def _agora() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _dispositivo(pedido: str | None) -> torch.device:
    if pedido in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pedido == "cuda" and not torch.cuda.is_available():
        raise ErroExecucao("dispositivo cuda pedido, mas não há GPU disponível")
    return torch.device(pedido)


def _conferir_particoes(pasta: Path, nomes: tuple[str, ...]) -> dict[str, str]:
    divergencias = particoes.conferir_particoes(pasta, nomes)
    if divergencias:
        raise ErroExecucao("partições não conferem com o MANIFEST: " + "; ".join(divergencias))
    splits = particoes.ler_manifesto(pasta)["splits"]
    return {nome: splits[nome]["sha256"] for nome in nomes}


def _conjunto(nome: str, docs: list, sha: str, config: Config, pasta_particoes: Path):
    congeladas = Path(pasta_particoes).resolve() == caminhos.PARTICOES.resolve()
    return tarefa.conjunto_referencia(nome, docs, particao_sha256=sha, max_gap=config.max_gap,
                                      conferir_tamanho=congeladas)


def config_de_registro(registro: dict[str, Any], classe: type = Config) -> Config:
    """A configuração gravada em `config.json` (campos de `como_dict`)."""
    return classe(**{f.name: registro[f.name] for f in dataclasses.fields(classe)})


def _selecao(montagem: Montagem, docs: list, config: Config, completos: entrada.Exemplos):
    """(índices, exemplos usados) da partição; índices None = todos."""
    if montagem.selecionar is None:
        return None, completos
    indices = list(montagem.selecionar(tarefa.candidatos(docs, max_gap=config.max_gap)))
    subconjunto.conferir_indices(indices, len(completos))
    return indices, completos.subconjunto(indices)


def _metricas_dev(y_completo: list[int], indices: list[int] | None, y_sub: list[int],
                  preds: list[int]) -> tuple[dict, dict | None, list[int]]:
    """Métricas no conjunto completo (remapeado, se houver seleção), do
    subconjunto e as predições no conjunto completo."""
    if indices is None:
        return metricas.avaliar(y_completo, preds), None, preds
    completas, _ = subconjunto.remapear(indices, len(y_completo), preds)
    return metricas.avaliar(y_completo, completas), metricas.avaliar(y_sub, preds), completas


def _pontuador(y_completo: list[int], indices: list[int] | None, y_sub: list[int]):
    def pontuar(preds: list[int]) -> dict[str, float]:
        m, m_sub, _ = _metricas_dev(y_completo, indices, y_sub, preds)
        pontos = {"dev_macro_f1": m["macro_f1"], "dev_negation_of_f1": m["f1_per_class"]["negation_of"]}
        if m_sub is not None:
            pontos.update({"dev_restrito_macro_f1": m_sub["macro_f1"],
                           "dev_restrito_negation_of_f1": m_sub["f1_per_class"]["negation_of"]})
        return pontos
    return pontuar


def treinar_execucao(raiz: str | Path, nome: str, config: Config, *, modelo: str | Path | None = None,
                     pasta_particoes: str | Path = caminhos.PARTICOES, retomar: bool = False,
                     checkpoint_a_cada: int | None = None, parar_apos_passo: int | None = None,
                     dispositivo: str | None = None, montagem: Montagem = CLASSIFICADOR) -> dict[str, Any]:
    """Treina (ou retoma) a execução `<raiz>/<nome>` da estratégia descrita
    por `montagem` e devolve o conteúdo de `treino.json`. `modelo` substitui o
    checkpoint de `config.encoder` (um caminho local, por exemplo)."""
    if not isinstance(config, montagem.classe_config):
        raise ErroExecucao(f"a estratégia {montagem.estrategia} usa {montagem.classe_config.__name__}, "
                           f"não {type(config).__name__}")
    pasta_particoes = Path(pasta_particoes)
    pasta = Path(raiz) / nome
    modelo_id = str(modelo) if modelo is not None else config.modelo
    shas = _conferir_particoes(pasta_particoes, PARTICOES_TREINO)
    registro = {"estrategia": montagem.estrategia, "config": config.como_dict(), "modelo": modelo_id,
                "particoes": shas, "estrategia_config": montagem.registro}

    if retomar:
        if not (pasta / diretorio.CONFIG).is_file():
            raise ErroExecucao(f"não há execução em {pasta} para retomar")
        gravado = diretorio.ler_config(pasta)["config"]
        if gravado != registro:
            divergentes = sorted(k for k in set(gravado) | set(registro) if gravado.get(k) != registro.get(k))
            raise ErroExecucao(f"a execução {pasta} foi criada com outra configuração ({divergentes}); "
                               "retome com a mesma estratégia e os mesmos parâmetros")
        anterior = ler_json(pasta / ARQUIVO_TREINO) if (pasta / ARQUIVO_TREINO).is_file() else {}
        if anterior.get("concluido"):
            raise ErroExecucao(f"o treino de {pasta} já foi concluído: não há o que retomar")
    else:
        if (pasta / diretorio.CONFIG).exists():
            raise ErroExecucao(f"a execução {pasta} já existe: para continuar, use retomar; "
                               "para recomeçar, escolha outro nome")
        diretorio.criar(raiz, nome, estrategia=montagem.estrategia, config=registro)
        anterior = {}

    sessao = {"tipo": "retomada" if retomar else "nova", "inicio": _agora()}
    with reprodutibilidade.coletar_avisos() as avisos:
        # Mesma ordem do legado: sementes → dados → tokenizer → modelo → loaders.
        reprodutibilidade.fixar_sementes(config.seed)
        disp = _dispositivo(dispositivo)
        docs = {p: particoes.ler_particao(pasta_particoes, p) for p in PARTICOES_TREINO}
        completos = {p: entrada.exemplos(docs[p], max_gap=config.max_gap, ctx_chars=config.ctx_chars)
                     for p in PARTICOES_TREINO}
        indices, usados = {}, {}
        for p in PARTICOES_TREINO:
            indices[p], usados[p] = _selecao(montagem, docs[p], config, completos[p])
        log.info("Exemplos: train=%d de %d | dev=%d de %d (o TEST não é lido pelo treino)",
                 len(usados["train"]), len(completos["train"]), len(usados["dev"]), len(completos["dev"]))

        tokenizer = modelos.carregar_tokenizer(modelo_id)
        rede = montagem.criar_modelo(modelo_id, tokenizer, config)
        rede.to(disp)
        identidade = {"execucao": registro, "modelo": modelos.identidade(modelo_id, tokenizer, rede)}
        n_params = modelos.contar_parametros(rede)
        descricao = (montagem.descrever(tokenizer, rede, usados) if montagem.descrever else None)

        loader_treino = entrada.criar_loader(tokenizer, usados["train"], max_length=config.max_length,
                                             batch_size=config.batch_size, embaralhar=True, seed=config.seed)
        loader_dev = entrada.criar_loader(tokenizer, usados["dev"], max_length=config.max_length,
                                          batch_size=config.batch_size, embaralhar=False, seed=config.seed)

        pesos = laco.pesos_balanced(usados["train"].rotulos) if config.class_weight == "balanced" else None
        perda = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(pesos, dtype=torch.float).to(disp) if pesos is not None else None)
        otimizador = torch.optim.AdamW(rede.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        total = len(loader_treino) * config.epochs
        aquecimento = int(config.warmup_ratio * total)
        agendador = get_linear_schedule_with_warmup(otimizador, aquecimento, total)
        log.info("Estratégia %s | dispositivo %s | %d parâmetros | %d passos (%d por época), aquecimento %d",
                 montagem.estrategia, disp, n_params, total, len(loader_treino), aquecimento)

        resultado = laco.treinar(
            modelo=rede, tokenizer=tokenizer, loader_treino=loader_treino, loader_dev=loader_dev,
            perda=perda, otimizador=otimizador, agendador=agendador, epocas=config.epochs,
            max_grad_norm=config.max_grad_norm, dispositivo=disp,
            pontuar_dev=_pontuador(completos["dev"].rotulos, indices["dev"], usados["dev"].rotulos),
            pasta_execucao=pasta, identidade=identidade, retomar=retomar,
            checkpoint_a_cada=checkpoint_a_cada, parar_apos_passo=parar_apos_passo)

        melhor_ep, melhor_f1 = laco.melhor_epoca(resultado.historico)
        restaurado = None
        if resultado.concluido:
            restaurado = _predicoes_dev(pasta, rede, resultado, loader_dev, disp, completos["dev"],
                                        indices["dev"], usados["dev"], docs["dev"], shas["dev"], config,
                                        modelo_id, pasta_particoes, montagem)

    sessao.update({"fim": _agora(), "status": "concluido" if resultado.concluido else "interrompido",
                   "a_partir_de": resultado.retomado_de, "ate": resultado.posicao,
                   "checkpoints_gravados": len(resultado.checkpoints),
                   "ambiente": reprodutibilidade.ambiente(), "avisos": avisos})
    registro_treino = {
        "nome": nome, "estrategia": montagem.estrategia, "modelo": modelo_id, "seed": config.seed,
        "dispositivo": str(disp), "config": config.como_dict(),
        "n_params": n_params, "n_candidatos": {p: len(completos[p]) for p in PARTICOES_TREINO},
        "pesos_classe": pesos,
        "passos": {"por_epoca": len(loader_treino), "total": total, "aquecimento": aquecimento},
        "identidade_modelo": identidade["modelo"],
        "dev_history": resultado.historico,
        "melhor_epoca": melhor_ep, "melhor_dev_macro_f1": melhor_f1,
        "restaurou_melhor_estado": restaurado,
        "concluido": resultado.concluido, "posicao": resultado.posicao,
        "test_avaliado": False,
        "sessoes": anterior.get("sessoes", []) + [sessao],
    }
    if montagem.selecionar is not None:
        registro_treino["subconjunto"] = {p: subconjunto.resumo(indices[p], completos[p].rotulos)
                                          for p in PARTICOES_TREINO}
    if descricao is not None:
        registro_treino["descricao"] = descricao
    gravar_json(pasta / ARQUIVO_TREINO, registro_treino)
    log.info("Treino %s: %s", "concluído" if resultado.concluido else "interrompido", pasta)
    return registro_treino


def _predicoes_dev(pasta: Path, rede: Any, resultado: laco.ResultadoTreino, loader_dev: Any, disp: Any,
                   dev: entrada.Exemplos, indices: list[int] | None, dev_usado: entrada.Exemplos,
                   docs_dev: list, sha_dev: str, config: Config, modelo_id: str,
                   pasta_particoes: Path, montagem: Montagem) -> bool:
    """DEV com os pesos da melhor época (os mesmos que o TEST usará)."""
    melhor_estado = resultado.melhor_estado or checkpoint.estado_do_melhor(pasta, rede)
    restaurado = bool(melhor_estado)
    if restaurado:
        rede.load_state_dict(melhor_estado)
    else:
        log.warning("Sem pesos da melhor época: as predições do DEV são da última época")
    melhor_ep, melhor_f1 = laco.melhor_epoca(resultado.historico)
    preds, probs = laco.prever(rede, loader_dev, disp, com_probs=True)
    m, m_sub, preds_completas = _metricas_dev(dev.rotulos, indices, dev_usado.rotulos, preds)
    if melhor_f1 is not None and abs(m["macro_f1"] - melhor_f1) > 1e-6:
        log.warning("macro-F1 do DEV recalculado (%.6f) difere do histórico (%.6f) da época %s",
                    m["macro_f1"], melhor_f1, melhor_ep)
    if indices is None:
        extra = {"split": "dev", "best_epoch": melhor_ep, "best_dev_macro_f1_history": melhor_f1,
                 "dev_macro_f1_recomputed": m["macro_f1"], "restored_best_state": restaurado}
    else:
        _, probs = subconjunto.remapear(indices, len(dev), preds, probs)
        extra = {**montagem.extras_sidecar, "best_epoch": melhor_ep, "restored_best_state": restaurado,
                 "split": "dev", "n_restrito": len(indices), "restricted_indices": list(indices),
                 "best_dev_macro_f1_history": melhor_f1, "dev_macro_f1_recomputed": m["macro_f1"],
                 "dev_negation_of_f1_recomputed": m["f1_per_class"]["negation_of"]}
    sidecar = predicoes.montar(model=modelo_id, seed=config.seed, y_true=dev.rotulos, y_pred=preds_completas,
                               probs=probs, extra=extra)
    diretorio.gravar_predicoes(pasta, "dev", sidecar, _conjunto("dev", docs_dev, sha_dev, config, pasta_particoes))
    diretorio.gravar_metricas(pasta, "dev", m)
    if m_sub is not None:
        diretorio.gravar_metricas(pasta, "dev_restrito", m_sub)
    return restaurado


def avaliar_test(pasta: str | Path, *, pasta_particoes: str | Path = caminhos.PARTICOES,
                 dispositivo: str | None = None, reavaliar: bool = False,
                 montagem: Montagem = CLASSIFICADOR) -> dict[str, Any]:
    """Avalia o TEST com o `melhor_modelo/` de uma execução concluída: grava
    `predicoes_test.json`, as métricas do TEST e uma linha na trilha. Uma
    segunda avaliação exige `reavaliar=True` e fica registrada na trilha.
    `montagem` é a da estratégia que treinou a execução (conferida contra
    `config.json`)."""
    pasta, pasta_particoes = Path(pasta), Path(pasta_particoes)
    if not (pasta / ARQUIVO_TREINO).is_file():
        raise ErroExecucao(f"{pasta} não é uma execução de treino (sem {ARQUIVO_TREINO})")
    treino = ler_json(pasta / ARQUIVO_TREINO)
    registro = diretorio.ler_config(pasta)["config"]
    if registro.get("estrategia") != montagem.estrategia or \
            registro.get("estrategia_config") != montagem.registro:
        raise ErroExecucao(f"a execução {pasta.name} é da estratégia {registro.get('estrategia')!r} "
                           f"(com a configuração gravada em config.json); a montagem pedida é "
                           f"{montagem.estrategia!r}")
    if not treino.get("concluido"):
        raise ErroExecucao("o treino desta execução não terminou: conclua-o (retomando) antes de avaliar o TEST")
    if not checkpoint.tem_melhor(pasta):
        raise ErroExecucao(f"não há {checkpoint.MELHOR}/ em {checkpoint.pasta_checkpoints(pasta)}")
    if (pasta / diretorio.arquivo_predicoes("test")).exists() and not reavaliar:
        raise ErroExecucao("o TEST desta execução já foi avaliado; uma nova avaliação exige reavaliar "
                           "e fica registrada na trilha como reavaliação")
    config = config_de_registro(registro["config"], montagem.classe_config)
    modelo_id = registro["modelo"]
    sha = _conferir_particoes(pasta_particoes, ("test",))["test"]

    disp = _dispositivo(dispositivo)
    docs = particoes.ler_particao(pasta_particoes, "test")
    test = entrada.exemplos(docs, max_gap=config.max_gap, ctx_chars=config.ctx_chars)
    indices, usado = _selecao(montagem, docs, config, test)
    tokenizer, rede = modelos.recarregar(checkpoint.pasta_checkpoints(pasta) / checkpoint.MELHOR,
                                         montagem.classe_modelo)
    rede.to(disp)
    if montagem.descrever is not None:
        treino["descricao_test"] = montagem.descrever(tokenizer, rede, {"test": usado})
    loader = entrada.criar_loader(tokenizer, usado, max_length=config.max_length,
                                  batch_size=config.batch_size, embaralhar=False, seed=config.seed)
    preds, probs = laco.prever(rede, loader, disp, com_probs=True)
    m, m_sub, preds_completas = _metricas_dev(test.rotulos, indices, usado.rotulos, preds)
    extra = None
    if indices is not None:
        _, probs = subconjunto.remapear(indices, len(test), preds, probs)
        extra = {**montagem.extras_sidecar, "best_epoch": treino["melhor_epoca"],
                 "restored_best_state": treino.get("restaurou_melhor_estado"), "split": "test",
                 "n_restrito": len(indices), "restricted_indices": list(indices)}
    sidecar = predicoes.montar(model=modelo_id, seed=config.seed, y_true=test.rotulos, y_pred=preds_completas,
                               probs=probs, extra=extra)
    diretorio.gravar_predicoes(pasta, "test", sidecar,
                               _conjunto("test", docs, sha, config, pasta_particoes), sobrescrever=reavaliar)
    diretorio.gravar_metricas(pasta, "test", m)
    if m_sub is not None:
        diretorio.gravar_metricas(pasta, "test_restrito", m_sub)
    linha = trilha.registrar(pasta / diretorio.TRILHA, execucao=pasta.name, model=modelo_id,
                             seed=config.seed, config_sha1=trilha.sha1_config(registro),
                             n_test=len(test), macro_f1=m["macro_f1"],
                             negation_of_f1=m["f1_per_class"]["negation_of"],
                             extra={"estrategia": montagem.estrategia, "melhor_epoca": treino["melhor_epoca"]})
    treino["test_avaliado"] = True
    gravar_json(pasta / ARQUIVO_TREINO, treino)
    log.info("TEST avaliado (avaliação %d desta configuração): macro-F1=%.4f | F1 negation_of=%.4f",
             linha["eval_index_for_config"], m["macro_f1"], m["f1_per_class"]["negation_of"])
    return {"metricas": m, "metricas_restrito": m_sub, "trilha": linha}
