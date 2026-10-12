"""Treino de um estágio do Two-Stage sobre a infraestrutura genérica.

`treinar_estagio` faz, para um estágio, o que `treino.classificador` faz para
o classificador da tarefa, com o mesmo laço (`treino.laco`, retomada exata),
os mesmos checkpoints (`treino.checkpoint`), a mesma reprodutibilidade e a
mesma ordem de consumo dos geradores: sementes → dados → tokenizer → modelo →
loaders → perda → otimizador → agendador. O que muda:

* o ponto de partida: só a pasta do pré-treino com `PROVENIENCIA.json`,
  conferida e verificada (`inicializacao`); um checkpoint ajustado é recusado;
* os exemplos e os rótulos do estágio (`estagios.exemplos_do_estagio`);
* a cabeça de 2 saídas e os pesos `balanced` calculados no TRAIN do estágio;
* o critério de escolha do checkpoint (`CAMPO_SELECAO` do estágio), passado
  ao laço por `campo_selecao` e gravado na identidade da execução;
* ao concluir, as predições do DEV do próprio estágio
  (`predicoes_dev_estagio.json`, binárias) — diagnóstico do estágio, não
  resultado da arquitetura. O sidecar da tarefa (3 rótulos) só existe na
  composição.

O arquivo do TEST nunca é aberto: o treino lê só TRAIN e DEV.

O diretório do estágio:

    <raiz>/<nome>/
        config.json                  estratégia (two_stage_e1 ou two_stage_e2), configuração,
                                     modelo, partições, proveniência dos pesos, marca de validação
        treino.json                  histórico do DEV, melhor época, inicialização verificada,
                                     sessões, hash do melhor_modelo/
        checkpoints/                 estado de retomada e melhor_modelo/
        predicoes_dev_estagio.json   DEV do estágio na melhor época
        metricas_estagio.json        métricas do DEV do estágio
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from transformers import get_linear_schedule_with_warmup

from reclin import entrada, modelos, particoes, pretreino
from reclin.config import Config
from reclin.estrategias.two_stage import (CAMPO_SELECAO, CONFIGS, DADOS_TREINO, DESCRICAO_SELECAO,
                                         ESTRATEGIA_ESTAGIO, ROTULOS)
from reclin.estrategias.two_stage import estagios, inicializacao
from reclin.estrategias.two_stage import metricas as metricas_estagio
from reclin.execucao import diretorio, predicoes, validacao
from reclin.treino import checkpoint, laco, reprodutibilidade
from reclin.treino.classificador import ErroExecucao
from reclin.util import caminhos, identidade_codigo
from reclin.util.io import gravar_json, ler_json

log = logging.getLogger(__name__)

ARQUIVO_TREINO = "treino.json"
PREDICOES_ESTAGIO = "predicoes_dev_estagio.json"
METRICAS_ESTAGIO = "metricas_estagio.json"
PARTICOES_TREINO = ("train", "dev")


def _agora() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def dispositivo(pedido: str | None) -> torch.device:
    if pedido in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pedido == "cuda" and not torch.cuda.is_available():
        raise ErroExecucao("dispositivo cuda pedido, mas não há GPU disponível")
    return torch.device(pedido)


def conferir_particoes(pasta: Path, nomes: tuple[str, ...]) -> dict[str, str]:
    divergencias = particoes.conferir_particoes(pasta, nomes)
    if divergencias:
        raise ErroExecucao("partições não conferem com o MANIFEST: " + "; ".join(divergencias))
    splits = particoes.ler_manifesto(pasta)["splits"]
    return {nome: splits[nome]["sha256"] for nome in nomes}


def registro_estagio(estagio: int) -> dict[str, Any]:
    return {"estagio": estagio, "rotulos": list(ROTULOS[estagio]), "dados_treino": DADOS_TREINO[estagio],
            "campo_selecao": CAMPO_SELECAO[estagio], "selecao_melhor_epoca": DESCRICAO_SELECAO[estagio],
            "inicializacao": "pesos originais do pré-treino (pasta com PROVENIENCIA.json), "
                             "sem checkpoint ajustado", "pesos_classe": "balanced no TRAIN do estágio"}


def carregar_modelo(modelo: str | Path, estagio: int) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Tokenizer, modelo, relatório de inicialização e proveniência — com as
    sementes já fixadas por quem chama."""
    proveniencia = inicializacao.recusar_origem(modelo)
    tokenizer = modelos.carregar_tokenizer(modelo)
    rede, info = estagios.criar_modelo(str(modelo), tokenizer, estagio)
    relatorio = inicializacao.verificar_carga(modelo, rede, info, tokenizer)
    relatorio["sha256_encoder_completo_inicial"] = inicializacao.hash_encoder(
        rede, relatorio["marcadores"]["vocabulario_original"])
    return tokenizer, rede, relatorio, proveniencia


def treinar_estagio(raiz: str | Path, nome: str, estagio: int, config: Config, *, modelo: str | Path,
                    pasta_particoes: str | Path = caminhos.PARTICOES, retomar: bool = False,
                    checkpoint_a_cada: int | None = None, parar_apos_passo: int | None = None,
                    dispositivo_pedido: str | None = None,
                    marca_validacao: dict[str, Any] | None = None) -> dict[str, Any]:
    """Treina (ou retoma) o estágio `estagio` em `<raiz>/<nome>` e devolve o
    conteúdo de `treino.json`."""
    if estagio not in CONFIGS:
        raise ErroExecucao(f"estágio desconhecido: {estagio!r}")
    if not isinstance(config, CONFIGS[estagio]):
        raise ErroExecucao(f"o estágio {estagio} usa {CONFIGS[estagio].__name__}, não {type(config).__name__}")
    pasta_particoes, pasta = Path(pasta_particoes), Path(raiz) / nome
    try:
        proveniencia = inicializacao.recusar_origem(modelo)
    except (inicializacao.ErroInicializacao, pretreino.ErroProveniencia) as erro:
        raise ErroExecucao(str(erro)) from erro
    shas = conferir_particoes(pasta_particoes, PARTICOES_TREINO)
    estrategia = ESTRATEGIA_ESTAGIO[estagio]
    campo = CAMPO_SELECAO[estagio]
    registro = {"estrategia": estrategia, "config": config.como_dict(), "modelo": str(modelo),
                "particoes": shas, "estrategia_config": registro_estagio(estagio),
                "proveniencia_pesos": proveniencia}
    if marca_validacao is not None:
        registro[validacao.CHAVE] = dict(marca_validacao)

    if retomar:
        if not (pasta / diretorio.CONFIG).is_file():
            raise ErroExecucao(f"não há execução em {pasta} para retomar")
        gravado = diretorio.ler_config(pasta)["config"]
        if gravado != registro:
            divergentes = sorted(k for k in set(gravado) | set(registro) if gravado.get(k) != registro.get(k))
            raise ErroExecucao(f"a execução {pasta} foi criada com outra configuração ({divergentes}); "
                               "retome com o mesmo estágio, o mesmo modelo e os mesmos parâmetros")
        anterior = ler_json(pasta / ARQUIVO_TREINO) if (pasta / ARQUIVO_TREINO).is_file() else {}
        if anterior.get("concluido"):
            raise ErroExecucao(f"o treino de {pasta} já foi concluído: não há o que retomar")
        try:
            diretorio.conferir_codigo(pasta, "a retomada do treino")
        except identidade_codigo.ErroCodigo as erro:
            raise ErroExecucao(str(erro)) from erro
    else:
        if (pasta / diretorio.CONFIG).exists():
            raise ErroExecucao(f"a execução {pasta} já existe: para continuar, use retomar; "
                               "para recomeçar, escolha outro nome")
        diretorio.criar(raiz, nome, estrategia=estrategia, config=registro)
        anterior = {}

    sessao = {"tipo": "retomada" if retomar else "nova", "inicio": _agora(),
              "codigo": identidade_codigo.atual()["identidade"]}
    with reprodutibilidade.coletar_avisos() as avisos:
        # Mesma ordem do baseline: sementes → dados → tokenizer → modelo → loaders.
        reprodutibilidade.fixar_sementes(config.seed)
        disp = dispositivo(dispositivo_pedido)
        docs = {p: particoes.ler_particao(pasta_particoes, p) for p in PARTICOES_TREINO}
        completos = {p: entrada.exemplos(docs[p], max_gap=config.max_gap, ctx_chars=config.ctx_chars)
                     for p in PARTICOES_TREINO}
        indices, usados = {}, {}
        for p in PARTICOES_TREINO:
            indices[p], usados[p] = estagios.exemplos_do_estagio(estagio, completos[p])
        log.info("Estágio %d | exemplos: train=%d de %d | dev=%d de %d (o TEST não é lido)", estagio,
                 len(usados["train"]), len(completos["train"]), len(usados["dev"]), len(completos["dev"]))

        try:
            tokenizer, rede, relatorio_inicial, _ = carregar_modelo(modelo, estagio)
        except (inicializacao.ErroInicializacao, pretreino.ErroProveniencia) as erro:
            raise ErroExecucao(str(erro)) from erro
        if anterior.get("inicializacao") not in (None, relatorio_inicial):
            raise ErroExecucao("a inicialização verificada nesta sessão difere da registrada na execução")
        rede.to(disp)
        identidade = {"execucao": registro, "modelo": modelos.identidade(str(modelo), tokenizer, rede)}
        n_params = modelos.contar_parametros(rede)

        loader_treino = entrada.criar_loader(tokenizer, usados["train"], max_length=config.max_length,
                                             batch_size=config.batch_size, embaralhar=True, seed=config.seed)
        loader_dev = entrada.criar_loader(tokenizer, usados["dev"], max_length=config.max_length,
                                          batch_size=config.batch_size, embaralhar=False, seed=config.seed)
        pesos = (laco.pesos_balanced(usados["train"].rotulos, n_rotulos=len(ROTULOS[estagio]))
                 if config.class_weight == "balanced" else None)
        perda = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(pesos, dtype=torch.float).to(disp) if pesos is not None else None)
        otimizador = torch.optim.AdamW(rede.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        total = len(loader_treino) * config.epochs
        aquecimento = int(config.warmup_ratio * total)
        agendador = get_linear_schedule_with_warmup(otimizador, aquecimento, total)
        log.info("Estágio %d | dispositivo %s | %d parâmetros | %d passos (%d por época), aquecimento %d",
                 estagio, disp, n_params, total, len(loader_treino), aquecimento)

        resultado = laco.treinar(
            modelo=rede, tokenizer=tokenizer, loader_treino=loader_treino, loader_dev=loader_dev,
            perda=perda, otimizador=otimizador, agendador=agendador, epocas=config.epochs,
            max_grad_norm=config.max_grad_norm, dispositivo=disp,
            pontuar_dev=estagios.pontuador(estagio, rede=rede, loader_dev=loader_dev, dispositivo=disp,
                                           y_dev=usados["dev"].rotulos,
                                           rotulos_tarefa_dev=completos["dev"].rotulos),
            pasta_execucao=pasta, identidade=identidade, retomar=retomar,
            checkpoint_a_cada=checkpoint_a_cada, parar_apos_passo=parar_apos_passo, campo_selecao=campo)

        melhor_ep, melhor_valor = laco.melhor_epoca(resultado.historico, campo)
        restaurado, final = None, None
        if resultado.concluido:
            restaurado, final = _predicoes_dev(pasta, estagio, rede, resultado, loader_dev, disp, usados["dev"],
                                               indices["dev"], melhor_ep, relatorio_inicial)

    sessao.update({"fim": _agora(), "status": "concluido" if resultado.concluido else "interrompido",
                   "a_partir_de": resultado.retomado_de, "ate": resultado.posicao,
                   "checkpoints_gravados": len(resultado.checkpoints),
                   "ambiente": reprodutibilidade.ambiente(), "avisos": avisos})
    registro_treino = {
        "nome": nome, "estrategia": estrategia, "estagio": estagio, "modelo": str(modelo), "seed": config.seed,
        "dispositivo": str(disp), "config": config.como_dict(), "campo_selecao": campo,
        "n_params": n_params, "n_candidatos": {p: len(completos[p]) for p in PARTICOES_TREINO},
        "n_exemplos": {p: len(usados[p]) for p in PARTICOES_TREINO},
        "contagem_por_rotulo": {p: {r: usados[p].rotulos.count(i) for i, r in enumerate(ROTULOS[estagio])}
                                for p in PARTICOES_TREINO},
        "pesos_classe": pesos,
        "passos": {"por_epoca": len(loader_treino), "total": total, "aquecimento": aquecimento},
        "identidade_modelo": identidade["modelo"],
        "inicializacao": relatorio_inicial,
        "dev_history": resultado.historico,
        "melhor_epoca": melhor_ep, "melhor_valor_selecao": melhor_valor,
        "restaurou_melhor_estado": restaurado,
        "concluido": resultado.concluido, "posicao": resultado.posicao,
        "sessoes": anterior.get("sessoes", []) + [sessao],
    }
    if final is not None:
        registro_treino["final"] = final
    gravar_json(pasta / ARQUIVO_TREINO, registro_treino)
    log.info("Estágio %d: treino %s em %s", estagio, "concluído" if resultado.concluido else "interrompido", pasta)
    return registro_treino


def _predicoes_dev(pasta: Path, estagio: int, rede: Any, resultado: laco.ResultadoTreino, loader_dev: Any,
                   disp: Any, dev: entrada.Exemplos, indices: list[int] | None, melhor_ep: int | None,
                   relatorio_inicial: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """DEV do estágio com os pesos da melhor época, e os hashes finais."""
    melhor_estado = resultado.melhor_estado or checkpoint.estado_do_melhor(pasta, rede)
    restaurado = bool(melhor_estado)
    if restaurado:
        rede.load_state_dict(melhor_estado)
    preds, probs = laco.prever(rede, loader_dev, disp, com_probs=True)
    positivo = 0
    m = metricas_estagio.binarias(dev.rotulos, preds, positivo=positivo)
    if estagio == 1:
        m["ap_relacao"] = metricas_estagio.ap([1 if y == positivo else 0 for y in dev.rotulos],
                                              [p[positivo] for p in probs])
    gravar_json(pasta / PREDICOES_ESTAGIO, {
        "estagio": estagio, "rotulos": list(ROTULOS[estagio]), "best_epoch": melhor_ep,
        "restored_best_state": restaurado, "indices_no_conjunto_completo": indices,
        "y_true": list(dev.rotulos), "y_pred": list(preds), "probs": predicoes.arredondar_probs(probs),
        "rotulo": "diagnóstico do estágio (não é resultado da arquitetura)"})
    gravar_json(pasta / METRICAS_ESTAGIO, {"dev": m, "rotulo": "diagnóstico do estágio"})
    final = {"sha256_melhor_modelo": pretreino.hash_pasta(checkpoint.pasta_checkpoints(pasta) / checkpoint.MELHOR),
             "sha256_encoder_completo_final": inicializacao.hash_encoder(
                 rede, relatorio_inicial["marcadores"]["vocabulario_original"])}
    final["encoder_mudou_no_treino"] = final["sha256_encoder_completo_final"] != \
        relatorio_inicial["sha256_encoder_completo_inicial"]
    return restaurado, final
