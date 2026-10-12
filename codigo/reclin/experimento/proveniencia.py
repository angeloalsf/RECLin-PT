"""O registro de proveniência `experimento.json`, ao lado do `config.json` de cada execução.

Formato verificado contra o que o projeto já grava (`execucao.diretorio`,
`treino.classificador`, `treino.checkpoint`, `pretreino`): este registro **não
repete nem substitui** `config.json`, `treino.json` nem `metricas.json` — o
inventário (`execucao.inventario`) continua lendo só esses. Ele acrescenta o que
falta para localizar e recuperar uma execução fora da máquina que a criou, e
aponta para os arquivos existentes com o SHA-256 do `config.json` (se o
`config.json` mudar, a divergência é detectada).

    <execucao>/
        config.json         (já existe) estratégia, configuração, partições, pesos, identidade do código
        treino.json         (já existe) histórico do DEV, melhor época, sessões com o ambiente
        checkpoints/        (já existe) ultimo.pt, ultimo.json, melhor_modelo/
        experimento.json    (novo) este registro

Campos (os pedidos para a proveniência; nenhum é inventado além deles e dos
de controle `schema`, `historico_de_estados`, `criado_em`, `atualizado_em`):

* `run_id`, `nome`, `alias_oficial`, `config_id`, `tentativa`, `chave_configuracao` — identificação (`ids`);
* `estrategia`, `seed`, `parametros_de_treinamento` — copiados do `config.json`;
* `codigo` — revisão exata (`revisao`, SHA de 40 caracteres, quando o código veio
  do tarball) e a identidade global do código gravada no `config.json`;
* `pesos_pretreinados` — repositório, commit e hash do manifesto (`pretreino`);
* `estado` e `historico_de_estados` — `planejada`, `iniciada`, `interrompida`,
  `concluida`, `falhou`; o estado derivado dos arquivos (`derivar_estado`) é a
  evidência, o gravado é o último declarado;
* `checkpoints` — onde estão: local, Drive (sincronização) e Hugging Face;
* `metricas` — resumo do `treino.json` e hash do `metricas.json` (os números
  continuam nesses arquivos);
* `publicacoes` — referências aos artefatos publicados, só de acréscimo;
* `ambiente` — Python, versões, GPU no momento do registro.

Nenhuma credencial entra: toda gravação passa por `segredos.exigir_sem_segredo`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from reclin.experimento import ids, segredos
from reclin.util.io import gravar_json, ler_json, sha256_arquivo

ARQUIVO = "experimento.json"
CONFIG = "config.json"
TREINO = "treino.json"
METRICAS = "metricas.json"
CHECKPOINTS = "checkpoints"
SCHEMA = "reclin-experimento/1"
ESTADOS = ("planejada", "iniciada", "interrompida", "concluida", "falhou")
TRANSICOES = {
    "planejada": {"iniciada", "falhou"},
    "iniciada": {"interrompida", "concluida", "falhou"},
    "interrompida": {"iniciada", "concluida", "falhou"},
    "concluida": set(),
    "falhou": set(),
}


class ErroProveniencia(ValueError):
    """Registro ausente, inválido ou transição de estado proibida."""


def _agora() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _gravar(pasta: Path, registro: dict[str, Any]) -> None:
    segredos.exigir_sem_segredo(registro)
    registro["atualizado_em"] = _agora()
    gravar_json(pasta / ARQUIVO, registro)


def ler(pasta: str | Path) -> dict[str, Any]:
    caminho = Path(pasta) / ARQUIVO
    if not caminho.is_file():
        raise ErroProveniencia(f"{pasta} não tem {ARQUIVO}")
    registro = ler_json(caminho)
    if registro.get("schema") != SCHEMA:
        raise ErroProveniencia(f"{caminho}: schema {registro.get('schema')!r} desconhecido (esperado {SCHEMA})")
    return registro


def codigo_do_ambiente(codigo: str | Path) -> dict[str, Any]:
    """O bloco `codigo` do registro: revisão exata quando o código veio do
    tarball (`REVISAO.json` na pasta acima de `codigo/`), senão origem `local`
    com revisão desconhecida (None) — a revisão nunca é adivinhada. Inclui a
    identidade própria da infraestrutura (`identidade`), só informativa."""
    from reclin.experimento import codigo_colab, identidade  # noqa: PLC0415

    infra = identidade.calcular(codigo)
    infra = {k: infra[k] for k in ("componente", "algoritmo", "identidade", "n_arquivos")}   # sem a lista de arquivos
    revisao = codigo_colab.ler_revisao(Path(codigo).resolve().parent)
    if revisao:
        return {"origem": "tarball", "repositorio": revisao["repositorio"], "revisao": revisao["sha"],
                "tarball_sha256": revisao["tarball_sha256"], "infraestrutura": infra}
    return {"origem": "local", "repositorio": None, "revisao": None, "tarball_sha256": None,
            "infraestrutura": infra}


def criar(pasta: str | Path, *, run: str, chave: Mapping[str, Any], codigo: Mapping[str, Any],
          ambiente: Mapping[str, Any] | None = None, alias_oficial: str | None = None,
          estado: str = "planejada", nota: str = "") -> dict[str, Any]:
    """Cria o `experimento.json` de uma execução. Falha se já existe (nunca é
    sobrescrito). A pasta pode ainda não ter `config.json` (o script de treino o
    cria); `vincular_config` completa os campos depois."""
    pasta = Path(pasta)
    if (pasta / ARQUIVO).exists():
        raise ErroProveniencia(f"{pasta / ARQUIVO} já existe; uma nova tentativa usa outro run_id")
    if estado not in ESTADOS:
        raise ErroProveniencia(f"estado {estado!r} desconhecido; opções: {ESTADOS}")
    partes = ids.decompor_run_id(run)
    cid = ids.config_id(chave)
    if cid != partes["config_id"] or chave["estrategia"] != partes["estrategia"]:
        raise ErroProveniencia(f"o run_id {run} não corresponde à chave de configuração (config_id {cid})")
    if alias_oficial is not None:
        ids.validar_nome(alias_oficial)
    agora = _agora()
    registro: dict[str, Any] = {
        "schema": SCHEMA, "run_id": run, "nome": pasta.name, "alias_oficial": alias_oficial,
        "config_id": cid, "tentativa": partes["tentativa"], "chave_configuracao": dict(chave),
        "estrategia": chave["estrategia"], "seed": (chave["config"] or {}).get("seed"),
        "parametros_de_treinamento": dict(chave["config"]), "pesos_pretreinados": dict(chave["pesos"]),
        "codigo": dict(codigo), "sha256_config": None,
        "estado": estado, "historico_de_estados": [{"estado": estado, "em": agora, "nota": nota}],
        "checkpoints": {"local": CHECKPOINTS, "drive": None, "hf": None},
        "metricas": None, "publicacoes": [], "ambiente": dict(ambiente) if ambiente else None,
        "criado_em": agora, "atualizado_em": agora}
    pasta.mkdir(parents=True, exist_ok=True)
    _gravar(pasta, registro)
    return registro


def vincular_config(pasta: str | Path) -> dict[str, Any]:
    """Depois que o script de treino criou o `config.json`: confere a
    estratégia, a semente e os pesos com a chave e grava o SHA-256 do arquivo
    e a identidade do código do `config.json`. Divergência é erro."""
    pasta = Path(pasta)
    registro = ler(pasta)
    cfg = ler_json(pasta / CONFIG)
    problemas = []
    if cfg.get("estrategia") != registro["estrategia"]:
        problemas.append(f"estratégia {cfg.get('estrategia')!r} ≠ {registro['estrategia']!r}")
    gravado = cfg.get("config") or {}
    efetiva = gravado.get("config") or {}
    if efetiva.get("seed") != registro["seed"]:
        problemas.append(f"seed {efetiva.get('seed')!r} ≠ {registro['seed']!r}")
    prov = gravado.get("proveniencia_pesos")
    if prov and any(prov.get(k) != registro["pesos_pretreinados"].get(k) for k in ("commit", "hash_manifesto")):
        problemas.append("proveniência dos pesos do config.json difere da chave")
    if efetiva and efetiva != registro["chave_configuracao"]["config"]:
        problemas.append("a configuração efetiva do config.json difere da registrada na chave")
    if problemas:
        raise ErroProveniencia(f"{pasta.name}: o config.json não corresponde à chave registrada: " + "; ".join(problemas))
    novo = registro["codigo"].get("identidade_global")
    identidade = cfg.get("codigo") or {}
    registro["codigo"] = {**registro["codigo"], "identidade_global": {
        "algoritmo": identidade.get("algoritmo"), "identidade": identidade.get("identidade")}}
    if novo and novo != registro["codigo"]["identidade_global"]:
        raise ErroProveniencia(f"{pasta.name}: a identidade do código do config.json mudou desde o registro")
    registro["sha256_config"] = sha256_arquivo(pasta / CONFIG)
    registro["parametros_de_treinamento"] = dict(efetiva) or registro["parametros_de_treinamento"]
    _gravar(pasta, registro)
    return registro


def mudar_estado(pasta: str | Path, estado: str, nota: str = "") -> dict[str, Any]:
    """Declara um novo estado (só transições previstas); o histórico é de acréscimo."""
    pasta = Path(pasta)
    registro = ler(pasta)
    if estado not in ESTADOS:
        raise ErroProveniencia(f"estado {estado!r} desconhecido; opções: {ESTADOS}")
    if estado == registro["estado"]:
        return registro
    if estado not in TRANSICOES[registro["estado"]]:
        raise ErroProveniencia(f"transição {registro['estado']} → {estado} não prevista")
    registro["estado"] = estado
    registro["historico_de_estados"].append({"estado": estado, "em": _agora(), "nota": nota})
    _gravar(pasta, registro)
    return registro


def derivar_estado(pasta: str | Path) -> dict[str, Any]:
    """O estado que os arquivos da execução evidenciam, sem consultar processo algum:
    `treino.json` concluído → `concluida`; `treino.json` incompleto ou checkpoint
    sem `treino.json` → `interrompida` (pode também estar em andamento: `ambiguo`);
    só `config.json` → `iniciada`; nada → `planejada`."""
    pasta = Path(pasta)
    treino = ler_json(pasta / TREINO) if (pasta / TREINO).is_file() else None
    if treino is not None:
        if treino.get("concluido"):
            return {"estado": "concluida", "evidencia": "treino.json com concluido=true", "ambiguo": False}
        return {"estado": "interrompida", "evidencia": "treino.json com concluido=false", "ambiguo": False}
    if (pasta / CHECKPOINTS / "ultimo.pt").is_file() or (pasta / CHECKPOINTS / "ultimo.json").is_file():
        return {"estado": "interrompida", "evidencia": "checkpoint sem treino.json (a sessão terminou sem gravá-lo, "
                "ou ainda está em andamento)", "ambiguo": True}
    if (pasta / CONFIG).is_file():
        return {"estado": "iniciada", "evidencia": "config.json sem checkpoint nem treino.json", "ambiguo": True}
    return {"estado": "planejada", "evidencia": "sem config.json", "ambiguo": False}


def resumo_de_metricas(pasta: str | Path) -> dict[str, Any]:
    """Resumo que aponta para os arquivos de métricas (os números ficam neles)."""
    pasta = Path(pasta)
    treino = ler_json(pasta / TREINO) if (pasta / TREINO).is_file() else None
    metricas = pasta / METRICAS
    return {
        "treino": None if treino is None else {
            "concluido": treino.get("concluido"), "melhor_epoca": treino.get("melhor_epoca"),
            "melhor_dev_macro_f1": treino.get("melhor_dev_macro_f1"),
            "epocas_no_historico": len(treino.get("dev_history") or []), "n_sessoes": len(treino.get("sessoes") or [])},
        "metricas_json": None if not metricas.is_file() else {
            "sha256": sha256_arquivo(metricas), "particoes": sorted(ler_json(metricas))}}


def registrar_metricas(pasta: str | Path) -> dict[str, Any]:
    pasta = Path(pasta)
    registro = ler(pasta)
    registro["metricas"] = resumo_de_metricas(pasta)
    _gravar(pasta, registro)
    return registro


def registrar_local_drive(pasta: str | Path, drive: Mapping[str, Any]) -> dict[str, Any]:
    """Registra a última sincronização com o Drive (`persistencia.sincronizar`)."""
    pasta = Path(pasta)
    registro = ler(pasta)
    registro["checkpoints"]["drive"] = dict(drive)
    _gravar(pasta, registro)
    return registro


def registrar_publicacao(pasta: str | Path, publicacao: Mapping[str, Any]) -> dict[str, Any]:
    """Acrescenta uma publicação (nunca substitui as anteriores)."""
    pasta = Path(pasta)
    registro = ler(pasta)
    registro["publicacoes"].append(dict(publicacao))
    if publicacao.get("destino") == "hf":
        registro["checkpoints"]["hf"] = {"repositorio": publicacao.get("repositorio"),
                                         "prefixo": publicacao.get("prefixo"), "commit": publicacao.get("commit")}
    _gravar(pasta, registro)
    return registro


def conferir(pasta: str | Path) -> list[str]:
    """Problemas de consistência do registro (vazio se confere)."""
    pasta = Path(pasta)
    try:
        registro = ler(pasta)
    except (ErroProveniencia, OSError, ValueError) as erro:
        return [str(erro)]
    problemas = []
    if registro.get("nome") != pasta.name:
        problemas.append(f"nome {registro.get('nome')!r} ≠ pasta {pasta.name!r}")
    try:
        if ids.config_id(registro["chave_configuracao"]) != registro.get("config_id"):
            problemas.append("config_id não confere com a chave de configuração gravada")
    except (ids.ErroId, KeyError) as erro:
        problemas.append(f"chave de configuração inválida: {erro}")
    if registro.get("sha256_config") and (pasta / CONFIG).is_file() \
            and sha256_arquivo(pasta / CONFIG) != registro["sha256_config"]:
        problemas.append("o config.json mudou depois de vinculado ao registro")
    if registro.get("estado") not in ESTADOS:
        problemas.append(f"estado desconhecido: {registro.get('estado')!r}")
    derivado = derivar_estado(pasta)
    if registro.get("estado") == "concluida" and derivado["estado"] != "concluida":
        problemas.append(f"declarada concluída, mas os arquivos evidenciam {derivado['estado']}: {derivado['evidencia']}")
    problemas += [f"segredo no registro: {m}" for m in segredos.contem_segredo(registro)]
    return problemas
