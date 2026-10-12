"""Identificação e planejamento das execuções, compartilhados por todas as estratégias.

Os notebooks não reimplementam nada disto: montam a chave de configuração,
perguntam ao plano o que fazer e (só então) criam a pasta da execução.

* `chave_da_execucao` — a chave determinística (`ids.chave_configuracao`) a
  partir de **objetos reais**: a configuração efetiva (`Config` ou o dicionário
  que ela grava), a configuração própria da estratégia, a proveniência dos
  pesos (`PROVENIENCIA.json` da pasta de pesos), a versão e o SHA-256 das
  partições (MANIFEST) e a identidade do código. Nada de caminho absoluto;
* `NOMES_OFICIAIS` — os nomes de pasta que o protocolo prevê, transcritos
  literalmente de `docs/protocolo/fase-biobertpt-seed42.md` (um teste confere
  que cada nome consta do protocolo). Não se cria nome oficial novo aqui: uma
  execução só recebe `alias_oficial` se o nome estiver nesta tabela, a
  estratégia e o encoder/semente do nome baterem com a configuração e a pasta
  ainda estiver livre; fora disso a pasta se chama como o `run_id`;
* `compativeis` / `planejar` — procuram, no disco local e no espelho do Drive,
  execuções **da mesma configuração** (mesmo `config_id`, recalculado da chave
  gravada, nunca confiando no nome da pasta) e decidem: `retomar`,
  `iniciar_planejada`, `ja_concluida` ou `nova_execucao` (próxima tentativa).
  A comparação é só pela chave; caminhos de sessão não participam.

`planejar` não escreve nada. `preparar` cria a pasta e o `experimento.json`
de uma execução nova (estado `planejada`); não treina.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from reclin.experimento import ids, persistencia, proveniencia

# nome da pasta oficial -> estratégia. Fonte: tabela de etapas do protocolo.
NOMES_OFICIAIS: dict[str, str] = {
    "baseline_biobertpt_seed42": "baseline",
    "baseline_calibrado_biobertpt_seed42": "baseline_calibrado",
    "restrito_biobertpt_seed42": "restrito",
    "pairaware_biobertpt_seed42": "pair_aware",
    "twostage_e1_biobertpt_seed42": "two_stage_e1",
    "twostage_e2_biobertpt_seed42": "two_stage_e2",
}
PADRAO_OFICIAL = re.compile(r"^[a-z0-9_]+?_(?P<encoder>[a-z]+)_seed(?P<seed>[0-9]+)$")
ESCOPO_CODIGO = "global-v1"
PARTICOES_DA_CHAVE = ("train", "dev")          # o que um treino usa; o TEST não entra na chave
ACOES = ("retomar", "iniciar_planejada", "ja_concluida", "nova_execucao")


class ErroPlano(ValueError):
    """Não foi possível montar a chave ou planejar a execução."""


# --------------------------------------------------------------------------- #
# Chave de configuração                                                       #
# --------------------------------------------------------------------------- #
def pesos_da_pasta(pasta_pesos: str | Path) -> dict[str, str]:
    """`{repositorio, commit, hash_manifesto}` dos pesos, conferidos contra o
    `PROVENIENCIA.json` da pasta. Sem manifesto, falha (peso sem revisão não
    tem identidade)."""
    from reclin import pretreino  # noqa: PLC0415

    registro = pretreino.registro_de(pasta_pesos)
    if not registro:
        raise ErroPlano(f"{Path(pasta_pesos).name}: sem PROVENIENCIA.json; baixe os pesos com "
                        "`scripts/pretreino.py baixar --revisao <commit>`")
    return dict(registro)


def dados_da_pasta(pasta_particoes: str | Path, particoes: tuple[str, ...] = PARTICOES_DA_CHAVE) -> dict[str, Any]:
    """`{versao, particoes: {nome: sha256}}` a partir do MANIFEST e da versão
    registrada. Só lê o MANIFEST (não abre o texto das partições)."""
    from reclin import particoes as part, versao_dados  # noqa: PLC0415

    versao = versao_dados.versao_da_pasta(pasta_particoes)
    if versao is None:
        raise ErroPlano(f"{Path(pasta_particoes).name}: o MANIFEST não é de uma versão registrada dos dados")
    splits = part.ler_manifesto(pasta_particoes).get("splits") or {}
    faltam = [n for n in particoes if n not in splits]
    if faltam:
        raise ErroPlano(f"o MANIFEST não traz as partições {faltam}")
    return {"versao": versao, "particoes": {n: splits[n]["sha256"] for n in particoes}}


def codigo_da_chave(codigo: str | Path) -> dict[str, str]:
    """A parte de código da chave: a identidade global v1 do `codigo/` indicado
    (a infraestrutura está fora dela; ver `identidade.py`)."""
    from reclin.util import identidade_codigo as ic  # noqa: PLC0415

    registro = ic.calcular(codigo)
    return {"escopo": ESCOPO_CODIGO, "algoritmo": registro["algoritmo"], "identidade": registro["identidade"]}


def chave_da_execucao(*, estrategia: str, config: Any, estrategia_config: Mapping[str, Any] | None,
                      pasta_pesos: str | Path | None = None, pesos: Mapping[str, Any] | None = None,
                      pasta_particoes: str | Path | None = None, dados: Mapping[str, Any] | None = None,
                      codigo: str | Path | None = None, identidade_codigo: Mapping[str, Any] | None = None
                      ) -> dict[str, Any]:
    """Monta a chave. Cada parte vem de uma pasta (lida aqui) ou é entregue já
    pronta (`pesos`, `dados`, `identidade_codigo`) — útil em planos hipotéticos
    e em testes. `config` é um `Config` (usa `como_dict()`) ou um dicionário."""
    efetiva = config.como_dict() if hasattr(config, "como_dict") else dict(config)
    if pesos is None:
        if pasta_pesos is None:
            raise ErroPlano("informe `pasta_pesos` ou `pesos`")
        pesos = pesos_da_pasta(pasta_pesos)
    if dados is None:
        if pasta_particoes is None:
            raise ErroPlano("informe `pasta_particoes` ou `dados`")
        dados = dados_da_pasta(pasta_particoes)
    if identidade_codigo is None:
        if codigo is None:
            raise ErroPlano("informe `codigo` (a pasta codigo/) ou `identidade_codigo`")
        identidade_codigo = codigo_da_chave(codigo)
    try:
        return ids.chave_configuracao(estrategia=estrategia, config=efetiva, estrategia_config=estrategia_config,
                                      pesos=pesos, dados=dados, codigo=identidade_codigo)
    except ids.ErroId as erro:
        raise ErroPlano(str(erro)) from erro


# --------------------------------------------------------------------------- #
# Nomes oficiais                                                              #
# --------------------------------------------------------------------------- #
def conferir_alias(alias: str, chave: Mapping[str, Any]) -> None:
    """Falha se `alias` não é um nome oficial do protocolo ou não corresponde à
    estratégia, ao encoder e à semente da configuração."""
    ids.validar_nome(alias)
    if alias not in NOMES_OFICIAIS:
        raise ErroPlano(f"{alias!r} não é um nome oficial do protocolo; execuções exploratórias usam o run_id")
    if NOMES_OFICIAIS[alias] != chave["estrategia"]:
        raise ErroPlano(f"{alias!r} é o nome oficial da estratégia {NOMES_OFICIAIS[alias]!r}, "
                        f"não de {chave['estrategia']!r}")
    m = PADRAO_OFICIAL.fullmatch(alias)
    config = chave.get("config") or {}
    if m and (config.get("encoder") != m["encoder"] or config.get("seed") != int(m["seed"])):
        raise ErroPlano(f"{alias!r} pressupõe encoder {m['encoder']!r} e semente {m['seed']}; a configuração tem "
                        f"{config.get('encoder')!r} e {config.get('seed')!r}")


# --------------------------------------------------------------------------- #
# Execuções compatíveis                                                       #
# --------------------------------------------------------------------------- #
def _raizes(raiz_local: str | Path, raiz_drive: str | Path | None) -> list[tuple[str, Path]]:
    raizes = [("local", Path(raiz_local))]
    if raiz_drive:
        raizes.append(("drive", persistencia.raiz_das_execucoes(raiz_drive)))
    return raizes


def _execucoes(raiz_local: str | Path, raiz_drive: str | Path | None) -> list[dict[str, Any]]:
    """Todas as execuções com `experimento.json` válido, em cada raiz. O
    `config_id` é recalculado da chave gravada: o campo e o nome da pasta
    não são confiados."""
    achadas = []
    for onde, raiz in _raizes(raiz_local, raiz_drive):
        if not raiz.is_dir():
            continue
        for pasta in sorted(p for p in raiz.iterdir() if p.is_dir() and not p.name.startswith(".")):
            try:
                registro = proveniencia.ler(pasta)
                cid = ids.config_id(registro["chave_configuracao"])
            except (proveniencia.ErroProveniencia, ids.ErroId, KeyError, ValueError, OSError):
                continue
            estado = registro.get("estado")
            if estado != "falhou":
                estado = proveniencia.derivar_estado(pasta)["estado"]
            achadas.append({"onde": onde, "nome": pasta.name, "run_id": registro.get("run_id"),
                            "config_id": cid, "tentativa": registro.get("tentativa"), "estado": estado,
                            "alias_oficial": registro.get("alias_oficial"),
                            "tem_ultimo": (pasta / proveniencia.CHECKPOINTS / "ultimo.pt").is_file(),
                            "coerente": cid == registro.get("config_id") and ids.decompor_run_id(
                                registro["run_id"])["config_id"] == cid})
    return achadas


def execucoes(raiz_local: str | Path, raiz_drive: str | Path | None = None) -> list[dict[str, Any]]:
    """Todas as execuções com `experimento.json` válido (local e Drive), cada uma
    com `onde`, `nome`, `run_id`, `config_id`, `tentativa`, `estado`, `alias_oficial`."""
    return _execucoes(raiz_local, raiz_drive)


def compativeis(chave: Mapping[str, Any], raiz_local: str | Path, raiz_drive: str | Path | None = None
                ) -> list[dict[str, Any]]:
    """As execuções (local e Drive) da mesma configuração, da tentativa mais
    recente para a mais antiga. Uma execução que existe nos dois lados aparece
    nos dois, cada uma com o seu `onde`."""
    cid = ids.config_id(chave)
    return sorted((e for e in _execucoes(raiz_local, raiz_drive) if e["config_id"] == cid),
                  key=lambda e: (e["tentativa"] or 0, e["onde"] == "local"), reverse=True)


# --------------------------------------------------------------------------- #
# Plano                                                                       #
# --------------------------------------------------------------------------- #
def planejar(chave: Mapping[str, Any], raiz_local: str | Path, raiz_drive: str | Path | None = None, *,
             alias_oficial: str | None = None, nova_tentativa: bool = False) -> dict[str, Any]:
    """Decide o que fazer com esta configuração, sem escrever nada.

    * há execução compatível `interrompida`/`iniciada` → `retomar` (a de maior
      tentativa; várias → aviso);
    * há `planejada` e nenhuma ativa → `iniciar_planejada`;
    * há `concluida` e nenhuma ativa → `ja_concluida` (não treina de novo);
    * só `falhou`, ou nada → `nova_execucao` com a próxima tentativa;
    * `nova_tentativa=True` força `nova_execucao` (o autor pediu recomeçar do zero).

    O nome oficial só é atribuído a uma execução nova se a pasta oficial estiver
    livre nas duas raízes; se não estiver, a pasta usa o `run_id` e o plano
    avisa. Uma pasta oficial ocupada por **outra configuração** é erro."""
    cid = ids.config_id(chave)
    estrategia = chave["estrategia"]
    todas = _execucoes(raiz_local, raiz_drive)
    mesmas = sorted((e for e in todas if e["config_id"] == cid),
                    key=lambda e: (e["tentativa"] or 0, e["onde"] == "local"), reverse=True)
    avisos: list[str] = [f"{e['nome']} ({e['onde']}): registro incoerente (config_id ou run_id)" for e in mesmas
                         if not e["coerente"]]
    if alias_oficial is not None:
        conferir_alias(alias_oficial, chave)
        for e in todas:
            if e["nome"] == alias_oficial and e["config_id"] != cid:
                raise ErroPlano(f"o nome oficial {alias_oficial!r} já pertence a outra configuração "
                                f"({e['run_id']}, {e['onde']}); nada é sobrescrito")
    por_estado = {estado: [e for e in mesmas if e["estado"] == estado] for estado in
                  ("iniciada", "interrompida", "planejada", "concluida", "falhou")}
    ativas = por_estado["interrompida"] + por_estado["iniciada"]
    alvo = None
    if nova_tentativa:
        acao = "nova_execucao"
    elif ativas:
        acao, alvo = "retomar", max(ativas, key=lambda e: (e["tentativa"] or 0, e["tem_ultimo"], e["onde"] == "local"))
        if len({e["run_id"] for e in ativas}) > 1:
            avisos.append("há mais de uma tentativa ativa desta configuração: o plano usa a mais recente "
                          f"({alvo['run_id']}); confira as outras")
    elif por_estado["planejada"]:
        acao, alvo = "iniciar_planejada", por_estado["planejada"][0]
    elif por_estado["concluida"]:
        acao, alvo = "ja_concluida", por_estado["concluida"][0]
    else:
        acao = "nova_execucao"
    if acao == "nova_execucao":
        tentativa = ids.proxima_tentativa([x for e in todas for x in (e["nome"], e["run_id"])],
                                          estrategia, cid)
        run = ids.run_id(estrategia, cid, tentativa)
        livre = alias_oficial is not None and not any(e["nome"] == alias_oficial for e in todas) \
            and not any(r.is_dir() and (r / alias_oficial).exists() for _, r in _raizes(raiz_local, raiz_drive))
        if alias_oficial is not None and not livre:
            avisos.append(f"o nome oficial {alias_oficial!r} já está em uso por outra tentativa desta configuração: "
                          "a nova tentativa usa o run_id como pasta")
        alias = alias_oficial if livre else None
        nome = alias or run
    else:
        run, nome, alias, tentativa = alvo["run_id"], alvo["nome"], alvo["alias_oficial"], alvo["tentativa"]
    return {"config_id": cid, "estrategia": estrategia, "acao": acao, "run_id": run, "tentativa": tentativa,
            "nome": nome, "alias_oficial": alias, "onde": (alvo or {}).get("onde"),
            "compativeis": mesmas, "avisos": avisos}


def preparar(plano: Mapping[str, Any], chave: Mapping[str, Any], raiz_local: str | Path, *,
             codigo: Mapping[str, Any], ambiente: Mapping[str, Any] | None = None, nota: str = "") -> Path:
    """Cria a pasta e o `experimento.json` (estado `planejada`) de uma execução
    **nova**. Para as demais ações devolve a pasta existente (local), sem
    escrever. Não treina, não baixa nada."""
    if plano["acao"] != "nova_execucao":
        return Path(raiz_local) / plano["nome"]
    pasta = Path(raiz_local) / plano["nome"]
    if pasta.exists() and any(pasta.iterdir()):
        raise ErroPlano(f"{pasta} já existe e não está vazia; nada é sobrescrito")
    proveniencia.criar(pasta, run=plano["run_id"], chave=chave, codigo=codigo, ambiente=ambiente,
                       alias_oficial=plano["alias_oficial"], nota=nota)
    return pasta
