"""Operações dos notebooks de estratégia (Colab), compartilhadas entre estratégias.

Cada notebook escolhe **uma** operação por vez (`OPERACOES`) e chama
`executar(operacao, contexto)`. Nada roda ao importar este módulo nem ao abrir
um notebook; uma operação só age quando é escolhida e executada. As operações
não reimplementam algoritmo: montam a identificação (`plano`), chamam os
scripts reais de `codigo/scripts/` (`raiz.rodar_script`) e registram a
proveniência e a persistência (`proveniencia`, `persistencia`).

Este módulo é neutro quanto à estratégia: o notebook entrega uma
`Especificacao` (nome da estratégia, script de treino e a classe de
configuração). Por isso ele não importa `reclin.estrategias`.

Garantias:

* a fase é só o BioBERTpt (`biobertpt`) e a semente 42: `encoder` e `seed` não
  são parâmetros editáveis; os demais campos da configuração só mudam na
  operação de nova execução, que exige um motivo e produz uma execução
  exploratória (nunca nome oficial, nunca elegível ao TEST);
* uma execução nova tem sempre pasta própria (`run_id` ou nome oficial livre);
  nada é sobrescrito;
* retomar só acontece se a chave de configuração da execução interrompida for
  igual à da configuração pretendida e o ambiente e o código forem compatíveis;
  senão a operação para e diz por quê;
* o TEST nunca é avaliado aqui: `preparar_test` só confere condições e imprime
  o que o notebook do TEST executará, e só dá a autorização como conferida
  quando o texto exato é informado; nenhuma operação lê números do TEST;
* a publicação só envia com a frase de confirmação exata, depois de mostrar o
  destino.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from reclin.experimento import ambiente, ids, persistencia, plano, proveniencia, publicacao_hf, raiz, segredos
from reclin.util import identidade_codigo as ic
from reclin.util.io import ler_json

ENCODER_DA_FASE = "biobertpt"
SEMENTE_DA_FASE = 42
INTERROMPIDO = 3                       # código de saída de `treinar.py`: interrompido, há checkpoint
CAMPOS_FIXOS = ("encoder", "seed")     # fixados pela fase; nunca editáveis
PREFIXO_AUTORIZACAO_TEST = "AUTORIZO O TEST DE"
SUFIXO_AUTORIZACAO_TEST = "COM O DEV CONGELADO"

OP_AMBIENTE = "1. Conferir ambiente e dependências"
OP_GPU = "2. Conferir GPU"
OP_LISTAR = "3. Identificar execuções anteriores"
OP_NOVO = "4. Novo treinamento (configuração do protocolo)"
OP_RETOMAR = "5. Retomar execução interrompida"
OP_NOVA = "6. Nova execução (outra tentativa ou outros parâmetros)"
OP_DEV = "7. Métricas do DEV"
OP_ESTADO = "8. Estado e logs da execução"
OP_TEST = "9. Preparar avaliação do TEST (só conferência; não avalia)"
OP_PUBLICACAO = "10. Preparar publicação (mostra o destino; não envia)"
OP_PESOS = "11. Obter os pesos do BioBERTpt (acessa o Hugging Face Hub)"
OP_SINCRONIZAR = "12. Sincronizar a execução com o Drive"
OP_RESTAURAR = "13. Restaurar a execução do Drive"
OP_INSTALAR = "14. Instalar dependências que faltam (pip)"
OP_PUBLICAR = "15. Publicar no Hugging Face (envia)"
OP_COMPOR = "16. Compor os estágios e escolher τ no DEV (Two-Stage; não lê o TEST)"
SEM_ESTRATEGIA = (OP_AMBIENTE, OP_GPU, OP_INSTALAR)       # não precisam da classe de configuração
OPERACOES = (OP_AMBIENTE, OP_GPU, OP_LISTAR, OP_NOVO, OP_RETOMAR, OP_NOVA, OP_DEV, OP_ESTADO, OP_TEST,
             OP_PUBLICACAO, OP_PESOS, OP_SINCRONIZAR, OP_RESTAURAR, OP_INSTALAR, OP_PUBLICAR, OP_COMPOR)


class ErroOperacao(RuntimeError):
    """A operação foi recusada ou falhou; a mensagem diz por quê e o que fazer."""


# --------------------------------------------------------------------------- #
# Especificação e contexto                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Especificacao:
    """O que muda de uma estratégia para outra: o nome, o script de treino e a
    classe de configuração (`reclin.config.Config` e suas subclasses). `estagio` só existe nas estratégias
    de estágios (Two-Stage): o script é então chamado como `<script> treinar --estagio N ...`, sem `--estrategia`."""
    estrategia: str
    script: str
    config_cls: type
    estrategia_config: Mapping[str, Any] | None = None
    estagio: int | None = None

    def config_protocolo(self) -> Any:
        """A configuração do protocolo: os padrões da classe, com o encoder e a semente da fase."""
        return self.config_cls(encoder=ENCODER_DA_FASE, seed=SEMENTE_DA_FASE)

    @property
    def campos(self) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(self.config_cls))

    @property
    def permitidos(self) -> tuple[str, ...]:
        """Os campos que a nova execução pode alterar: todos os que o script de treino expõe
        (um por campo da configuração), exceto `encoder` e `seed`."""
        return tuple(c for c in self.campos if c not in CAMPOS_FIXOS)

    @property
    def alias_oficial(self) -> str | None:
        achados = [n for n, e in plano.NOMES_OFICIAIS.items() if e == self.estrategia]
        return achados[0] if len(achados) == 1 else None


@dataclass
class Contexto:
    """Tudo o que as operações usam, vindo da célula de configuração do notebook."""
    codigo: Path
    raiz_local: Path
    raiz_drive: Path | None
    espec: Especificacao | None
    pasta_pesos: Path
    pasta_particoes: Path
    revisao_pesos: str = ""
    repositorio_pesos: str = ""
    parametros: Mapping[str, Any] = field(default_factory=dict)
    motivo: str = ""
    execucao: str = ""
    checkpoint_a_cada: int = 500
    espelhar_a_cada_min: float = 10.0
    linhas_do_log: int = 60
    repositorio_hf: str = ""
    confirmacao: str = ""
    autorizacao_test: str = ""
    modo_ambiente: str = "minimo"
    estagio1: str = ""                 # Two-Stage: nome (pasta) do estágio 1 concluído, para a composição
    estagio2: str = ""                 # Two-Stage: nome (pasta) do estágio 2 concluído, para a composição
    prefixo_hf: str = ""               # pasta no repositório do HF (vazio: runs/<nome>; estágio: etapa1/etapa2)


# --------------------------------------------------------------------------- #
# Utilidades                                                                  #
# --------------------------------------------------------------------------- #
def _imprimir(objeto: Any) -> None:
    print(segredos.limpar(json.dumps(objeto, ensure_ascii=False, indent=2, sort_keys=True, default=str)))


def _titulo(texto: str) -> None:
    print(f"\n=== {texto} ===")


def _agora() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _tabela(linhas: list[Mapping[str, Any]], colunas: list[str]) -> None:
    if not linhas:
        print("(nada a mostrar)")
        return
    texto = [[("" if linha.get(c) is None else str(linha.get(c))) for c in colunas] for linha in linhas]
    larguras = [max(len(c), *(len(t[i]) for t in texto)) for i, c in enumerate(colunas)]
    print("  ".join(c.ljust(w) for c, w in zip(colunas, larguras)))
    print("  ".join("-" * w for w in larguras))
    for t in texto:
        print("  ".join(v.ljust(w) for v, w in zip(t, larguras)))


def _versao_da_fase() -> str:
    from reclin import versao_dados  # noqa: PLC0415

    return versao_dados.VERSAO_DA_FASE


def _exigir_drive(ctx: Contexto) -> Path:
    if ctx.raiz_drive is None:
        raise ErroOperacao("o Drive não está montado: monte-o na preparação (célula 2) antes de treinar. "
                           "Sem ele os checkpoints ficariam só no disco temporário do Colab.")
    return ctx.raiz_drive


def _raiz_de(ctx: Contexto, onde: str) -> Path:
    return ctx.raiz_local if onde == "local" else persistencia.raiz_das_execucoes(_exigir_drive(ctx))


# --------------------------------------------------------------------------- #
# Configuração e chave                                                        #
# --------------------------------------------------------------------------- #
def config_da_operacao(espec: Especificacao, parametros: Mapping[str, Any]) -> Any:
    """A configuração efetiva: a do protocolo, com os `parametros` aplicados.
    Recusa campo que não seja um parâmetro permitido (inclusive `encoder` e `seed`)."""
    parametros = dict(parametros or {})
    for chave in parametros:
        if chave in CAMPOS_FIXOS:
            raise ErroOperacao(f"{chave!r} é fixado pela fase ({ENCODER_DA_FASE}, semente {SEMENTE_DA_FASE}) e não "
                               "pode ser alterado")
        if chave not in espec.permitidos:
            raise ErroOperacao(f"parâmetro desconhecido: {chave!r}. Permitidos: {list(espec.permitidos)}")
    try:
        return dataclasses.replace(espec.config_protocolo(), **parametros)
    except (ValueError, TypeError) as erro:
        raise ErroOperacao(f"configuração inválida: {erro}") from erro


def _chave(ctx: Contexto, config: Any) -> dict[str, Any]:
    if not ctx.pasta_pesos.is_dir():
        raise ErroOperacao(f"os pesos não estão em {ctx.pasta_pesos}: use a operação '{OP_PESOS}'")
    try:
        chave = plano.chave_da_execucao(estrategia=ctx.espec.estrategia, config=config,
                                        estrategia_config=ctx.espec.estrategia_config,
                                        pasta_pesos=ctx.pasta_pesos, pasta_particoes=ctx.pasta_particoes,
                                        codigo=ctx.codigo)
    except (plano.ErroPlano, ValueError, OSError) as erro:
        raise ErroOperacao(str(erro)) from erro
    if chave["dados"]["versao"] != _versao_da_fase():
        raise ErroOperacao(f"as partições são da versão {chave['dados']['versao']!r}; a fase usa {_versao_da_fase()!r}")
    return chave


def _achatar(valor: Any, prefixo: str = "") -> dict[str, Any]:
    if isinstance(valor, Mapping):
        saida: dict[str, Any] = {}
        for k, v in valor.items():
            saida.update(_achatar(v, f"{prefixo}.{k}" if prefixo else str(k)))
        return saida
    return {prefixo: valor}


def diferencas_de_chave(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Campos em que duas chaves de configuração diferem (`campo: a → b`)."""
    fa, fb = _achatar(a), _achatar(b)
    return [f"{k}: {fa.get(k)!r} → {fb.get(k)!r}" for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]


def _explicar_incompatibilidade(ctx: Contexto, chave: Mapping[str, Any]) -> list[str]:
    """Para cada execução ativa (interrompida/iniciada) da estratégia, o que a torna incompatível."""
    linhas = []
    for e in plano.execucoes(ctx.raiz_local, ctx.raiz_drive):
        if e["estado"] not in ("interrompida", "iniciada") or not e["run_id"] \
                or ids_estrategia(e["run_id"]) != ctx.espec.estrategia:
            continue
        gravada = proveniencia.ler(_raiz_de(ctx, e["onde"]) / e["nome"])["chave_configuracao"]
        linhas.append(f"{e['nome']} ({e['onde']}, {e['estado']}): " + ("; ".join(diferencas_de_chave(gravada, chave))
                                                                    or "sem diferença na chave"))
    return linhas


def ids_estrategia(run_id: str) -> str:
    return ids.decompor_run_id(run_id)["estrategia"]


def argumentos_de_treino(ctx: Contexto, config: Any, nome: str, *, retomar: bool) -> list[str]:
    """Os argumentos reais de `scripts/treinar.py`: todos os campos da configuração, explícitos
    (o script cria uma opção por campo), mais os de caminho e de checkpoint. Um campo `None` cujo padrão
    também é `None` (opcional, como os da cabeça da Pair-Aware) fica de fora: o script o assume `None`."""
    if ctx.espec.estagio is None:
        argumentos = ["--estrategia", ctx.espec.estrategia]
    else:                                  # `two_stage.py treinar --estagio N` (o subcomando vem primeiro)
        argumentos = ["treinar", "--estagio", str(int(ctx.espec.estagio))]
    argumentos += ["--nome", nome, "--saida", str(ctx.raiz_local),
                   "--particoes", str(ctx.pasta_particoes), "--modelo", str(ctx.pasta_pesos)]
    padrao = ctx.espec.config_cls()
    for campo in ctx.espec.campos:
        valor = getattr(config, campo)
        if valor is None and getattr(padrao, campo) is None:
            continue                       # campo opcional no padrão (ex.: `mlp_hidden`): o script já assume None
        argumentos += ["--" + campo.replace("_", "-"), str(valor)]
    argumentos += ["--checkpoint-a-cada", str(int(ctx.checkpoint_a_cada))]
    if retomar:
        argumentos.append("--retomar")
    return argumentos


# --------------------------------------------------------------------------- #
# 1, 2 e 14 — ambiente, GPU, instalação                                       #
# --------------------------------------------------------------------------- #
def conferir_ambiente(ctx: Contexto) -> dict[str, Any]:
    relatorio = ambiente.verificar(ctx.codigo, modo=ctx.modo_ambiente)
    _titulo("Ambiente")
    atual = relatorio["ambiente"]
    print(f"Python {atual['python']} | {atual['plataforma']} | no Colab: {atual['colab']}")
    _tabela([{"pacote": p, "versão": v} for p, v in sorted(atual["versoes"].items())], ["pacote", "versão"])
    for aviso in relatorio["avisos"]:
        print("AVISO:", aviso)
    for erro in relatorio["erros"]:
        print("ERRO:", erro)
    plano_instalacao = relatorio["plano"]
    if plano_instalacao["instalar"]:
        print("\nPlano de instalação (nada foi instalado):")
        for linha in plano_instalacao["motivos"]:
            print(" -", linha)
        print(f"Use a operação '{OP_INSTALAR}' se concordar.")
    print("\nResultado:", "ok" if relatorio["ok"] else "há pendências")
    return relatorio


def instalar_dependencias(ctx: Contexto) -> dict[str, Any]:
    relatorio = ambiente.verificar(ctx.codigo, modo=ctx.modo_ambiente)
    if not relatorio["plano"]["instalar"]:
        print("Nada a instalar: as dependências atendem ao mínimo do projeto.")
        return relatorio
    _titulo("Instalando")
    resultado = ambiente.instalar(relatorio["plano"], executar=True)
    _imprimir(resultado)
    depois = ambiente.verificar(ctx.codigo, modo=ctx.modo_ambiente)
    if relatorio["plano"]["reiniciar"]:
        print("\nATENÇÃO: reinicie o ambiente de execução (Ambiente de execução → Reiniciar sessão) e rode de novo "
              "a preparação antes de treinar.")
    print("Resultado:", "ok" if depois["ok"] else "ainda há pendências: " + "; ".join(depois["erros"]))
    return depois


def conferir_gpu(ctx: Contexto) -> dict[str, Any]:
    atual = ambiente.coletar()
    gpu = ambiente.classificar_gpu(atual.get("gpu"))
    _titulo("GPU")
    print("GPU:", atual.get("gpu") or "nenhuma", "| CUDA:", atual.get("cuda"), "| cuDNN:", atual.get("cudnn"))
    print("Classe:", gpu["classe"])
    if gpu["aviso"]:
        print("AVISO:", gpu["aviso"])
    if gpu["classe"] == "sem_gpu":
        print("ERRO: sem GPU, o treinamento não deve ser iniciado (Ambiente de execução → Alterar tipo → GPU).")
    return {"gpu": atual.get("gpu"), "classe": gpu["classe"], "aviso": gpu["aviso"]}


def _exigir_gpu(ctx: Contexto) -> None:
    relatorio = ambiente.verificar(ctx.codigo, modo=ctx.modo_ambiente, exigir_gpu=True,
                                   ambiente_atual=ambiente.coletar())
    if not relatorio["ok"]:
        raise ErroOperacao("ambiente não está pronto para treinar:\n - " + "\n - ".join(relatorio["erros"]))
    for aviso in relatorio["avisos"]:
        print("AVISO:", aviso)


# --------------------------------------------------------------------------- #
# 11 — pesos                                                                  #
# --------------------------------------------------------------------------- #
def obter_pesos(ctx: Contexto) -> dict[str, Any]:
    """Garante os pesos na revisão exata, com `PROVENIENCIA.json` conferido: reaproveita a pasta local,
    depois a cópia do Drive, e só então baixa (`scripts/pretreino.py baixar`, o único comando que acessa o Hub)."""
    from reclin import pretreino  # noqa: PLC0415

    if not ctx.revisao_pesos:
        raise ErroOperacao("informe REVISAO_PESOS (SHA completo, 40 caracteres) na célula de configuração; "
                           "ela é a revisão congelada da fase e é comunicada ao autor antes do primeiro treino")
    try:
        pretreino.validar_commit(ctx.revisao_pesos)
    except pretreino.ErroProveniencia as erro:
        raise ErroOperacao(str(erro)) from erro
    repositorio = ctx.repositorio_pesos or pretreino.REPOSITORIO_BIOBERTPT
    destino = ctx.pasta_pesos
    cache = (_exigir_drive(ctx) / "pesos" / destino.name) if ctx.raiz_drive else None
    origem = "local"

    def conferido(pasta: Path) -> dict[str, str] | None:
        if not pasta.is_dir() or not pretreino.tem_manifesto(pasta):
            return None
        try:
            return pretreino.conferir(pasta)
        except pretreino.ErroProveniencia:
            return None

    registro = conferido(destino)
    if registro is None and cache is not None and conferido(cache) is not None:
        origem = "drive"
        print(f"Copiando os pesos do Drive ({cache}) …")
        if destino.exists() and any(destino.iterdir()):
            raise ErroOperacao(f"{destino} existe, não confere com o manifesto e não está vazio: remova-o antes")
        tmp = destino.with_name(destino.name + ".copiando")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(cache, tmp)
        if destino.exists():
            destino.rmdir()
        tmp.rename(destino)
        registro = conferido(destino)
    if registro is None:
        origem = "hub"
        print(f"Baixando {repositorio}@{ctx.revisao_pesos} (acessa o Hugging Face Hub) …")
        raiz.rodar_script(ctx.codigo, "pretreino.py", "baixar", "--repo", repositorio, "--revisao", ctx.revisao_pesos,
                          "--destino", str(destino))
        registro = conferido(destino)
    if registro is None:
        raise ErroOperacao(f"os pesos em {destino} não conferem com o manifesto")
    if registro["commit"] != ctx.revisao_pesos:
        raise ErroOperacao(f"os pesos em {destino} são do commit {registro['commit']}, não de {ctx.revisao_pesos}")
    if cache is not None and conferido(cache) is None:
        print(f"Guardando uma cópia no Drive ({cache}) …")
        tmp = cache.with_name(cache.name + ".copiando")
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(destino, tmp)
        if cache.exists():
            shutil.rmtree(cache)
        tmp.rename(cache)
        if conferido(cache) is None:
            raise ErroOperacao(f"a cópia dos pesos no Drive não confere: {cache}")
    resultado = {**registro, "pasta": str(destino), "origem": origem, "copia_no_drive": str(cache) if cache else None}
    _imprimir(resultado)
    return resultado


# --------------------------------------------------------------------------- #
# Execuções: resolver e listar                                                #
# --------------------------------------------------------------------------- #
def _resolver_execucao(ctx: Contexto) -> tuple[str, Path, str]:
    """(nome, pasta, onde) da execução indicada em `ctx.execucao` (nome ou run_id), ou a única existente."""
    todas = [e for e in plano.execucoes(ctx.raiz_local, ctx.raiz_drive) if e["run_id"]
             and ids_estrategia(e["run_id"]) == ctx.espec.estrategia]
    nomes = sorted({e["nome"] for e in todas})
    if ctx.execucao:
        achadas = [e for e in todas if ctx.execucao in (e["nome"], e["run_id"])]
        if not achadas:
            raise ErroOperacao(f"execução {ctx.execucao!r} não encontrada. Existentes: {nomes or 'nenhuma'}")
    elif len(nomes) == 1:
        achadas = todas
    else:
        raise ErroOperacao("informe EXECUCAO (nome ou run_id). Existentes: " + (", ".join(nomes) or "nenhuma"))
    escolhida = sorted(achadas, key=lambda e: e["onde"] != "local")[0]
    return escolhida["nome"], _raiz_de(ctx, escolhida["onde"]) / escolhida["nome"], escolhida["onde"]


def listar_execucoes(ctx: Contexto) -> list[dict[str, Any]]:
    _titulo("Execuções anteriores")
    linhas = persistencia.listar(ctx.raiz_local, ctx.raiz_drive)
    linhas = [linha for linha in linhas if (linha["run_id"] or "").startswith(ctx.espec.estrategia + "__")
              or linha["alias_oficial"] or not linha["run_id"]]
    quadro = [{"nome": l["nome"], "run_id": l["run_id"], "oficial": l["alias_oficial"],
               "local": (l["local"] or {}).get("estado"), "drive": (l["drive"] or {}).get("estado"),
               "passo": max((l[x] or {}).get("passo_global", 0) for x in ("local", "drive")
                            if l[x]) if (l["local"] or l["drive"]) else None,
               "situação": l["situacao"], "retomável": l["retomavel"]} for l in linhas]
    _tabela(quadro, ["nome", "run_id", "oficial", "local", "drive", "passo", "situação", "retomável"])
    try:
        chave = _chave(ctx, config_da_operacao(ctx.espec, ctx.parametros))
        compat = plano.compativeis(chave, ctx.raiz_local, ctx.raiz_drive)
        print(f"\nconfig_id da configuração atual: {ids.config_id(chave)}")
        print("Compatíveis com ela:", [f"{e['nome']} ({e['onde']}, {e['estado']})" for e in compat] or "nenhuma")
    except ErroOperacao as erro:
        print("\n(configuração atual não identificada:", erro, ")")
    return linhas


# --------------------------------------------------------------------------- #
# 4, 5, 6 — treinar                                                           #
# --------------------------------------------------------------------------- #
def _fechar(ctx: Contexto, pasta: Path, codigo_de_saida: int) -> dict[str, Any]:
    """Registra o estado final da sessão e espelha no Drive (também se o treino foi interrompido ou falhou)."""
    avisos: list[str] = []
    if (pasta / "config.json").is_file():
        try:
            proveniencia.vincular_config(pasta)
        except proveniencia.ErroProveniencia as erro:
            avisos.append(f"vínculo com o config.json: {erro}")
    derivado = proveniencia.derivar_estado(pasta)["estado"]
    tem_checkpoint = (pasta / proveniencia.CHECKPOINTS / "ultimo.pt").is_file()
    if codigo_de_saida == 0 and derivado == "concluida":
        proveniencia.mudar_estado(pasta, "concluida", "treino concluído")
        try:
            proveniencia.registrar_metricas(pasta)
        except (proveniencia.ErroProveniencia, OSError, ValueError) as erro:
            avisos.append(f"métricas não registradas: {erro}")
    elif codigo_de_saida == INTERROMPIDO or tem_checkpoint:
        proveniencia.mudar_estado(pasta, "interrompida", f"saída {codigo_de_saida}; há checkpoint para retomar")
    else:
        proveniencia.mudar_estado(pasta, "falhou", f"saída {codigo_de_saida}; sem checkpoint")
    espelho = None
    try:
        if (pasta / "config.json").is_file():
            espelho = persistencia.sincronizar_e_registrar(pasta, _exigir_drive(ctx), incluir_ultimo=True)
    except (persistencia.ErroSincronizacao, OSError, ErroOperacao) as erro:
        avisos.append(f"ESPELHO NO DRIVE FALHOU: {erro}")
    registro = proveniencia.ler(pasta)
    return {"nome": pasta.name, "run_id": registro["run_id"], "estado": registro["estado"],
            "codigo_de_saida": codigo_de_saida, "espelho_no_drive": espelho, "avisos": avisos}


def _executar_treino(ctx: Contexto, pasta: Path, config: Any, *, retomar: bool) -> dict[str, Any]:
    proveniencia.mudar_estado(pasta, "iniciada", "retomada" if retomar else "início do treino")
    argumentos = argumentos_de_treino(ctx, config, pasta.name, retomar=retomar)
    drive = _exigir_drive(ctx)

    def espelhar() -> None:
        if (pasta / "config.json").is_file():
            persistencia.sincronizar_e_registrar(pasta, drive, incluir_ultimo=True)
            print(f"[espelho] {pasta.name} sincronizado com o Drive")

    _titulo(("Retomando " if retomar else "Treinando ") + pasta.name)
    print("scripts/" + ctx.espec.script, " ".join(argumentos))
    saida = 1
    try:
        saida = raiz.rodar_script(ctx.codigo, ctx.espec.script, *argumentos, verificar=False,
                                  log=pasta / "logs" / f"treino-{_agora()}.log",
                                  periodico=(max(60.0, ctx.espelhar_a_cada_min * 60.0), espelhar))
    finally:
        resultado = _fechar(ctx, pasta, saida)
    _titulo("Resultado")
    _imprimir(resultado)
    if resultado["estado"] == "interrompida":
        print(f"\nTreino interrompido: use a operação '{OP_RETOMAR}' (mesma configuração).")
    return resultado


def compor_dev(ctx: Contexto) -> dict[str, Any]:
    """Two-Stage: com os dois estágios concluídos (`ctx.estagio1`, `ctx.estagio2`), roda
    `two_stage.py compor-dev`, que escolhe τ no DEV pelo fluxo real e o congela em `limiar.json`, e espelha a
    composição no Drive. **Não lê o TEST** (nem existe aqui comando que o leia)."""
    drive = _exigir_drive(ctx)
    pastas = {}
    for numero, nome in ((1, ctx.estagio1), (2, ctx.estagio2)):
        if not nome:
            raise ErroOperacao(f"informe o nome do estágio {numero} concluído (ESTAGIO{numero})")
        pasta = ctx.raiz_local / nome
        if not (pasta / proveniencia.ARQUIVO).is_file():
            raise ErroOperacao(f"o estágio {numero} ({nome}) não está no disco local: restaure-o do Drive antes")
        registro = proveniencia.ler(pasta)
        esperado = f"two_stage_e{numero}"
        if registro["estrategia"] != esperado:
            raise ErroOperacao(f"{nome} é da estratégia {registro['estrategia']!r}, não de {esperado!r}")
        derivado = proveniencia.derivar_estado(pasta)["estado"]
        if derivado != "concluida":
            raise ErroOperacao(f"o estágio {numero} ({nome}) não está concluído (os arquivos evidenciam {derivado!r})")
        pastas[numero] = pasta
    nome_composicao = f"twostage_{ENCODER_DA_FASE}_seed{SEMENTE_DA_FASE}"
    destino = ctx.raiz_local / nome_composicao
    if (destino / "config.json").exists():
        raise ErroOperacao(f"a composição {nome_composicao} já existe: o τ dela já foi escolhido e congelado; "
                           "nada é sobrescrito")
    _exigir_gpu(ctx)
    argumentos = ["compor-dev", "--estagio1", str(pastas[1]), "--estagio2", str(pastas[2]), "--nome", nome_composicao,
                  "--saida", str(ctx.raiz_local), "--particoes", str(ctx.pasta_particoes)]
    _titulo("Compondo os estágios no DEV")
    print("scripts/two_stage.py", " ".join(argumentos))
    saida = raiz.rodar_script(ctx.codigo, "two_stage.py", *argumentos, verificar=False,
                              log=ctx.raiz_local.parent / "logs-composicao" / f"{nome_composicao}-{_agora()}.log")
    if saida != 0:
        raise ErroOperacao(f"a composição terminou com código {saida}; veja a saída acima. Nada foi espelhado.")
    espelho = persistencia.sincronizar(destino, drive, incluir_ultimo=False)
    resultado: dict[str, Any] = {"nome": nome_composicao, "estagio1": ctx.estagio1, "estagio2": ctx.estagio2,
                                 "espelho_no_drive": {k: v for k, v in espelho.items() if k != "manifesto_sha256"}}
    limiar = ler_json(destino / "limiar.json")
    resultado["tau"] = limiar.get("limiar")
    dev = (ler_json(destino / "metricas.json").get("dev") or {})
    resultado["dev"] = {k: dev[k] for k in ("macro_f1", "micro_f1", "weighted_f1", "mcc", "f1_per_class") if k in dev}
    _titulo("Resultado (DEV, fluxo real)")
    _imprimir(resultado)
    print("\nτ ficou congelado em limiar.json. O TEST não foi lido; a avaliação dele é etapa separada, uma única vez.")
    return resultado


def _preparar_treino(ctx: Contexto, modo: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    espec = ctx.espec
    _exigir_drive(ctx)
    config = config_da_operacao(espec, ctx.parametros)
    protocolo = config == espec.config_protocolo()
    if modo == "novo" and not protocolo:
        raise ErroOperacao(f"PARAMETROS não está vazio/igual ao protocolo, e '{OP_NOVO}' usa a configuração do "
                           f"protocolo. Esvazie PARAMETROS ou use '{OP_NOVA}' (execução exploratória).")
    if modo == "nova" and not ctx.motivo.strip():
        raise ErroOperacao("MOTIVO é obrigatório numa nova execução: ele fica registrado no histórico da execução")
    _exigir_gpu(ctx)
    chave = _chave(ctx, config)
    alias = espec.alias_oficial if (modo == "novo" and protocolo) else None
    try:
        p = plano.planejar(chave, ctx.raiz_local, ctx.raiz_drive, alias_oficial=alias, nova_tentativa=(modo == "nova"))
    except (plano.ErroPlano, ValueError) as erro:
        raise ErroOperacao(str(erro)) from erro
    _titulo("Configuração efetiva")
    _imprimir({k: v for k, v in config.como_dict().items() if k != "modelo"})
    _titulo("Plano")
    _imprimir({"natureza": "protocolo" if protocolo else "exploratória", "config_id": p["config_id"],
               "acao": p["acao"], "run_id": p["run_id"], "pasta": p["nome"], "nome_oficial": p["alias_oficial"],
               "tentativa": p["tentativa"], "avisos": p["avisos"]})
    return config, chave, p


def treinar_novo(ctx: Contexto) -> dict[str, Any]:
    config, chave, p = _preparar_treino(ctx, "novo")
    if p["acao"] == "retomar":
        raise ErroOperacao(f"já existe a execução {p['nome']} desta configuração, interrompida: use '{OP_RETOMAR}'")
    if p["acao"] == "ja_concluida":
        raise ErroOperacao(f"a execução {p['nome']} desta configuração já está concluída; nada é sobrescrito. "
                           f"Para outra tentativa, use '{OP_NOVA}'")
    pasta = plano.preparar(p, chave, ctx.raiz_local, codigo=proveniencia.codigo_do_ambiente(ctx.codigo),
                           ambiente=ambiente.coletar(), nota="natureza=protocolo")
    return _executar_treino(ctx, pasta, config, retomar=False)


def treinar_nova_execucao(ctx: Contexto) -> dict[str, Any]:
    config, chave, p = _preparar_treino(ctx, "nova")
    protocolo = config == ctx.espec.config_protocolo()
    nota = f"natureza={'protocolo (nova tentativa)' if protocolo else 'exploratoria'}; motivo={ctx.motivo.strip()}"
    pasta = plano.preparar(p, chave, ctx.raiz_local, codigo=proveniencia.codigo_do_ambiente(ctx.codigo),
                           ambiente=ambiente.coletar(), nota=nota)
    if not protocolo:
        print("\nExecução EXPLORATÓRIA: não recebe nome oficial, não é elegível à avaliação do TEST nem à fase científica.")
    return _executar_treino(ctx, pasta, config, retomar=False)


def retomar_treino(ctx: Contexto) -> dict[str, Any]:
    config, chave, p = _preparar_treino(ctx, "retomar")
    if p["acao"] != "retomar":
        causas = _explicar_incompatibilidade(ctx, chave)
        raise ErroOperacao(
            "não há execução interrompida COMPATÍVEL com esta configuração; nada foi retomado nem reutilizado.\n"
            + ("Execuções interrompidas e o que difere da configuração pretendida:\n - " + "\n - ".join(causas)
               if causas else "Nenhuma execução interrompida desta estratégia foi encontrada.")
            + f"\nCorrija PARAMETROS para a configuração da execução, ou use '{OP_NOVA}'.")
    if ctx.execucao and ctx.execucao not in (p["nome"], p["run_id"]):
        raise ErroOperacao(f"a execução pedida ({ctx.execucao}) não é a compatível com esta configuração "
                           f"({p['nome']}); confira EXECUCAO e PARAMETROS")
    nome = p["nome"]
    diagnostico = persistencia.diagnosticar_retomada(nome, ctx.raiz_local, ctx.raiz_drive,
                                                     identidade_atual=ic.calcular(ctx.codigo)["identidade"])
    _titulo("Diagnóstico da retomada")
    _imprimir({k: diagnostico[k] for k in ("fonte", "pode_retomar", "bloqueios", "avisos")})
    if not diagnostico["pode_retomar"]:
        raise ErroOperacao("a retomada não é segura:\n - " + "\n - ".join(diagnostico["bloqueios"])
                           + "\nNenhum peso foi reutilizado.")
    if diagnostico["fonte"] == "drive":
        try:
            persistencia.restaurar(_exigir_drive(ctx), nome, ctx.raiz_local, progresso=print)
        except persistencia.ErroSincronizacao as erro:
            raise ErroOperacao(f"não foi possível restaurar do Drive: {erro}") from erro
    return _executar_treino(ctx, ctx.raiz_local / nome, config, retomar=True)


# --------------------------------------------------------------------------- #
# 7 e 8 — métricas do DEV, estado e logs                                      #
# --------------------------------------------------------------------------- #
def metricas_dev(ctx: Contexto) -> dict[str, Any]:
    """Histórico por época e métricas do DEV. Só o DEV: nenhuma chave ou arquivo do TEST é lido."""
    nome, pasta, onde = _resolver_execucao(ctx)
    _titulo(f"Métricas do DEV — {nome} ({onde})")
    resultado: dict[str, Any] = {"nome": nome}
    if (pasta / "treino.json").is_file():
        treino = ler_json(pasta / "treino.json")
        historico = treino.get("dev_history") or []
        if ctx.espec.estagio is None:
            _tabela(historico, ["epoch", "train_loss", "dev_loss", "dev_macro_f1", "dev_negation_of_f1", "duration_s"])
            print(f"\nconcluído: {treino.get('concluido')} | posição: {treino.get('posicao')} | "
                  f"melhor época (DEV, macro-F1): {treino.get('melhor_epoca')} ({treino.get('melhor_dev_macro_f1')})")
        else:                              # estágio: as métricas do histórico são as do próprio estágio
            colunas = ["epoch", "train_loss", "dev_loss"] + sorted(
                {k for h in historico for k, v in h.items() if isinstance(v, (int, float))}
                - {"epoch", "train_loss", "dev_loss"})
            _tabela(historico, colunas)
            print(f"\nconcluído: {treino.get('concluido')} | posição: {treino.get('posicao')} | melhor época (DEV, "
                  f"{treino.get('campo_selecao')}): {treino.get('melhor_epoca')} ({treino.get('melhor_valor_selecao')})")
        resultado.update({"historico": historico, "melhor_epoca": treino.get("melhor_epoca"),
                          "concluido": treino.get("concluido")})
    else:
        print("treino.json ainda não existe (o treino não terminou uma sessão); o histórico por época aparece no log.")
    if (pasta / "metricas.json").is_file():
        dev = ler_json(pasta / "metricas.json").get("dev")
        if dev:
            chaves = [k for k in ("macro_f1", "micro_f1", "weighted_f1", "mcc", "f1_per_class") if k in dev]
            print("\nMétricas do DEV da melhor época:")
            _imprimir({k: dev[k] for k in chaves})
            resultado["dev"] = {k: dev[k] for k in chaves}
    print("\nO DEV serve para acompanhar o treino e escolher a época; não escolhe hiperparâmetros nem decide "
          "entre configurações (isso não é feito com o TEST, e esta fase não tem busca de hiperparâmetros).")
    return resultado


def estado_e_logs(ctx: Contexto) -> dict[str, Any]:
    nome, pasta, onde = _resolver_execucao(ctx)
    _titulo(f"Estado — {nome} ({onde})")
    derivado = proveniencia.derivar_estado(pasta)
    resumo: dict[str, Any] = {"nome": nome, "estado_pelos_arquivos": derivado["estado"],
                              "evidencia": derivado["evidencia"], "ambiguo": derivado["ambiguo"]}
    if (pasta / proveniencia.ARQUIVO).is_file():
        registro = proveniencia.ler(pasta)
        resumo.update({"run_id": registro["run_id"], "config_id": registro["config_id"],
                       "estado_registrado": registro["estado"], "nome_oficial": registro.get("alias_oficial"),
                       "tentativa": registro["tentativa"],
                       "historico_de_estados": registro["historico_de_estados"][-5:],
                       "checkpoints": registro["checkpoints"], "codigo": registro["codigo"],
                       "pesos": registro["pesos_pretreinados"], "publicacoes": registro["publicacoes"]})
    if (pasta / proveniencia.CHECKPOINTS / "ultimo.json").is_file():
        resumo["ultimo_checkpoint"] = ler_json(pasta / proveniencia.CHECKPOINTS / "ultimo.json").get("posicao")
    _imprimir(resumo)
    logs = sorted((pasta / "logs").glob("*.log")) if (pasta / "logs").is_dir() else []
    _titulo("Logs")
    _tabela([{"arquivo": p.name, "bytes": p.stat().st_size} for p in logs], ["arquivo", "bytes"])
    if logs:
        linhas = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, ctx.linhas_do_log):]
        print(f"\nÚltimas {len(linhas)} linhas de {logs[-1].name}:\n")
        print(segredos.limpar("\n".join(linhas)))
    else:
        print("(sem logs nesta execução)")
    return resumo


# --------------------------------------------------------------------------- #
# 12 e 13 — Drive                                                             #
# --------------------------------------------------------------------------- #
def sincronizar(ctx: Contexto) -> dict[str, Any]:
    nome, pasta, onde = _resolver_execucao(ctx)
    if onde != "local":
        raise ErroOperacao(f"{nome} só existe no Drive; não há o que sincronizar")
    try:
        resultado = persistencia.sincronizar_e_registrar(pasta, _exigir_drive(ctx), progresso=print)
    except persistencia.ErroSincronizacao as erro:
        raise ErroOperacao(str(erro)) from erro
    _imprimir(resultado)
    return resultado


def restaurar(ctx: Contexto) -> dict[str, Any]:
    nome = ctx.execucao
    if not nome:
        raise ErroOperacao("informe EXECUCAO (o nome da pasta no Drive)")
    try:
        resultado = persistencia.restaurar(_exigir_drive(ctx), nome, ctx.raiz_local, progresso=print)
    except persistencia.ErroSincronizacao as erro:
        raise ErroOperacao(str(erro)) from erro
    _imprimir(resultado)
    return resultado


# --------------------------------------------------------------------------- #
# 9 — preparar o TEST (não avalia)                                            #
# --------------------------------------------------------------------------- #
def frase_de_autorizacao_test(run_id: str) -> str:
    return f"{PREFIXO_AUTORIZACAO_TEST} {run_id} {SUFIXO_AUTORIZACAO_TEST}"


def autorizacao_test_valida(run_id: str, texto: str) -> bool:
    """A autorização do autor para o TEST de `run_id`: o texto exato (o notebook do TEST usa esta mesma função)."""
    return (texto or "").strip() == frase_de_autorizacao_test(run_id)


def preparar_test(ctx: Contexto) -> dict[str, Any]:
    """Confere as condições para avaliar o TEST de uma execução e imprime o resultado. **Não avalia o TEST**:
    não chama `avaliar_test.py` nem lê partição, predição ou métrica do TEST."""
    from reclin import versao_dados  # noqa: PLC0415
    from reclin.execucao import validacao  # noqa: PLC0415

    nome, pasta, onde = _resolver_execucao(ctx)
    checagens: list[dict[str, Any]] = []

    def checar(item: str, ok: bool, detalhe: str = "") -> None:
        checagens.append({"condição": item, "ok": "sim" if ok else "NÃO", "detalhe": detalhe})

    registro = proveniencia.ler(pasta)
    run = registro["run_id"]
    checar("execução está no disco local", onde == "local", "restaure do Drive (operação 13) se estiver só lá")
    checar("estratégia da execução", registro["estrategia"] == ctx.espec.estrategia, registro["estrategia"])
    derivado = proveniencia.derivar_estado(pasta)["estado"]
    checar("treino concluído (treino.json)", derivado == "concluida", derivado)
    checar("tem checkpoints/melhor_modelo/", (pasta / "checkpoints" / "melhor_modelo").is_dir())
    conteudo = ler_json(pasta / "config.json") if (pasta / "config.json").is_file() else {}
    gravada = (conteudo.get("config") or {})
    checar("não é validação técnica", not validacao.da_config(gravada))
    checar("configuração igual à do protocolo",
           gravada.get("config") == ctx.espec.config_protocolo().como_dict(),
           "execuções exploratórias não são elegíveis")
    checar("pesos com revisão e manifesto gravados", bool(gravada.get("proveniencia_pesos")))
    versoes = versao_dados.registradas(versao_dados.versoes_do_registro(conteudo).values())
    checar("dados da versão da fase", versoes == {versao_dados.VERSAO_DA_FASE}, ", ".join(sorted(versoes)) or "nenhuma")
    gravado_codigo = (conteudo.get("codigo") or {}).get("identidade")
    checar("identidade do código = a do código atual", gravado_codigo == ic.calcular(ctx.codigo)["identidade"])
    checar("TEST ainda não avaliado", not (pasta / "predicoes_test.json").exists()
           and not (pasta / "avaliacoes_test.jsonl").exists())
    sincronizada = bool(registro["checkpoints"].get("drive"))
    checar("execução espelhada no Drive", sincronizada, "operação 12")
    codigo_conferir = raiz.rodar_script(ctx.codigo, "pretreino.py", "conferir", "--execucoes", str(pasta),
                                        verificar=False)
    checar("scripts/pretreino.py conferir --execucoes", codigo_conferir == 0, f"saída {codigo_conferir}")
    _titulo(f"Condições para o TEST — {nome}")
    _tabela(checagens, ["condição", "ok", "detalhe"])
    tecnicamente_pronta = all(c["ok"] == "sim" for c in checagens)
    frase = frase_de_autorizacao_test(run)
    autorizada = autorizacao_test_valida(run, ctx.autorizacao_test)
    print("\nPronta tecnicamente:", "sim" if tecnicamente_pronta else "NÃO")
    print("Autorização explícita do autor:", "conferida" if autorizada else "pendente")
    print(f"Texto exato exigido: {frase!r}")
    print("\nEste notebook NÃO avalia o TEST. A avaliação é uma operação separada (reclin_teste.ipynb), só depois que "
          "todas as escolhas do DEV estiverem congeladas; o comando será:\n"
          f"  scripts/avaliar_test.py --execucao {pasta} --particoes {ctx.pasta_particoes}")
    return {"nome": nome, "run_id": run, "pronta_tecnicamente": tecnicamente_pronta, "autorizada": autorizada,
            "checagens": checagens, "executado": False}


# --------------------------------------------------------------------------- #
# 10 e 15 — publicação                                                        #
# --------------------------------------------------------------------------- #
def frase_de_publicacao(nome: str, repositorio: str) -> str:
    return f"PUBLICAR {nome} EM {repositorio}"


def _prefixo_hf(ctx: Contexto, pasta: Path | None = None) -> str | None:
    """O prefixo da publicação: `runs/<nome>` (None) ou, num estágio do Two-Stage, `etapa1`/`etapa2`."""
    n = ctx.espec.estagio
    if n is None:
        return ctx.prefixo_hf or None
    if ctx.prefixo_hf and ctx.prefixo_hf != f"etapa{n}":
        raise ErroOperacao(f"o estágio {n} só se publica em etapa{n}/ (recebi {ctx.prefixo_hf!r})")
    if pasta is not None and proveniencia.ler(pasta)["estrategia"] != ctx.espec.estrategia:
        raise ErroOperacao(f"{pasta.name} não é uma execução de {ctx.espec.estrategia}")
    return f"etapa{n}"


def preparar_publicacao(ctx: Contexto) -> dict[str, Any]:
    if not ctx.repositorio_hf:
        raise ErroOperacao("informe REPOSITORIO_HF (usuario/nome do repositório privado de artefatos) na célula de "
                           "configuração; ele só é exigido para publicar")
    nome, pasta, onde = _resolver_execucao(ctx)
    if onde != "local":
        raise ErroOperacao(f"{nome} só existe no Drive: restaure-a (operação 13) antes de planejar a publicação")
    p = publicacao_hf.planejar(pasta, ctx.repositorio_hf, prefixo=_prefixo_hf(ctx, pasta))
    _titulo(f"Plano de publicação — {nome}")
    print(f"Destino: https://huggingface.co/{ctx.repositorio_hf}/tree/main/{p['prefixo']} (repositório privado)")
    print(f"{p['n_arquivos']} arquivos, {p['bytes'] / 1e6:.1f} MB. Nada foi enviado.")
    _tabela([{"arquivo": a, "MB": f"{t / 1e6:.2f}"} for a, t in sorted(p["arquivos"].items())], ["arquivo", "MB"])
    if p["excluidos"]:
        print("\nNão publicados por padrão:")
        _tabela(p["excluidos"], ["arquivo", "motivo"])
    for problema in p["problemas"]:
        print("PROBLEMA:", problema)
    print("\nPode publicar:", "sim" if p["pode_publicar"] else "NÃO")
    print(f"Para enviar, escolha '{OP_PUBLICAR}' e informe CONFIRMACAO = {frase_de_publicacao(nome, ctx.repositorio_hf)!r}")
    return {k: p[k] for k in ("repositorio", "prefixo", "n_arquivos", "bytes", "problemas", "pode_publicar")}


def publicar(ctx: Contexto) -> dict[str, Any]:
    if not ctx.repositorio_hf:
        raise ErroOperacao("informe REPOSITORIO_HF na célula de configuração")
    nome, pasta, onde = _resolver_execucao(ctx)
    esperado = frase_de_publicacao(nome, ctx.repositorio_hf)
    if ctx.confirmacao.strip() != esperado:
        raise ErroOperacao(f"publicação não confirmada. Confira o destino com '{OP_PUBLICACAO}' e informe "
                           f"CONFIRMACAO = {esperado!r}")
    if onde != "local":
        raise ErroOperacao(f"{nome} só existe no Drive: restaure-a antes de publicar")
    prefixo = _prefixo_hf(ctx, pasta)
    extra = ["--prefixo", prefixo] if prefixo else []
    codigo = raiz.rodar_script(ctx.codigo, "experimento.py", "publicar", nome, "--repositorio", ctx.repositorio_hf,
                               *extra, "--raiz-local", str(ctx.raiz_local), "--executar", verificar=False)
    if codigo != 0:
        raise ErroOperacao(f"a publicação terminou com código {codigo}; veja a saída acima (o Secret HF_TOKEN, com "
                           "escrita, precisa estar configurado e liberado para este notebook)")
    return {"nome": nome, "repositorio": ctx.repositorio_hf, "executado": True}


# --------------------------------------------------------------------------- #
# Despacho                                                                    #
# --------------------------------------------------------------------------- #
_DESPACHO: dict[str, Callable[[Contexto], Any]] = {
    OP_AMBIENTE: conferir_ambiente, OP_GPU: conferir_gpu, OP_LISTAR: listar_execucoes, OP_NOVO: treinar_novo,
    OP_RETOMAR: retomar_treino, OP_NOVA: treinar_nova_execucao, OP_DEV: metricas_dev, OP_ESTADO: estado_e_logs,
    OP_TEST: preparar_test, OP_PUBLICACAO: preparar_publicacao, OP_PESOS: obter_pesos,
    OP_SINCRONIZAR: sincronizar, OP_RESTAURAR: restaurar, OP_INSTALAR: instalar_dependencias, OP_PUBLICAR: publicar,
    OP_COMPOR: compor_dev,
}


def executar(operacao: str, ctx: Contexto) -> Any:
    """Executa a operação escolhida. Uma recusa vira mensagem clara (`ERRO:`), sem traceback;
    um erro inesperado continua sendo levantado."""
    if operacao not in _DESPACHO:
        raise ErroOperacao(f"operação desconhecida: {operacao!r}. Opções: {list(OPERACOES)}")
    print(f">>> {operacao}")
    try:
        if ctx.espec is None and operacao not in SEM_ESTRATEGIA:
            raise ErroOperacao("a estratégia não pôde ser importada (faltam dependências, como o torch): use a "
                               f"operação '{OP_INSTALAR}', reinicie a sessão se pedido e rode a preparação de novo")
        return _DESPACHO[operacao](ctx)
    except (ErroOperacao, plano.ErroPlano, proveniencia.ErroProveniencia, persistencia.ErroSincronizacao,
            publicacao_hf.ErroPublicacao, segredos.ErroSegredo, raiz.ErroRaiz) as erro:
        print(segredos.limpar(f"\nERRO: {erro}"))
        return None
