"""Apresentação do notebook do baseline por fases (Colab): o que se mostra, não o que se calcula.

O notebook em fases orquestra as operações de `operacoes` e usa este módulo para
o que é só apresentação ou conferência de leitura:

* `contexto` — monta o `Contexto` das operações (a estratégia vem do notebook);
* `semente_da_fase`, `instrucoes_secret`, `segredo_no_ambiente`;
* `verificar_conexoes` — Drive, GitHub e Hugging Face (Secrets presentes, acesso), com o que falta;
* `instalar_se_faltar` — instala sozinha o que falta e avisa se a sessão precisa reiniciar;
* `avisar_revisao_pesos` — avisa se o commit atual dos pesos no Hub difere do congelado;
* `conferir_particoes` — as partições da fase estão no código clonado?;
* `execucao_atual` — a execução mais recente (local ou Drive), sem campo `EXECUCAO`;
* `recomendar_ramo` — qual operação de treino (4, 5 ou 6) a situação pede;
* `quadro_pre_treino` / `exigir_quadro` — tudo o que vai valer, antes de treinar;
* `curvas_dev` — curvas do DEV lidas de `treino.json` (nada do TEST).

Nada aqui treina, calibra, baixa modelo, escreve numa execução ou lê o TEST. As
funções que podem ser recusadas (`ErroOperacao`) imprimem `ERRO:` e devolvem
None, como `operacoes.executar`.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from reclin.experimento import ambiente, colab, ids, operacoes, persistencia, plano, proveniencia, raiz, segredos, versionamento
from reclin.experimento.operacoes import ErroOperacao
from reclin.util.io import ler_json

MODOS = {operacoes.OP_NOVO: "novo", operacoes.OP_RETOMAR: "retomar", operacoes.OP_NOVA: "nova"}
OPCOES_DE_TREINO = tuple(MODOS)
PARTICOES = ("train", "dev", "test")
ESTADOS_ATIVOS = ("interrompida", "iniciada")


def _recusavel(funcao: Callable[..., Any]) -> Callable[..., Any]:
    """Uma recusa vira mensagem clara (`ERRO:`), sem traceback; um erro inesperado continua sendo levantado."""
    @functools.wraps(funcao)
    def interna(*args: Any, **kw: Any) -> Any:
        try:
            return funcao(*args, **kw)
        except (ErroOperacao, plano.ErroPlano, segredos.ErroSegredo, raiz.ErroRaiz) as erro:
            print(segredos.limpar(f"\nERRO: {erro}"))
            return None
    return interna


def _titulo(texto: str) -> None:
    print(f"\n=== {texto} ===")


def _exigir_estrategia(ctx: operacoes.Contexto) -> None:
    if ctx.espec is None:
        raise ErroOperacao("a estratégia não pôde ser importada: rode a célula 4 (instalação) e depois a 5")


# --------------------------------------------------------------------------- #
# Contexto, semente, Secrets                                                  #
# --------------------------------------------------------------------------- #
def pasta_dos_pesos(raiz_pesos: str | Path, revisao: str, nome: str = "biobertpt-all") -> Path:
    """Caminho fixo dos pesos (o resume compara o caminho do modelo), por revisão. `nome` é o do repositório
    sem o dono (`biobertpt-clin` para a variante clínica)."""
    return Path(raiz_pesos) / f"{nome}@{revisao or 'sem-revisao'}"


def contexto(codigo: str | Path, raiz_local: str | Path, raiz_drive: str | Path | None, raiz_pesos: str | Path,
             revisao_pesos: str, espec: operacoes.Especificacao | None, **extra: Any) -> operacoes.Contexto:
    """O `Contexto` das operações. A especificação da estratégia (`espec`) é entregue pelo notebook, que
    importa a estratégia na hora (este módulo, como `operacoes`, é neutro e não a importa); com `espec`
    None só as operações de ambiente, GPU e instalação funcionam."""
    codigo = Path(codigo)
    nome_pesos = extra.pop("nome_pesos", "biobertpt-all")
    return operacoes.Contexto(
        codigo=codigo, raiz_local=Path(raiz_local), raiz_drive=Path(raiz_drive) if raiz_drive else None,
        espec=espec, pasta_pesos=pasta_dos_pesos(raiz_pesos, revisao_pesos, nome_pesos),
        pasta_particoes=codigo / "dados" / "particoes-v2", revisao_pesos=revisao_pesos, **extra)


def semente_da_fase() -> tuple[int, str]:
    """`(semente, pasta oficial)`: a semente vem da constante do protocolo e é conferida contra o sufixo
    `seedNN` do nome da pasta oficial do baseline. Divergência é erro de configuração do projeto."""
    oficiais = [n for n, e in plano.NOMES_OFICIAIS.items() if e == "baseline"]
    if len(oficiais) != 1:
        raise ErroOperacao(f"esperava um nome oficial para o baseline, achei {oficiais}")
    achado = re.search(r"seed(\d+)$", oficiais[0])
    if not achado or int(achado.group(1)) != operacoes.SEMENTE_DA_FASE:
        raise ErroOperacao(f"a pasta oficial {oficiais[0]!r} não aponta para a semente "
                           f"{operacoes.SEMENTE_DA_FASE} do protocolo")
    return operacoes.SEMENTE_DA_FASE, oficiais[0]


def instrucoes_secret(nome: str) -> str:
    """Passo a passo para criar o Secret `nome` e liberá-lo para o notebook (nunca pede o valor no chat)."""
    s = segredos.CATALOGO[nome]
    return "\n".join([
        f"Como configurar o Secret {nome} ({s.finalidade}):",
        "  1. No Colab, abra o ícone de chave (Secrets) na barra lateral esquerda.",
        "  2. Clique em «Adicionar novo secret».",
        f"  3. Nome: {nome}   Valor: o token ({s.permissao_minima}). Nunca cole o token no notebook nem no chat.",
        "  4. Ligue a chave «Acesso pelo notebook» (Notebook access) para ESTE notebook.",
        "  5. Rode a célula de novo."])


def _preenchido(valor: str) -> bool:
    return bool(valor and valor.strip()) and "<PREENCHER" not in valor


def verificar_conexoes(*, repositorio: str, branch_resultados: str, git_nome: str, git_email: str,
                       repositorio_hf: str, pasta_no_drive: str) -> dict[str, Any]:
    """Monta o Drive e confere GitHub e Hugging Face, dizendo exatamente o que falta. `bloqueios` impedem
    seguir (Drive, branch de resultados inválida); `pendencias` só impedem os passos de salvar (8b e 8c) e
    cada uma diz o que fazer. Nenhum valor de Secret é impresso nem exportado."""
    bloqueios: list[str] = []
    pendencias: list[str] = []
    saida: dict[str, Any] = {"raiz_drive": None, "bloqueios": bloqueios, "pendencias": pendencias}

    def linha(ok: bool, texto: str) -> None:
        print(f"  [{'ok' if ok else 'FALTA'}] {texto}")

    _titulo("Drive")
    try:
        saida["raiz_drive"] = colab.montar_drive() / pasta_no_drive
        saida["raiz_drive"].mkdir(parents=True, exist_ok=True)
        linha(True, f"Drive montado; execuções espelhadas em {saida['raiz_drive']}")
    except colab.ErroColab as erro:
        linha(False, f"Drive: {erro}")
        bloqueios.append("Drive não montado: rode a célula de novo e autorize o acesso na janela do Colab")

    _titulo("GitHub")
    try:
        versionamento._validar_destino(repositorio, branch_resultados)
        linha(True, f"resultados vão para a branch {branch_resultados!r} (main e master são recusadas)")
    except versionamento.ErroVersionamento as erro:
        linha(False, str(erro))
        bloqueios.append(str(erro))
    escrita = segredos.obter(segredos.GITHUB_ESCRITA)
    linha(escrita is not None, "Secret GITHUB_TOKEN_ESCRITA (só para o passo 8b)")
    if escrita is None:
        pendencias.append("GITHUB_TOKEN_ESCRITA ausente (necessário no passo 8b).\n"
                          + instrucoes_secret(segredos.GITHUB_ESCRITA))
    for nome, valor, campo in (("nome", git_nome, "GIT_NOME"), ("e-mail", git_email, "GIT_EMAIL")):
        linha(_preenchido(valor), f"{campo} (autor do commit)")
        if not _preenchido(valor):
            pendencias.append(f"preencha {campo} na célula 1 (o {nome} que assina o commit no passo 8b)")

    _titulo("Hugging Face")
    linha(_preenchido(repositorio_hf), "REPOSITORIO_HF")
    if not _preenchido(repositorio_hf):
        pendencias.append("preencha REPOSITORIO_HF na célula 1 (usuario/nome do repositório PRIVADO; passo 8c)")
    token = segredos.obter(segredos.HUGGING_FACE)
    linha(token is not None, "Secret HF_TOKEN (só para o passo 8c)")
    if token is None:
        pendencias.append("HF_TOKEN ausente (necessário no passo 8c).\n" + instrucoes_secret(segredos.HUGGING_FACE))
    elif _preenchido(repositorio_hf):
        try:
            quem = segredos.verificar_escrita_hf(token)
            linha(True, f"token de escrita válido (usuário {quem['usuario']})")
            from huggingface_hub import HfApi  # noqa: PLC0415

            try:
                info = HfApi().repo_info(repositorio_hf, repo_type="model", token=token.valor())
                if getattr(info, "private", True):
                    linha(True, f"repositório {repositorio_hf} existe e é privado")
                else:
                    linha(False, f"repositório {repositorio_hf} é PÚBLICO")
                    pendencias.append(f"{repositorio_hf} é público: a publicação o recusa. Torne-o privado ou crie outro")
            except Exception as erro:  # noqa: BLE001 — a classe do erro do Hub varia entre versões
                if type(erro).__name__ == "RepositoryNotFoundError":
                    linha(True, f"repositório {repositorio_hf} ainda não existe: o passo 8c o cria como PRIVADO")
                else:
                    linha(False, f"não consegui consultar {repositorio_hf} ({type(erro).__name__})")
                    pendencias.append(f"sem acesso ao repositório {repositorio_hf}: confira o nome e as permissões do token")
        except segredos.ErroSegredo as erro:
            linha(False, segredos.limpar(str(erro), token))
            pendencias.append(f"HF_TOKEN sem escrita ou recusado: {segredos.limpar(str(erro), token)}")
        except ImportError:
            linha(False, "huggingface_hub ainda não instalado (a etapa 4 o instala); refaça esta verificação depois")

    _titulo("Resumo")
    for b in bloqueios:
        print("PARE:", b)
    for p in pendencias:
        print("A FAZER ANTES DO PASSO 8:", p)
    if not bloqueios and not pendencias:
        print("Tudo certo: Drive, GitHub e Hugging Face prontos.")
    elif not bloqueios:
        print("\nO treino pode seguir; as pendências só travam salvar no GitHub e publicar no Hugging Face.")
    return saida


@_recusavel
def instalar_se_faltar(ctx: operacoes.Contexto) -> dict[str, Any]:
    """Instala sozinha as dependências que faltam (sem confirmação) e diz se é preciso reiniciar a sessão."""
    relatorio = ambiente.verificar(ctx.codigo, modo=ctx.modo_ambiente)
    _titulo("Dependências")
    if not relatorio["plano"]["instalar"]:
        for aviso in relatorio["avisos"]:
            print("AVISO:", aviso)
        for erro in relatorio["erros"]:
            print("ERRO:", erro)
        print("Dependências ok: nada a instalar." if relatorio["ok"] else "Há pendências que a instalação não resolve.")
        return {"ok": relatorio["ok"], "reiniciar": False}
    reiniciar = bool(relatorio["plano"]["reiniciar"])
    print("Faltam dependências; instalando agora:")
    for motivo in relatorio["plano"]["motivos"]:
        print(" -", motivo)
    depois = operacoes.instalar_dependencias(ctx)
    return {"ok": bool(depois["ok"]), "reiniciar": reiniciar}


@_recusavel
def avisar_revisao_pesos(ctx: operacoes.Contexto) -> str | None:
    """Compara o commit atual dos pesos no Hub com `REVISAO_PESOS` e avisa se mudou. Só consulta metadados
    (nada é baixado); sem rede, segue em silêncio relativo. O commit usado fica registrado em
    `PROVENIENCIA.json` ao baixar e é o que vale para a fase."""
    from reclin import pretreino  # noqa: PLC0415

    repositorio = ctx.repositorio_pesos or pretreino.REPOSITORIO_BIOBERTPT
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        atual = HfApi().model_info(repositorio).sha
    except Exception as erro:  # noqa: BLE001
        print(f"(não consegui consultar o commit atual de {repositorio} no Hub: {type(erro).__name__}; "
              f"a fase usa {ctx.revisao_pesos[:12]}…)")
        return None
    if atual == ctx.revisao_pesos:
        print(f"Pesos: {repositorio}@{ctx.revisao_pesos[:12]}… é o commit atual do Hub.")
    else:
        print(f"AVISO: o commit atual de {repositorio} ({atual[:12]}…) difere de REVISAO_PESOS ({ctx.revisao_pesos[:12]}…).\n"
              "       A fase continua usando REVISAO_PESOS: o download confere os hashes e o commit baixado fica "
              "registrado (PROVENIENCIA.json) e congelado para a fase. Só troque REVISAO_PESOS por decisão sua.")
    return atual


@contextlib.contextmanager
def segredo_no_ambiente(nome: str) -> Iterator[None]:
    """Exporta o Secret `nome` por `os.environ` só durante o bloco e o remove ao sair (também em erro).
    Nunca imprime o valor. Falta do Secret levanta `ErroSegredo` com a instrução de configuração."""
    credencial = segredos.obter(nome, obrigatorio=True)
    anterior = os.environ.get(nome)
    os.environ[nome] = credencial.valor()
    try:
        yield
    finally:
        if anterior is None:
            os.environ.pop(nome, None)
        else:
            os.environ[nome] = anterior


# --------------------------------------------------------------------------- #
# Partições                                                                   #
# --------------------------------------------------------------------------- #
@_recusavel
def conferir_particoes(ctx: operacoes.Contexto, *, sha_codigo: str = "") -> dict[str, Any]:
    """Confere as partições da fase (`dados/particoes-v2/`) que vieram no código do SHA: os arquivos existem
    e o SHA-256 de cada um bate com o MANIFEST (`scripts/preparar_dados.py conferir`). Imprime o número
    de documentos por partição (do MANIFEST; o conteúdo do TEST não é lido além da conferência do hash).
    Não gera partições: elas são versionadas e congeladas."""
    pasta = ctx.pasta_particoes
    _titulo(f"Partições — {pasta.name}")
    esperados = [pasta / f"{n}.jsonl" for n in PARTICOES] + [pasta / "MANIFEST.json"]
    faltam = [a.name for a in esperados if not a.is_file()]
    if faltam:
        raise ErroOperacao(
            f"o código clonado (commit {sha_codigo[:12] or '?'}…) não contém as partições da fase em "
            f"dados/particoes-v2/ (faltam: {', '.join(faltam)}). Use, em BRANCH_CODIGO (célula 1), uma branch cujo "
            "commit as inclua e rode de novo desde a célula 2. As partições não são geradas aqui: são versionadas e congeladas.")
    codigo = raiz.rodar_script(ctx.codigo, "preparar_dados.py", "--pasta", str(pasta), "conferir", verificar=False)   # --pasta vem ANTES do subcomando
    if codigo != 0:
        raise ErroOperacao(f"a conferência das partições terminou com código {codigo} (a saída do script pode não aparecer "
                           "na célula; rode `!python codigo/scripts/preparar_dados.py --pasta codigo/dados/particoes-v2 conferir` "
                           "dentro do repositório para ver): as partições do código clonado podem estar alteradas ou ser de outra versão.")
    manifesto = ler_json(pasta / "MANIFEST.json")
    if manifesto.get("versao") != operacoes._versao_da_fase():
        raise ErroOperacao(f"as partições são da versão {manifesto.get('versao')!r}; a fase usa "
                           f"{operacoes._versao_da_fase()!r}")
    contagens = manifesto.get("contagens") or {}
    linhas = [{"partição": n, "documentos": (contagens.get(n) or {}).get("documentos"),
               "candidatos": (contagens.get(n) or {}).get("candidatos")} for n in PARTICOES]
    operacoes._tabela(linhas, ["partição", "documentos", "candidatos"])
    print(f"\nVersão dos dados: {manifesto.get('versao')} | partições conferidas contra o MANIFEST.")
    return {"ok": True, "versao": manifesto.get("versao"), "contagens": {l["partição"]: l["documentos"] for l in linhas}}


# --------------------------------------------------------------------------- #
# Recomendação de ramo                                                        #
# --------------------------------------------------------------------------- #
@_recusavel
def recomendar_ramo(ctx: operacoes.Contexto) -> dict[str, Any]:
    """Qual operação de treino (4, 5 ou 6) a situação pede, para os `PARAMETROS` atuais. Só lê.

    * execução interrompida **compatível** com estes parâmetros → 5;
    * parâmetros diferentes do protocolo → 6 (execução exploratória; a 4 não vale);
    * protocolo já concluído → 6 (nova tentativa);
    * senão → 4."""
    _exigir_estrategia(ctx)
    config = operacoes.config_da_operacao(ctx.espec, ctx.parametros)
    protocolo = config == ctx.espec.config_protocolo()
    chave = operacoes._chave(ctx, config)
    cid = ids.config_id(chave)
    compat = plano.compativeis(chave, ctx.raiz_local, ctx.raiz_drive)
    ativas = [e for e in compat if e["estado"] in ESTADOS_ATIVOS]
    concluidas = [e for e in compat if e["estado"] == "concluida"]
    outras = sorted({e["nome"] for e in plano.execucoes(ctx.raiz_local, ctx.raiz_drive)
                     if e["run_id"] and operacoes.ids_estrategia(e["run_id"]) == ctx.espec.estrategia
                     and e["config_id"] != cid})
    if ativas:
        recomendada = operacoes.OP_RETOMAR
        motivo = ("há execução interrompida COMPATÍVEL com estes parâmetros: "
                  + ", ".join(f"{e['nome']} ({e['onde']}, {e['estado']})" for e in ativas))
    elif not protocolo:
        recomendada = operacoes.OP_NOVA
        motivo = ("PARAMETROS difere do protocolo: a execução é EXPLORATÓRIA (fora da fase científica, não elegível "
                  "ao TEST) e só a 6 a cria")
    elif concluidas:
        recomendada = operacoes.OP_NOVA
        motivo = (f"a execução do protocolo já está concluída ({concluidas[0]['nome']}); nada é sobrescrito. "
                  "A 6 faz outra tentativa")
    else:
        recomendada = operacoes.OP_NOVO
        motivo = "ainda não há execução do protocolo"
    _titulo("Recomendação")
    print(f"Use: {recomendada}\nPorque: {motivo}.")
    if outras:
        print(f"\nExistem execuções de OUTRA configuração: {', '.join(outras)}.\n"
              "Para retomar uma delas, ponha em PARAMETROS (célula 1) os valores dela e use a 5.")
    return {"recomendada": recomendada, "motivo": motivo, "compativeis": [e["nome"] for e in compat], "outras": outras}


# --------------------------------------------------------------------------- #
# Quadro pré-treino                                                           #
# --------------------------------------------------------------------------- #
def assinatura(ctx: operacoes.Contexto, operacao: str) -> str:
    """Impressão digital das escolhas da célula 6a (para provar que o quadro mostrado é o que vai valer)."""
    material = {"operacao": operacao, "parametros": dict(ctx.parametros or {}), "motivo": ctx.motivo.strip(),
                "checkpoint_a_cada": int(ctx.checkpoint_a_cada), "espelhar_a_cada_min": float(ctx.espelhar_a_cada_min),
                "revisao_pesos": ctx.revisao_pesos, "pasta_pesos": str(ctx.pasta_pesos)}
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _linhas(pares: list[tuple[str, Any]]) -> None:
    largura = max(len(k) for k, _ in pares)
    for chave, valor in pares:
        print(f"  {chave.ljust(largura)} : {valor}")


@_recusavel
def quadro_pre_treino(ctx: operacoes.Contexto, operacao: str, *, sha_codigo: str = "",
                      repositorio: str = "") -> dict[str, Any]:
    """Imprime tudo o que vai valer no treino e os impedimentos previstos. Não cria nada: o plano é só de
    leitura (`plano.planejar`). As recusas de verdade continuam sendo as das operações 4, 5 e 6."""
    if operacao not in MODOS:
        raise ErroOperacao(f"escolha uma das operações de treino: {list(OPCOES_DE_TREINO)}")
    _exigir_estrategia(ctx)
    modo = MODOS[operacao]
    impedimentos: list[str] = []
    config = chave = p = None
    protocolo = None
    try:
        config = operacoes.config_da_operacao(ctx.espec, ctx.parametros)
        protocolo = config == ctx.espec.config_protocolo()
        chave = operacoes._chave(ctx, config)
        alias = ctx.espec.alias_oficial if (modo == "novo" and protocolo) else None
        p = plano.planejar(chave, ctx.raiz_local, ctx.raiz_drive, alias_oficial=alias, nova_tentativa=(modo == "nova"))
    except (ErroOperacao, plano.ErroPlano, ValueError) as erro:
        impedimentos.append(str(erro))
    if ctx.raiz_drive is None:
        impedimentos.append("o Drive não está montado (célula 3)")
    if modo == "novo" and protocolo is False:
        impedimentos.append(f"PARAMETROS difere do protocolo: a operação 4 só vale com os valores do protocolo. Use "
                            f"'{operacoes.OP_NOVA}' (execução EXPLORATÓRIA: fora da fase científica e não elegível ao "
                            "TEST) ou volte PARAMETROS ao protocolo")
    if modo == "nova" and not ctx.motivo.strip():
        impedimentos.append("MOTIVO é obrigatório numa nova execução")
    if p is not None:
        if modo == "novo" and p["acao"] == "retomar":
            impedimentos.append(f"já existe a execução {p['nome']} desta configuração, interrompida: use a 5")
        if modo == "novo" and p["acao"] == "ja_concluida":
            impedimentos.append(f"a execução {p['nome']} desta configuração já está concluída: use a 6")
        if modo == "retomar" and p["acao"] != "retomar":
            impedimentos.append("não há execução interrompida COMPATÍVEL com estes parâmetros (a 5 mostrará o que difere)")
    atual = ambiente.coletar()
    gpu = ambiente.classificar_gpu(atual.get("gpu"))
    if gpu["classe"] == "sem_gpu":
        impedimentos.append("sem GPU: Ambiente de execução → Alterar tipo → T4 GPU")

    if p is None:
        natureza = "indefinida"
    elif protocolo and modo == "novo":
        natureza = f"PROTOCOLO (oficial: {p['alias_oficial']})" if p["alias_oficial"] else "PROTOCOLO (pasta com o run_id)"
    elif protocolo:
        natureza = "protocolo (nova tentativa)" if modo == "nova" else "protocolo (retomada)"
    else:
        natureza = "EXPLORATÓRIA (sem nome oficial; fora da fase científica; não elegível ao TEST)"
    drive = persistencia.raiz_das_execucoes(ctx.raiz_drive) / p["nome"] if (p and ctx.raiz_drive) else "—"
    _titulo("Quadro pré-treino — confira antes de treinar")
    _linhas([
        ("Operação", operacao),
        ("Natureza", natureza),
        ("Código (commit)", (f"{repositorio}@{sha_codigo}" if sha_codigo else str(ctx.codigo))),
        ("Pesos", f"{ctx.repositorio_pesos or 'pucpr/biobertpt-all'}@{ctx.revisao_pesos or '(sem revisão)'}"),
        ("Dados", f"{chave['dados']['versao']} (SHA-256 train {chave['dados']['particoes']['train'][:12]}…, "
                  f"dev {chave['dados']['particoes']['dev'][:12]}…)" if chave else "—"),
        ("GPU", f"{atual.get('gpu') or 'nenhuma'} ({gpu['classe']})"),
        ("config_id", p["config_id"] if p else "—"),
        ("Ação", (p["acao"] if p else "—")),
        ("Execução (pasta)", f"{p['nome']}  [run_id {p['run_id']}, tentativa {p['tentativa']}]" if p else "—"),
        ("Disco local", str(ctx.raiz_local / p["nome"]) if p else "—"),
        ("Destino no Drive", drive),
        ("Checkpoint", f"a cada {int(ctx.checkpoint_a_cada)} passos e no fim de cada época"),
        ("Espelho no Drive", f"a cada {max(1.0, float(ctx.espelhar_a_cada_min)):g} min e ao fim da sessão"),
        ("Motivo", ctx.motivo.strip() or "—"),
    ])
    if config is not None:
        print("\n  Parâmetros efetivos (o que o scripts/treinar.py receberá):")
        base = ctx.espec.config_protocolo().como_dict()
        for campo, valor in config.como_dict().items():
            if campo == "modelo":
                continue
            marca = "= protocolo" if base.get(campo) == valor else f"≠ protocolo ({base.get(campo)!r})"
            print(f"    {campo.ljust(14)} {valor!s:<12} {marca}")
    if p and p["avisos"]:
        for aviso in p["avisos"]:
            print("\n  AVISO:", aviso)
    if impedimentos:
        print("\nIMPEDIMENTOS (a célula 6b vai recusar enquanto houver):")
        for i in impedimentos:
            print(" -", segredos.limpar(i))
    else:
        print("\nPronto: a célula 6b executa exatamente isto. Se algo acima não é o que você quer, mude as escolhas "
              "e rode a 6a de novo.")
    return {"operacao": operacao, "modo": modo, "assinatura": assinatura(ctx, operacao), "pronto": not impedimentos,
            "impedimentos": impedimentos, "natureza": natureza, "config_id": p["config_id"] if p else None,
            "pasta": p["nome"] if p else None, "run_id": p["run_id"] if p else None}


def exigir_quadro(quadro: Mapping[str, Any] | None, ctx: operacoes.Contexto, operacao: str) -> None:
    """Recusa treinar sem o quadro da 6a, com o quadro de outras escolhas, ou com impedimentos."""
    if not quadro:
        raise ErroOperacao("rode a célula 6a antes: o treino só começa depois de você ver o quadro pré-treino")
    if quadro.get("assinatura") != assinatura(ctx, operacao):
        raise ErroOperacao("as escolhas mudaram desde o quadro da célula 6a: rode a 6a de novo e confira o quadro")
    if not quadro.get("pronto"):
        raise ErroOperacao("o quadro da célula 6a tem impedimentos: " + "; ".join(quadro.get("impedimentos") or []))


# --------------------------------------------------------------------------- #
# Execução escolhida e curvas do DEV                                          #
# --------------------------------------------------------------------------- #
def execucao_atual(ctx: operacoes.Contexto, *, restaurar: bool = False) -> str:
    """O nome da execução **mais recente** (por `atualizado_em`, local ou Drive) da estratégia **e do encoder**, sem campo
    `EXECUCAO`. Com `restaurar=True`, uma execução que só existe no Drive é trazida para o disco local
    (verificada) antes de devolver o nome. Levanta `ErroOperacao` se não há execução."""
    _exigir_estrategia(ctx)
    achadas = [e for e in plano.execucoes(ctx.raiz_local, ctx.raiz_drive)
               if e["run_id"] and operacoes.ids_estrategia(e["run_id"]) == ctx.espec.estrategia]
    if not achadas:
        raise ErroOperacao("nenhuma execução encontrada: treine primeiro (célula 6b) ou, numa sessão nova, "
                           "confira PASTA_NO_DRIVE")
    recentes: dict[str, str] = {}
    for e in achadas:
        pasta = (ctx.raiz_local if e["onde"] == "local" else persistencia.raiz_das_execucoes(ctx.raiz_drive)) / e["nome"]
        registro = proveniencia.ler(pasta)
        # outra variante de encoder (mesma estratégia) nunca é "a execução atual" deste notebook
        if (registro.get("parametros_de_treinamento") or {}).get("encoder", operacoes.ENCODER_DA_FASE) != ctx.espec.encoder:
            continue
        quando = registro.get("atualizado_em") or ""
        recentes[e["nome"]] = max(recentes.get(e["nome"], ""), quando)
    if not recentes:
        raise ErroOperacao(f"nenhuma execução do encoder {ctx.espec.encoder!r} encontrada (há execuções da estratégia "
                           "com outro encoder, que não são usadas aqui): treine primeiro")
    nome = max(recentes, key=lambda n: (recentes[n], n))
    if restaurar and not (ctx.raiz_local / nome / proveniencia.ARQUIVO).is_file():
        print(f"A execução mais recente ({nome}) só existe no Drive: restaurando para o disco local…")
        try:
            persistencia.restaurar(ctx.raiz_drive, nome, ctx.raiz_local, progresso=print)
        except persistencia.ErroSincronizacao as erro:
            raise ErroOperacao(f"não foi possível restaurar {nome} do Drive: {erro}") from erro
    print(f"Execução usada: {nome}")
    return nome


def pasta_local_da_execucao(ctx: operacoes.Contexto) -> Path:
    """A pasta local da execução indicada em `ctx.execucao`. Levanta `ErroOperacao`."""
    _exigir_estrategia(ctx)
    nome, pasta, onde = operacoes._resolver_execucao(ctx)
    if onde != "local":
        raise ErroOperacao(f"{nome} só existe no Drive: restaure-a antes")
    return pasta


@_recusavel
def curvas_dev(ctx: operacoes.Contexto) -> dict[str, Any] | None:
    """train_loss × dev_loss, macro-F1 do DEV e F1 de `negation_of` do DEV por época, de `treino.json`.
    Só o DEV: nenhum arquivo do TEST é aberto."""
    _exigir_estrategia(ctx)
    nome, pasta, onde = operacoes._resolver_execucao(ctx)
    _titulo(f"Curvas do DEV — {nome} ({onde})")
    caminho = pasta / "treino.json"
    if not caminho.is_file():
        print("treino.json ainda não existe (ele é gravado quando uma sessão de treino termina): sem curvas por enquanto. "
              "O andamento por época está no log (arquivo logs/treino.log da execução).")
        return None
    treino = ler_json(caminho)
    historico = treino.get("dev_history") or []
    if not historico:
        print("treino.json sem histórico por época.")
        return None
    import matplotlib.pyplot as plt  # noqa: PLC0415

    epocas = [h["epoch"] for h in historico]
    melhor = treino.get("melhor_epoca")
    figura, eixos = plt.subplots(1, 3, figsize=(15, 4))
    eixos[0].plot(epocas, [h["train_loss"] for h in historico], marker="o", label="train_loss")
    eixos[0].plot(epocas, [h["dev_loss"] for h in historico], marker="o", label="dev_loss")
    eixos[0].set_title("Perda: treino × DEV")
    eixos[1].plot(epocas, [h["dev_macro_f1"] for h in historico], marker="o", color="tab:green")
    eixos[1].set_title("macro-F1 no DEV")
    eixos[2].plot(epocas, [h.get("dev_negation_of_f1") for h in historico], marker="o", color="tab:red")
    eixos[2].set_title("F1 de negation_of no DEV")
    for eixo in eixos:
        eixo.set_xlabel("época")
        eixo.set_xticks(epocas)
        eixo.grid(alpha=0.3)
        if melhor in epocas:
            eixo.axvline(melhor, color="gray", linestyle="--", linewidth=1)
    eixos[0].legend()
    figura.suptitle(f"{nome} — melhor época pelo DEV: {melhor} (só DEV; nada do TEST)")
    figura.tight_layout()
    plt.show()
    return {"nome": nome, "melhor_epoca": melhor, "epocas": len(historico)}
