"""Salvar os resultados de uma execução no GitHub (célula 8b do notebook do baseline).

O código de treino vem de um clone raso da branch de código (`codigo_colab.clonar_codigo`). Para guardar os
resultados usa-se um **clone de trabalho separado** (raso), que nunca alimenta o treino: ele só recebe os arquivos de resultado e os envia a uma
**branch própria** (criada a partir do SHA do código quando ainda não existe).

Garantias:

* só entram os arquivos de `ARQUIVOS_DA_EXECUCAO` (configuração, histórico do treino, proveniência,
  predições do DEV e métricas do DEV), em `codigo/resultados/execucoes/<nome>/`; nunca checkpoints
  (`ultimo.pt`), pesos, logs, artefatos do TEST nem o que `publicacao_hf` já exclui da publicação;
* a execução precisa estar concluída; `metricas.json` com a chave `test` é recusado; `predicoes_dev.json`
  não pode conter texto (só rótulos, predições e probabilidades); nenhum arquivo pode conter segredo;
* o commit mostra antes o que será enviado (`conferir`); a frase de confirmação é opcional (`confirmacao=None` a dispensa);
* o token de escrita (`GITHUB_TOKEN_ESCRITA`) não é gravado em `.git/config`, na URL do remote, em argv nem
  em `os.environ` da sessão: vai por `GIT_ASKPASS` (script temporário, permissão 700, apagado em `finally`)
  e existe só no ambiente do subprocesso do git;
* nada é sobrescrito: arquivo diferente do que já está no clone é recusado; reexecutar sem mudança não cria
  commit.

O módulo só usa o Git quando `salvar` é chamada, e `salvar` só é chamada pela célula 8b.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

from reclin.experimento import codigo_colab, proveniencia, publicacao_hf, segredos
from reclin.util.io import ler_json, sha256_arquivo

ARQUIVOS_DA_EXECUCAO = ("config.json", "treino.json", "experimento.json", "predicoes_dev.json", "metricas.json")
# Perfis de salvamento: só os arquivos que a pasta já tem; nada é criado para ser versionado.
#  * execucao   — baseline e demais execuções com `experimento.json` (o padrão);
#  * estagio    — um estágio do Two-Stage: configuração e histórico do treino por época;
#  * composicao — a composição do Two-Stage: τ congelado, tabela da busca (dentro de `limiar.json`),
#                 métricas do DEV do pipeline e as predições/probabilidades do DEV.
PERFIS: dict[str, tuple[str, ...]] = {
    "execucao": ARQUIVOS_DA_EXECUCAO,
    "estagio": ("config.json", "treino.json"),
    "composicao": ("limiar.json", "metricas.json", "predicoes_dev.json"),
}
BASE_DESTINO = "codigo/resultados/execucoes"
BRANCHES_RECUSADAS = ("main", "master")
LIMITE_DE_TEXTO = 200
_PROIBIDOS = (re.compile(r"(^|/)checkpoints(/|$)"), re.compile(r"(^|/)logs(/|$)"),
              re.compile(r"\.(pt|pth|bin|safetensors|ckpt)$"), re.compile(r"(^|/)SYNC\.json$"),
              re.compile(r"(^|[_/.-])test([_./-]|$)"))

Executor = Callable[[list[str], "Path | None", "dict[str, str]"], "tuple[int, str]"]


class ErroVersionamento(RuntimeError):
    """O salvamento foi recusado ou o Git falhou; a mensagem diz o que fazer."""


# --------------------------------------------------------------------------- #
# Conferência (só leitura)                                                    #
# --------------------------------------------------------------------------- #
def frase_de_salvamento(nome: str, repositorio: str, branch: str) -> str:
    return f"SALVAR {nome} EM {repositorio}@{branch}"


def caminho_no_repositorio(nome: str, arquivo: str = "") -> str:
    base = f"{BASE_DESTINO}/{nome}"
    return f"{base}/{arquivo}" if arquivo else base


def _so_numeros(valor: Any) -> bool:
    if isinstance(valor, bool) or isinstance(valor, (int, float)) or valor is None:
        return True
    if isinstance(valor, (list, tuple)):
        return all(_so_numeros(v) for v in valor)
    return False


def sem_texto_clinico(predicoes: dict[str, Any]) -> list[str]:
    """Motivos pelos quais o sidecar parece conter texto: string longa, ou lista com algo além de números
    (exceto `labels`, os 3 nomes de rótulo)."""
    motivos = []
    for chave, valor in predicoes.items():
        if chave == "labels":
            continue
        if isinstance(valor, str) and len(valor) > LIMITE_DE_TEXTO:
            motivos.append(f"{chave}: texto de {len(valor)} caracteres")
        elif isinstance(valor, (list, tuple)) and not _so_numeros(valor):
            motivos.append(f"{chave}: lista com elementos que não são números")
    return motivos


LIMITE_DE_AVISO = 600
_AVISO_DE_BIBLIOTECA = re.compile(r"(User|Future|Deprecation|Runtime)Warning: ")


def _sem_avisos_de_biblioteca(conteudo: Any) -> Any:
    """Cópia do `treino.json` sem os avisos de biblioteca de `sessoes[*].avisos` (ex.: UserWarning do PyTorch).
    Só some o item que começa com `<Tipo>Warning: ` e tem até `LIMITE_DE_AVISO` caracteres; qualquer outro texto
    longo (neste campo ou em outro) continua sendo achado por `_textos_longos`."""
    if not isinstance(conteudo, dict) or not isinstance(conteudo.get("sessoes"), list):
        return conteudo
    sessoes = []
    for sessao in conteudo["sessoes"]:
        if isinstance(sessao, dict) and isinstance(sessao.get("avisos"), list):
            avisos = ["" if (isinstance(a, str) and len(a) <= LIMITE_DE_AVISO and _AVISO_DE_BIBLIOTECA.match(a)) else a
                      for a in sessao["avisos"]]
            sessao = {**sessao, "avisos": avisos}
        sessoes.append(sessao)
    return {**conteudo, "sessoes": sessoes}


def _textos_longos(valor: Any, caminho: str = "") -> list[str]:
    """Strings com mais de `LIMITE_DE_TEXTO` caracteres em qualquer nível (texto clínico não cabe nestes arquivos)."""
    achados: list[str] = []
    if isinstance(valor, str):
        if len(valor) > LIMITE_DE_TEXTO:
            achados.append(f"{caminho or '(raiz)'}: texto de {len(valor)} caracteres")
    elif isinstance(valor, dict):
        for k, v in valor.items():
            achados += _textos_longos(v, f"{caminho}.{k}" if caminho else str(k))
    elif isinstance(valor, (list, tuple)):
        for i, v in enumerate(valor):
            achados += _textos_longos(v, f"{caminho}[{i}]")
    return achados


def _predicoes_da_composicao_sem_texto(conteudo: dict[str, Any]) -> list[str]:
    """Como `sem_texto_clinico`, aceitando só `estagios` (os nomes das duas pastas) como lista de nomes."""
    resto = dict(conteudo)
    motivos = []
    nomes = resto.pop("estagios", [])
    if not isinstance(nomes, list) or not all(isinstance(n, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}", n)
                                              for n in nomes):
        motivos.append("estagios: não é uma lista de nomes de pasta")
    return motivos + sem_texto_clinico(resto) + _textos_longos(resto)


def _motivo_de_exclusao(relativo: str) -> str:
    if re.search(r"(^|/)checkpoints(/|$)|\.(pt|pth|bin|safetensors|ckpt)$", relativo):
        return "pesos/checkpoints: nunca versionados"
    if relativo in publicacao_hf.ARTEFATOS_DO_TEST or re.search(r"(^|[_/.-])test([_./-]|$)", relativo):
        return "artefato do TEST: nunca versionado aqui"
    if relativo.startswith("logs/"):
        return "logs: ficam no Drive"
    if relativo == "SYNC.json":
        return "controle do espelho no Drive"
    return "fora da lista de resultados versionáveis"


def plano_de_salvamento(pasta: str | Path, perfil: str = "execucao") -> dict[str, Any]:
    """O que seria versionado e por que algo impede (`problemas`). Não escreve nada."""
    if perfil not in PERFIS:
        raise ErroVersionamento(f"perfil de salvamento desconhecido: {perfil!r}. Opções: {list(PERFIS)}")
    lista = PERFIS[perfil]
    pasta = Path(pasta)
    problemas: list[str] = []
    arquivos: dict[str, dict[str, Any]] = {}
    registro: dict[str, Any] | None = None
    if not pasta.is_dir():
        return {"nome": pasta.name, "pasta": str(pasta), "arquivos": {}, "excluidos": [], "registro": None,
                "problemas": [f"a pasta {pasta} não existe"]}
    if perfil != "composicao":
        try:
            registro = proveniencia.ler(pasta)
        except proveniencia.ErroProveniencia as erro:
            problemas.append(f"experimento.json inválido: {erro}")
    if registro is not None and registro.get("estado") != "concluida":
        problemas.append(f"a execução está {registro.get('estado')!r}: só se salva execução concluída")
    for relativo in lista:
        arquivo = pasta / relativo
        if not arquivo.is_file():
            problemas.append(f"falta {relativo}")
            continue
        arquivos[relativo] = {"bytes": arquivo.stat().st_size, "sha256": sha256_arquivo(arquivo)}
        try:
            conteudo = ler_json(arquivo)
        except (ValueError, OSError) as erro:
            problemas.append(f"{relativo} ilegível: {erro}")
            continue
        achados = segredos.contem_segredo(conteudo)
        if achados:
            problemas.append(f"{relativo} parece conter segredo ({'; '.join(achados)})")
        if perfil != "execucao":
            verificado = _sem_avisos_de_biblioteca(conteudo) if relativo == "treino.json" else conteudo
            problemas += [f"{relativo} contém texto? {m}" for m in _textos_longos(verificado)
                          if not (relativo == "predicoes_dev.json")]
        if relativo == "treino.json" and not conteudo.get("concluido"):
            problemas.append("treino.json diz que o treino não foi concluído")
        if perfil == "estagio" and relativo == "config.json" and "validacao" in ((conteudo.get("config") or {})):
            problemas.append("config.json é de uma execução de validação técnica: não é resultado da fase")
        if perfil == "composicao" and relativo == "limiar.json":
            if "validacao" in conteudo:
                problemas.append("limiar.json é de uma composição de validação técnica: não é resultado da fase")
            if "limiar" not in conteudo or "tabela" not in conteudo:
                problemas.append("limiar.json sem 'limiar' ou 'tabela': τ não está congelado")
        if relativo == "metricas.json":
            if "test" in conteudo:
                problemas.append("metricas.json tem a chave 'test': resultados do TEST não são versionados por aqui "
                                 "(nada foi alterado no arquivo)")
            if "dev" not in conteudo:
                problemas.append("metricas.json não tem a chave 'dev'")
        if relativo == "predicoes_dev.json":
            motivos = (_predicoes_da_composicao_sem_texto(conteudo) if perfil == "composicao"
                       else sem_texto_clinico(conteudo))
            problemas += [f"predicoes_dev.json contém texto? {m}" for m in motivos]
    if perfil == "composicao" and (pasta / "predicoes_test.json").exists():
        problemas.append("há artefatos do TEST nesta composição: não é versionada por aqui")
    excluidos = [{"arquivo": a.relative_to(pasta).as_posix(), "motivo": _motivo_de_exclusao(a.relative_to(pasta).as_posix())}
                 for a in sorted(pasta.rglob("*")) if a.is_file() and a.relative_to(pasta).as_posix() not in arquivos]
    return {"nome": pasta.name, "pasta": str(pasta), "arquivos": arquivos, "excluidos": excluidos,
            "registro": registro, "problemas": problemas}


def _validar_destino(repositorio: str, branch: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repositorio):
        raise ErroVersionamento(f"REPOSITORIO {repositorio!r} deve ser usuario/nome")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch) or ".." in branch or branch.endswith(("/", ".lock")):
        raise ErroVersionamento(f"BRANCH_RESULTADOS {branch!r} não é um nome de branch válido")
    if branch in BRANCHES_RECUSADAS:
        raise ErroVersionamento(f"os resultados não vão para a branch {branch!r}: use uma branch própria "
                                "(por exemplo 'resultados-colab'); a integração à principal é manual")


def _validar_identidade(nome: str, email: str) -> None:
    if not nome.strip() or "\n" in nome or nome.startswith("-"):
        raise ErroVersionamento("informe GIT_NOME na célula 1 (o nome que assinará o commit)")
    if not re.fullmatch(r"[^\s@]+@[^\s@]+", email.strip()):
        raise ErroVersionamento("informe GIT_EMAIL na célula 1 (o e-mail que assinará o commit)")


def conferir(pasta: str | Path, *, repositorio: str, branch: str, sha_codigo: str,
             imprimir: Callable[[str], None] = print, perfil: str = "execucao") -> dict[str, Any]:
    """Mostra o que seria enviado ao GitHub e o que NUNCA vai. Não usa o Git nem a rede."""
    _validar_destino(repositorio, branch)
    p = plano_de_salvamento(pasta, perfil)
    nome = p["nome"]
    imprimir(f"\n=== Salvar no GitHub — {nome} ===")
    imprimir(f"Repositório: {repositorio} | branch: {branch} (criada a partir do SHA {sha_codigo[:12]}… se não existir)")
    imprimir(f"Destino no repositório: {caminho_no_repositorio(nome)}/")
    imprimir("\nSerão versionados (git add -f, lista explícita):")
    for relativo, info in p["arquivos"].items():
        imprimir(f"  {relativo:<20} {info['bytes']:>12,} bytes  sha256 {info['sha256'][:16]}…")
    if p["excluidos"]:
        imprimir("\nNunca versionados (ficam fora):")
        for item in p["excluidos"]:
            imprimir(f"  {item['arquivo']:<45} {item['motivo']}")
    for problema in p["problemas"]:
        imprimir(f"PROBLEMA: {problema}")
    p["pode_salvar"] = not p["problemas"]
    imprimir("\nPode salvar: " + ("sim" if p["pode_salvar"] else "NÃO"))
    return p


# --------------------------------------------------------------------------- #
# Git                                                                         #
# --------------------------------------------------------------------------- #
def executor_padrao(argumentos: list[str], cwd: Path | None, env_extra: dict[str, str]) -> tuple[int, str]:
    """Roda `git <argumentos>`; o ambiente extra existe só neste subprocesso."""
    env = {**os.environ, **env_extra, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C.UTF-8"}
    r = subprocess.run(["git", *argumentos], cwd=cwd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900, check=False)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


@contextlib.contextmanager
def _askpass(credencial: segredos.Credencial) -> Iterator[dict[str, str]]:
    """`GIT_ASKPASS` temporário (ver `codigo_colab.askpass`): o token só existe no ambiente do subprocesso do Git."""
    with codigo_colab.askpass(credencial.valor()) as env:
        yield env


class _Sessao:
    def __init__(self, *, executor: Executor, env: dict[str, str], identidade: list[str],
                 credencial: segredos.Credencial, imprimir: Callable[[str], None]):
        self.executor, self.env, self.identidade = executor, env, identidade
        self.credencial, self.imprimir = credencial, imprimir

    def git(self, *argumentos: str, cwd: Path | None = None, aceitar: tuple[int, ...] = (0,)) -> str:
        self.imprimir("$ git " + " ".join(argumentos))
        codigo, saida = self.executor([*self.identidade, *argumentos], cwd, self.env)
        saida = segredos.limpar(saida, self.credencial).strip()
        if saida:
            self.imprimir(saida)
        if codigo not in aceitar:
            raise ErroVersionamento(f"git {argumentos[0]} terminou com código {codigo}"
                                    + (f": {saida.splitlines()[-1]}" if saida else ""))
        return saida if codigo == 0 else str(codigo)


def _identidade_do_git(nome: str, email: str) -> list[str]:
    return ["-c", "credential.helper=", "-c", f"user.name={nome.strip()}", "-c", f"user.email={email.strip()}"]


def _preparar_clone(s: _Sessao, *, url: str, clone: Path, branch: str, sha_codigo: str, remoto_existe: bool) -> None:
    if (clone / ".git").is_dir():
        atual = s.git("remote", "get-url", "origin", cwd=clone)
        if atual != url:
            raise ErroVersionamento(f"o clone em {clone} aponta para {atual}, não para {url}: remova a pasta ou "
                                    "corrija REPOSITORIO")
        cabeca = s.git("rev-parse", "--abbrev-ref", "HEAD", cwd=clone)
        if cabeca != branch:
            raise ErroVersionamento(f"o clone em {clone} está na branch {cabeca!r}, não em {branch!r}: remova a pasta "
                                    "ou corrija BRANCH_RESULTADOS")
        s.imprimir(f"(reutilizando o clone de trabalho em {clone})")
        return
    if clone.exists() and any(clone.iterdir()):
        raise ErroVersionamento(f"{clone} existe, não é um clone e não está vazio: remova-o antes")
    clone.parent.mkdir(parents=True, exist_ok=True)
    s.git("clone", "--depth", "1", "--no-checkout", url, str(clone))
    if remoto_existe:
        s.git("fetch", "--depth", "1", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}", cwd=clone)
        s.git("checkout", "-B", branch, f"refs/remotes/origin/{branch}", cwd=clone)
    else:
        s.git("fetch", "--depth", "1", "origin", sha_codigo, cwd=clone)
        s.git("checkout", "-B", branch, "FETCH_HEAD", cwd=clone)


def _copiar(pasta: Path, destino: Path, arquivos: dict[str, dict[str, Any]], imprimir: Callable[[str], None]) -> list[str]:
    destino.mkdir(parents=True, exist_ok=True)
    for relativo, info in arquivos.items():
        alvo = destino / relativo
        if alvo.exists():
            if sha256_arquivo(alvo) != info["sha256"]:
                raise ErroVersionamento(f"{alvo.name} já existe no clone com conteúdo diferente: nada é sobrescrito. "
                                        "Confira a execução e a branch")
            imprimir(f"  {relativo}: igual ao que já está no clone")
            continue
        temporario = alvo.with_name(alvo.name + ".copiando")
        shutil.copyfile(pasta / relativo, temporario)
        os.replace(temporario, alvo)
        if sha256_arquivo(alvo) != info["sha256"]:
            raise ErroVersionamento(f"a cópia de {relativo} não confere com o original")
        imprimir(f"  {relativo}: copiado")
    return list(arquivos)


def salvar(pasta: str | Path, *, repositorio: str, branch: str, sha_codigo: str, clone: str | Path,
           nome_git: str, email_git: str, credencial: segredos.Credencial, confirmacao: str | None = None,
           executor: Executor = executor_padrao, imprimir: Callable[[str], None] = print,
           perfil: str = "execucao") -> dict[str, Any]:
    """Copia os resultados para o clone de trabalho, faz `add -f`, commit, `pull --rebase --autostash` e push."""
    _validar_destino(repositorio, branch)
    _validar_identidade(nome_git, email_git)
    if not re.fullmatch(r"[0-9a-f]{40}", sha_codigo):
        raise ErroVersionamento("SHA do código deve ter 40 caracteres hexadecimais")
    pasta, clone = Path(pasta), Path(clone)
    p = plano_de_salvamento(pasta, perfil)
    if p["problemas"]:
        raise ErroVersionamento("nada foi salvo:\n - " + "\n - ".join(p["problemas"]))
    nome = p["nome"]
    esperado = frase_de_salvamento(nome, repositorio, branch)
    if confirmacao is not None and confirmacao.strip() != esperado:
        raise ErroVersionamento(f"confirmação diferente da esperada ({esperado!r}): nada foi salvo")
    url = f"https://github.com/{repositorio}.git"
    registro = p["registro"]
    caminhos = [caminho_no_repositorio(nome, a) for a in p["arquivos"]]
    imprimir(f"\n=== Salvando {nome} em {repositorio}@{branch} ===")
    with _askpass(credencial) as env:
        s = _Sessao(executor=executor, env=env, identidade=_identidade_do_git(nome_git, email_git),
                    credencial=credencial, imprimir=imprimir)
        remoto_existe = s.git("ls-remote", "--exit-code", "--heads", url, branch, aceitar=(0, 2)) != "2"
        _preparar_clone(s, url=url, clone=clone, branch=branch, sha_codigo=sha_codigo, remoto_existe=remoto_existe)
        imprimir("\nCopiando os resultados:")
        _copiar(pasta, clone / BASE_DESTINO / nome, p["arquivos"], imprimir)
        s.git("add", "-f", "--", *caminhos, cwd=clone)
        preparados = [l for l in s.git("diff", "--cached", "--name-only", cwd=clone).splitlines() if l]
        fora = [a for a in preparados if a not in caminhos]
        if fora:
            raise ErroVersionamento("há arquivos preparados no clone que não são desta execução (" + ", ".join(fora)
                                    + "): nada foi commitado. Remova o clone e rode de novo")
        proibidos = [a for a in preparados if any(r.search(a) for r in _PROIBIDOS)]
        if proibidos:
            raise ErroVersionamento("arquivos proibidos preparados: " + ", ".join(proibidos))
        novo_commit = bool(preparados)
        if novo_commit:
            ident = (f"run_id: {registro['run_id']}\nconfig_id: {registro['config_id']}\n" if registro else "")
            mensagem = (f"Resultados de {nome} (DEV)\n\n{ident}"
                        f"código: {sha_codigo}\nSomente DEV: sem TEST, pesos nem checkpoints.")
            s.git("commit", "-m", mensagem, cwd=clone)
        else:
            imprimir("Nada novo para commitar: os arquivos já estão no clone.")
        if remoto_existe:
            s.git("pull", "--rebase", "--autostash", "origin", branch, cwd=clone)
        s.git("push", "-u", "origin", branch, cwd=clone)
        commit = s.git("rev-parse", "HEAD", cwd=clone)
    resultado = {"nome": nome, "repositorio": repositorio, "branch": branch, "commit": commit,
                 "novo_commit": novo_commit, "arquivos": caminhos,
                 "url": f"https://github.com/{repositorio}/tree/{branch}/{caminho_no_repositorio(nome)}"}
    imprimir("\nResultado:\n" + segredos.limpar(json.dumps(resultado, ensure_ascii=False, indent=2), credencial))
    return resultado
