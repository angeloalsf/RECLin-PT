"""Inventário das execuções do código novo, com a verificação de proveniência
que as análises e os relatórios exigem antes de ler qualquer número.

Só entram execuções no formato do código novo (`<raiz>/<nome>/config.json`,
`execucao.diretorio`). Qualquer outra coisa encontrada na raiz é recusada e
listada com o motivo: sidecars soltos e arquivos com os nomes do legado
(`*.preds.json`, `*.dev_preds.json`, `*.test_evals.jsonl`,
`significance_*.json`, `CALIBRACAO_filtro.*`, `FASE2_test_summary.json`,
`summary_by_seed.json`), pastas sem `config.json` e qualquer caminho dentro de
`codigo/testes/referencia/` (as referências do legado, que só os testes de
equivalência leem). Não existe opção para aceitar esses arquivos.

Para cada execução, `inspecionar` confere, sem treinar nem avaliar nada:

* cada sidecar gravado (`predicoes_dev.json`, `predicoes_test.json`) tem o
  formato válido e corresponde ao conjunto de referência registrado no
  `config.json` (número de candidatos, `y_true` pelo hash e `max_gap`);
* o SHA-256 da partição desse conjunto é o do MANIFEST da pasta de partições
  informada (por padrão, as congeladas);
* as métricas gravadas em `metricas.json`, quando existem, são exatamente as
  recalculadas do sidecar (`avaliacao.metricas.avaliar`);
* um TEST com predições tem a trilha `avaliacoes_test.jsonl` (sem ela, as
  predições do TEST não foram avaliadas pelo comando próprio e não são usadas);
  mais de uma linha na trilha vira aviso de reavaliação;
* uma execução treinada tem o `treino.json`, e o treino está concluído;
* todas as partições que a execução usou (treino e conjuntos de referência)
  são de uma só versão dos dados (`reclin.versao_dados`); a versão é herdada
  pelas derivadas e precisa ser a mesma das dependências;
* a identidade do código (D6, `util.identidade_codigo`) está no `config.json`,
  é consistente com o manifesto gravado e é a mesma nas sessões do treino e
  nas linhas da trilha do TEST (uma execução com partes de versões diferentes
  do código tem problema);
* nenhuma configuração traz um arquivo de calibração externo (`calibracao`,
  o modo de reprodução do legado do filtro e da regra pura), cuja proveniência
  não é verificável;
* o filtro de pistas aponta, em cada sidecar, o SHA-256 das predições da base
  de onde veio (`base_sha256`), igual ao do sidecar da base no inventário.

E classifica a execução em uma de três naturezas, que nunca se misturam numa
mesma tabela (`Inventario.selecionar`):

* `validacao` — execução de validação técnica (`execucao.validacao`), própria
  ou herdada de uma dependência;
* `cientifica` — compatível com a fase atual: semente e encoder da
  configuração padrão (`Config`: 42 e BioBERTpt), proveniência dos pesos
  registrada (C9), partições congeladas da versão dos dados da fase
  (`versao_dados.VERSAO_DA_FASE`, a v2) e produzida pela versão de referência
  do código (por padrão, a atual);
* `fora_da_fase` — válida, mas fora dessas condições (o modelo minúsculo dos
  testes, outra semente, outras partições, outra versão do código, ou sem a
  identidade do código — as execuções anteriores à D6, que nunca são aceitas
  como equivalentes às novas); os motivos ficam em `motivos_fora_da_fase`.

As dependências de execuções derivadas (o filtro de pistas e o baseline
calibrado sobre uma base; a composição do Two-Stage sobre dois estágios) são
lidas do `config.json` e precisam estar no mesmo inventário, válidas e com a
mesma natureza.

Uma seleção para um relatório (`Inventario.selecionar`) exige a mesma
natureza, a mesma semente, a mesma proveniência dos pesos e a mesma identidade
do código em todas as execuções.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from reclin import particoes, versao_dados
from reclin.avaliacao import metricas
from reclin.config import Config
from reclin.execucao import diretorio, predicoes, trilha, validacao
from reclin.tarefa import MAX_GAP
from reclin.util import caminhos, identidade_codigo
from reclin.util.io import ler_json, sha256_arquivo, sha256_json

NATUREZAS = ("cientifica", "validacao", "fora_da_fase")
REFERENCIAS_DO_LEGADO = caminhos.CODIGO / "testes" / "referencia"
PADROES_LEGADO = ("*.preds.json", "*.dev_preds.json", "*.test_evals.jsonl", "significance_*.json",
                  "CALIBRACAO_filtro.*", "FASE2_test_summary.json", "summary_by_seed.json")
ARQUIVO_TREINO = "treino.json"

# Como cada estratégia derivada registra as execuções de que depende (o
# formato gravado pelas estratégias; ver `scripts/pretreino.py conferir`).
DEPENDENCIAS = {
    "filtro_pistas": lambda r: [r["base"]],
    "baseline_calibrado": lambda r: [r["base"]["nome"]],
    "two_stage": lambda r: [r["estagios"]["1"]["nome"], r["estagios"]["2"]["nome"]],
}

# As derivadas que gravam a marca de validação herdada no próprio config.json
# (`validacao.herdar`); o filtro não grava e herda a marca da base aqui.
REGISTRAM_MARCA = ("baseline_calibrado", "two_stage")


class ErroInventario(ValueError):
    """Uma seleção pedida não pode ser usada (ausente, inválida ou misturada)."""


@dataclass
class Execucao:
    nome: str
    pasta: Path
    estrategia: str
    registro: dict[str, Any]
    seed: int | None = None
    encoder: str | None = None
    proveniencia: dict[str, Any] | None = None
    marca_validacao: dict[str, Any] | None = None
    dependencias: list[str] = field(default_factory=list)
    sidecars: dict[str, dict[str, Any]] = field(default_factory=dict)
    test_avaliacoes: int = 0
    treino: dict[str, Any] | None = None
    problemas: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    natureza: str = "fora_da_fase"
    motivos_fora_da_fase: list[str] = field(default_factory=list)
    codigo: str | None = None                  # identidade do código (D6)
    versao_dados: str | None = None            # versão dos dados (`reclin.versao_dados`)

    @property
    def valida(self) -> bool:
        return not self.problemas

    def tem(self, particao: str) -> bool:
        """Há predições utilizáveis da partição (o TEST só se foi avaliado)."""
        return particao in self.sidecars and (particao != "test" or self.test_avaliacoes > 0)

    def predicoes(self, particao: str) -> dict[str, Any]:
        """O sidecar da partição, conferido contra o hash lido no inventário."""
        if not self.tem(particao):
            raise ErroInventario(f"{self.nome}: sem predições utilizáveis de {particao}")
        caminho = self.pasta / diretorio.arquivo_predicoes(particao)
        if sha256_arquivo(caminho) != self.sidecars[particao]["sha256"]:
            raise ErroInventario(f"{caminho} mudou depois do inventário")
        return predicoes.ler(caminho)

    def origem(self) -> dict[str, Any]:
        """O registro de proveniência que acompanha cada número publicado."""
        return {"execucao": self.nome, "estrategia": self.estrategia, "natureza": self.natureza,
                "seed": self.seed, "encoder": self.encoder,
                "proveniencia_pesos": self.proveniencia, "dependencias": list(self.dependencias),
                "sidecars": {p: {"sha256": s["sha256"], "particao_sha256": s["particao_sha256"]}
                             for p, s in sorted(self.sidecars.items())},
                "avaliacoes_test": self.test_avaliacoes, "codigo": self.codigo,
                "versao_dados": self.versao_dados,
                "sha256_config": sha256_arquivo(self.pasta / diretorio.CONFIG)}


def _dentro(caminho: Path, pasta: Path) -> bool:
    try:
        caminho.resolve().relative_to(pasta.resolve())
        return True
    except ValueError:
        return False


def recusar_caminho(caminho: str | Path) -> str | None:
    """O motivo para recusar `caminho` como entrada de análise, ou None."""
    caminho = Path(caminho)
    if _dentro(caminho, REFERENCIAS_DO_LEGADO):
        return "dentro de codigo/testes/referencia/ (referências do legado, só para os testes de equivalência)"
    if caminho.is_file() and any(fnmatch.fnmatch(caminho.name, p) for p in PADROES_LEGADO):
        return "arquivo com o formato ou o nome dos resultados do legado"
    return None


def _manifesto_sha(pasta_particoes: Path) -> dict[str, str]:
    return {p: s["sha256"] for p, s in particoes.ler_manifesto(pasta_particoes)["splits"].items()}


def _congeladas(pasta_particoes: Path) -> bool:
    """O MANIFEST da pasta é, byte a byte, o da versão dos dados da fase
    (`versao_dados.VERSAO_DA_FASE`, a v2). A v1 é congelada, mas não é a da
    fase: execuções sobre ela ficam fora da fase."""
    return versao_dados.versao_da_pasta(pasta_particoes) == versao_dados.VERSAO_DA_FASE


def inspecionar(pasta: str | Path, *, pasta_particoes: str | Path = caminhos.PARTICOES) -> Execucao:
    """Lê e confere uma execução (ver o módulo). Não levanta exceção por
    problema de conteúdo: os problemas ficam em `Execucao.problemas`."""
    pasta = Path(pasta)
    motivo = recusar_caminho(pasta)
    if motivo:
        raise ErroInventario(f"{pasta}: {motivo}")
    if not (pasta / diretorio.CONFIG).is_file():
        raise ErroInventario(f"{pasta} não é uma execução do código novo (sem {diretorio.CONFIG})")
    conteudo = diretorio.ler_config(pasta)
    registro = conteudo.get("config") or {}
    estrategia = conteudo.get("estrategia") or registro.get("estrategia") or "?"
    ex = Execucao(nome=conteudo.get("nome", pasta.name), pasta=pasta, estrategia=estrategia, registro=registro)
    if ex.nome != pasta.name:
        ex.problemas.append(f"o config.json chama a execução de {ex.nome!r}, e a pasta é {pasta.name!r}")
    ex.marca_validacao = validacao.da_config(registro)
    registro_codigo = conteudo.get("codigo")
    if registro_codigo:
        ex.codigo = registro_codigo.get("identidade")
        ex.problemas.extend(f"identidade do código: {p}" for p in identidade_codigo.conferir_registro(registro_codigo))
    if "calibracao" in registro:
        ex.problemas.append("configuração com um arquivo de calibração externo (modo de reprodução do legado): "
                            "proveniência não verificável")
    modelo = registro.get("modelo")
    if isinstance(modelo, str) and _dentro(Path(modelo), REFERENCIAS_DO_LEGADO):
        ex.motivos_fora_da_fase.append("pesos das referências do legado (modelo minúsculo dos testes)")

    config = registro.get("config") if isinstance(registro.get("config"), dict) else None
    if config is not None:                                    # execução treinada
        ex.seed, ex.encoder = config.get("seed"), config.get("encoder")
    else:
        ex.seed = registro.get("seed")
    ex.proveniencia = registro.get("proveniencia_pesos")
    if estrategia in DEPENDENCIAS:
        try:
            ex.dependencias = list(DEPENDENCIAS[estrategia](registro))
        except (KeyError, TypeError):
            ex.problemas.append(f"config.json de {estrategia} sem as dependências registradas")

    versoes = versao_dados.versoes_do_registro(conteudo)
    ex.problemas.extend(versao_dados.conferir_uma_versao(versoes))
    ex.versao_dados = versao_dados.versao_unica(versoes)
    for particao, gravada in (conteudo.get("versoes_dados") or {}).items():
        if gravada != versoes.get(f"conjunto.{particao}"):
            ex.problemas.append(f"versoes_dados[{particao}] = {gravada!r} não corresponde ao SHA-256 da partição "
                                "do conjunto de referência")

    shas = _manifesto_sha(Path(pasta_particoes))
    for particao_treino, sha in (registro.get("particoes") or {}).items():
        if shas.get(particao_treino) != sha:
            ex.problemas.append(f"a partição {particao_treino} do treino não é a do MANIFEST de {pasta_particoes}")

    gravadas = {}
    if (pasta / diretorio.METRICAS).is_file():
        gravadas = diretorio.ler_metricas(pasta)
    conjuntos = conteudo.get("conjuntos") or {}
    for particao in diretorio.PARTICOES:
        arquivo = pasta / diretorio.arquivo_predicoes(particao)
        if not arquivo.is_file():
            continue
        try:
            sidecar = predicoes.ler(arquivo)
        except (ValueError, KeyError) as erro:
            ex.problemas.append(f"{arquivo.name}: {erro}")
            continue
        conjunto = conjuntos.get(particao)
        if not conjunto:
            ex.problemas.append(f"{arquivo.name} sem o conjunto de referência no config.json")
            continue
        divergencias = predicoes.conferir_conjunto(sidecar, conjunto)
        if conjunto.get("max_gap") != MAX_GAP:
            divergencias.append(f"max_gap {conjunto.get('max_gap')} diferente de {MAX_GAP}")
        if conjunto.get("particao_sha256") != shas.get(particao):
            divergencias.append(f"a partição {particao} do conjunto não é a do MANIFEST de {pasta_particoes}")
        if particao in gravadas and gravadas[particao] != metricas.avaliar(sidecar["y_true"], sidecar["y_pred"]):
            divergencias.append(f"metricas.json[{particao}] não é o recalculado das predições")
        if divergencias:
            ex.problemas.extend(f"{arquivo.name}: {d}" for d in divergencias)
            continue
        ex.sidecars[particao] = {"sha256": sha256_arquivo(arquivo), "n": len(sidecar["y_true"]),
                                 "particao_sha256": conjunto["particao_sha256"],
                                 "y_true_sha256": sha256_json(sidecar["y_true"]),
                                 "base_sha256": sidecar.get("base_sha256")}

    linhas = trilha.ler(pasta / diretorio.TRILHA)
    ex.test_avaliacoes = len(linhas)
    if "test" in ex.sidecars and not linhas:
        ex.avisos.append("predicoes_test.json sem trilha de avaliação: o TEST não é usado")
    if linhas and "test" not in ex.sidecars:
        ex.problemas.append("trilha de avaliação do TEST sem predicoes_test.json válido")
    if len(linhas) > 1:
        ex.avisos.append(f"o TEST desta execução foi avaliado {len(linhas)} vezes (reavaliação registrada na trilha)")
    if ex.codigo and any(l.get("codigo") != ex.codigo for l in linhas):
        ex.problemas.append("a trilha do TEST tem avaliações de outra versão do código (ou sem a identidade)")

    if config is not None and not (pasta / ARQUIVO_TREINO).is_file():
        ex.problemas.append("execução treinada sem treino.json")
    if (pasta / ARQUIVO_TREINO).is_file():
        ex.treino = ler_json(pasta / ARQUIVO_TREINO)
        if not ex.treino.get("concluido"):
            ex.problemas.append("treino não concluído")
        sessoes = ex.treino.get("sessoes") or []
        if ex.codigo and any(s.get("codigo") != ex.codigo for s in sessoes):
            ex.problemas.append("o treino tem sessões de outra versão do código (ou sem a identidade)")
    return ex


def _resolver(execucoes: dict[str, Execucao], congeladas: bool, codigo_referencia: str | None) -> None:
    """Herda seed, encoder, proveniência e marca das dependências e classifica."""
    padrao = Config()

    def visitar(ex: Execucao, pilha: tuple[str, ...]) -> None:
        if ex.nome in pilha:
            ex.problemas.append("dependência circular: " + " -> ".join(pilha + (ex.nome,)))
            return
        for dep in ex.dependencias:
            base = execucoes.get(dep)
            if base is None:
                ex.problemas.append(f"depende de {dep}, que não está no inventário")
                continue
            visitar(base, pilha + (ex.nome,))
            if not base.valida:
                ex.problemas.append(f"depende de {dep}, que tem problemas")
            if ex.encoder is None:
                ex.encoder = base.encoder
            if ex.seed is None:
                ex.seed = base.seed
            elif base.seed is not None and base.seed != ex.seed:
                ex.problemas.append(f"semente {ex.seed} diferente da de {dep} ({base.seed})")
            if ex.proveniencia is None:
                ex.proveniencia = base.proveniencia
            elif base.proveniencia is not None and base.proveniencia != ex.proveniencia:
                ex.problemas.append(f"proveniência dos pesos diferente da de {dep}")
            if base.marca_validacao and not ex.marca_validacao and ex.estrategia not in REGISTRAM_MARCA:
                ex.marca_validacao = {"origens": {dep: base.marca_validacao},
                                      "rotulo": "derivada de execuções de validação técnica"}
            if bool(base.marca_validacao) != bool(ex.marca_validacao):
                ex.problemas.append(f"mistura execução de validação e da fase científica ({dep})")
            if ex.codigo != base.codigo:
                ex.problemas.append(f"identidade do código diferente da de {dep}")
            if ex.versao_dados is None:
                ex.versao_dados = base.versao_dados
            elif base.versao_dados is not None and base.versao_dados != ex.versao_dados:
                ex.problemas.append(f"versão dos dados {ex.versao_dados} diferente da de {dep} "
                                    f"({base.versao_dados})")
            if ex.estrategia == "filtro_pistas":
                for particao, s in ex.sidecars.items():
                    origem = base.sidecars.get(particao, {}).get("sha256")
                    if s.get("base_sha256") is None:
                        ex.problemas.append(f"predicoes_{particao}.json do filtro sem o SHA-256 das predições "
                                            "da base (base_sha256): origem não verificável")
                    elif s["base_sha256"] != origem:
                        ex.problemas.append(f"predicoes_{particao}.json do filtro não veio das predições "
                                            f"atuais de {dep}")

    for ex in execucoes.values():
        visitar(ex, ())
    for ex in execucoes.values():
        motivos = list(ex.motivos_fora_da_fase)
        if ex.codigo is None:
            motivos.append("sem identidade do código (execução anterior à D6)")
        elif ex.codigo != codigo_referencia:
            motivos.append(f"produzida por outra versão do código ({ex.codigo[:12]}…; referência "
                           f"{(codigo_referencia or '-')[:12]}…)")
        if ex.seed != padrao.seed:
            motivos.append(f"semente {ex.seed} (a fase usa {padrao.seed})")
        if ex.encoder != padrao.encoder:
            motivos.append(f"encoder {ex.encoder} (a fase usa {padrao.encoder})")
        if not ex.proveniencia:
            motivos.append("sem proveniência dos pesos registrada (C9)")
        if not congeladas:
            motivos.append("partições diferentes das congeladas")
        if ex.versao_dados is not None and ex.versao_dados != versao_dados.VERSAO_DA_FASE:
            motivos.append(f"dados da versão {ex.versao_dados} (a fase usa {versao_dados.VERSAO_DA_FASE})")
        ex.motivos_fora_da_fase = motivos
        ex.natureza = "validacao" if ex.marca_validacao else ("fora_da_fase" if motivos else "cientifica")
    # remove duplicatas mantendo a ordem
    for ex in execucoes.values():
        ex.problemas = list(dict.fromkeys(ex.problemas))


@dataclass
class Inventario:
    raiz: Path
    pasta_particoes: Path
    particoes_congeladas: bool
    execucoes: dict[str, Execucao]
    recusados: list[dict[str, str]]
    codigo_referencia: str | None = None

    def obter(self, nome: str) -> Execucao:
        if nome not in self.execucoes:
            raise ErroInventario(f"a execução {nome} não está em {self.raiz}")
        return self.execucoes[nome]

    def selecionar(self, nomes: Iterable[str], *, natureza: str = "cientifica",
                   ausentes: str = "pular") -> list[Execucao]:
        """As execuções pedidas, todas válidas, da mesma natureza e com a mesma
        semente e proveniência. `ausentes="pular"` ignora nomes que não estão
        no inventário (as tabelas marcam esses sistemas como pendentes);
        `"falhar"` recusa."""
        if natureza not in NATUREZAS:
            raise ValueError(f"natureza {natureza!r}; opções: {NATUREZAS}")
        escolhidas = []
        for nome in nomes:
            if nome not in self.execucoes:
                if ausentes == "falhar":
                    raise ErroInventario(f"a execução {nome} não está em {self.raiz}")
                continue
            ex = self.execucoes[nome]
            if not ex.valida:
                raise ErroInventario(f"{nome} tem problemas: " + "; ".join(ex.problemas))
            if ex.natureza != natureza:
                raise ErroInventario(f"{nome} é de natureza {ex.natureza!r}, não {natureza!r}: "
                                     "naturezas diferentes não entram no mesmo relatório")
            escolhidas.append(ex)
        sementes = {ex.seed for ex in escolhidas}
        if len(sementes) > 1:
            raise ErroInventario(f"sementes diferentes na mesma seleção: {sorted(map(str, sementes))}")
        provs = {sha256_json(ex.proveniencia) for ex in escolhidas if ex.proveniencia}
        if len(provs) > 1:
            raise ErroInventario("proveniências de pesos diferentes na mesma seleção")
        codigos = {ex.codigo for ex in escolhidas}
        if len(codigos) > 1:
            raise ErroInventario("versões diferentes do código na mesma seleção")
        versoes = {ex.versao_dados for ex in escolhidas}
        if len(versoes) > 1:
            raise ErroInventario(f"versões diferentes dos dados na mesma seleção: {sorted(map(str, versoes))}")
        return escolhidas

    def resumo(self) -> dict[str, Any]:
        return {
            "raiz": caminhos.relativo_ao_codigo(self.raiz),
            "particoes": caminhos.relativo_ao_codigo(self.pasta_particoes),
            "particoes_congeladas": self.particoes_congeladas,
            "versao_dados": versao_dados.descrever(self.pasta_particoes)["versao"],
            "codigo_referencia": self.codigo_referencia,
            "execucoes": {n: {"estrategia": e.estrategia, "natureza": e.natureza, "valida": e.valida,
                              "codigo": e.codigo, "versao_dados": e.versao_dados,
                              "particoes": sorted(p for p in e.sidecars if e.tem(p)),
                              "seed": e.seed, "encoder": e.encoder, "dependencias": e.dependencias,
                              "problemas": e.problemas, "avisos": e.avisos,
                              "motivos_fora_da_fase": e.motivos_fora_da_fase}
                          for n, e in sorted(self.execucoes.items())},
            "recusados": self.recusados,
        }


def inventariar(raiz: str | Path = caminhos.RESULTADOS / "execucoes", *,
                pasta_particoes: str | Path = caminhos.PARTICOES,
                codigo_referencia: str | None = None) -> Inventario:
    """Todas as execuções diretamente sob `raiz` (uma por pasta) e os itens
    recusados. Uma raiz dentro das referências do legado é recusada.
    `codigo_referencia` é a identidade do código com que as execuções da fase
    científica precisam ter sido produzidas (padrão: a do código atual)."""
    codigo_referencia = codigo_referencia or identidade_codigo.atual()["identidade"]
    raiz, pasta_particoes = Path(raiz), Path(pasta_particoes)
    motivo = recusar_caminho(raiz)
    if motivo:
        raise ErroInventario(f"{raiz}: {motivo}")
    if _dentro(pasta_particoes, REFERENCIAS_DO_LEGADO):
        raise ErroInventario(f"{pasta_particoes}: partições dentro das referências do legado")
    execucoes, recusados = {}, []
    if raiz.is_dir():
        for item in sorted(raiz.iterdir()):
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            motivo = recusar_caminho(item)
            if motivo is None and item.is_file():
                motivo = "arquivo solto: só pastas de execução com config.json são lidas"
            elif motivo is None and not (item / diretorio.CONFIG).is_file():
                motivo = "pasta sem config.json: não é uma execução do código novo"
            if motivo:
                recusados.append({"caminho": item.name, "motivo": motivo})
                continue
            ex = inspecionar(item, pasta_particoes=pasta_particoes)
            execucoes[ex.nome] = ex
    _resolver(execucoes, _congeladas(pasta_particoes), codigo_referencia)
    return Inventario(raiz=raiz, pasta_particoes=pasta_particoes,
                      particoes_congeladas=_congeladas(pasta_particoes),
                      execucoes=execucoes, recusados=recusados, codigo_referencia=codigo_referencia)


def origens(execucoes: Iterable[Execucao]) -> list[dict[str, Any]]:
    return [ex.origem() for ex in execucoes]


def conferir_origens(registradas: Iterable[Mapping[str, Any]], inventario: Inventario) -> list[str]:
    """Divergências entre a proveniência gravada num relatório e o estado
    atual das execuções (sidecars e config.json pelo hash)."""
    problemas = []
    for o in registradas:
        nome = o["execucao"]
        if nome not in inventario.execucoes:
            problemas.append(f"{nome} não está mais no inventário")
            continue
        atual = inventario.execucoes[nome].origem()
        for chave in ("sidecars", "sha256_config", "avaliacoes_test", "natureza", "codigo"):
            if atual[chave] != o.get(chave):
                problemas.append(f"{nome}: {chave} mudou desde a geração")
    return problemas
