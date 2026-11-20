"""Congelamento da configuração final e trava da avaliação do TEST.

A avaliação do TEST científico (o das partições congeladas do projeto) só é
aceita para **uma** execução, registrada antes em um arquivo de congelamento, e
só com o código de liberação desse registro. Sem o registro, com outra
execução, com a execução alterada depois do congelamento ou com o código
errado, a avaliação é recusada **antes** de qualquer leitura da partição.

O que a trava garante:

* nenhuma avaliação acidental: quem chama `avaliar_test` precisa apontar o
  registro e digitar o código, que só existe depois do congelamento;
* uma configuração só: o registro guarda as impressões digitais da execução
  (configuração, predições do DEV, pesos do melhor modelo, limiar) e a
  avaliação recusa qualquer divergência;
* uma vez só: a trilha final fica ao lado do registro; uma segunda avaliação
  exige `reavaliar=True` e fica registrada.

O que ela **não** substitui: a autorização do autor. O código de liberação é
derivado do registro; a regra do projeto continua sendo usá-lo apenas depois
da autorização explícita.

As partições técnicas dos testes automatizados (subconjuntos com MANIFEST
próprio, em pastas temporárias) não são o TEST científico e não passam pela
trava: `test_cientifico` compara o hash do TEST registrado no MANIFEST da pasta
com o dos MANIFEST oficiais (v1 e v2). Só os MANIFEST são lidos; o arquivo do
TEST não é aberto por este módulo.

Este módulo não importa torch e não lê partições.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from reclin.util import caminhos, identidade_codigo
from reclin.util.io import gravar_json, ler_json, ler_jsonl, sha256_arquivo

PASTA = caminhos.RESULTADOS / "congelamento"
ARQUIVO = "configuracao_final.json"
TRILHA = "avaliacoes_test_final.jsonl"
PASTA_CHECKPOINTS, MELHOR_MODELO = "checkpoints", "melhor_modelo"     # os de reclin.treino.checkpoint (que importa torch)
VERSAO = 1
# Arquivos da execução cujas impressões digitais entram no congelamento, quando existem.
ARQUIVOS_DA_EXECUCAO = ("config.json", "predicoes_dev.json", "limiar.json")
MANIFESTO = "MANIFEST.json"


class ErroCongelamento(ValueError):
    """A avaliação do TEST foi recusada pela trava de congelamento."""


def caminho_padrao() -> Path:
    return PASTA / ARQUIVO


# ------------------------------------------------------------------ o TEST científico

def _sha_test_do_manifesto(pasta: Path) -> str | None:
    manifesto = pasta / MANIFESTO
    if not manifesto.is_file():
        return None
    return ((ler_json(manifesto).get("splits") or {}).get("test") or {}).get("sha256")


def test_cientifico(pasta_particoes: str | Path,
                    oficiais: tuple[Path, ...] | None = None) -> bool:
    """A pasta contém o TEST das partições congeladas do projeto (v1 ou v2)?

    Verdadeiro quando a pasta é uma das oficiais ou quando o hash do TEST no
    seu MANIFEST é o de uma delas (uma cópia, por exemplo no Colab). Uma pasta
    sem MANIFEST legível é tratada como científica: na dúvida, a trava vale."""
    pasta = Path(pasta_particoes)
    oficiais = oficiais if oficiais is not None else (caminhos.PARTICOES_V2, caminhos.PARTICOES_V1)
    if any(pasta.resolve() == o.resolve() for o in oficiais):
        return True
    sha = _sha_test_do_manifesto(pasta)
    if sha is None:
        return True
    return sha in {s for s in (_sha_test_do_manifesto(o) for o in oficiais) if s}


# ------------------------------------------------------------------ o registro

def impressoes_da_execucao(pasta: str | Path) -> dict[str, Any]:
    """As impressões digitais de uma execução: só registros e predições do DEV."""
    pasta = Path(pasta)
    if not (pasta / "config.json").is_file():
        raise ErroCongelamento(f"{pasta} não é uma execução (sem config.json)")
    arquivos = {nome: sha256_arquivo(pasta / nome) for nome in ARQUIVOS_DA_EXECUCAO if (pasta / nome).is_file()}
    treino = ler_json(pasta / "treino.json") if (pasta / "treino.json").is_file() else {}
    melhor = pasta / PASTA_CHECKPOINTS / MELHOR_MODELO
    if melhor.is_dir():                    # os pesos presentes: o hash é calculado, não só lido do registro
        from reclin import pretreino       # sem torch
        sha_modelo = pretreino.hash_pasta(melhor)
    else:
        sha_modelo = (treino.get("final") or {}).get("sha256_melhor_modelo")
    return {"pasta": pasta.name, "arquivos_sha256": arquivos, "sha256_melhor_modelo": sha_modelo}


def _canonico(registro: Mapping[str, Any]) -> bytes:
    corpo = {k: v for k, v in registro.items() if k != "liberacao"}
    return json.dumps(corpo, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def codigo_de_liberacao(registro: Mapping[str, Any]) -> str:
    """O código que precisa ser informado para avaliar o TEST desta configuração."""
    return hashlib.sha256(_canonico(registro)).hexdigest()[:12]


def criar(pasta_execucao: str | Path, *, responsavel: str, justificativa: str,
          pasta_particoes: str | Path = caminhos.PARTICOES,
          descricao: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """O registro de congelamento de uma execução. Não lê o TEST.

    Recusa uma execução cujo TEST já foi avaliado, cujo treino não terminou ou
    que não tem predições do DEV."""
    pasta = Path(pasta_execucao)
    if not responsavel.strip() or not justificativa.strip():
        raise ErroCongelamento("o congelamento exige o responsável e a justificativa da escolha")
    impressoes = impressoes_da_execucao(pasta)
    if "predicoes_dev.json" not in impressoes["arquivos_sha256"]:
        raise ErroCongelamento(f"{pasta.name} não tem predicoes_dev.json: não há como registrar a seleção no DEV")
    if (pasta / "predicoes_test.json").exists():
        raise ErroCongelamento(f"o TEST de {pasta.name} já foi avaliado: uma execução avaliada não é congelada")
    treino = ler_json(pasta / "treino.json") if (pasta / "treino.json").is_file() else None
    if treino is not None and (not treino.get("concluido") or treino.get("test_avaliado")):
        raise ErroCongelamento(f"{pasta.name}: treino não concluído ou TEST marcado como avaliado em treino.json")
    if treino is not None and not impressoes["sha256_melhor_modelo"]:
        raise ErroCongelamento(
            f"{pasta.name} não tem {PASTA_CHECKPOINTS}/{MELHOR_MODELO}: congele na máquina em que os pesos do "
            "melhor modelo estão, para que o registro guarde o hash deles")
    config = ler_json(pasta / "config.json")
    manifesto = Path(pasta_particoes) / MANIFESTO
    return {
        "versao_do_registro": VERSAO,
        "congelado_em": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "responsavel": responsavel.strip(),
        "justificativa": justificativa.strip(),
        "execucao": impressoes,
        "estrategia": config.get("estrategia"),
        "nome_registrado": config.get("nome"),
        "codigo_da_execucao": (config.get("codigo") or {}).get("identidade"),
        "codigo_no_congelamento": identidade_codigo.atual()["identidade"],
        "particoes": {"manifest_sha256": sha256_arquivo(manifesto) if manifesto.is_file() else None,
                      "test_sha256_registrado_no_manifest": _sha_test_do_manifesto(Path(pasta_particoes))},
        "descricao": dict(descricao or {}),
        "regra": "o TEST é avaliado uma única vez, só para esta execução, depois da autorização explícita do autor",
    }


def gravar(registro: Mapping[str, Any], caminho: str | Path | None = None) -> Path:
    """Grava o registro. Recusa substituir um congelamento existente."""
    caminho = Path(caminho) if caminho is not None else caminho_padrao()
    if caminho.exists():
        raise ErroCongelamento(f"já existe um congelamento em {caminho}: ele não é substituído por este código")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    gravar_json(caminho, {**registro, "liberacao": {"codigo": codigo_de_liberacao(registro),
                                                    "uso": "informe em --liberacao apenas depois da autorização do autor"}})
    return caminho


# ------------------------------------------------------------------ a trava

def _trilha(caminho_registro: Path) -> Path:
    return caminho_registro.parent / TRILHA


def avaliacoes_registradas(caminho_registro: str | Path) -> list[dict[str, Any]]:
    trilha = _trilha(Path(caminho_registro))
    return list(ler_jsonl(trilha)) if trilha.exists() else []


def sha_de_test_cientifico(sha: str | None, oficiais: tuple[Path, ...] | None = None) -> bool:
    """O hash é o do TEST de uma das partições oficiais? Sem hash, vale a trava."""
    oficiais = oficiais if oficiais is not None else (caminhos.PARTICOES_V2, caminhos.PARTICOES_V1)
    return sha is None or sha in {s for s in (_sha_test_do_manifesto(o) for o in oficiais) if s}


def exigir_liberacao(pasta_execucao: str | Path, pasta_particoes: str | Path | None, *,
                     congelamento: str | Path | None = None, liberacao: str | None = None,
                     reavaliar: bool = False, sha_test: str | None = None) -> dict[str, Any] | None:
    """Recusa a avaliação do TEST científico sem congelamento e liberação válidos.

    Devolve o registro de congelamento, ou `None` quando a pasta de partições é
    técnica (testes automatizados). Deve ser a primeira instrução de toda
    função que lê o TEST. Quem não recebe a pasta das partições (o baseline
    calibrado, que parte das predições da base) informa `sha_test`, o hash do
    TEST registrado pela base."""
    cientifico = test_cientifico(pasta_particoes) if pasta_particoes is not None \
        else sha_de_test_cientifico(sha_test)
    if not cientifico:
        return None
    pasta = Path(pasta_execucao)
    caminho = Path(congelamento) if congelamento is not None else caminho_padrao()
    if not caminho.is_file():
        raise ErroCongelamento(
            "avaliação do TEST recusada: não há configuração final congelada "
            f"({caminho}). Congele a execução escolhida com scripts/congelar.py e, depois da "
            "autorização do autor, informe --liberacao.")
    registro = ler_json(caminho)
    esperado = codigo_de_liberacao(registro)
    if (registro.get("liberacao") or {}).get("codigo") != esperado:
        raise ErroCongelamento(f"avaliação do TEST recusada: o registro {caminho.name} foi alterado depois de gravado")
    if registro["execucao"]["pasta"] != pasta.name:
        raise ErroCongelamento(
            f"avaliação do TEST recusada: a configuração congelada é {registro['execucao']['pasta']!r}, "
            f"não {pasta.name!r}. O TEST só é avaliado para a configuração final.")
    atuais = impressoes_da_execucao(pasta)
    if atuais != registro["execucao"]:
        raise ErroCongelamento(
            f"avaliação do TEST recusada: a execução {pasta.name} mudou depois do congelamento "
            "(configuração, predições do DEV, limiar ou pesos do melhor modelo diferem do registro)")
    if not liberacao or liberacao.strip() != esperado:
        raise ErroCongelamento(
            "avaliação do TEST recusada: código de liberação ausente ou incorreto. O código está no "
            f"registro {caminho.name} e só deve ser usado depois da autorização explícita do autor.")
    anteriores = avaliacoes_registradas(caminho)
    if anteriores and not reavaliar:
        raise ErroCongelamento(
            f"avaliação do TEST recusada: a configuração final já foi avaliada em {anteriores[0].get('timestamp')}. "
            "Uma nova avaliação exige reavaliar e fica registrada na trilha final.")
    return {**registro, "_caminho": str(caminho)}


def registrar_avaliacao(registro: Mapping[str, Any] | None, *, macro_f1: float, negation_of_f1: float,
                        n_test: int) -> dict[str, Any] | None:
    """Acrescenta a avaliação à trilha final (ao lado do registro). Sem registro
    (partições técnicas), não faz nada."""
    if registro is None:
        return None
    caminho = Path(registro["_caminho"])
    anteriores = avaliacoes_registradas(caminho)
    linha = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "indice": len(anteriores) + 1,
             "execucao": registro["execucao"]["pasta"], "liberacao": (registro.get("liberacao") or {}).get("codigo"),
             "n_test": int(n_test), "test_macro_f1": round(float(macro_f1), 6),
             "test_negation_of_f1": round(float(negation_of_f1), 6),
             "codigo": identidade_codigo.atual()["identidade"]}
    with open(_trilha(caminho), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(linha, ensure_ascii=False) + "\n")
    return linha
