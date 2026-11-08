"""Auditoria das execuções gravadas e consolidação dos resultados do DEV (Etapa 3).

Lê só o que as execuções já gravaram (`config.json`, `treino.json`,
`experimento.json`, `metricas.json`, `limiar.json`, `predicoes_dev.json`):
não treina, não roda modelo, não toca o TEST e não altera nenhum arquivo das
execuções. Três partes:

1. **Inventário** (`inventariar`): por pasta, o tipo (execução treinada,
   estágio do Two-Stage, composição do Two-Stage), o encoder e a revisão dos
   pesos, a estratégia, a semente, os hashes das partições, a identidade do
   código e o commit, as épocas, a melhor época, as sessões (nova/retomada), a
   conclusão, `test_avaliado` e o que *não* está no repositório (checkpoints e
   logs, que ficam no Drive). Confere os vínculos entre as composições do
   Two-Stage e os estágios pelos hashes (`sha256_config`, `sha256_melhor_modelo`),
   independentemente do nome da pasta.
2. **Métricas** (`metricas_sistema`): recalcula P, R, F1, suporte, TP, FP e FN
   por classe, macro-F1, MCC e a matriz de confusão a partir de `y_true` e
   `y_pred` gravados, com `reclin.avaliacao.metricas.avaliar`, e confere
   igualdade exata com o `metricas.json` da execução e com os hashes do conjunto
   de referência. Para `associated_with`, a precisão média (AP) a partir das
   probabilidades, quando o sidecar as tem.
3. **Comparações pareadas no DEV** (`comparar_pareado`): diferenças de F1 de
   `negation_of`, de F1 de `associated_with` e de macro-F1, com bootstrap
   pareado por candidato (o do protocolo: `rng.integers(0, n, n)` por
   reamostra, semente 42, 10.000 reamostras; o F1 de `negation_of` sai idêntico
   ao de `significancia.comparar`), McNemar exato e, quando se informa o
   documento de cada candidato, bootstrap por documento (reamostra os
   documentos inteiros, porque os candidatos de um mesmo documento não são
   independentes). **Descritivas:** o DEV escolheu as épocas, τ e t; as
   comparações do protocolo são no TEST, ainda não autorizado.

As saídas têm só números agregados, ids de documentos e identificadores de
execução: nenhum texto clínico.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reclin.avaliacao import metricas as met
from reclin.avaliacao import significancia
from reclin.tarefa import ASSOC, LABELS, NEG, NOREL
from reclin.util.io import sha256_json

N_BOOT = significancia.N_BOOT
SEMENTE = 42

ENCODERS = {
    "pucpr/biobertpt-all": "BioBERTpt-all",
    "pucpr/biobertpt-clin": "BioBERTpt-clin",
    "neuralmind/bert-base-portuguese-cased": "BERTimbau",
}

ROTULO_ESTRATEGIA = {
    "baseline": "Baseline",
    "pair_aware": "Pair-Aware linear",
    "pair_aware_mlp": "Pair-Aware MLP",
    "two_stage": "Two-Stage",
    "two_stage_e1": "Two-Stage, estágio 1",
    "two_stage_e2": "Two-Stage, estágio 2",
    "baseline_calibrado": "Baseline calibrado",
    "filtro_pistas": "Filtro léxico",
}

PAPEL = {
    "pair_aware": "arquitetura oficial",
    "baseline": "referência experimental",
    "two_stage": "estratégia experimental",
    "pair_aware_mlp": "variante experimental",
    "baseline_calibrado": "controle",
    "filtro_pistas": "controle",
}


# --------------------------------------------------------------------------- #
# Leitura                                                                     #
# --------------------------------------------------------------------------- #
def _ler(pasta: Path, nome: str) -> Any:
    caminho = pasta / nome
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.is_file() else None


def sha256_arquivo(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _hash_obj(obj: Any) -> str:
    """O hash de `two_stage.composicao._hash` (JSON canônico, sem espaços)."""
    texto = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def tipo_pasta(pasta: Path) -> str:
    if (pasta / "limiar.json").is_file() and not (pasta / "config.json").is_file():
        return "composicao_two_stage"
    registro = _ler(pasta, "config.json") or {}
    estrategia = registro.get("estrategia")
    if estrategia in ("two_stage_e1", "two_stage_e2"):
        return "estagio_two_stage"
    if estrategia:
        return "execucao"
    return "desconhecido"


def _encoder(repositorio: str | None) -> str | None:
    return ENCODERS.get(repositorio or "", repositorio)


# --------------------------------------------------------------------------- #
# 1. Inventário                                                               #
# --------------------------------------------------------------------------- #
def _resumo_sessoes(treino: Mapping[str, Any]) -> list[dict[str, Any]]:
    saida = []
    for s in treino.get("sessoes") or []:
        de = s.get("a_partir_de") or {}
        saida.append({"tipo": s.get("tipo"), "status": s.get("status"), "inicio": s.get("inicio"),
                      "fim": s.get("fim"), "retomada_do_passo": de.get("passo_global"),
                      "checkpoints_gravados": s.get("checkpoints_gravados"),
                      "identidade_codigo": s.get("codigo")})
    return saida


def ler_execucao(pasta: str | Path) -> dict[str, Any]:
    """O registro de inventário de uma pasta de `resultados/execucoes/`."""
    pasta = Path(pasta)
    tipo = tipo_pasta(pasta)
    reg: dict[str, Any] = {"pasta": pasta.name, "tipo": tipo,
                           "arquivos": sorted(p.name for p in pasta.iterdir() if p.is_file())}
    config = _ler(pasta, "config.json")
    treino = _ler(pasta, "treino.json")
    experimento = _ler(pasta, "experimento.json")
    limiar = _ler(pasta, "limiar.json")
    preds = _ler(pasta, "predicoes_dev.json")

    if config:
        c = config["config"]
        prov = c.get("proveniencia_pesos") or {}
        hp = c.get("config") if isinstance(c.get("config"), dict) else {}
        reg.update({
            "nome_registrado": config.get("nome"),
            "estrategia": config.get("estrategia"),
            "encoder": _encoder(prov.get("repositorio")),
            "pesos": {"repositorio": prov.get("repositorio"), "commit": prov.get("commit"),
                      "hash_manifesto": prov.get("hash_manifesto")},
            "seed": hp.get("seed", c.get("seed")),
            "hiperparametros": {k: hp.get(k) for k in
                                ("epochs", "batch_size", "lr", "max_length", "ctx_chars", "max_gap",
                                 "class_weight", "warmup_ratio", "weight_decay", "max_grad_norm")} if hp
            else {k: c.get(k) for k in ("base", "regra", "lexico_sha1", "min_freq", "n_formas", "max_gap")
                  if k in c},
            "particoes_sha256": c.get("particoes"),
            "versoes_dados": config.get("versoes_dados"),
            "conjunto_dev": (config.get("conjuntos") or {}).get("dev"),
            "identidade_codigo": (config.get("codigo") or {}).get("identidade"),
            "n_arquivos_codigo": (config.get("codigo") or {}).get("n_arquivos"),
            "sha256_config_canonico": _hash_obj(c),
        })
    if treino:
        hist = treino.get("dev_history") or []
        campo = treino.get("campo_selecao") or "dev_macro_f1"
        reg["treino"] = {
            "concluido": treino.get("concluido"),
            "epocas_previstas": (treino.get("posicao") or {}).get("epocas"),
            "epocas_registradas": len(hist),
            "passos_por_epoca": (treino.get("posicao") or {}).get("passos_por_epoca"),
            "n_params": treino.get("n_params"),
            "campo_selecao": campo,
            "historico_selecao": [h.get(campo) for h in hist],
            "melhor_epoca": treino.get("melhor_epoca"),
            "melhor_valor": treino.get("melhor_dev_macro_f1", treino.get("melhor_valor_selecao")),
            "restaurou_melhor_estado": treino.get("restaurou_melhor_estado"),
            "test_avaliado": treino.get("test_avaliado"),
            "n_candidatos": treino.get("n_candidatos"),
            "sessoes": _resumo_sessoes(treino),
            "sha256_melhor_modelo": (treino.get("final") or {}).get("sha256_melhor_modelo"),
            "dev_history": hist,
        }
        # A melhor época registrada é a de maior valor (empate -> a primeira)?
        valores = reg["treino"]["historico_selecao"]
        if valores and all(v is not None for v in valores):
            reg["treino"]["melhor_epoca_confere"] = (
                int(np.argmax(valores)) + 1 == treino.get("melhor_epoca"))
    if experimento:
        cod = experimento.get("codigo") or {}
        ck = experimento.get("checkpoints") or {}
        reg["experimento"] = {
            "estado": experimento.get("estado"), "run_id": experimento.get("run_id"),
            "alias_oficial": experimento.get("alias_oficial"),
            "commit": cod.get("revisao"), "branch": cod.get("branch"),
            "identidade_global": (cod.get("identidade_global") or {}).get("identidade"),
            "historico_de_estados": [(h.get("estado"), h.get("em")) for h in
                                     experimento.get("historico_de_estados") or []],
            "checkpoints": {"drive": (ck.get("drive") or {}).get("destino"),
                            "n_arquivos_drive": (ck.get("drive") or {}).get("n_arquivos"),
                            "hf": ck.get("hf")},
            "publicacoes": experimento.get("publicacoes"),
            "gpu": (experimento.get("ambiente") or {}).get("gpu"),
            "versoes": (experimento.get("ambiente") or {}).get("versoes"),
            "sha256_metricas_registrado": ((experimento.get("metricas") or {}).get("metricas_json") or {})
            .get("sha256"),
        }
        if (pasta / "metricas.json").is_file():
            reg["experimento"]["sha256_metricas_confere"] = (
                sha256_arquivo(pasta / "metricas.json") == reg["experimento"]["sha256_metricas_registrado"])
    if limiar:
        v = limiar.get("vinculos") or {}
        reg.update({
            "estrategia": "two_stage",
            "encoder": _encoder((v.get("proveniencia_pesos") or {}).get("repositorio")),
            "pesos": v.get("proveniencia_pesos"),
            "seed": v.get("seed"),
            "conjunto_dev": v.get("dev"),
            "limiar": {"tau": limiar.get("limiar"), "congelado_em": limiar.get("congelado_em"),
                       "grade": (limiar.get("configuracao_busca") or {}).get("grade"),
                       "criterio": (limiar.get("configuracao_busca") or {}).get("criterio")},
            "estagios": {k: v.get(k) for k in ("estagio1", "estagio2")},
        })
    if preds:
        reg["predicoes"] = {"n": len(preds.get("y_pred") or []),
                            "tem_probabilidades": bool(preds.get("probs")),
                            "modelo": preds.get("model"),
                            "melhor_epoca": preds.get("best_epoch")}
    reg["fora_do_repositorio"] = ["checkpoints (Drive)", "logs de console do Colab"] \
        if tipo in ("execucao", "estagio_two_stage") else []
    return reg


def vincular_estagios(registros: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Para cada composição do Two-Stage, a pasta de cada estágio, achada pelos
    hashes (`sha256_config` canônico e `sha256_melhor_modelo`), não pelo nome."""
    estagios = [r for r in registros if r["tipo"] == "estagio_two_stage"]
    saida = []
    for comp in (r for r in registros if r["tipo"] == "composicao_two_stage"):
        linha: dict[str, Any] = {"composicao": comp["pasta"]}
        for chave in ("estagio1", "estagio2"):
            alvo = comp["estagios"][chave]
            achados = [e["pasta"] for e in estagios
                       if e.get("sha256_config_canonico") == alvo.get("sha256_config")
                       and (e.get("treino") or {}).get("sha256_melhor_modelo") == alvo.get("sha256_melhor_modelo")]
            e = next((x for x in estagios if x["pasta"] in achados), None)
            linha[chave] = {"nome_no_limiar": alvo.get("nome"), "pastas": achados,
                            "nome_confere": alvo.get("nome") in achados,
                            "melhor_epoca_confere": bool(e) and
                            (e.get("treino") or {}).get("melhor_epoca") == alvo.get("melhor_epoca")}
        saida.append(linha)
    return saida


def diferencas_codigo(pastas: Sequence[Path]) -> dict[str, Any]:
    """Para cada par de identidades de código, os arquivos do manifesto que
    diferem (pelo SHA-256 de cada arquivo registrado em `config.json`)."""
    manifestos: dict[str, dict[str, str]] = {}
    for p in pastas:
        codigo = (_ler(p, "config.json") or {}).get("codigo") or {}
        if codigo.get("identidade") and codigo.get("arquivos"):
            manifestos.setdefault(codigo["identidade"], codigo["arquivos"])
    ids = sorted(manifestos)
    pares = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            A, B = manifestos[a], manifestos[b]
            pares[f"{a[:12]}|{b[:12]}"] = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
    return {"identidades": {k[:12]: len(v) for k, v in manifestos.items()}, "arquivos_diferentes": pares}


def inventariar(raiz: str | Path, extras: Sequence[str | Path] = ()) -> dict[str, Any]:
    raiz = Path(raiz)
    pastas = sorted(p for p in raiz.iterdir() if p.is_dir()) + [Path(p) for p in extras]
    registros = [ler_execucao(p) for p in pastas]
    return {"execucoes": registros, "vinculos_two_stage": vincular_estagios(registros),
            "codigo": diferencas_codigo(pastas)}


# --------------------------------------------------------------------------- #
# 2. Métricas                                                                 #
# --------------------------------------------------------------------------- #
def average_precision(y_bin: np.ndarray, score: np.ndarray) -> float:
    """AP como no `sklearn.metrics.average_precision_score` (degraus nos
    limiares distintos, sem interpolação)."""
    ordem = np.argsort(-score, kind="mergesort")
    s, y = score[ordem], y_bin[ordem]
    fim = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tps = np.cumsum(y)[fim]
    fps = (fim + 1) - tps
    if tps[-1] == 0:
        return 0.0
    precisao = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precisao))


def metricas_sistema(pasta: str | Path, conjunto_dev: Mapping[str, Any] | None = None) -> dict[str, Any]:
    pasta = Path(pasta)
    preds = _ler(pasta, "predicoes_dev.json")
    if preds is None:
        raise FileNotFoundError(f"{pasta.name}: sem predicoes_dev.json")
    if list(preds["labels"]) != list(LABELS):
        raise ValueError(f"{pasta.name}: rótulos {preds['labels']} != {LABELS}")
    y_true, y_pred = preds["y_true"], preds["y_pred"]
    m = met.avaliar(y_true, y_pred)
    saida: dict[str, Any] = {
        "pasta": pasta.name,
        "sha256_predicoes": sha256_arquivo(pasta / "predicoes_dev.json"),
        "n": m["n"], "macro_f1": m["macro_f1"], "mcc": m["mcc"], "micro_f1": m["micro_f1"],
        "weighted_f1": m["weighted_f1"], "por_classe": m["por_classe"],
        "matriz_confusao": m["confusion_matrix"]["matrix"],
    }
    gravado = _ler(pasta, "metricas.json")
    if gravado is not None and "dev_diagnostico_estagios" in gravado:
        saida["diagnostico_estagios"] = gravado["dev_diagnostico_estagios"]
    if gravado is not None:
        cr = gravado["dev"]["classification_report"]
        saida["confere_metricas_json"] = cr == m["classification_report"]
        if "confusion_matrix" in gravado["dev"]:
            saida["confere_matriz_gravada"] = gravado["dev"]["confusion_matrix"]["matrix"] == \
                m["confusion_matrix"]["matrix"]
    if conjunto_dev is not None:
        saida["confere_y_true_sha256"] = sha256_json(list(y_true)) == conjunto_dev.get("y_true_sha256")
    if "dev_macro_f1_recomputed" in preds:
        saida["confere_macro_f1_sidecar"] = preds["dev_macro_f1_recomputed"] == m["macro_f1"]
    probs = preds.get("probs")
    if probs and len(probs) == len(y_true) and len(probs[0]) == len(LABELS):
        p = np.asarray(probs, dtype=float)
        yt = np.asarray(y_true)
        saida["ap"] = {r: average_precision((yt == i).astype(int), p[:, i]) for i, r in enumerate(LABELS)}
    if "limiar_tau" in preds:
        saida["tau"] = preds["limiar_tau"]
        saida["n_roteados_estagio2"] = len(preds.get("indices_estagio2") or [])
    if "limiar_t" in preds:
        saida["t"] = preds["limiar_t"]
    return saida


def origem_falsos_positivos_assoc(y_true: Sequence[int], y_pred: Sequence[int],
                                  pares: Sequence[tuple[str, str, str]],
                                  gaps: Sequence[int] | None = None) -> dict[str, Any]:
    """De onde vêm os falsos positivos de `associated_with`.

    `pares[i] = (doc_id, e1_id, e2_id)` do candidato i. Classifica cada FP:
    o verdadeiro é `negation_of`; o par inverso (e2, e1) é `associated_with`
    anotado (erro só de direção); o par inverso é `negation_of` anotado;
    ou nenhum dos dois (par sem relação anotada em nenhuma direção).
    """
    rotulo = {p: y for p, y in zip(pares, y_true)}
    contagem = {"verdadeiro_negation_of": 0, "inverso_associated_with": 0,
                "inverso_negation_of": 0, "sem_relacao_nas_duas_direcoes": 0}
    por_gap: dict[str, list[int]] = {}
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if gaps is not None:
            faixa = "0-1" if gaps[i] <= 1 else "2-5" if gaps[i] <= 5 else "6-10" if gaps[i] <= 10 \
                else "11-25"
            por_gap.setdefault(faixa, [0, 0, 0])
            if p == ASSOC:
                por_gap[faixa][0 if t == ASSOC else 1] += 1
            elif t == ASSOC:
                por_gap[faixa][2] += 1
        if p != ASSOC or t == ASSOC:
            continue
        if t == NEG:
            contagem["verdadeiro_negation_of"] += 1
            continue
        doc, a, b = pares[i]
        inverso = rotulo.get((doc, b, a))
        if inverso == ASSOC:
            contagem["inverso_associated_with"] += 1
        elif inverso == NEG:
            contagem["inverso_negation_of"] += 1
        else:
            contagem["sem_relacao_nas_duas_direcoes"] += 1
    saida: dict[str, Any] = {"fp_total": sum(contagem.values()), "fp_por_origem": contagem}
    if gaps is not None:
        saida["por_distancia"] = {k: {"tp": v[0], "fp": v[1], "fn": v[2],
                                      "precision": met._dividir(v[0], v[0] + v[1]),
                                      "recall": met._dividir(v[0], v[0] + v[2])}
                                  for k, v in sorted(por_gap.items(), key=lambda kv: int(kv[0].split("-")[0]))}
    return saida


# --------------------------------------------------------------------------- #
# 3. Comparações pareadas                                                     #
# --------------------------------------------------------------------------- #
def _f1_de_contagens(cm: np.ndarray) -> np.ndarray:
    """F1 por classe a partir de matrizes 3x3 achatadas (..., 9)."""
    cm = cm.reshape(cm.shape[:-1] + (3, 3))
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    suporte = cm.sum(axis=-1)
    preditos = cm.sum(axis=-2)
    den = suporte + preditos
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(den > 0, 2.0 * tp / np.where(den > 0, den, 1), 0.0)
    return f1


def _resumo(observado: float, amostras: np.ndarray) -> dict[str, Any]:
    baixo, alto = np.percentile(amostras, [2.5, 97.5])
    # A mesma convenção de `significancia.bootstrap_pareado`.
    if observado >= 0:
        p = 2.0 * float(np.mean(amostras <= 0.0))
    else:
        p = 2.0 * float(np.mean(amostras >= 0.0))
    leitura = "positiva" if baixo > 0 else "negativa" if alto < 0 else "não detectada"
    return {"diferenca": float(observado), "media_bootstrap": float(np.mean(amostras)),
            "ic95": [float(baixo), float(alto)], "p_bilateral": min(1.0, p), "diferenca_na_seed42": leitura}


METRICAS_COMPARADAS = ("f1_negation_of", "f1_associated_with", "macro_f1")


def _deltas(f1_a: np.ndarray, f1_b: np.ndarray) -> dict[str, np.ndarray]:
    return {"f1_negation_of": f1_a[..., NEG] - f1_b[..., NEG],
            "f1_associated_with": f1_a[..., ASSOC] - f1_b[..., ASSOC],
            "macro_f1": f1_a.mean(axis=-1) - f1_b.mean(axis=-1)}


def comparar_pareado(y_true: Sequence[int], y_a: Sequence[int], y_b: Sequence[int], *,
                     grupos: Sequence[str] | None = None, n_boot: int = N_BOOT,
                     seed: int = SEMENTE) -> dict[str, Any]:
    """A − B no mesmo conjunto: bootstrap por candidato (sequência do
    protocolo), por documento (se `grupos`) e McNemar exato."""
    yt, a, b = (np.asarray(v, dtype=np.int64) for v in (y_true, y_a, y_b))
    if not (yt.shape == a.shape == b.shape):
        raise ValueError("y_true, y_a e y_b têm tamanhos diferentes")
    n = yt.size
    cod_a, cod_b = 3 * yt + a, 3 * yt + b
    obs = _deltas(_f1_de_contagens(np.bincount(cod_a, minlength=9)),
                  _f1_de_contagens(np.bincount(cod_b, minlength=9)))

    rng = np.random.default_rng(seed)
    amostras = {k: np.empty(n_boot) for k in METRICAS_COMPARADAS}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)          # mesma sequência de significancia.bootstrap_pareado
        d = _deltas(_f1_de_contagens(np.bincount(cod_a[idx], minlength=9)),
                    _f1_de_contagens(np.bincount(cod_b[idx], minlength=9)))
        for k in METRICAS_COMPARADAS:
            amostras[k][i] = d[k]
    saida: dict[str, Any] = {
        "n": int(n), "n_boot": n_boot, "seed": seed,
        "por_candidato": {k: _resumo(float(obs[k]), amostras[k]) for k in METRICAS_COMPARADAS},
        "mcnemar": significancia.mcnemar_exato(a == yt, b == yt),
    }
    if grupos is not None:
        if len(grupos) != n:
            raise ValueError("grupos com tamanho diferente de y_true")
        docs, inv = np.unique(np.asarray(grupos), return_inverse=True)
        cm_a = np.zeros((docs.size, 9), dtype=np.int64)
        cm_b = np.zeros((docs.size, 9), dtype=np.int64)
        np.add.at(cm_a, (inv, cod_a), 1)
        np.add.at(cm_b, (inv, cod_b), 1)
        rng_doc = np.random.default_rng(seed)
        pesos = np.stack([np.bincount(rng_doc.integers(0, docs.size, docs.size), minlength=docs.size)
                          for _ in range(n_boot)])                    # (n_boot, D)
        d = _deltas(_f1_de_contagens(pesos @ cm_a), _f1_de_contagens(pesos @ cm_b))
        saida["por_documento"] = {k: _resumo(float(obs[k]), d[k]) for k in METRICAS_COMPARADAS}
        saida["n_documentos"] = int(docs.size)
    return saida


def tabela_tau(limiar: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A grade de τ do Two-Stage, como gravada em `limiar.json` (DEV)."""
    linhas = []
    for linha in limiar.get("tabela") or []:
        c = linha["contagens"]
        assoc = c["associated_with"]
        linhas.append({
            "tau": linha["limiar"], "n_roteados": linha.get("n_roteados"),
            "f1_negation_of": linha["f1_negation_of"], "macro_f1": linha["macro_f1"],
            "assoc_tp": assoc["tp"], "assoc_fp": assoc["fp"], "assoc_fn": assoc["fn"],
            "assoc_precision": met._dividir(assoc["tp"], assoc["tp"] + assoc["fp"]),
            "assoc_recall": met._dividir(assoc["tp"], assoc["tp"] + assoc["fn"]),
            "neg_tp": c["negation_of"]["tp"], "neg_fp": c["negation_of"]["fp"], "neg_fn": c["negation_of"]["fn"],
        })
    return linhas
