#!/usr/bin/env python3
"""
Gera as referências de validação da nova implementação a partir do legado.

O legado é o projeto anterior, no repositório original
(https://github.com/angeloalsf/RECLin-PT, commit a5f055c). Ele não faz parte
deste repositório: este script recebe o caminho de um checkout dele em
`--legado` e é a ÚNICA ponte entre os dois. Importa e executa o código do
legado para registrar o que ele produz, e grava tudo nesta pasta. O código novo
nunca importa o legado; os testes de equivalência comparam o que o código novo
produz com os arquivos gravados aqui, e por isso rodam sem o legado.

O QUE É GERADO
--------------
dados.json
    Por partição (train/dev/test), com a janela `max_gap=25` dos experimentos:
    número de documentos e de candidatos, contagem por rótulo, SHA-256 da
    sequência de candidatos (doc, e1, e2, rótulo, na ordem de
    `iter_candidate_pairs`), SHA-256 das janelas marcadas
    (`build_marked_window`, `ctx_chars=128`) e do `y_true`, conferido contra os
    sidecars oficiais. Também: o léxico de pistas induzido do TRAIN para cada
    `min_freq` calibrado, o `lexico_sha1` do léxico congelado (`min_freq=3`),
    os índices do espaço restrito e os pesos `balanced` dos dois espaços.
    Por fim, o MANIFEST das partições do legado e a configuração registrada
    dos quatro baselines (`config` e `config_sha1` da trilha `test_evals`).

modelo_minusculo/
    BERT minúsculo e aleatório (receita de `src/pair_aware/test_pair_aware.py`
    do legado) com tokenizer de vocabulário sintético, salvo com
    `save_pretrained`. É a entrada comum das referências de treino.

treino/
    Sidecars do legado treinando o modelo minúsculo em CPU, sem retomada:
    baseline (espaço completo, cabeça [CLS]), fine-tuning restrito e Pair-Aware.
    Cada treino roda duas vezes; as predições precisam sair idênticas, senão a
    referência não serve para teste de equivalência e o script falha.

resultados_legado/
    Cópia sem alteração dos resultados do legado que a nova implementação
    precisa reproduzir (etapa 2): os quatro baselines (métricas, sidecars do
    TEST e do DEV, trilha), os sidecars do filtro e da regra pura com o
    resumo da fase 2 e a calibração do filtro (que congelou o léxico), as 26
    comparações de significância, a agregação entre sementes, as cinco
    execuções do fine-tuning restrito e a Pair-Aware. Mesma estrutura de
    pastas de `results/`.

referencias.json
    Ambiente, comandos, SHA-256 de cada arquivo gerado, resultado da checagem
    de determinismo e dos dois testes de CPU do legado.

Nos registros, `legado/` designa a raiz desse checkout (onde ficam `src/`,
`data/` e `results/`).

Uso (no ambiente com as versões de `codigo/requirements.txt`, as mesmas do
legado):

    git clone https://github.com/angeloalsf/RECLin-PT RECLin-PT-legado
    git -C RECLin-PT-legado checkout a5f055c
    python codigo/testes/referencia/gerar_referencias.py --legado RECLin-PT-legado
    python codigo/testes/referencia/gerar_referencias.py --legado RECLin-PT-legado --partes dados
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[2]

# Definidos por usar_legado() a partir de --legado.
LEGADO = SPLITS = RESULTS = Path()


def usar_legado(raiz: Path) -> None:
    global LEGADO, SPLITS, RESULTS
    raiz = raiz.resolve()
    faltando = [p for p in ("src/relation_extraction.py", "data/splits/MANIFEST.json", "results")
                if not (raiz / p).exists()]
    if faltando:
        raise SystemExit(f"{raiz} não parece o legado (commit a5f055c): falta {', '.join(faltando)}")
    LEGADO, SPLITS, RESULTS = raiz, raiz / "data" / "splits", raiz / "results"

MAX_GAP = 25
CTX_CHARS = 128
MIN_FREQS_CALIBRADOS = (1, 2, 3, 5, 10)
MIN_FREQ_CONGELADO = 3
SPLITS_NOMES = ("train", "dev", "test")

# Uma thread: matmul em CPU pode mudar nos últimos bits com o número de
# threads. Fixar em 1 torna a referência reproduzível em outra máquina.
AMBIENTE_TREINO = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}


def sha256_bytes(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def sha256_arquivo(caminho: Path) -> str:
    return sha256_bytes(caminho.read_bytes())


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, ensure_ascii=False,
                                   separators=(",", ":")).encode("utf-8"))


def gravar_json(caminho: Path, obj) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


# --------------------------------------------------------------------------- #
# Parte A: dados, candidatos, janelas, léxico, espaço restrito                 #
# --------------------------------------------------------------------------- #
def gerar_dados() -> dict:
    for p in (LEGADO / "src", LEGADO / "src" / "finetuning_restrito"):
        sys.path.insert(0, str(p))
    import _isolamento  # noqa: F401  (legado: desvia loggers antes do núcleo)
    from candidates import iter_candidate_pairs
    from negation_lexicon import count_cue_forms, induce_lexicon
    from relation_extraction import LABEL2ID, LABELS, build_marked_window, read_jsonl
    from restricted_space import carregar_espacos, carregar_lexico, lexico_sha1, pesos_balanced

    manifesto = json.loads((SPLITS / "MANIFEST.json").read_text(encoding="utf-8"))
    sidecars = {"dev": RESULTS / "baseline_biobertpt_seed42.dev_preds.json",
                "test": RESULTS / "baseline_biobertpt_seed42.preds.json"}

    saida = {"max_gap": MAX_GAP, "ctx_chars": CTX_CHARS, "labels": LABELS,
             "particoes": {}}
    for nome in SPLITS_NOMES:
        arquivo = SPLITS / f"{nome}.jsonl"
        docs = list(read_jsonl(arquivo))
        linhas, janelas, y_true = [], [], []
        contagem = {l: 0 for l in LABELS}
        for doc in docs:
            for c in iter_candidate_pairs(doc, max_gap=MAX_GAP):
                linhas.append([doc["doc_id"], c["e1"]["id"], c["e2"]["id"], c["label"]])
                janelas.append(build_marked_window(doc["text"], c["e1"], c["e2"], CTX_CHARS))
                y_true.append(LABEL2ID[c["label"]])
                contagem[c["label"]] += 1
        sha_arquivo = sha256_arquivo(arquivo)
        if sha_arquivo != manifesto["splits"][nome]["sha256"]:
            raise SystemExit(f"{arquivo} não confere com o MANIFEST do legado")
        entrada = {
            "arquivo_sha256": sha_arquivo,
            "n_documentos": len(docs),
            "n_candidatos": len(linhas),
            "contagem_por_rotulo": contagem,
            "candidatos_sha256": sha256_json(linhas),
            "janelas_sha256": sha256_json(janelas),
            "y_true_sha256": sha256_json(y_true),
        }
        if nome in sidecars:
            ref = json.loads(sidecars[nome].read_text(encoding="utf-8"))
            entrada["y_true_confere_com"] = sidecars[nome].relative_to(LEGADO).as_posix()
            if ref["y_true"] != y_true:
                raise SystemExit(f"y_true de {nome} diverge de {sidecars[nome]}")
        saida["particoes"][nome] = entrada
        print(f"  {nome}: {len(docs)} docs, {len(linhas)} candidatos", flush=True)

    train_docs = list(read_jsonl(SPLITS / "train.jsonl"))
    formas = count_cue_forms(train_docs, MAX_GAP)
    saida["lexico"] = {
        "contagem_formas_train": dict(sorted(formas.items(), key=lambda kv: (-kv[1], kv[0]))),
        "por_min_freq": {str(m): sorted(induce_lexicon(train_docs, MAX_GAP, m))
                         for m in MIN_FREQS_CALIBRADOS},
    }
    lex = carregar_lexico(train_docs, max_gap=MAX_GAP, min_freq=MIN_FREQ_CONGELADO,
                          guarda=RESULTS / "CALIBRACAO_filtro.json")
    saida["lexico"]["congelado"] = {"min_freq": MIN_FREQ_CONGELADO,
                                    "formas": lex, "lexico_sha1": lexico_sha1(lex)}

    espacos = carregar_espacos(SPLITS, lex, max_gap=MAX_GAP, ctx_chars=CTX_CHARS)
    saida["espaco_restrito"] = {
        nome: {"n_restrito": e.n_restrito, "indices_sha256": sha256_json(e.indices),
               "janelas_sha256": sha256_json(e.textos), "resumo": e.resumo()}
        for nome, e in espacos.items()}

    y_train = [LABEL2ID[c["label"]] for d in train_docs
               for c in iter_candidate_pairs(d, max_gap=MAX_GAP)]
    saida["pesos_balanced"] = {
        "espaco_completo": dict(zip(LABELS, pesos_balanced(y_train))),
        "espaco_restrito": dict(zip(LABELS, pesos_balanced(espacos["train"].y_restrito))),
    }

    saida["manifesto_particoes"] = manifesto
    saida["config_baselines"] = config_baselines()
    return saida


def config_baselines() -> dict:
    """Configuração dos quatro baselines como o legado a registrou.

    `config` vem do `<out>.json`. `config_sha1` vem da primeira linha da trilha
    `<out>.test_evals.jsonl` (a avaliação do próprio baseline; as linhas
    seguintes são do filtro, com outra configuração). O hash do legado cobre
    também `weight_decay`, `warmup_ratio` e `splits_dir`, que não aparecem em
    `config`: é ele que permite conferir esses três valores.
    """
    saida = {}
    for encoder in ("biobertpt", "bertimbau"):
        for seed in (42, 43):
            nome = f"baseline_{encoder}_seed{seed}"
            resultado = json.loads((RESULTS / f"{nome}.json").read_text(encoding="utf-8"))
            trilha = (RESULTS / f"{nome}.test_evals.jsonl").read_text(encoding="utf-8")
            primeira = json.loads(trilha.splitlines()[0])
            if primeira["eval_index"] != 1 or primeira["model"] != resultado["model"]:
                raise SystemExit(f"trilha de {nome} não começa pela avaliação do baseline")
            saida[nome] = {"model": resultado["model"], "seed": resultado["seed"],
                           "config": resultado["config"],
                           "config_sha1": primeira["config_sha1"],
                           "receita_config_sha1": "relation_extraction.append_test_eval_log"}
    return saida


# --------------------------------------------------------------------------- #
# Parte B: modelo minúsculo                                                   #
# --------------------------------------------------------------------------- #
def gerar_modelo(destino: Path) -> None:
    """Mesma receita de src/pair_aware/test_pair_aware.tokenizer_e_encoder do legado,
    salvo SEM os marcadores, como um checkpoint real do Hub: o treino os
    acrescenta com add_special_tokens."""
    import string

    import torch
    from transformers import BertConfig, BertModel, BertTokenizer
    chars = sorted(set(string.ascii_lowercase + string.digits + string.punctuation))
    palavras = ["sem", "nega", "nao", "febre", "dor", "edema", "tosse", "paciente", "refere"]
    vocab = (["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + palavras + chars
             + ["##" + c for c in chars if c.isalnum()])
    tok = BertTokenizer(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=512)
    torch.manual_seed(0)
    if destino.exists():
        shutil.rmtree(destino)
    BertModel(cfg).save_pretrained(destino)
    tok.save_pretrained(destino)


# --------------------------------------------------------------------------- #
# Parte C: treino do modelo minúsculo pelo legado                             #
# --------------------------------------------------------------------------- #
def comandos_treino(modelo: Path, saida: Path) -> dict[str, list[str]]:
    py = sys.executable
    # lr 1e-3 (e não 2e-5): com o lr dos experimentos o modelo minúsculo não
    # aprende nada em poucas épocas, prevê só no_relation e a referência não
    # exercitaria a escolha da melhor época nem o argmax.
    comum = ["--model", str(modelo), "--seed", "42", "--lr", "1e-3"]
    return {
        "baseline_minusculo_seed42": [
            py, "src/baseline_biobertpt.py", "--splits-dir", "data/splits",
            "--epochs", "2", "--batch-size", "64", "--max-gap", "25",
            "--max-length", "128", *comum,
            "--out", str(saida / "baseline_minusculo_seed42.json")],
        "restrito_minusculo_seed42": [
            py, "src/finetuning_restrito/train_restrito.py", "--encoder", "biobertpt",
            "--epochs", "10", *comum,
            "--out", str(saida / "restrito_minusculo_seed42.json")],
        "pairaware_minusculo_seed42": [
            py, "src/pair_aware/train_pair_aware.py", "--encoder", "biobertpt",
            "--epochs", "3", "--avaliar-test", *comum,
            "--out", str(saida / "pairaware_minusculo_seed42.json")],
    }


SUFIXOS_COMPARADOS = (".preds.json", ".dev_preds.json")
SUFIXOS_GUARDADOS = (".json", ".preds.json", ".dev_preds.json")


def rodar(cmd: list[str], log: Path) -> float:
    env = dict(os.environ, **AMBIENTE_TREINO)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=LEGADO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"falhou ({r.returncode}): {' '.join(cmd)} -- ver {log}")
    return round(time.time() - t0, 1)


def gerar_treino(modelo: Path, destino: Path, so: set[str] | None = None) -> dict:
    destino.mkdir(parents=True, exist_ok=True)
    registro = {}
    with tempfile.TemporaryDirectory(prefix="ref_treino_") as tmp:
        tmp = Path(tmp)
        execs = {r: comandos_treino(modelo, tmp / r) for r in ("a", "b")}
        for nome in execs["a"]:
            if so and nome not in so:
                continue
            hashes, duracoes = {}, {}
            for rodada in ("a", "b"):
                (tmp / rodada).mkdir(exist_ok=True)
                print(f"  treino {nome} (rodada {rodada})...", flush=True)
                duracoes[rodada] = rodar(execs[rodada][nome], tmp / rodada / f"{nome}.log")
                hashes[rodada] = {s: sha256_arquivo(tmp / rodada / f"{nome}{s}")
                                  for s in SUFIXOS_COMPARADOS}
            if hashes["a"] != hashes["b"]:
                raise SystemExit(f"{nome}: duas rodadas com a mesma semente deram "
                                 f"predições diferentes {hashes} -- referência inválida")
            for s in SUFIXOS_GUARDADOS:
                shutil.copyfile(tmp / "a" / f"{nome}{s}", destino / f"{nome}{s}")
            cmd = ["python"] + [c.replace(str(tmp / "a"), "<saida>").replace(str(modelo), "<modelo_minusculo>")
                               for c in execs["a"][nome][1:]]
            registro[nome] = {"comando": cmd, "cwd": "legado/",
                              "deterministico_em_duas_rodadas": True,
                              "duracao_s": duracoes}
    return registro


# --------------------------------------------------------------------------- #
# Parte E: resultados do legado (etapa 2)                                     #
# --------------------------------------------------------------------------- #
N_COMPARACOES = 26


def lista_resultados() -> list[str]:
    """Arquivos de `results/` copiados, relativos a ele. Ficam de fora os
    `archive_*` (rodadas substituídas), as comparações de DEV dos critérios de
    parada e os `filtro_dev/`, que pertencem às etapas das estratégias."""
    nomes = []
    for encoder in ("biobertpt", "bertimbau"):
        for seed in (42, 43):
            nomes += [f"baseline_{encoder}_seed{seed}{s}"
                      for s in (".json", ".preds.json", ".dev_preds.json", ".test_evals.jsonl")]
            nomes.append(f"filtro_{encoder}_seed{seed}.preds.json")
    nomes += ["regra_pura.preds.json", "regra_pura.test_evals.jsonl",
              "FASE2_test_summary.json", "summary_by_seed.json",
              "CALIBRACAO_filtro.json"]       # guarda do léxico congelado (etapa 3)
    comparacoes = sorted(p.name for p in RESULTS.glob("significance_*.json"))
    if len(comparacoes) != N_COMPARACOES:
        raise SystemExit(f"esperadas {N_COMPARACOES} comparações em results/, há {len(comparacoes)}")
    nomes += comparacoes
    for seed in range(42, 47):
        nomes += [f"finetuning_restrito/restrito_biobertpt_seed{seed}{s}"
                  for s in (".json", ".preds.json", ".dev_preds.json", ".test_evals.jsonl")]
    nomes += ["pair_aware/pairaware_biobertpt_seed42.json",
              "pair_aware/pairaware_biobertpt_seed42.dev_preds.json"]
    return nomes


def copiar_resultados(destino: Path) -> dict:
    if destino.exists():
        shutil.rmtree(destino)
    nomes = lista_resultados()
    for nome in nomes:
        alvo = destino / nome
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RESULTS / nome, alvo)
        if sha256_arquivo(alvo) != sha256_arquivo(RESULTS / nome):
            raise SystemExit(f"cópia de {nome} não confere")
    return {"origem": "results/", "arquivos": len(nomes), "comparacoes": N_COMPARACOES}


# --------------------------------------------------------------------------- #
# Parte D: testes de CPU do legado                                            #
# --------------------------------------------------------------------------- #
def rodar_testes_legado() -> dict:
    """Roda os dois testes de CPU do legado e devolve o arquivo que um deles
    regrava (verificacao_cpu.json) ao conteúdo original: o legado é somente
    leitura."""
    testes = {"test_remapeamento": "src/finetuning_restrito/test_remapeamento.py",
              "test_pair_aware": "src/pair_aware/test_pair_aware.py"}
    regravado = RESULTS / "pair_aware" / "verificacao_cpu.json"
    original = regravado.read_bytes()
    resultado = {}
    try:
        for nome, script in testes.items():
            env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([sys.executable, script], cwd=LEGADO, env=env,
                               capture_output=True, text=True)
            linhas = [ln for ln in (r.stdout + r.stderr).splitlines()
                      if "checagens" in ln and ("OK" in ln or "FALH" in ln)]
            resultado[nome] = {"codigo_saida": r.returncode,
                               "resumo": linhas[-1].split("] ", 3)[-1].split(" -> ")[0] if linhas else None}
            if r.returncode != 0:
                raise SystemExit(f"{nome} falhou no legado:\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    finally:
        regravado.write_bytes(original)
    return resultado


# --------------------------------------------------------------------------- #
def ambiente() -> dict:
    import importlib
    versoes = {}
    for m in ("torch", "transformers", "tokenizers", "safetensors", "numpy",
              "scipy", "sklearn", "huggingface_hub"):
        try:
            versoes[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001
            versoes[m] = None
    return {"python": platform.python_version(), "plataforma": platform.platform(),
            "versoes": versoes, "variaveis_treino": AMBIENTE_TREINO}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--legado", type=Path, default=None,
                    help="checkout do repositório original no commit a5f055c "
                         "(default: a pasta legado/ ao lado de codigo/, se existir)")
    ap.add_argument("--partes", default="dados,modelo,treino,testes,resultados",
                    help="subconjunto de: dados, modelo, treino, testes, resultados")
    ap.add_argument("--treinos", default=None,
                    help="na parte treino, so estes (separados por virgula)")
    args = ap.parse_args()
    partes = set(args.partes.split(","))
    legado = args.legado or REPO / "legado"
    if not legado.exists():
        raise SystemExit("informe --legado com um checkout do repositório original no commit a5f055c")
    usar_legado(legado)

    manifesto_path = AQUI / "referencias.json"
    manifesto = (json.loads(manifesto_path.read_text(encoding="utf-8"))
                 if manifesto_path.exists() else {})
    manifesto["origem"] = {"repositorio": "https://github.com/angeloalsf/RECLin-PT",
                           "commit": "a5f055c", "legado": "raiz do checkout passado em --legado"}
    manifesto["ambiente"] = ambiente()

    if "dados" in partes:
        print("Parte A: dados", flush=True)
        gravar_json(AQUI / "dados.json", gerar_dados())
    if "modelo" in partes:
        print("Parte B: modelo minúsculo", flush=True)
        gerar_modelo(AQUI / "modelo_minusculo")
    if "treino" in partes:
        print("Parte C: treino pelo legado", flush=True)
        so = set(args.treinos.split(",")) if args.treinos else None
        manifesto.setdefault("treino", {}).update(
            gerar_treino(AQUI / "modelo_minusculo", AQUI / "treino", so))
    if "resultados" in partes:
        print("Parte E: resultados do legado", flush=True)
        manifesto["resultados_legado"] = copiar_resultados(AQUI / "resultados_legado")
    if "testes" in partes:
        print("Parte D: testes de CPU do legado", flush=True)
        manifesto["testes_cpu_legado"] = rodar_testes_legado()

    manifesto["arquivos_sha256"] = {
        p.relative_to(AQUI).as_posix(): sha256_arquivo(p)
        for p in sorted(AQUI.rglob("*"))
        if p.is_file() and p.name not in ("referencias.json", "README.md")
        and p.suffix != ".py" and "__pycache__" not in p.parts}
    gravar_json(manifesto_path, manifesto)
    print(f"Referências em {AQUI.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
