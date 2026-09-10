#!/usr/bin/env python3
"""
Nucleo compartilhado dos baselines de extracao de relacao (RECLin-PT).

Este modulo concentra TODA a logica de treino e avaliacao usada pelos baselines
de encoder (BioBERTpt clinico e BERTimbau geral). Os scripts de entrada
(`baseline_biobertpt.py`, `baseline_bertimbau.py`) sao apenas finas cascas que
escolhem o modelo e os caminhos de saida e chamam `run(args, log)` daqui.

POR QUE UM NUCLEO UNICO
-----------------------
A pergunta de pesquisa e: "o pre-treinamento clinico importa para extracao de
relacoes em textos medicos em portugues?". A resposta so e valida se a UNICA
diferenca entre os dois experimentos for o checkpoint de pre-treino. Mantendo
representacao de entrada, loss, scheduler, early stopping, salvamento, metricas,
seed e logging num so lugar, a paridade entre BioBERTpt e BERTimbau e garantida
por construcao -- nao por disciplina de copiar-colar.

ITENS MANTIDOS IDENTICOS ENTRE OS BASELINES
-------------------------------------------
- Entity markers / representacao de entrada: [E1] ... [/E1] e [E2] ... [/E2]
  como tokens especiais (Soares et al., 2019, "Matching the Blanks"), com janela
  de +-ctx_chars ao redor do par.
- Loss: CrossEntropy com class_weight=balanced (negativos `no_relation` dominam).
- Scheduler: linear com warmup (get_linear_schedule_with_warmup).
- Early stopping / selecao de modelo: melhor epoca pelo macro-F1 no DEV; o TEST
  e reportado UMA unica vez.
- Salvamento de checkpoints: `best_model/` (pesos do melhor epoch, formato HF) e
  `last_checkpoint/` (estado completo de retomada), ambos gravados de forma
  ATOMICA.
- Metricas: Micro-F1, Macro-F1, F1 por classe, classification_report (sklearn),
  matriz de confusao e MCC (Matthews, robusto a desbalanceamento).
- Curvas de treino/validacao: train_loss E dev_loss por epoca no `dev_history`.
- Predicoes do TEST salvas em um sidecar (`<out>.preds.json`) para o teste de
  significancia (ver `src/significance.py`: McNemar + bootstrap pareado).
- Seed de reprodutibilidade: set_all_seeds (python/numpy/torch/cuda + cudnn
  deterministico).
- Logging estruturado centralizado: tudo passa por src/utils/logger.py ->
  terminal + logs/pipeline.log.

CHECKPOINT / RETOMADA (--ckpt-dir)
----------------------------------
Ao fim de CADA epoca grava `<ckpt-dir>/last_checkpoint/` com o estado completo de
treino, de forma atomica (escreve em .tmp e renomeia). Quando uma epoca bate o
melhor dev macro-F1, grava tambem `<ckpt-dir>/best_model/` (somente pesos, no
formato `save_pretrained`). Ao iniciar, se `last_checkpoint/` existir e a config
bater, retoma da PROXIMA epoca. Aponte --ckpt-dir para o Google Drive para
sobreviver a quedas do runtime do Colab.

SIDECARS DE INSTRUMENTACAO (--save-split-preds / --save-probs / --test-eval-log)
-------------------------------------------------------------------------------
Alem do `<out>.json` de metricas, `run` grava arquivos AUXILIARES. Nenhum deles
influencia treino, loss, scheduler ou selecao de melhor epoca -- sao registro do
que ja aconteceu:

- `<out>.preds.json`     -- predicoes do TEST (contrato de src/significance.py).
- `<out>.dev_preds.json` -- predicoes do DEV geradas com os pesos da MELHOR
  epoca (a mesma restaurada antes do test), nao da ultima. Sem elas nao ha como
  calibrar honestamente um filtro lexico ou um limiar de decisao: calibrar no
  test vaza o conjunto de avaliacao para dentro do metodo, e o `dev_history` so
  guarda agregados (dev_loss, dev_macro_f1), nunca as predicoes individuais.
- campo `probs` -- distribuicao softmax por instancia, na ordem de LABELS e
  arredondada a 4 casas, para limiares do tipo P(negation_of) > tau com tau
  escolhido no dev.
- `<out>.test_evals.jsonl` -- uma linha por avaliacao final no TEST (timestamp,
  hash da config, metricas). Nao impede reavaliar o test; deixa registro de
  quantas vezes isso aconteceu, que e o que a analise de multiplicidade precisa
  poder auditar.

Os defaults ligam tudo. `--save-split-preds test --no-save-probs
--no-test-eval-log` reproduz byte-a-byte a saida da versao anterior do modulo.

BACKUP DO MELHOR CHECKPOINT NO HF HUB (--hf-backup-repo)
--------------------------------------------------------
Camada EXTRA de seguranca ALEM do --ckpt-dir no Drive, nunca no lugar dele.
Passando `--hf-backup-repo <usuario>/<repo>`, toda vez que uma epoca bate o
melhor dev macro-F1 o `best_model/` recem-gravado tambem sobe, em segundo plano,
para um repositorio de modelo PRIVADO no Hugging Face Hub. O envio acontece na
hora, a cada nova melhor epoca -- nao ao final -- justamente para o caso de o
runtime cair na epoca seguinte.

Toda a logica vive em `src/hf_backup.py`, que so e importado quando a flag e
usada: sem ela nao ha import de `huggingface_hub` nem chamada de rede nova. O
envio roda numa thread e NUNCA derruba o treino -- qualquer falha (rede, cota,
rate limit, token invalido) vira WARNING. O token vem de --hf-token ou de
$HF_TOKEN e nao e registrado no log em nenhum nivel.

O backup depende de --ckpt-dir: e o `save_best_model` que dispara o envio, e ele
so roda quando ha --ckpt-dir. Passar --hf-backup-repo sem --ckpt-dir emite um
WARNING explicito na largada em vez de falhar em silencio.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import iter_candidate_pairs  # noqa: E402
from utils.logger import get_logger  # noqa: E402

# Logger padrao do nucleo. Cada entry-point passa o seu (com o nome do baseline)
# para `run`, que reatribui o global abaixo -- assim o campo [MODULO] do log
# reflete qual baseline esta rodando.
log = get_logger("relation_extraction")

# --------------------------------------------------------------------------- #
# Espaco de rotulos e marcadores de entidade                                  #
# --------------------------------------------------------------------------- #
LABELS = ["negation_of", "associated_with", "no_relation"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

E1_OPEN, E1_CLOSE, E2_OPEN, E2_CLOSE = "[E1]", "[/E1]", "[E2]", "[/E2]"
MARKER_TOKENS = [E1_OPEN, E1_CLOSE, E2_OPEN, E2_CLOSE]

# Nomes das pastas de checkpoint (mantidos identicos entre os baselines).
BEST_MODEL_DIR = "best_model"
LAST_CKPT_DIR = "last_checkpoint"
STATE_FILE = "training_state.pt"


# --------------------------------------------------------------------------- #
# Dados / representacao de entrada                                             #
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_marked_window(text, e1, e2, ctx_chars):
    """Recorta janela centrada no par e insere os marcadores de entidade."""
    lo, hi = min(e1["start"], e2["start"]), max(e1["end"], e2["end"])
    ws, we = max(0, lo - ctx_chars), min(len(text), hi + ctx_chars)
    window = text[ws:we]
    inserts: dict[int, list[tuple[int, str]]] = {}

    def add(i, marker, prio):
        inserts.setdefault(i, []).append((prio, marker))

    add(e1["start"] - ws, E1_OPEN + " ", 1)
    add(e1["end"] - ws, " " + E1_CLOSE, 0)
    add(e2["start"] - ws, E2_OPEN + " ", 1)
    add(e2["end"] - ws, " " + E2_CLOSE, 0)

    out = []
    for i in range(len(window) + 1):
        if i in inserts:
            for _, m in sorted(inserts[i]):
                out.append(m)
        if i < len(window):
            out.append(window[i])
    return "".join(out)


def build_dataset(docs, max_gap, ctx_chars):
    texts, labels = [], []
    for doc in docs:
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            texts.append(build_marked_window(doc["text"], c["e1"], c["e2"], ctx_chars))
            labels.append(LABEL2ID[c["label"]])
    return texts, labels


# --------------------------------------------------------------------------- #
# Reprodutibilidade                                                           #
# --------------------------------------------------------------------------- #
def set_all_seeds(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(tokenizer, texts, labels, max_length, batch_size, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return texts[i], labels[i]

    def collate(batch):
        bt, bl = zip(*batch)
        enc = tokenizer(list(bt), truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(bl)

    g = torch.Generator().manual_seed(seed)
    return DataLoader(DS(), batch_size=batch_size, shuffle=shuffle,
                      generator=g, collate_fn=collate)


def predict(model, loader, device, *, return_probs=False):
    """So predicoes (usado no TEST e, na melhor epoca, no DEV).

    Com `return_probs=True` devolve `(preds, probs)`, onde `probs` e a softmax
    sobre as 3 classes por instancia, na ordem de LABELS e na MESMA ordem de
    `preds` (os loaders de avaliacao rodam com shuffle=False). O default mantem
    a assinatura antiga: quem so quer o argmax continua recebendo uma lista.

    A softmax e monotonica, entao `argmax(probs) == argmax(logits)`: acrescentar
    as probabilidades nao pode alterar nenhuma predicao. A funcao roda em
    `model.eval()` e sob `no_grad`, sem dropout e sem consumir RNG global.
    """
    import torch
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for ids, attn, _ in loader:
            logits = model(input_ids=ids.to(device),
                           attention_mask=attn.to(device)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            if return_probs:
                probs.extend(torch.softmax(logits.float(), dim=-1).cpu().tolist())
    if return_probs:
        return preds, probs
    return preds


def evaluate(model, loader, device, loss_fn):
    """Predicoes + loss media (usado no DEV, para a curva de validacao)."""
    import torch
    model.eval()
    preds = []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for ids, attn, lab in loader:
            logits = model(input_ids=ids.to(device),
                           attention_mask=attn.to(device)).logits
            loss = loss_fn(logits, lab.to(device))
            total_loss += loss.item()
            n_batches += 1
            preds.extend(logits.argmax(-1).cpu().tolist())
    return preds, total_loss / max(1, n_batches)


# --------------------------------------------------------------------------- #
# Matriz de confusao (renderizacao em texto para o log)                        #
# --------------------------------------------------------------------------- #
def render_confusion_matrix(cm, labels):
    """Devolve a matriz de confusao como string alinhada (linhas=verdadeiro,
    colunas=predito). Usada no log; o array tambem vai para o JSON de saida."""
    short = [l[:8] for l in labels]
    head = "true\\pred".ljust(12) + "".join(s.rjust(10) for s in short)
    lines = [head]
    for i, l in enumerate(labels):
        row = l[:11].ljust(12) + "".join(str(int(cm[i][j])).rjust(10)
                                          for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Instrumentacao: sidecars de predicao/probabilidade e registro de avaliacao   #
# --------------------------------------------------------------------------- #
def best_epoch_from_history(history):
    """Reconstroi QUAL epoca foi eleita a melhor, a partir do `dev_history`.

    Replica exatamente o criterio do laco de treino (`macro > best_f1`, com
    best_f1 inicial -1.0): em caso de empate vence a PRIMEIRA epoca. Derivar do
    historico -- em vez de guardar um contador novo -- e o que faz a informacao
    sobreviver a uma retomada: `history` ja viaja dentro do `last_checkpoint`,
    um contador novo nao viajaria, e acrescenta-lo mudaria o formato do
    checkpoint. Devolve (epoca, dev_macro_f1) ou (None, None).
    """
    best_ep, best_f1 = None, -1.0
    for entry in history or []:
        f1 = entry.get("dev_macro_f1")
        if f1 is None:
            continue
        if f1 > best_f1:
            best_f1, best_ep = f1, int(entry["epoch"])
    return best_ep, (best_f1 if best_ep is not None else None)


def round_probs(probs, ndigits=4):
    """Arredonda as probabilidades para serializacao.

    ~19 mil instancias x 3 classes x 2 splits x 4 execucoes: em float64 puro os
    sidecars passam de 10 MB por execucao (e `results/` e versionado). Com 4
    casas cada split fica em ~0,5 MB, resolucao muito acima do passo de qualquer
    limiar calibrado no dev. Efeito colateral: as linhas deixam de somar
    exatamente 1,0 -- renormalize se isso importar para o seu uso.
    """
    return [[round(float(p), ndigits) for p in row] for row in probs]


def build_preds_payload(*, args, y_true_ids, y_pred_ids, probs=None, extra=None):
    """Monta o payload de um sidecar de predicoes.

    As cinco chaves originais (model, seed, labels, y_true, y_pred) vem primeiro
    e na MESMA ordem de sempre: com `--save-split-preds test --no-save-probs` o
    `<out>.preds.json` sai byte-a-byte igual ao da versao anterior deste modulo,
    que e a condicao para comparar com os resultados ja escritos no TCC.
    `src/significance.py` le por chave, entao o campo extra `probs` e inerte
    para ele.
    """
    payload = {
        "model": args.model, "seed": args.seed, "labels": LABELS,
        "y_true": [int(i) for i in y_true_ids],
        "y_pred": [int(i) for i in y_pred_ids],
    }
    if extra:
        payload.update(extra)
    if probs is not None:
        payload["probs"] = round_probs(probs)
    return payload


def append_test_eval_log(out_path, *, args, n_test, macro_f1, negation_of_f1):
    """Anexa uma linha a `<out>.test_evals.jsonl` a cada avaliacao final no TEST.

    O que isto E: uma trilha de auditoria. Cada linha traz data/hora, hash da
    configuracao de treino e as metricas obtidas, e `eval_index` conta quantas
    avaliacoes ja existiam para aquele `--out`. Um `eval_index` maior que 1 com o
    mesmo `config_sha1` e exatamente o cenario que a analise de multiplicidade
    precisa declarar: a mesma configuracao teve o test medido mais de uma vez.

    O que isto NAO E: uma trava. Nao impede reavaliar, e nao ve um "espiar" feito
    abrindo o `<out>.json` a mao. O caminho e derivado de `--out`, entao o
    registro acompanha o resultado a que se refere e nao depende do diretorio de
    trabalho nem de nenhum estado global do processo.
    """
    import hashlib
    log_path = Path(out_path).with_suffix(".test_evals.jsonl")
    cfg = dict(_config_guard(args))
    cfg.update({"class_weight": args.class_weight,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "splits_dir": str(args.splits_dir)})
    digest = hashlib.sha1(
        json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    previous = 0
    if log_path.is_file():
        previous = sum(1 for ln in log_path.read_text(encoding="utf-8").splitlines()
                       if ln.strip())
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "eval_index": previous + 1,
        "out": str(out_path),
        "model": args.model,
        "seed": args.seed,
        "config_sha1": digest,
        "n_test": int(n_test),
        "test_macro_f1": round(float(macro_f1), 6),
        "test_negation_of_f1": round(float(negation_of_f1), 6),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" explicito: `*.jsonl` nao esta coberto pelo .gitattributes,
    # entao deixar o Python traduzir a quebra de linha faria o arquivo virar
    # CRLF se algum dia isto rodar no Windows.
    with open(log_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if entry["eval_index"] > 1:
        log.warning("ATENCAO (multiplicidade): esta e a avaliacao #%d no TEST "
                    "registrada em %s. Se as anteriores usaram a mesma config "
                    "(config_sha1=%s), o test ja foi visto mais de uma vez para "
                    "esta configuracao -- declare isso ao reportar.",
                    entry["eval_index"], log_path, digest)
    else:
        log.info("Avaliacao no TEST registrada em %s (eval_index=1, "
                 "config_sha1=%s)", log_path, digest)


# --------------------------------------------------------------------------- #
# Checkpoint / retomada (pastas best_model/ e last_checkpoint/)               #
# --------------------------------------------------------------------------- #
def _config_guard(args):
    # NAO inclui --save-split-preds / --save-probs / --test-eval-log de
    # proposito: essas flags so decidem O QUE E GRAVADO DEPOIS que o treino
    # termina, e nao tocam em pesos, loss, scheduler nem selecao de epoca. Um
    # last_checkpoint gravado com uma combinacao e identico ao gravado com
    # outra. Coloca-las aqui faria a guarda REJEITAR um checkpoint valido (o
    # caminho de mismatch descarta o checkpoint e recomeca do zero, em silencio)
    # por um motivo que nao tem nada a ver com o modelo -- horas de T4 perdidas.
    return {
        "model": args.model, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "max_gap": args.max_gap, "ctx_chars": args.ctx_chars,
        "lr": args.lr, "seed": args.seed,
    }


def save_last_checkpoint(ckpt_dir, *, epoch, model, optimizer, scheduler,
                         best_f1, history, args):
    """Grava o estado COMPLETO de retomada em <ckpt-dir>/last_checkpoint/
    de forma atomica (escreve training_state.pt.tmp e renomeia)."""
    import torch
    last = Path(ckpt_dir) / LAST_CKPT_DIR
    last.mkdir(parents=True, exist_ok=True)
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    payload = {
        "epoch": epoch,                         # ultima epoca CONCLUIDA
        "model": model.state_dict(),            # modelo ATUAL (para continuar)
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_f1": best_f1,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": cuda_rng,
        },
        "config_guard": _config_guard(args),
        # Seed tambem no topo do payload (alem de dentro de config_guard): a
        # guarda de retomada precisa distinguir "seed divergente" (erro) de
        # "checkpoint antigo, sem seed registrada" (so aviso).
        "seed": args.seed,
    }
    final = last / STATE_FILE
    tmp = last / (STATE_FILE + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, final)
    log.info("last_checkpoint salvo (epoca %d concluida) em %s", epoch, final)


def save_best_model(ckpt_dir, model, tokenizer, *, hf_backup=None, hf_note=None):
    """Grava SOMENTE os pesos do melhor epoch em <ckpt-dir>/best_model/ no
    formato HuggingFace (save_pretrained), de forma atomica via pasta .tmp.

    Com `hf_backup` (uma instancia de `hf_backup.HFBackup`, criada em `run` so
    quando --hf-backup-repo e passado), dispara TAMBEM uma copia para o repo
    privado no HF Hub, imediatamente. A chamada retorna na hora -- o envio roda
    em outra thread -- e nunca levanta: falha de rede vira WARNING e o treino
    continua. `hf_note` vira a mensagem de commit, o que faz o historico do repo
    virar um registro de qual epoca gerou cada backup.
    """
    base = Path(ckpt_dir)
    base.mkdir(parents=True, exist_ok=True)
    best = base / BEST_MODEL_DIR
    tmp = base / (BEST_MODEL_DIR + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    if best.exists():
        shutil.rmtree(best)
    os.replace(tmp, best)
    log.info("best_model atualizado em %s", best)
    if hf_backup is not None:
        hf_backup.backup_model_async(model, tokenizer, note=hf_note)


def load_last_checkpoint(ckpt_dir, *, model, optimizer, scheduler, args):
    """Carrega <ckpt-dir>/last_checkpoint/ se existir e a config bater.
    Retorna (start_epoch, best_f1, history) ou None para comecar do zero."""
    import torch
    final = Path(ckpt_dir) / LAST_CKPT_DIR / STATE_FILE
    if not final.is_file():
        log.info("Nenhum last_checkpoint em %s -- comecando do zero", final)
        return None

    ckpt = torch.load(final, map_location="cpu", weights_only=False)

    # --- guarda de seed: divergencia aqui e ERRO, nao "comeca do zero" ---
    # Retomar epocas treinadas sob outra seed produz um treino que nao
    # corresponde a nenhuma das duas seeds -- e o resultado seria irreprodutivel
    # sem que nada no log denunciasse. Checkpoints gravados antes desta guarda
    # podem nao ter a seed registrada: nesse caso avisamos e seguimos, para nao
    # invalidar o que ja esta salvo em disco / no Drive.
    saved_seed = ckpt.get("seed", ckpt.get("config_guard", {}).get("seed"))
    if saved_seed is None:
        log.warning("last_checkpoint em %s nao tem seed registrada (formato "
                    "antigo) -- nao foi possivel validar contra --seed %s; "
                    "prosseguindo.", ckpt_dir, args.seed)
    elif int(saved_seed) != int(args.seed):
        raise RuntimeError(
            f"Checkpoint em {ckpt_dir} foi salvo com seed={saved_seed}, mas esta "
            f"execucao esta usando --seed {args.seed}. Retomar assim geraria um "
            f"treino inconsistente. Use a seed correta ou aponte para outro "
            f"--ckpt-dir (ex.: checkpoints/<modelo>_seed{args.seed})."
        )

    guard = ckpt.get("config_guard", {})
    want = _config_guard(args)
    mismatch = [k for k, v in want.items() if guard.get(k) != v]
    if mismatch:
        log.warning("last_checkpoint encontrado mas config divergente em %s -- "
                    "IGNORANDO checkpoint e comecando do zero.", mismatch)
        return None

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    rng = ckpt.get("rng", {})
    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as e:  # noqa: BLE001
        log.warning("Nao foi possivel restaurar estados de RNG: %s", e)

    start_epoch = int(ckpt["epoch"]) + 1
    log.info("last_checkpoint carregado: %d epoca(s) ja concluida(s), "
             "best_dev_macro_f1=%.4f -> retomando da epoca %d",
             ckpt["epoch"], ckpt["best_f1"], start_epoch)
    return start_epoch, ckpt["best_f1"], ckpt["history"]


def load_best_state(ckpt_dir):
    """Le os pesos de <ckpt-dir>/best_model/ (apos retomada sem novo melhor).
    Retorna um state_dict ou None se a pasta nao existir."""
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification
    best = Path(ckpt_dir) / BEST_MODEL_DIR
    if not (best / "config.json").is_file():
        return None
    m = AutoModelForSequenceClassification.from_pretrained(best)
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


# --------------------------------------------------------------------------- #
# CLI compartilhada                                                           #
# --------------------------------------------------------------------------- #
def build_arg_parser(*, default_model):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=None,
                    help="Pasta para best_model/ e last_checkpoint/ (ideal: "
                         "Google Drive). Habilita retomada automatica.")
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--max-gap", type=int, default=75)
    ap.add_argument("--ctx-chars", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--seed", type=int, default=42)

    # --- instrumentacao ---------------------------------------------------- #
    # Nenhuma destas altera treino, loss, scheduler, selecao de melhor epoca ou
    # qualquer hiperparametro: elas so decidem quais ARQUIVOS AUXILIARES sao
    # gravados depois que o treino acabou. Os defaults ligam tudo; a combinacao
    # `--save-split-preds test --no-save-probs --no-test-eval-log` reproduz
    # byte-a-byte a saida da versao anterior do modulo.
    ap.add_argument("--save-split-preds", choices=["test", "dev+test"],
                    default="dev+test",
                    help="Quais splits ganham sidecar de predicoes. 'dev+test' "
                         "(default) grava tambem <out>.dev_preds.json com o DEV "
                         "da MELHOR epoca, necessario para calibrar regras "
                         "pos-hoc sem tocar no test. 'test' reproduz o "
                         "comportamento antigo.")
    ap.add_argument("--save-probs", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Grava tambem a softmax por instancia nos sidecars "
                         "(campo 'probs', ordem de LABELS: negation_of, "
                         "associated_with, no_relation), arredondada a 4 casas. "
                         "Use --no-save-probs para o formato antigo.")
    ap.add_argument("--test-eval-log", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Anexa uma linha a <out>.test_evals.jsonl a cada "
                         "avaliacao final no TEST (auditoria de multiplicidade). "
                         "Use --no-test-eval-log para desligar.")

    # --- backup do checkpoint no Hugging Face Hub -------------------------- #
    # Camada EXTRA alem do --ckpt-dir no Drive. Sem --hf-backup-repo nada muda:
    # nenhum import de huggingface_hub, nenhuma chamada de rede.
    ap.add_argument("--hf-backup-repo", default=None,
                    help="Repositorio de modelo PRIVADO no HF Hub que recebe uma "
                         "copia de best_model/ a cada nova melhor epoca (ex.: "
                         "angeloalsf/reclin-pt-biobertpt-seed42). Omitido = "
                         "backup desligado. Exige --ckpt-dir.")
    ap.add_argument("--hf-token", default=None,
                    help="Token de ESCRITA do HF Hub. Se omitido, e lido de "
                         "$HF_TOKEN ou $HUGGING_FACE_HUB_TOKEN. O valor nunca e "
                         "registrado no log.")
    return ap


# --------------------------------------------------------------------------- #
# Pipeline principal                                                          #
# --------------------------------------------------------------------------- #
def run(args, logger=None):
    """Executa o baseline fim a fim. `logger` permite que cada entry-point
    registre o log com o nome do seu baseline ([MODULO] no log)."""
    global log
    if logger is not None:
        log = logger

    t0 = time.time()
    log.info("=== Baseline iniciado (model=%s) ===", args.model)
    log.info("Config: model=%s | epochs=%d | batch_size=%d | max_length=%d | "
             "lr=%g | class_weight=%s | max_gap=%d | ctx_chars=%d | seed=%d | ckpt_dir=%s",
             args.model, args.epochs, args.batch_size, args.max_length, args.lr,
             args.class_weight, args.max_gap, args.ctx_chars, args.seed, args.ckpt_dir)
    log.info("Instrumentacao: save_split_preds=%s | save_probs=%s | "
             "test_eval_log=%s", args.save_split_preds, args.save_probs,
             args.test_eval_log)

    # ----- backup do melhor checkpoint no HF Hub (opcional) -----
    # Sem --hf-backup-repo este bloco inteiro nao roda: `hf_backup` sequer e
    # importado. Com a flag, o repo privado e criado ja aqui -- antes de carregar
    # dados e modelo -- para que um token invalido apareca AGORA e nao depois de
    # uma hora de treino. Nada aqui pode levantar: ver src/hf_backup.py.
    hf_backup = None
    if args.hf_backup_repo:
        from hf_backup import HFBackup, resolve_token  # noqa: E402
        hf_token, token_origem = resolve_token(args.hf_token)
        log.info("Backup HF: repo=%s | token=%s", args.hf_backup_repo,
                 ("presente (via %s)" % token_origem) if hf_token
                 else "AUSENTE (sera tentado o login em cache, se houver)")
        if not args.ckpt_dir:
            log.warning("Backup HF: --hf-backup-repo foi passado mas --ckpt-dir NAO. "
                        "O best_model/ so e gravado (e so e enviado) quando ha "
                        "--ckpt-dir, entao NENHUM backup sera feito nesta execucao. "
                        "Passe --ckpt-dir para ativar o backup.")
        else:
            hf_backup = HFBackup(args.hf_backup_repo, hf_token, logger=log)
        del hf_token  # o token nao precisa continuar vivo neste escopo

    import torch
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, matthews_corrcoef)
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info("Dispositivo: CUDA (%s)", torch.cuda.get_device_name(0))
    else:
        log.warning("Dispositivo: CPU (sem GPU) -- treino sera lento")

    # ----- dados -----
    sp = Path(args.splits_dir)
    log.info("Carregando splits de %s", sp)
    Xtr, ytr = build_dataset(list(read_jsonl(sp / "train.jsonl")), args.max_gap, args.ctx_chars)
    Xdv, ydv = build_dataset(list(read_jsonl(sp / "dev.jsonl")), args.max_gap, args.ctx_chars)
    Xte, yte = build_dataset(list(read_jsonl(sp / "test.jsonl")), args.max_gap, args.ctx_chars)
    log.info("Candidatos gerados: train=%d | dev=%d | test=%d", len(ytr), len(ydv), len(yte))
    tr_dist = {l: int(np.sum(np.array(ytr) == LABEL2ID[l])) for l in LABELS}
    log.info("Distribuicao (train): %s", tr_dist)

    # ----- tokenizer -----
    log.info("Carregando tokenizer: %s", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    n_added = tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    log.info("Tokenizer carregado | tokens de marcacao adicionados: %d", n_added)

    # ----- modelo -----
    log.info("Carregando modelo (3 classes): %s", args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    model.resize_token_embeddings(len(tok))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Modelo carregado | parametros: %.1fM | vocab: %d", n_params / 1e6, len(tok))

    tr_loader = make_loader(tok, Xtr, ytr, args.max_length, args.batch_size, True, args.seed)
    dv_loader = make_loader(tok, Xdv, ydv, args.max_length, args.batch_size, False, args.seed)
    te_loader = make_loader(tok, Xte, yte, args.max_length, args.batch_size, False, args.seed)

    # ----- loss com peso de classe -----
    if args.class_weight == "balanced":
        counts = np.bincount(ytr, minlength=len(LABELS)).astype(float)
        counts[counts == 0] = 1.0
        w = counts.sum() / (len(LABELS) * counts)
        weights = torch.tensor(w, dtype=torch.float).to(device)
        log.info("Pesos de classe (balanced): %s",
                 {l: round(float(w[i]), 3) for i, l in enumerate(LABELS)})
    else:
        weights = None
        log.info("Sem pesos de classe (class_weight=none)")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(args.warmup_ratio * total), total)
    log.info("Otimizador AdamW | passos totais=%d | warmup=%d",
             total, int(args.warmup_ratio * total))

    # ----- retomada de checkpoint -----
    start_epoch, best_f1, best_state, history = 1, -1.0, None, []
    if args.ckpt_dir:
        loaded = load_last_checkpoint(args.ckpt_dir, model=model, optimizer=opt,
                                      scheduler=sched, args=args)
        if loaded is not None:
            start_epoch, best_f1, history = loaded

    ydv_str = [ID2LABEL[i] for i in ydv]

    if start_epoch > args.epochs:
        log.info("Todas as %d epocas ja concluidas (checkpoint) -- pulando treino, "
                 "indo direto para avaliacao final.", args.epochs)
    else:
        log.info("--- Treino: epocas %d..%d (%d passos/epoca) ---",
                 start_epoch, args.epochs, len(tr_loader))
        for ep in range(start_epoch, args.epochs + 1):
            ep_t0 = time.time()
            log.info("Epoca %d/%d: inicio", ep, args.epochs)
            model.train()
            running = 0.0
            for step, (ids, attn, lab) in enumerate(tr_loader, 1):
                opt.zero_grad()
                logits = model(input_ids=ids.to(device),
                               attention_mask=attn.to(device)).logits
                loss = loss_fn(logits, lab.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                running += loss.item()
                if step % 20 == 0:
                    log.info("  epoca %d | passo %d/%d | loss media=%.4f",
                             ep, step, len(tr_loader), running / step)

            train_loss = running / max(1, len(tr_loader))
            log.info("Epoca %d: avaliando no dev...", ep)
            dev_pred_ids, dev_loss = evaluate(model, dv_loader, device, loss_fn)
            dev_pred = [ID2LABEL[i] for i in dev_pred_ids]
            macro = f1_score(ydv_str, dev_pred, labels=LABELS, average="macro", zero_division=0)
            neg = f1_score(ydv_str, dev_pred, labels=["negation_of"], average="macro", zero_division=0)
            dur = time.time() - ep_t0
            history.append({"epoch": ep, "train_loss": train_loss, "dev_loss": dev_loss,
                            "dev_macro_f1": macro, "dev_negation_of_f1": neg,
                            "duration_s": round(dur, 1)})
            log.info("Epoca %d: fim | train_loss=%.4f | dev_loss=%.4f | dev_macro_f1=%.4f | "
                     "dev_negation_of_f1=%.4f | duracao=%.1fs",
                     ep, train_loss, dev_loss, macro, neg, dur)
            if macro > best_f1:
                best_f1 = macro
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                log.info("Epoca %d: novo MELHOR modelo (dev_macro_f1=%.4f)", ep, macro)
                if args.ckpt_dir:
                    save_best_model(args.ckpt_dir, model, tok,
                                    hf_backup=hf_backup,
                                    hf_note="epoca %d | dev_macro_f1=%.4f | seed=%d"
                                            % (ep, macro, args.seed))

            # last_checkpoint por epoca (apos avaliar e atualizar o melhor)
            if args.ckpt_dir:
                save_last_checkpoint(args.ckpt_dir, epoch=ep, model=model, optimizer=opt,
                                     scheduler=sched, best_f1=best_f1, history=history, args=args)

    # ----- restaura melhor epoca e avalia no test -----
    if best_state is None and args.ckpt_dir:
        best_state = load_best_state(args.ckpt_dir)
        if best_state is not None:
            log.info("Melhores pesos recuperados de best_model/ (sem novo melhor nesta sessao)")
    restored_best_state = bool(best_state)
    if best_state:
        model.load_state_dict(best_state)
        log.info("Melhor modelo (dev_macro_f1=%.4f) restaurado para avaliacao final", best_f1)

    # ----- predicoes do DEV na MELHOR epoca (ANTES de tocar no test) -----
    # Roda aqui, logo depois do load_state_dict(best_state), para que os pesos
    # que produzem estas predicoes sejam inequivocamente os mesmos que serao
    # reportados no test -- e nao os da ultima epoca de treino.
    #
    # Por que isto nao muda o resultado do test: `predict` roda em eval() e sob
    # no_grad (sem dropout) e nao consome o RNG global; a criacao do iterador do
    # DataLoader consome apenas o `torch.Generator` proprio do `dv_loader`, e o
    # `te_loader` tem o seu. Logo as predicoes do test saem identicas as da
    # versao anterior do modulo.
    best_epoch, best_epoch_f1 = best_epoch_from_history(history)
    dev_pred_ids, dev_probs, dev_macro_recomputed = None, None, None
    if args.save_split_preds == "dev+test":
        log.info("--- Predicoes do DEV na melhor epoca (epoca=%s) ---", best_epoch)
        if not restored_best_state:
            log.warning("Nenhum best_state para restaurar: as predicoes do dev "
                        "correspondem aos pesos ATUAIS (ultima epoca), nao a "
                        "melhor epoca. O sidecar registra "
                        "'restored_best_state': false -- nao calibre nada em "
                        "cima dele sem verificar.")
        if args.save_probs:
            dev_pred_ids, dev_probs = predict(model, dv_loader, device,
                                              return_probs=True)
        else:
            dev_pred_ids = predict(model, dv_loader, device)
        dev_macro_recomputed = float(f1_score(
            ydv_str, [ID2LABEL[i] for i in dev_pred_ids],
            labels=LABELS, average="macro", zero_division=0))
        log.info("DEV (melhor epoca) macro-F1 recomputado=%.4f", dev_macro_recomputed)
        # Guarda-corpo barato: se os pesos restaurados forem mesmo os da melhor
        # epoca, este numero tem de bater com o que ficou no dev_history.
        if best_epoch_f1 is not None and abs(dev_macro_recomputed - best_epoch_f1) > 1e-6:
            log.warning("macro-F1 do dev recomputado (%.6f) DIFERE do registrado "
                        "no dev_history para a epoca %s (%.6f). Os pesos avaliados "
                        "podem nao ser os da melhor epoca -- investigue antes de "
                        "calibrar qualquer regra sobre este sidecar.",
                        dev_macro_recomputed, best_epoch, best_epoch_f1)

    log.info("--- Avaliacao final no TEST ---")
    if args.save_probs:
        y_pred_ids, te_probs = predict(model, te_loader, device, return_probs=True)
    else:
        y_pred_ids, te_probs = predict(model, te_loader, device), None
    y_pred = [ID2LABEL[i] for i in y_pred_ids]
    y_true = [ID2LABEL[i] for i in yte]

    macro = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, labels=LABELS, average="micro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_class = {l: float(per_class[i]) for i, l in enumerate(LABELS)}
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_list = cm.tolist()
    log.info("Matriz de confusao (linhas=verdadeiro, colunas=predito):\n%s",
             render_confusion_matrix(cm, LABELS))

    result = {
        "model": args.model, "seed": args.seed, "device": str(device),
        "config": {"max_gap": args.max_gap, "ctx_chars": args.ctx_chars,
                   "max_length": args.max_length, "epochs": args.epochs,
                   "batch_size": args.batch_size, "lr": args.lr,
                   "class_weight": args.class_weight},
        "n_params": int(n_params),
        "n_candidates": {"train": len(ytr), "dev": len(ydv), "test": len(yte)},
        "dev_history": history,
        "test_macro_f1": macro, "test_micro_f1": micro,
        "test_weighted_f1": weighted, "test_mcc": mcc,
        "test_f1_per_class": per_class,
        "sklearn_report": report,
        "confusion_matrix": {"labels": LABELS, "matrix": cm_list},
        # Rastreabilidade das flags de instrumentacao. Fica FORA de `config` de
        # proposito: scripts/_artifacts.py:load_pipeline_config() exige que o
        # `config` das 4 execucoes seja IDENTICO e aborta a geracao dos
        # artefatos do TCC se divergir. Rodar uma execucao com --no-save-probs
        # (por espaco, por exemplo) travaria `make` por uma diferenca que nao
        # tem efeito nenhum sobre os numeros -- um alarme falso caro.
        "instrumentation": {
            "save_split_preds": args.save_split_preds,
            "save_probs": bool(args.save_probs),
            "test_eval_log": bool(args.test_eval_log),
            "best_epoch": best_epoch,
            "restored_best_state": restored_best_state,
            # Onde esta o backup dos pesos que produziram estes numeros. Nunca
            # contem o token -- so o nome do repo (None quando desligado).
            "hf_backup_repo": args.hf_backup_repo,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                         sort_keys=True), encoding="utf-8")
    log.info("Resultados salvos em %s", args.out)

    # ----- predicoes do TEST (sidecar) para o teste de significancia -----
    preds_path = Path(args.out).with_suffix(".preds.json")
    preds_payload = build_preds_payload(args=args, y_true_ids=yte,
                                        y_pred_ids=y_pred_ids, probs=te_probs)
    preds_path.write_text(json.dumps(preds_payload, ensure_ascii=False), encoding="utf-8")
    log.info("Predicoes do test salvas em %s (para src/significance.py)%s",
             preds_path, " [com probs]" if te_probs is not None else "")

    # ----- predicoes do DEV na melhor epoca (sidecar de CALIBRACAO) -----
    if dev_pred_ids is not None:
        dev_preds_path = Path(args.out).with_suffix(".dev_preds.json")
        dev_payload = build_preds_payload(
            args=args, y_true_ids=ydv, y_pred_ids=dev_pred_ids, probs=dev_probs,
            extra={"split": "dev",
                   "best_epoch": best_epoch,
                   "best_dev_macro_f1_history": best_epoch_f1,
                   "dev_macro_f1_recomputed": dev_macro_recomputed,
                   "restored_best_state": restored_best_state})
        dev_preds_path.write_text(json.dumps(dev_payload, ensure_ascii=False),
                                  encoding="utf-8")
        log.info("Predicoes do dev (melhor epoca=%s) salvas em %s -- calibre "
                 "regras pos-hoc AQUI, nunca no test", best_epoch, dev_preds_path)

    # ----- registro de auditoria da avaliacao no TEST -----
    if args.test_eval_log:
        append_test_eval_log(args.out, args=args, n_test=len(yte),
                             macro_f1=macro,
                             negation_of_f1=per_class["negation_of"])

    # ----- resumo final -----
    log.info("=== RESULTADO (test) ===")
    log.info("Macro-F1=%.4f | Micro-F1=%.4f | Weighted-F1=%.4f | MCC=%.4f",
             macro, micro, weighted, mcc)
    for l in LABELS:
        marca = "  <<< NEGACAO" if l == "negation_of" else ""
        log.info("  F1 %-18s %.4f%s", l, per_class[l], marca)
    log.info("Tempo total: %.1fs", time.time() - t0)

    # Espera o ultimo envio ao HF Hub terminar. Fica DEPOIS do resumo de
    # proposito: os resultados aparecem no log na hora, sem esperar a rede.
    if hf_backup is not None:
        hf_backup.close()
    return 0
