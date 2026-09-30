#!/usr/bin/env python3
"""
Fine-tuning dedicado no ESPACO RESTRITO de candidatos (e1 e pista de negacao).

FRENTE EXPERIMENTAL E ISOLADA
-----------------------------
Nada aqui altera o Cap. 6 nem a fase 2. O modulo so LE `src/` (importando, nunca
copiando) e so GRAVA em `results/finetuning_restrito/` e no `--ckpt-dir`. Uma
saida apontada para qualquer outro lugar de `results/` e recusada.

O QUE E REUSADO DE `src/relation_extraction.py` (import, nao copia)
--------------------------------------------------------------------
`build_marked_window` (via `restricted_space`), `set_all_seeds` (determinismo:
cuDNN deterministico + `use_deterministic_algorithms(True, warn_only=True)`),
`CUBLAS_WORKSPACE_CONFIG`/`PYTHONHASHSEED` (setados no import do modulo),
`collect_environment`, `make_loader`, `predict`, `evaluate`, `save_best_model`
(que dispara o backup no HF Hub), `save_last_checkpoint`,
`load_last_checkpoint`, `load_best_state`, `best_epoch_from_history`,
`build_preds_payload`, `render_confusion_matrix` e o proprio
`build_arg_parser`, para que as flags e o significado delas sejam os mesmos
dos baselines.

O laco de treino em si e reescrito porque `run()` monta o dataset do espaco
completo por dentro e nao aceita outro. A reescrita segue `run()` passo a passo:
mesma ordem de sementes, dados, tokenizer, modelo e loaders (a ordem importa para
o consumo de RNG), mesmo otimizador, scheduler, clipping e criterio de melhor
epoca.

O QUE MUDA EM RELACAO AOS BASELINES (e so isto)
-----------------------------------------------
1. Os exemplos de treino, dev e test sao so os do espaco restrito.
2. `--epochs` default 10 (era 3): cada epoca agora tem ~107 passos, nao 2.386.
3. `class_weight=balanced` e recalculado no train RESTRITO. Os pesos efetivos vao
   para o JSON (`restricted_space.pesos_classe_efetivos`) porque essa mudanca nao
   aparece em nenhuma flag.
4. A melhor epoca e escolhida pelo macro-F1 do DEV REMAPEADO ao espaco completo
   (19.064 pares), o mesmo criterio e o mesmo conjunto dos baselines. Os numeros
   do dev restrito tambem ficam no `dev_history`, como informacao.

SAIDAS (todas em results/finetuning_restrito/)
----------------------------------------------
- `<out>.json`            metricas no TEST remapeado + bloco `restricted_space`
                          + `environment` (mesmo formato dos baselines).
- `<out>.preds.json`      TEST remapeado ao espaco completo (19.210), contrato de
                          `src/significance.py`.
- `<out>.dev_preds.json`  DEV da melhor epoca, remapeado (19.064). E o insumo do
                          criterio de parada.
- `<out>.test_evals.jsonl` trilha de multiplicidade.
- `<out>.train_log.txt`   log completo da execucao, INCLUINDO os UserWarning de
                          `use_deterministic_algorithms`, que nunca foram
                          capturados nas rodadas dos baselines. (`*.log` esta no
                          .gitignore; `.txt` nao, entao o log e versionavel.)

Uso (Colab T4):
    python src/finetuning_restrito/train_restrito.py --encoder bertimbau \
        --seed 42 --ckpt-dir /content/drive/MyDrive/RECLin-PT/checkpoints_restrito_bertimbau_seed42 \
        --hf-backup-repo angeloalsf/reclin-pt-restrito-bertimbau-seed42
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolamento  # noqa: E402  (antes de qualquer import do nucleo)
from _isolamento import REPO, RESULTS_DIR, exigir_saida_isolada  # noqa: E402

import relation_extraction as core  # noqa: E402
from relation_extraction import (ID2LABEL, LABEL2ID, LABELS,  # noqa: E402
                                 MARKER_TOKENS, best_epoch_from_history,
                                 build_arg_parser, build_preds_payload,
                                 collect_environment, evaluate, load_best_state,
                                 load_last_checkpoint, make_loader, predict,
                                 read_jsonl, render_confusion_matrix,
                                 save_best_model, save_last_checkpoint,
                                 set_all_seeds)
from restricted_space import (CTX_CHARS, MAX_GAP, MIN_FREQ,  # noqa: E402
                              carregar_espacos, carregar_lexico, conferir_y_true,
                              lexico_sha1, pesos_balanced)

log = _isolamento.get()

ENCODERS = {
    "biobertpt": "pucpr/biobertpt-all",
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
}
OUT_TEMPLATE = "restrito_{encoder}_seed{seed}.json"
GUARD_FILE = "finetuning_restrito_guard.json"
FORMATO_VERSAO = 1


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser():
    ap = build_arg_parser(default_model=None)
    ap.description = __doc__.split("\n")[1]
    ap.add_argument("--encoder", choices=sorted(ENCODERS), required=True,
                    help="Encoder de pre-treino. Define --model (salvo se --model "
                         "for passado explicitamente) e o nome do arquivo de saida.")
    ap.add_argument("--lexicon-min-freq", type=int, default=MIN_FREQ,
                    help="min_freq do lexico (default 3 = calibracao da fase 2; "
                         "nao recalibrar aqui).")
    ap.add_argument("--lexicon-guard",
                    default=str(REPO / "results" / "CALIBRACAO_filtro.json"),
                    help="JSON da calibracao congelada da fase 2, para conferir "
                         "min_freq e tamanho do lexico. Ausente = so aviso.")
    ap.add_argument("--baseline-dir", default=str(REPO / "results"),
                    help="Onde estao os sidecars dos baselines, usados so para "
                         "conferir que o y_true reconstruido e o mesmo.")
    # Defaults = configuracao congelada dos baselines (results/baseline_*.json),
    # nao os defaults historicos do argparse do nucleo (75/192/32/3).
    ap.set_defaults(max_gap=MAX_GAP, ctx_chars=CTX_CHARS, max_length=128,
                    batch_size=64, epochs=10, splits_dir=str(REPO / "data" / "splits"))
    return ap


def resolver_args(args):
    if args.model is None:
        args.model = ENCODERS[args.encoder]
    if args.out is None:
        args.out = str(RESULTS_DIR / OUT_TEMPLATE.format(encoder=args.encoder,
                                                         seed=args.seed))
    exigir_saida_isolada(args.out)
    if args.ckpt_dir:
        exigir_saida_isolada(args.ckpt_dir)
    if args.hf_backup_repo and "restrito" not in args.hf_backup_repo:
        raise SystemExit(
            f"--hf-backup-repo {args.hf_backup_repo} recusado: o nome precisa "
            f"conter 'restrito' para nunca misturar com os repos dos baselines "
            f"(angeloalsf/reclin-pt-<modelo>-seed<N>). Use, por exemplo, "
            f"angeloalsf/reclin-pt-restrito-{args.encoder}-seed{args.seed}.")
    return args


def config_restrito(args, lex_sha):
    """Configuracao que define o experimento (entra no `config` e no hash)."""
    return {"max_gap": args.max_gap, "ctx_chars": args.ctx_chars,
            "max_length": args.max_length, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
            "class_weight": args.class_weight,
            "espaco": "restrito: e1 no lexico induzido do train",
            "lexicon_min_freq": args.lexicon_min_freq, "lexicon_sha1": lex_sha,
            "selecao_melhor_epoca": "dev_macro_f1 remapeado ao espaco completo",
            "remapeamento": "fora do espaco restrito -> no_relation"}


# --------------------------------------------------------------------------- #
# Guarda do --ckpt-dir                                                        #
# --------------------------------------------------------------------------- #
def checar_ckpt_dir(args, cfg):
    """Impede que esta frente retome (ou sobrescreva) checkpoint de outra.

    O caso perigoso e apontar --ckpt-dir para a pasta de um baseline: a guarda de
    config do nucleo veria epochs 3 != 10, descartaria o last_checkpoint em
    silencio e, na primeira melhor epoca, `save_best_model` faria rmtree no
    best_model/ DO BASELINE. Aqui: pasta com best_model/ ou last_checkpoint/ e
    sem o arquivo de guarda desta frente -> aborta. Com o arquivo, ele precisa
    bater com a config atual.
    """
    ck = Path(args.ckpt_dir)
    guard = ck / GUARD_FILE
    tem_estado = (ck / core.BEST_MODEL_DIR).exists() or (ck / core.LAST_CKPT_DIR).exists()
    want = {"formato": FORMATO_VERSAO, "encoder": args.encoder, "model": args.model,
            "seed": args.seed, "config": cfg}
    if guard.is_file():
        have = json.loads(guard.read_text(encoding="utf-8"))
        if have != want:
            diff = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
            raise SystemExit(
                f"{guard} foi gravado por outra configuracao desta frente "
                f"(chaves divergentes: {diff}). Use outro --ckpt-dir.")
    elif tem_estado:
        raise SystemExit(
            f"{ck} ja tem best_model/ ou last_checkpoint/ mas nao tem {GUARD_FILE}: "
            f"provavelmente e a pasta de um BASELINE. Recusado para nao "
            f"sobrescrever o best_model/ dele. Use uma pasta nova, ex. "
            f".../checkpoints_restrito_{args.encoder}_seed{args.seed}.")
    else:
        ck.mkdir(parents=True, exist_ok=True)
        guard.write_text(json.dumps(want, ensure_ascii=False, indent=2,
                                    sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Metricas                                                                    #
# --------------------------------------------------------------------------- #
def metricas(y_true_ids, y_pred_ids):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, matthews_corrcoef)
    yt = [ID2LABEL[i] for i in y_true_ids]
    yp = [ID2LABEL[i] for i in y_pred_ids]
    per = f1_score(yt, yp, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, yp, labels=LABELS, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, labels=LABELS, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "f1_per_class": {l: float(per[i]) for i, l in enumerate(LABELS)},
        "report": classification_report(yt, yp, labels=LABELS, zero_division=0,
                                        output_dict=True),
        "cm": confusion_matrix(yt, yp, labels=LABELS),
    }


def append_test_eval_log(out_path, *, args, cfg, n_test, macro_f1, negation_of_f1):
    """Mesmo formato de `relation_extraction.append_test_eval_log`, com o
    contador `eval_index_for_config` da fase 2. O hash inclui a configuracao do
    espaco restrito (a versao do nucleo so enxerga `_config_guard`, que nao
    distingue esta frente de um baseline de 10 epocas)."""
    log_path = Path(out_path).with_suffix(".test_evals.jsonl")
    full_cfg = dict(cfg, model=args.model, seed=args.seed,
                    splits_dir=Path(args.splits_dir).name)
    digest = hashlib.sha1(json.dumps(full_cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    linhas = []
    if log_path.is_file():
        linhas = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    mesmo = sum(1 for ln in linhas if json.loads(ln).get("config_sha1") == digest)
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "eval_index": len(linhas) + 1, "eval_index_for_config": mesmo + 1,
             "out": str(out_path), "model": args.model, "seed": args.seed,
             "config_sha1": digest, "n_test": int(n_test),
             "test_macro_f1": round(float(macro_f1), 6),
             "test_negation_of_f1": round(float(negation_of_f1), 6),
             "espaco": "restrito remapeado ao completo"}
    with open(log_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if entry["eval_index_for_config"] > 1:
        log.warning("ATENCAO (multiplicidade): avaliacao #%d no TEST para a "
                    "config %s em %s -- declare ao reportar.",
                    entry["eval_index_for_config"], digest, log_path)
    return entry


def environment_extra():
    import importlib
    import os

    def versao(nome):
        try:
            return getattr(importlib.import_module(nome), "__version__", None)
        except Exception:
            return None

    # os.environ["PYTHONHASHSEED"] sempre le "0" aqui, porque relation_extraction
    # faz setdefault no import; o que diz se ele VALEU e a flag do interpretador,
    # fixada na partida do processo.
    return {"huggingface_hub": versao("huggingface_hub"),
            "tokenizers": versao("tokenizers"),
            "lxml": versao("lxml"),
            "matplotlib": versao("matplotlib"),
            "pythonhashseed_valeu_no_processo": sys.flags.hash_randomization == 0,
            "pythonhashseed_no_ambiente": os.environ.get("PYTHONHASHSEED")}


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def run(args):
    t0 = time.time()
    out = Path(args.out)

    # Lexico e guarda do --ckpt-dir ANTES de abrir o log da execucao: uma
    # execucao recusada nao deixa arquivo nenhum em results/finetuning_restrito/.
    sp = Path(args.splits_dir)
    lex = carregar_lexico(read_jsonl(sp / "train.jsonl"), max_gap=args.max_gap,
                          min_freq=args.lexicon_min_freq, guarda=args.lexicon_guard)
    lex_sha = lexico_sha1(lex)
    cfg = config_restrito(args, lex_sha)
    if args.ckpt_dir:
        checar_ckpt_dir(args, cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    _isolamento.anexar_arquivo(out.with_suffix(".train_log.txt"))
    coletor = _isolamento.capturar_avisos()
    # save_best_model/save_last_checkpoint logam pelo `log` global do nucleo; ele
    # ja aponta para o logger "relation_extraction", que _isolamento desviou de
    # logs/pipeline.log para o console + o log desta execucao.

    log.info("=== Fine-tuning RESTRITO iniciado (encoder=%s, model=%s, seed=%d) ===",
             args.encoder, args.model, args.seed)
    log.info("Lexico: %d formas, min_freq=%d, sha1=%s: %s", len(lex),
             args.lexicon_min_freq, lex_sha, ", ".join(lex))
    log.info("Config: %s", json.dumps(cfg, ensure_ascii=False, sort_keys=True))

    hf_backup = None
    if args.hf_backup_repo:
        from hf_backup import HFBackup, resolve_token
        token, origem = resolve_token(args.hf_token)
        log.info("Backup HF: repo=%s | token=%s", args.hf_backup_repo,
                 ("presente (via %s)" % origem) if token else "AUSENTE")
        if not args.ckpt_dir:
            log.warning("Backup HF pedido sem --ckpt-dir: nenhum backup sera feito.")
        else:
            hf_backup = HFBackup(args.hf_backup_repo, token, logger=log)
        del token

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    # Mesma ordem de run(): sementes -> dados -> tokenizer -> modelo -> loaders.
    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info("Dispositivo: CUDA (%s)", torch.cuda.get_device_name(0))
    else:
        log.warning("Dispositivo: CPU (sem GPU)")

    esp = carregar_espacos(sp, lex, max_gap=args.max_gap, ctx_chars=args.ctx_chars)
    for split, nome in (("dev", "dev_preds"), ("test", "preds")):
        for enc in ENCODERS:
            ref = Path(args.baseline_dir) / f"baseline_{enc}_seed42.{nome}.json"
            if ref.is_file():
                if not conferir_y_true(esp[split], ref):
                    raise SystemExit(f"y_true reconstruido do {split} difere de {ref}: "
                                     f"o remapeamento nao seria comparavel.")
                log.info("y_true do %s confere com %s", split, ref.name)
                break
        else:
            log.warning("Nenhum sidecar de baseline para conferir o y_true do %s.", split)

    tr, dv, te = esp["train"], esp["dev"], esp["test"]
    log.info("Distribuicao (train restrito): %s", tr.contagens()["restrito"])

    tok = AutoTokenizer.from_pretrained(args.model)
    n_added = tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    log.info("Tokenizer %s | marcadores adicionados: %d", args.model, n_added)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    model.resize_token_embeddings(len(tok))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Modelo carregado | parametros: %.1fM", n_params / 1e6)

    tr_loader = make_loader(tok, tr.textos, tr.y_restrito, args.max_length, args.batch_size, True, args.seed)
    dv_loader = make_loader(tok, dv.textos, dv.y_restrito, args.max_length, args.batch_size, False, args.seed)
    te_loader = make_loader(tok, te.textos, te.y_restrito, args.max_length, args.batch_size, False, args.seed)

    if args.class_weight == "balanced":
        w = pesos_balanced(tr.y_restrito)
        weights = torch.tensor(w, dtype=torch.float).to(device)
    else:
        w, weights = None, None
    pesos_efetivos = ({l: round(float(w[i]), 6) for i, l in enumerate(LABELS)}
                      if w is not None else None)
    log.info("Pesos de classe efetivos (%s, recalculados no train RESTRITO): %s",
             args.class_weight, pesos_efetivos)
    raros = {l: n for l, n in tr.contagens()["restrito"].items() if 0 < n < 50}
    if raros and w is not None:
        log.warning("Classe(s) rara(s) no train restrito %s: o balanced da a elas "
                    "peso %s. Registrado no JSON; leia junto com o resultado.",
                    raros, {l: pesos_efetivos[l] for l in raros})
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    log.info("AdamW | passos totais=%d | warmup=%d | passos/epoca=%d",
             total, int(args.warmup_ratio * total), len(tr_loader))

    start_epoch, best_f1, best_state, history = 1, -1.0, None, []
    if args.ckpt_dir:
        loaded = load_last_checkpoint(args.ckpt_dir, model=model, optimizer=opt,
                                      scheduler=sched, args=args)
        if loaded is not None:
            start_epoch, best_f1, history = loaded

    from sklearn.metrics import f1_score

    def f1s(y_true_ids, y_pred_ids):
        yt = [ID2LABEL[i] for i in y_true_ids]
        yp = [ID2LABEL[i] for i in y_pred_ids]
        return (float(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)),
                float(f1_score(yt, yp, labels=["negation_of"], average="macro", zero_division=0)))

    for ep in range(start_epoch, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        running = 0.0
        for step, (ids, attn, lab) in enumerate(tr_loader, 1):
            opt.zero_grad()
            logits = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
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

        dev_pred_r, dev_loss = evaluate(model, dv_loader, device, loss_fn)
        dev_pred_full, _ = dv.remapear(dev_pred_r)
        macro, neg = f1s(dv.y_true_full, dev_pred_full)
        macro_r, neg_r = f1s(dv.y_restrito, dev_pred_r)
        dur = time.time() - ep_t0
        history.append({"epoch": ep, "train_loss": train_loss,
                        "dev_loss": dev_loss,              # so no espaco restrito
                        "dev_macro_f1": macro,             # remapeado (selecao)
                        "dev_negation_of_f1": neg,         # remapeado
                        "dev_restrito_macro_f1": macro_r,
                        "dev_restrito_negation_of_f1": neg_r,
                        "duration_s": round(dur, 1)})
        log.info("Epoca %d: train_loss=%.4f | dev_loss(restrito)=%.4f | "
                 "dev_macro_f1(remap)=%.4f | dev_negation_of_f1(remap)=%.4f | "
                 "restrito: macro=%.4f neg=%.4f | %.1fs",
                 ep, train_loss, dev_loss, macro, neg, macro_r, neg_r, dur)
        if macro > best_f1:
            best_f1 = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log.info("Epoca %d: novo MELHOR modelo (dev_macro_f1 remapeado=%.4f)", ep, macro)
            if args.ckpt_dir:
                save_best_model(args.ckpt_dir, model, tok, hf_backup=hf_backup,
                                hf_note="restrito | epoca %d | dev_macro_f1=%.4f | "
                                        "dev_neg_f1=%.4f | seed=%d"
                                        % (ep, macro, neg, args.seed))
        if args.ckpt_dir:
            save_last_checkpoint(args.ckpt_dir, epoch=ep, model=model, optimizer=opt,
                                 scheduler=sched, best_f1=best_f1, history=history, args=args)

    if best_state is None and args.ckpt_dir:
        best_state = load_best_state(args.ckpt_dir)
    restored = bool(best_state)
    if best_state:
        model.load_state_dict(best_state)
        log.info("Melhor epoca restaurada (dev_macro_f1 remapeado=%.4f)", best_f1)
    best_epoch, best_epoch_f1 = best_epoch_from_history(history)

    # DEV da melhor epoca ANTES do test (mesma disciplina de run()).
    dev_pred_r, dev_probs_r = predict(model, dv_loader, device, return_probs=True)
    dev_pred_full, dev_probs_full = dv.remapear(dev_pred_r, dev_probs_r)
    dev_macro_rec, dev_neg_rec = f1s(dv.y_true_full, dev_pred_full)
    if best_epoch_f1 is not None and abs(dev_macro_rec - best_epoch_f1) > 1e-6:
        log.warning("macro-F1 do dev recomputado (%.6f) difere do dev_history (%.6f): "
                    "os pesos avaliados podem nao ser os da melhor epoca.",
                    dev_macro_rec, best_epoch_f1)

    te_pred_r, te_probs_r = predict(model, te_loader, device, return_probs=True)
    te_pred_full, te_probs_full = te.remapear(te_pred_r, te_probs_r)
    m = metricas(te.y_true_full, te_pred_full)
    m_r = metricas(te.y_restrito, te_pred_r)
    log.info("Matriz de confusao TEST remapeado (linhas=verdadeiro):\n%s",
             render_confusion_matrix(m["cm"], LABELS))

    resumo_esp = {s: e.resumo() for s, e in esp.items()}
    result = {
        "model": args.model, "encoder": args.encoder, "seed": args.seed,
        "device": str(device),
        "config": cfg,
        "n_params": int(n_params),
        "n_candidates": {s: e.n_full for s, e in esp.items()},
        "dev_history": history,
        "test_macro_f1": m["macro_f1"], "test_micro_f1": m["micro_f1"],
        "test_weighted_f1": m["weighted_f1"], "test_mcc": m["mcc"],
        "test_f1_per_class": m["f1_per_class"],
        "sklearn_report": m["report"],
        "confusion_matrix": {"labels": LABELS, "matrix": m["cm"].tolist()},
        "restricted_space": {
            "lexico": {"fonte": "negation_lexicon.induce_lexicon(train)",
                       "max_gap": args.max_gap, "min_freq": args.lexicon_min_freq,
                       "n_formas": len(lex), "sha1": lex_sha, "formas": lex},
            "n_candidates_restrito": {s: e.n_restrito for s, e in esp.items()},
            "por_split": resumo_esp,
            "teto_recall_negation_of": {s: r["teto_recall_negation_of"]
                                        for s, r in resumo_esp.items()},
            "class_weight": args.class_weight,
            "pesos_classe_efetivos": pesos_efetivos,
            "pesos_classe_espaco_completo_baselines": {
                "negation_of": 40.55, "associated_with": 7.17, "no_relation": 0.3526},
            "remapeamento": {"fora_do_espaco": "no_relation",
                             "probs_fora_do_espaco": [0.0, 0.0, 1.0],
                             "y_true": "gold do espaco completo, nunca remapeado"},
            "ressalva_metricas": (
                "associated_with fica quase todo fora do espaco restrito: F1 de "
                "associated_with, macro-F1, micro/weighted, MCC e o McNemar de "
                "significance.py NAO sao comparaveis aos dos baselines. A "
                "metrica comparavel e o F1 de negation_of."),
            "test_restrito": {"macro_f1": m_r["macro_f1"],
                              "f1_per_class": m_r["f1_per_class"],
                              "confusion_matrix": m_r["cm"].tolist()},
        },
        "instrumentation": {
            "save_split_preds": "dev+test", "save_probs": True, "test_eval_log": True,
            "best_epoch": best_epoch, "best_epoch_is_last": best_epoch == args.epochs,
            "restored_best_state": restored,
            "dev_macro_f1_recomputed": dev_macro_rec,
            "dev_negation_of_f1_recomputed": dev_neg_rec,
            "hf_backup_repo": args.hf_backup_repo,
            "avisos_capturados": coletor.mensagens,
        },
        "environment": collect_environment(),
        # Irma de `environment`, que fica no formato EXATO dos baselines. Fecha a
        # lacuna registrada em 20/09 (collect_environment nao grava
        # huggingface_hub/lxml/matplotlib, pinados no requirements.txt) e
        # confirma se PYTHONHASHSEED foi exportado ANTES do processo, que e o
        # unico jeito de ele valer.
        "environment_extra": environment_extra(),
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8", newline="\n")

    extra_comum = {"espaco": "restrito remapeado ao completo",
                   "encoder": args.encoder, "best_epoch": best_epoch,
                   "restored_best_state": restored}
    te_payload = build_preds_payload(
        args=args, y_true_ids=te.y_true_full, y_pred_ids=te_pred_full,
        probs=te_probs_full,
        extra=dict(extra_comum, split="test", n_restrito=te.n_restrito,
                   restricted_indices=te.indices))
    out.with_suffix(".preds.json").write_text(
        json.dumps(te_payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    dv_payload = build_preds_payload(
        args=args, y_true_ids=dv.y_true_full, y_pred_ids=dev_pred_full,
        probs=dev_probs_full,
        extra=dict(extra_comum, split="dev", n_restrito=dv.n_restrito,
                   restricted_indices=dv.indices,
                   best_dev_macro_f1_history=best_epoch_f1,
                   dev_macro_f1_recomputed=dev_macro_rec,
                   dev_negation_of_f1_recomputed=dev_neg_rec))
    out.with_suffix(".dev_preds.json").write_text(
        json.dumps(dv_payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    append_test_eval_log(out, args=args, cfg=cfg, n_test=te.n_full,
                         macro_f1=m["macro_f1"],
                         negation_of_f1=m["f1_per_class"]["negation_of"])

    log.info("=== RESULTADO (test remapeado, %d pares) ===", te.n_full)
    log.info("F1 negation_of=%.4f  (dev: %.4f, melhor epoca %s de %d)",
             m["f1_per_class"]["negation_of"], dev_neg_rec, best_epoch, args.epochs)
    log.info("Avisos capturados: %d distintos", len(coletor.mensagens))
    log.info("Saidas em %s | tempo total %.1fs", out.parent, time.time() - t0)
    if hf_backup is not None:
        hf_backup.close()
    return 0


def main(argv=None) -> int:
    args = resolver_args(build_parser().parse_args(argv))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
