#!/usr/bin/env python3
"""Auditoria das execuções gravadas e consolidação dos resultados do DEV (Etapa 3).

Lê as pastas de `resultados/execucoes/` (na branch `resultados-colab`) e, se
informadas, as execuções derivadas só do DEV (`tcc/analise/execucoes_derivadas_dev/`).
Não treina, não roda modelo, não lê o TEST e não altera nenhum arquivo lido.

    python codigo/diagnosticos/auditar_execucoes.py \\
        --execucoes <checkout de resultados-colab>/codigo/resultados/execucoes \\
        --derivadas tcc/analise/execucoes_derivadas_dev/baseline_calibrado_biobertpt_seed42 \\
        --derivadas tcc/analise/execucoes_derivadas_dev/filtro_biobertpt_seed42 \\
        --particoes codigo/dados/particoes-v2

`--particoes` é opcional: com ele, a partição DEV (conferida contra o MANIFEST)
dá o documento de cada candidato, para o bootstrap por documento e para a
origem dos falsos positivos de `associated_with`. Só a partição DEV é aberta.

Saídas em `--saida` (padrão `codigo/resultados/auditoria_execucoes/`), todas só
com números agregados, ids de execução e ids de documento:
inventario.json, metricas_dev.json, comparacoes_dev.json, two_stage_tau.json,
associated_with_dev.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

CODIGO = Path(__file__).resolve().parents[1]
if str(CODIGO) not in sys.path:
    sys.path.insert(0, str(CODIGO))

from reclin import particoes as particoes_mod  # noqa: E402
from reclin import tarefa  # noqa: E402
from reclin.analises import auditoria_execucoes as aud  # noqa: E402
from reclin.util.io import gravar_json, ler_json, sha256_arquivo, sha256_json  # noqa: E402

log = logging.getLogger("reclin.auditoria")
PASTA_PADRAO = CODIGO / "resultados" / "auditoria_execucoes"

# Comparações descritivas no DEV: (A, B, grupo). Chaves: (estratégia, encoder).
PA, BL, TS, MLP = "pair_aware", "baseline", "two_stage", "pair_aware_mlp"
SP = "pair_aware_sem_pesos"     # Pair-Aware linear treinada sem pesos de classe (exploratório)
CD, PP = "cls_direto", "pair_aware_pooler"     # ablação do pooler: as duas células novas do fatorial 2×2
ALL, CLIN, BERT = "BioBERTpt-all", "BioBERTpt-clin", "BERTimbau"
COMPARACOES = (
    [((PA, e), (BL, e), "Pair-Aware linear × Baseline (mesmo encoder)") for e in (ALL, CLIN, BERT)]
    + [((PA, e), (TS, e), "Pair-Aware linear × Two-Stage (mesmo encoder)") for e in (ALL, CLIN, BERT)]
    + [((PA, ALL), (MLP, ALL), "Pair-Aware linear × Pair-Aware MLP"),
       ((PA, ALL), ("baseline_calibrado", ALL), "Pair-Aware linear × controles"),
       ((PA, ALL), ("filtro_pistas", ALL), "Pair-Aware linear × controles")]
    + [((TS, e), (BL, e), "Two-Stage × Baseline (mesmo encoder)") for e in (ALL, CLIN, BERT)]
    + [((TS, ALL), ("baseline_calibrado", ALL), "Two-Stage × controles"),
       (("baseline_calibrado", ALL), (BL, ALL), "Baseline calibrado × Baseline")]
    + [((SP, ALL), (PA, ALL), "Pair-Aware sem pesos × Pair-Aware ponderada"),
       ((SP, ALL), (BL, ALL), "Pair-Aware sem pesos × demais"),
       ((SP, ALL), (TS, ALL), "Pair-Aware sem pesos × demais"),
       ((SP, ALL), ("baseline_calibrado", ALL), "Pair-Aware sem pesos × demais")]
    + [((s, a), (s, b), "Encoders (mesma estratégia)")
       for s in (PA, BL, TS) for a, b in ((CLIN, ALL), (BERT, ALL), (CLIN, BERT))]
    # Ablação do pooler (Etapa 5, M3): os quatro efeitos simples do fatorial marcadores × pooler.
    + [((PA, ALL), (CD, ALL), "Ablação: efeito dos marcadores, sem pooler"),
       ((PP, ALL), (BL, ALL), "Ablação: efeito dos marcadores, com pooler"),
       ((CD, ALL), (BL, ALL), "Ablação: efeito de retirar o pooler, sem marcadores"),
       ((PA, ALL), (PP, ALL), "Ablação: efeito de retirar o pooler, com marcadores")]
    # Controles calculados também para os outros dois encoders (Etapa 5, M4), só no DEV.
    + [c for e in (CLIN, BERT) for c in (
        ((PA, e), ("baseline_calibrado", e), "Pair-Aware linear × controles"),
        ((PA, e), ("filtro_pistas", e), "Pair-Aware linear × controles"),
        ((TS, e), ("baseline_calibrado", e), "Two-Stage × controles"),
        (("baseline_calibrado", e), (BL, e), "Baseline calibrado × Baseline"))]
)


def _estrategia_derivada(pasta: Path) -> str:
    registro = ler_json(pasta / "config.json") if (pasta / "config.json").is_file() else {}
    nome = registro.get("estrategia") or (registro.get("config") or {}).get("estrategia") or ""
    if "calibrado" in pasta.name or "calibrado" in nome:
        return "baseline_calibrado"
    if "filtro" in pasta.name or "filtro" in nome:
        return "filtro_pistas"
    return nome or pasta.name


def _sistemas(inventario: dict, derivadas: list[Path]) -> dict[tuple[str, str], Path]:
    sistemas: dict[tuple[str, str], Path] = {}
    for r in inventario["execucoes"]:
        if r["tipo"] in ("execucao", "composicao_two_stage") and "predicoes_dev.json" in r["arquivos"]:
            if r.get("pasta") in {p.name for p in derivadas}:
                continue
            chave = (aud.chave_estrategia(r), r["encoder"])
            if chave in sistemas:
                raise SystemExit(f"dois sistemas com a mesma chave {chave}: {sistemas[chave].name} e {r['pasta']}")
            sistemas[chave] = Path(r["_caminho"])
    for p in derivadas:
        base = ler_json(p / "predicoes_dev.json")
        modelo = base.get("model", "")
        encoder = next((v for k, v in aud.ENCODERS.items() if k.split("/")[1] in modelo), None)
        sistemas[(_estrategia_derivada(p), encoder)] = p
    return sistemas


def _dev(pasta: Path) -> tuple[list[tarefa.Candidato], dict]:
    divergencias = particoes_mod.conferir_particoes(pasta, ["dev"])
    if divergencias:
        raise SystemExit("partição DEV não confere com o MANIFEST:\n  " + "\n  ".join(divergencias))
    docs = particoes_mod.ler_particao(pasta, "dev")
    cands = tarefa.candidatos(docs)
    prov = {"particao_sha256": sha256_arquivo(particoes_mod.arquivo_particao(pasta, "dev")),
            "manifest_sha256": sha256_arquivo(pasta / particoes_mod.ARQUIVO_MANIFESTO),
            "n_candidatos": len(cands),
            "candidatos_sha256": sha256_json([tarefa.chave_candidato(c) for c in cands]),
            "y_true_sha256": sha256_json(tarefa.y_true(cands))}
    return cands, prov


TOLERANCIA = 1e-9     # recálculo × registro: acima disso é divergência, não ruído numérico


def _coincide(novo, antigo, tol: float = TOLERANCIA) -> bool:
    """Mesmo conteúdo, com números iguais até `tol` (ruído de ponto flutuante entre ambientes)."""
    if isinstance(novo, bool) or isinstance(antigo, bool) or novo is None or antigo is None:
        return novo == antigo
    if isinstance(novo, (int, float)) and isinstance(antigo, (int, float)):
        return abs(novo - antigo) <= tol
    if isinstance(novo, dict) and isinstance(antigo, dict):
        return novo.keys() == antigo.keys() and all(_coincide(novo[k], antigo[k], tol) for k in novo)
    if isinstance(novo, (list, tuple)) and isinstance(antigo, (list, tuple)):
        return len(novo) == len(antigo) and all(_coincide(a, b, tol) for a, b in zip(novo, antigo))
    return novo == antigo


def preservar_registrados(novos: dict, antigos: dict, o_que: str) -> dict:
    """Mantém, byte a byte, os registros já gravados e acrescenta os novos.

    Todo registro antigo é recalculado e conferido: se coincide (até `TOLERANCIA`),
    fica o valor gravado; se diverge, a auditoria para, porque um resultado
    histórico não pode mudar em silêncio. Um registro antigo ausente do recálculo
    também para a auditoria."""
    saida = {}
    for chave, antigo in antigos.items():
        if chave not in novos:
            raise SystemExit(f"{o_que}: o registro {chave!r} existe no arquivo gravado e não foi recalculado")
        if not _coincide(novos[chave], antigo):
            raise SystemExit(f"{o_que}: o recálculo de {chave!r} diverge do registro gravado (tolerância {TOLERANCIA})")
        saida[chave] = antigo
    acrescentados = [k for k in novos if k not in antigos]
    saida.update({k: novos[k] for k in acrescentados})
    if antigos:
        log.info("%s: %d registros gravados conferidos e mantidos; %d acrescentados%s", o_que, len(antigos),
                 len(acrescentados), f" ({', '.join(map(str, acrescentados))})" if acrescentados else "")
    return saida


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execucoes", type=Path, required=True)
    ap.add_argument("--derivadas", type=Path, action="append", default=[])
    ap.add_argument("--particoes", type=Path, default=None, help="pasta das partições v2 (só o DEV é lido)")
    ap.add_argument("--saida", type=Path, default=PASTA_PADRAO)
    ap.add_argument("--n-boot", type=int, default=aud.N_BOOT)
    ap.add_argument("--origem", default=None,
                    help="de onde vêm as execuções, para o registro (ex.: resultados-colab@<commit>)")
    ap.add_argument("--recalcular-tudo", action="store_true",
                    help="regrava todos os registros; por padrão, os já gravados em --saida são conferidos "
                         "contra o recálculo e mantidos, e só os novos são acrescentados")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    inventario = aud.inventariar(args.execucoes, extras=args.derivadas)
    caminhos = {p.name: p for p in sorted(args.execucoes.iterdir()) if p.is_dir()} | {p.name: p for p in args.derivadas}
    for r in inventario["execucoes"]:
        r["_caminho"] = str(caminhos[r["pasta"]])
    sistemas = _sistemas(inventario, args.derivadas)
    for r in inventario["execucoes"]:
        del r["_caminho"]
    por_pasta = {p.name: chave for chave, p in sistemas.items()}
    for r in inventario["execucoes"]:
        if r["pasta"] in {p.name for p in args.derivadas}:
            r["tipo"] = "derivada_dev"
            r["estrategia"], r["encoder"] = por_pasta[r["pasta"]]

    cands, prov_dev = (None, None)
    if args.particoes is not None:
        cands, prov_dev = _dev(args.particoes)

    metricas = {}
    preds = {}
    for (estrategia, encoder), pasta in sorted(sistemas.items()):
        conjunto = next((r.get("conjunto_dev") for r in inventario["execucoes"] if r["pasta"] == pasta.name), None)
        m = aud.metricas_sistema(pasta, conjunto)
        m.update({"estrategia": estrategia, "encoder": encoder,
                  "rotulo": f"{aud.ROTULO_ESTRATEGIA.get(estrategia, estrategia)} ({encoder})",
                  "papel": aud.PAPEL.get(estrategia)})
        p = ler_json(pasta / "predicoes_dev.json")
        preds[(estrategia, encoder)] = p
        if prov_dev is not None:
            m["confere_y_true_particao"] = sha256_json(list(p["y_true"])) == prov_dev["y_true_sha256"]
        metricas[f"{estrategia}|{encoder}"] = m

    grupos = [c["doc_id"] for c in cands] if cands is not None else None
    comparacoes = []
    for a, b, grupo in COMPARACOES:
        if a not in preds or b not in preds:
            log.warning("comparação %s × %s: sistema ausente", a, b)
            continue
        pa, pb = preds[a], preds[b]
        if pa["y_true"] != pb["y_true"]:
            raise SystemExit(f"{a} e {b} não têm o mesmo y_true")
        r = aud.comparar_pareado(pa["y_true"], pa["y_pred"], pb["y_pred"], grupos=grupos, n_boot=args.n_boot)
        comparacoes.append({"grupo": grupo, "a": f"{a[0]}|{a[1]}", "b": f"{b[0]}|{b[1]}", **r})
        log.info("%s × %s: ΔF1 neg = %+.4f", a, b, r["por_candidato"]["f1_negation_of"]["diferenca"])

    taus = {}
    for (estrategia, encoder), pasta in sistemas.items():
        if (pasta / "limiar.json").is_file() and estrategia == "two_stage":
            lim = ler_json(pasta / "limiar.json")
            taus[encoder] = {"pasta": pasta.name, "tau": lim["limiar"], "grade": aud.tabela_tau(lim)}

    assoc = {}
    if cands is not None:
        pares = [(c["doc_id"], c["e1"]["id"], c["e2"]["id"]) for c in cands]
        gaps = [tarefa.entity_gap(c["e1"], c["e2"]) for c in cands]
        for chave, p in sorted(preds.items()):
            assoc[f"{chave[0]}|{chave[1]}"] = aud.origem_falsos_positivos_assoc(p["y_true"], p["y_pred"], pares, gaps)
        rot = dict(zip(pares, tarefa.y_true(cands)))
        assoc["_referencia"] = {
            "associated_with_candidatas_no_dev": sum(1 for v in rot.values() if v == tarefa.ASSOC),
            "com_o_par_inverso_tambem_associated_with": sum(
                1 for (d, a, b), v in rot.items() if v == tarefa.ASSOC and rot.get((d, b, a)) == tarefa.ASSOC),
        }

    args.saida.mkdir(parents=True, exist_ok=True)
    if not args.recalcular_tudo:
        def gravado(nome: str, campo: str):
            return ler_json(args.saida / nome)[campo] if (args.saida / nome).is_file() else None
        antigo = gravado("metricas_dev.json", "sistemas")
        if antigo is not None:
            metricas = preservar_registrados(metricas, antigo, "métricas do DEV")
        antigo = gravado("comparacoes_dev.json", "comparacoes")
        if antigo is not None:
            par = lambda c: f"{c['a']} × {c['b']}"                                   # noqa: E731
            mantidas = preservar_registrados({par(c): c for c in comparacoes}, {par(c): c for c in antigo},
                                             "comparações do DEV")
            comparacoes = list(mantidas.values())
        antigo = gravado("associated_with_dev.json", "sistemas") if assoc else None
        if antigo is not None:
            assoc = preservar_registrados(assoc, antigo, "falsos positivos de associated_with")
        antigo = gravado("two_stage_tau.json", "two_stage")
        if antigo is not None:
            taus = preservar_registrados(taus, antigo, "limiares do Two-Stage")
        antigo = gravado("inventario.json", "execucoes")
        if antigo is not None:
            por_pasta_inv = preservar_registrados({r["pasta"]: r for r in inventario["execucoes"]},
                                                  {r["pasta"]: r for r in antigo}, "inventário")
            inventario = {**inventario, "execucoes": list(por_pasta_inv.values())}
    # Só os nomes das pastas: os caminhos absolutos dependem da máquina.
    prov = {"execucoes": args.origem or args.execucoes.name,
            "derivadas": [p.name for p in args.derivadas],
            "dev": prov_dev, "n_boot": args.n_boot, "seed_bootstrap": aud.SEMENTE}
    gravar_json(args.saida / "inventario.json", {"proveniencia": prov, **inventario})
    gravar_json(args.saida / "metricas_dev.json", {"proveniencia": prov, "sistemas": metricas})
    gravar_json(args.saida / "comparacoes_dev.json", {"proveniencia": prov, "comparacoes": comparacoes})
    gravar_json(args.saida / "two_stage_tau.json", {"proveniencia": prov, "two_stage": taus})
    if assoc:
        gravar_json(args.saida / "associated_with_dev.json", {"proveniencia": prov, "sistemas": assoc})
    log.info("saídas em %s", args.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
