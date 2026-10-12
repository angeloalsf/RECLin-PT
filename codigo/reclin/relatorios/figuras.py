"""Figuras (PNG e PDF) com um estilo único, tamanho de fonte por figura.

O matplotlib é importado só dentro das funções (o pacote não depende dele, e
importar `reclin.relatorios` não o carrega). Sem o matplotlib, `disponivel()`
é falso e os scripts registram as figuras como pendentes.

Os dados de cada figura vêm de outra parte do pacote (curvas de
`relatorios.curvas`, contagens de `analises`); aqui só se desenha.

Origem no legado (só a finalidade): `make_figures.py` e as três versões de
`apply_style` (fonte 13, 11 e 12), unificadas em `estilo(tamanho_fonte)`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMATOS = ("pdf", "png")


def disponivel() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def estilo(tamanho_fonte: int = 12) -> dict[str, Any]:
    """Parâmetros do matplotlib (`rcParams`) do estilo dos relatórios."""
    return {"font.size": tamanho_fonte, "axes.titlesize": tamanho_fonte, "axes.labelsize": tamanho_fonte,
            "legend.fontsize": tamanho_fonte - 1, "xtick.labelsize": tamanho_fonte - 1,
            "ytick.labelsize": tamanho_fonte - 1, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight",
            "figure.figsize": (6.4, 4.0), "axes.formatter.use_locale": False}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _virgula(eixo) -> None:
    from matplotlib.ticker import FuncFormatter
    eixo.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def _gravar(fig, destino: Path, nome: str) -> list[Path]:
    destino.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for formato in FORMATOS:
        caminho = destino / f"{nome}.{formato}"
        # metadados fixos para que o PDF não mude de bytes a cada geração
        meta = {"CreationDate": None} if formato == "pdf" else {"Software": None}
        fig.savefig(caminho, format=formato, metadata=meta)
        caminhos.append(caminho)
    return caminhos


def curvas(historicos: Sequence[Mapping[str, Any]], destino: str | Path, *, nome: str = "curvas_dev",
           tamanho_fonte: int = 12) -> list[Path]:
    """A métrica de seleção no DEV por época, uma linha por execução, com a
    época escolhida destacada."""
    plt = _pyplot()
    with plt.rc_context(estilo(tamanho_fonte)):
        fig, ax = plt.subplots()
        for h in historicos:
            ax.plot(h["epocas"], h["selecao"], marker="o", label=f"{h['rotulo']} ({h['campo_selecao']})")
            if h.get("melhor_epoca") in h["epocas"]:
                i = h["epocas"].index(h["melhor_epoca"])
                ax.scatter([h["epocas"][i]], [h["selecao"][i]], s=120, facecolors="none", edgecolors="black")
        ax.set_xlabel("Época")
        ax.set_ylabel("Métrica de seleção (DEV)")
        _virgula(ax.yaxis)
        ax.legend()
        caminhos = _gravar(fig, Path(destino), nome)
        plt.close(fig)
    return caminhos


def barras(contagens: Mapping[str, Mapping[str, float]], destino: str | Path, *, nome: str,
           rotulo_y: str, tamanho_fonte: int = 12) -> list[Path]:
    """Barras agrupadas: `contagens[grupo][serie]`."""
    plt = _pyplot()
    grupos = list(contagens)
    series = list(dict.fromkeys(s for g in grupos for s in contagens[g]))
    largura = 0.8 / max(1, len(series))
    with plt.rc_context(estilo(tamanho_fonte)):
        fig, ax = plt.subplots()
        for j, serie in enumerate(series):
            xs = [i + j * largura for i in range(len(grupos))]
            ax.bar(xs, [contagens[g].get(serie, 0) for g in grupos], width=largura, label=serie)
        ax.set_xticks([i + largura * (len(series) - 1) / 2 for i in range(len(grupos))], grupos)
        ax.set_ylabel(rotulo_y)
        _virgula(ax.yaxis)
        if len(series) > 1:
            ax.legend()
        caminhos = _gravar(fig, Path(destino), nome)
        plt.close(fig)
    return caminhos
