# `artigo-sbc/` — artigo SBC do RECLin-PT

Esta pasta contém o artigo (`artigo.tex`, `artigo.bib`, `artigo.bbl`), o
template da SBC (`sbc-template.sty`, `sbc.bst`), as figuras (`figs/`) e as
tabelas (`tables/`) — ambas **saída derivada** de `results/*.json` — e as
versões entregues em `entregas/`.

Os comandos abaixo são executados **a partir da raiz do repositório**, salvo
onde indicado.

## Regerar as tabelas e figuras

Dois scripts derivam os artefatos de `artigo-sbc/` diretamente de
`results/*.json`, em vez de transcrever os números à mão. **São eles que
produzem as tabelas e figuras que o artigo compila** — não há mais versão
manual paralela:

```bash
python scripts/make_tables.py      # -> artigo-sbc/tables/
python scripts/make_figures.py     # -> artigo-sbc/figs/
```

**`make_tables.py`** grava um fragmento `.tex` por tabela: `tab_resultados.tex`
(métricas dos dois baselines no teste, com o melhor valor de cada coluna em
negrito) e `tab_signif.tex` (comparação pareada — McNemar e bootstrap no F1 de
`negation_of`). Cada fragmento é um ambiente `table` completo, com `\caption` e
`\label`; `artigo.tex` os inclui por `\input{tables/tab_resultados.tex}` e
`\input{tables/tab_signif.tex}`.

**`make_figures.py`** grava os três PDFs do artigo: `f1_por_classe.pdf` (barras
agrupadas, F1 por classe nos dois baselines) e `cm_biobertpt.pdf` /
`cm_bertimbau.pdf` (matrizes de confusão normalizadas por linha — a diagonal é
o recall por classe). São exatamente os arquivos que `artigo.tex` referencia
via `\includegraphics{figs/...}`.

Os dois aceitam `--seed` (padrão 42, a semente do artigo), `--results-dir`
(padrão `results/`) e `--out-dir`. Além disso, `make_tables.py` aceita
`--check`, que imprime as tabelas no terminal sem gravar nada, e
`make_figures.py` aceita `--format {pdf,png,svg}` e `--dpi`. Nenhum dos dois
treina nada: só leem `results/` e escrevem nas pastas de saída.

> **Rodar sem argumentos sobrescreve os artefatos reais do artigo.** É o
> comportamento pretendido: os scripts são a fonte única, e `artigo-sbc/figs/`
> e `artigo-sbc/tables/` são saída derivada — não edite os `.tex` de
> `tables/` à mão, a próxima execução descarta a edição. Para só conferir sem
> tocar no artigo, gere em outro lugar com `--out-dir` (ex.:
> `--out-dir /tmp/conferencia`) e compare.

Um terceiro script, `scripts/aggregate_seeds.py`, agrega as métricas **entre**
as sementes (média e desvio do Macro-F1 e do F1 por classe, para os dois
baselines) em `results/summary_by_seed.json`. Não gera artefato LaTeX — os
números de dispersão citados no texto saem dele:

```bash
python scripts/aggregate_seeds.py --check   # imprime sem gravar
```

Os três também podem ser chamados pelos alvos `make tables`, `make figures` e
`make aggregate`, que só reexecutam o que estiver desatualizado em relação a
`results/` (ver "make" no README da raiz). `latexmk` continua sem saber disso —
regenere antes de compilar.

Trocar a semente do artigo é, portanto, um comando: `make_tables.py --seed 43 &&
make_figures.py --seed 43` regenera os cinco artefatos coerentes entre si. Os
`\label` (`tab:resultados`, `tab:signif`) e os nomes de arquivo não mudam com a
semente, então as remissões `\ref{}` do texto continuam válidas — mas o texto
corrido em volta discute os números da semente 42 e teria de ser revisto à mão.

## Compilar o artigo

**Pré-requisito:** uma distribuição TeX Live com `latexmk` (a compilação
publicada foi feita com TeX Live 2022/dev, pdfTeX 3.141592653-2.6-1.40.22 e
latexmk 4.76). O `sbc-template.sty` e o `sbc.bst` já estão no repositório, e o
`artigo.bbl` é versionado — não é preciso instalar nada da SBC à parte.

```bash
cd artigo-sbc
latexmk -pdf artigo.tex     # -> artigo-sbc/artigo.pdf
latexmk -c                  # apaga .aux/.log/.fls/... e mantém o .pdf
```

O `latexmk` resolve sozinho as passadas de `pdflatex` e `bibtex` até as
referências cruzadas estabilizarem. Rode-o **de dentro de `artigo-sbc/`**: os
caminhos em `artigo.tex` (`figs/...`, `tables/...`) são relativos a esse
diretório.

Se tiver acabado de mexer nos números, regenere os artefatos antes de compilar
(seção anterior) — o `latexmk` não sabe que eles derivam de `results/*.json` e
não os reconstrói sozinho.

