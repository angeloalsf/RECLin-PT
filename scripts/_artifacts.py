"""
Camada compartilhada pelos geradores de tabela e figura do projeto.

Consumidores:

- `make_tables.py` / `make_figures.py`  -> artefatos do artigo SBC
- `make_tcc_eda.py`                     -> caracterizacao do corpus (TCC)
- `make_tcc_artifacts.py`               -> resultados e significancia (TCC)
- `make_tcc_curves.py`                  -> curvas de treino por epoca (TCC)

Concentra o que essas ferramentas precisam saber em comum:

- onde ficam os arquivos de `results/` e qual e a convencao de nomes
  (`baseline_<modelo>_seed<N>.json`, `significance_<A>_vs_<B>_seed<N>.json`);
- onde fica o corpus canonico e os splits congelados em `data/`, e como
  valida-los contra `data/splits/MANIFEST.json`;
- como um checkpoint do Hugging Face vira o rotulo usado nos textos
  ("pucpr/biobertpt-all" -> "BioBERTpt (clinico)");
- a ordem canonica das classes nos eixos e nas colunas;
- formatacao numerica em portugues (virgula decimal) e os involucros
  flutuantes ABNT (`table` / `quadro`) que o TCC usa.

Manter isso num modulo unico garante que tabela e figura nunca divirjam
no rotulo de um modelo, na ordem das classes ou no valor de um
hiperparametro.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Raiz do repositorio: este arquivo vive em <raiz>/scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Raiz do TCC. Os geradores `make_tcc_*.py` escrevem por padrao dentro
# dela, nos mesmos diretorios que `main.tex` inclui.
TCC_SRC = REPO_ROOT / "tcc" / "src"
TCC_TABLES_DIR = TCC_SRC / "tabelas"
TCC_EDA_FIGS_DIR = TCC_SRC / "imagens" / "eda"
TCC_RESULT_FIGS_DIR = TCC_SRC / "imagens" / "resultados"

# Ordem canonica das classes. E a mesma usada em `confusion_matrix.labels`
# e a que aparece nos eixos das figuras publicadas: a classe-alvo primeiro,
# a majoritaria por ultimo.
CLASS_ORDER = ["negation_of", "associated_with", "no_relation"]

# Checkpoint HF -> rotulo do artigo. A chave e o campo `model` do JSON,
# entao o rotulo acompanha o que foi realmente treinado, e nao o nome do
# arquivo.
MODEL_LABELS = {
    "pucpr/biobertpt-all": "BioBERTpt (clínico)",
    "neuralmind/bert-base-portuguese-cased": "BERTimbau (geral)",
}

# Apelido curto usado nos nomes de arquivo de `results/` e das figuras.
MODEL_SLUGS = {
    "pucpr/biobertpt-all": "biobertpt",
    "neuralmind/bert-base-portuguese-cased": "bertimbau",
}

# Ordem em que os baselines aparecem nas tabelas e nas figuras: clinico
# primeiro (e a hipotese sob teste), geral depois.
BASELINE_ORDER = ["biobertpt", "bertimbau"]

# Sementes efetivamente treinadas. 42 e a de referencia (a reportada como
# resultado principal); 43 e a analise de robustez a inicializacao.
SEEDS = [42, 43]
REFERENCE_SEED = 42

# Splits congelados, na ordem em que aparecem nas tabelas do TCC.
SPLIT_ORDER = ["train", "dev", "test"]
SPLIT_LABELS = {"train": "Treino", "dev": "Validação", "test": "Teste"}

# Tipos de relacao ANOTADOS no corpus. Nao inclui `no_relation`, que nao e
# anotacao: e o rotulo atribuido aos pares candidatos sem relacao gold.
RELATION_TYPES = ["negation_of", "associated_with"]


class MissingResultError(FileNotFoundError):
    """Arquivo esperado em `results/` nao existe.

    Levantado com uma mensagem que diz qual comando reproduz o arquivo,
    para o script nunca falhar com um traceback opaco.
    """


def load_json(path: Path) -> dict:
    if not path.exists():
        raise MissingResultError(
            f"arquivo nao encontrado: {path}\n"
            f"  Rode o baseline/significancia correspondente antes "
            f"(ver 'Como rodar' no README) ou aponte --results-dir para "
            f"outro diretorio."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def baseline_path(results_dir: Path, slug: str, seed: int) -> Path:
    return results_dir / f"baseline_{slug}_seed{seed}.json"


def load_baseline(results_dir: Path, slug: str, seed: int) -> dict:
    """Carrega um `baseline_<slug>_seed<N>.json` e valida o minimo."""
    data = load_json(baseline_path(results_dir, slug, seed))

    model = data.get("model")
    if model not in MODEL_LABELS:
        raise ValueError(
            f"modelo desconhecido em {baseline_path(results_dir, slug, seed)}: "
            f"{model!r}. Acrescente-o a MODEL_LABELS em scripts/_artifacts.py."
        )

    # Guarda-corpo: o JSON traz a seed usada de verdade. Se ela nao bate com
    # o nome do arquivo, alguem renomeou um resultado a mao -- e um erro que
    # contamina silenciosamente tabela e figura.
    if int(data.get("seed", seed)) != seed:
        raise ValueError(
            f"{baseline_path(results_dir, slug, seed).name} diz seed="
            f"{data['seed']}, mas o nome do arquivo diz seed={seed}."
        )

    labels = data["confusion_matrix"]["labels"]
    if sorted(labels) != sorted(CLASS_ORDER):
        raise ValueError(
            f"classes inesperadas em {slug} seed{seed}: {labels}. "
            f"Esperado: {CLASS_ORDER}."
        )

    return data


def load_baselines(results_dir: Path, seed: int) -> dict[str, dict]:
    """Carrega os dois baselines de uma semente, na ordem canonica."""
    return {slug: load_baseline(results_dir, slug, seed) for slug in BASELINE_ORDER}


def load_significance(results_dir: Path, seed: int) -> dict:
    """Carrega o relatorio de significancia da semente pedida."""
    return load_json(
        results_dir / f"significance_biobertpt_vs_bertimbau_seed{seed}.json"
    )


def load_pipeline_config(results_dir: Path, seeds: list[int] | None = None) -> dict:
    """Config de pipeline efetivamente usada, lida dos proprios resultados.

    `max_gap`, `ctx_chars`, `max_length`, `batch_size`, `lr` e `epochs` sao
    citados no texto do TCC. Digitar esses valores a mao no LaTeX foi
    exatamente o que deixou o capitulo de metodologia descrever uma janela
    (`max_gap=200`) diferente da que os experimentos usaram. Aqui eles saem
    do campo `config` dos JSONs de resultado.

    Exige que os quatro experimentos (2 modelos x 2 sementes) compartilhem a
    mesma config: se divergirem, os resultados nao sao comparaveis e o texto
    nao poderia citar "os hiperparametros" no singular.
    """
    seeds = seeds or SEEDS
    configs = {}
    for slug in BASELINE_ORDER:
        for seed in seeds:
            data = load_baseline(results_dir, slug, seed)
            configs[f"{slug}_seed{seed}"] = data["config"]

    reference_name, reference = next(iter(configs.items()))
    for name, config in configs.items():
        if config != reference:
            divergent = sorted(
                key
                for key in set(config) | set(reference)
                if config.get(key) != reference.get(key)
            )
            raise ValueError(
                f"config divergente entre {reference_name} e {name} "
                f"nos campos {divergent}. Os experimentos nao estao na mesma "
                f"regua: o texto nao pode citar um valor unico."
            )
    return dict(reference)


def read_jsonl(path: Path):
    """Itera registros de um `.jsonl`, ignorando linhas em branco."""
    if not path.exists():
        raise MissingResultError(
            f"arquivo nao encontrado: {path}\n"
            f"  Rode `make data` (parse + splits) antes, ou aponte "
            f"--data-dir para outro diretorio."
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict]:
    """Representacao canonica completa (`data/processed/dataset.jsonl`)."""
    return list(read_jsonl(data_dir / "processed" / "dataset.jsonl"))


def load_splits(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, list[dict]]:
    """Splits congelados, conferidos contra `MANIFEST.json`.

    O MANIFEST registra o SHA-256 de cada particao. Conferi-lo aqui e o que
    liga o Apendice de hashes ao numero impresso na tabela: se alguem
    regerar os splits sem atualizar o manifesto (ou vice-versa), o gerador
    de tabelas para em vez de publicar um numero orfao.
    """
    splits_dir = data_dir / "splits"
    manifest = load_json(splits_dir / "MANIFEST.json")

    splits = {}
    for name in SPLIT_ORDER:
        path = splits_dir / f"{name}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest["splits"][name]["sha256"]
        if digest != expected:
            raise ValueError(
                f"{path.name}: SHA-256 {digest[:12]}... nao bate com o "
                f"MANIFEST ({expected[:12]}...). Os splits mudaram sem que o "
                f"manifesto fosse atualizado -- regere um ou outro antes de "
                f"gerar tabelas."
            )
        splits[name] = list(read_jsonl(path))
    return splits


def entity_gap(e1: dict, e2: dict) -> int:
    """Distancia em caracteres entre dois spans (0 se houver sobreposicao).

    Replica `src/candidates.entity_gap`. Duplicado de proposito: `scripts/`
    gera artefatos de texto e nao deve depender de `src/` estar no
    `sys.path` (o TCC e compilado em container sem o pacote instalado). A
    equivalencia das duas implementacoes e conferida por
    `make_tcc_eda.py --check-against-results`.
    """
    if e1["start"] <= e2["start"]:
        return max(0, e2["start"] - e1["end"])
    return max(0, e1["start"] - e2["end"])


# --------------------------------------------------------------------------- #
# Involucros LaTeX (convencoes ABNT do TCC)                                    #
# --------------------------------------------------------------------------- #
# ABNT: legenda ACIMA do quadro/tabela, fonte ABAIXO. Numerico -> `table`
# ("Tabela"); qualitativo/estruturado -> `quadro` ("Quadro"), ambiente
# habilitado em tcc/src/macros.tex. Essa distincao e a diferenca de
# formatacao entre os artefatos do TCC e os do artigo SBC, que usa `table`
# para tudo e legenda abaixo.

TEX_HEADER = (
    "%% GERADO AUTOMATICAMENTE por scripts/{script} -- nao edite a mao.\n"
    "%% Para atualizar: python scripts/{script}\n"
    "%% Fonte dos dados: {sources}\n"
)


def abnt_float(
    body: str,
    *,
    caption: str,
    label: str,
    source: str,
    env: str = "table",
    placement: str = "htbp",
    resize: bool = False,
) -> str:
    """Envolve um `tabular` no flutuante ABNT usado pelo TCC.

    `resize=True` aplica um \\resizebox que SO REDUZ: a largura alvo e
    \\textwidth apenas quando a tabela natural e mais larga que o texto; caso
    contrario a largura natural e preservada. Sem o \\ifdim, \\resizebox{\\textwidth}
    tambem AMPLIA tabelas estreitas, deixando a fonte maior que a do corpo
    (convencao registrada no OUTLINE.md).
    """
    content = (
        rf"\resizebox{{\ifdim\width>\textwidth\textwidth\else\width\fi}}{{!}}{{%"
        + "\n" + body + "}"
        if resize
        else body
    )
    return "\n".join(
        [
            rf"\begin{{{env}}}[{placement}]",
            r"  \centering",
            rf"  \caption{{{caption}}}",
            rf"  \label{{{label}}}",
            content,
            r"  \\[4pt]",
            rf"  {{\footnotesize Fonte: {source}}}",
            rf"\end{{{env}}}",
            "",
        ]
    )


def write_tex(path: Path, body: str, *, script: str, sources: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        TEX_HEADER.format(script=script, sources=sources) + body,
        encoding="utf-8",
    )


def label_for(data: dict) -> str:
    """Rotulo de artigo a partir do campo `model` do JSON."""
    return MODEL_LABELS[data["model"]]


def slug_for(data: dict) -> str:
    return MODEL_SLUGS[data["model"]]


def row_normalized(matrix: list[list[int]]) -> list[list[float]]:
    """Normaliza a matriz de confusao por linha (cada linha soma 1).

    Assim a diagonal e o recall por classe -- e a leitura declarada na
    legenda das Figuras 2 e 3 do artigo.
    """
    normalized = []
    for row in matrix:
        total = sum(row)
        normalized.append([value / total if total else 0.0 for value in row])
    return normalized


def reorder(labels: list[str], matrix: list[list]) -> list[list]:
    """Reordena linhas e colunas da matriz para CLASS_ORDER."""
    index = [labels.index(name) for name in CLASS_ORDER]
    return [[matrix[i][j] for j in index] for i in index]


def display_path(path: Path) -> str:
    """Caminho relativo a raiz do repo quando possivel, absoluto quando nao.

    `--out-dir` pode apontar para fora do repositorio (uma pasta temporaria,
    por exemplo), e nesse caso `relative_to` levanta ValueError.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ptbr(value: float, decimals: int = 3) -> str:
    """Formata com virgula decimal, como no corpo do artigo."""
    return f"{value:.{decimals}f}".replace(".", ",")


def ptbr_int(value: int) -> str:
    """Inteiro com ponto como separador de milhar (padrao pt-BR)."""
    return f"{value:,}".replace(",", ".")


def tt(name: str) -> str:
    """Nome de classe/tipo em monoespaco, com o `_` escapado para LaTeX."""
    return rf"\texttt{{{name.replace('_', chr(92) + '_')}}}"
