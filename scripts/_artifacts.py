"""
Camada compartilhada entre `make_tables.py` e `make_figures.py`.

Concentra o que as duas ferramentas precisam saber em comum:

- onde ficam os arquivos de `results/` e qual e a convencao de nomes
  (`baseline_<modelo>_seed<N>.json`, `significance_<A>_vs_<B>_seed<N>.json`);
- como um checkpoint do Hugging Face vira o rotulo usado no artigo
  ("pucpr/biobertpt-all" -> "BioBERTpt (clinico)");
- a ordem canonica das classes nos eixos e nas colunas;
- formatacao numerica em portugues (virgula decimal).

Manter isso num modulo unico garante que tabela e figura nunca divirjam
no rotulo de um modelo ou na ordem das classes.
"""

from __future__ import annotations

import json
from pathlib import Path

# Raiz do repositorio: este arquivo vive em <raiz>/scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

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
