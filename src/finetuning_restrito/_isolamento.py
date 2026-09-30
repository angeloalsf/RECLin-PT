"""
Isolamento de efeitos colaterais da frente experimental `finetuning_restrito`.

POR QUE ESTE MODULO EXISTE
--------------------------
Os modulos do nucleo (`relation_extraction`, `negation_lexicon`,
`significance`) criam o logger na hora do import via `utils.logger.get_logger`,
que abre `logs/pipeline.log` em modo append. Importa-los "crus" faria esta
frente escrever num arquivo que ja existe no repositorio, e a regra desta frente
e nao modificar nenhum arquivo existente.

`get_logger` e idempotente: se o logger com aquele nome JA tem handler, ele
devolve o logger sem acrescentar o handler de arquivo. Este modulo aproveita
isso e registra, ANTES de qualquer import do nucleo, um handler de console para
cada nome. O efeito e que as mensagens do nucleo continuam aparecendo (terminal
e, quando `anexar_arquivo` e chamado, o log proprio da execucao em
`results/finetuning_restrito/`), mas `logs/pipeline.log` nunca e aberto.

Tambem liga `sys.dont_write_bytecode`, para que importar `src/*.py` nao regrave
os `.pyc` que ja existem em `src/__pycache__/`.

REGRA DE USO: importar este modulo PRIMEIRO, antes de qualquer `import` do
nucleo. Todos os pontos de entrada desta pasta fazem isso.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True

AQUI = Path(__file__).resolve().parent
SRC = AQUI.parent
REPO = SRC.parent
RESULTS_DIR = REPO / "results" / "finetuning_restrito"

for _p in (str(SRC), str(AQUI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Mesmo formato de src/utils/logger.py, para o log desta frente ser lido do
# mesmo jeito que o log do pipeline.
FORMATO = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
FORMATO_DATA = "%Y-%m-%d %H:%M:%S"

# `py.warnings` recebe os UserWarning de `torch.use_deterministic_algorithms(
# True, warn_only=True)` quando `logging.captureWarnings(True)` esta ligado.
# Essa lista nunca foi capturada nas rodadas dos baselines (ver
# divergencia_pos_determinismo_20set); aqui ela vai para o log da execucao.
NOMES = ("relation_extraction", "negation_lexicon", "significance",
         "hf_backup", "finetuning_restrito", "py.warnings")

_fmt = logging.Formatter(FORMATO, datefmt=FORMATO_DATA)
for _nome in NOMES:
    _lg = logging.getLogger(_nome)
    if not _lg.handlers:
        _h = logging.StreamHandler(sys.stdout)
        _h.setFormatter(_fmt)
        _lg.addHandler(_h)
    _lg.setLevel(logging.INFO)
    _lg.propagate = False


class ColetorAvisos(logging.Handler):
    """Guarda a mensagem de cada aviso capturado (sem repeticao, em ordem)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.mensagens: list[str] = []

    def emit(self, record):
        msg = record.getMessage().strip()
        if msg not in self.mensagens:
            self.mensagens.append(msg)


def get(nome: str = "finetuning_restrito") -> logging.Logger:
    return logging.getLogger(nome)


def anexar_arquivo(caminho: str | Path) -> None:
    """Acrescenta um handler de arquivo (append, UTF-8, LF) a todos os NOMES.

    Idempotente para o mesmo caminho. O arquivo e criado pelo chamador dentro de
    `results/finetuning_restrito/` (ou num diretorio temporario, em teste).
    """
    caminho = Path(caminho).resolve()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    for nome in NOMES:
        lg = logging.getLogger(nome)
        ja = any(isinstance(h, logging.FileHandler)
                 and Path(h.baseFilename).resolve() == caminho
                 for h in lg.handlers)
        if ja:
            continue
        fh = logging.FileHandler(caminho, mode="a", encoding="utf-8")
        fh.setFormatter(_fmt)
        # O Formatter nao traduz \n; o FileHandler abre em modo texto, entao no
        # Windows sairia CRLF. Forcar LF mantem o arquivo coerente com o resto
        # do repositorio (e com o .gitattributes desta pasta).
        fh.terminator = "\n"
        if hasattr(fh, "stream") and fh.stream is not None:
            fh.stream.reconfigure(newline="\n")
        lg.addHandler(fh)


def capturar_avisos() -> ColetorAvisos:
    """Liga `logging.captureWarnings` e devolve o coletor de mensagens."""
    import warnings
    logging.captureWarnings(True)
    # "default" = mostra cada aviso uma vez por local de origem. Sem isto, o
    # filtro padrao do Python esconderia os DeprecationWarning e repetiria
    # os UserWarning do torch a cada passo.
    warnings.simplefilter("default")
    coletor = ColetorAvisos()
    logging.getLogger("py.warnings").addHandler(coletor)
    return coletor


def dentro_de(caminho: str | Path, base: str | Path) -> bool:
    try:
        Path(caminho).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def exigir_saida_isolada(caminho: str | Path) -> None:
    """Recusa qualquer saida dentro de `results/` que nao seja desta frente.

    Caminhos fora do repositorio (por exemplo um diretorio temporario de teste)
    sao aceitos. O que nao pode acontecer e esta frente gravar por cima, ou ao
    lado, dos resultados fechados do Cap. 6 e da fase 2.
    """
    if dentro_de(caminho, REPO / "results") and not dentro_de(caminho, RESULTS_DIR):
        raise SystemExit(
            f"Saida recusada: {caminho} fica em results/ mas fora de "
            f"results/finetuning_restrito/. Esta frente nao grava em nenhum "
            f"outro lugar de results/.")
    if dentro_de(caminho, REPO) and not (dentro_de(caminho, RESULTS_DIR)
                                         or dentro_de(caminho, AQUI)):
        raise SystemExit(
            f"Saida recusada: {caminho} fica dentro do repositorio mas fora das "
            f"duas pastas desta frente (src/finetuning_restrito/ e "
            f"results/finetuning_restrito/).")
