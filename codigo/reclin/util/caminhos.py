"""Caminhos padrão do repositório, calculados a partir deste arquivo.

Valem para a instalação editável (`pip install -e codigo/`), que é a forma
documentada de usar o pacote: assim o destino é o mesmo qualquer que seja o
diretório de trabalho. Todas as funções que leem ou gravam arquivos recebem o
caminho como argumento; estes valores são só os defaults usados pelos scripts.

Origem no legado: caminhos relativos ao diretório de trabalho em cada script e
as constantes de `scripts/_artifacts.py`.
"""
from __future__ import annotations

from pathlib import Path

CODIGO = Path(__file__).resolve().parents[2]      # codigo/
RAIZ = CODIGO.parent                                # RECLin-PT/

DADOS = CODIGO / "dados"
BRUTOS = DADOS / "brutos"              # XML do SemClinBr (fora do git: licença restrita)
PROCESSADOS = DADOS / "processados"    # dataset.jsonl (fora do git: contém o texto do corpus)
PARTICOES = DADOS / "particoes"        # train/dev/test.jsonl + MANIFEST.json (versionados)
DATASET = PROCESSADOS / "dataset.jsonl"

RESULTADOS = CODIGO / "resultados"
TCC = RAIZ / "tcc"


def relativo_ao_codigo(caminho: str | Path) -> str:
    """Caminho em formato POSIX relativo a `codigo/`, para gravar em registros
    sem expor o caminho absoluto da máquina. Fora de `codigo/`, devolve o
    próprio caminho em formato POSIX."""
    caminho = Path(caminho).resolve()
    try:
        return caminho.relative_to(CODIGO).as_posix()
    except ValueError:
        return caminho.as_posix()
