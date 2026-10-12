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
# Pasta dos XML usada na reconstrução da v2 (também fora do git: o padrão
# `SemClinBr-xml-public-v1/` está no .gitignore). `--xml` aceita outra.
XML_SEMCLINBR = CODIGO / "SemClinBr-xml-public-v1"
PROCESSADOS = DADOS / "processados"    # dataset.jsonl (fora do git: contém o texto do corpus)
DATASET = PROCESSADOS / "dataset.jsonl"                           # v1 (conversão do legado)
DATASET_V2 = PROCESSADOS / "dataset-v2-texto-fonte.jsonl"         # v2 (reclin/corpus)
INTERMEDIARIO_V2 = PROCESSADOS / "semclinbr-intermediario-v2.jsonl"

# Versões das partições (`reclin.versao_dados`). Ambas versionadas e congeladas:
# v1-conversao-legado — as partições do legado, com o desalinhamento de offsets
#                       do D9; ficam para a regressão técnica e o histórico;
# v2-texto-fonte      — reconstruídas dos XML com a regra C1 (D9).
PARTICOES_V1 = DADOS / "particoes"
PARTICOES_V2 = DADOS / "particoes-v2"
# As partições da fase científica atual: o padrão de todos os scripts que
# treinam, avaliam ou analisam. É `versao_dados.PASTAS[VERSAO_DA_FASE]`.
PARTICOES = PARTICOES_V2

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
