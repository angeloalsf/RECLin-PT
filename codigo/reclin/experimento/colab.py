"""O que depende do Google Colab: detecção, Drive e Secrets.

`google.colab` só é importado dentro das funções, e nada daqui roda ao
importar o módulo. Fora do Colab, `montar_drive` falha com mensagem clara e
`ler_secret` devolve "indisponível" (os segredos podem então vir de variável de
ambiente, ver `segredos`).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PONTO_DRIVE = "/content/drive"


class ErroColab(RuntimeError):
    """Operação exclusiva do Colab usada fora dele, ou falha ao montar o Drive."""


def em_colab() -> bool:
    if "google.colab" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def montar_drive(ponto: str | Path = PONTO_DRIVE) -> Path:
    """Monta o Google Drive (pede autorização na interface do Colab) e devolve
    `<ponto>/MyDrive`. Se já está montado, não monta de novo."""
    ponto = Path(ponto)
    meu_drive = ponto / "MyDrive"
    if meu_drive.is_dir():
        return meu_drive
    if not em_colab():
        raise ErroColab("o Drive só pode ser montado dentro do Google Colab; fora dele, "
                        "informe uma pasta local como raiz de persistência")
    from google.colab import drive  # noqa: PLC0415

    drive.mount(str(ponto))
    if not meu_drive.is_dir():
        raise ErroColab(f"o Drive não ficou disponível em {meu_drive}; autorize o acesso e rode de novo")
    return meu_drive


def ler_secret(nome: str) -> tuple[str | None, str]:
    """`(valor, motivo)`. O valor é None quando o Secret não existe, o notebook
    não tem acesso a ele ou não se está no Colab; `motivo` diz qual (sem o valor)."""
    if not em_colab():
        return None, "fora do Colab"
    from google.colab import userdata  # noqa: PLC0415

    try:
        valor = userdata.get(nome)
    except Exception as erro:  # noqa: BLE001 — as classes de erro do Colab variam entre versões
        tipo = type(erro).__name__
        if tipo == "SecretNotFoundError":
            return None, f"Secret {nome!r} não existe"
        if tipo == "NotebookAccessError":
            return None, f"o notebook não tem acesso ao Secret {nome!r} (ative 'Acesso pelo notebook')"
        return None, f"falha ao ler o Secret {nome!r}: {tipo}"
    if not valor:
        return None, f"Secret {nome!r} vazio"
    return valor, "colab-secrets"
