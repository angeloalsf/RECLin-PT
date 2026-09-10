#!/usr/bin/env python3
"""
Backup do melhor checkpoint no Hugging Face Hub (camada EXTRA, nao substituta).

POR QUE ESTE MODULO E SEPARADO
------------------------------
`src/relation_extraction.py` e o nucleo de treino/avaliacao: tudo la dentro
precisa ser deterministico e offline. Backup e outra coisa -- e rede, e falha, e
lento, e opcional. Manter isso num modulo proprio garante tres coisas que seriam
dificeis de sustentar se o codigo morasse la:

1. `huggingface_hub` so e importado DENTRO das funcoes que o usam. Sem
   `--hf-backup-repo` o modulo nem chega a ser importado por
   `relation_extraction`, entao nao ha import novo nem chamada de rede.
2. A falha de upload nunca escapa para o laco de treino. Toda excecao vira
   `log.warning` e o treino continua. Perder um backup e recuperavel; perder a
   epoca de treino por causa do backup nao e.
3. O envio roda em segundo plano, numa thread. O laco de treino nao espera a
   rede.

O QUE E ENVIADO E QUANDO
------------------------
Toda vez que uma epoca bate o melhor dev macro-F1, `save_best_model` grava
`<ckpt-dir>/best_model/` e chama `HFBackup.backup_model_async`. Assim, se o
runtime do Colab cair na epoca seguinte, o melhor checkpoint ATE AQUELE PONTO ja
esta fora do ambiente -- nao so ao final do treino.

O RETRATO NAO PASSA PELO DRIVE
------------------------------
O envio precisa de uma copia estavel dos pesos: sem ela, a proxima melhor epoca
faria `shutil.rmtree` em cima da pasta que a thread esta lendo. Essa copia e
gerada com um `save_pretrained` extra direto para o disco LOCAL e efemero do
runtime (`tempfile`), nunca para dentro de `--ckpt-dir`. Dois motivos: copiar de
volta do Drive e lento (FUSE), e cota de Drive foi exatamente o que ja custou um
checkpoint neste projeto. O retrato e apagado assim que o envio termina.

TOKEN
-----
O token nunca e logado, em nenhum nivel. Toda mensagem de erro passa por
`_scrub`, que troca o valor do token por `***` antes de chegar ao log -- porque
excecoes de HTTP as vezes carregam a URL ou o corpo da requisicao.

Uso avulso (fora do treino), por exemplo para reenviar um backup a mao:

    from hf_backup import upload_best_model, resolve_token
    token, _ = resolve_token()
    upload_best_model(Path("checkpoints/biobertpt_seed42/best_model"),
                      "angeloalsf/reclin-pt-biobertpt-seed42", token)
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.logger import get_logger  # noqa: E402

log = get_logger("hf_backup")

# Variaveis de ambiente aceitas, em ordem de precedencia. A segunda e a que o
# proprio huggingface_hub usa, entao quem ja tem o ambiente configurado nao
# precisa de nada novo.
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def resolve_token(explicit=None):
    """Descobre o token a usar. Devolve (token, origem) -- `origem` e um rotulo
    seguro para o log (o nome da flag ou da variavel), NUNCA o valor.

    Ordem: --hf-token > $HF_TOKEN > $HUGGING_FACE_HUB_TOKEN > None. `None` nao e
    erro: o `huggingface_hub` ainda pode usar um login em cache
    (`huggingface-cli login` / `notebook_login()`).
    """
    if explicit:
        return explicit, "--hf-token"
    for var in TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value, "$" + var
    return None, None


def folder_size_mb(folder):
    """Tamanho total dos arquivos de uma pasta, em MB (so para o log)."""
    try:
        return sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file()) / 1e6
    except OSError:
        return float("nan")


def upload_best_model(folder, repo_id, token, *, commit_message=None,
                      create=True, logger=None):
    """Sobe o CONTEUDO de `folder` para o repo PRIVADO `repo_id` no HF Hub.

    Esta e a versao simples e BLOQUEANTE, util para reenviar um backup a mao.
    O treino usa a classe `HFBackup` abaixo, que chama esta funcao numa thread.

    Usa `HfApi.upload_folder`, e nao `upload_file` por arquivo, porque ele fecha
    a operacao inteira num UNICO commit: ou o commit entra, ou nao entra. Nao
    existe revisao "meio enviada" visivel no repo, que e o modo de falha que
    interessa evitar aqui. Arquivos grandes sobem por LFS com upload
    multipart/retomavel, entao uma queda no meio nao recomeca do zero.

    Levanta a excecao original em caso de falha -- quem chama decide o que fazer.
    """
    lg = logger or log
    from huggingface_hub import HfApi, create_repo  # import local: ver docstring

    if create:
        create_repo(repo_id, repo_type="model", private=True, exist_ok=True,
                    token=token)
    t0 = time.time()
    HfApi(token=token).upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=commit_message or "backup automatico do best_model",
    )
    lg.info("Backup HF: %s enviado para %s (%.0f MB em %.0fs)",
            Path(folder).name, repo_id, folder_size_mb(folder), time.time() - t0)


class HFBackup:
    """Envio em segundo plano do best_model para um repo privado no HF Hub.

    Contrato: NENHUM metodo publico levanta excecao. Falhou, virou WARNING.

    Um envio de cada vez. Se uma nova melhor epoca aparecer enquanto o envio
    anterior ainda nem comecou, o retrato antigo e DESCARTADO -- subir um
    checkpoint que ja foi superado seria gastar banda para guardar a coisa
    errada.
    """

    def __init__(self, repo_id, token, *, logger=None, attempts=3,
                 backoff_s=15.0, staging_dir=None):
        self.repo_id = repo_id
        self._token = token                 # nunca vai para o log
        self.log = logger or log
        self.attempts = attempts
        self.backoff_s = backoff_s
        self.staging_dir = staging_dir      # None = disco local do runtime
        self._lock = threading.Lock()
        self._pending = None                # (raiz_do_retrato, nota) ou None
        self._worker = None
        self._repo_ready = False
        self.uploads_ok = 0
        self.uploads_failed = 0
        self._prepare_repo()

    # ------------------------------------------------------------------ #
    # Interno                                                             #
    # ------------------------------------------------------------------ #
    def _scrub(self, text):
        """Remove o token de qualquer texto antes de ele chegar ao log."""
        s = str(text)
        if self._token:
            s = s.replace(self._token, "***")
        return s

    def _prepare_repo(self):
        """Cria o repo privado ja na largada, para que um token invalido
        apareca AGORA e nao depois de uma hora de treino."""
        try:
            from huggingface_hub import create_repo
            create_repo(self.repo_id, repo_type="model", private=True,
                        exist_ok=True, token=self._token)
            self._repo_ready = True
            self.log.info("Backup HF: repo privado pronto -> %s", self.repo_id)
        except Exception as e:  # noqa: BLE001
            self.log.warning("Backup HF: nao foi possivel preparar o repo %s agora "
                             "(%s: %s). O treino segue normalmente; a criacao sera "
                             "tentada de novo no primeiro envio.",
                             self.repo_id, type(e).__name__, self._scrub(e))

    def _snapshot(self, model, tokenizer):
        """Grava um retrato dos pesos no disco local e devolve a raiz temporaria."""
        root = Path(tempfile.mkdtemp(prefix="hf_backup_", dir=self.staging_dir))
        folder = root / "best_model"
        folder.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(folder)
        tokenizer.save_pretrained(folder)
        return root

    def _drain(self):
        """Laco da thread: esvazia a fila (que tem no maximo um item)."""
        while True:
            with self._lock:
                item, self._pending = self._pending, None
            if item is None:
                return
            root, note = item
            try:
                self._upload(root / "best_model", note)
            finally:
                shutil.rmtree(root, ignore_errors=True)

    def _upload(self, folder, note):
        """Envia com retentativa e backoff. Nunca levanta."""
        message = "best_model: " + (note or "backup automatico")
        backoff = self.backoff_s
        for k in range(1, self.attempts + 1):
            try:
                upload_best_model(folder, self.repo_id, self._token,
                                  commit_message=message,
                                  create=not self._repo_ready, logger=self.log)
                self._repo_ready = True
                self.uploads_ok += 1
                self.log.info("Backup HF: %s | tentativa %d/%d", note or "-", k,
                              self.attempts)
                return
            except Exception as e:  # noqa: BLE001
                if k < self.attempts:
                    self.log.warning("Backup HF: falha no envio (tentativa %d/%d) "
                                     "%s: %s -- nova tentativa em %.0fs. "
                                     "O TREINO SEGUE.", k, self.attempts,
                                     type(e).__name__, self._scrub(e), backoff)
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    self.uploads_failed += 1
                    self.log.warning("Backup HF: envio ABANDONADO apos %d tentativas "
                                     "(%s: %s). O best_model local segue intacto e o "
                                     "treino continua normalmente.", self.attempts,
                                     type(e).__name__, self._scrub(e))

    # ------------------------------------------------------------------ #
    # Publico                                                             #
    # ------------------------------------------------------------------ #
    def backup_model_async(self, model, tokenizer, note=None):
        """Enfileira um backup dos pesos ATUAIS. Retorna na hora; nunca levanta."""
        try:
            root = self._snapshot(model, tokenizer)
        except Exception as e:  # noqa: BLE001
            self.log.warning("Backup HF: nao foi possivel preparar o retrato local "
                             "dos pesos (%s: %s) -- backup desta epoca pulado, "
                             "treino segue.", type(e).__name__, self._scrub(e))
            return
        try:
            with self._lock:
                if self._pending is not None:
                    old_root, _ = self._pending
                    self.log.info("Backup HF: o envio anterior ainda nao comecou; "
                                  "descartando o retrato superado.")
                    shutil.rmtree(old_root, ignore_errors=True)
                self._pending = (root, note)
                # O log do enfileiramento sai AQUI dentro, antes de a thread
                # comecar: caso contrario o worker consegue logar a primeira
                # tentativa antes desta linha e a trilha de log fica fora de
                # ordem -- confuso justamente durante um treino de horas.
                self.log.info("Backup HF: envio de %.0f MB enfileirado (%s)",
                              folder_size_mb(root / "best_model"), note or "-")
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._drain,
                                                    name="hf-backup", daemon=True)
                    self._worker.start()
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(root, ignore_errors=True)
            self.log.warning("Backup HF: nao foi possivel enfileirar o envio "
                             "(%s: %s) -- treino segue.",
                             type(e).__name__, self._scrub(e))

    def close(self, timeout_s=900.0):
        """Espera o envio pendente terminar, com teto. Nunca levanta.

        A thread e daemon de proposito: se o envio travar, o processo ainda
        encerra. O preco e que um envio ainda em curso morre junto -- por isso o
        aviso explicito abaixo, e por isso o `best_model` local continua sendo a
        copia de referencia.
        """
        try:
            worker = self._worker
            if worker is not None and worker.is_alive():
                self.log.info("Backup HF: aguardando o envio em andamento (ate %.0fs)...",
                              timeout_s)
                worker.join(timeout_s)
                if worker.is_alive():
                    self.log.warning("Backup HF: o envio nao terminou em %.0fs. Como a "
                                     "thread e daemon, o processo encerra assim mesmo e "
                                     "este ultimo backup pode ficar incompleto. O "
                                     "best_model local esta intacto.", timeout_s)
            self.log.info("Backup HF: %d envio(s) concluido(s), %d falha(s) | repo: %s",
                          self.uploads_ok, self.uploads_failed, self.repo_id)
        except Exception as e:  # noqa: BLE001
            self.log.warning("Backup HF: erro ao encerrar o backup (%s: %s) -- ignorado.",
                             type(e).__name__, self._scrub(e))
