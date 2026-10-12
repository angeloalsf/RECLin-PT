# Infraestrutura compartilhada para Google Colab (Etapa 8.2, prompt 2)

Estado: implementada; validação mínima executada (importação, `--help` e testes
focados de infraestrutura, ver `docs/entregas/etapa-8.2-infraestrutura.md`; a
suíte completa **não** foi executada). Cobre o que todos os notebooks vão compartilhar; não
contém notebook de treino.

## 1. Visão geral

| Necessidade | Módulo (`reclin/experimento/`) | Observação |
|---|---|---|
| Obter o código: clone da branch (`clonar_codigo`) ou tarball de um SHA (`preparar_codigo`) | `codigo_colab.py` | só biblioteca padrão; baixado sozinho pela 2ª célula do notebook do baseline |
| Achar a raiz do código, chamar scripts | `raiz.py` | `localizar_codigo`, `ativar`, `rodar_script` |
| Drive / Secrets do Colab | `colab.py` | imports de `google.colab` só quando chamados |
| Credenciais | `segredos.py` | GitHub e Hugging Face separados; valor nunca impresso |
| Ambiente e GPU | `ambiente.py` | plano de instalação (dry-run por padrão), versões, GPU |
| Identificadores | `ids.py` | `config_id` e `run_id` |
| Chave, nomes oficiais, plano | `plano.py` | `chave_da_execucao`, `NOMES_OFICIAIS`, `planejar`, `preparar` |
| Identidade da infraestrutura | `identidade.py` | só proveniência; fora da identidade v1 |
| Operações dos notebooks de estratégia | `operacoes.py` | `executar(operacao, contexto)`; usado por `reclin_baseline.ipynb` (etapa 8.3) |
| Proveniência por execução | `proveniencia.py` | arquivo `experimento.json` na pasta da execução |
| Disco local ⇄ Drive | `persistencia.py` | cópia atômica com verificação SHA-256 |
| Hugging Face (privado) | `publicacao_hf.py` | plano por padrão; envia só com confirmação |
| Linha de comando | `scripts/experimento.py` | `ambiente`, `listar`, `planejar`, `estado`, `retomada`, `sincronizar`, `restaurar`, `publicar`, `recuperar` |

Nenhum módulo executa operação científica ao ser importado ou ao abrir um
notebook. Treino, calibração, download de pesos/corpus e TEST continuam
exclusivamente nos scripts existentes, chamados por decisão explícita.

## 2. Obter o código nos notebooks

O notebook ainda não tem o código; por isso a célula baixa **apenas** `codigo_colab.py` (API de conteúdo do GitHub,
`ref=<branch>`) e usa esse módulo para trazer o resto. Há dois caminhos, ambos gravando `REVISAO.json`:

* **`clonar_codigo(repositorio, branch, destino, token=None)`** — usado por `reclin_baseline.ipynb` (etapa 8.3.2):
  clone raso da **branch** `BRANCH_CODIGO`; o commit clonado (HEAD) é registrado em `REVISAO.json`
  (`origem: "git"`) e, por `proveniencia.codigo_do_ambiente`, em `experimento.json` (`codigo.revisao`) de cada
  execução. Um clone existente é reaproveitado **sem atualizar** (o código não muda no meio da sessão), desde que
  seja do mesmo repositório e da mesma branch e não tenha arquivos rastreados alterados; tag é recusada. O token
  (opcional, só para repositório privado) vai por `GIT_ASKPASS` (script 700, apagado ao sair) e existe só no ambiente
  do subprocesso. Como a branch se move, a reprodutibilidade vem da **identidade do código** (`vincular_config`
  grava a identidade global em `config.json`/`experimento.json`): a retomada a confere e recusa execução incompatível.
* **`preparar_codigo(repositorio, sha, token=None, arquivo_bootstrap=None)`** — tarball de um SHA exato (40 hex;
  branch ou tag recusados): confere o prefixo do SHA servido, extrai em pasta temporária (recusa `..`, caminhos
  absolutos, links), renomeia só no fim, reaproveita extração anterior do mesmo SHA e exige que o `codigo_colab.py`
  baixado seja idêntico ao da árvore extraída. `REVISAO.json` traz repositório, SHA, SHA-256 do tarball e arquivos.
  Continua disponível para notebooks que queiram código fixo por commit.

## 3. Secrets

Configure em **Colab → Secrets (ícone de chave)** e ative "Acesso pelo
notebook" apenas nos notebooks que precisam. Nunca envie o valor no chat.

| Secret | Quando | Escopo mínimo |
|---|---|---|
| `GITHUB_TOKEN` | só se o repositório for privado (clonar o código) | token fine-grained, **somente este repositório**, *Contents: Read-only* |
| `GITHUB_TOKEN_ESCRITA` | só para salvar resultados no GitHub (célula 8b do notebook do baseline) | token fine-grained, **somente este repositório**, *Contents: Read and write*; nunca o mesmo `GITHUB_TOKEN` de leitura |
| `HF_TOKEN` | só para `publicar` (`--executar`) | token fine-grained, **somente o repositório de artefatos**, com escrita |

- Os três são independentes; cada operação pede só o que precisa
  (`segredos.OPERACOES`). Treinar, retomar, sincronizar com o Drive e planejar
  publicação não exigem nenhum.
- O valor é guardado numa `Credencial` que não se imprime; mensagens, logs,
  `experimento.json` e `SYNC.json` passam por `segredos.limpar`, e os registros
  recusam textos com padrão de token (`ghp_`, `github_pat_`, `hf_`…).
- Nada é publicado automaticamente: `publicar` sem `--executar` só mostra o
  plano.
- `segredos.obter` lê o Secret do Colab (`google.colab.userdata`); se não puder
  (por exemplo, num subprocesso, onde o `userdata` não funciona), vale a variável
  de ambiente de mesmo nome. O notebook do baseline exporta o `HF_TOKEN` por
  `apresentacao.segredo_no_ambiente` **só durante** a publicação (célula 8c) e o
  remove ao terminar; o `GITHUB_TOKEN_ESCRITA` não passa por `os.environ`: vai ao
  Git por `GIT_ASKPASS` (ver `reclin/experimento/versionamento.py`).

## 4. Como os notebooks usam a infraestrutura

1. **Preparar**: obter o código (seção 2); depois `scripts/experimento.py ambiente`
   (plano de instalação em modo `minimo`, relatório de GPU/versões; instala só
   com `--instalar`; T4 é a referência, L4/A100 geram aviso e ficam registrados).
2. **Antes de treinar** (lógica única, a mesma para todas as estratégias):
   `plano.chave_da_execucao(estrategia=…, config=<Config>, estrategia_config=…,
   pasta_pesos=…, pasta_particoes=…, codigo=CODIGO)` monta a chave com a
   configuração efetiva, a revisão dos pesos (`PROVENIENCIA.json`), a versão e
   o SHA-256 das partições (MANIFEST) e a identidade do código — sem caminhos
   absolutos. `plano.planejar(chave, raiz_local, raiz_drive, alias_oficial=…)`
   diz, sem escrever nada, se é `retomar`, `iniciar_planejada`, `ja_concluida`
   ou `nova_execucao` (próxima tentativa); `plano.preparar` cria a pasta e o
   `experimento.json` da execução nova. `retomada <nome>` (dry-run) confirma
   que a retomada é segura.
3. **Treinar**: `raiz.rodar_script(CODIGO, "<script>.py", ...)` chama os
   scripts reais, em disco local do Colab (rápido). Credenciais não vão por
   argumento.
4. **Persistir**: `sincronizar <nome>` periodicamente e ao final;
   `proveniencia.mudar_estado` / `registrar_metricas`.
5. **Recuperar** numa sessão nova: `restaurar <nome>` → `retomada <nome>` →
   continuar.
6. **Publicar** (opcional, com confirmação): `publicar <nome> --repositorio
   usuario/repo` mostra o plano; `--executar` envia.

## 5. Armazenamento e recuperação de checkpoints

Camadas, da mais rápida à mais durável:

| Camada | Conteúdo | Observação |
|---|---|---|
| Disco local do Colab (`resultados/execucoes/<nome>/`) | execução completa | perde-se ao fim da sessão |
| Drive (`$RECLIN_DRIVE`, padrão `/content/drive/MyDrive/RECLin-PT`) | cópia da execução + `SYNC.json` | inclui `ultimo.pt` (necessário para retomar) |
| Hugging Face (repositório privado, prefixo `runs/<nome>/`) | pesos finais, métricas, configs, `MANIFESTO_PUBLICACAO.json` | **sem** `ultimo.pt` e **sem** artefatos do TEST por padrão |

Sincronização com o Drive: cópia para `*.sync-tmp`, conferência SHA-256,
`os.replace`; `SYNC.json` é escrito por último (sua presença marca cópia
íntegra); nunca regride um estado mais avançado; nunca apaga. Divergência
entre local e Drive é informada e só é resolvida com `--substituir`.

Retomada: `diagnosticar_retomada` lê `ultimo.json` (`posicao.passo_global`),
detecta interrupção e compara o ambiente da última sessão com o atual.
Bloqueia retomada se mudou versão menor do Python, torch, transformers,
tokenizers, numpy ou GPU; reaproveitar etapa concluída só avisa.

Publicação no Hugging Face: repositório precisa ser privado (se não existir, é
criado privado; se existir e for público, a publicação é recusada); um único
commit; recusa sobrescrever execução já publicada; verifica o remoto depois do
envio. Pesos grandes nunca vão ao GitHub.

### Identificação e nomes oficiais

- `config_id` (12 hex) = SHA-256 da chave canônica: estratégia, configuração
  efetiva, configuração própria da estratégia, pesos (repositório, commit,
  hash do manifesto), dados (versão e SHA-256 de `train` e `dev`; o TEST não
  entra) e identidade do código. O caminho do modelo é removido e qualquer
  caminho absoluto é recusado: a mesma configuração dá o mesmo `config_id` em
  qualquer sessão.
- `run_id` = `<estrategia>__<config_id>__t<n>`. Retomar não cria tentativa;
  recomeçar do zero cria a seguinte. Execuções de configurações diferentes
  nunca compartilham `config_id`.
- Nomes oficiais: só os quatro do protocolo (`baseline_biobertpt_seed42`,
  `baseline_calibrado_biobertpt_seed42`, `restrito_biobertpt_seed42`,
  `pairaware_biobertpt_seed42`), conferidos por teste contra o texto do
  protocolo. Uma execução recebe o nome oficial como pasta e `alias_oficial`
  só se a estratégia, o encoder e a semente baterem e a pasta estiver livre;
  uma pasta oficial ocupada por outra configuração é erro, e uma nova
  tentativa da mesma configuração usa o `run_id` como pasta. Execuções
  exploratórias usam o `run_id`.
- Retomada: `planejar` e `compativeis` olham o disco local e o espelho do
  Drive e comparam pela chave gravada em `experimento.json` (o `config_id` é
  recalculado; nome de pasta e caminho de sessão não contam).

## 6. `experimento.json`

Esquema `reclin-experimento/1`, na pasta da execução (fora do `config.json`,
`treino.json` e `metricas.json`, que não são alterados). Campos: `run_id`,
`config_id`, `estrategia`, `chave` (config canônica, seed, parâmetros de
treino, pesos com revisão, dados), `codigo` (repositório, SHA, hash de
identidade), `ambiente`, `estado` (`planejada → iniciada → interrompida |
concluida | falhou`), `historico` de transições, `metricas` (resumo e
referência ao arquivo), `drive` (local e SHA de `SYNC.json`), `publicacoes`
(repositório, revisão, manifesto). Sem caminhos absolutos na chave e sem
credenciais.

## 7. Limitações conhecidas

- A infraestrutura está **fora** da identidade de código v1 e tem identidade
  própria, só informativa (`identidade-por-componente.md`, seção 11); editá-la
  não invalida treinos. A identidade `v2` por componente continua na 8.3.
- O `experimento.json` local difere do publicado no Hub (a publicação é
  registrada depois do envio).
- Como `tomllib` não existe no Python 3.10, os requisitos mínimos de
  `pyproject.toml` são lidos por expressão regular.
- Existe um `reclin/experimento/__pycache__/codigo_colab.cpython-310.pyc`
  gerado por engano (inofensivo; fora das duas identidades). Pode ser apagado
  manualmente.
- O nome do repositório do Hugging Face é configuração explícita
  (`--repositorio usuario/repo`), exigida só para publicar.
