# Observações do legado — integração Hugging Face e Google Drive (etapa 8.2)

> **Consulta somente de leitura**, autorizada pelo autor: `src/hf_backup.py` e a
> célula 11 dos notebooks 01 e 02, commit `a5f055c` do repositório
> `angeloalsf/RECLin-PT-original`. Nada foi copiado, executado ou alterado;
> nenhum código do legado foi trazido para o projeto.

> **Decisão do autor (8.2):** para a integração com Hugging Face e Drive, vale
> apenas o comportamento efetivamente observado nestes materiais; não há
> investigação adicional de outras células.

## 0. Como a leitura foi feita (limite de confiança)

O ambiente não clona repositórios (sem Git) e os arquivos foram obtidos por
leitura remota de `raw.githubusercontent.com` no commit `a5f055c`, por uma
ferramenta que **resume** o conteúdo conforme a pergunta. Portanto, os itens
"observado" abaixo são **o que a ferramenta relatou do arquivo**, não leitura
literal do texto-fonte por mim. Antes de qualquer decisão que dependa de um
detalhe (por exemplo, número de tentativas), conferir no arquivo original.

Arquivos nomeados no README do commit: `notebooks/01_baseline_biobertpt_colab.ipynb`,
`02_baseline_bertimbau_colab.ipynb`, `03_significance_colab.ipynb`. O README
diz que os notebooks 01 e 02 "montam o Google Drive, treinam com retomada
automática por época" e "opcionalmente" enviam o modelo final ao Hugging Face.

## 1. A "célula 11" dos notebooks 01 e 02

Fatos observados (ambos os notebooks, relato coincidente):

- Com índice **zero-based**, a célula 11 é uma célula **markdown**, "6. Token do
  Hugging Face Hub (Colab Secrets)". **Não tem código.** A célula de código com
  a autenticação é a de índice 12. (A referência original à "célula 11" era
  ambígua; o autor a considerou **esclarecida** na 8.2 e dispensou investigar
  outras células.)
- Instrui: criar token HF com escopo de **escrita**, criar o secret
  `HF_TOKEN` no Colab, ativar "Notebook access", rodar a célula seguinte.
- Diz que o backup é do `best_model/` para um repositório **privado**, a cada
  nova melhor época; o **Drive é o destino principal** e o Hub é camada extra.
- Sem o secret, o treino segue e só o backup extra é pulado. A célula não fala
  de erros de upload.

Célula de código seguinte (índice 12), relatada para o notebook 02 (o relato
do 01 só a inferiu):

- Lê `HF_TOKEN` pela API `userdata` do Colab e a coloca no ambiente do
  processo; o texto diz que o token **não é impresso nem passado na linha de
  comando**. A leitura é protegida por `try/except`: em falha, imprime o tipo
  da exceção e segue.
- O usuário do HF está **fixo no código** (`angeloalsf`); o nome do repositório
  é `reclin-pt-<modelo>-seed<N>` com a seed extraída do fim do nome da pasta de
  checkpoints; sem seed identificável, o backup é desativado com mensagem.
- Caminho dos checkpoints no Drive: `MyDrive/RECLin-PT/checkpoints_<modelo>_seed42`.
- Sem token ou sem seed: imprime motivo e como corrigir, esvazia a flag de
  backup e o treino segue.
- **Dúvida não resolvida:** a ferramenta disse que "o repositório" é clonado em
  `/content/`; isso provavelmente se refere ao **código** do projeto, não ao
  repositório do HF (o `hf_backup.py` não usa clone). Não confirmado.

## 2. `src/hf_backup.py`

### 2.1 Autenticação

| Item | Observado |
| --- | --- |
| Fontes do token, em ordem | argumento explícito (`--hf-token`), `$HF_TOKEN`, `$HUGGING_FACE_HUB_TOKEN`; senão `None` |
| Retorno | `(token, rótulo da origem)`; o rótulo é o nome da fonte, nunca o valor |
| Sem token | não é erro: `huggingface_hub` pode usar login em cache |
| Colab `userdata` | **ausente** neste arquivo (a leitura dos Secrets está no notebook) |
| Vazamento | o token nunca é logado; uma função de limpeza troca o token por `***` nas mensagens de aviso, porque exceções HTTP podem trazer detalhes da requisição |

### 2.2 Repositório

- `create_repo` com `repo_type="model"`, `private=True` **fixo no código**,
  `exist_ok=True`. O `repo_id` vem do chamador; o padrão de nome aparece só no
  exemplo da docstring.
- A preparação ocorre ao construir o objeto; falha vira aviso e não interrompe
  o treino; se falhou, a criação é tentada de novo no primeiro upload.

### 2.3 O que é publicado e como

- `HfApi.upload_folder` (um commit por envio) da pasta **inteira** `best_model/`
  (saída de `save_pretrained` do modelo e do tokenizer), na **raiz** do
  repositório, sem filtros de padrão.
- A origem do envio é um **instantâneo temporário** (`tempfile.mkdtemp`), fora
  da pasta de checkpoints, apagado em `finally` após cada tentativa.
- Mensagem de commit: `"best_model: " + nota`, ou um texto padrão.
- Envio em **thread daemon** (`hf-backup`), uma por vez, com **uma vaga
  pendente**: se chega um instantâneo mais novo antes do anterior começar, o
  antigo é descartado e registrado como substituído.
- Gatilho: segundo a docstring, `save_best_model` aciona o envio a cada novo
  melhor macro-F1 do DEV (o chamador não está nesse arquivo; não verificado).
- **Não** publica `ultimo.pt` nem otimizador/estado de retomada: só o melhor
  modelo.

### 2.4 Tratamento de falhas

- Padrão: 3 tentativas, espera inicial 15 s, dobrando (15 s, 30 s); **toda**
  exceção é tentada de novo, sem distinguir tipo.
- Nada sobe ao laço de treino: falhas viram avisos limpos de token; o aviso
  final diz que o treino continua e que o `best_model` local está intacto. A
  função de envio em si lança em falha; só o invólucro `HFBackup` captura.
- Falha ao criar o instantâneo: o backup daquela época é pulado com aviso.
- `close(timeout_s=900)` espera até 900 s pela thread, avisa se ainda ativa e
  registra contagens de sucesso/falha e o repositório. Por ser daemon, um envio
  não concluído é **morto** na saída do processo (dito na docstring).
- O chamador de `close()` não está no arquivo (inferido).

### 2.5 O que **não** existe no legado

- Nenhuma soma de verificação, manifesto, verificação pós-envio ou comparação
  de hash.
- Nenhuma revisão, branch ou tag por época ou por execução: os envios sucessivos
  são commits na branch padrão e **sobrescrevem os mesmos caminhos** (o
  histórico só existe pelos commits).
- Nenhum clone Git local; todo envio é por API.
- Nenhum limite de tamanho; LFS fica por conta de `huggingface_hub`.
- Nenhum rastreio de proveniência (revisão do código, dos pesos, dos dados) no
  repositório publicado.

## 3. Fatos × inferências × recomendações

| # | Fato (observado) | Inferência | Recomendação como insumo para o Prompt 2 |
| --- | --- | --- | --- |
| 1 | Token: argumento → `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN`; Colab `userdata` só no notebook; token limpo (`***`) em mensagens | o modelo "Secrets no notebook → ambiente do processo → biblioteca" funciona | manter a origem do token nos Secrets; **nunca** em argumento de linha de comando, arquivo ou saída; limpar `***` também em tracebacks; registrar só o **rótulo** da origem. Isso é compatível com a decisão 3 |
| 2 | Usuário do HF fixo no notebook | acopla o notebook ao dono | configurar o repositório por parâmetro/Secret, sem usuário fixo |
| 3 | Repositório privado fixo no código; `exist_ok=True` | privado por padrão | manter privado por padrão e **não configurável para público** sem autorização explícita |
| 4 | Pasta `best_model/` inteira, raiz do repositório, mesmos caminhos reescritos a cada envio | execuções diferentes podem colidir; histórico só por commits | um **prefixo por `run_id`** (`runs/<run_id>/…`), nunca reutilizado; sem sobrescrita de execuções anteriores (decisão 5) |
| 5 | Só `best_model/`, sem `ultimo.pt` | coerente com a decisão 5 | manter; publicar `melhor_modelo/` + manifesto; `ultimo.pt` só sob pedido explícito |
| 6 | Sem manifesto nem verificação | o envio pode ter sido parcial sem que ninguém saiba | `manifesto.json` por execução (arquivos, tamanhos, SHA-256, identidade de código, chave do componente, pesos, dados, ambiente) e verificação pós-envio (listar arquivos e comparar tamanhos/hashes) |
| 7 | Thread daemon, 1 vaga pendente, descarta o antigo | no Colab, desconexão mata envios em curso | publicar **ao fim** de cada execução e de forma explícita pelo notebook (operação "preparar publicação de artefatos"), com retomada de envio; não depender de thread daemon |
| 8 | Retries 3× com recuo, toda exceção retentada | erros permanentes (token inválido, 401/403) são retentados sem sentido | tentar de novo só erros transitórios; falha permanente aborta o envio com mensagem clara |
| 9 | Falha não interrompe o treino (aviso) | adequado para backup de apoio | manter para backup **intermediário**; a **publicação final** de uma execução oficial deve falhar de forma visível |
| 10 | Drive é o destino principal; Hub é camada extra | o legado treinava **no Drive** | decisão 4 inverte: treinar em disco local e sincronizar para o Drive por arquivo temporário + renomeação + verificação; HF permanece camada adicional |
| 11 | Nome de repositório por seed `reclin-pt-<modelo>-seed<N>` | uma seed por repositório | um único repositório privado para a fase, com prefixos por execução; o `config_id` e o `run_id` fazem parte do caminho |

Itens sem equivalente no legado e que a infraestrutura precisará definir do
zero: sincronização para o Drive com arquivos temporários e verificação de
integridade, detecção de execução interrompida na retomada, política de
versões/GPU na retomada, tarball do SHA exato do código.

## 4. Limites desta consulta

- Detalhes numéricos (tentativas, espera, 900 s) vêm de relato resumido; não
  usar como fatos sem conferência no arquivo.
- Não foi possível listar a árvore `notebooks/` pela API nem pela página do
  GitHub; os nomes dos notebooks vieram do README do commit.
- A célula de código 12 do notebook 01 não foi relatada diretamente; o seu
  conteúdo foi inferido a partir do relato do notebook 02.
