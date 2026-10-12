# Especificação — compatibilidade entre estágios do Two-Stage e chave de reaproveitamento (etapa 8.2)

> **Estado: ESPECIFICAÇÃO FECHADA na etapa 8.2 (12/10/2026), com as decisões
> finais do autor. Nada implementado.** Esta matriz **não** altera
> `CAMPOS_BASE` nem `composicao.conferir_par` hoje; descreve o que a
> implementação (etapa 8.4) deverá fazer.
>
> **Decisões finais:** X1 aprovada; **X2 e X3 não serão implementadas**;
> `max_length` e `batch_size` permanecem **compartilhados** enquanto a
> composição depender dos valores do estágio 1; duas listas explícitas
> (campos compartilhados e campos específicos de cada estágio); o
> reaproveitamento exige compatibilidade de configuração efetiva, dados,
> pesos, tokenizer e identidade do componente, além da integridade do modelo
> salvo.

## 1. Evidência lida no código atual

| Fato | Onde |
| --- | --- |
| `CAMPOS_BASE` (11): `encoder, seed, max_gap, ctx_chars, max_length, batch_size, lr, weight_decay, warmup_ratio, max_grad_norm, class_weight`; qualquer diferença entre estágios é erro | `two_stage/composicao.py`, `conferir_par` |
| `epochs` **não** está em `CAMPOS_BASE` e já difere (estágio 1: 3; estágio 2: 10) | `two_stage/__init__.py`, `ConfigEstagio1/2` |
| Também exigidos iguais: `registro["particoes"]` (SHA-256 de train/dev), mesma proveniência de pesos (`pretreino.mesma_proveniencia`), mesma marca de validação (`validacao.herdar`), tipo de configuração correto por estágio | `conferir_par` |
| A composição lê `Pipeline.config = e1["registro"]["config"]` e usa **essa** configuração para `max_length`, `batch_size` e `seed` na inferência **dos dois estágios** (`_prever`) | `composicao.py`, `Pipeline` |
| Os exemplos do DEV são construídos **uma vez** com `max_gap` e `ctx_chars` da configuração do estágio 1; o conjunto de referência usa `max_gap` | `composicao.py`, `_particao` |
| `lr`, `weight_decay`, `warmup_ratio`, `max_grad_norm`, `class_weight` só são lidos no treino (perda, otimizador, agendador, recorte de gradiente); a composição não os usa | `two_stage/treino.py` (linhas ~179–192), `treino/laco.py` |
| `registro_estagio(n)` grava texto fixo `pesos_classe: "balanced no TRAIN do estágio"`, independente de `class_weight` | `two_stage/treino.py` |
| O registro de cada estágio contém `modelo` como string de caminho e a retomada compara o registro inteiro | `two_stage/treino.py` |
| Tokenizer/rede: `modelos.identidade` grava `vocabulario_sha256`, `ids_marcadores`, `config_sha256`, `n_parametros` (sem `_name_or_path` nem versão do `transformers`) | `modelos.py` |
| `ler_estagio` já exige: config.json do código novo, estratégia certa, treino concluído, `melhor_modelo/` presente e **hash do `melhor_modelo/` igual ao de `treino.json`**, e identidade de código | `composicao.py` |
| `limiar.json` já congela τ com vínculos (hashes do `melhor_modelo/`, hashes de configuração, seed, proveniência, identidade do DEV) | `composicao._vinculos` |

## 2. Matriz de compatibilidade

Legenda — **Regra**: *Idêntico* (deve coincidir entre estágio 1 e 2), *Pode variar* (cada estágio tem o seu, registrado), *Condicionado* (só pode variar se uma mudança descrita for feita e aprovada), *Fixo* (não varia sob a E1). **Evid.**: **L** = lido no código (seção 1); **I** = inferência a confirmar na 8.4.

### 2.1 Campos compartilhados

| Campo | Regra | Justificativa e evidência |
| --- | --- | --- |
| versão dos dados (`v2-texto-fonte`) e SHA-256 do MANIFEST | Idêntico | a composição avalia as duas redes sobre o mesmo conjunto de referência; `conferir_par` já exige partições iguais. **L** |
| partições train/dev (SHA-256 de cada) | Idêntico | idem. **L** |
| `max_gap` | Idêntico | define o conjunto de candidatos e de referência; os exemplos são construídos uma vez com o valor do estágio 1. **L** |
| `ctx_chars` | Idêntico | idem (janela de contexto da representação do exemplo). **L** |
| `max_length` | Idêntico (decisão final) | a composição usa o `max_length` do estágio 1 para inferir **os dois** estágios (`Pipeline`, `_prever`); variar exigiria mudar a composição, o que não será feito (X3 não implementada). **L** |
| `batch_size` | Idêntico (decisão final) | idem: a inferência da composição usa o `batch_size` do estágio 1 para os dois (X2 não implementada). **L** |
| representação do exemplo (marcadores, janela) | Idêntico (fixo no código) | `entrada.py`, parte de N1; mudança invalida os dois estágios. **L/I** |
| política de pré-processamento do texto | Idêntico (embutido na versão dos dados) | texto-fonte congelado nas partições; coberto pelo SHA-256 das partições. **I** |
| encoder (`pucpr/biobertpt-all`) | Idêntico | em `CAMPOS_BASE`; a fase é restrita ao BioBERTpt. **L** |
| pesos de origem: revisão (40 hex) e manifesto `PROVENIENCIA.json` | Idêntico | `pretreino.mesma_proveniencia`; cada estágio parte dos pesos originais, nunca de checkpoint ajustado (`inicializacao.recusar_origem`). **L** |
| tokenizer (hash do vocabulário com marcadores, ids dos marcadores) | Idêntico | derivado dos pesos + `MARCADORES`; deve coincidir porque a composição tokeniza o mesmo texto para cada rede. Hoje **não** é comparado entre estágios — a 8.4 deve compará-lo explicitamente. **L/I** |
| `seed` | Idêntico | em `CAMPOS_BASE`; a fase é seed 42. **L** |
| natureza (`validacao` / `desenvolvimento` / `cientifica`) | Idêntico | `validacao.herdar`; a E1 estende o mesmo princípio à natureza `desenvolvimento`: estágio de naturezas diferentes não compõe. **L/I** |
| tipo da configuração por estágio (`ConfigEstagio1/2`) | Fixo por estágio | `conferir_par`. **L** |
| ambiente (Python, `torch`, `transformers`, `tokenizers`, `numpy`, GPU) | Pode variar, **registrado**; aviso (ver `identidade-por-componente.md`, seção 5) | só afeta a variação numérica; não é parte da chave, é parte do registro. **I** |

### 2.2 Campos específicos de cada estágio

| Campo | Regra | Justificativa e evidência |
| --- | --- | --- |
| `epochs` | Pode variar | já difere por protocolo (3 e 10); não está em `CAMPOS_BASE`. **L** |
| `lr` | Pode variar (**exceção X1**) | só usado no otimizador do estágio; não é lido na composição. **L** |
| `weight_decay` | Pode variar (X1) | idem. **L** |
| `warmup_ratio` | Pode variar (X1) | só no agendador do estágio. **L** |
| `max_grad_norm` | Pode variar (X1) | só no recorte de gradiente do laço. **L** |
| `class_weight` (`balanced`/`none`) | Pode variar (X1), com correção de registro | só na perda do estágio; **mas** `registro_estagio` grava "balanced" como texto fixo — a 8.4 deve passar a gravar o valor efetivo. **L** |
| critério de seleção do checkpoint no DEV | Fixo por estágio | `CAMPO_SELECAO`: AP de `relacao` (estágio 1), F1 de `negation_of` nos pares com relação anotada (estágio 2); mudá-lo é alterar a estratégia, não um hiperparâmetro. **L** |
| dados de treino do estágio | Fixo por estágio | `DADOS_TREINO`: todos os pares (1) × só relação anotada (2). **L** |
| limiares próprios | Não existem no estágio (argmax) | o único limiar é τ, da composição. **L** |
| τ, grade de τ, desempates | Da composição, **não** do estágio | `avaliacao/limiar.py`; protocolo. **L** |
| regra de roteamento e casas de arredondamento (4) | Fixo | `CASAS_ROTEAMENTO`; muda a identidade da composição. **L** |

### 2.3 Duas listas explícitas e a exceção X1 (decisão final)

`CAMPOS_BASE` não é removido às cegas: é **substituído** por duas listas, cada
uma com validação própria:

- **`CAMPOS_COMPARTILHADOS`** (obrigatoriamente iguais nos dois estágios):
  `encoder`, `seed`, `max_gap`, `ctx_chars`, `max_length`, `batch_size`, mais,
  agora verificados explicitamente, partições (SHA-256) e versão dos dados,
  proveniência dos pesos (revisão + manifesto), hash do tokenizer e natureza.
- **`CAMPOS_POR_ESTAGIO`** (podem diferir; gravados na identidade da
  configuração de **cada** estágio e exibidos em `compor-dev`): `epochs`,
  `lr`, `weight_decay`, `warmup_ratio`, `max_grad_norm`, `class_weight`.

**X1 (aprovada):** `lr`, `weight_decay`, `warmup_ratio`, `max_grad_norm` e
`class_weight` podem variar entre estágios **desde que** façam parte da
identidade da configuração efetiva de cada estágio (entram em `config_efetiva`
da chave, seção 3). Condição que preserva a validade: são lidos só no treino de
cada estágio (**L**), e o registro de `class_weight` deixa de ser texto fixo
(`registro_estagio`) para gravar o valor efetivo.

**X2 e X3: não implementadas.** `batch_size` e `max_length` ficam
compartilhados enquanto a composição usar os valores do estágio 1.

Não há exceção para `encoder`, `seed`, `max_gap`, `ctx_chars`, dados, pesos ou
tokenizer: alterá-los muda o objeto comparado, não a otimização.

O erro de incompatibilidade passa a dizer **qual regra** falhou (campo
compartilhado divergente, partição, proveniência, tokenizer, natureza, hash do
`melhor_modelo/`, código), e não só "divergem na configuração base".

## 3. Chave de reaproveitamento de um estágio

O reaproveitamento é de um **estágio concluído** (treino terminado) numa nova
composição, sem retreino. A chave identifica exatamente:

```
chave_estagio = sha256_canonico({
  "estagio": 1 | 2,
  "estrategia": "two_stage_e1" | "two_stage_e2",
  "config_efetiva": { todos os campos de Config do estágio + o tipo ConfigEstagio<n>,
                       SEM caminhos locais },
  "estrategia_config": registro_estagio(n),              # rótulos, dados, critério de seleção
  "dados": { "versao": "v2-texto-fonte", "manifest_sha256": ...,
             "particoes": {"train": sha256, "dev": sha256}, "max_gap_do_conjunto": ... },
  "pesos": { "encoder": "pucpr/biobertpt-all", "revisao": <40 hex>,
             "manifesto_sha256": <do PROVENIENCIA.json> },
  "tokenizer": { "vocabulario_sha256": ..., "ids_marcadores": [...] },
  "codigo": { "algoritmo": "sha256-manifesto-v2", "componente": "estagio",
              "arquivos": [{"caminho","sha256"}, ...] },   # hash integral; ver identidade-por-componente.md
  "natureza": "desenvolvimento" | "cientifica" | "validacao"
})
```

Observações:

- `config_efetiva` **não contém** o caminho local dos pesos; hoje `modelo` é o
  caminho em `str`. A 8.4 deve substituí-lo por (revisão + manifesto) no
  cálculo da chave e guardar o caminho apenas como informação de sessão.
- O nome de execução (`twostage_..._e1_...`) **não** entra na chave: o
  identificador `config_id` (12 hex da chave) e o `run_id` são derivados; os
  nomes oficiais do protocolo viram alias.
- GPU e versões de bibliotecas **não** estão na chave: ficam no `ambiente`;
  geram aviso no reuso e bloqueio na retomada (`identidade-por-componente.md`,
  seção 5).

### 3.1 O que é verificado no momento do uso (não faz parte da chave)

A existência de `checkpoints/` ou de `melhor_modelo/` **não prova**
compatibilidade. Antes de reaproveitar, todas devem ser verdadeiras:

1. `config.json` do código novo e `experimento.json` com a chave gravada na
   criação; a chave **recalculada agora** (com o código atual) é igual à gravada.
2. `treino.json` presente (`treino.json` só existe ao fim da sessão) e treino
   concluído (nenhuma sessão interrompida pendente).
3. Hash recalculado do `melhor_modelo/` igual ao de `treino.json` (já feito por
   `ler_estagio`).
4. Partições conferidas contra o MANIFEST no momento do uso (já feito) e
   revisão/manifesto dos pesos conferidos contra a proveniência gravada.
5. Natureza compatível com o uso pretendido (nenhuma execução `validacao` em
   comparação ou TEST; `desenvolvimento` nunca em TEST sem promoção).
6. Nenhuma marca de erro/invalidação posterior no registro do estágio.
7. Para o par: regras da seção 2 (compartilhados idênticos; exceções X1 registradas).

Falha em qualquer item → o estágio **não** é reaproveitado, com mensagem que
diz qual item falhou. Nenhum item é ignorável por parâmetro.

### 3.2 O reaproveitamento é uma dependência, não uma cópia

A composição grava os `chave_estagio` e os hashes dos `melhor_modelo/` dos
dois estágios. Um estágio reaproveitado nunca é alterado, movido nem
sobrescrito; uma nova tentativa de treino do mesmo estágio gera outro
`run_id` (`…__t<n+1>`) sem apagar o anterior.

### 3.3 Retomada de treino (distinta de reaproveitamento)

A chave é a mesma, mais: hash do estado salvo (`ultimo.json`), **mesma GPU
(tipo/modelo) e mesmas versões** das bibliotecas numéricas (bloqueio, não
aviso), e consistência do registro **sem** o caminho local.

## 4. Itens de implementação (8.4) e registro de escolhas

1. Comparar o tokenizer entre estágios (lacuna atual).
2. Gravar o valor efetivo de `class_weight` em `registro_estagio`.
3. Substituir o caminho local dos pesos na identidade de configuração por
   revisão + manifesto.
4. A E1 não inclui `max_length`, `ctx_chars`, `max_gap` nem `batch_size` no
   espaço de busca (ver emenda, seção 3).
