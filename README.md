# RECLin-PT — Extração de Relações em Notas Clínicas em Português

Trabalho de Conclusão de Curso (TCC) de **Angelo Antonio Lima Silveira Filho**,
Bacharelado em Sistemas de Informação — **IFES, Campus Cachoeiro de
Itapemirim**. Orientação: **Prof. Cristiano Colombo**.

O RECLin-PT é uma arquitetura de extração de relações em notas clínicas em
português, com foco na relação `negation_of`, avaliada sobre o corpus
SemClinBr.

## O que este trabalho investiga

A pergunta de partida é simples de enunciar e difícil de responder sem um
experimento controlado:

> **Pré-treinamento em domínio clínico dá vantagem real sobre um encoder de
> domínio geral na extração de relações em notas clínicas em português?**

A intuição diz que sim: um modelo que já leu texto médico deveria entender
melhor um prontuário. Mas encoders clínicos em português são treinados com
muito menos dados que os de domínio geral, e essa troca — vocabulário
especializado *versus* volume de pré-treinamento — não é obviamente favorável a
nenhum dos lados.

Para responder, o trabalho monta um **estudo controlado**: dois encoders em
português são treinados na **mesma tarefa**, com os **mesmos dados**, os
**mesmos hiperparâmetros** e a **mesma implementação**, mudando **somente o
checkpoint de pré-treino**:

- **BioBERTpt** (`pucpr/biobertpt-all`) — encoder **clínico**.
- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) — encoder de **domínio
  geral** (pré-treinado no brWaC).

A tarefa é classificação de relações entre pares de entidades em notas clínicas
do corpus **SemClinBr**, num espaço de **3 rótulos**: `negation_of`,
`associated_with` e `no_relation`. A métrica central é o **F1 da classe
`negation_of`** no conjunto de teste.

Sobre essa base, o RECLin-PT reúne o conhecimento sobre como a negação aparece
no corpus (o léxico de pistas) e compara **estratégias experimentais** para
detectar `negation_of` — baseline, filtro de pistas, regra pura, fine-tuning
restrito, Pair-Aware e, no futuro, Two-Stage — sob o mesmo protocolo. Nenhuma
delas define o projeto: a de melhor desempenho compõe o modelo entregue.

## Estado do repositório

O projeto está sendo **reconstruído** sobre uma nova arquitetura, em etapas
pequenas e verificáveis. Cada etapa é validada contra o projeto anterior (o
"legado"), que fica no repositório original como referência.

| Pasta | Conteúdo | Situação |
| --- | --- | --- |
| `codigo/` | Implementação nova: núcleo, especialização em `negation_of`, estratégias experimentais e modelo final | Em construção. Etapas 1 a 5 concluídas: tarefa, partições congeladas, configuração base, formato das execuções, avaliação (métricas, significância e agregação), o léxico de pistas de negação, as estratégias sem GPU (filtro de pistas e regra pura, com a calibração no DEV) e a infraestrutura de treino (entrada, modelos, laço com retomada exata, avaliação do TEST em comando separado), validados contra o legado. Ver [`codigo/README.md`](codigo/README.md) |
| `tcc/` | Texto do TCC em LaTeX | Ainda não incluída. Entra quando uma etapa precisar dela (tabelas, figuras e texto); até lá o texto está no repositório original |

O que cada etapa entregou está em [`codigo/docs/entregas/`](codigo/docs/entregas/).

## Arquitetura

O pacote `reclin` (em `codigo/`) tem três níveis, e um nível só importa os de
cima:

- **Núcleo** — a tarefa e a infraestrutura para resolvê-la e avaliá-la:
  `tarefa`, `corpus`, `particoes`, `entrada`, `modelos`, `treino/`,
  `execucao`, `avaliacao/`, `config`, `util/`.
- **Especialização** — o conhecimento sobre como a negação aparece no corpus:
  `negacao/` (léxico de pistas e o predicado `e_pista`).
- **Estratégias experimentais** — `estrategias/`: cada uma combina núcleo e
  especialização para prever e grava execuções comparáveis, no conjunto
  completo de candidatos. Uma estratégia nunca importa outra.

O modelo entregue é escolhido entre as estratégias pelas execuções e
referenciado num único arquivo, `modelo_final.py`.

## Pré-requisitos

**Python 3.10 ou mais novo; recomendado 3.12 ou 3.13.** Os resultados
registrados do legado foram produzidos com Python 3.13 no Google Colab (numpy
2.1.3, scipy 1.16.3, scikit-learn 1.6.1, torch 2.11.0), conforme o campo
`environment` de cada execução. O código novo foi testado em 3.10, 3.12 e 3.13.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e "codigo/[testes]" -c codigo/requirements.txt
```

O pacote depende de numpy, scipy, torch e transformers; o `-c` fixa as versões
dos experimentos (detalhes, o tamanho do torch no Linux e a exceção do Python
3.10 em [`codigo/README.md`](codigo/README.md)).

**Corpus SemClinBr.** Não é distribuído aqui: é de **acesso restrito** e precisa
ser solicitado aos autores do corpus (Oliveira et al.). Os arquivos `.xml` vão
em `codigo/dados/brutos/`, que está no `.gitignore` — dados clínicos nunca
devem ser versionados.

Você **não precisa do corpus** para reproduzir os experimentos: as partições
derivadas (`codigo/dados/particoes/*.jsonl`) são versionadas, junto de
`codigo/dados/particoes/MANIFEST.json` — o SHA-256 de cada partição, para
conferir que são os mesmos arquivos que produziram os resultados. O corpus só é
necessário para regerar o `dataset.jsonl` e as partições.

**Hardware e tempo (etapas de treino).** O treino do legado foi validado em
**GPU NVIDIA T4** (Google Colab, ~16 GB VRAM): cada época levou ≈48 min, cada
baseline (3 épocas) ≈2h25, e os quatro baselines (2 modelos × 2 sementes)
custaram ≈10 h de GPU. Em CPU o treino roda, mas é impraticável para o tamanho
do conjunto (152.686 candidatos de treino, com `max_gap=25`).

## Como usar

Confira as partições e rode os testes (detalhes em
[`codigo/README.md`](codigo/README.md)):

```bash
python codigo/scripts/preparar_dados.py conferir
python -m pytest codigo
```

As 26 comparações de significância do TCC podem ser refeitas a partir dos
resultados do legado e conferidas byte a byte:

```bash
python codigo/scripts/comparar.py protocolo \
    --resultados codigo/testes/referencia/resultados_legado \
    --saida comparacoes --conferir codigo/testes/referencia/resultados_legado
```

O léxico de pistas congelado (11 formas) e a sua cobertura em cada partição:

```bash
python codigo/scripts/analisar.py lexico
```

As estratégias sem GPU — a calibração no DEV, o filtro de pistas sobre os
baselines do legado e a regra pura —, conferidas byte a byte contra os
arquivos do legado:

```bash
REF=codigo/testes/referencia/resultados_legado
python codigo/scripts/calibrar.py --resultados $REF --saida reproducao/CALIBRACAO_filtro.json \
    --conferir $REF/CALIBRACAO_filtro.json
python codigo/scripts/filtro_pistas.py --resultados $REF --calibracao $REF/CALIBRACAO_filtro.json \
    --saida reproducao/execucoes --conferir $REF
python codigo/scripts/regra_pura.py --calibracao $REF/CALIBRACAO_filtro.json \
    --saida reproducao/execucoes --conferir $REF
```

O treino e a avaliação do TEST, em comandos separados (detalhes, checkpoints e
retomada em [`codigo/README.md`](codigo/README.md)):

```bash
python codigo/scripts/treinar.py --nome classificador_biobertpt_seed42 --encoder biobertpt --seed 42
python codigo/scripts/avaliar_test.py --execucao codigo/resultados/execucoes/classificador_biobertpt_seed42
```

As estratégias treinadas (baseline, fine-tuning restrito, Pair-Aware) e a
geração dos artefatos do TCC entram nas etapas seguintes.

## Métricas e como interpretar

- O **F1 de `negation_of`** e o **Macro-F1** são as métricas principais.
  Micro-F1, Weighted-F1 e accuracy são dominados por `no_relation` (~99% dos
  pares) e servem só de contexto — não são a manchete.
- **MCC** é um número único robusto a desbalanceamento.
- **Curvas**: `train_loss` e `dev_loss` por época (para ver overfitting) e
  macro-F1 e F1 de `negation_of` no DEV (seleção de época).
- **Significância**: o **McNemar** diz se os padrões de erro de dois sistemas
  diferem; o **bootstrap pareado** dá o intervalo de 95% e o p-valor da
  diferença no F1 de `negation_of`. Se o IC95 não cruza zero, a vantagem é
  significativa. O pareamento só é legítimo entre execuções sobre o mesmo
  conjunto de referência (19.210 pares de teste com `max_gap=25`, mesmo
  `y_true`); predições com outro `max_gap` não são comparáveis.
- **Múltiplas sementes**: com duas sementes, o desvio é **dispersão observada**,
  não intervalo de confiança.

## Decisões principais

- **Partição em nível de documento** (não de relação): evita vazamento de
  vocabulário do mesmo prontuário entre train/test. Estratificada pela presença
  de `negation_of`. Semente fixa 42, e o resultado registrado em
  `codigo/dados/particoes/MANIFEST.json`.
- **Candidatos negativos gerados, não anotados**: o SemClinBr só anota relações
  positivas; os pares `no_relation` são gerados como pares **ordenados** dentro
  de uma janela `max_gap`. A direção importa para `negation_of`.
- **Paridade por construção**: os encoders passam pelo mesmo caminho de código
  e pela mesma configuração; só o checkpoint de pré-treino muda. Marcadores de
  posição (`[E1]`/`[E2]`, sem tipo semântico) + janela de contexto,
  CrossEntropy com `class_weight=balanced`, melhor época escolhida pelo
  macro-F1 no **dev**; teste avaliado uma única vez.
- **Artefatos do TCC derivados por script**, nunca transcritos à mão: uma
  fonte única (as execuções gravadas) para tabelas, figuras e números do texto.

## Limitações conhecidas

- **Duas sementes por modelo.** Duas execuções não bastam para estimar a
  variância de inicialização com intervalo de confiança.
- **Uma única cabeça de classificação nos baselines, sem busca de
  hiperparâmetros.** É por design: uma busca por modelo quebraria a paridade
  que sustenta a comparação. O custo é que nenhum dos dois baselines está
  necessariamente no seu melhor ponto de operação.
- **`max_gap=25` limita os pares candidatos a entidades próximas.** Relações de
  longa distância estão fora do espaço de avaliação. O valor era 20 e foi
  elevado para 25 após uma análise de sensibilidade, que mostrou que 20 já
  cortava 11,6% das relações anotadas. Com 25, o teto de recall do conjunto de
  teste é de 92,0% — `negation_of` não perde nenhuma relação, e a perda
  restante é toda de `associated_with` (100 de 1.098, 9,11%).
- **Três classes.** O espaço de rótulos é um recorte do SemClinBr; ampliá-lo
  muda a dificuldade da tarefa e pediria reexecutar tudo.

## Projeto legado

A implementação anterior está no repositório original,
[github.com/angeloalsf/RECLin-PT](https://github.com/angeloalsf/RECLin-PT),
no commit `a5f055c`: baselines BioBERTpt e BERTimbau, filtro de pistas, regra
pura, fine-tuning restrito e cabeça Pair-Aware, com seus resultados, a análise
de `max_gap`, o artigo SBC descontinuado e o texto do TCC. Ela é a referência da
lógica já desenvolvida e dos comportamentos que a nova implementação precisa
reproduzir, e não recebe manutenção.

O código novo não importa nada do legado. O que dele é necessário para validar
a reconstrução está registrado em `codigo/testes/referencia/`, gerado pelo
único script que lê o legado (`gerar_referencias.py`). Os números do TCC serão
regerados com o código novo.

## Licença

O **código** e a **documentação** deste repositório são MIT — veja
[`LICENSE`](LICENSE).

> **Os dados têm licença separada.** O corpus **SemClinBr não é distribuído
> aqui**, possui licença própria e restrita, e deve ser obtido separadamente
> junto aos autores do corpus. A licença MIT deste repositório **não se estende a
> ele**. As partições versionadas em `codigo/dados/particoes/` contêm o texto
> das notas, como no repositório original.

Para citar este trabalho, use os metadados de [`CITATION.cff`](CITATION.cff).
