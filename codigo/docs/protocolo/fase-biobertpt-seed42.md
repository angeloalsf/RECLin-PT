# Protocolo pré-registrado — fase BioBERTpt, seed 42 (validação inicial)

Registrado na etapa 6.1 (11/10/2026), **antes de qualquer treino científico**.
Fixa o escopo, a revisão dos pesos, os critérios de seleção e de parada de
cada estratégia, os limiares escolhidos no DEV, a sequência de execução, as
comparações e a forma de interpretar os resultados da fase científica (etapa
8). Tudo o que aqui se escolhe usa **só o DEV**. Qualquer mudança neste
documento antes da etapa 8 depende de aprovação explícita do autor.

O bloco JSON no fim deste documento é conferido contra o código por
`testes/unidade/test_protocolo_fase.py`: se o código mudar um critério sem que
este registro mude, a suíte falha.

## 1. Escopo da fase

| | Decisão |
| --- | --- |
| Rótulo dos resultados | **"validação inicial, seed 42"** |
| Encoder | só o BioBERTpt (`pucpr/biobertpt-all`). O BERTimbau fica fora do escopo experimental atual |
| Semente | só a 42. A semente continua parâmetro do código; rodar três sementes (42, 43, 44) é uma possível fase posterior, fora do escopo atual |
| Sistemas comparados | baseline padrão, baseline calibrado, filtro léxico sobre o baseline novo, restrito, Pair-Aware e Two-Stage |
| Origem dos resultados | **só o código novo**, com as configurações registradas. Nenhum resultado do legado entra nos experimentos ou nas comparações; os históricos ficam preservados como documentação e como referência da reprodução técnica da etapa 7. Execução ausente roda na fase científica, com autorização específica; nunca é substituída por um resultado do legado |
| Reaproveitamento | só de execuções **do código novo** concluídas e compatíveis (mesmo `config.json`, mesmas partições, seed 42) |
| Léxico | o congelado: `min_freq=3`, 11 formas, `lexico_sha1` `70c93fa807de`, induzido só do TRAIN. Não é recalibrado nesta fase. O `min_freq=3` vem da calibração histórica (etapa 3); é um parâmetro fixo, não um resultado usado na comparação |
| Modelo final | a regra de escolha fica fora do escopo e será definida numa etapa posterior. O TEST não influencia nenhuma escolha |

## 2. Revisão exata dos pesos (proveniência)

1. `scripts/pretreino.py baixar --revisao <SHA de 40 caracteres>` baixa o
   snapshot do commit (branch e tag são recusados) para
   `codigo/modelos_pretreinados/biobertpt-all@<SHA>/` (fora do git) e grava
   `PROVENIENCIA.json` com o commit, o SHA-256 e o tamanho de cada arquivo e o
   hash do manifesto. É o único passo com acesso ao Hugging Face Hub; o commit
   é comunicado ao autor antes do primeiro treino e fica congelado para a fase.
2. Todas as execuções treinadas da fase (baseline, restrito, Pair-Aware e os
   dois estágios) usam `--modelo <essa pasta>`. Antes de cada carga, os hashes
   de todos os arquivos são recalculados (divergência = recusa) e a carga é
   feita sem rede (`HF_HUB_OFFLINE=1`).
3. O commit e o hash do manifesto ficam no `config.json` de cada execução
   (`proveniencia_pesos`); retomar com outra cópia é recusado.
4. Antes da comparação final, `scripts/pretreino.py conferir --execucoes ...`
   exige, de todas as execuções: código novo, nenhuma marca de validação
   técnica, a configuração do protocolo (seção 3), a mesma semente, o mesmo
   commit e o mesmo hash de manifesto.

Limitação aceita: a ligação entre os arquivos locais e o commit depende do
`snapshot_download(revision=<SHA>)` no momento do download.

## 3. Critérios de seleção e de parada (só DEV)

| Estratégia | Escolha do checkpoint | Parada do treino | O que fica congelado antes do TEST |
| --- | --- | --- | --- |
| Baseline padrão | maior macro-F1 do DEV; empate → a primeira época | teto fixo de 3 épocas, sem parada antecipada | época |
| Baseline calibrado | sem treino: o limiar t sobre as probabilidades do DEV do baseline padrão (seção 4) | — | t |
| Filtro léxico | sem checkpoint: pós-processa o baseline padrão | — | nada (léxico congelado) |
| Restrito | maior macro-F1 do DEV **remapeado** ao conjunto completo (fora do espaço → `no_relation`); empate → a primeira época | teto fixo de 10 épocas, sem parada antecipada | época |
| Pair-Aware | igual ao restrito | igual ao restrito | época |
| Two-Stage, estágio 1 | maior AP da classe `relacao` no DEV completo; empate → a primeira | teto fixo de 3 épocas | época |
| Two-Stage, estágio 2 | maior F1 de `negation_of` nos pares do DEV com relação anotada (argmax); empate → a primeira | teto fixo de 10 épocas | época |
| Two-Stage, composição | τ (seção 4) | — | τ |

- A seleção do restrito e da Pair-Aware é a implementada e validada na etapa
  6 (a mesma métrica do baseline). Ela difere da métrica primária (F1 de
  `negation_of`); a diferença fica registrada.
- No Two-Stage, cada estágio é escolhido pela sua própria métrica. **Isso não
  garante a melhor combinação para o pipeline completo.** Não há otimização
  conjunta de épocas no DEV; o único parâmetro do pipeline escolhido no DEV é
  τ.
- Justificativa dos tetos: estágio 1 = os mesmos dados e passos do baseline
  (o mesmo teto evita que a diferença venha de mais treino; a documentação
  histórica indica que os baselines ainda melhoravam na 3.ª época, limitação
  comum aos dois); estágio 2 = dimensionado pelo volume de atualizações (1.310
  passos). Se a melhor época do estágio 2 for a 10.ª, isso é registrado e
  qualquer aumento depende de aprovação.
- **Sem "portão de pesquisa"**: os critérios de parada de frente do legado
  (várias sementes) não se aplicam. O TEST de todos os sistemas é avaliado uma
  única vez, depois que todas as escolhas do DEV estiverem congeladas, sem
  condição.

## 4. Limiares escolhidos no DEV (τ e t)

Uma só busca (`reclin/avaliacao/limiar.py`) para os dois:

- grade: 0,05; 0,10; …; 0,95 (19 valores);
- critério: maior F1 de `negation_of` da saída final (3 rótulos) no DEV, pelas
  contagens;
- desempates, nesta ordem: maior macro-F1; menor |limiar − 0,5|; maior limiar;
- comparação exata, por frações das contagens.

**τ (Two-Stage):** um par vai ao estágio 2 quando P(relacao), arredondada a 4
casas (a convenção dos sidecars), é ≥ τ; os demais ficam `no_relation`; o
estágio 2 decide pelo argmax. Para cada τ, o pipeline completo roda pelo
**fluxo real** (o estágio 2 só sobre os pares roteados), o mesmo código do
TEST.

**t (baseline calibrado):** P(`negation_of`) ≥ t → `negation_of`; senão o
argmax entre `associated_with` e `no_relation` (empate exato →
`associated_with`, a convenção do argmax). Entradas: as probabilidades de 4
casas que o baseline já grava no DEV e no TEST (sem GPU extra).

**Congelamento:** `limiar.json` guarda o limiar, a tabela completa, a
configuração da busca (com hash) e os vínculos — hashes dos `melhor_modelo/`
dos estágios ou do `predicoes_dev.json` da base, hash das configurações, seed,
proveniência e identidade do DEV. A aplicação ao TEST recalcula tudo e recusa
qualquer divergência. Não existe opção para informar um limiar à mão.

## 5. Rodada de validação técnica (passo 1V)

Antes da fase científica, com o BioBERTpt real, **só até o DEV**: baseline (1
época), estágio 1 (1 época), estágio 2 (2 épocas, para exercitar a seleção e o
desempate "primeira época"), `compor-dev` e a calibração de t.

- As épocas vêm de `--validacao-epocas N`, gravado no `config.json` como
  execução de validação; as constantes do protocolo não mudam.
- Nomes: `baseline_validacao_1ep`, `twostage_e1_validacao_1ep`,
  `twostage_e2_validacao_2ep`, `twostage_validacao`,
  `baseline_calibrado_validacao_1ep`.
- Recomendado: rodar com uma pasta de partições só com `train.jsonl`,
  `dev.jsonl` e o `MANIFEST.json` congelado (o código confere o tamanho dos
  conjuntos quando o MANIFEST é o congelado, byte a byte).
- Execuções de validação **nunca** entram na etapa 8: `avaliar_test.py`,
  `two_stage.py avaliar-test`, `baseline_calibrado.py aplicar-test`,
  `pretreino.py conferir --execucoes` e `comparar.py` as recusam; τ e t da
  validação não valem para execuções científicas (vínculos).
- Resultados rotulados **"validação técnica, N época(s)"**: verificam que o
  pipeline funciona (formatos, sidecars, vínculos, recusas), não avaliam
  desempenho.
- Opcional, aprovado para quando o passo 1V for autorizado: interromper e
  retomar o estágio 2 para verificar a retomada em GPU.

## 6. Sequência da fase científica

Cada passo em GPU depende de autorização específica. Estimativas em T4, não
medidas.

| Passo | Execução | Estimativa |
| --- | --- | --- |
| 0 | `pretreino.py baixar` num SHA exato; commit comunicado ao autor | minutos |
| 1 | `two_stage.py verificar-inicializacao` com os pesos reais | minutos |
| 1V | rodada de validação técnica (seção 5); revisão do relatório antes do passo 2 | ~1,7 h |
| 2 | `baseline_biobertpt_seed42` | ~2,4 h |
| 3 | calibração de t (`baseline_calibrado_biobertpt_seed42`) e filtro no DEV | minutos, sem GPU |
| 4 | `restrito_biobertpt_seed42` e `pairaware_biobertpt_seed42` | ~0,5 h cada |
| 5 | estágios 1 e 2; `compor-dev` (τ congelado) | ~2,9 h |
| 6 | `pretreino.py conferir --execucoes` e conferência de que tudo está congelado | minutos |
| 7 | TEST, uma vez cada: baseline padrão, baseline calibrado, filtro, restrito, Pair-Aware, Two-Stage | minutos |
| 8 | comparações (seção 7) | minutos |

## 7. Comparações e interpretação

Métrica primária: F1 de `negation_of` no TEST. Complementares (descritivas):
precisão e recall de `negation_of`, macro-F1 e MCC.

| Comparação (BioBERTpt, seed 42, código novo) | Tipo |
| --- | --- |
| Two-Stage × baseline padrão | principal |
| Two-Stage × baseline calibrado | principal (controle de calibração) |
| Baseline calibrado × baseline padrão | exploratória (efeito da calibração sozinha) |
| Two-Stage × filtro; × restrito; × Pair-Aware | exploratórias (com restrito e Pair-Aware, só o F1 de `negation_of` é comparável) |

- Estatística: `significancia.comparar` — bootstrap pareado do F1 de
  `negation_of` (10.000 reamostras, IC95% por percentis, semente 42 pela regra
  de `protocolo.semente_bootstrap`) e McNemar exato.
- Redação: "na seed 42, diferença positiva / negativa / não detectada" (IC95%
  acima de 0, abaixo de 0, ou contendo o 0).
- Só a comparação com o baseline calibrado permite atribuir o efeito à
  arquitetura e não à calibração; sem diferença positiva contra ele, o texto
  não atribui o efeito à arquitetura.
- As duas comparações principais são reportadas sem correção de
  multiplicidade, e isso é declarado.
- O bootstrap e o McNemar medem a incerteza sobre os exemplos do TEST, dados
  os modelos treinados; não medem a variação do treino (inicialização, ordem
  dos lotes, dropout). Com uma semente não se separa o efeito da arquitetura
  da sorte da semente; não se alega robustez entre sementes.
- Os números do DEV do Two-Stage e do baseline calibrado são otimistas (τ e t
  escolhidos nele) e entram só como diagnóstico, rotulados; os de cada estágio
  nunca são apresentados como resultado da arquitetura.
- A configuração do baseline da seed 42 já teve o TEST avaliado no legado;
  isso é registrado como dado histórico, sem uso de nenhum número do legado.

## 8. Pendências registradas

- **C11 — revisão acadêmica (etapa 9):** sem o BERTimbau, a comparação entre
  encoders, pergunta central do texto atual do TCC, não terá resultado novo
  nesta fase. O TCC não é alterado agora.
- **C13 — modelo final (etapa 10):** a regra de escolha será definida numa
  etapa posterior.

## 9. Registro verificável

```json
{
  "versao": 1,
  "fase": "validação inicial, seed 42",
  "encoder": "biobertpt",
  "modelo": "pucpr/biobertpt-all",
  "seed": 42,
  "parada_antecipada": false,
  "portao_de_pesquisa": false,
  "desempate_de_epocas": "primeira época (máximo estrito)",
  "estrategias": {
    "baseline": {
      "epocas": 3,
      "campo_selecao": "dev_macro_f1"
    },
    "restrito": {
      "epocas": 10,
      "campo_selecao": "dev_macro_f1",
      "selecao_melhor_epoca": "dev_macro_f1 remapeado ao espaco completo",
      "remapeamento": "fora do espaco restrito -> no_relation"
    },
    "pair_aware": {
      "epocas": 10,
      "campo_selecao": "dev_macro_f1",
      "selecao_melhor_epoca": "dev_macro_f1 remapeado ao espaco completo",
      "remapeamento": "fora do espaco restrito -> no_relation"
    },
    "two_stage_e1": {
      "epocas": 3,
      "campo_selecao": "dev_ap_relacao"
    },
    "two_stage_e2": {
      "epocas": 10,
      "campo_selecao": "dev_negation_of_f1_relacoes_anotadas"
    },
    "filtro_pistas": {
      "lexico": {
        "min_freq": 3,
        "n_formas": 11,
        "lexico_sha1": "70c93fa807de"
      }
    },
    "baseline_calibrado": {
      "regra": "baseline_calibrado: P(negation_of) >= t -> negation_of; senão argmax entre associated_with e no_relation (empate -> associated_with); probabilidades de 4 casas do sidecar da base"
    }
  },
  "limiar": {
    "grade": [
      0.05,
      0.1,
      0.15,
      0.2,
      0.25,
      0.3,
      0.35,
      0.4,
      0.45,
      0.5,
      0.55,
      0.6,
      0.65,
      0.7,
      0.75,
      0.8,
      0.85,
      0.9,
      0.95
    ],
    "criterio": "maior F1 de negation_of (contagens) da saída final no DEV",
    "desempates": [
      "maior macro-F1 (contagens)",
      "menor |limiar - 0,5|",
      "maior limiar"
    ],
    "regra_two_stage": "two_stage: P(relacao) do estágio 1, arredondada a 4 casas, >= τ -> estágio 2 (argmax entre negation_of e associated_with); senão no_relation",
    "casas_roteamento": 4
  },
  "validacao_tecnica": {
    "epocas": {
      "baseline": 1,
      "two_stage_e1": 1,
      "two_stage_e2": 2
    },
    "nomes": {
      "baseline": "baseline_validacao_1ep",
      "two_stage_e1": "twostage_e1_validacao_1ep",
      "two_stage_e2": "twostage_e2_validacao_2ep",
      "two_stage": "twostage_validacao",
      "baseline_calibrado": "baseline_calibrado_validacao_1ep"
    }
  }
}
```
