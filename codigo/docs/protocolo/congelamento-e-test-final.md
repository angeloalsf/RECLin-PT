# Congelamento da configuração final e avaliação única do TEST

Etapa 5 (recomendação G2), 16/11/2026. Estado: **nada congelado; TEST não avaliado**.

## Regra

O TEST das partições científicas é avaliado **uma única vez**, só para a
configuração final, depois de autorização explícita do autor. Ele não compara
experimentos. A regra já estava no protocolo; a partir desta etapa o código a
impõe.

## O que a trava faz

`reclin/execucao/congelamento.py` é chamado como primeira instrução das três
funções que avaliam um modelo no TEST: `treino.classificador.avaliar_test`
(`scripts/avaliar_test.py`), `two_stage.composicao.avaliar_test`
(`scripts/two_stage.py avaliar-test`) e `baseline_calibrado.aplicar_test`
(`scripts/baseline_calibrado.py aplicar-test`). Para partições científicas (v1 e
v2, reconhecidas pelo caminho ou pelo SHA-256 do TEST no MANIFEST; pasta sem
MANIFEST também é tratada como científica), a avaliação é recusada quando:

| Situação | Mensagem |
| --- | --- |
| não existe `resultados/congelamento/configuracao_final.json` | não há configuração final congelada |
| o registro foi editado depois de gravado | o registro foi alterado |
| a execução pedida não é a congelada | a configuração congelada é outra |
| `config.json`, `predicoes_dev.json`, `limiar.json` ou os pesos do melhor modelo mudaram | a execução mudou depois do congelamento |
| `--liberacao` ausente ou incorreto | código de liberação ausente ou incorreto |
| já existe avaliação em `avaliacoes_test_final.jsonl` | a configuração final já foi avaliada (`--reavaliar` registra a repetição) |

Partições técnicas (subconjuntos com MANIFEST próprio, usados nos testes) não
são travadas. O módulo não importa torch e não abre nenhuma partição: lê só o
MANIFEST e os registros da execução.

## Procedimento

0. **Emenda E3 em vigor.** `emenda-e3-pair-aware.md` ainda é uma **proposta**. Ela
   exige, antes do congelamento, as decisões E3-D3 (critérios estatísticos) e
   E3-D6 (critério de seleção da configuração final, no DEV), a aprovação
   registrada do autor e o SHA-256 do texto aprovado na seção 8 do protocolo.
   Sem isso, vale o protocolo da fase, organizado em torno do Two-Stage.
1. **Decisão do autor**, registrada em `docs/protocolo/` (critério, configuração
   escolhida e execução; só DEV), como pede a seção A5 da proposta.
2. **Congelar**, na máquina em que estão os pesos do melhor modelo (a pasta
   `checkpoints/melhor_modelo/` da execução; os resultados versionados não a
   contêm, ela está no Drive e no Hugging Face):

   ```bash
   python codigo/scripts/congelar.py --execucao <pasta da execução final> \
       --responsavel "<nome>" --justificativa "<critério e evidência, só DEV>"
   ```

   O comando não lê o TEST. Grava o registro (hashes de `config.json`,
   `predicoes_dev.json`, `limiar.json` quando existe, e do melhor modelo; código
   da execução e do congelamento; MANIFEST das partições) e imprime o **código de
   liberação**. Um congelamento existente não é substituído.
3. **Versionar** o registro (commit manual do autor) antes de avaliar.
4. **Autorização explícita** do autor para a avaliação.
5. **Avaliar, uma vez**:

   ```bash
   python codigo/scripts/avaliar_test.py --execucao <pasta da execução final> --liberacao <código>
   ```

   A avaliação grava `predicoes_test.json`, as métricas e uma linha em
   `resultados/congelamento/avaliacoes_test_final.jsonl`.
6. Relatar o resultado como está, inclusive a análise sem as três notas com cópia
   no treino, e não voltar ao DEV para ajustar nada.

`python codigo/scripts/congelar.py --execucao <pasta> --conferir` confere um
congelamento existente sem gravar.

## Recomendação preliminar da configuração definitiva

**Pair-Aware Linear, BioBERTpt-all (rev. `36b11cf`), semente 42, `max_gap=25`,
perda ponderada (`balanced`), execução `pairaware_biobertpt_seed42`** — a
configuração de referência (A3) da proposta de emenda E3. É preliminar: a recomendação final depende da
adjudicação (M1) e fica registrada depois dela; a ablação (M3) explica a origem
da diferença para o baseline e não entra na escolha (as duas variantes não são
elegíveis ao TEST).

Evidência (DEV, uma semente):

| | P neg | R neg | F1 neg | Macro-F1 |
| --- | --- | --- | --- | --- |
| Pair-Aware Linear ponderada | 0,742 | 0,940 | 0,829 | 0,741 |
| Pair-Aware Linear sem pesos | 0,794 | 0,873 | 0,832 | 0,752 |

- F1 de `negation_of` (métrica principal): +0,002 a favor da sem pesos, não
  detectada (documento: [−0,025; +0,028]). Macro-F1: +0,011, não detectada
  ([−0,005; +0,027]).
- A diferença é de ponto de operação e é detectada nos dois sentidos: precisão
  +0,052 [+0,026; +0,079], recall −0,067 [−0,113; −0,029] (por documento).
- Os 34 falsos positivos da sem pesos estão todos entre os 49 da ponderada; os 9
  falsos negativos da ponderada estão todos entre os 19 da sem pesos.

Sem diferença detectada na métrica principal, a recomendação é manter a
configuração pré-registrada, em torno da qual todos os controles foram
calculados. A sem pesos atende à prioridade de precisão, ao custo de recall, e é
uma alternativa legítima **se adotada por emenda antes do TEST**.

### Sugestão para as decisões pendentes da E3

- **E3-D3:** alternativa (b) da proposta, sem a parte ainda não implementada:
  bootstrap pareado por candidato e por documento, com "detectada" só quando os
  dois intervalos excluem o zero. É o que a auditoria já calcula. A correção de
  Holm não está implementada; se for exigida, é trabalho adicional.
- **E3-D6:** alternativa (a): manter a configuração de referência, salvo se um
  experimento a superar no DEV, no F1 de `negation_of`, pelo critério de E3-D3.
  Para o experimento sem pesos, cujo resultado já é conhecido, o critério teria
  de ser declarado como pós-DEV; por esse critério ele não supera a referência.

### Regra de decisão proposta (a fixar antes de abrir as decisões da adjudicação)

Trocar para a execução sem pesos (`pair_aware__a3ac671a77e2__t2`) somente se, sob
a referência adjudicada por consenso (`adjudicacao_ponto_de_operacao`), o F1 de
`negation_of` dela superar o da ponderada com IC95% por documento excluindo o
zero. Caso contrário, manter a ponderada. **Esta regra é uma proposta e depende
da aprovação do autor.**

## O que a trava não cobre (achados; decisão do autor)

- `scripts/regra_pura.py` avalia DEV e TEST por padrão e não tem trava. A suíte
  de equivalência com o legado (`testes/equivalencia/`) o executa sobre as
  partições v1, inclusive o TEST, para comparar byte a byte com os resultados
  do legado.
- `scripts/filtro_pistas.py --particao test` exige que o TEST da base já tenha
  sido avaliado; com a trava, isso só ocorre para a execução congelada.
- Leituras estruturais do `test.jsonl`: conferência de SHA-256 do MANIFEST,
  estatísticas do corpus (`scripts/analisar.py estatisticas`,
  `diagnosticar.py` na contagem descritiva de distâncias) e testes de unidade
  das partições (`test_dados_v2`). Não calculam desempenho.
- `analisar.py --incluir-test` e `diagnosticar.py --test-autorizado` dependem só
  da opção explícita.

## Atualização de 22/11/2026 — configuração final decidida

O autor decidiu a configuração final: **Pair-Aware Linear sem pesos de classe
(`pair_aware__a3ac671a77e2__t2`) com a regra P(`negation_of`) ≥ 0,80**. As
decisões E3-D3 e E3-D6, o critério do limiar e a evidência do DEV estão em
`emenda-configuracao-final.md`. A sugestão anterior desta página para E3-D6
(manter a referência ponderada) não foi a adotada; a regra de troca ligada à
adjudicação deixa de ser necessária para a seleção.

O que mudou no código para isso (ver o relatório da Etapa 5, adendo de 22/11):

- `estrategias/pair_aware_limiar.py` e `scripts/pair_aware_limiar.py`: a regra e
  a busca de t (critério `avaliacao.limiar.PrecisaoComPisoDeRecall`);
- `treino/reempacotamento.py` e `scripts/reempacotar.py`: a base, treinada com
  outro código, passa ao código atual depois de reproduzir o DEV;
- `execucao/congelamento.py`: uma configuração derivada é congelada com as
  impressões da base (incluindo os pesos); a trava aceita, sob a mesma
  liberação e uma vez cada, a base (primeiro) e a derivada (depois); do
  `config.json` entram a configuração e a identidade do código, não o arquivo
  inteiro (a avaliação do TEST acrescenta a ele o conjunto do TEST);
- `notebooks/final/reclin_configuracao_final.ipynb`: o procedimento no Colab,
  só DEV, até o registro de congelamento.
