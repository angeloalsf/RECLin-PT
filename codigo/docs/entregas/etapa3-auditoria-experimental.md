# Etapa 3 — Auditoria experimental e atualização do TCC (02/11/2026)

Auditoria das execuções gravadas na branch `resultados-colab` e atualização do TCC com a Pair-Aware linear como arquitetura oficial.

**O que não foi feito:**
- nenhum treino;
- nenhuma avaliação no TEST, e o arquivo do TEST não foi lido;
- nenhuma partição, anotação, predição, checkpoint ou configuração oficial foi alterada;
- nenhum resultado existente foi apagado ou sobrescrito;
- nenhum comando Git foi executado na pasta de trabalho.

**Acesso aos dados:**
- Única exceção autorizada: um clone **só leitura** da branch `resultados-colab` no ambiente isolado da sessão (commit `ad7c2a20d9dd78133f1781d646654fbc28f8cbc8`, de 22/10/2026).
- A partição DEV v2 e o `MANIFEST.json` foram copiados para lá só para leitura, com autorização, para obter o documento de cada candidato. Foram apagados ao final.

## 1. Inventário (16 pastas + 2 derivadas)

Fonte: `codigo/resultados/auditoria_execucoes/inventario.json`. A mesma tabela está no Apêndice F do TCC.

| Pasta | Tipo | Estratégia | Encoder | Commit | Identidade | Sessão registrada | Melhor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_biobertpt_seed42` | treinada | baseline | BioBERTpt-all | `cc3d8b3` | `8a6c19` | nova | 3/3 |
| `baseline_BioBERTpt-clin` | treinada | baseline | BioBERTpt-clin | `3e3cb39` | `0526d9` | retomada (passo 3.000) | 3/3 |
| `baseline_BERTimbau` | treinada | baseline | BERTimbau-base | `c2cc230` | `0526d9` | retomada (6.500) | 3/3 |
| `pairaware_biobertpt_seed42` | treinada | Pair-Aware linear | BioBERTpt-all | `cb88c52` | `ecbb98` | nova | 3/3 |
| `pair_aware__6d11685f9cdb__t1` | treinada | Pair-Aware linear | BioBERTpt-clin | `54faa5e` | `0526d9` | retomada (3.500) | 3/3 |
| `pair_aware__f7a5acb7f500__t1` | treinada | Pair-Aware linear | BERTimbau-base | `5a42f28` | `0526d9` | nova | 3/3 |
| `pair_aware_mlp_BioBERTpt-all` | treinada | Pair-Aware MLP | BioBERTpt-all | `e16836d` | `22339f` | retomada (4.500) | 3/3 |
| `twostage_e1_biobertpt_seed42` | estágio 1 | Two-Stage | BioBERTpt-all | — | `8a6c19` | retomada (6.000) | 2/3 |
| `two_stage_e1__82490c70baed__t1` | estágio 1 | Two-Stage | BioBERTpt-clin | — | `0526d9` | retomada (6.000) | 2/3 |
| `two_stage_e1_BERTimbau` | estágio 1 | Two-Stage | BERTimbau-base | — | `0526d9` | retomada (6.000) | 3/3 |
| `twostage_e2_biobertpt_seed42` | estágio 2 | Two-Stage | BioBERTpt-all | — | `8a6c19` | nova | 8/10 |
| `two_stage_e2__65e72d6c30e8__t1` | estágio 2 | Two-Stage | BioBERTpt-clin | — | `0526d9` | nova | 6/10 |
| `two_stage_e2_BERTimbau` | estágio 2 | Two-Stage | BERTimbau-base | — | `0526d9` | nova | 5/10 |
| `two_stage_limiar_BioBERTpt-all` | composição | Two-Stage | BioBERTpt-all | — | — | — | τ = 0,95 |
| `two_stage_limiar_BioBERTpt-clin` | composição | Two-Stage | BioBERTpt-clin | — | — | — | τ = 0,95 |
| `twostage_bertimbau_seed42` | composição | Two-Stage | BERTimbau-base | — | — | — | τ = 0,85 |
| `baseline_calibrado_biobertpt_seed42` (`tcc/analise/execucoes_derivadas_dev/`) | derivada | baseline calibrado | BioBERTpt-all | — | `8a6c19` | — | t = 0,90 |
| `filtro_biobertpt_seed42` (idem) | derivada | filtro léxico | BioBERTpt-all | — | `8a6c19` | — | léxico D4 |

**Verificado:**
- Todas as execuções treinadas e estágios estão **concluídos** e usam a semente 42.
- Todas usam as mesmas partições: treino `b8aaafce…`, DEV `19feb024…`, versão `v2-texto-fonte`.
- Cada encoder tem uma única revisão de pesos:

  | Encoder | Revisão |
  | --- | --- |
  | `pucpr/biobertpt-all` | `36b11cf` |
  | `pucpr/biobertpt-clin` | `b59beec` |
  | `neuralmind/bert-base-portuguese-cased` | `94d69c9` |

- As 7 execuções de três rótulos registram `test_avaliado = false`.
- Nenhuma pasta tem métrica de TEST.
- A melhor época registrada é a de maior valor de seleção em todas as execuções.
- O SHA-256 de `metricas.json` confere com o registrado em `experimento.json` nas 7 execuções que o têm.

**Classificação:**

| Situação | Execuções |
| --- | --- |
| Concluídas e validadas (métricas recalculadas e iguais) | 7 treinadas, 3 composições, 2 derivadas |
| Concluídas, sem métrica própria (estágios) | 6 |
| Interrompidas sem retomada / incompletas | nenhuma |

**Checkpoints e logs: não verificáveis daqui.**
- Os checkpoints ficam no Drive. O `experimento.json` registra o destino e o número de arquivos sincronizados.
- `hf = null` em todas as execuções.
- Não há logs de console no repositório.

## 2. Métricas recalculadas (DEV, semente 42)

Fonte: `metricas_dev.json`.

**Como foram recalculadas:**
- A partir de `y_true` e `y_pred` gravados, com `reclin.avaliacao.metricas.avaliar`.
- O `classification_report` é **idêntico** ao de `metricas.json` nos 12 sistemas, bem como a matriz de confusão gravada.
- O `y_true` de cada sistema confere com:
  - o `y_true_sha256` do conjunto registrado;
  - o DEV reconstruído da partição congelada (`candidatos_sha256` `d3401d56…`).

| Sistema | P neg | R neg | F1 neg | VP/FP/FN | F1 assoc | F1 no_rel | Macro-F1 | MCC |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Pair-Aware linear, BioBERTpt-all | 0,742 | 0,940 | 0,829 | 141/49/9 | 0,436 | 0,956 | 0,741 | 0,497 |
| Pair-Aware linear, BioBERTpt-clin | 0,732 | 0,947 | 0,826 | 142/52/8 | 0,454 | 0,958 | 0,746 | 0,514 |
| Pair-Aware linear, BERTimbau-base | 0,713 | 0,960 | 0,818 | 144/58/6 | 0,435 | 0,955 | 0,736 | 0,497 |
| Baseline, BioBERTpt-all | 0,696 | 0,947 | 0,802 | 142/62/8 | 0,408 | 0,946 | 0,719 | 0,480 |
| Baseline, BioBERTpt-clin | 0,722 | 0,933 | 0,814 | 140/54/10 | 0,426 | 0,952 | 0,731 | 0,489 |
| Baseline, BERTimbau-base | 0,708 | 0,920 | 0,800 | 138/57/12 | 0,417 | 0,951 | 0,723 | 0,480 |
| Two-Stage, BioBERTpt-all (τ = 0,95) | 0,798 | 0,867 | 0,831 | 130/33/20 | 0,366 | 0,978 | 0,725 | 0,462 |
| Two-Stage, BioBERTpt-clin (τ = 0,95) | 0,768 | 0,907 | 0,832 | 136/41/14 | 0,380 | 0,977 | 0,730 | 0,464 |
| Two-Stage, BERTimbau-base (τ = 0,85) | 0,753 | 0,913 | 0,825 | 137/45/13 | 0,461 | 0,968 | 0,751 | 0,505 |
| Pair-Aware MLP, BioBERTpt-all | 0,715 | 0,953 | 0,817 | 143/57/7 | 0,438 | 0,956 | 0,737 | 0,498 |
| Baseline calibrado, BioBERTpt-all (t = 0,90) | 0,723 | 0,920 | 0,809 | 138/53/12 | 0,407 | 0,946 | 0,721 | 0,479 |
| Filtro léxico, BioBERTpt-all | 0,750 | 0,900 | 0,818 | 135/45/15 | 0,408 | 0,946 | 0,724 | 0,477 |

**Suporte no DEV:** 150 `negation_of`, 763 `associated_with`, 18.151 `no_relation`.

**Precisão, recall, TP, FP e FN de `associated_with` e `no_relation`:** tabela `tab:dev_classes` do TCC e `metricas_dev.json`.

**Conferências com versões anteriores:** os números do BioBERTpt-all e do baseline BERTimbau coincidem com os do TCC da Etapa 2.

### Comparações pareadas (descritivas, DEV)

Fonte: `comparacoes_dev.json`. Implementação:
- **Bootstrap por candidato:** 10.000 reamostras, semente 42. É o do protocolo; o F1 de `negation_of` sai **idêntico** ao de `significancia.comparar` (conferido por teste).
- **Bootstrap por documento:** novo, reamostra os 100 documentos do DEV.
- **McNemar exato.**

Leituras (diferença "detectada" = IC95% exclui o zero):

**Pair-Aware linear × baseline (mesmo encoder)**
- Macro-F1 e F1 de `associated_with`: diferença positiva nos três encoders, nos dois níveis de reamostragem.
- F1 de `negation_of`: diferença detectada só com o BioBERTpt-all (+0,027; IC cand. [+0,005; +0,051]; IC doc. [+0,001; +0,056]).

**Pair-Aware linear × Two-Stage**
- F1 de `negation_of`: nada detectado (−0,001 a −0,007).
- Com os BioBERTpt, a Pair-Aware linear tem F1 de `associated_with` maior (+0,071 e +0,074, detectado nos dois níveis).
- Com o BERTimbau, o Two-Stage tem macro-F1 e `associated_with` maiores (detectado só por candidato).

**Pair-Aware linear × MLP:** nada detectado.

**Pair-Aware linear × baseline calibrado**
- F1 de `negation_of`: +0,020, com limite inferior do IC por candidato **igual a 0,0** (não detectado).
- Macro-F1: diferença positiva.

**Encoders**
- Nenhuma das 9 comparações detecta diferença no F1 de `negation_of`.
- O BioBERTpt-clin tem `associated_with` maior que o BioBERTpt-all na Pair-Aware linear e no baseline, e maior que o BERTimbau na Pair-Aware linear (nos dois níveis).
- No Two-Stage, as diferenças entre encoders acompanham o τ.

**Bootstrap por documento:** dá intervalos em geral mais largos. Diferenças detectadas só por candidato (não por documento):
- Pair-Aware × Two-Stage (BioBERTpt-clin, macro-F1);
- Pair-Aware × Two-Stage (BERTimbau, `associated_with` e macro-F1);
- BioBERTpt-clin × BERTimbau (Pair-Aware, macro-F1);
- Two-Stage × baseline (BioBERTpt-all e -clin, `associated_with`);
- Two-Stage × baseline calibrado (`associated_with`).

### `associated_with` (precisão baixa)

Fonte: `associated_with_dev.json`.

**Origem dos falsos positivos**

| Origem | Parcela dos FP |
| --- | --- |
| Par cuja **direção inversa** está anotada como `associated_with` | 25%–31% |
| Par sem relação nas duas direções | 69%–75% |
| É `negation_of` | ≤ 1,4% |

- No DEV, só 4 das 763 `associated_with` candidatas têm o par inverso também anotado.

**Precisão por distância** (estratégias de três rótulos):

| Distância | Precisão |
| --- | --- |
| 0–1 caractere | 0,35–0,39 |
| 11–25 caracteres | 0,14–0,17 |

- **AP de `associated_with`:** 0,396–0,446. É maior na Pair-Aware linear que no baseline nos três encoders.

**Two-Stage**
- O τ congelado (critério: F1 de `negation_of`) ficou no **extremo da grade (0,95)** com os dois BioBERTpt.
- Margens sobre τ = 0,90: 0,0014 e 0,0038 de F1.
- Custo:

  | | τ = 0,90 | τ = 0,95 |
  | --- | --- | --- |
  | Pares roteados (BioBERTpt-all) | 1.303 | 592 |
  | Macro-F1 | — | cerca de 0,025 a menos |
  | Recall de `associated_with` | — | cerca de 0,3 |

- O estágio 2 acerta o tipo em mais de 98% dos pares roteados com relação. As perdas estão no estágio 1 e no limiar.

## 3. Divergências encontradas

1. **Nomes dos estágios do Two-Stage BERTimbau.**
   - O `limiar.json` de `twostage_bertimbau_seed42` registra `two_stage_e1__727707a32ba8__t1` e `two_stage_e2__9c00a5279316__t1`.
   - As pastas e o campo `nome` dos registros são `two_stage_e1_BERTimbau` e `two_stage_e2_BERTimbau`.
   - O `sha256_config` canônico e o `sha256_melhor_modelo` coincidem, o que identifica os mesmos estágios: renomeação depois da composição. **Resolução proposta:** registrar a correspondência; não renomear nada.
2. **Sessões não registradas.**
   - 7 treinos foram retomados e registram só a sessão final. `treino.json` é gravado ao fim de cada sessão (`reclin/treino/classificador.py`), então a sessão que caiu no Colab não deixa registro.
   - O estado do RNG é salvo no checkpoint. A retomada exige a mesma identidade de código, e todas conferem.
   - **Limitação de reprodutibilidade estrita, documentada.**
3. **Estágios do Two-Stage.** Não têm `experimento.json` (commit desconhecido) nem o campo `test_avaliado`.
4. **Quatro identidades de código em sete commits.** As diferenças, conferidas pelos manifestos gravados, são:

   | De | Para | Arquivos que mudam |
   | --- | --- | --- |
   | `8a6c19` | `ecbb98` | `estrategias/pair_aware/*`, `restrito.py`, `execucao/subconjunto.py`, `__init__.py`, `scripts/treinar.py`, `scripts/avaliar_test.py` (a D10) |
   | `ecbb98` | `22339f` | `pair_aware_mlp/*`, `__init__.py`, os dois scripts |
   | `22339f` | `0526d9` | só `reclin/config.py` (os novos encoders) |

   A identidade atual da pasta de trabalho é `0526d9…` (71 arquivos).
5. **Código na branch `resultados-colab`.** O código da branch está na identidade `8a6c19` (69 arquivos), anterior à D10. O protocolo dessa branch não tem a D10 nem a E2. A branch serve só para resultados; a fonte do protocolo é a pasta de trabalho.
6. **Arquitetura oficial × protocolo.**
   - O protocolo pré-registrado toma o **Two-Stage** como objeto das duas comparações principais no TEST (× baseline, × baseline calibrado). A comparação com a Pair-Aware é exploratória.
   - A designação da Pair-Aware linear como arquitetura oficial não está no protocolo.
   - **Decisão do autor:** emenda E3 antes de qualquer TEST (ver a seção 6).
7. **Negation 2.646 × 2.299.** As unidades são diferentes:
   - 2.646 ocorrências do rótulo, depois de separar as anotações multirrótulo por "|";
   - 2.299 entidades, porque 347 têm o rótulo repetido.
   - O mesmo vale para `Abbreviation` (12.592 × 12.290), `Finding` (6.827 × 6.747) e TPP (4.759 × 4.716).
   - **Corrigido no TCC (§4.5.2).**
8. **Scripts antigos do TCC.**
   - `tcc/analise/analise_dev.py` (local) e `inventario_execucoes.md`/`resultados_dev.json` usam nomes antigos de pasta (`twostage_biobertpt_seed42` como BioBERTpt-all, `pair_aware_mlp__6d9b07b48c37__t1`, `baseline__fdcac6602f78__t1`).
   - Na branch atual, `twostage_bertimbau_seed42` é a composição do **BERTimbau** e a do BioBERTpt-all é `two_stage_limiar_BioBERTpt-all`.
   - Esses arquivos ficam preservados como histórico. As tabelas novas vêm de `gerar_tabelas_etapa3.py`.
9. **Texto anterior do TCC.**
   - Dizia "três execuções retomadas" (são sete) e "duas execuções fora do escopo" (são oito treinos de encoder variante mais a MLP).
   - Dizia também "BioBERTpt-clin ainda não avaliado".
   - **Corrigido.**

## 4. Arquivos

**Criados**
- `codigo/reclin/analises/auditoria_execucoes.py`: inventário, vínculos, recálculo, AP, bootstrap por candidato e por documento, origem dos FP, grade de τ. Fica fora da identidade do código.
- `codigo/diagnosticos/auditar_execucoes.py`: ponto de entrada; só o DEV.
- `codigo/testes/unidade/test_auditoria_execucoes.py`: 10 testes sintéticos.
- `codigo/resultados/auditoria_execucoes/`: `inventario.json`, `metricas_dev.json`, `comparacoes_dev.json`, `two_stage_tau.json`, `associated_with_dev.json`. Só números, ids de execução e de documento; nenhum texto clínico.
- `codigo/docs/entregas/etapa3-auditoria-experimental.md`: esta nota.
- `tcc/analise/gerar_tabelas_etapa3.py`
- `tcc/tcc/src/apendices/apendice_f_inventario.tex`
- `tcc/tcc/src/tabelas/`: `dev_sistemas`, `dev_classes`, `dev_comparacoes_pa`, `dev_comparacoes_ts`, `dev_comparacoes_encoders`, `dev_assoc_fp`, `tau_encoders`, `two_stage_diag`, `dev_historico_execucoes`, `dev_estagios`, `inventario_execucoes` (`.tex`).
- `tcc/tcc/src/imagens/resultados/tau_two_stage_encoders.png`

**Modificados**

| Arquivo | Mudança | Justificativa |
| --- | --- | --- |
| `pre_textuais/resumo.tex` | Resumo e abstract: Pair-Aware linear oficial; 3 encoders × 3 estratégias; o Two-Stage deixa de ser a referência | resultados novos; pedido da etapa |
| `textuais/introducao.tex` | Objetivos específicos: arquitetura oficial, três encoders, auditoria; "estratégia de referência preliminar" removida | idem |
| `textuais/metodologia.tex` | Encoders; frase da `Negation` (ocorrências × entidades); escopo do protocolo (variantes de encoder); parágrafo sobre a designação oficial fora do protocolo; bootstrap por documento | divergências 6 e 7; nova análise |
| `textuais/proposta_prototipo.tex` | Papéis das estratégias; BioBERTpt-clin com checkpoint e resultado; paridade com 3 encoders; subseção "Pair-Aware linear: a arquitetura oficial" | idem |
| `textuais/experimentos_e_resultados.tex` | Reescrito: inventário; épocas; visão geral (12 sistemas); Pair-Aware frente às demais; encoders; `associated_with`; Two-Stage com 3 encoders; exploratório | resultados novos |
| `textuais/discussao.tex` | Reescrito: arquitetura oficial; Two-Stage e limiar; encoder; `associated_with` (direção); limitações (ii), (iii), (vi) novas; ameaças com o inventário e o bootstrap por documento | idem |
| `textuais/conclusao.tex` | Reescrita com os resultados novos; trabalhos futuros: emenda E3, controles por encoder, direção de `associated_with` | idem |
| `apendices/apendice_c_tau.tex`, `apendice_d_historico.tex`, `apendice_e_exploratoria.tex` | Usam as tabelas novas (3 encoders) | idem |
| `imagens/figuras/ciclo-desenvolvimento.tex` | Notas da figura (fine-tuning 3×3; controles só com o BioBERTpt-all) | idem |
| `main.tex` | Inclui o Apêndice F | inventário |
| `macros.tex` | `\usepackage{rotating}` | tabelas de comparação em página girada |
| `codigo/diagnosticos/README.md`, `codigo/README.md`, `codigo/docs/entregas/README.md` | Seção e linhas da auditoria | documentação |

**Removidos:** nenhum.

**Tabelas antigas preservadas, não mais incluídas no texto:** `dev_negacao`, `dev_outras_classes`, `dev_comparacoes`, `dev_encoders`, `dev_history`, `dev_estagio1`, `dev_estagio2`, `tau_completo`. A figura antiga `tau_two_stage.png` também ficou.

## 5. Testes e verificações

| Verificação | Resultado |
| --- | --- |
| `test_auditoria_execucoes.py` (10), `test_identidade_codigo.py`, `test_avaliacao.py` | **56 passaram** |
| `test_pacote.py` (com `PYTHONPATH=codigo`) | 120 passaram, 9 falharam **por ambiente** (sem `LICENSE`, partições, referências do legado e `reclin.experimento` completo na cópia de trabalho); o teste de importação de todos os módulos, inclusive o novo, passou |
| Equivalência do bootstrap por candidato com `significancia.comparar` | idêntica (teste + conferência nos dados reais) |
| AP contra `sklearn.average_precision_score` | igual até 1e-16 (fora da suíte) |
| Identidade do código | `0526d9123ecc…`, 71 arquivos (inalterada) |
| Determinismo | as tabelas regeradas de uma segunda execução são idênticas (md5) |
| Compilação do TCC | `TCC_etapa3.pdf`, 86 páginas, sem avisos de referência indefinida |
| PDF de diferenças | `TCC_etapa3_diff.pdf`, 94 páginas, contra o TCC da Etapa 2 |
| Revisão visual | caps. 6–8, resumo e apêndices C–F |

**Não executado:**
- a suíte completa (`make -C codigo testes`), que exige torch e os dados;
- os testes de integração e de equivalência.

## 6. Pendências e decisões do autor (recomendação para a Etapa 4)

1. **Emenda E3 ao protocolo, antes de qualquer TEST**, para definir:
   - a Pair-Aware linear como arquitetura principal;
   - as comparações principais: Pair-Aware linear × baseline e × baseline calibrado, do mesmo encoder;
   - o critério de escolha do encoder (F1 de `negation_of`? macro-F1? custo?);
   - as variantes de encoder como parte formal da fase.

   Sem a emenda, o TEST continua organizado em torno do Two-Stage.
2. **Confirmar no texto** que a designação "oficial" foi feita com os resultados do DEV disponíveis. O TCC afirma isso com base nas datas dos registros: os resultados do BioBERTpt-all existiam desde 14–15/10, e o TCC da Etapa 2 ainda tratava o Two-Stage como referência.
3. **Controles para os outros encoders.** Calcular o baseline calibrado e o filtro léxico para o BioBERTpt-clin e o BERTimbau (sem GPU), se a emenda os exigir.
4. **Proveniência antes do TEST:**
   - conferir os checkpoints no Drive (`pretreino.py conferir --execucoes`);
   - decidir se as quatro identidades de código são aceitas com as diferenças documentadas, ou se o protocolo exige uma única.
5. **Correspondência de nomes do Two-Stage BERTimbau** (divergência 1): registrar na emenda ou num arquivo de vínculos.
6. Pendências anteriores continuam:
   - Makefile (alvo `diagnosticos`);
   - comentário do 11,6% em `tarefa.py`;
   - nota sobre a redistribuição do corpus;
   - slides.
7. **Repositório:** pode voltar a ser privado. O clone na nuvem é descartado com a sessão.

## 7. Adendo (02/11/2026): emenda E3 e cronologia da arquitetura oficial

Duas atividades pedidas depois do relatório. Nenhum treino, nenhum TEST, nenhum código científico alterado.

1. **Emenda E3.**
   - Arquivo: `docs/protocolo/emenda-e3-pair-aware.md`. É uma **proposta, não em vigor**.
   - O protocolo `fase-biobertpt-seed42.md` e o seu bloco JSON não foram alterados.
   - Decisões pendentes do autor: E3-D1 a E3-D5.
2. **Cronologia.**
   - Arquivo: `docs/analises/cronologia-arquitetura-oficial.md`.
   - A afirmação do TCC está correta: o primeiro registro da designação é de 01/11/2026, e o último treino no DEV terminou em 22/10/2026.
   - O texto foi precisado em `metodologia.tex` (§4.6), `discussao.tex` (§7.1 e limitação ii) e `conclusao.tex` (trabalhos futuros).
   - Fica para decisão do autor uma frase de transparência sobre a D10, não aplicada.

## 8. Adendo (02/11/2026): revisão 2 da proposta E3

A E3 foi revista com as decisões do autor de 02/11/2026. Continua sendo uma proposta, não aprovada, e o protocolo vigente não foi alterado.

**Decisões incorporadas:**
- Pair-Aware linear como arquitetura principal; o Two-Stage não justifica a complexidade adicional;
- BioBERTpt-all como encoder de referência;
- configuração de referência atual (Pair-Aware linear, BioBERTpt-all, semente 42), não definitiva;
- etapa de aprimoramento com novos experimentos, só em TRAIN e DEV;
- modelo final definido depois do congelamento;
- TEST avaliado uma única vez, só para o modelo final.

**Pendentes:**
- E3-D3: critérios estatísticos;
- E3-D6: critério de seleção da configuração final.

**Consequência declarada:** sem comparação confirmatória entre sistemas no TEST.

**TCC atualizado:** resumo, introdução (objetivo), §4.6, cap. 5, introdução do cap. 6 e figura do ciclo, §7.1–7.3, limitações e cap. 8.
