# Etapa 2 — Correções metodológicas e controles experimentais (01/11/2026; finalizada)

Implementação das correções da auditoria metodológica da Etapa 2, sem treino, sem avaliação do TEST e sem alterar partições, anotações, predições, configurações ou a arquitetura oficial. Nenhum comando Git foi executado.

## Resumo e decisões técnicas

1. **Tipos semânticos.** Preservados nas partições como estão: campo `type`, cópia do `tag` do XML. Continuam fora da entrada das quatro configurações; nada em `reclin/entrada.py`, `estrategias/` ou `treino/` foi tocado.
   - Nova área experimental independente: `codigo/diagnosticos/`.
   - **Controles informados pelas anotações**, que usam os tipos reais do SemClinBr sem treinar modelos:
     - tipo `Negation` → pista;
     - tipo `Negation` + posição → entidade seguinte;
     - relações previstas pelo tipo nos pares candidatos.
   - Só TRAIN e DEV; o código recusa o TEST.
2. **TEST.** Emenda E2 no protocolo (seção 8):
   - o TEST não é inédito;
   - segundo os registros, nenhuma configuração atual foi avaliada nele;
   - a Pair-Aware linear ainda não foi avaliada no TEST e não é avaliada nesta etapa.
   
   Não foi criada seção sobre versões anteriores, e nenhum número anterior é usado.
3. **Partições.** 800/100/100 e semente 42 mantidas.
   - Detector de quase duplicatas que só registra.
   - Análise de sensibilidade sem 9534, 9535 e 9536 **preparada** e pré-registrada como secundária (E2). No TEST, só roda com `--test-autorizado`; não foi executada.
4. **`max_gap = 25` mantido.** Documentação corrigida:
   - distância simétrica;
   - cobertura por classe;
   - custo da ampliação.
   
   O "11,6%" foi **reproduzido**: 1.324 de 11.458 relações do **corpus inteiro, TEST incluído**, com distância simétrica acima de 20.
5. **Reprodutibilidade.** Um ponto de entrada (`diagnosticos/diagnosticar.py`) para sementes, `max_gap`, tipos semânticos e quase duplicatas. Saídas só com números agregados e ids.
6. **Identidade do código preservada.**
   - Os módulos novos ficam em `reclin/analises/` (fora da identidade) e o ponto de entrada fora de `scripts/`.
   - Identidade recalculada depois das mudanças: `0526d9123eccfebace39bf072343d254d1bab650530d43c84d6cd022c0a6ef3c`, 71 arquivos, a mesma de antes.
   - Por isso o comentário com o "11,6%" em `reclin/tarefa.py` **não** foi editado; a correção está na documentação.

## Arquivos

**Criados**
- `codigo/reclin/analises/controles_anotacao.py`
- `codigo/reclin/analises/quase_duplicatas.py`
- `codigo/reclin/analises/sensibilidade_semente.py`
- `codigo/reclin/analises/exclusao_documentos.py`
- `codigo/diagnosticos/diagnosticar.py`
- `codigo/diagnosticos/README.md`
- `codigo/testes/unidade/test_diagnosticos.py`
- `codigo/docs/analises/controles-tipo-semantico.md`
- `codigo/docs/analises/janela-max-gap.md`
- `codigo/docs/analises/integridade-particoes.md`
- `codigo/docs/entregas/etapa2-correcoes-metodologicas.md` (esta nota)
- `codigo/resultados/diagnosticos/controles_tipo_semantico.json`
- `codigo/resultados/diagnosticos/quase_duplicatas.json`
- `codigo/resultados/diagnosticos/sensibilidade_semente.json`
- `codigo/resultados/diagnosticos/sensibilidade_max_gap.json`

**Modificados** (implementação e finalização)

| Arquivo | Mudança |
| --- | --- |
| `codigo/reclin/analises/__init__.py` | docstring: módulos novos |
| `codigo/reclin/analises/max_gap.py` | nova função `distancias_anotadas`; `sensibilidade` intacta |
| `codigo/docs/protocolo/fase-biobertpt-seed42.md` | cabeçalho e emenda E2 (revisada na finalização); bloco JSON intacto |
| `codigo/docs/entregas/README.md` | linha desta entrega |
| `codigo/README.md` | linhas `docs/analises/` e `diagnosticos/`; linha da `pair_aware_mlp` com a dependência permitida |
| `README.md` (raiz) | encoders, as quatro configurações e os controles, sementes, limitações e `max_gap` alinhados ao protocolo atual |
| `codigo/testes/unidade/test_pacote.py` | *finalização*: regra de dependências com lista explícita de funções compartilhadas e dois testes novos |
| `codigo/docs/estrategias/README.md` | *finalização*: papel de cada estratégia e separação dos controles diagnósticos |
| `codigo/docs/analises/janela-max-gap.md` | *finalização*: seção 4 (origem do 11,6%) centrada no cálculo e no conjunto usado |
| `codigo/docs/analises/controles-tipo-semantico.md` | *finalização*: correspondência com os controles pedidos e sugestões futuras (não implementadas) |
| `codigo/docs/entregas/etapa2-correcoes-metodologicas.md` | esta nota, atualizada para o estado final |

**Não modificado:** `codigo/Makefile`, protegido contra escrita pelas ferramentas remotas (ver pendências).

**Removidos:** nenhum.

## Métricas dos controles (dados reais, partições v2)

| Controle | Partição | TP | FP | FN | P | R | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tipo → pista (entidades) | TRAIN | 1.125 | 716 | 42 | 0,611 | 0,964 | 0,748 |
| | DEV | 136 | 94 | 5 | 0,591 | 0,965 | 0,733 |
| Tipo → `negation_of` (pares) | TRAIN | 1.213 | 6.191 | 42 | 0,164 | 0,967 | 0,280 |
| | DEV | 145 | 751 | 5 | 0,162 | 0,967 | 0,277 |
| Tipo + posição | TRAIN | 1.089 | 684 | 166 | 0,614 | 0,868 | 0,719 |
| | DEV | 130 | 92 | 20 | 0,586 | 0,867 | 0,699 |

Interpretação e limitações: `docs/analises/controles-tipo-semantico.md`.

## Testes e verificações

**Ambiente:** o shell do computador do autor não iniciou nesta sessão. Os testes rodaram numa cópia de trabalho no ambiente isolado da sessão:
- Python 3.13, numpy 2.5.3, pytest 9.1.1, **sem torch**;
- as partições v1 e v2 foram copiadas só para leitura na implementação, com autorização do autor, e **apagadas ao final**;
- a finalização não precisou dos dados.

### Implementação (executados uma vez; arquivos não modificados depois)

| Verificação | Resultado |
| --- | --- |
| `testes/unidade/test_diagnosticos.py` (23 testes, incluídos os de dados reais e o `lento` do detector) | **23 aprovados** |
| `test_analises`, `test_particoes`, `test_tarefa`, `test_util`, `test_versao_dados`, `test_avaliacao`, `test_negacao`, `test_identidade_codigo` | aprovados, exceto 1 pulado (`tokenizers` ausente) |
| `diagnosticar.py tudo` | gerou as 4 saídas; as partições conferem com o MANIFEST antes da leitura |
| Ausência de texto clínico | `test_quase_duplicatas_reais`: nenhum trecho de 30 caracteres de nenhum documento na saída |

### Finalização (testes direcionados)

| Verificação | Resultado |
| --- | --- |
| `test_pacote.py::test_um_nivel_so_importa_os_de_cima` (84 módulos) | **84 aprovados**, inclusive `estrategias/pair_aware_mlp/cabeca.py`, que falhava antes |
| `test_pacote.py::test_compartilhamento_entre_estrategias_so_de_funcoes_auxiliares_usadas` (novo) | **aprovado** |
| `test_pacote.py::test_a_pair_aware_linear_nao_importa_outras_estrategias` (novo) | **aprovado** |
| Mutações, numa cópia descartada | as três foram detectadas (ver abaixo) |
| Bloco JSON do protocolo | lido e válido (um bloco) |
| `testes/unidade/test_protocolo_fase.py` (protocolo modificado) | **5 aprovados**, num ambiente isolado com as versões fixadas de `requirements.txt`: torch 2.11.0, transformers 5.16.1, numpy 2.1.3, scipy 1.16.3; Python 3.13 |
| Identidade do código | recalculada: `0526d9123eccfebace39bf072343d254d1bab650530d43c84d6cd022c0a6ef3c`, 71 arquivos (**inalterada**) |
| Partições no computador do autor | tamanhos e datas de modificação iguais aos de antes da etapa (`dados/particoes-v2/`) |

**Mutações, cada uma revertida em seguida:**

| Mutação | Resultado |
| --- | --- |
| (1) A MLP importa `PairAwareClassifier` da linear | o teste falha |
| (2) A linear importa a MLP | dois testes falham. A regra antiga **não** pegava esse caso: comparava por prefixo, e `reclin.estrategias.pair_aware_mlp` começa com `reclin.estrategias.pair_aware` |
| (3) Uma classe na lista de permitidos | o teste falha |

**Não executados**

| Teste | Motivo |
| --- | --- |
| Suíte completa e `make -C codigo testes` | não pedidos nesta finalização |
| Os demais testes que exigem torch, como a importação de todos os módulos e o `--help` dos scripts de treino | não pedidos nesta finalização |
| Testes de integração e de equivalência | não pedidos nesta finalização |
| Os 23 testes de `test_diagnosticos.py` | aprovados na implementação; o arquivo e os módulos que ele testa não mudaram |

## Regra de dependências entre estratégias: justificativa

**Problema:** `test_pacote.py::test_um_nivel_so_importa_os_de_cima` falhava porque `estrategias/pair_aware_mlp/cabeca.py` importa de `estrategias/pair_aware/cabeca.py` quatro funções:
- `_carregar_encoder`;
- `ids_dos_marcadores`;
- `posicoes_dos_marcadores`;
- `contar_marcadores_ausentes`.

**Por que mudar a regra, e não o código:**
- O compartilhamento é deliberado e documentado (`docs/entregas/pair-aware-mlp.md`, "reaproveita… sem cópia").
- Ele garante que as duas variantes leiam o par **exatamente** da mesma forma: mesma carga do encoder sem pooler, mesma posição dos marcadores, mesmo *fallback*. A única diferença entre elas fica sendo a cabeça.
- Copiar as funções duplicaria lógica e arriscaria divergência silenciosa.
- Mover as funções para um módulo comum mudaria arquivos de estratégia e a identidade do código das execuções (`0526d9…`).

**Solução:** uma exceção **explícita e estreita** no teste (`COMPARTILHADAS_ENTRE_ESTRATEGIAS`):
- Só a `pair_aware_mlp` pode importar, e só essas quatro funções de `reclin.estrategias.pair_aware.cabeca`.
- Qualquer outro nome continua proibido, como a cabeça `PairAwareClassifier`, a configuração ou a montagem.
- Dois testes novos garantem que a lista continua estreita:
  - cada nome é uma função definida no módulo de origem e de fato importada (sem entradas obsoletas);
  - a dependência tem uma direção só: a Pair-Aware linear oficial não importa outras estratégias.
- A comparação de prefixo da regra foi corrigida (`proprio + "."`), o que fechou a brecha da mutação (2).

**Nada mudou** nas cabeças, na representação dos pares, nos pesos, nos checkpoints ou no funcionamento das quatro configurações.

## Critérios de aceitação

| Critério | Situação |
| --- | --- |
| Controles semânticos preservados em `codigo/diagnosticos/`, separados das quatro configurações | atendido |
| Tipos preservados nas partições e fora da entrada da Pair-Aware linear | atendido: nenhum arquivo de entrada, estratégia ou treino alterado; partições com as mesmas datas e tamanhos |
| Documentação do TEST centrada no protocolo atual; Pair-Aware linear "ainda não avaliada"; sem afirmar que o TEST é inédito | atendido (E2 revisada) |
| Detector que só registra; sensibilidade sem 9534–9536 preparada e não executada no TEST | atendido |
| `max_gap = 25` mantido; definição simétrica, impacto por classe, cobertura e custo documentados; origem do 11,6% com o conjunto usado | atendido |
| Falha de `test_pacote` resolvida na regra de dependências, sem mudar arquitetura | atendido |
| Testes direcionados executados | atendido (tabela acima) |
| README sem informação metodológica contraditória | atendido nos pontos revistos: encoders, configurações, sementes, limitações, `max_gap`, estado |
| Suíte completa | **não executada** (fora do escopo desta finalização); não declarada aprovada |

## Pendências que permanecem

1. **`codigo/Makefile`:** protegido contra escrita remota. Para ter `make diagnosticos`, acrescente à mão o trecho abaixo, e `diagnosticos` à lista `.PHONY`:

   ```make
   diagnosticos:
   	$(PY) $(CODIGO)/diagnosticos/diagnosticar.py tudo --pasta $(CODIGO)/resultados/diagnosticos
   ```

2. **Comentário do "11,6%" em `reclin/tarefa.py`:** não editado, porque mudaria a identidade do código. A origem correta está em `docs/analises/janela-max-gap.md` e na E2.
3. **Demais testes com torch:** fora desta finalização; rodam na suíte completa. `test_protocolo_fase.py` já foi executado e aprovado.
4. **Protocolo, seção 7:** a frase anterior a esta etapa sobre o TEST do baseline foi mantida, por ser texto já registrado; a E2 a complementa. Remover ou reescrever é decisão do autor.
5. **`README.md` da raiz:** a seção "Projeto legado" e a nota de hardware do treino continuam como estavam; não foram criadas nem ampliadas.
6. **Versionamento** de `codigo/resultados/diagnosticos/` (só agregados e ids): decisão do autor.

## Sincronização com o TCC (02/11/2026)

O TCC (`tcc/tcc/src/`) foi lido e avaliado contra o código, o protocolo e os diagnósticos da Etapa 2. **Alterações acadêmicas feitas nas fontes LaTeX:**

| Arquivo do TCC | Trecho | O que mudou e por quê |
| --- | --- | --- |
| `textuais/metodologia.tex`, §4.5.3 Particionamento | dois parágrafos e duas tabelas novas | partição única; 80/10/10 como convenção; sensibilidade à semente (Tabela `tab:sensibilidade_semente`); quase duplicatas (Tabela `tab:quase_duplicatas`) e análise secundária pré-registrada sem 9534–9536 |
| `textuais/metodologia.tex`, §4.5.4 Geração de pares candidatos | propriedade (iii) | a distância é **simétrica** (o texto dizia "fim de e1 → início de e2") |
| `textuais/metodologia.tex`, §4.5.4 e Quadro `quad:exemplo_candidatos` | exemplo trocado | o exemplo antigo (documento 8906) é uma negação com **direção invertida** (o conceito é e1) e um diagnóstico, não uma negação típica. O novo usa o trecho já mostrado na Figura de anotação (documento 9091, ids 9513, 9514, 9554), conferido nos dados v2: seis candidatos, só 9514 → 9554 é `negation_of` |
| `textuais/metodologia.tex`, §4.5.5 Teto de recall | dois parágrafos | cobertura × recall (DEV 155 → 153 → 150; recall sobre candidatos × sobre anotadas; no teste os dois coincidem para `negation_of`) e a origem do 11,6% (1.324/11.458, corpus inteiro, **teste incluído**) |
| `textuais/metodologia.tex`, nova §4.5.6 `sec:controles_tipo` | subseção e tabela `tab:controles_tipo` | os tipos ficam nas partições e fora da entrada; os controles diagnósticos informados pelas anotações, com TP/FP/FN/P/R/F1 (treino e DEV) e interpretação separada dos resultados |
| `textuais/metodologia.tex`, §4.6 Protocolo | frase sobre o teste | "nenhum sistema foi avaliado no teste" → nenhuma configuração deste trabalho, entre elas a Pair-Aware, foi avaliada; o teste **não é inédito**; análise principal e secundária |
| `textuais/proposta_prototipo.tex`, §5.3 Marcadores | item 1 e Quadro `quad:entrada` | justificativa da ausência de tipo (remete aos controles); entrada do novo exemplo |
| `textuais/experimentos_e_resultados.tex` | introdução do capítulo | mesma correção da frase sobre o teste |
| `textuais/discussao.tex`, Limitações e Ameaças | itens (ix) e (x); um parágrafo | inconsistências de anotação (38/5 invertidas, cerca de 37% das pistas sem relação, concordância de cerca de 0,57 em `associated_with` e 1,0 em `negation_of` nas notas duplicadas); partição única e semente; duplicatas e janela fixada com contagem que incluiu o teste |
| `textuais/conclusao.tex` | contribuição (v) e trabalhos futuros | diagnósticos reproduzíveis; análise sem as notas duplicadas e adjudicação cega; tipos semânticos só como **proposta** futura (variante separada, `Negation` mascarado) |
| `tabelas/controles_tipo.tex`, `tabelas/sensibilidade_semente.tex`, `tabelas/quase_duplicatas.tex` (novas) | — | geradas por `tcc/analise/gerar_tabelas_diagnosticos.py` a partir de `codigo/resultados/diagnosticos/*.json` |
| `tabelas/max_gap.tex` | coluna `no_relation`/relação | 15 casas decimais → 1; correção em `tcc/analise/gerar_artefatos_tcc.py`; os números são os mesmos |

**Verificação:** compilação com `latexmk` e `pdflatex` + `bibtex`, pela configuração do projeto (`.latexmkrc`), em TeX Live 2023 no ambiente isolado.
- O original e o modificado compilam sem erros, sem referências indefinidas e sem *overfull*.
- O PDF passou de 69 para 74 páginas.
- As páginas alteradas foram renderizadas e conferidas.
- O `src/main.pdf` versionado **não** foi substituído: deve ser regerado no pipeline do autor (Docker).

**Não alterado, por estar fora do escopo da Etapa 2** (divergências registradas para a etapa de resultados e narrativa):
1. O TCC ainda apresenta o Two-Stage como "referência preliminar", no resumo e nos caps. 6–8, enquanto a Pair-Aware linear é a arquitetura oficial.
2. As seis execuções novas (BioBERTpt-clin nas três estratégias; BERTimbau na Pair-Aware e no Two-Stage) não estão no TCC, que ainda diz "BioBERTpt-clin ainda não foi avaliado" e "duas execuções fora do escopo".
3. A nota de rodapé da §4.3 diz que o corpus "não é redistribuído". Depende da decisão sobre a exposição dos dados no repositório (Etapa 1).
4. A §4.5.2 cita 2.646 menções `Negation`. Esta etapa contou 2.299 **entidades** com esse tipo (1.841 + 230 + 228). A diferença é compatível com rótulos repetidos dentro do mesmo atributo, mas não foi reconferida.
5. Os slides não foram alterados; eles ainda usam o exemplo antigo.

## Situação final da Etapa 2

**Encerrada, dentro do escopo, aguardando avaliação do autor:**
- controles semânticos preservados;
- correções metodológicas documentadas;
- pendência de organização (`test_pacote`) resolvida sem mudar a arquitetura;
- testes direcionados executados e aprovados;
- identidade do código inalterada.

O TCC foi sincronizado com os itens da Etapa 2 (seção anterior). As divergências de narrativa e de resultados listadas ali ficam para a etapa de resultados. A Etapa 3 não foi iniciada.
