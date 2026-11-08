# Adjudicação dos falsos positivos de `negation_of` — Pair-Aware linear, DEV

Análise **secundária e exploratória**. Prepara os falsos positivos (FP) de
`negation_of` da execução oficial `pairaware_biobertpt_seed42` no DEV para
adjudicação humana cega e define como reconciliar as decisões depois. Não altera
dados, rótulos, partições, MANIFEST, protocolo, execuções nem métricas oficiais.
Não treina, não faz inferência e não usa o TEST.

Código: `codigo/reclin/analises/adjudicacao_negation_of.py` (fora da identidade
do código, como todo o `reclin/analises/`).

## 1. Fonte dos resultados

| Item | Valor |
| --- | --- |
| Execução | `pairaware_biobertpt_seed42` (estratégia `pair_aware`, cabeça linear v2, sem léxico) |
| Origem | branch `resultados-colab` (ponta `995c57dd119e…`), pasta `codigo/resultados/execucoes/pairaware_biobertpt_seed42/`, última alteração `d1d809f597641f…` (14/10/2026). Lida por exportação somente leitura, autorizada em 20/10/2026, sem checkout nem alteração de branch |
| Revisão do código no treino | `cb88c52e57ce…` (branch `main`); identidade `ecbb98cae7bc…` (69 arquivos) |
| Seed / encoder / pesos | 42 / `pucpr/biobertpt-all` @ `36b11cfdfa49…` |
| Melhor época | 3 (macro-F1 DEV 0,7406) |
| Dados | `v2-texto-fonte`; DEV `19feb024ca9d…`; MANIFEST `18264860a3d8…` |
| Conjunto do DEV | 19.064 candidatos, `max_gap` 25, candidatos `d3401d56b109…`, `y_true` `a6012dcff283…` |

SHA-256 dos arquivos da execução (também em `restrito/rastreabilidade.json`):

| Arquivo | SHA-256 |
| --- | --- |
| `config.json` | `4d5d1ba0158cf798ff6c6777d6e28ec9895d81405392d62c1ec306adf9a2e409` |
| `experimento.json` | `0d0ff1c0188ebfd6bc0ce5531fabd940aae7a7f3a5e5e35fe36ef24cd382d320` |
| `metricas.json` | `c1e1ff06ad944ab5bd8f1ee51e6d0b1488f3b9e03db6edce40db79a6f47c0679` |
| `predicoes_dev.json` | `017cee17142390e68295eb6b985e198d6c9359ffbd890ffcd43c8bb602ea4e44` |
| `treino.json` | `ad2753d8c9e3020b1aee13d9534b741b7b8fa6f44a129bb68af666d633a8ab08` |

Conferências feitas antes de extrair os casos: o inventário (`execucao.inventario.inspecionar`)
não aponta problemas; a partição DEV confere com o MANIFEST; o `y_true` do sidecar é
o dos 19.064 candidatos recalculados da partição; e as contagens recalculadas são
as do `metricas.json`.

**Identidade do código atual × a do treino.** Hoje a identidade é `0526d9123ecc…`.
Diferem `reclin/config.py`, `reclin/estrategias/__init__.py`,
`reclin/estrategias/pair_aware_mlp/*`, `scripts/avaliar_test.py` e `scripts/treinar.py`.
Nenhum deles participa da enumeração dos candidatos (`reclin/tarefa.py` e
`reclin/particoes.py` são os mesmos), e o conjunto recalculado confere pelos hashes.

## 2. Contagem dos falsos positivos

Recalculada a partir de `predicoes_dev.json` contra os rótulos do DEV:

| | n |
| --- | --- |
| VP (`negation_of` anotado e predito) | 141 |
| **FP (predito `negation_of`, anotado outro)** | **49** (44 `no_relation`, 5 `associated_with`) |
| FN | 9 |
| Precisão oficial de `negation_of` | 141 / 190 = 0,7421 |

Os 49 FP estão em 26 documentos (14 com um FP; 12 com 2 a 5). O número confere
com o `metricas.json` e com a análise anterior: não há divergência.

## 3. Artefatos

Em `codigo/resultados/analises/adjudicacao_negation_of/pairaware_biobertpt_seed42/`:

| Arquivo | Para quem | Conteúdo |
| --- | --- | --- |
| `cega/planilha_adjudicacao.xlsx` | adjudicadores | abas Instruções, Casos (107 linhas), Notas (texto integral de 53 notas) e Categorias |
| `restrito/rastreabilidade.json` | só a análise | proveniência, contagens, métricas oficiais, amostragem e, por caso: documento, ids e offsets de E1/E2, índice no conjunto do DEV, grupo, rótulo anotado, predito, probabilidades, SHA-256 do caso cego |
| `restrito/rastreabilidade.csv` | só a análise | o mesmo, uma linha por caso (UTF-8 com BOM) |
| `restrito/decisoes_congeladas.json` | gerado por `congelar` | decisões humanas conferidas (ainda não existe) |
| `restrito/reconciliacao.{json,md}` | gerado por `reconciliar` | métricas secundárias (ainda não existe) |

As duas pastas existem para que a cega possa ser entregue sozinha. **Os arquivos
contêm texto do SemClinBr**, e `codigo/resultados/` não está no `.gitignore`.
Decida antes de versionar (ver §8).

## 4. Como a cegueira é garantida

1. **Nada do sistema na planilha.** Ela não contém a relação predita, as
   probabilidades, o rótulo anotado, a condição de FP, as relações anotadas da nota
   nem classificações anteriores (a planilha não usa `analises/falsos_positivos.py`).
2. **Controles para ocultar também o rótulo anotado.** Todo FP tem, por definição,
   um rótulo anotado diferente de `negation_of`. Por isso os 49 FP aparecem misturados
   a **49 VP sorteados** (semente 42, dentre 141) e aos **9 FN**. Nada na planilha
   distingue esses grupos. Como cerca de metade dos casos tem `negation_of` anotado,
   saber que o caso está na planilha não diz nada sobre a anotação.
3. **Códigos neutros.** Documentos `D01…D53` em ordem sorteada; casos `C001…C107`
   pela posição no texto dentro de cada documento. Ids de documento e de entidade
   do corpus ficam só na rastreabilidade, para que não se possa consultar a anotação
   do XML pelo id.
4. **Categorias em formulação neutra.** Elas descrevem o par, sem dizer se ele foi
   anotado ou predito (§5). A leitura como categorias de FP é feita só na reconciliação.
5. **Conferência automática na geração** (`verificar_planilha_cega`), que falha se:
   houver abas ocultas, comentários ou nomes definidos; algum rótulo da tarefa
   aparecer em qualquer célula; algum termo revelador (previsão, probabilidade,
   falso/verdadeiro, positivo/negativo, controle, modelo etc.) aparecer fora do texto
   clínico ou nas propriedades do arquivo; algum código coincidir com id do corpus;
   ou o texto relido divergir do original.

## 5. Ficha de adjudicação

Colunas A–K (somente leitura): caso, documento, E1 (texto, tipo), E2 (texto, tipo),
ordem no texto, contexto anterior (≤ 250 caracteres), trecho com as entidades marcadas
pelos offsets (`⟦E1⟧…⟦/E1⟧`, `⟦E2⟧…⟦/E2⟧`), contexto posterior (≤ 250) e observação
sobre o texto (vazia em todos os 107 casos: todas as entidades estão alinhadas aos
offsets). O texto é o da partição, sem normalização; os contextos não são cortados
por frase. Casos da mesma nota têm o mesmo `D..`, e a aba Notas lista os casos de
cada nota com o texto completo.

Colunas L–Q (a preencher): adjudicador (obrigatório e único na cópia), data
(opcional), decisão principal (obrigatória, com lista de validação), problemas
secundários (códigos 1–4 separados por `;`, diferentes da principal), justificativa
e observações.

| Código na planilha | Formulação neutra | Leitura quando o caso é FP |
| --- | --- | --- |
| 1 | Negação válida: E1 é pista e E2 é o alvo negado | Negação clinicamente correta, mas ausente na anotação |
| 2 | Há negação expressa por E1, mas o alvo é outra entidade | Alvo incorreto |
| 3 | E1 é pista, mas E2 está fora da sua abrangência (ou ela foi estendida ou restringida indevidamente) | Escopo incorreto |
| 4 | Não há evidência suficiente de que E1 negue E2 | Relação incorreta ou ausência de negação |
| `inconclusivo` | Contexto insuficiente ou ambiguidade não resolvida | — |
| `revisao_necessaria` | Exige discussão adicional (segunda revisão) | — |

Os códigos 1–4 contam como conclusivos. Nenhuma categoria é preenchida
automaticamente.

## 6. Procedimento

1. Entregue a cada adjudicador **só** a sua cópia de `cega/planilha_adjudicacao.xlsx`,
   por exemplo `adjudicacao_A.xlsx` e `adjudicacao_B.xlsx`. A pasta `restrito/` não
   deve ser entregue.
2. Cada adjudicador decide sozinho, só pelo texto, sem consultar o corpus, as
   predições, outras análises ou a outra cópia. Os casos ambíguos recebem
   `inconclusivo` ou `revisao_necessaria`.
3. Com dois adjudicadores, as duas cópias são fechadas antes da discussão. Se houver
   consenso, ele vai numa terceira cópia (adjudicador = `consenso`), e as
   individuais ficam preservadas.
4. Congele as decisões. Planilhas incompletas, com o texto alterado ou com códigos
   inválidos são recusadas, e a saída nunca é sobrescrita:

   ```
   cd codigo
   python -m reclin.analises.adjudicacao_negation_of congelar \
     --rastreabilidade resultados/analises/adjudicacao_negation_of/pairaware_biobertpt_seed42/restrito/rastreabilidade.json \
     --planilha CAMINHO/adjudicacao_A.xlsx [--planilha CAMINHO/adjudicacao_B.xlsx] [--consenso CAMINHO/consenso.xlsx] \
     --saida resultados/analises/adjudicacao_negation_of/pairaware_biobertpt_seed42/restrito/decisoes_congeladas.json
   ```

5. Só então reconcilie:

   ```
   python -m reclin.analises.adjudicacao_negation_of reconciliar \
     --rastreabilidade .../restrito/rastreabilidade.json \
     --congeladas .../restrito/decisoes_congeladas.json \
     --saida .../restrito/reconciliacao.json --markdown .../restrito/reconciliacao.md
   ```

Com um único adjudicador, a reconciliação registra `fonte_das_decisoes =
adjudicador_unico` e um aviso explícito de que o resultado não é consenso nem
validação clínica. Com dois e sem consenso, ela apresenta os resultados de cada um
e a concordância (observada e kappa de Cohen, geral e só nos FP), sem decisão final.

## 7. Métricas da reconciliação

Por conjunto de decisões (cada adjudicador e, se houver, o consenso):

- FP revisados (de 49); contagem e proporção por decisão; inconclusivos; revisão necessária;
  problemas secundários; cruzamento com o rótulo anotado (`no_relation` × `associated_with`).
- **Precisão original** = 141 / 190 (inalterada).
- **Precisão secundária (exploratória)** = (VP + FP com decisão 1) / (VP + FP). Os casos
  inconclusivos, em revisão e com decisões 2 a 4 continuam FP, e os VP não são reavaliados.
- Predições reinterpretadas = FP com decisão 1 (com a lista dos casos).
- Controles: distribuição das decisões nos VP e nos FN e a concordância com a anotação,
  como indicador da calibração do adjudicador. Não altera a precisão.
- Exemplos (até 3 por decisão) com o trecho marcado, **só no JSON restrito**. O Markdown
  não traz texto clínico.

As proporções valem apenas para estes 49 FP e não são extrapoláveis ao corpus nem a
outras execuções. Divergências com a anotação ficam nesta análise secundária. Os
rótulos e as métricas oficiais continuam como estão.

## 8. Reprodução, limitações e decisões pendentes

**Reprodução.** Os arquivos da execução não estão na pasta de trabalho (só na branch
`resultados-colab`). Para gerar de novo, exporte a pasta da execução para um
diretório chamado `pairaware_biobertpt_seed42` e rode:

```
python -m reclin.analises.adjudicacao_negation_of preparar --execucao PASTA_EXPORTADA \
  --saida OUTRA_PASTA [--controles-vp 49] [--semente 42] [--registro-origem "..."]
```

A geração é determinística: duas execuções produziram `rastreabilidade.json` e
`.csv` idênticos byte a byte. O `.xlsx` muda só nos carimbos de data internos; o
conteúdo é conferido pelo `sha256_casos_cegos` (`74f3eda76570…`). Os SHA-256 da §1
confirmam que a exportação é a mesma.

**Limitações.**

- Duas características visíveis (e exigidas na ficha) se distribuem de forma desigual
  entre os grupos. "E2 antes de E1" ocorre em 5 FP, 3 FN e nenhum VP sorteado. E1 sem o
  tipo `Negation` ocorre em 5 FP, 3 FN e nenhum VP. Um adjudicador que conheça as
  convenções do corpus pode inferir algo nesses casos. A mistura com controles reduz o
  viés, mas não o elimina.
- O tipo das entidades vem da anotação de entidades do corpus (`Negation` em E1). Isso
  é anotação de entidade, não de relação, mas pode influenciar a decisão.
- Os contextos têm janela fixa de 250 caracteres. A nota inteira está na aba Notas.
- A cegueira depende de o adjudicador não abrir o corpus nem a pasta `restrito/`. O código
  não tem como impedir isso.
- Com um único adjudicador, o resultado não é consenso.
- Não rodei a suíte de testes nem criei testes novos para o módulo, por instrução.

**Decisões que dependem de aprovação.**

1. Os controles (49 VP sorteados + 9 FN) aumentam a carga de 49 para 107 casos. As
   alternativas são `--controles-vp 0` (só FP + FN, e a anotação fica parcialmente
   inferível) ou todos os 141 VP (199 casos).
2. O versionamento de `codigo/resultados/analises/adjudicacao_negation_of/`, que contém
   texto do SemClinBr. Talvez convenha manter `cega/` e `restrito/` fora do controle de versão.
3. O número de adjudicadores e quem faz o consenso.
