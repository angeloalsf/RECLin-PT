# Cronologia da designação da Pair-Aware linear como arquitetura oficial

Data: 02/11/2026. Escopo: verificar, com as evidências disponíveis no projeto, a afirmação do TCC de que a designação foi feita com os resultados do DEV já conhecidos.

**Restrições desta análise:**
- nenhum comando Git foi executado e o GitHub não foi consultado (regras 11–13 do projeto), portanto o histórico de commits **não** foi usado;
- nenhum arquivo de resultado foi alterado.

## 1. Afirmação verificada

TCC da Etapa 3 (`metodologia.tex`, §4.6; `discussao.tex`, §7.1 e limitação ii; `conclusao.tex`, §8.2):

> "A designação foi feita com os resultados do DEV disponíveis e, por isso, não constitui evidência confirmatória."

## 2. Evidências, em ordem cronológica

Legenda: **[R]** registro com data explícita no próprio documento ou no arquivo de resultado; **[C]** conteúdo de um documento cuja data é a do registro que o contém; **[L]** limitação (data não verificável).

| Data | Evidência | Fonte | O que mostra |
| --- | --- | --- | --- |
| antes de 10/10/2026 | O projeto legado tinha uma execução da Pair-Aware antiga (espaço restrito, cabeça MLP) com predições no DEV v1 | `codigo/docs/entregas/etapa-6.md`, l. 118 ("Pair-Aware oficial: DEV"); `entregas/auditoria-tcc-completa.md`, tabela de decisões (D10 "após avaliação da versão anterior no DEV v1") [C] | A Pair-Aware já tinha resultado de DEV, na versão anterior e com os dados v1, antes da configuração atual |
| 11/10/2026 | Protocolo registrado. Comparações principais: **Two-Stage** × baseline e × calibrado; Pair-Aware só em comparação exploratória; encoder só BioBERTpt-all; regra do modelo final adiada (C13). Nenhuma "arquitetura oficial" | `codigo/docs/protocolo/fase-biobertpt-seed42.md`, seções 1, 7 e 8 [R] | No registro, nenhuma estratégia é oficial, e o Two-Stage é o foco das comparações |
| 12/10/2026 | Rascunho da emenda E1: "Esta emenda **não** torna o Two-Stage a arquitetura oficial"; a "oficialização do Two-Stage como arquitetura do RECLin-PT" exige registro próprio | `codigo/docs/protocolo/rascunho-emenda-two-stage-desenvolvimento.md`, seções 1 e 9 [R] | Em 12/10 não havia arquitetura oficial; a oficialização era tratada como decisão futura, que dependeria de registro |
| 13/10/2026 | D10: a Pair-Aware passa a ser a linear (todos os pares, sem léxico, sem MLP), "antes de qualquer treino científico" dela. O documento não a designa oficial | `codigo/docs/entregas/d10-pair-aware.md`; protocolo, seção 3 [R] | A **configuração** atual da Pair-Aware antecede o seu treino. O **papel** de arquitetura oficial não é atribuído aqui |
| 13/10/2026, 11:10 UTC | τ do Two-Stage (BioBERTpt-all) congelado: resultado completo do Two-Stage no DEV | `two_stage_limiar_BioBERTpt-all/limiar.json`, `congelado_em` [R] | O resultado do Two-Stage no DEV existia antes do treino da Pair-Aware linear |
| 13/10, 21:37 – 14/10, 00:39 UTC | Treino da Pair-Aware linear (BioBERTpt-all) | `pairaware_biobertpt_seed42/treino.json`, sessões [R] | Primeiro resultado da Pair-Aware linear no DEV |
| 15/10 – 22/10/2026 | Demais execuções no DEV. A última é a Pair-Aware linear com o BioBERTpt-clin, concluída em 22/10, 13:02 UTC | `treino.json` de cada execução; `limiar.json` das composições (16/10 e 19/10) [R] | Em 22/10, os resultados das 9 combinações encoder × estratégia estavam disponíveis |
| 22/10/2026 | Branch `resultados-colab`, commit `ad7c2a2`. Nenhum documento chama a Pair-Aware linear de oficial. O TCC da branch diz que "Os resultados sustentam o uso do Two-Stage como estratégia de referência preliminar" | clone só leitura da branch (Etapa 3): `tcc/src/textuais/discussao.tex` l. 78, `introducao.tex` l. 135, `conclusao.tex` l. 12, 31 e 53; busca por "oficial" em todos os `.md` [R] | Com os resultados do DEV quase todos conhecidos, o registro ainda tratava o Two-Stage como referência |
| sem data verificável (≤ 01/11/2026) | Os slides da apresentação já seguiam a "narrativa Pair-Aware". O slide 21 traz 19 comparações, inclusive da Pair-Aware linear com o BioBERTpt-clin | `entregas/auditoria-tcc-completa.md`, seções 7 e 9.2 [C]; data de criação dos slides não verificada [L] | Se os slides registram a decisão, ela usou o conjunto 3 × 3 do DEV, que só existiu depois de 22/10, 13:02 UTC |
| 01/11/2026 | Auditoria v2: "A auditoria assume a Pair-Aware como narrativa do trabalho, **conforme você decidiu**" | `entregas/auditoria-tcc-completa.md`, cabeçalho [R] | Primeiro registro datado da decisão, que já está tomada nesse momento |
| 01/11/2026 | Etapa 1 da auditoria: "Pair-Aware linear (oficial)" | `entregas/auditoria-repositorio-etapa1.md`, seção 3 [R] | Primeiro registro datado com o termo "oficial" |
| 01–02/11/2026 | README da raiz e `docs/estrategias/README.md`: "arquitetura oficial do RECLin-PT". O TCC da Etapa 2 ainda trazia o Two-Stage como referência preliminar | arquivos atuais; `TCC_etapa2_sincronizado.pdf` (resumo, §7.2, §8) [C] | A designação está nos documentos do repositório apenas na versão atual; quando a frase entrou não é verificável sem o Git [L] |

**Datas de modificação dos arquivos** no computador do autor não servem como evidência. Muitos documentos têm o mesmo horário (por exemplo, 18/10/2026 23:53:02), o que indica cópia ou atualização em lote, não autoria.

## 3. Conclusão

**A afirmação é correta e sustentada pelos registros disponíveis.**
- O primeiro registro datado da designação, e de qualquer decisão pela Pair-Aware como arquitetura do trabalho, é de **01/11/2026**.
- Isso é posterior ao término de todos os treinos no DEV (último em **22/10/2026**).
- Até 22/10, os registros não continham arquitetura oficial e tratavam o Two-Stage como referência preliminar.
- Em 12/10, a oficialização de uma arquitetura era explicitamente uma decisão futura, que dependeria de registro próprio.

**O que as evidências não permitem afirmar:**
1. **A data exata da decisão.** Ela está entre 22/10 (último registro sem a designação) e 01/11/2026 (primeiro registro com ela). Os slides sugerem que ela usou o conjunto completo do DEV, mas a data deles não foi verificada.
2. **Que a decisão foi motivada pelos resultados do DEV.** Os registros provam a ordem (designação depois dos resultados), não a motivação. A redação deve dizer "registrada depois de conhecidos os resultados", não "escolhida por causa deles".

**Uma nuance relacionada, sobre a configuração e não sobre a designação.** A D10 fixou a configuração atual da Pair-Aware antes do seu treino, como o TCC diz. Mas a versão anterior da Pair-Aware já tinha resultado no DEV v1 do legado (evidência na linha 1 da tabela). A frase do TCC "aprovada antes de seu treino" é correta, mas incompleta para fins de transparência. A correção sugerida está na seção 4 e não foi aplicada.

## 4. Redação no TCC

A afirmação atual está correta. Para torná-la verificável e evitar uma leitura de motivação, ela foi **precisada** com as datas, nos trechos diretamente relacionados:

| Trecho | Antes | Depois |
| --- | --- | --- |
| `metodologia.tex`, §4.6 | "A designação foi feita com os resultados do DEV disponíveis e, por isso, não constitui evidência confirmatória." | A designação não consta do protocolo nem dos registros do projeto até 22/10/2026, quando terminou o último treino no DEV; até então, os registros e o texto tratavam o Two-Stage como referência preliminar. Os primeiros registros dela são de 01/11/2026. Ela foi, portanto, registrada com os resultados do DEV já conhecidos e não constitui evidência confirmatória. Menção à proposta de emenda E3 |
| `discussao.tex`, §7.1 | "foi tomada com os resultados do DEV disponíveis" | "foi registrada depois de conhecidos os resultados do DEV (Seção 4.6)"; menção à E3 |
| `discussao.tex`, limitação (ii) | "foi feita com os resultados do DEV disponíveis" | "foi registrada depois de conhecidos os resultados do DEV"; comparações principais pré-registradas ainda são as do Two-Stage, com a E3 proposta e não aprovada |
| `conclusao.tex`, trabalhos futuros | "registrar … uma emenda ao protocolo que …" | "aprovar e registrar, antes de qualquer avaliação no teste, a emenda E3, já redigida, que …" |

**Não aplicada, para decisão do autor (transparência sobre a D10).** Acrescentar ao §4.6, depois de "aprovada antes de seu treino":

> "A configuração anterior da Pair-Aware, restrita aos pares com pista léxica e com cabeça MLP, já tinha sido avaliada no DEV numa versão anterior dos dados, no projeto legado; a escolha da configuração atual, portanto, não foi cega ao DEV."

**Fontes dessa frase:** `docs/entregas/etapa-6.md`, l. 118, e `entregas/auditoria-tcc-completa.md`. O valor do F1 da versão antiga não foi reverificado e não deve entrar no texto.
