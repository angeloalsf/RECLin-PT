# Etapa 4 — Recomendações técnicas e plano de evolução do RECLin-PT (13/11/2026)

Complementa `etapa4-auditoria-integral.md`. Cada recomendação indica a evidência e a
classe: **[agora]** (sem novo experimento, sem GPU, sem TEST), **[experimento]** (novo
treino ou nova inferência, só treino e DEV) ou **[autorização]** (toca o TEST ou muda
decisão registrada).

## 1. O que a evidência atual diz sobre a prioridade (precisão de `negation_of`)

Análise feita nesta etapa sobre as probabilidades do DEV já gravadas (BioBERTpt-all,
semente 42, 150 pares `negation_of`):

| Configuração | Precisão | Recall | F1 |
|---|---|---|---|
| Pair-Aware ponderada, argmax (referência) | 0,742 | 0,940 | 0,829 |
| Pair-Aware ponderada, P(neg) ≥ 0,95 | 0,768 | 0,907 | 0,832 |
| Pair-Aware sem pesos, argmax | 0,794 | 0,873 | 0,832 |
| Pair-Aware sem pesos, P(neg) ≥ 0,80 | 0,845 | 0,800 | 0,822 |
| Two-Stage, τ = 0,95 | 0,798 | 0,867 | 0,831 |

Leitura:

1. **Peso de classe, limiar e Two-Stage percorrem a mesma curva.** Nenhuma opção
   altera o F1 de forma detectável (sem pesos − ponderada: +0,002 [−0,028; +0,031]).
   Ganha-se precisão pagando recall quase um por um.
2. **Há um núcleo de 34 falsos positivos** que aparece nas duas execuções da
   Pair-Aware. Mudar o ponto de operação não os remove.
3. **A referência é ruidosa nessa classe**: 37% das entidades `Negation` não participam
   de nenhuma `negation_of` anotada (TCC, seção dos controles pelo tipo `Negation`), e o cofre registra que 30 dos
   49 falsos positivos são pista sem relação anotada.

Conclusão: o próximo ganho de precisão que se pode demonstrar não vem de treinar
mais, e sim de **medir quanto do erro é da anotação**. Limiares escolhidos no DEV são
otimistas e, com 150 positivos, uma diferença de 0,03 de F1 equivale a poucas
predições.

## 2. Recomendações

### Governança e integridade

| # | Recomendação | Evidência | Classe |
|---|---|---|---|
| G1 | Tornar privados `RECLin-PT` e `RECLin-PT-original`; decidir a limpeza do histórico; alinhar README e `.gitignore` | repositórios públicos com `codigo/dados/` versionado | agora (ação sua) |
| G2 | Trava técnica no `scripts/avaliar_test.py`: exigir um identificador da configuração congelada e recusar sem ele | a trilha registra, mas não impede | agora (código, com sua aprovação) |
| G3 | Registrar o experimento sem pesos no protocolo, no inventário e no cofre; rodar `diagnosticos/auditar_execucoes.py` localmente para o bootstrap por documento | execução existe só na branch de resultados | agora (local, sem GPU) |
| G4 | Confirmar a regra do Ifes sobre declaração de uso de IA generativa | texto sem declaração | agora (ação sua) |

### Código

| # | Recomendação | Evidência | Classe |
|---|---|---|---|
| K1 | Corrigir o teste que falha por colisão de nome (`referencia` em `sensibilidade_semente.py`) | 1 falha real em 601 testes aprovados | agora |
| K2 | Adiar o `import torch` em `estrategias/pair_aware` | 2 módulos de teste não coletam sem torch | agora |
| K3 | Gravar probabilidades com precisão total (ou logits) nas próximas execuções | precisão média de `negation_of` do baseline varia de 0,820 a 0,875 conforme o desempate | agora (código); regravar as atuais exige reinferência |
| K4 | Rodar a suíte completa (torch + dados) e registrar o resultado | não executada nesta etapa nem nas duas anteriores | agora (local) |
| K5 | Declarar `lxml` e `matplotlib`; retirar `scikit-learn`; atualizar o README de raiz | pendências já anotadas no cofre | agora |
| K6 | Dividir `experimento/operacoes.py` e `analises/adjudicacao_negation_of.py` (≈ 49 KB cada) | 28 funções de alta complexidade | agora, baixa prioridade |

### Método e experimentos

| # | Recomendação | Evidência | Classe |
|---|---|---|---|
| M1 | Adjudicação cega dos falsos positivos de `negation_of` (os 34 comuns primeiro) | Seção 1; planilha cega já preparada | agora (trabalho humano, sem GPU) |
| M2 | Mais sementes (5) para Pair-Aware e baseline com o BioBERTpt-all | diferença de +0,027 detectada só em 1 de 3 encoders, com 1 semente | experimento |
| M3 | Ablação do pooler: baseline sem pooler e Pair-Aware com pooler | duas diferenças entre as cabeças; efeito confundido | experimento (2 treinos) |
| M4 | Controles (calibrado e filtro) para BioBERTpt-clin e BERTimbau | hoje só existem para o BioBERTpt-all | agora (local, sem GPU) |
| M5 | Avaliar `associated_with` sem direção, como análise secundária | cerca de um quarto dos falsos positivos tem a direção inversa anotada | agora (local, sem GPU) |
| M6 | Orçamento maior que 3 épocas, com seleção pelo DEV | melhor época foi a última em todas as execuções de 3 rótulos | experimento |
| M7 | Encoder moderno multilíngue (mmBERT-base) como variante | melhor desempenho em NER clínico no SemClinBr segundo Almeida et al. (2026), já citado no TCC | experimento |
| M8 | Critério de limiar que considere as três classes | τ = 0,95 no extremo da grade, com perda de `associated_with` | experimento (só DEV, sem treino) |
| M9 | Congelar a configuração e avaliar uma vez no TEST | quinto objetivo do TCC cumprido em parte | autorização |

## 3. Plano de evolução

**Fase A — Higiene (dias, sem GPU, sem TEST).** G1 a G4, K1 a K5. Saída: repositório
privado e coerente, suíte completa executada, experimento sem pesos registrado.
Critério de saída: suíte verde no seu ambiente e inventário com 17 pastas explicadas.

**Fase B — Entender o erro (1 a 2 semanas, sem GPU).** M1, M4, M5. Saída: taxa de
falsos positivos que são omissão da anotação, com intervalo; controles nos três
encoders. Decisão ao final: se a maior parte dos 34 for omissão, a precisão real já é
maior que a medida e o foco muda para relatar isso; se não for, seguir para a Fase C
com foco em precisão.

**Fase C — Robustez (GPU, só treino e DEV).** M2 e M3 primeiro; M6, M7 e M8 só se a
Fase B indicar espaço. Sugestão de ordem: 5 sementes × {Pair-Aware, baseline} com o
BioBERTpt-all (10 treinos), depois os 2 da ablação. O custo deve ser estimado pelo
tempo registrado em `treino.json` das execuções atuais; não o medi nesta etapa.

**Fase D — Congelamento e TEST (autorização explícita).** Escolher o ponto de operação
(ponderada ou sem pesos) com base nas Fases B e C, registrar a configuração, avaliar
uma única vez e relatar também a análise sem as 3 notas com cópia no treino.

**Fase E — Fechamento do TCC.** Substituir a Seção 6 pelos números finais, preencher
banca e ficha, compilar no Docker, ensaiar a apresentação (15 min).

**Fase F — Depois do TCC.** Fluxo completo NER → relações com medida de propagação de
erro; comparação com modelos generativos em poucas amostras; artigo.

Se o prazo não permitir as Fases C e D antes da entrega, o texto revisado já sustenta
a entrega como estudo no DEV, com o TEST reservado. Essa é uma decisão sua e do
orientador.

## 4. Avaliação do vídeo explicativo

**Recomendação: vale produzir, com escopo curto.** Um vídeo de 2 a 3 minutos que
mostre o caminho de um par pela cadeia resolve o ponto mais difícil de explicar em
slides (como os marcadores e a cabeça leem o par) e serve à defesa e ao repositório.

Roteiro proposto (7 cenas):

1. Uma nota clínica curta aparece; duas entidades acendem.
2. O gerador percorre os pares até 25 caracteres; a maioria vira `no_relation`.
3. O par recebe os marcadores `[E1]` e `[E2]`.
4. O encoder processa a sequência; três vetores são destacados: `[CLS]`, `[E1]`, `[E2]`.
5. Baseline e Pair-Aware lado a lado: um vetor contra três concatenados.
6. A saída: três probabilidades, rótulo `negation_of`.
7. O compromisso precisão × recall, com os quatro pontos da tabela da Seção 1.

Cuidados: usar apenas o trecho já citado no TCC (documento 9091) ou texto sintético,
nunca outro texto do corpus; identificar os números como de DEV; não exibir endereço
de repositório enquanto G1 estiver pendente; não narrar o resultado do TEST antes de
ele existir.

Produção: a animação pode ser feita em código (gera um arquivo de vídeo, sem
ferramenta paga) em uma etapa própria. Não exige novo experimento. Depende de duas
decisões suas: a identidade visual (sugiro a do deck revisado) e se a cena 7 entra
agora ou só depois da Fase D. Posso produzir o primeiro corte na próxima etapa, se
autorizado.
