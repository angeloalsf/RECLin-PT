# Etapa 8.4 — notebook do Two-Stage (Colab)

Data: 12/10/2026. Sem pacote; arquivos na pasta de trabalho. **Nada foi treinado, avaliado, baixado nem publicado.**

## O que foi feito
`notebooks/reclin_two_stage.ipynb`, no padrão do `reclin_baseline.ipynb` (seções 1–6 idênticas, salvo o repositório de backup HF):

| Seção | Conteúdo |
|---|---|
| 1–6 | clone, GPU, Drive, dependências, partições + pesos originais, token HF (`HF_BACKUP_REPO = angeloalsf/reclin-pt-twostage-biobertpt-seed42`) |
| 7 | estágio 1 (`relacao` × `no_relation`, 3 épocas), novo/retomada/nada, com quadro pré-treino |
| 8 | estágio 2 (`negation_of` × `associated_with`, 10 épocas), mesmo comportamento |
| 9 | `compor-dev`: escolhe e congela τ no DEV; exige `CONFIRMO_COMPOR = True` e estágios do protocolo |
| 10 | histórico, curvas por estágio, τ × F1 e métricas do fluxo real (só DEV) |
| 11 | GitHub: **não implementado** (decisão pendente, abaixo) |
| 12 | backup HF dos dois estágios, só com `PUBLICAR_NO_HF = True` |

`reclin/experimento/operacoes.py` (extensão mínima, genérica; o baseline não muda):
- `Especificacao.estagio` (opcional): com ele, o treino chama `two_stage.py treinar --estagio N ...` (sem `--estrategia`).
- `Contexto.estagio1/estagio2` e a operação `OP_COMPOR` (`compor_dev`): confere que os dois estágios estão concluídos e são `two_stage_e1/e2`, recusa composição já existente, roda `two_stage.py compor-dev` e espelha a composição no Drive (SHA-256).
- `metricas_dev`: colunas genéricas para estágios (chaves próprias de cada estágio).

Nenhum algoritmo científico, dado, MANIFEST, protocolo ou `reclin.ipynb` foi alterado.

## Arquivos
Criados: `notebooks/reclin_two_stage.ipynb`, `testes/unidade/test_operacoes_two_stage.py`, esta nota.
Modificados: `reclin/experimento/operacoes.py`, `docs/entregas/README.md`.
Removidos: nenhum.

## Verificações executadas (CPU, sem rede, sem Git)
- `test_operacoes_two_stage.py` (novo): 12 passaram.
- `test_operacoes_baseline.py`: 51 passaram (regressão do baseline).
- `test_experimento.py` + `test_experimento_plano.py`: 51 passaram.
- Notebook: JSON válido, 25 células, ids únicos, sintaxe das 12 células de código, uma só célula com `pip install`.
- **Suíte completa não executada.** Nada foi rodado no Colab.

## Divergências e decisões que dependem do autor
1. **Nomes das pastas:** `plano.NOMES_OFICIAIS` só tem baseline/baseline_calibrado/restrito/pair_aware, e o protocolo não define nomes oficiais do Two-Stage. Os estágios usam pasta `run_id`. A composição usa o nome do script (`twostage_biobertpt_seed42`).
2. **Épocas:** o texto do protocolo cita limite 2; a tabela e o código usam 3 (E1) e 10 (E2). Usei 3 e 10 e sinalizei no notebook (divergência já registrada).
3. **GitHub (seção 11):** os estágios não têm `metricas.json`/`predicoes_dev.json` e a composição não tem `experimento.json`/`treino.json`; o fluxo do baseline não se aplica. Preciso que você defina quais arquivos versionar.
4. **Composição fora da proveniência:** não tem `experimento.json`; ela só é espelhada no Drive, não publicada no HF.
5. Repositório HF novo para o Two-Stage (confirme o nome).
6. E1 segue inativa; D4/D9 pendentes, fora do escopo.

## Limitações
Tudo da integração com o Colab (clone, GPU, treino de verdade, `compor-dev`, espelho, HF) é **inferido pela leitura do código** e depende de teste real. A retomada de estágio usa `treinar_estagio`, que compara o registro inteiro do `config.json`; só será confirmada num treino real interrompido.
