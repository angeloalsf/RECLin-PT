# Entregas

Cada etapa da reconstrução é entregue como um ZIP com o **projeto completo**
até aquele ponto: extraído numa pasta limpa, ele tem tudo o que é preciso para
instalar, executar e testar o que já foi implementado. Não é preciso combinar
pacotes anteriores. As entregas ficam separadas por etapa e versão; uma versão
nova da mesma etapa substitui a anterior.

| Etapa | Versão | Data | Pacote | Situação | Nota |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 09/10/2026 | `RECLin-PT-etapa-1.zip` | Substituída pela versão 2: não trazia `CITATION.cff` e `LICENSE`, então não era autossuficiente | — |
| 1 | 2 | 09/10/2026 | `RECLin-PT-etapa-1-v2.zip` | Atual | [etapa-1.md](etapa-1.md) |

## O que nunca vai no pacote

| Item | Por quê | Onde obter |
| --- | --- | --- |
| Projeto legado | É só referência; o que dele os testes usam está em `codigo/testes/referencia/` | Repositório original, commit `a5f055c` |
| XML do SemClinBr e `dataset.jsonl` (`codigo/dados/brutos/`, `codigo/dados/processados/`) | Licença restrita do corpus | Autores do corpus; ver o README da raiz |
| Ambientes virtuais, caches (`__pycache__`, `.pytest_cache`, `*.egg-info`) | Gerados na instalação | `pip install -e "codigo/[testes]"` |
| Pesos de modelos e checkpoints | Centenas de MB; gerados pelo treino | Treino (etapas seguintes) ou Hugging Face Hub |
| Dependências externas (Python, pytest e, nas etapas de treino, torch e transformers) | Instaladas pelo `pip` | `codigo/requirements.txt` e `codigo/pyproject.toml` |

## Como cada pacote é verificado antes da entrega

1. O ZIP é montado a partir dos arquivos versionados do projeto, e é recusado
   se contiver algum item da tabela acima ou se faltar um arquivo obrigatório.
2. É extraído numa pasta vazia e comparado byte a byte com o projeto.
3. Nessa pasta, em ambientes virtuais novos com cada versão de Python testada:
   instalação, importação de todos os módulos, `--help` de todos os scripts,
   conferência das partições e a suíte completa de testes.

Os resultados de cada entrega, com o que foi e o que não foi executado, estão
na nota da etapa.
