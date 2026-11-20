# Repositório privado e dados do SemClinBr (Etapa 5, G1)

Estado em 16/11/2026: os repositórios **ainda estão públicos**; o autor informou
que os tornará privados. Nada foi alterado no GitHub nesta etapa.

## Por que é necessário

O SemClinBr é de acesso restrito. O repositório versiona, de propósito, arquivos
com texto das notas clínicas:

- `codigo/dados/particoes/` e `codigo/dados/particoes-v2/` (treino, DEV e TEST);
- arquivos de referência dos testes em `codigo/testes/referencia/`;
- `codigo/resultados/analises/adjudicacao_negation_of/` (planilha e
  rastreabilidade, com trechos do DEV).

Enquanto o repositório for público, esses textos estão sendo redistribuídos. O
mesmo vale para o repositório legado (`RECLin-PT-original`) e para qualquer
repositório de modelo ou de dados no Hugging Face que contenha texto do corpus.

## O que fazer (ações do autor)

1. **Tornar privados** `reclin-pt` e `RECLin-PT-original` (GitHub → Settings →
   General → Danger Zone → Change repository visibility). As duas branches
   (`main` e `resultados-colab`) passam a ser privadas juntas.
2. Conferir se há **forks** públicos (aba Insights → Forks). Tornar o repositório
   privado não apaga forks nem clones já feitos.
3. Conferir a visibilidade dos repositórios no **Hugging Face** usados como
   cópia dos pesos (os pesos não contêm texto do corpus; os cartões e arquivos
   auxiliares, conferir).
4. O `GITHUB_PAT` dos notebooks precisa ter acesso de leitura (e de escrita, para
   publicar resultados) ao repositório privado. Os notebooks já clonam por
   token.
5. Decidir sobre o **histórico**. Com o repositório privado, reescrever o
   histórico não é necessário para cessar a redistribuição. Só faz sentido se o
   repositório for voltar a ser público; nesse caso seria preciso retirar as
   partições e os arquivos com texto de todo o histórico (por exemplo com
   `git filter-repo --invert-paths --path codigo/dados/particoes --path codigo/dados/particoes-v2 ...`),
   o que troca todos os hashes de commit e quebra a identidade de código
   registrada nas execuções. **Não recomendado** enquanto os experimentos
   dependerem desses registros; a alternativa para divulgação é um repositório
   público novo, só com código, sem dados.

## O que foi alinhado nesta etapa

- `README.md` da raiz: a pasta `tcc/` consta como incluída; a seção do corpus
  diz que o repositório contém texto do SemClinBr e não pode ser público.
- `.gitignore`: os comentários sobre as partições e sobre
  `codigo/resultados/analises/` dizem o mesmo. Nenhuma regra de ignorar foi
  alterada.
