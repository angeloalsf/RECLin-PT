# Configuração do latexmk para o TCC do RECLin-PT.
# Lido automaticamente quando latexmk é invocado a partir de tcc/src/
# (latexmk procura .latexmkrc na CWD e nas pastas acima).

# Saída em ../build/  → mantém tcc/src/ limpo no git status.
$out_dir = '../build';
$aux_dir = '../build';

# Pipeline ABNT (template IFES usa abntex2cite + estilo abntex2-alf,
# que é bibtex tradicional — NÃO biber).
$pdf_mode = 1;
$bibtex_use = 2;            # sempre rodar bibtex
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

# Extensões geradas pelo template que precisam ser limpas no `latexmk -C`.
$clean_ext = 'bbl run.xml synctex.gz acn acr alg glg glo gls ist lol brf';
