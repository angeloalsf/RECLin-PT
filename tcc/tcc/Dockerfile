# Imagem oficial TeX Live "full" — inclui abnTeX2, latexmk, biber, bibtex,
# fontes brasileiras e todos os pacotes que o template do IFES (iftex.cls)
# requer (mathptmx, helvet, backref, subfig, pdfpages, microtype, etc.).
#
# Por que pinar a imagem: garante que a compilação do TCC em 2027/2028
# produza o MESMO PDF que produz hoje. Se quiser atualizar, basta trocar
# o digest abaixo.
FROM texlive/texlive:latest-full

# Workdir dentro do container — é onde o docker-compose monta tcc/src/.
WORKDIR /workdir/src

# Locale para acentuação correta em mensagens do latexmk.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
