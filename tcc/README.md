# TCC — RECLin-PT

Diretório do Trabalho de Conclusão de Curso em LaTeX, baseado no template oficial do **IFES (iftex.cls)** sobre **abnTeX2**.

Toda compilação roda dentro de um container Docker com `texlive/texlive:latest-full`. **Nada é instalado no Windows.**

## Pré-requisitos

Só uma coisa: **Docker Desktop** (Windows/Mac) ou Docker Engine (Linux) com `docker compose`. Nada mais — sem TeX Live, sem MiKTeX, sem `make`.

Verifique:

```bash
docker --version
docker compose version
```

## Comandos principais

Sempre rodar a partir da pasta `tcc/`. Os quatro serviços abaixo estão definidos em `docker-compose.yml`.

| Comando | O que faz |
|---|---|
| `docker compose build` | (Primeira vez apenas) Constrói a imagem TeX Live — baixa ~5 GB. |
| `docker compose run --rm build` | Compila `src/main.tex` uma vez. PDF em `build/main.pdf`. |
| `docker compose run --rm watch` | Recompila a cada `save` nos `.tex`. Ctrl+C para sair. |
| `docker compose run --rm clean` | Remove arquivos auxiliares e PDF. |
| `docker compose run --rm shell` | Abre bash dentro do container (debug, `tlmgr search`, etc.). |

> **Dica Windows/PowerShell:** se digitar muito `docker compose run --rm watch` te incomoda, defina aliases no seu `$PROFILE`:
> ```powershell
> function tcc-build { docker compose run --rm build }
> function tcc-watch { docker compose run --rm watch }
> function tcc-clean { docker compose run --rm clean }
> function tcc-shell { docker compose run --rm shell }
> ```
> Mas é totalmente opcional — o repositório não depende disso.

## Estrutura

```
tcc/
├── Dockerfile               FROM texlive/texlive:latest-full
├── docker-compose.yml       4 services: build / watch / clean / shell
├── .latexmkrc               saída em ../build
├── .gitignore               ignora build/ e aux files
├── .vscode/settings.json    LaTeX Workshop → Docker
├── OUTLINE.md               mapa dos capítulos
├── src/                     FONTE LaTeX
│   ├── main.tex             ← raiz da compilação
│   ├── macros.tex           ← dados do autor/título/orientador (IFES)
│   ├── iftex.cls            ← classe do IFES (NÃO EDITAR)
│   ├── easyReview.sty       ← NÃO EDITAR
│   ├── bibliografia.bib     ← referências (bibtex tradicional)
│   ├── pre_textuais/        ← dedicatória, resumo, siglas, símbolos…
│   ├── textuais/            ← capítulos
│   ├── imagens/             ← PNG/JPG + figuras TikZ em imagens/figuras/
│   ├── tabelas/             ← .tex GERADOS pelo pipeline Python
│   └── apendices/           ← apêndices autorais (SHA256SUMS, léxico…)
└── build/                   ← gerado, NÃO versionado
    └── main.pdf
```

## Convenções

- **Template IFES é intocável:** `iftex.cls`, `easyReview.sty` e a estrutura de `main.tex` permanecem como o template oficial entrega. Se for preciso ajustar comportamento, prefere-se sobrescrever em `macros.tex` ou criar novo `.sty` ao lado.
- **Bibliografia:** usar `bibtex` (estilo `abntex2-alf`), não biber. Adicionar entradas em `src/bibliografia.bib`.
- **Tabelas:** todas as tabelas com dados do experimento são geradas pelos scripts em `../scripts/` e gravadas em `src/tabelas/*.tex`. Os capítulos incluem via `\input{tabelas/nome}`. Nunca editar à mão.
- **Imagens:** PNG/JPG em `src/imagens/`. Figuras compostas (TikZ, subfigs) em `src/imagens/figuras/`. O `\graphicspath{}` em `main.tex` cobre `imagens/`.
- **Apêndices:** material autoral em `src/apendices/`. Incluir via `\apendice` + `\input{apendices/...}` ao final de `main.tex`.

## Fluxo recomendado

1. Abrir VSCode na raiz do RECLin-PT.
2. Em um terminal: `cd tcc && docker compose run --rm watch`.
3. Editar `.tex` em `src/textuais/` — o PDF em `build/main.pdf` recarrega sozinho.
4. Quando o pipeline Python gera tabelas novas, elas caem em `src/tabelas/` e o `watch` recompila.
5. Antes de enviar ao orientador: `git tag tcc-vYYYYMMDD-orientador && git push --tags` e anexar `build/main.pdf` no email.

## Como o Claude trabalha aqui

Sempre que abro o TCC, leio:

1. `tcc/OUTLINE.md` — saber onde mexer.
2. O capítulo específico em `src/textuais/`.
3. `src/macros.tex` se a edição envolve dados institucionais.

Nunca mexo em `iftex.cls`, `easyReview.sty` ou no esqueleto de `main.tex` sem pedido explícito.

## Origem do template

Template `iftex` por Humberto da Silva Neto — https://github.com/hsneto/iftex (LPPL v1.3c). Baseado em [abnTeX2](https://www.abntex.net.br/).
