# Report and slides

Built artifacts for the ICE-8111 final exemption project.

## Files

| File | What |
|---|---|
| `report.md`    | 4–6 page report in Markdown (same content as the .tex) |
| `report.tex`   | LaTeX article version of the same report (12 pt) |
| `slides.tex`   | 10-slide Beamer deck (Metropolis theme) |
| `references.bib` | Bibliography (copied from `docs/references.bib`) |
| `figures/`     | All PNGs included by the documents |
| `Makefile`     | `make` to produce `report.pdf` via latexmk |

## Rendering report.md to PDF

The Markdown file is portable but for submission you'll want a paginated PDF. Three options ranked by friction:

```bash
# 1. Pandoc + a TeX engine — best fidelity
pandoc report.md -o report.pdf \
  --pdf-engine=xelatex --resource-path=. \
  -V geometry:margin=2.4cm -V fontsize=12pt

# 2. Pandoc + WeasyPrint — no TeX required (brew install pandoc weasyprint)
pandoc report.md -o report.pdf --pdf-engine=weasyprint

# 3. VS Code / Cursor "Markdown PDF" extension — one-click export
```

If you submit the Markdown directly, GitHub renders it natively with the figures inlined.

## Building locally

Requires a TeX distribution with `latexmk`, `biber`, and the Metropolis Beamer theme (TeX Live ≥ 2020 or MacTeX).

```bash
brew install --cask mactex-no-gui     # macOS, one-off (~5 GB)
cd report
make                                   # produces report.pdf
pdflatex slides.tex && pdflatex slides.tex   # slides
```

## Building without installing LaTeX

Fastest path: [Overleaf](https://www.overleaf.com). Upload this whole `report/` folder as a new project, set the compiler to **pdfLaTeX**, click **Recompile**. The Metropolis Beamer theme is preinstalled on Overleaf.

## Editing checklist before submission

- `report.tex` line 24: replace `AM:~\texttt{<TODO>}` with your student registration number.
- `slides.tex` line 18: same `AM` placeholder.
- Optional: if you overflow the 6-page limit, tighten `\setlength{\parskip}{0.35em}` to `0.25em` or set narrower `geometry` margins (e.g. `2cm`).
