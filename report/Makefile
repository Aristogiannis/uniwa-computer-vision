.PHONY: report clean watch

REPORT  = report
TEXFILE = $(REPORT).tex
PDFFILE = $(REPORT).pdf

# Build the report PDF via latexmk (handles biblatex + biber automatically).
# Requires: a TeX Live distribution (MacTeX on macOS, TeX Live on Linux).
report: $(PDFFILE)

$(PDFFILE): $(TEXFILE) references.bib $(wildcard figures/*.png)
	latexmk -pdf -bibtex -file-line-error -halt-on-error $(TEXFILE)

# Continuous build — re-runs on save.
watch: $(TEXFILE)
	latexmk -pdf -bibtex -pvc -file-line-error $(TEXFILE)

clean:
	latexmk -C $(TEXFILE)
	rm -f $(REPORT).bbl $(REPORT).run.xml $(REPORT).bcf
