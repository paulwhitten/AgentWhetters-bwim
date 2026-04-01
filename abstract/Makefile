
PNG=$(patsubst %.puml,%.png,$(wildcard *.puml))
BIB=$(wildcard *.bib)
DEPS=$(if $(BIB),$(patsubst %.tex,%.bbl,$(wildcard *.tex)),)
PDF=$(patsubst %.tex,%.pdf,$(wildcard *.tex))

all: $(PNG) $(PDF) $(DEPS)
	@echo "BUILDING all DEPS=$(DEPS)"

%.png: %.puml
	@echo "BUILDING png $@ $<"
	plantuml $^

%.bbl: $(BIB)
	@echo "BUILDING bbl $@ $<"
	pdflatex $(patsubst %.bbl,%.tex,$@)
	bibtex $(basename $@)
	pdflatex $(patsubst %.bbl,%.tex,$@)
	rm $(patsubst %.bbl,%.pdf,$@)

%.pdf: %.tex $(DEPS)
	@echo "BUILDING pdf $@ $<"
	pdflatex $<

clean:
	rm -f $(PDF) *.log *.aux *.dvi *.bbl *.fls *.blg *.fdb_latexmk *.spl *.gz *.out
