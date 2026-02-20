.PHONY: all clean verified

all: paper.pdf verified.html

paper.tex: paper.org
	emacs --batch -l org \
		--eval '(setq org-confirm-babel-evaluate nil)' \
		--eval '(setq org-export-with-broken-links t)' \
		--eval '(find-file "paper.org")' \
		--eval '(org-latex-export-to-latex)'

paper.pdf: paper.tex references.bib
	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper

verified.html: paper.org claims.json build_verified.py
	python build_verified.py

clean:
	rm -f paper.tex paper.pdf paper.aux paper.bbl paper.blg paper.log paper.out paper.toc
	rm -f verified.html .test_results.json
