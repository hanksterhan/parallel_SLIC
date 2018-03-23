#
# make: builds ps and pdf versions using latex vs. pdflatex
# make spell: runs ispell on .tex file
#
TARGET=proposal

all: 
	pdflatex ${TARGET}.tex
	bibtex ${TARGET} 
	pdflatex ${TARGET}
	pdflatex ${TARGET}
	#dvips -o ${TARGET}.ps ${TARGET} 
	#ps2pdf ${TARGET}.ps

spell:
	ispell ${TARGET}.tex

clean: 
	$(RM) *.dvi *.log *.aux *.bbl  *.blg ${TARGET}.pdf ${TARGET}.ps ${TARGET}.out *.bak

