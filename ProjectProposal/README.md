This directory contains a document that you should use for
writing your project proposal and annotated bibliography.

proposal.tex: latex source file, annotated with comments and can be built
           in either one or two column mode (see comment at top of file)

proposal.bib: contains bibtex entries.  entries from here are cited in 
              proposal.tex and all cited entries will be generated into a 
              References listing at the end of compiled paper.[pdf,ps,dvi].  
              see the Reference section at the bottom of proposal.tex for how 
              this is added. 

make:  build .pdf  (calls pdflatex and bibtex)
make clean:  clean up all built files

To view: use evince or acroread:
acroread proposal.pdf

This is based on the paper.tex example from:
  /home/newhall/public/latex_examples/paper/
  this example has more latex examples of figures, tables, math mode
  etc.


More information
================
  * my links for document and figure editing tools:
  http://www.cs.swarthmore.edu/~newhall/unixlinks.html#doc
  * my research and writing links and resources:
  http://www.cs.swarthmore.edu/~newhall/unixlinks.html#research
  * latex and bibtex resources:
    * http://www.latex-project.org/
    * http://www.bibtex.org          
    * http://www.cs.cornell.edu/Info/Misc/LaTeX-Tutorial/LaTeX-Home.html
