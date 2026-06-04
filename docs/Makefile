# Minimal Sphinx Makefile
# Usage:
#   make html       -> build HTML docs  (output: build/html/)
#   make latexpdf   -> build PDF        (requires LaTeX)
#   make clean      -> remove build dir
#   make livehtml   -> auto-rebuild on file changes (requires sphinx-autobuild)

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html livehtml latexpdf man Makefile

# Default target
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm -rf $(BUILDDIR)
	@echo "Build directory removed."

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Build finished. Open build/html/index.html in your browser."

livehtml:
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

latexpdf:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX files through pdflatex..."
	@$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo "PDF built at build/latex/OOPAO.pdf"

man:
	@$(SPHINXBUILD) -b man "$(SOURCEDIR)" "$(BUILDDIR)/man" $(SPHINXOPTS) $(O)

# Catch-all: pass other targets straight to sphinx-build
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
