QUARTO_PYTHON := /Users/skojaku-admin/miniforge3/envs/advnetsci/bin/python

.PHONY: render preview clean

render:
	QUARTO_PYTHON=$(QUARTO_PYTHON) quarto render slide08.qmd

preview:
	QUARTO_PYTHON=$(QUARTO_PYTHON) quarto preview slide08.qmd

clean:
	rm -rf slide08_files slide08.html
