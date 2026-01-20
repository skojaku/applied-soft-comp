# Notebooks

This directory contains course notebooks in Quarto markdown format (.qmd files). These files combine narrative text with executable code and can be converted to Jupyter notebooks (.ipynb) for interactive use.

## Converting .qmd to .ipynb

You have two main options for converting Quarto markdown files to Jupyter notebooks.

### Option 1: Using Quarto (Recommended)

Quarto provides a built-in conversion command that preserves all formatting and metadata.

Convert a single file:

```bash
quarto convert 01-toolkit.qmd
```

This creates `01-toolkit.ipynb` in the same directory.

Convert all .qmd files in the notebooks directory:

```bash
cd notebooks
for file in *.qmd; do quarto convert "$file"; done
```

### Option 2: Using Jupytext

If you have jupytext installed, you can use it for bidirectional conversion between .qmd and .ipynb formats.

Install jupytext if needed:

```bash
uv pip install jupytext
```

Convert a single file:

```bash
jupytext --to notebook 01-toolkit.qmd
```

Convert all .qmd files:

```bash
jupytext --to notebook *.qmd
```

The advantage of jupytext is that it supports synchronization between formats, allowing you to keep both .qmd and .ipynb versions in sync as you edit.

## Working with .qmd Files Directly

Many modern editors and IDEs can work with .qmd files directly without conversion:

**VS Code**: Install the Quarto extension to edit and execute .qmd files interactively.

**JupyterLab**: Install the jupytext extension to open .qmd files as notebooks.

**RStudio**: Has native support for Quarto documents.

## Why .qmd Format?

Quarto markdown format offers several advantages for course materials:

**Version control friendly**: Plain text format works well with git, making it easy to track changes and collaborate.

**Reproducible**: All code, output, and narrative are in one file with clear execution order.

**Flexible output**: Can render to HTML, PDF, slides, or notebooks from the same source.

**Educational clarity**: The clear separation between text and code blocks makes the learning structure explicit.

## Need Help?

If you encounter issues with conversion or execution, check that:

1. Your virtual environment is activated
2. All dependencies are installed (`uv pip install -e .` from the project root)
3. Quarto is installed (download from https://quarto.org if needed)
4. You're running commands from the correct directory
