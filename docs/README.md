# OOPAO Sphinx Documentation

This folder contains the Sphinx source for the OOPAO documentation.

## Folder layout

```
docs/
├── Makefile          # Linux / macOS build
├── make.bat          # Windows build
├── source/
│   ├── conf.py       # Sphinx configuration
│   ├── index.rst     # Top-level toctree
│   ├── telescope.rst
│   ├── source.rst
│   ├── atmosphere.rst
│   ├── ...           # one .rst per module
│   ├── calibration/
│   ├── closed_loop/
│   ├── tools/
│   └── mis_reg_algo/
└── build/            # generated output (git-ignored)
```

## Quick start

### 1. Install dependencies

```bash
pip install sphinx furo sphinx-autobuild
```

> **Theme**: The config uses [Furo](https://pradyunsg.me/furo/).  
> To use the built-in theme instead, change `html_theme = 'furo'` to  
> `html_theme = 'alabaster'` in `source/conf.py` (no extra install needed).

### 2. Build HTML

```bash
# Linux / macOS
cd docs/
make html

# Windows
cd docs\
make.bat html
```

Output is written to `docs/build/html/`. Open `build/html/index.html` in any browser.

### 3. Live preview (auto-rebuild on save)

```bash
make livehtml
# Then open http://127.0.0.1:8000 in your browser
```

### 4. Build PDF (requires a LaTeX distribution)

```bash
make latexpdf
# Output: docs/build/latex/OOPAO.pdf
```

## Updating the docs

- **Edit existing pages**: modify the relevant `.rst` file in `source/`.
- **Add a new module**: create a new `.rst` file and add it to the appropriate `toctree` in `index.rst` or a sub-index.
- **Pull docstrings from code**: the `autodoc` extension is enabled. You can use `.. autoclass::` or `.. autofunction::` directives in any `.rst` file to pull docstrings directly from the source.

Example:

```rst
.. autoclass:: OOPAO.Telescope.Telescope
   :members:
   :undoc-members:
```

## Hosting on Read the Docs

1. Add a `.readthedocs.yaml` at the repo root (see [RTD docs](https://docs.readthedocs.io/en/stable/config-file/v2.html)).
2. Point it at `docs/` as the documentation directory.
3. Specify `sphinx` as the builder and list `furo` under `python.install`.
