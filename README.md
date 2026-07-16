# factorylib

Useful math models for factory games. Intended more as a playground than a production-ready library. No stable public API exists.

## Setup

Recommended: [uv](https://docs.astral.sh/uv/) manages the venv and dependencies for you.

```sh
uv sync --extra dev
```

Without uv, a plain venv works too:

```sh
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

To run the linter on all files, which may cause some files to change:

```sh
uv run prek run --all-files
```

To run all tests:

```sh
uv run pytest
```

(drop the `uv run` prefix for either command if you're using the plain venv instead)

## Optional: Graphviz diagrams

`factorylib.endfield`'s CLI can render a production plan as a Graphviz diagram (`--diagram PATH`). This needs two separate installs:

1. `uv sync --extra diagram` (or `pip install -e ".[diagram]"`) installs the Python `graphviz` package -- a thin wrapper around the `dot` command, not the renderer itself.
2. The actual Graphviz binaries (providing the `dot` executable) aren't distributed on PyPI at all, so they must be installed separately via your OS package manager:
   - Debian/Ubuntu: `apt-get install graphviz`
   - macOS: `brew install graphviz`
   - Windows: `choco install graphviz`

If `dot` isn't installed, `--diagram PATH` still writes the raw `.dot` source instead of a rendered image -- open it in any online Graphviz viewer, or rerun once `dot` is installed to render it properly.

When running from a source checkout (not a wheel install), the CLI also writes this diagram by default to `output/wuling-diagram.png` (gitignored) unless `--diagram PATH` or `--no-diagram` says otherwise.
