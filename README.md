# factorylib

Useful math models for factory games. Intended more as a playground than a production-ready library. No stable public API exists.

To set up a venv and install locally including dependencies:

```sh
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

To run the linter on all files, which may cause some files to change:

```sh
prek run --all-files
```

To run all tests:

```sh
pytest
```