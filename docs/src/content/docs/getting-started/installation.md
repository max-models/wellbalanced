---
title: Installation
description: Create a Python environment and install wellbalanced.
---

Create and activate a Python environment:

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the package and its requirements with pip:

```bash
pip install -e .
```

It needs [plasmacoll](https://github.com/max-models/plasmacoll) (cross sections and
rate coefficients) and SciPy, which pip installs with it.

## Optional extras

| Extra  | Installs                                                |
| ------ | ------------------------------------------------------- |
| `test` | `pytest` and `coverage`                                 |
| `docs` | the notebook runner and griffe for the API reference    |
| `dev`  | formatters and linters, plus the `test` and `docs` extras |

```bash
pip install -e ".[dev]"
```
