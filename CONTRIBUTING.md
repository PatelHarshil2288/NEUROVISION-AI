# Contributing

## Setup

```bash
python -m pip install -e .
python -m pip install pytest
```

## Validation

Run before submitting changes:

```bash
python -m pytest -q
python -m compileall -q src tests
python -m build
```

For Windows executable validation:

```powershell
python -m PyInstaller neurovision.spec --clean --noconfirm
```

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Do not commit generated `data/`, `build/`, `dist/`, caches, or local checkpoints.
- Use the `main` branch and describe the result, validation, and any platform limitations.
