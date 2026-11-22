# Contributing

## Setup

This repository contains separate web frontend and Python backend projects. Follow [backend setup](backend/README.md) and [frontend setup](frontend/README.md) for their respective dependencies.

## Validation

Run the relevant checks before submitting changes.

From `backend`:

```powershell
python -m pytest
```

From `frontend`:

```powershell
npm run lint
npm run typecheck
npm test
npm run build
```

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Do not commit generated backend data/checkpoints, frontend dependencies/build output, caches, or local environment files.
- Use the `main` branch and describe the result, validation, and any platform limitations.
