# NEUROVISION AI

NEUROVISION AI is being rebuilt as a browser-based neural-network learning and visualization platform. The web implementation is a greenfield project; it does not depend on the earlier desktop application's model, dataset cache, or checkpoints.

[![Backend CI](https://github.com/PatelHarshil2288/NEUROVISION-AI/actions/workflows/backend.yml/badge.svg?branch=main)](https://github.com/PatelHarshil2288/NEUROVISION-AI/actions/workflows/backend.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](backend/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Implementation status

- **Phase 1 — AI/backend foundation:** complete. The independent Python backend provides MNIST data preparation, an MLP, train/validation-based checkpoint selection, evaluation, and versioned FastAPI inference/activation endpoints.
- **Phase 2 — Digit Studio:** complete. The Next.js frontend supports drawing and image upload, MNIST preprocessing, real API predictions, top-five probabilities, and session history.
- **Phase 3 — 3D Neural Observatory:** complete. The WebGL view renders all 986 neurons and a deterministic sample of connections.
- **Phase 4 — Live Signal Visualization:** complete. The Observatory connects the current Digit Studio input to real model activations and trained weights, with sampled contribution details and staged signal playback.
- **Phase 5 — Neuron Inspector:** complete. Select individual neurons in 3D or with accessible controls to inspect activations, biases, all incoming weights and contributions, and the model's computations.
- **Phase 6 — Training Laboratory:** complete for bounded local experiments. Observe real training, validation, gradients, parameter updates, and lab checkpoints; it is disabled by default and is not public training infrastructure.
- **Phase 7 — Activation and Weight Explorer:** complete. Explore activation functions and derivatives in a hypothetical sandbox, and inspect distributions and connections from the loaded inference checkpoint separately.
- **Phase 8 — Model Analytics:** complete. Explore checkpoint-matched MNIST test metrics, a confusion matrix, real misclassifications, prediction probabilities, and local input sensitivity.
- **Phase 9 — Verified evaluation and local hardening:** complete. Evaluation provenance, SQLite persistence, alternative-architecture training, local-only write paths, and single-worker safeguards are in place.
- **Phase 10 — Model versioning and rollback:** complete. Activate approved checkpoints, inspect activation history, and roll back to the previous model; active selection persists locally across API restarts. See [PHASE_10_GUIDE.md](PHASE_10_GUIDE.md).
- Later phases will follow the approved web-platform plan; no public deployment or external service has been configured.

The reference training run selected epoch 8 by validation accuracy and achieved **97.57% accuracy** on the untouched 10,000-image MNIST test set (seed 42). This is a measured result for that run, not a guarantee for every machine or future configuration.

## Backend quick start

From the `backend` directory on Windows with Python 3.12 or newer:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
neurovision-train-mnist
uvicorn neurovision_api.main:app --app-dir .\src --host 127.0.0.1 --port 8000
```

The training command downloads a fresh MNIST copy into `backend/data/`, uses a seeded 55,000/5,000 train/validation split, saves best-validation and latest checkpoints under `backend/artifacts/`, then evaluates the selected checkpoint on the official test set. Full metrics and misclassified examples are saved to `backend/artifacts/test-evaluation.json`. Generated data and checkpoints are excluded from Git.

To enable the `/analytics` test-results view, generate a separate checkpoint-identified report using the existing checkpoint and test data:

```powershell
python -m neurovision_api.scripts.evaluate_mnist
```

The command refuses to overwrite an existing report. See the backend README for regenerating it after a checkpoint change.

API docs: `http://127.0.0.1:8000/docs`

## Frontend quick start

Install Node.js 22.11 or newer separately from Python. In a second PowerShell window:

```powershell
Set-Location .\frontend
Copy-Item .env.example .env.local
npm ci
npm run dev
```

Open `http://localhost:3000`. Start the Python API in another window using the backend instructions above; the frontend reports when the API or trained checkpoint is unavailable instead of inventing predictions.

To use the Training Lab locally, set `$env:NEUROVISION_TRAINING_ENABLED = "1"` in the backend shell before starting the API, then open `/training`. Run a single direct loopback server only; do not enable this feature behind a proxy or on a public server. It requires the already-downloaded MNIST data and saves separate, replaceable lab checkpoints without changing the prediction model. See [backend/README.md](backend/README.md) for limits and security restrictions.

Run frontend quality checks from `frontend`:

```powershell
npm run lint
npm run typecheck
npm test
npm run build
```

Run backend tests from `backend`:

```powershell
python -m pytest
```

See [backend/README.md](backend/README.md) for the API contract and configuration details. The former desktop application, checkpoint cache, and packaging workflow have been removed from the current project tree.

## Project resources

- [License](LICENSE)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
