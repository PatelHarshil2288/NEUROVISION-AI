# NEUROVISION AI Backend

Phase 1 establishes the independent AI backend for the browser-based platform. It contains a CPU-capable MNIST MLP, a reproducible train/validation split, checkpointed training, and versioned FastAPI inference endpoints. It does not import the earlier desktop application.

## Windows setup

Use Python 3.12 or newer:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Train locally on MNIST. The official test set is evaluated only after the best validation checkpoint has been selected:

```powershell
neurovision-train-mnist
```

Choose another supported architecture with `neurovision-train-mnist --architecture cnn`, `rnn`, or `transformer`. All training checkpoints include architecture metadata and can be loaded by the inference service. The detailed activation and weight-visualization endpoints remain MLP-only; prediction and saliency support all four architectures.

The script downloads MNIST under `backend/data/` and writes `best.pt`, `latest.pt`, and the complete test metrics/misclassified-example report (`test-evaluation.json`) under `backend/artifacts/`. These generated files are intentionally ignored by Git.

The Phase 1 reference run selected epoch 8 and reached 97.57% accuracy on the untouched official test set.

Start the API after training:

```powershell
uvicorn neurovision_api.main:app --app-dir .\src --host 127.0.0.1 --port 8000
```

API documentation is available at `http://127.0.0.1:8000/docs`. Without a trained checkpoint, health and model-discovery endpoints remain available, while prediction requests return HTTP 503 rather than producing random model output.

Run backend tests:

```powershell
python -m pytest
```

## API contract

- `GET /api/v1/health` reports service and checkpoint readiness.
- `GET /api/v1/models` reports whether a trained checkpoint is loaded and its parameter count.
- `POST /api/v1/predict` accepts a 28×28 grayscale image with values in `[0, 1]` and returns real logits, probabilities, the predicted class, model version, and measured inference time.
- `POST /api/v1/activations` returns detailed MLP activations; other architectures receive HTTP 501.
- `GET /api/v1/weights` returns actual MLP layer weights and biases; other architectures receive HTTP 501.
- `POST /api/v1/saliency` accepts the same pixels and optional `target_class` (0–9; defaults to the predicted class), returning the gradient of that class's logit with respect to raw pixel intensities. It is local sensitivity, not a definitive explanation.
- `GET /api/v1/analytics/evaluation` returns the SQLite-backed report for the active model or evaluates the approved active checkpoint server-side on the existing official test files. It is restricted to direct local clients.
- `GET /api/v1/analytics/misclassifications/{sample_index}` returns an image only for a recorded error and when the local test-data hashes match the report.
- `GET /api/v1/comparison/checkpoints` lists checkpoints from the approved artifact directories (`artifacts/`, `artifacts/training-lab/`, and `artifacts/models/`).
- `POST /api/v1/comparison/evaluation-reports` accepts only a checkpoint SHA-256 and evaluates that approved local checkpoint against the server's existing official MNIST test files. It is restricted to direct local clients; `GET` on the same path lists persisted reports.
- `POST /api/v1/comparison/reports` compares two saved reports only when their dataset and split fingerprints match, and persists the comparison. `GET` on the same path returns comparison history.
- `GET /api/v1/comparison/summary` and `/api/v1/comparison/available-models` report stored evaluation and comparison state.
- `GET /api/v1/model-registry` lists the active checkpoint, approved checkpoints, and activation history.
- `POST /api/v1/model-registry/activate` accepts an approved checkpoint SHA-256; `POST /api/v1/model-registry/rollback` restores the previous activation. Both operations are local-only and persist across API restarts.

Incoming pixel values use the same MNIST normalization as training (mean `0.1307`, standard deviation `0.3081`). Prediction requests do not include model weights, and the API does not store submitted images.

## Analytics report

After training, run `python -m neurovision_api.scripts.evaluate_mnist` from `backend` to evaluate a checkpoint against the existing official MNIST test split. It writes `artifacts/analytics-report.json`, separately from the original training report. The generated report records the checkpoint and test-file SHA-256 hashes, metrics and errors; no data is downloaded. The command refuses to overwrite an existing output.

The Analytics page asks the backend to verify and evaluate the loaded checkpoint, then lets you select other approved checkpoints for evaluation. Create another architecture checkpoint locally with:

```powershell
neurovision-train-mnist --architecture cnn
```

Reports and comparison history are stored transactionally in `artifacts/comparison.sqlite3` and survive API restarts. Client-submitted JSON evaluation reports are not accepted. To preserve reports from the earlier JSON-backed version, run `python -m neurovision_api.scripts.migrate_comparison_json` once from `backend`. The migration re-evaluates each report's approved local checkpoint against the current official test files, then reconstructs comparison history from those verified results; it leaves the source JSON files untouched and fails explicitly if required checkpoints or data are missing. Evaluation requires already-downloaded MNIST test files and never downloads data from a web request.

The model registry restores its selected checkpoint after restart and fails closed if that checkpoint is no longer available. A SQLite PID lease enforces one backend process per artifacts directory; a second live process cannot start against the same registry. This is local model version control, not a multi-user deployment or authentication system.

## Local Training Laboratory

The Phase 6 lab is **disabled by default** and is not suitable for public or proxied deployment. To opt in on a trusted local machine, first download MNIST via the training script above, then start exactly one backend process directly on loopback:

```powershell
$env:NEUROVISION_TRAINING_ENABLED = "1"
uvicorn neurovision_api.main:app --app-dir .\src --host 127.0.0.1 --port 8000
```

Open the frontend's `/training` page. The lab permits one CPU job at a time, at most three jobs per checkpoint directory, with one to three epochs per job. It uses 2,048 training and 512 validation examples from the seeded, disjoint training/validation split (never the official test set), batch size 128, and Adam at learning rate 0.001. A SQLite process lease prevents a second live backend worker from starting training against the same checkpoint directory; the quota persists across restarts. Job details remain in memory and are not resumable; no cancellation is provided. Batch gradients, the selected parameter update and completed-epoch metrics are real values from the experiment.

The lab requires existing local MNIST files and never downloads them in response to a web request. Its `artifacts/training-lab/best.pt` and `latest.pt` are separate from the inference checkpoint and are overwritten by the next lab experiment. Starting a job does not change the loaded prediction model. Checkpoint metadata records the subset sizes.

`GET /api/v1/training` reports local job status and `POST /api/v1/training` accepts only `{"epochs": 1}` through `{"epochs": 3}`. Both endpoints require direct loopback clients and a local host, reject forwarded requests, and check browser origins. These checks and per-process limits are **not** authentication or public quotas: do not expose the enabled service to the internet, run it behind a reverse proxy, or run multiple worker processes. Public training requires separate authentication, shared durable quotas and resource-isolated workers before it can be enabled.
