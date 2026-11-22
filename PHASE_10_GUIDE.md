# Phase 10 — Model Versioning and Rollback

Phase 10 adds a local model registry on top of the approved checkpoint catalog. It allows a user to switch the active inference model, review recent activations, and roll back without replacing checkpoint files.

## Registry API

All registry endpoints require direct loopback access, a local `Host`/`Origin`, and no forwarded-proxy headers.

```http
GET  /api/v1/model-registry
POST /api/v1/model-registry/activate
POST /api/v1/model-registry/rollback
```

Activation accepts only a checkpoint hash from the local catalog:

```json
{
  "checkpoint_hash": "<64-character SHA-256>"
}
```

The registry returns the active checkpoint, available approved checkpoints, and recent activation history. Activation loads and validates the checkpoint before switching the in-memory inference service. Rollback activates the most recent prior distinct checkpoint. Prediction and saliency calls immediately use the newly active model; model identifiers include the selected architecture.

## Persistence and startup

The active checkpoint pointer and activation history share `backend/artifacts/comparison.sqlite3` with evaluation reports and comparison records. The API restores the selected checkpoint on restart and fails closed if that approved artifact is missing or cannot be loaded. A configured `NEUROVISION_MNIST_CHECKPOINT` explicitly overrides the persisted selection.

A SQLite PID lease allows only one live backend process per artifacts directory. A second worker fails startup rather than serving a different in-memory model than the registry's active checkpoint. This feature is local model version control, not authentication or public deployment infrastructure.

## Checkpoint scope

Only valid `.pt` checkpoints under `artifacts/`, `artifacts/training-lab/`, and `artifacts/models/` can be activated. The registry does not modify or delete checkpoint files. CNN, RNN, Transformer, and MLP checkpoints can be activated; detailed activation and weight visualization remains MLP-only.

See [backend/README.md](backend/README.md) for setup, training, evaluation, migration, and deployment restrictions.
