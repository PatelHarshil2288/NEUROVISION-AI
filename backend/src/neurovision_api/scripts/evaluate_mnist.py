from __future__ import annotations

import argparse
from pathlib import Path

from neurovision_api.core.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR, get_checkpoint_path
from neurovision_api.services.evaluation import evaluate_mnist_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the existing official MNIST test split.")
    parser.add_argument("--checkpoint", type=Path, default=get_checkpoint_path())
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS_DIR / "analytics-report.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new output path or move the old report.")

    report = evaluate_mnist_checkpoint(args.checkpoint, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(report.model_dump_json(indent=2))
        destination.write("\n")
    print(f"Saved {report.num_samples} test results to {args.output}; accuracy {report.accuracy:.4%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
