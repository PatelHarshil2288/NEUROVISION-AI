from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from neurovision_api.services.analytics import EvaluationReport
from neurovision_api.services.evaluation import evaluate_mnist_checkpoint
from neurovision_api.services.model_comparison import ModelComparison, ModelComparisonStore


def migrate_legacy_comparison_json(
    store: ModelComparisonStore,
    *,
    data_dir: Path,
    evaluator: Callable[[Path, Path], EvaluationReport] = evaluate_mnist_checkpoint,
) -> tuple[int, int]:
    """Re-evaluate legacy reports before migrating their reports and history to SQLite."""
    artifacts_dir = store.artifacts_dir
    legacy_report_dir = artifacts_dir / "evaluation-reports"
    legacy_comparison_dir = artifacts_dir / "comparisons"
    reports_by_version: dict[str, EvaluationReport] = {}

    report_files = sorted(legacy_report_dir.glob("*.json")) if legacy_report_dir.is_dir() else []
    for report_file in report_files:
        if report_file.stat().st_size > 5_000_000:
            raise ValueError(f"Legacy evaluation report is too large: {report_file.name}.")
        legacy_report = EvaluationReport.model_validate_json(
            report_file.read_text(encoding="utf-8")
        )
        if report_file.stem != legacy_report.checkpoint_sha256:
            raise ValueError(f"Legacy report filename does not match its checkpoint hash: {report_file.name}.")
        checkpoint_path = store.checkpoint_path(legacy_report.checkpoint_sha256)
        verified_report = evaluator(checkpoint_path, data_dir)
        if verified_report.checkpoint_sha256 != legacy_report.checkpoint_sha256:
            raise ValueError(f"Re-evaluation produced a different checkpoint hash: {report_file.name}.")
        if verified_report.model_version != legacy_report.model_version:
            raise ValueError(f"Re-evaluation produced a different model version: {report_file.name}.")
        store.save_report(verified_report)
        if verified_report.model_version in reports_by_version:
            raise ValueError(f"Legacy reports contain an ambiguous model version: {verified_report.model_version}.")
        reports_by_version[verified_report.model_version] = verified_report

    comparison_files = (
        sorted(legacy_comparison_dir.glob("*.json"))
        if legacy_comparison_dir.is_dir()
        else []
    )
    existing_comparisons = {
        comparison.comparison_id: comparison for comparison in store.list_comparisons()
    }
    migrated_comparisons = 0
    for comparison_file in comparison_files:
        if comparison_file.stat().st_size > 5_000_000:
            raise ValueError(f"Legacy comparison is too large: {comparison_file.name}.")
        legacy_comparison = ModelComparison.model_validate(
            json.loads(comparison_file.read_text(encoding="utf-8"))
        )
        try:
            report_a = reports_by_version[legacy_comparison.model_a_version]
            report_b = reports_by_version[legacy_comparison.model_b_version]
        except KeyError as exc:
            raise ValueError(
                f"Legacy comparison references a report that was not verified: {comparison_file.name}."
            ) from exc
        verified_comparison = store.compare_reports(
            report_a,
            report_b,
            name_a=legacy_comparison.model_a_name,
            name_b=legacy_comparison.model_b_name,
        ).model_copy(update={
            "comparison_id": legacy_comparison.comparison_id,
            "compared_at": legacy_comparison.compared_at,
        })
        existing_comparison = existing_comparisons.get(verified_comparison.comparison_id)
        if existing_comparison is not None:
            if existing_comparison != verified_comparison:
                raise ValueError(f"Stored comparison conflicts with verified legacy data: {comparison_file.name}.")
            continue
        store.save_comparison(verified_comparison)
        existing_comparisons[verified_comparison.comparison_id] = verified_comparison
        migrated_comparisons += 1

    return len(report_files), migrated_comparisons
