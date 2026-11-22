from __future__ import annotations

from neurovision_api.core.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR
from neurovision_api.services.comparison_migration import migrate_legacy_comparison_json
from neurovision_api.services.model_comparison import ModelComparisonStore


def main() -> int:
    store = ModelComparisonStore(artifacts_dir=DEFAULT_ARTIFACTS_DIR)
    reports, comparisons = migrate_legacy_comparison_json(
        store,
        data_dir=DEFAULT_DATA_DIR,
    )
    print(f"Verified and migrated {reports} reports and {comparisons} comparisons to SQLite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
