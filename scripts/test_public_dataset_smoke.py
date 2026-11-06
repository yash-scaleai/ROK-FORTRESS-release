"""Smoke test: load the public ROK-FORTRESS TSV and validate variant expansion."""

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rok_fortress_eval.core.data_loader import load_and_expand_csv  # noqa: E402

PUBLIC_TSV = REPO_ROOT / "data" / "rok_fortress_public.tsv"


def main() -> None:
    assert PUBLIC_TSV.exists(), f"public TSV missing at {PUBLIC_TSV}"

    variants = load_and_expand_csv(str(PUBLIC_TSV))
    assert variants, "loader returned no variants"

    by_variant = Counter(v.variant_name for v in variants)
    by_task_type = Counter(v.task_type for v in variants)
    unique_task_ids = {v.task_id for v in variants}

    expected_variant_names = {
        "original_en", "translated_kr", "cultural_adapted_en", "transcreated_kr",
    }
    found_variant_names = set(by_variant)
    unexpected = found_variant_names - expected_variant_names
    assert not unexpected, f"unexpected variant names: {unexpected}"

    for v in variants:
        assert v.task_id, "empty task_id"
        assert v.adversarial_prompt, f"empty adversarial_prompt for {v.task_id}/{v.variant_name}"
        assert v.rubric, f"empty rubric for {v.task_id}/{v.variant_name}"

    print(f"OK: {len(variants)} variants from {len(unique_task_ids)} tasks")
    print(f"by_variant_name: {dict(by_variant)}")
    print(f"by_task_type: {dict(by_task_type)}")


if __name__ == "__main__":
    main()
