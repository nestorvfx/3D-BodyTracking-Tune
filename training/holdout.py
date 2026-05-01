"""Hold-out enforcement for the v2 student training pipeline.

Loads the SOTA benchmark's manifest + subset and refuses to start training
if any candidate train take_uid intersects the held-out set.

Usage at training-job startup:
    from training.holdout import assert_no_leakage
    assert_no_leakage(train_take_uids,
                      manifest_path="benchmark/frames_manifest.json",
                      subset_path="benchmark/subset.json")
    # raises RuntimeError exit-99 on leak
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_forbidden_uids(manifest_path: Path, subset_path: Path) -> set[str]:
    """Load the union of held-out take_uids from both files."""
    forbidden: set[str] = set()
    if Path(manifest_path).exists():
        m = json.loads(Path(manifest_path).read_text())
        # frames_manifest.json is keyed by take_uid
        forbidden |= set(m.keys())
    if Path(subset_path).exists():
        s = json.loads(Path(subset_path).read_text())
        # subset.json has {"take_uids": [...]}
        if isinstance(s, dict) and "take_uids" in s:
            forbidden |= set(s["take_uids"])
        elif isinstance(s, list):
            forbidden |= set(s)
    return forbidden


def assert_no_leakage(train_uids, manifest_path: Path, subset_path: Path) -> None:
    forbidden = load_forbidden_uids(manifest_path, subset_path)
    leak = forbidden.intersection(set(train_uids))
    if leak:
        msg = (f"VAL TAKE UIDS LEAKED INTO TRAIN — abort\n"
               f"  forbidden corpus size: {len(forbidden)}\n"
               f"  train corpus size:     {len(set(train_uids))}\n"
               f"  intersection:          {len(leak)}\n"
               f"  examples: {sorted(leak)[:5]}")
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)
    print(f"[holdout] OK: 0/{len(set(train_uids))} train uids in held-out set "
          f"(checked against {len(forbidden)} forbidden)")


def _self_test():
    """Two-shot test: clean train list passes, poisoned train list raises."""
    HERE = Path(__file__).resolve().parent
    manifest = HERE.parent / "benchmark" / "frames_manifest.json"
    subset   = HERE.parent / "benchmark" / "subset.json"
    if not (manifest.exists() and subset.exists()):
        print("[self_test] benchmark files missing; skipping")
        return
    forbidden = load_forbidden_uids(manifest, subset)
    print(f"[self_test] forbidden N={len(forbidden)}")

    # Clean
    fake_clean = ["aaaa-train-1", "bbbb-train-2"]
    assert_no_leakage(fake_clean, manifest, subset)

    # Poisoned: pick one held-out uid
    poisoned = list(fake_clean) + [next(iter(forbidden))]
    try:
        assert_no_leakage(poisoned, manifest, subset)
    except RuntimeError as e:
        print("[self_test] poisoned input correctly raised:")
        print("   ", str(e).splitlines()[0])
        return
    raise SystemExit("[self_test] FAILED: poisoned input did not raise")


if __name__ == "__main__":
    _self_test()
