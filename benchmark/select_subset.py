"""Pick a deterministic subset of body-pose val takes to benchmark on.

Writes `subset.json` listing the chosen take_uids.  Random with a fixed
seed so the benchmark is reproducible across reruns.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.ego_exo_io import list_body_takes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-root", type=Path,
                   default=HERE / "raw" / "annotations")
    p.add_argument("--n", type=int, default=50,
                   help="How many val takes to include in the benchmark.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=HERE / "subset.json")
    args = p.parse_args()

    all_uids = list_body_takes(args.annotations_root)
    if not all_uids:
        print(f"[error] no val body takes under {args.annotations_root}")
        return 2

    import random
    rng = random.Random(args.seed)
    pool = list(all_uids)
    rng.shuffle(pool)
    chosen = sorted(pool[: args.n])

    args.out.write_text(json.dumps({
        "n_total":     len(all_uids),
        "n_selected":  len(chosen),
        "seed":        args.seed,
        "take_uids":   chosen,
    }, indent=2))
    print(f"[ok] {len(chosen)} of {len(all_uids)} val body takes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
