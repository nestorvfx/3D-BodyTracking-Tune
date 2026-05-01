"""Build a diversity-filtered frame manifest from the extracted val takes.

What this guards against:
1. **Scenario skew**: random 50-take sampling over-represents dance and
   under-represents bouldering/cooking.  We cap takes per
   (scenario × university) combo to spread coverage.
2. **Subject/environment over-fit**: 21 of 25 dance takes in the random
   subset come from a single university (same studio, same lighting).
   The (scenario × university) cap fixes this too.
3. **Temporal pose redundancy**: GT is annotated at 10 Hz over a ~10 sec
   window per take.  Consecutive frames can be < 1 cm apart in pose.
   We require min temporal stride AND min root-relative pose displacement
   between picked frames.

Output: `frames_manifest.json`:
    { "<take_uid>": {"<cam_name>": [frame_idx_1, frame_idx_2, ...], ...}, ... }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.ego_exo_io import load_body_gt, load_camera_pose, is_exo_cam
from lib.keypoint_map import gt_to_coco17, HIP_L, HIP_R


# Map common take_name tokens to Ego-Exo4D's 8 official scenarios.  Anything
# unmatched falls into "other"; the audit will flag if "other" is non-empty.
SCENARIO_PATTERNS = [
    ("basketball", re.compile(r"basketball", re.I)),
    ("dance",      re.compile(r"dance",      re.I)),
    ("bike",       re.compile(r"bike",       re.I)),
    ("cooking",    re.compile(r"cook",       re.I)),
    ("music",      re.compile(r"music|guitar|piano", re.I)),
    ("soccer",     re.compile(r"soccer",     re.I)),
    ("bouldering", re.compile(r"climb|bould", re.I)),
    # Health procedures (CPR/PCR/COVID) are grouped — Ego-Exo4D treats
    # them as one "covid-19 procedures" scenario family.
    ("health",     re.compile(r"covid|cpr|pcr|first[_-]?aid", re.I)),
]


def scenario_for(take_name: str) -> str:
    for s, rx in SCENARIO_PATTERNS:
        if rx.search(take_name):
            return s
    return "other"


def university_for(take_name: str) -> str:
    return take_name.split("_")[0]


def root_center_kp(kp: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Centre on mid-hip; return NaN if either hip absent."""
    if not (present[HIP_L] and present[HIP_R]):
        return np.full_like(kp, np.nan)
    root = 0.5 * (kp[HIP_L] + kp[HIP_R])
    return kp - root


def pose_distance(kp1: np.ndarray, p1: np.ndarray,
                  kp2: np.ndarray, p2: np.ndarray) -> float:
    """Root-relative mean joint distance between two pose snapshots, only on
    joints visible in BOTH.  Returns NaN if either hip is absent or fewer
    than 6 joints are co-visible.  Missing joints (NaN) are masked out via
    `common`; we don't reject just because some non-common joint is NaN."""
    if not (p1[HIP_L] and p1[HIP_R] and p2[HIP_L] and p2[HIP_R]):
        return float("nan")
    common = p1 & p2
    if int(common.sum()) < 6:
        return float("nan")
    root1 = 0.5 * (kp1[HIP_L] + kp1[HIP_R])
    root2 = 0.5 * (kp2[HIP_L] + kp2[HIP_R])
    return float(np.linalg.norm(
        (kp1[common] - root1) - (kp2[common] - root2), axis=-1).mean())


def pick_diverse_frames(gt: dict[int, dict], k: int,
                        min_stride: int, min_pose_disp_m: float) -> list[int]:
    """Return up to k frame indices from gt with:
       - Δ frame-idx ≥ min_stride between consecutive picks
       - Δ root-relative mean joint distance ≥ min_pose_disp_m (metres)
    """
    sorted_idxs = sorted(gt.keys())
    if not sorted_idxs:
        return []

    picked: list[int] = [sorted_idxs[0]]
    last_kp, last_pres = gt_to_coco17(gt[picked[0]]["annotation3D"])

    for fi in sorted_idxs[1:]:
        if fi - picked[-1] < min_stride:
            continue
        cur_kp, cur_pres = gt_to_coco17(gt[fi]["annotation3D"])
        d = pose_distance(last_kp, last_pres, cur_kp, cur_pres)
        if np.isnan(d) or d < min_pose_disp_m:
            continue
        picked.append(fi)
        last_kp, last_pres = cur_kp, cur_pres
        if len(picked) >= k:
            break
    return picked


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subset",       type=Path, default=HERE / "subset.json")
    p.add_argument("--frames-root",  type=Path, default=HERE / "frames")
    p.add_argument("--anno-root",    type=Path, default=HERE / "raw" / "annotations")
    p.add_argument("--out-manifest", type=Path, default=HERE / "frames_manifest.json")
    p.add_argument("--out-audit",    type=Path, default=HERE / "diversity_audit.json")
    p.add_argument("--max-per-combo", type=int, default=2,
                   help="Max takes per (scenario × university).")
    p.add_argument("--frames-per-take", type=int, default=6,
                   help="Max diverse frames per take (per cam).")
    p.add_argument("--min-stride",   type=int, default=10,
                   help="Min Δframe-idx between picked frames "
                        "(GT is at 10 Hz so 10 ≈ 1 sec).")
    p.add_argument("--min-pose-disp-m", type=float, default=0.05,
                   help="Min root-relative pose displacement (metres).")
    p.add_argument("--min-frames-per-take", type=int, default=2,
                   help="Drop takes that can't yield this many diverse frames.")
    args = p.parse_args()

    sub = json.loads(args.subset.read_text())
    take_uids = sub["take_uids"]
    print(f"[diversity] auditing {len(take_uids)} takes")

    # ── 1. Scenario × university audit ───────────────────────────────────
    by_combo: dict[tuple[str, str], list[str]] = defaultdict(list)
    take_meta: dict[str, dict] = {}
    for uid in take_uids:
        cp_path = (args.anno_root / "ego_pose" / "val" / "camera_pose"
                   / f"{uid}.json")
        if not cp_path.exists():
            continue
        cp = load_camera_pose(cp_path)
        name = cp["take_name"]
        sc = scenario_for(name)
        uni = university_for(name)
        by_combo[(sc, uni)].append(uid)
        take_meta[uid] = {"take_name": name, "scenario": sc, "university": uni,
                          "cams": list(cp["cams"].keys())}

    # ── 2. Stratified take pool (cap per scenario×university) ────────────
    rng = np.random.default_rng(123)
    chosen: list[str] = []
    for combo, uids in sorted(by_combo.items()):
        order = rng.permutation(len(uids))
        for j in order[: args.max_per_combo]:
            chosen.append(uids[j])
    chosen.sort()
    print(f"[diversity] stratified take pool: {len(chosen)} takes "
          f"(from {len(take_uids)} starting)")

    # ── 3. Per-take frame selection with stride + pose-displacement gates ─
    manifest: dict[str, dict[str, list[int]]] = {}
    take_motion: dict[str, float] = {}
    dropped_low_motion: list[str] = []
    for uid in chosen:
        body_path = (args.anno_root / "ego_pose" / "val" / "body" / "annotation"
                     / f"{uid}.json")
        gt = load_body_gt(body_path)
        idxs = pick_diverse_frames(gt, args.frames_per_take,
                                   args.min_stride, args.min_pose_disp_m)
        if len(idxs) < args.min_frames_per_take:
            dropped_low_motion.append(uid)
            continue
        # Mean motion across the picked frames (sanity stat).
        disps = []
        for a, b in zip(idxs[:-1], idxs[1:]):
            kp_a, p_a = gt_to_coco17(gt[a]["annotation3D"])
            kp_b, p_b = gt_to_coco17(gt[b]["annotation3D"])
            d = pose_distance(kp_a, p_a, kp_b, p_b)
            if not np.isnan(d):
                disps.append(d)
        take_motion[uid] = float(np.mean(disps)) if disps else 0.0
        # Match against extracted JPEGs per cam.
        take_dir = args.frames_root / uid
        cams_used: dict[str, list[int]] = {}
        if take_dir.exists():
            for cam_dir in sorted(take_dir.iterdir()):
                if not cam_dir.is_dir() or not is_exo_cam(cam_dir.name):
                    continue
                avail = {int(p.stem) for p in cam_dir.glob("*.jpg")}
                kept  = [fi for fi in idxs if fi in avail]
                if kept:
                    cams_used[cam_dir.name] = kept
        if cams_used:
            manifest[uid] = cams_used

    # ── 4. Diversity audit ───────────────────────────────────────────────
    final_uids = list(manifest.keys())
    by_scen = defaultdict(int)
    by_uni  = defaultdict(int)
    for uid in final_uids:
        m = take_meta.get(uid, {})
        by_scen[m.get("scenario", "?")] += 1
        by_uni [m.get("university", "?")] += 1

    n_total_frames = sum(len(idxs)
                         for cams in manifest.values()
                         for idxs in cams.values())
    audit = {
        "starting_take_pool":          len(take_uids),
        "after_combo_cap":             len(chosen),
        "after_low_motion_drop":       len(final_uids),
        "low_motion_dropped":          dropped_low_motion,
        "scenario_distribution":       dict(by_scen),
        "university_distribution":     dict(by_uni),
        "scenarios_missing":           sorted(
            {s for s, _ in SCENARIO_PATTERNS} - set(by_scen.keys())),
        "frames_per_take_target":      args.frames_per_take,
        "min_stride_frames":           args.min_stride,
        "min_pose_disp_m":             args.min_pose_disp_m,
        "max_per_combo":               args.max_per_combo,
        "total_frames_in_manifest":    n_total_frames,
        "mean_motion_m_per_take":      {uid: round(v, 4) for uid, v in
                                        sorted(take_motion.items())},
        "take_meta": {uid: take_meta[uid] for uid in final_uids},
    }
    args.out_manifest.write_text(json.dumps(manifest, indent=2))
    args.out_audit.write_text(json.dumps(audit, indent=2))

    print(f"[diversity] final benchmark: {len(final_uids)} takes, "
          f"{n_total_frames} frame-evaluations")
    print(f"[diversity] scenario coverage: {dict(by_scen)}")
    print(f"[diversity] university coverage: {dict(by_uni)}")
    if audit["scenarios_missing"]:
        print(f"[warn] scenarios with NO takes: {audit['scenarios_missing']}")
    if dropped_low_motion:
        print(f"[info] {len(dropped_low_motion)} takes dropped for low pose "
              f"motion (couldn't reach min frames)")
    print(f"[diversity] manifest -> {args.out_manifest}")
    print(f"[diversity] audit    -> {args.out_audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
