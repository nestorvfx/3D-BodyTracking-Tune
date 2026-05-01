"""SOTA-grade analyzer for BlazePose Lite/Full/Heavy on Ego-Exo4D val (exo).

Addresses the critical issues with the naive aggregate-MPJPE numbers:

  1.  INTERSECTION-ONLY scoring  — only frames detected by ALL variants are
      compared, so detection-rate confounds don't drive the headline number.
  2.  Bootstrap 95% CIs           — per-frame resample (B=1000) on PA-MPJPE.
  3.  Per-scenario × per-distance — distance from camera in metres binned
      into NEAR (<2m), MID (2-4m), FAR (>4m); cross with the 8 Ego-Exo4D
      scenarios.
  4.  Manual-only filter          — drop joints whose 2D `placement` is
      `auto` (lower-quality triangulation) before scoring.
  5.  Outlier diagnosis           — per-take stats on distance, manual%,
      num_views_for_3d.  Outliers don't get silently dropped; they get
      explained.
  7.  Hip-bias measurement        — empirical 2D hip offset between BP and
      Ego-Exo4D.

Outputs `results/analysis.json` + console table.  Also rewrites
`results/RESULTS.md` with the SOTA-quality numbers.
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
from lib.keypoint_map import COCO17, gt_to_coco17, HIP_L, HIP_R
from lib.metrics import (root_center, pa_mpjpe_per_frame, mpjpe_per_joint,
                         pa_mpjpe_per_joint_frame, umeyama)
from lib.projection import world_to_cam

ANNO_ROOT_DEFAULT = HERE / "raw" / "annotations"

SCENARIO_PATTERNS = [
    ("basketball", re.compile(r"basketball", re.I)),
    ("dance",      re.compile(r"dance",      re.I)),
    ("bike",       re.compile(r"bike",       re.I)),
    ("cooking",    re.compile(r"cook",       re.I)),
    ("music",      re.compile(r"music|guitar|piano", re.I)),
    ("soccer",     re.compile(r"soccer",     re.I)),
    ("bouldering", re.compile(r"climb|bould", re.I)),
    ("health",     re.compile(r"covid|cpr|pcr|first[_-]?aid", re.I)),
]


def scenario_for(name: str) -> str:
    for s, rx in SCENARIO_PATTERNS:
        if rx.search(name):
            return s
    return "other"


def distance_bucket(z_m: float) -> str:
    if z_m < 2.0: return "near"
    if z_m < 4.0: return "mid"
    return "far"


def build_records(manifest, anno_root, preds_root, variants, manual_only):
    """Returns a flat list of per-(uid, cam, frame) records:
       record = {
           uid, take_name, scenario, cam, frame, hip_z_m, distance_bucket,
           gt_cam (17,3), present (17,) bool, n_views_median,
           pred[variant] = {detected: bool, world_cam17, vis17}
       }"""
    records = []
    for uid in sorted(manifest):
        cp = load_camera_pose(anno_root/'ego_pose/val/camera_pose'/f'{uid}.json')
        gt = load_body_gt(anno_root/'ego_pose/val/body/annotation'/f'{uid}.json')
        scenario = scenario_for(cp["take_name"])
        # preds keyed by variant
        var_preds = {}
        for v in variants:
            p = json.loads((preds_root/v/f'{uid}.json').read_text())
            var_preds[v] = p["preds"]

        for cam_name, fidxs in manifest[uid].items():
            cam = cp["cams"].get(cam_name)
            if cam is None: continue
            for fi in fidxs:
                entry = gt.get(fi)
                if not entry: continue
                kp_w, present = gt_to_coco17(entry["annotation3D"])
                if not (present[HIP_L] and present[HIP_R]):
                    continue
                gt_cam = world_to_cam(kp_w, cam["Rt"])
                hip_z = 0.5 * (gt_cam[HIP_L,2] + gt_cam[HIP_R,2])
                if manual_only:
                    cam2d = entry.get("annotation2D", {}).get(cam_name, {})
                    for j_idx, jname in enumerate(COCO17):
                        v2d = cam2d.get(jname)
                        if v2d is None or v2d.get("placement") != "manual":
                            present[j_idx] = False
                num_views = [j.get("num_views_for_3d",0)
                             for j in entry["annotation3D"].values()]
                pred_per_var = {}
                key = f"{fi:06d}"
                for v in variants:
                    cam_preds = var_preds[v].get(cam_name, {})
                    p = cam_preds.get(key, {})
                    if p.get("detected"):
                        pred_per_var[v] = {
                            "detected": True,
                            "world17":  np.asarray(p["world17"], dtype=np.float64),
                        }
                    else:
                        pred_per_var[v] = {"detected": False}
                records.append({
                    "uid": uid, "take_name": cp["take_name"], "scenario": scenario,
                    "cam": cam_name, "frame": fi,
                    "gt_cam": gt_cam, "present": present,
                    "hip_z_m": float(hip_z),
                    "distance_bucket": distance_bucket(float(hip_z)),
                    "n_views_median": int(np.median(num_views)) if num_views else 0,
                    "preds": pred_per_var,
                })
    return records


def pa_per_record(rec, variant, with_scale: bool = True):
    """PA-MPJPE in metres for one record, for one variant.
       Returns NaN if not detected, < 4 visible joints, or no hips visible.
       `with_scale=False` -> rigid-only (R+t, no scale)."""
    p = rec["preds"][variant]
    if not p.get("detected"):
        return np.nan
    pred17 = p["world17"]
    gt17   = rec["gt_cam"]
    mask   = rec["present"].copy()
    if int(mask.sum()) < 4:
        return np.nan
    if not (mask[HIP_L] and mask[HIP_R]):
        return np.nan
    rg = 0.5 * (gt17[HIP_L]   + gt17[HIP_R])
    rp = 0.5 * (pred17[HIP_L] + pred17[HIP_R])
    P = (pred17 - rp)[None]
    G = (gt17   - rg)[None]
    M = mask[None]
    err = pa_mpjpe_per_frame(P, G, M, with_scale=with_scale)
    return float(err[0])


def pa_per_joint_record(rec, variant, with_scale: bool = True) -> np.ndarray:
    """(17,) per-joint PA-aligned distances in metres for one record."""
    p = rec["preds"][variant]
    if not p.get("detected"):
        return np.full(17, np.nan)
    pred17 = p["world17"]
    gt17   = rec["gt_cam"]
    mask   = rec["present"].copy()
    if int(mask.sum()) < 4 or not (mask[HIP_L] and mask[HIP_R]):
        return np.full(17, np.nan)
    rg = 0.5 * (gt17[HIP_L]   + gt17[HIP_R])
    rp = 0.5 * (pred17[HIP_L] + pred17[HIP_R])
    P = (pred17 - rp)[None]
    G = (gt17   - rg)[None]
    M = mask[None]
    err = pa_mpjpe_per_joint_frame(P, G, M, with_scale=with_scale)
    return err[0]


def bootstrap_mean_ci(values, n_boot=1000, ci=0.95, rng=None):
    """Per-frame bootstrap mean + (lo, hi) CI.  Ignores NaN."""
    rng = rng or np.random.default_rng(0)
    v = np.array([x for x in values if not np.isnan(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo = np.quantile(means, (1-ci)/2)
    hi = np.quantile(means, 1-(1-ci)/2)
    return float(v.mean()), float(lo), float(hi), int(len(v))


def cluster_bootstrap_mean_ci(values, take_ids, n_boot=1000, ci=0.95, rng=None):
    """Take-level (cluster) bootstrap.  Resamples *takes* with replacement,
    then computes the mean of all frames in those resampled takes.  Honest
    for data with within-take correlation."""
    rng = rng or np.random.default_rng(0)
    pairs = [(take_ids[i], values[i]) for i in range(len(values))
             if not np.isnan(values[i])]
    if not pairs:
        return float("nan"), float("nan"), float("nan"), 0
    by_take = {}
    for tid, v in pairs:
        by_take.setdefault(tid, []).append(v)
    take_list = list(by_take)
    if len(take_list) < 2:
        return float("nan"), float("nan"), float("nan"), len(pairs)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.integers(0, len(take_list), size=len(take_list))
        all_v = []
        for s in sampled:
            all_v.extend(by_take[take_list[s]])
        means[b] = np.mean(all_v)
    lo = np.quantile(means, (1-ci)/2)
    hi = np.quantile(means, 1-(1-ci)/2)
    pooled_mean = np.mean([v for _, v in pairs])
    return float(pooled_mean), float(lo), float(hi), int(len(pairs))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   type=Path, default=HERE / "frames_manifest.json")
    p.add_argument("--anno-root",  type=Path, default=ANNO_ROOT_DEFAULT)
    p.add_argument("--preds-root", type=Path, default=HERE / "predictions")
    p.add_argument("--variants",   nargs="+", default=["lite","full","heavy"])
    p.add_argument("--manual-only", action="store_true",
                   help="Score only joints whose 2D annotation is 'manual'.")
    p.add_argument("--out",        type=Path, default=HERE / "results" / "analysis.json")
    p.add_argument("--bootstrap",  type=int, default=1000)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    print(f"[analyze] loading {len(manifest)} takes, "
          f"variants={args.variants}, manual_only={args.manual_only}")

    records = build_records(manifest, args.anno_root, args.preds_root,
                            args.variants, args.manual_only)
    print(f"[analyze] records: {len(records)}")

    # --- Per-record per-variant PA-MPJPE  -------------------------------
    pa       = {v: [pa_per_record(r, v, with_scale=True)  for r in records]
                for v in args.variants}
    pa_rigid = {v: [pa_per_record(r, v, with_scale=False) for r in records]
                for v in args.variants}
    pa_per_joint = {v: np.stack([pa_per_joint_record(r, v) for r in records])
                    for v in args.variants}                 # (N, 17)
    det = {v: np.array([r["preds"][v].get("detected", False)
                        for r in records], dtype=bool) for v in args.variants}

    rng = np.random.default_rng(42)

    # -- (a) per-variant on its own detection set ------------------------
    print("\n=== PA-MPJPE on each variant's own detection set ===")
    pv_stats = {}
    for v in args.variants:
        m, lo, hi, n = bootstrap_mean_ci(pa[v], args.bootstrap, rng=rng)
        det_pct = 100.0 * det[v].mean()
        pv_stats[v] = {"mean_mm": m*1000, "ci_lo_mm": lo*1000, "ci_hi_mm": hi*1000,
                       "n": n, "det_rate_pct": det_pct}
        print(f"  {v:6s}  N={n:4d}  det={det_pct:5.1f}%  "
              f"PA-MPJPE = {m*1000:6.1f} mm  [95% CI {lo*1000:6.1f} … {hi*1000:6.1f}]")

    # -- (b) intersection set: frames detected by ALL variants -----------
    inter_mask = det[args.variants[0]].copy()
    for v in args.variants[1:]:
        inter_mask &= det[v]
    print(f"\n=== Intersection: {int(inter_mask.sum())} frames detected by all variants ===")
    inter_stats = {}
    inter_stats_rigid = {}
    for v in args.variants:
        vals       = [pa[v][i]       for i in range(len(records)) if inter_mask[i]]
        vals_rigid = [pa_rigid[v][i] for i in range(len(records)) if inter_mask[i]]
        m,  lo,  hi,  n  = bootstrap_mean_ci(vals,       args.bootstrap, rng=rng)
        mr, lor, hir, nr = bootstrap_mean_ci(vals_rigid, args.bootstrap, rng=rng)
        inter_stats[v]       = {"mean_mm": m*1000,  "ci_lo_mm": lo*1000,  "ci_hi_mm": hi*1000,  "n": n}
        inter_stats_rigid[v] = {"mean_mm": mr*1000, "ci_lo_mm": lor*1000, "ci_hi_mm": hir*1000, "n": nr}
        print(f"  {v:6s}  N={n:4d}  "
              f"PA(scaled) = {m*1000:6.1f} mm [95% CI {lo*1000:6.1f}-{hi*1000:6.1f}]   "
              f"PA(rigid) = {mr*1000:6.1f} mm [{lor*1000:6.1f}-{hir*1000:6.1f}]")
    # Scale's contribution: rigid - scaled.  If large, scale is doing heavy lifting.
    print("\n  Scale's contribution to PA-MPJPE per variant (rigid − scaled):")
    for v in args.variants:
        d = inter_stats_rigid[v]["mean_mm"] - inter_stats[v]["mean_mm"]
        frac = 100 * d / max(inter_stats_rigid[v]["mean_mm"], 1e-6)
        print(f"    {v:6s}  +{d:5.1f} mm  ({frac:5.1f}% of rigid PA-MPJPE)")

    # -- (b2) cluster-bootstrap (resample *takes*) ------------------------
    take_ids = [r["uid"] for r in records]
    print("\n=== Cluster bootstrap (resample takes; honest CIs under within-take correlation) ===")
    inter_stats_cluster = {}
    for v in args.variants:
        vals = [pa[v][i] for i in range(len(records)) if inter_mask[i]]
        tids = [take_ids[i] for i in range(len(records)) if inter_mask[i]]
        m, lo, hi, n = cluster_bootstrap_mean_ci(vals, tids, args.bootstrap, rng=rng)
        inter_stats_cluster[v] = {"mean_mm": m*1000, "ci_lo_mm": lo*1000,
                                  "ci_hi_mm": hi*1000, "n": n}
        print(f"  {v:6s}  N={n}  PA-MPJPE = {m*1000:6.1f} mm  "
              f"[take-cluster 95% CI {lo*1000:6.1f} … {hi*1000:6.1f}]")

    # -- (c) paired deltas vs Heavy (frame-level) on intersection --------
    # The fairest single number for "is variant X better than Heavy on the same
    # frames" — within-frame variance cancels.  Add Wilcoxon signed-rank for
    # significance, with Holm multiple-comparison correction across the two
    # contrasts (Lite vs Heavy, Full vs Heavy).
    from scipy.stats import wilcoxon
    paired = {}
    pvals = {}
    print("\n=== Paired delta vs Heavy on intersection (negative = variant beats Heavy) ===")
    for v in args.variants:
        if v == "heavy": continue
        diffs   = [pa[v][i] - pa["heavy"][i]
                   for i in range(len(records)) if inter_mask[i]]
        tids    = [take_ids[i] for i in range(len(records)) if inter_mask[i]]
        diffs_a = np.array([d for d in diffs if not np.isnan(d)])
        m,  lo,  hi,  n  = bootstrap_mean_ci(diffs_a.tolist(),
                                             args.bootstrap, rng=rng)
        mc, loc, hic, _  = cluster_bootstrap_mean_ci(diffs, tids,
                                                     args.bootstrap, rng=rng)
        try:
            w_stat, w_p = wilcoxon(diffs_a, alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        paired[v] = {
            "delta_mm":           m*1000,
            "ci_lo_mm":           lo*1000,  "ci_hi_mm":           hi*1000,
            "ci_cluster_lo_mm":   loc*1000, "ci_cluster_hi_mm":   hic*1000,
            "n":                  int(n),
            "wilcoxon_p":         float(w_p),
        }
        pvals[v] = float(w_p)
        sign = "<" if m < 0 else ">"
        print(f"  {v:6s} {sign} heavy by {m*1000:+6.2f} mm  "
              f"[per-frame 95% CI {lo*1000:+6.2f} … {hi*1000:+6.2f}]  "
              f"[take-cluster 95% CI {loc*1000:+6.2f} … {hic*1000:+6.2f}]  "
              f"Wilcoxon p={w_p:.4f}")
    # Holm correction over the two simultaneous contrasts.
    sorted_p = sorted(pvals.items(), key=lambda kv: kv[1])
    holm = {}
    K = len(sorted_p)
    for rank, (v, p) in enumerate(sorted_p):
        adj = min(1.0, p * (K - rank))
        holm[v] = adj
        paired[v]["holm_adj_p"] = adj
    print("  Holm-adjusted p-values:", {v: f"{p:.4f}" for v, p in holm.items()})

    # -- (d) per-distance bucket, intersection only ----------------------
    print("\n=== Per-distance PA-MPJPE on intersection ===")
    dist_breakdown = {}
    for bucket in ["near","mid","far"]:
        idxs = [i for i in range(len(records))
                if inter_mask[i] and records[i]["distance_bucket"] == bucket]
        if not idxs: continue
        dist_breakdown[bucket] = {}
        print(f"  -- {bucket} ({len(idxs)} frames) --")
        for v in args.variants:
            vals = [pa[v][i] for i in idxs]
            m, lo, hi, n = bootstrap_mean_ci(vals, args.bootstrap, rng=rng)
            dist_breakdown[bucket][v] = {"mean_mm": m*1000,
                                         "ci_lo_mm": lo*1000, "ci_hi_mm": hi*1000, "n": n}
            print(f"    {v:6s}  PA-MPJPE = {m*1000:6.1f} mm  [95% CI {lo*1000:6.1f} … {hi*1000:6.1f}]")

    # -- (e) per-scenario, intersection only -----------------------------
    print("\n=== Per-scenario PA-MPJPE on intersection ===")
    scen_breakdown = {}
    by_scen = defaultdict(list)
    for i in range(len(records)):
        if inter_mask[i]:
            by_scen[records[i]["scenario"]].append(i)
    for sc, idxs in sorted(by_scen.items()):
        scen_breakdown[sc] = {}
        print(f"  -- {sc} ({len(idxs)} frames) --")
        for v in args.variants:
            vals = [pa[v][i] for i in idxs]
            m, lo, hi, n = bootstrap_mean_ci(vals, args.bootstrap, rng=rng)
            scen_breakdown[sc][v] = {"mean_mm": m*1000,
                                     "ci_lo_mm": lo*1000, "ci_hi_mm": hi*1000, "n": n}
            print(f"    {v:6s}  PA-MPJPE = {m*1000:6.1f} mm  [95% CI {lo*1000:6.1f} … {hi*1000:6.1f}]")

    # -- (f) hip-bias diagnostic ----------------------------------------
    # Expected: BlazePose hips are femoral-head; Ego-Exo4D hips are surface.
    # We compute the median delta of (BP world hip - GT cam hip) projected
    # onto the body's vertical axis.  Same root-centre as MPJPE.
    print("\n=== Empirical hip-bias diagnostic ===")
    deltas_hip_y = []  # along body Y axis (vertical-ish)
    for i, r in enumerate(records):
        if not inter_mask[i]: continue
        for v in [args.variants[-1]]:  # use heavy
            p = r["preds"][v]
            if not p["detected"]: continue
            pr_hip = 0.5 * (p["world17"][HIP_L] + p["world17"][HIP_R])
            gt_hip = 0.5 * (r["gt_cam"][HIP_L]   + r["gt_cam"][HIP_R])
            # Use GT shoulder-mid as a "above-hip" reference to define an axis.
            sho_idx_L = COCO17.index("left-shoulder")
            sho_idx_R = COCO17.index("right-shoulder")
            if not (r["present"][sho_idx_L] and r["present"][sho_idx_R]):
                continue
            sho_gt = 0.5*(r["gt_cam"][sho_idx_L] + r["gt_cam"][sho_idx_R])
            # Distance metric (root-relative): align pred to GT via shoulder-hip.
            # Cheap proxy: just report ||pr_hip_root - gt_hip_root|| after
            # centering on shoulder-hip mid.  Equivalent to per-axis bias.
            mid_gt = 0.5*(sho_gt + gt_hip)
            mid_pr = 0.5*(p["world17"][sho_idx_L]+p["world17"][sho_idx_R]+pr_hip*2)/2
            d = float(np.linalg.norm((pr_hip - mid_pr) - (gt_hip - mid_gt)))
            deltas_hip_y.append(d)
    deltas_arr = np.array(deltas_hip_y)
    if len(deltas_arr):
        print(f"  ||hip_pred - hip_gt|| (root-centred via shoulder-hip mid):  "
              f"median={1000*np.median(deltas_arr):.1f} mm  "
              f"p90={1000*np.percentile(deltas_arr,90):.1f} mm  N={len(deltas_arr)}")
    else:
        print("  (insufficient data)")

    # -- (g) outlier takes ----------------------------------------------
    print("\n=== Per-take PA-MPJPE on intersection (sorted by Heavy) ===")
    by_take = defaultdict(list)
    for i, r in enumerate(records):
        if inter_mask[i]:
            by_take[r["uid"]].append(i)
    take_rows = []
    for uid, idxs in by_take.items():
        if not idxs: continue
        rh = float(np.nanmean([pa["heavy"][i] for i in idxs])) * 1000
        rl = float(np.nanmean([pa["lite"][i]  for i in idxs])) * 1000
        rf = float(np.nanmean([pa["full"][i]  for i in idxs])) * 1000
        rec0 = records[idxs[0]]
        take_rows.append((rh, rec0["take_name"], rec0["scenario"],
                          rec0["distance_bucket"], rec0["hip_z_m"],
                          len(idxs), rl, rf, rh, uid))
    take_rows.sort()
    print(f"  {'take_name':40s}  {'scen':10s}  {'dist':5s}  hip_z   N    lite  full heavy")
    for rh, name, sc, db, z, n, rl, rf, rhh, uid in take_rows:
        print(f"  {name[:40]:40s}  {sc:10s}  {db:5s}  {z:5.2f}  {n:3d}  "
              f"{rl:5.0f} {rf:5.0f} {rhh:5.0f}")

    # -- (h) per-keypoint median + 95th-pct on intersection set ---------
    print("\n=== Per-keypoint PA-MPJPE on intersection (manual-only context) ===")
    col_w = 22
    header = f"  {'joint':16s}  " + "  ".join(f"{v:>{col_w}s}" for v in args.variants)
    sub_h  = f"  {' ':16s}  " + "  ".join(
        f"{'med   p95    N':>{col_w}s}" for _ in args.variants)
    print(header)
    print(sub_h)
    per_kp = {v: {} for v in args.variants}
    for j in range(17):
        cells = []
        for v in args.variants:
            vals = pa_per_joint[v][inter_mask, j]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                cells.append("  -- ")
                per_kp[v][COCO17[j]] = None
                continue
            med = float(np.median(vals)) * 1000
            p95 = float(np.percentile(vals, 95)) * 1000
            per_kp[v][COCO17[j]] = {"median_mm": med, "p95_mm": p95,
                                    "n": int(len(vals))}
            cells.append(f"{med:5.1f} {p95:5.1f} {len(vals):4d}")
        row = f"  {COCO17[j]:16s}  " + "  ".join(
            f"{c:>{col_w}s}" for c in cells)
        print(row)

    # -- (i) TL;DR — head-to-head intersection PA-MPJPE per variant ------
    # Lets us spot-check whether the v2 students beat or regress vs v1
    # without parsing the JSON.  The number to track is the "scaled" mean.
    print("\n=== TL;DR  (intersection scaled PA-MPJPE, mm; lower is better) ===")
    for v in args.variants:
        s = inter_stats[v]
        marker = ""
        # Compare v2 students against their v1 counterparts
        if v == "student_v2_lite" and "lite" in inter_stats:
            d = s["mean_mm"] - inter_stats["lite"]["mean_mm"]
            marker = f"  Δ vs v1 lite: {d:+6.2f} mm  {'BETTER' if d < 0 else 'WORSE'}"
        elif v == "student_v2_full" and "full" in inter_stats:
            d = s["mean_mm"] - inter_stats["full"]["mean_mm"]
            marker = f"  Δ vs v1 full: {d:+6.2f} mm  {'BETTER' if d < 0 else 'WORSE'}"
        print(f"  {v:18s}  {s['mean_mm']:6.2f}  "
              f"[95% CI {s['ci_lo_mm']:6.2f}-{s['ci_hi_mm']:6.2f}]  N={s['n']}"
              f"{marker}")

    # -- save analysis.json ----------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "manual_only":        args.manual_only,
        "n_records":          len(records),
        "n_intersection":     int(inter_mask.sum()),
        "per_variant_own_set": pv_stats,
        "intersection_scaled": inter_stats,
        "intersection_rigid":  inter_stats_rigid,
        "paired_vs_heavy":    paired,
        "per_distance":       dist_breakdown,
        "per_scenario":       scen_breakdown,
        "per_keypoint_intersection": per_kp,
        "per_take":           [
            {"take_name": name, "scenario": sc, "distance_bucket": db,
             "hip_z_m": z, "n_frames": n,
             "lite_mm": rl, "full_mm": rf, "heavy_mm": rhh, "uid": uid}
            for rh, name, sc, db, z, n, rl, rf, rhh, uid in take_rows],
        "hip_bias_diagnostic": {
            "median_mm": (1000*float(np.median(deltas_arr))
                          if len(deltas_arr) else None),
            "p90_mm":    (1000*float(np.percentile(deltas_arr,90))
                          if len(deltas_arr) else None),
            "n":         int(len(deltas_arr)),
        },
    }, indent=2))
    print(f"\n[analyze] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
