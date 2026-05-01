"""Aggregate predictions vs Ego-Exo4D GT into MPJPE / PA-MPJPE / PCK reports.

For each subset take, for each exo cam, for each annotated frame:
  - Load GT 3D in world frame -> transform to camera frame using static [R|t]
  - Filter to COCO-17 joints; mark absent joints as not-present
  - Centre BOTH GT and pred on their respective mid-hip (root-relative MPJPE).
  - Procrustes-align pred->gt for PA-MPJPE.

Outputs `results/<variant>_summary.json` per variant + a comparison.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.ego_exo_io import load_body_gt, load_camera_pose, is_exo_cam
from lib.keypoint_map import gt_to_coco17, COCO17, HIP_L, HIP_R
from lib.metrics import (root_center, mpjpe_per_joint, pa_mpjpe_per_frame,
                         aggregate_summary)
from lib.projection import world_to_cam


def score_variant(variant: str, subset_uids: list[str],
                  preds_root: Path, anno_root: Path,
                  manual_only: bool) -> dict:
    pred_dir = preds_root / variant
    err_chunks: list[np.ndarray] = []   # (N_t, 17) per-take per-frame per-joint err [m]
    pa_chunks:  list[np.ndarray] = []   # (N_t,)   per-take per-frame PA-MPJPE [m]
    mask_chunks: list[np.ndarray] = []
    detect_total = detect_hits = 0
    per_take_summaries: list[dict] = []

    for uid in subset_uids:
        pred_path = pred_dir / f"{uid}.json"
        gt_path   = anno_root / "ego_pose" / "val" / "body" / "annotation" / f"{uid}.json"
        cp_path   = anno_root / "ego_pose" / "val" / "camera_pose" / f"{uid}.json"
        if not (pred_path.exists() and gt_path.exists() and cp_path.exists()):
            continue
        preds = json.loads(pred_path.read_text())["preds"]
        gt    = load_body_gt(gt_path)
        cp    = load_camera_pose(cp_path)

        pred_list, gt_list, mask_list = [], [], []
        for cam_name, cam_preds in preds.items():
            if not is_exo_cam(cam_name):
                continue
            cam = cp["cams"].get(cam_name)
            if cam is None:
                continue
            Rt = cam["Rt"]
            for frame_str, pred in cam_preds.items():
                fi = int(frame_str)
                if fi not in gt:
                    continue
                detect_total += 1
                gt_world17, present = gt_to_coco17(gt[fi]["annotation3D"])
                gt_cam17 = world_to_cam(gt_world17, Rt)
                if not (present[HIP_L] and present[HIP_R]):
                    continue
                if pred.get("detected"):
                    detect_hits += 1
                    pred17_cam = np.asarray(pred["world17"], dtype=np.float64)
                else:
                    pred17_cam = np.full((17, 3), np.nan, dtype=np.float64)

                if manual_only:
                    cam2d = gt[fi].get("annotation2D", {}).get(cam_name, {})
                    for j_idx, name in enumerate(COCO17):
                        v = cam2d.get(name)
                        if v is None or v.get("placement") != "manual":
                            present[j_idx] = False

                pred_list.append(pred17_cam)
                gt_list.append(gt_cam17)
                mask_list.append(present)

        if not pred_list:
            continue
        P = np.stack(pred_list)
        G = np.stack(gt_list)
        M = np.stack(mask_list)

        # Root-relative on each side.
        Pc = root_center(P)
        Gc = root_center(G)
        per_joint = mpjpe_per_joint(Pc, Gc, M)               # (N_t, 17) m
        pa        = pa_mpjpe_per_frame(Pc, Gc, M)            # (N_t,)    m

        err_chunks.append(per_joint)
        pa_chunks.append(pa)
        mask_chunks.append(M)
        per_take_summaries.append({
            "take_uid":     uid,
            "n_frames":     int(P.shape[0]),
            "mpjpe_mm":     float(np.nanmean(per_joint) * 1000.0),
            "pa_mpjpe_mm":  float(np.nanmean(pa)        * 1000.0),
        })

    if not err_chunks:
        return {"variant": variant, "error": "no predictions matched any GT"}

    err  = np.concatenate(err_chunks,  axis=0)
    pa   = np.concatenate(pa_chunks,   axis=0)
    msk  = np.concatenate(mask_chunks, axis=0)
    summary = aggregate_summary(err, msk, pa)
    summary["variant"]            = variant
    summary["takes_scored"]       = len(per_take_summaries)
    summary["detection_rate_pct"] = (100.0 * detect_hits / max(detect_total, 1))
    summary["per_take"]           = per_take_summaries
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subset",      type=Path, default=HERE / "subset.json")
    p.add_argument("--preds-root",  type=Path, default=HERE / "predictions")
    p.add_argument("--anno-root",   type=Path, default=HERE / "raw" / "annotations")
    p.add_argument("--out-root",    type=Path, default=HERE / "results")
    p.add_argument("--variants",    nargs="+", default=["lite","full","heavy"])
    p.add_argument("--manual-only", action="store_true",
                   help="Score only joints whose 2D annotation is 'manual'.")
    args = p.parse_args()

    sub = json.loads(args.subset.read_text())
    take_uids = sub["take_uids"]
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for v in args.variants:
        print(f"\n=== {v} ===")
        s = score_variant(v, take_uids, args.preds_root, args.anno_root,
                          args.manual_only)
        out = args.out_root / f"{v}_summary.json"
        out.write_text(json.dumps(s, indent=2))
        if "error" not in s:
            print(f"  takes={s['takes_scored']}  frames={s['frames_scored']}  "
                  f"det={s['detection_rate_pct']:.1f}%  "
                  f"MPJPE={s['mpjpe_mm']:.1f} mm  "
                  f"PA-MPJPE={s['pa_mpjpe_mm']:.1f} mm  "
                  f"PCK150={s['pck150_mm_pct']:.1f}%  "
                  f"AUC0-200={s['auc_0_200']:.3f}")
            rows.append((v, s))
        else:
            print(f"  [error] {s['error']}")

    if rows:
        csv = ["variant,takes,frames,det_rate_pct,mpjpe_mm,pa_mpjpe_mm,"
               "pck50_mm,pck150_mm,auc_0_200"]
        for v, s in rows:
            csv.append(",".join(map(str, [
                v, s["takes_scored"], s["frames_scored"],
                f"{s['detection_rate_pct']:.2f}",
                f"{s['mpjpe_mm']:.2f}",
                f"{s['pa_mpjpe_mm']:.2f}",
                f"{s['pck50_mm_pct']:.2f}",
                f"{s['pck150_mm_pct']:.2f}",
                f"{s['auc_0_200']:.4f}",
            ])))
        (args.out_root / "comparison.csv").write_text("\n".join(csv) + "\n")
        print(f"\n[ok] comparison -> {args.out_root / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
