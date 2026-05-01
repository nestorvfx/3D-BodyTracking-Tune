"""C6 + C7 ablations: inference-mode (IMAGE vs VIDEO) and detector
confidence threshold sensitivity.

Runs Heavy on a small set of takes from the manifest under each setting
and prints a delta table.  Outputs `results/ablation_<axis>.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.keypoint_map import BP_INDEX_FOR_COCO, gt_to_coco17, HIP_L, HIP_R, COCO17
from lib.ego_exo_io import load_body_gt, load_camera_pose
from lib.metrics import pa_mpjpe_per_frame
from lib.projection import world_to_cam

ASSETS = HERE.parent / "assets"
DEFAULT_TASK = ASSETS / "pose_landmarker_heavy.task"


def make_landmarker(model_path: Path, *, mode: str = "image",
                    det_conf: float = 0.5, track_conf: float = 0.5):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    rm = {"image": mp_vision.RunningMode.IMAGE,
          "video": mp_vision.RunningMode.VIDEO}[mode]
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=rm,
        num_poses=1,
        output_segmentation_masks=False,
        min_pose_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts), mp


def evaluate_setting(mode, det_conf, manifest, frames_root, anno_root):
    landmarker, mp = make_landmarker(DEFAULT_TASK, mode=mode, det_conf=det_conf)
    pa_vals = []
    n_frames = n_detected = 0
    for uid in sorted(manifest):
        cp = load_camera_pose(anno_root/'ego_pose/val/camera_pose'/f'{uid}.json')
        gt = load_body_gt(anno_root/'ego_pose/val/body/annotation'/f'{uid}.json')
        for cam_name, fidxs in manifest[uid].items():
            cam = cp["cams"].get(cam_name)
            if cam is None: continue
            sorted_idxs = sorted(fidxs)
            for fi in sorted_idxs:
                jpg = frames_root/uid/cam_name/f"{fi:06d}.jpg"
                if not jpg.exists(): continue
                img_bgr = cv2.imread(str(jpg))
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                res = landmarker.detect(mp_img)
                n_frames += 1
                entry = gt.get(fi)
                if not entry: continue
                kp_w, present = gt_to_coco17(entry["annotation3D"])
                if not (present[HIP_L] and present[HIP_R]): continue
                gt_cam = world_to_cam(kp_w, cam["Rt"])
                if not res.pose_world_landmarks:
                    pa_vals.append(np.nan)
                    continue
                n_detected += 1
                lm = res.pose_world_landmarks[0]
                pred33 = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float64)
                pred17 = pred33[BP_INDEX_FOR_COCO]
                rg = 0.5*(gt_cam[HIP_L] + gt_cam[HIP_R])
                rp = 0.5*(pred17[HIP_L] + pred17[HIP_R])
                P = (pred17 - rp)[None]
                G = (gt_cam - rg)[None]
                M = present[None]
                err = pa_mpjpe_per_frame(P, G, M)
                pa_vals.append(float(err[0]))
    landmarker.close()
    arr = np.array([v for v in pa_vals if not np.isnan(v)])
    return {
        "mode":         mode,
        "det_conf":     det_conf,
        "n_frames":     int(n_frames),
        "n_detected":   int(n_detected),
        "det_rate_pct": 100.0 * n_detected / max(n_frames, 1),
        "pa_mpjpe_mm":  float(arr.mean()*1000) if len(arr) else float('nan'),
        "pa_median_mm": float(np.median(arr)*1000) if len(arr) else float('nan'),
        "pa_n":         int(len(arr)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   type=Path, default=HERE / "frames_manifest.json")
    p.add_argument("--frames-root",type=Path, default=HERE / "frames")
    p.add_argument("--anno-root",  type=Path, default=HERE / "raw" / "annotations")
    p.add_argument("--out-dir",    type=Path, default=HERE / "results")
    p.add_argument("--limit-takes",type=int, default=None,
                   help="Cap number of manifest takes to ablate on.")
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    full_manifest = json.loads(args.manifest.read_text())
    if args.limit_takes:
        keys = sorted(full_manifest.keys())[:args.limit_takes]
        manifest = {k: full_manifest[k] for k in keys}
    else:
        manifest = full_manifest
    print(f"[ablation] {len(manifest)} takes")

    # VIDEO mode is incompatible with our sparse manifest: it requires
    # monotonic per-stream timestamps over a contiguous frame sequence, but
    # the manifest holds 5 non-adjacent frames per (take, cam) tuple.  Adding
    # VIDEO mode would require re-extracting dense frame ranges per take and
    # only ablates a smoothing pass we don't apply in IMAGE mode anyway.
    # We ablate only the detector confidence threshold here.
    settings = [
        ("image", 0.3),
        ("image", 0.5),    # default
        ("image", 0.7),
    ]
    results = []
    for mode, det in settings:
        t0 = time.time()
        r = evaluate_setting(mode, det, manifest, args.frames_root, args.anno_root)
        r["wall_sec"] = time.time() - t0
        results.append(r)
        print(f"  mode={mode:5s}  det_conf={det:.1f}  N={r['n_frames']:4d}  "
              f"det={r['det_rate_pct']:5.1f}%  PA-MPJPE={r['pa_mpjpe_mm']:6.1f} mm  "
              f"({r['wall_sec']:.0f}s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[ablation] -> {out}")

    # Summary
    base = next(r for r in results if r["mode"]=="image" and abs(r["det_conf"]-0.5)<1e-6)
    print(f"\nDeltas vs (image, det_conf=0.5):")
    for r in results:
        if r is base: continue
        d_mpjpe = r["pa_mpjpe_mm"] - base["pa_mpjpe_mm"]
        d_det   = r["det_rate_pct"] - base["det_rate_pct"]
        print(f"  mode={r['mode']:5s} det={r['det_conf']:.1f}  "
              f"ΔPA-MPJPE={d_mpjpe:+5.1f} mm  Δdet={d_det:+5.1f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
