"""Render skeleton-overlay PNGs for visual verification (criterion F3 + C2).

For a list of (uid, cam, frame) tuples, draw on the JPEG:
  - GT 17 keypoints projected from world via static [K|Rt|dist]   (cyan)
  - BlazePose Heavy's prediction projected to image                (yellow)
  - Joint labels for joints with high per-joint error              (red text)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.ego_exo_io import load_body_gt, load_camera_pose
from lib.keypoint_map import COCO17, gt_to_coco17, HIP_L, HIP_R
from lib.projection import world_to_cam, project_with_distortion

# COCO-17 skeleton edges by index pairs
EDGES = [
    (5,7),(7,9),(6,8),(8,10),         # arms
    (5,6),(11,12),(5,11),(6,12),      # torso
    (11,13),(13,15),(12,14),(14,16),  # legs
    (0,1),(0,2),(1,3),(2,4),          # face
]

CYAN  = (255, 255, 0)
YELLOW = (0, 255, 255)
RED   = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def draw_skeleton(img, kp_2d, present, color, thickness=2, radius=4):
    H, W = img.shape[:2]
    for a, b in EDGES:
        if not (present[a] and present[b]):
            continue
        pa = (int(kp_2d[a,0]), int(kp_2d[a,1]))
        pb = (int(kp_2d[b,0]), int(kp_2d[b,1]))
        if (0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H):
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    for j in range(17):
        if not present[j]: continue
        x, y = int(kp_2d[j,0]), int(kp_2d[j,1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   type=Path, default=HERE / "frames_manifest.json")
    p.add_argument("--anno-root",  type=Path, default=HERE / "raw" / "annotations")
    p.add_argument("--frames-root",type=Path, default=HERE / "frames")
    p.add_argument("--preds-root", type=Path, default=HERE / "predictions")
    p.add_argument("--variant",    default="heavy")
    p.add_argument("--out-dir",    type=Path, default=HERE / "results" / "overlays")
    p.add_argument("--per-take",   type=int, default=2,
                   help="Render this many frames per take.")
    p.add_argument("--scale",      type=float, default=1.0,
                   help="Resolution factor: K is in native units; downscaled frames "
                        "are 0.207x.  Set 0.207 if rendering on 448p frames.")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render] {len(manifest)} takes, {args.per_take} per take -> {args.out_dir}")

    n_done = 0
    for uid in sorted(manifest):
        cp = load_camera_pose(args.anno_root/'ego_pose/val/camera_pose'/f'{uid}.json')
        gt = load_body_gt(args.anno_root/'ego_pose/val/body/annotation'/f'{uid}.json')
        preds = json.loads((args.preds_root/args.variant/f'{uid}.json').read_text())["preds"]
        for cam_name, fidxs in manifest[uid].items():
            cam = cp["cams"].get(cam_name)
            if cam is None: continue
            for fi in fidxs[:args.per_take]:
                jpg_path = args.frames_root/uid/cam_name/f"{fi:06d}.jpg"
                if not jpg_path.exists(): continue
                img = cv2.imread(str(jpg_path))
                if img is None: continue
                H, W = img.shape[:2]
                # GT 3D -> projected 2D (in native px); scale to extracted frame.
                entry = gt.get(fi)
                if not entry: continue
                kp_w, present = gt_to_coco17(entry["annotation3D"])
                kp_c = world_to_cam(kp_w, cam["Rt"])
                kp_im = project_with_distortion(kp_c, cam["K"], cam["dist"])
                # Find downscaling factor by comparing W to K's principal point cx (≈ native_W/2)
                cx = cam["K"][0, 2]
                scale = W / (2.0 * cx)              # auto-detect downscale
                kp_im_scaled = kp_im * scale
                draw_skeleton(img, kp_im_scaled, present, CYAN, thickness=2, radius=4)

                # Pred — image landmarks are normalized 0..1 in original ROI/image
                pred_entry = preds.get(cam_name, {}).get(f"{fi:06d}", {})
                if pred_entry.get("detected"):
                    img17 = np.asarray(pred_entry["img17"])
                    pw, ph = pred_entry.get("img17_w", W), pred_entry.get("img17_h", H)
                    pred_2d = np.column_stack([img17[:,0]*pw, img17[:,1]*ph])
                    pres_pred = np.ones(17, dtype=bool)
                    draw_skeleton(img, pred_2d, pres_pred, YELLOW, thickness=2, radius=3)
                    # Label
                    cv2.putText(img, "GT (cyan) / heavy (yellow)", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
                else:
                    cv2.putText(img, "GT (cyan) / heavy: NO DETECT", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 1, cv2.LINE_AA)
                lbl = f"{cp['take_name']}  {cam_name}  f{fi}  hip_z={kp_c[HIP_L,2]:.1f}m"
                cv2.putText(img, lbl, (10, H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
                out_name = f"{cp['take_name']}__{cam_name}__f{fi:06d}.jpg"
                cv2.imwrite(str(args.out_dir / out_name), img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                n_done += 1
    print(f"[render] {n_done} overlays written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
