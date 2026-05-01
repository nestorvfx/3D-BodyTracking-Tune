"""Run BlazePose Lite/Full/Heavy on the extracted JPEG frames.

For each subset take, for each exo cam, for each annotated frame: run
the MediaPipe Pose Landmarker (.task) end-to-end (detector + landmarker),
take the 33 world_landmarks (camera-frame metres, mid-hip origin), slice
to 17 COCO joints, save predictions JSON.  No GT used at inference time.

We also save `pose_image_landmarks` (image-normalized 0..1) so a future
eval can compute 2D PCK if needed.
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

from lib.keypoint_map import BP_INDEX_FOR_COCO

ASSETS = HERE.parent / "assets"
MODEL_PATHS = {
    "lite":  ASSETS / "pose_landmarker_lite.task",
    "full":  ASSETS / "pose_landmarker_full.task",
    "heavy": ASSETS / "pose_landmarker_heavy.task",
}


def make_landmarker(model_path: Path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def landmark_to_arr(lms) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence]
                     for lm in lms], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--manifest",   type=Path, default=HERE / "frames_manifest.json",
                   help="Diversity-filtered frame manifest (from select_frames.py).")
    p.add_argument("--frames-root", type=Path, default=HERE / "frames")
    p.add_argument("--out-root",    type=Path, default=HERE / "predictions")
    p.add_argument("--limit-takes", type=int, default=None,
                   help="Sanity-cap on number of takes (debug only).")
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    import mediapipe as mp                                # noqa: F401  (verify import)
    if not args.manifest.exists():
        print(f"[error] manifest not found: {args.manifest} "
              f"— run select_frames.py first.")
        return 2
    manifest: dict[str, dict[str, list[int]]] = json.loads(args.manifest.read_text())
    take_uids = sorted(manifest.keys())
    if args.limit_takes:
        take_uids = take_uids[: args.limit_takes]

    out_dir = args.out_root / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_eval] variant={args.variant} model={MODEL_PATHS[args.variant].name}")
    print(f"[run_eval] takes={len(take_uids)}  frames_root={args.frames_root}")

    landmarker = make_landmarker(MODEL_PATHS[args.variant])
    t0 = time.time()
    n_frames_total = 0
    for i, uid in enumerate(take_uids):
        out_path = out_dir / f"{uid}.json"
        if out_path.exists():
            print(f"  [{i+1}/{len(take_uids)}] {uid}  (cached, skipping)")
            continue
        take_dir = args.frames_root / uid
        if not take_dir.exists():
            print(f"  [{i+1}/{len(take_uids)}] {uid}  (no frames; skipping)")
            continue

        per_take = {"take_uid": uid, "variant": args.variant, "preds": {}}
        n_in_take = 0
        cams_for_take = manifest[uid]
        for cam_name, frame_idxs in sorted(cams_for_take.items()):
            cam_dir = take_dir / cam_name
            if not cam_dir.is_dir():
                continue
            cam_preds: dict[str, dict] = {}
            for fi in sorted(frame_idxs):
                jpg = cam_dir / f"{fi:06d}.jpg"
                if not jpg.exists():
                    continue
                img_bgr = cv2.imread(str(jpg))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=img_rgb)
                res = landmarker.detect(mp_img)
                if not res.pose_world_landmarks:
                    cam_preds[jpg.stem] = {"detected": False}
                    n_in_take += 1
                    continue
                world33 = landmark_to_arr(res.pose_world_landmarks[0])  # (33,5)
                img33   = landmark_to_arr(res.pose_landmarks[0])        # (33,5)
                world17 = world33[BP_INDEX_FOR_COCO]                    # (17,5)
                img17   = img33  [BP_INDEX_FOR_COCO]                    # (17,5)
                cam_preds[jpg.stem] = {
                    "detected": True,
                    "world17": world17[:, :3].tolist(),
                    "vis17":   world17[:,  3].tolist(),
                    "img17":   img17[:, :3].tolist(),
                    "img17_w": int(img_bgr.shape[1]),
                    "img17_h": int(img_bgr.shape[0]),
                }
                n_in_take += 1
            per_take["preds"][cam_name] = cam_preds

        out_path.write_text(json.dumps(per_take))
        n_frames_total += n_in_take
        dt = time.time() - t0
        rate = n_frames_total / max(dt, 1e-6)
        print(f"  [{i+1}/{len(take_uids)}] {uid}  frames={n_in_take}  "
              f"cum={n_frames_total}  ({rate:.1f} fps)")

    landmarker.close()
    print(f"\n[done] {n_frames_total} frames in "
          f"{(time.time()-t0)/60:.1f} min  -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
