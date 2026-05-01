"""Per-take: download exo videos via egoexo CLI, extract just the
GT-annotated frames as JPEG, delete the videos, move on.

Caps peak transient disk at one-take's-video-worth (~50-300 MB) regardless
of total subset size.  Persistent output: `frames/<take_uid>/<cam>/<frame>.jpg`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.ego_exo_io import load_body_gt, load_camera_pose, is_exo_cam

DEFAULT_EGOEXO = (
    Path("C:/Users/Mihajlo/miniconda3/envs/bodytrack/Scripts/egoexo.exe"))


def pull_take_videos(out_dir: Path, take_uid: str, egoexo: Path) -> int:
    """Calls the egoexo CLI to download `downscaled_takes/448` for one take."""
    cmd = [str(egoexo), "-o", str(out_dir),
           "--parts", "downscaled_takes/448",
           "--uids", take_uid,
           "--views", "exo",
           "-y"]
    return subprocess.run(cmd, check=False).returncode


def extract_take(take_uid: str, raw_root: Path, frames_root: Path,
                 anno_root: Path, jpeg_quality: int) -> dict:
    """For one take: open every exo cam mp4, decode the GT-annotated frames,
    write JPEGs.  Returns a per-take manifest dict."""
    cam_pose_path = anno_root / "ego_pose" / "val" / "camera_pose" / f"{take_uid}.json"
    body_path     = anno_root / "ego_pose" / "val" / "body" / "annotation" / f"{take_uid}.json"
    cp = load_camera_pose(cam_pose_path)
    gt = load_body_gt(body_path)
    take_name = cp["take_name"]
    gt_frame_idxs = sorted(gt.keys())

    take_video_dir = (raw_root / "takes" / take_name
                      / "frame_aligned_videos" / "downscaled" / "448")

    manifest = {
        "take_uid":   take_uid,
        "take_name":  take_name,
        "cams":       {},
        "n_gt_frames": len(gt_frame_idxs),
    }
    out_take_dir = frames_root / take_uid
    out_take_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in cp["cams"]:
        if not is_exo_cam(cam_name):
            continue
        mp4 = take_video_dir / f"{cam_name}.mp4"
        if not mp4.exists():
            print(f"  [warn] {cam_name}.mp4 not found at {mp4}")
            continue
        cam_out = out_take_dir / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            print(f"  [warn] cv2 failed to open {mp4}")
            continue
        n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok_count = 0
        skipped = 0
        for fi in gt_frame_idxs:
            if fi >= n_frames_video:
                skipped += 1
                continue
            out_jpg = cam_out / f"{fi:06d}.jpg"
            if out_jpg.exists():
                ok_count += 1
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                skipped += 1
                continue
            cv2.imwrite(str(out_jpg), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            ok_count += 1
        cap.release()
        manifest["cams"][cam_name] = {
            "n_video_frames": n_frames_video,
            "n_extracted":    ok_count,
            "n_skipped":      skipped,
        }
        print(f"  {cam_name}: extracted {ok_count} (skipped {skipped})")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=Path, default=HERE / "subset.json")
    p.add_argument("--raw-root", type=Path, default=HERE / "raw")
    p.add_argument("--frames-root", type=Path, default=HERE / "frames")
    p.add_argument("--egoexo", type=Path, default=DEFAULT_EGOEXO)
    p.add_argument("--keep-videos", action="store_true",
                   help="Don't delete videos after extraction.")
    p.add_argument("--jpeg-quality", type=int, default=90)
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    if not args.subset.exists():
        print(f"[error] subset.json not found: {args.subset} "
              f"— run select_subset.py first.")
        return 2
    if not args.egoexo.exists():
        print(f"[error] egoexo CLI not at {args.egoexo}")
        return 3
    sub = json.loads(args.subset.read_text())
    take_uids = sub["take_uids"]
    args.frames_root.mkdir(parents=True, exist_ok=True)

    anno_root = args.raw_root / "annotations"
    manifests = []
    t0 = time.time()
    for i, uid in enumerate(take_uids):
        print(f"\n=== [{i+1}/{len(take_uids)}] {uid} ===")
        cam_pose_path = (anno_root / "ego_pose" / "val" / "camera_pose"
                         / f"{uid}.json")
        if not cam_pose_path.exists():
            print(f"  [warn] no camera_pose JSON for {uid}; skipping")
            continue
        cp = load_camera_pose(cam_pose_path)
        take_name = cp["take_name"]
        take_video_dir = (args.raw_root / "takes" / take_name
                          / "frame_aligned_videos" / "downscaled" / "448")

        # Skip download if any exo cam mp4 already exists for this take.
        already_have = any(
            (take_video_dir / f"{c}.mp4").exists()
            for c in cp["cams"] if is_exo_cam(c))
        if not already_have:
            print(f"  pulling videos for {take_name} ...")
            rc = pull_take_videos(args.raw_root, uid, args.egoexo)
            if rc != 0:
                print(f"  [warn] egoexo returned {rc}; skipping take")
                continue
        else:
            print(f"  already have videos for {take_name}, reusing")

        try:
            mani = extract_take(uid, args.raw_root, args.frames_root,
                                anno_root, args.jpeg_quality)
        except Exception as e:
            print(f"  [error] extract failed: {e}")
            continue
        manifests.append(mani)

        if not args.keep_videos:
            take_dir = args.raw_root / "takes" / take_name
            if take_dir.exists():
                shutil.rmtree(take_dir, ignore_errors=True)
                print(f"  deleted videos for {take_name}")

    out_manifest = args.frames_root / "_manifest.json"
    out_manifest.write_text(json.dumps(manifests, indent=2))
    dt = time.time() - t0
    print(f"\n[done] {len(manifests)} takes in {dt/60:.1f} min "
          f"-> {args.frames_root}")
    print(f"manifest: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
