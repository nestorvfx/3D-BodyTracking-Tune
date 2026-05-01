"""Recreate `benchmark/frames/` on Vast.ai from Ego-Exo4D val split.

The 26 take_uids and the per-frame manifest are committed to the repo
(`benchmark/subset.json`, `benchmark/frames_manifest.json`) so the result
is byte-stable with the local extraction.  We just need to fetch the
videos + annotations from Ego-Exo4D and decode the listed frames.

This is the linux-friendly batched cousin of `benchmark/stream_extract.py`
(which hardcodes a Windows path to the egoexo CLI and goes one-take-at-
a-time).  The egoexo CLI's per-call metadata fetch (~30 s) dominates a
per-take loop; one batched call amortises that across all 26 takes.

Output (matching what `train.py:benchmark_eval()` and `benchmark/run_eval.py`
expect, both of which read relative to `BlazePose tune/benchmark/`):

  benchmark/raw/annotations/ego_pose/val/{body,camera_pose}/<uid>.json
  benchmark/frames/<uid>/<cam>/<frame:06d>.jpg

Idempotent: re-running with frames already on disk is a fast no-op.
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
BENCH = HERE.parent / "benchmark"
sys.path.insert(0, str(BENCH))

from lib.ego_exo_io import load_camera_pose, is_exo_cam


def pull_val_annotations(out_dir: Path, egoexo: str) -> int:
    """One egoexo call for the val annotations + metadata (~80 MB)."""
    cmd = [egoexo, "-o", str(out_dir),
           "--parts", "annotations", "metadata",
           "--benchmarks", "egopose",
           "--splits", "val",
           "-y"]
    return subprocess.run(cmd, check=False).returncode


def pull_take_videos_batch(out_dir: Path, take_uids: list[str],
                            egoexo: str, num_workers: int = 16) -> int:
    """Single egoexo call for all 26 takes' downscaled exo videos."""
    cmd = [egoexo, "-o", str(out_dir),
           "--parts", "downscaled_takes/448",
           "--uids", *take_uids,
           "--views", "exo",
           "--num_workers", str(num_workers),
           "-y"]
    return subprocess.run(cmd, check=False).returncode


def extract_frames_for_take(uid: str, raw_root: Path, frames_root: Path,
                            anno_root: Path, manifest_for_uid: dict,
                            jpeg_quality: int) -> tuple[int, int]:
    """For one take: open each exo cam mp4, extract only the frames listed
    in the manifest (NOT all GT frames), write JPEGs.  Returns
    (extracted_count, missing_count)."""
    cp_path = anno_root / "ego_pose" / "val" / "camera_pose" / f"{uid}.json"
    if not cp_path.exists():
        return 0, sum(len(v) for v in manifest_for_uid.values())
    cp = load_camera_pose(cp_path)
    take_name = cp["take_name"]
    take_video_dir = (raw_root / "takes" / take_name
                      / "frame_aligned_videos" / "downscaled" / "448")
    if not take_video_dir.exists():
        return 0, sum(len(v) for v in manifest_for_uid.values())

    out_take_dir = frames_root / uid
    out_take_dir.mkdir(parents=True, exist_ok=True)

    n_extracted = n_missing = 0
    for cam_name, frame_idxs in manifest_for_uid.items():
        if not is_exo_cam(cam_name):
            continue
        mp4 = take_video_dir / f"{cam_name}.mp4"
        cam_out = out_take_dir / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)
        if not mp4.exists():
            n_missing += len(frame_idxs)
            continue
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            n_missing += len(frame_idxs)
            continue
        n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in sorted(int(f) for f in frame_idxs):
            out_jpg = cam_out / f"{fi:06d}.jpg"
            if out_jpg.exists():
                n_extracted += 1
                continue
            if fi >= n_video_frames:
                n_missing += 1
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                n_missing += 1
                continue
            cv2.imwrite(str(out_jpg), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            n_extracted += 1
        cap.release()
    return n_extracted, n_missing


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subset",   type=Path, default=BENCH / "subset.json")
    p.add_argument("--manifest", type=Path, default=BENCH / "frames_manifest.json")
    p.add_argument("--raw-root", type=Path, default=BENCH / "raw",
                   help="Where egoexo writes annotations + transient videos.")
    p.add_argument("--frames-root", type=Path, default=BENCH / "frames")
    p.add_argument("--egoexo",   type=str,  default="egoexo")
    p.add_argument("--keep-videos", action="store_true")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--num-workers",  type=int, default=16)
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    if not args.subset.exists():
        print(f"[fatal] {args.subset} missing — repo not checked out fully?")
        return 2
    if not args.manifest.exists():
        print(f"[fatal] {args.manifest} missing — repo not checked out fully?")
        return 2

    sub  = json.loads(args.subset.read_text())
    take_uids = sub["take_uids"]
    manifest  = json.loads(args.manifest.read_text())
    print(f"[bench] {len(take_uids)} takes  /  "
          f"{sum(len(c) for u in manifest.values() for c in u.values())} "
          f"frame-evaluations")

    args.frames_root.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    anno_root = args.raw_root / "annotations"

    # ── 1) val annotations ────────────────────────────────────────────────
    val_body_dir = anno_root / "ego_pose" / "val" / "body" / "annotation"
    needs_anno = not val_body_dir.exists() or not any(val_body_dir.glob("*.json"))
    if needs_anno:
        print("[bench] downloading val annotations + metadata ...")
        rc = pull_val_annotations(args.raw_root, args.egoexo)
        if rc != 0:
            print(f"[fatal] egoexo annotations rc={rc}")
            return 3
    else:
        print(f"[bench] annotations already present at {val_body_dir}")

    # ── 2) classify which takes still need video fetch ────────────────────
    needs_fetch: list[str] = []
    take_name_for_uid: dict[str, str] = {}
    for uid in take_uids:
        cp_path = anno_root / "ego_pose" / "val" / "camera_pose" / f"{uid}.json"
        if not cp_path.exists():
            print(f"  [warn] no camera_pose for {uid}; skipping")
            continue
        cp = load_camera_pose(cp_path)
        take_name_for_uid[uid] = cp["take_name"]
        # If every (cam, frame) in the manifest already has its JPG, skip fetch.
        cams = manifest.get(uid, {})
        all_present = True
        for cam_name, fidxs in cams.items():
            if not is_exo_cam(cam_name):
                continue
            for fi in fidxs:
                if not (args.frames_root / uid / cam_name
                        / f"{int(fi):06d}.jpg").exists():
                    all_present = False
                    break
            if not all_present:
                break
        if not all_present:
            needs_fetch.append(uid)
    print(f"[bench] needs_fetch={len(needs_fetch)}  "
          f"already_extracted={len(take_uids) - len(needs_fetch)}")

    # ── 3) batched video download (one egoexo call) ───────────────────────
    if needs_fetch:
        print(f"[bench] downloading {len(needs_fetch)} takes' exo videos "
              f"in one egoexo call (num_workers={args.num_workers}) ...")
        t0 = time.time()
        rc = pull_take_videos_batch(args.raw_root, needs_fetch,
                                    args.egoexo, args.num_workers)
        print(f"[bench] download rc={rc}  ({(time.time()-t0)/60:.1f} min)")
        if rc != 0:
            print("[bench] continuing despite non-zero rc; will extract whatever "
                  "videos landed on disk")

    # ── 4) extract per-manifest frames + cleanup ──────────────────────────
    n_extracted = n_missing = 0
    t0 = time.time()
    for i, uid in enumerate(take_uids):
        if uid not in take_name_for_uid:
            continue
        n_ex, n_miss = extract_frames_for_take(
            uid, args.raw_root, args.frames_root, anno_root,
            manifest.get(uid, {}), args.jpeg_quality)
        n_extracted += n_ex
        n_missing   += n_miss
        print(f"  [{i+1}/{len(take_uids)}] {uid}  "
              f"extracted={n_ex}  missing={n_miss}  "
              f"cum_ok={n_extracted}  cum_miss={n_missing}")
        if not args.keep_videos:
            tk = take_name_for_uid[uid]
            tk_dir = args.raw_root / "takes" / tk
            if tk_dir.exists():
                shutil.rmtree(tk_dir, ignore_errors=True)

    dt = time.time() - t0
    print(f"\n[bench] DONE  extracted={n_extracted}  missing={n_missing}  "
          f"({dt/60:.1f} min)")
    print(f"[bench] frames -> {args.frames_root}")
    print(f"[bench] annotations -> {anno_root}")
    if n_missing > 0:
        print(f"[bench] WARNING: {n_missing} manifest frames could not be "
              f"extracted (video missing or out-of-range)")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
