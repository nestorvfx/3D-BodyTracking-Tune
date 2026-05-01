"""Stream-download + extract Ego-Exo4D **train** split body-pose frames.

Differs from `benchmark/stream_extract.py` (val-only):
  - All ~3072 train takes (vs 24-take diversity-filtered val manifest)
  - All annotated GT frames per take (vs 5-12 sampled)
  - Per-take: download videos → extract every annotated frame → delete videos

Hold-out enforced: any take_uid present in `benchmark/frames_manifest.json`
or `benchmark/subset.json` is REFUSED.  Exits 99 on intersection.

Output: /data/egoexo/frames/<take_uid>/<cam>/<frame_idx>.jpg
        /data/egoexo/manifest_train.jsonl  (one line per (uid, cam, frame))
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
sys.path.insert(0, str(HERE.parent / "benchmark"))
sys.path.insert(0, str(HERE.parent / "training"))

from lib.ego_exo_io import load_body_gt, load_camera_pose, is_exo_cam
from holdout import load_forbidden_uids


def pull_take_videos(out_dir: Path, take_uid: str, egoexo: str) -> int:
    cmd = [egoexo, "-o", str(out_dir),
           "--parts", "downscaled_takes/448",
           "--uids", take_uid,
           "--views", "exo",
           "-y"]
    return subprocess.run(cmd, check=False).returncode


def extract_one(uid: str, raw_root: Path, frames_root: Path,
                anno_root: Path, manifest_lines: list,
                jpeg_quality: int = 90) -> tuple[int, int]:
    cp_path = anno_root / "ego_pose" / "train" / "camera_pose" / f"{uid}.json"
    body_path = anno_root / "ego_pose" / "train" / "body" / "annotation" / f"{uid}.json"
    if not (cp_path.exists() and body_path.exists()):
        return 0, 1
    cp = load_camera_pose(cp_path)
    gt = load_body_gt(body_path)
    take_name = cp["take_name"]
    take_video_dir = (raw_root / "takes" / take_name
                      / "frame_aligned_videos" / "downscaled" / "448")
    if not take_video_dir.exists():
        return 0, 2
    frame_idxs = sorted(gt.keys())
    out_take_dir = frames_root / uid
    out_take_dir.mkdir(parents=True, exist_ok=True)

    n_extracted = 0
    for cam_name in cp["cams"]:
        if not is_exo_cam(cam_name):
            continue
        mp4 = take_video_dir / f"{cam_name}.mp4"
        if not mp4.exists():
            continue
        cam_out = out_take_dir / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            continue
        n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in frame_idxs:
            if fi >= n_frames_video:
                continue
            out_jpg = cam_out / f"{fi:06d}.jpg"
            if out_jpg.exists():
                manifest_lines.append({
                    "take_uid": uid, "cam": cam_name, "frame": fi,
                    "image_path": str(out_jpg.relative_to(frames_root)),
                })
                n_extracted += 1
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(out_jpg), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            manifest_lines.append({
                "take_uid": uid, "cam": cam_name, "frame": fi,
                "image_path": str(out_jpg.relative_to(frames_root)),
            })
            n_extracted += 1
        cap.release()
    return n_extracted, 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--anno-root",   type=Path, required=True,
                   help="Directory containing ego_pose/train/{body,camera_pose}.")
    p.add_argument("--raw-root",    type=Path, default=Path("/data/egoexo"))
    p.add_argument("--frames-root", type=Path,
                   default=Path("/data/egoexo/frames"))
    p.add_argument("--manifest-out", type=Path,
                   default=Path("/data/egoexo/manifest_train.jsonl"))
    p.add_argument("--egoexo",      type=str, default="egoexo")
    p.add_argument("--limit-takes", type=int, default=None,
                   help="Cap takes (debugging only).")
    p.add_argument("--keep-videos", action="store_true")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--forbidden-manifest", type=Path,
                   default=HERE.parent / "benchmark" / "frames_manifest.json")
    p.add_argument("--forbidden-subset",   type=Path,
                   default=HERE.parent / "benchmark" / "subset.json")
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    # ── List all train uids that have body GT ─────────────────────────
    body_dir = args.anno_root / "ego_pose" / "train" / "body" / "annotation"
    if not body_dir.exists():
        print(f"[fatal] {body_dir} missing — pull annotations first via egoexo "
              f"--parts annotations --benchmarks egopose --splits train")
        return 2
    train_uids = sorted(p.stem for p in body_dir.glob("*.json"))
    print(f"[extract] {len(train_uids)} train body-GT takes")

    # ── Hold-out check ────────────────────────────────────────────────
    forbidden = load_forbidden_uids(args.forbidden_manifest, args.forbidden_subset)
    leak = forbidden.intersection(set(train_uids))
    if leak:
        print(f"[FATAL] {len(leak)} train uids are in held-out set: "
              f"{sorted(leak)[:5]} ...", file=sys.stderr)
        return 99
    print(f"[extract] hold-out OK: 0/{len(train_uids)} in forbidden corpus "
          f"({len(forbidden)} forbidden)")

    if args.limit_takes:
        train_uids = train_uids[: args.limit_takes]
    args.frames_root.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    manifest_lines: list[dict] = []
    n_done = n_skip = n_total_frames = 0
    t0 = time.time()
    for i, uid in enumerate(train_uids):
        cp_path = args.anno_root / "ego_pose" / "train" / "camera_pose" / f"{uid}.json"
        if not cp_path.exists():
            n_skip += 1
            continue
        cp = load_camera_pose(cp_path)
        take_name = cp["take_name"]
        take_video_dir = (args.raw_root / "takes" / take_name
                          / "frame_aligned_videos" / "downscaled" / "448")

        # Already extracted?
        already_extracted = (args.frames_root / uid).exists() and any(
            (args.frames_root / uid / cam).is_dir()
            for cam in cp["cams"] if is_exo_cam(cam))
        if already_extracted:
            # Re-walk to populate manifest from disk
            for cam in cp["cams"]:
                if not is_exo_cam(cam): continue
                cam_dir = args.frames_root / uid / cam
                if not cam_dir.is_dir(): continue
                for jpg in sorted(cam_dir.glob("*.jpg")):
                    manifest_lines.append({
                        "take_uid": uid, "cam": cam,
                        "frame": int(jpg.stem),
                        "image_path": str(jpg.relative_to(args.frames_root)),
                    })
                    n_total_frames += 1
            n_done += 1
            continue

        # Pull videos
        if not take_video_dir.exists():
            rc = pull_take_videos(args.raw_root, uid, args.egoexo)
            if rc != 0:
                print(f"  [warn] egoexo rc={rc} for {uid}; skipping")
                n_skip += 1
                continue

        # Extract
        try:
            n_ex, err = extract_one(uid, args.raw_root, args.frames_root,
                                    args.anno_root, manifest_lines,
                                    jpeg_quality=args.jpeg_quality)
        except Exception as e:
            print(f"  [error] extract {uid}: {e}")
            n_skip += 1
            continue
        if err > 0:
            n_skip += 1
            continue
        n_total_frames += n_ex
        n_done += 1

        # Cleanup
        if not args.keep_videos:
            take_dir = args.raw_root / "takes" / take_name
            if take_dir.exists():
                shutil.rmtree(take_dir, ignore_errors=True)

        if (i + 1) % 25 == 0 or (i + 1) == len(train_uids):
            dt = time.time() - t0
            print(f"  [{i+1}/{len(train_uids)}] done={n_done} skip={n_skip}  "
                  f"frames={n_total_frames}  ({dt/60:.1f} min, "
                  f"~{n_total_frames/max(dt,1):.0f} fps)")

    # Write manifest
    with args.manifest_out.open("w") as fh:
        for r in manifest_lines:
            fh.write(json.dumps(r) + "\n")
    print(f"\n[done] {n_done} takes, {n_total_frames} frames "
          f"-> {args.frames_root}")
    print(f"[done] manifest: {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
