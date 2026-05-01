"""Bulk-batched download + extract Ego-Exo4D **train** split body-pose frames.

Differs from `benchmark/stream_extract.py` (val-only):
  - All ~3072 train takes (vs 24-take diversity-filtered val manifest)
  - All annotated GT frames per take (vs 5-12 sampled)
  - **Batched download**: N takes per single `egoexo` invocation (vs 1-per-call).
    Default batch=200 → ~30 GB peak transient (fits 512 GB box with margin).
    50× faster than per-take loop because the egoexo CLI's ~30-sec metadata
    fetch is amortised across each batch instead of paid per take.
  - **Bumped --num_workers** to 32 (egoexo default is 15) for more S3 parallelism.

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


def pull_take_videos_batch(out_dir: Path, take_uids: list[str],
                            egoexo: str, num_workers: int = 32) -> int:
    """Download exo videos for MANY take_uids in a single egoexo invocation.

    The egoexo CLI accepts a list of --uids and uses --num_workers (default 15)
    for parallel S3 fetches.  One metadata fetch per call (~30 sec) amortised
    across the whole batch — orders of magnitude faster than per-take loop."""
    cmd = [egoexo, "-o", str(out_dir),
           "--parts", "downscaled_takes/448",
           "--uids", *take_uids,
           "--views", "exo",
           "--num_workers", str(num_workers),
           "-y"]
    return subprocess.run(cmd, check=False).returncode


def pull_take_videos(out_dir: Path, take_uid: str, egoexo: str) -> int:
    """Single-take wrapper, kept for backward compat."""
    return pull_take_videos_batch(out_dir, [take_uid], egoexo, num_workers=15)


def extract_one(uid: str, raw_root: Path, frames_root: Path,
                anno_root: Path,
                jpeg_quality: int = 90) -> tuple[int, int, list]:
    """Returns (n_extracted, err_code, manifest_entries).
    err_code: 0=ok, 1=missing GT/cp, 2=video dir missing.
    Returns the manifest entries instead of mutating a shared list, so
    the function is process-pool safe."""
    manifest_entries: list = []
    cp_path = anno_root / "ego_pose" / "train" / "camera_pose" / f"{uid}.json"
    body_path = anno_root / "ego_pose" / "train" / "body" / "annotation" / f"{uid}.json"
    if not (cp_path.exists() and body_path.exists()):
        return 0, 1, manifest_entries
    cp = load_camera_pose(cp_path)
    gt = load_body_gt(body_path)
    take_name = cp["take_name"]
    take_video_dir = (raw_root / "takes" / take_name
                      / "frame_aligned_videos" / "downscaled" / "448")
    if not take_video_dir.exists():
        return 0, 2, manifest_entries
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
                manifest_entries.append({
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
            manifest_entries.append({
                "take_uid": uid, "cam": cam_name, "frame": fi,
                "image_path": str(out_jpg.relative_to(frames_root)),
            })
            n_extracted += 1
        cap.release()
    return n_extracted, 0, manifest_entries


def _extract_one_worker(args_tuple):
    """ProcessPool wrapper.  Imports happen lazily per-worker."""
    uid, raw_root, frames_root, anno_root, jpeg_quality = args_tuple
    try:
        n_ex, err, lines = extract_one(uid, raw_root, frames_root,
                                       anno_root, jpeg_quality)
        return uid, n_ex, err, lines
    except Exception as e:
        return uid, 0, 99, [f"[error] {uid}: {e}"]


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
    p.add_argument("--batch-size", type=int, default=200,
                   help="Number of takes to download per single egoexo CLI "
                        "invocation.  Larger = fewer metadata fetches = faster, "
                        "but higher peak transient disk (~150 MB per take).  "
                        "Default 200 → ~30 GB peak per batch, fits a 512 GB box.")
    p.add_argument("--num-workers", type=int, default=32,
                   help="Parallel S3 download workers per egoexo call. "
                        "Default 32 (egoexo's own default is 15).")
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

    # ── First pass: classify uids by status (already-extracted / needs-fetch) ─
    needs_fetch: list[str] = []
    already_extracted_uids: list[str] = []
    no_camera_pose_uids: list[str] = []
    take_name_for_uid: dict[str, str] = {}
    for uid in train_uids:
        cp_path = args.anno_root / "ego_pose" / "train" / "camera_pose" / f"{uid}.json"
        if not cp_path.exists():
            no_camera_pose_uids.append(uid)
            continue
        cp = load_camera_pose(cp_path)
        take_name_for_uid[uid] = cp["take_name"]
        already = (args.frames_root / uid).exists() and any(
            (args.frames_root / uid / cam).is_dir()
            for cam in cp["cams"] if is_exo_cam(cam))
        if already:
            already_extracted_uids.append(uid)
        else:
            needs_fetch.append(uid)
    n_skip = len(no_camera_pose_uids)
    print(f"[extract] needs_fetch={len(needs_fetch)} already_extracted={len(already_extracted_uids)} "
          f"no_camera_pose={len(no_camera_pose_uids)}")

    # ── Re-walk already-extracted takes to populate the manifest ────────
    for uid in already_extracted_uids:
        cp = load_camera_pose(args.anno_root / "ego_pose" / "train" / "camera_pose" / f"{uid}.json")
        for cam in cp["cams"]:
            if not is_exo_cam(cam): continue
            cam_dir = args.frames_root / uid / cam
            if not cam_dir.is_dir(): continue
            for jpg in sorted(cam_dir.glob("*.jpg")):
                manifest_lines.append({
                    "take_uid": uid, "cam": cam, "frame": int(jpg.stem),
                    "image_path": str(jpg.relative_to(args.frames_root)),
                })
                n_total_frames += 1
        n_done += 1

    # ── Process needs_fetch in batches ────────────────────────────────
    # Extract phase parallelism: ProcessPoolExecutor across takes.  Each
    # take is independent (its own MP4s, JSONs, output dir), so this scales
    # near-linearly with CPU count.  Default = min(cpus, 16) — 200-take
    # batch on a 16-core box drops from ~5 min single-process to ~30 sec.
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    EXTRACT_NPROC = int(os.environ.get(
        "EGOEXO_EXTRACT_NPROC", min(os.cpu_count() or 4, 16)))
    print(f"[extract] extract pool: {EXTRACT_NPROC} workers")

    BATCH = max(1, args.batch_size)
    n_batches = (len(needs_fetch) + BATCH - 1) // BATCH
    for batch_idx in range(n_batches):
        batch_uids = needs_fetch[batch_idx * BATCH : (batch_idx + 1) * BATCH]
        b_start = time.time()
        print(f"\n=== batch {batch_idx+1}/{n_batches}  ({len(batch_uids)} takes) ===")
        rc = pull_take_videos_batch(args.raw_root, batch_uids,
                                    args.egoexo, num_workers=args.num_workers)
        if rc != 0:
            print(f"  [warn] egoexo batch rc={rc}; will still attempt to "
                  f"extract whatever videos landed on disk")
        d_dl = time.time() - b_start
        ex_start = time.time()

        # Extract every take in parallel (videos on disk → independent work).
        n_batch_extracted_takes = 0
        n_batch_extracted_frames = 0
        worker_args = [(uid, args.raw_root, args.frames_root,
                        args.anno_root, args.jpeg_quality)
                       for uid in batch_uids]
        n_done_in_batch = 0
        with ProcessPoolExecutor(max_workers=EXTRACT_NPROC) as ex:
            futures = {ex.submit(_extract_one_worker, wa): wa[0]
                       for wa in worker_args}
            for fut in as_completed(futures):
                uid, n_ex, err, lines = fut.result()
                if err == 99:
                    print(f"  {lines[0] if lines else '[error] '+uid}")
                    n_skip += 1
                    continue
                if err > 0:
                    n_skip += 1
                    continue
                manifest_lines.extend(lines)
                n_total_frames += n_ex
                n_done += 1
                n_batch_extracted_takes += 1
                n_batch_extracted_frames += n_ex
                n_done_in_batch += 1
                # Per-take progress every 25 takes (one line, not flooding)
                if n_done_in_batch % 25 == 0:
                    print(f"    extract progress: {n_done_in_batch}/"
                          f"{len(batch_uids)} takes  +{n_batch_extracted_frames} frames")
        d_ex = time.time() - ex_start

        # Cleanup the whole batch's videos (bounds peak disk to ~BATCH × ~150 MB)
        if not args.keep_videos:
            for uid in batch_uids:
                tk_name = take_name_for_uid.get(uid)
                if not tk_name:
                    continue
                take_dir = args.raw_root / "takes" / tk_name
                if take_dir.exists():
                    shutil.rmtree(take_dir, ignore_errors=True)

        d_total = time.time() - b_start
        cum_dt = time.time() - t0
        print(f"  [batch {batch_idx+1}/{n_batches}]  "
              f"takes={n_batch_extracted_takes}/{len(batch_uids)}  "
              f"frames+={n_batch_extracted_frames}  "
              f"cum_done={n_done} cum_frames={n_total_frames}  "
              f"(dl {d_dl:.0f}s, ex {d_ex:.0f}s, total {d_total:.0f}s, "
              f"{cum_dt/60:.1f} min cum)")

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
