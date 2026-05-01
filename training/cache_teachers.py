"""Pre-cache multi-teacher outputs to disk so the dataloader can read them
in O(ms) instead of running mediapipe live during training.

Per frame:
  - Heavy: world33 + img33 (essential body teacher)
  - Hand:  world_left, world_right, img_left, img_right (optional)
  - Face:  img_face (478 landmarks, optional)

Output schema: one .npz per frame_id under <out_dir>/<frame_id>.npz
              + a global manifest.json with {frame_id: {"teachers": [...]}}

Designed for Vast.ai: runs on full Ego-Exo4D + synth corpus once at the
top of training (4-8 hr); subsequent training epochs just load the npz.
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

from teachers import HeavyTeacher, HandTeacher, FaceTeacher, download_teacher_weights


def cache_one(img_rgb: np.ndarray, heavy=None, hand=None, face=None) -> dict:
    """Cache teacher outputs.  When both Heavy + Hand are run, also compute
    the body-axis-aligned hand fingertip positions for BP indices 17-22:
       BP 17/18 (pinky tip L/R) ← MediaPipe Hand idx 20
       BP 19/20 (index tip L/R) ← MediaPipe Hand idx 8
       BP 21/22 (thumb tip L/R) ← MediaPipe Hand idx 4

    Hand Landmarker emits world coords in a wrist-relative frame
    (origin = wrist).  Body-axis approximation: fingertip_body =
    BP_wrist_body + (HL_world[tip] - HL_world[0]).  Treats hand-frame
    as ≈ body-frame, valid for typical poses (hand hangs naturally
    aligned with the body); degrades for unusual hand orientations.
    """
    out: dict[str, np.ndarray] = {}
    if heavy is not None:
        h = heavy(img_rgb)
        if h is not None:
            out["world33"] = h["world33"]
            out["img33"]   = h["img33"]
    if hand is not None:
        hh = hand(img_rgb)
        if hh is not None:
            for k, v in hh.items():
                out[f"hand_{k}"] = v
            # Build body-axis hand targets if Heavy also fired
            if "world33" in out:
                _build_hand_body(out)
    if face is not None:
        f = face(img_rgb)
        if f is not None:
            out["face"] = f["img_face"]
    return out


def _build_hand_body(out: dict) -> None:
    """Populate `out["hand_bp33_xyz_body"]` / `["hand_bp33_present"]` with
    BP indices 17-22 set to body-axis-aligned fingertip positions.

    SOTA hand-frame alignment (per audit gap 1):
      1. Estimate forearm direction in body-axis from Heavy's
         elbow→wrist vector  → defines hand_y (palm-towards-fingers).
      2. Cross with body_x = (1, 0, 0)  → hand_z (out-of-palm).
      3. Cross hand_y × hand_z  → hand_x (across palm).
      4. Build R_hand_to_body, rotate Hand Landmarker's wrist-relative
         fingertip into body frame, then translate by Heavy's wrist.

    Falls back to translation-only when the elbow is missing.  Captures
    the major rotation that broke the prior approximation at >30° wrist
    roll (the dominant Hand-KD failure mode per the audit).
    """
    world33 = out["world33"]
    bp33_xyz   = np.zeros((33, 3), dtype=np.float32)
    bp33_present = np.zeros(33, dtype=np.float32)
    fingertip_to_bp = {
        4:  (21, 22),   # thumb (L, R)
        8:  (19, 20),   # index
        20: (17, 18),   # pinky
    }
    BP_WRIST_L, BP_WRIST_R, BP_ELBOW_L, BP_ELBOW_R = 15, 16, 13, 14
    wrist_l = world33[BP_WRIST_L, :3]; wrist_r = world33[BP_WRIST_R, :3]
    elbow_l = world33[BP_ELBOW_L, :3]; elbow_r = world33[BP_ELBOW_R, :3]

    def _hand_to_body_R(elbow, wrist):
        """Build R_hand_to_body from forearm direction.  Returns identity
        on degenerate input."""
        forearm = wrist - elbow
        n = np.linalg.norm(forearm)
        if n < 1e-3:
            return np.eye(3, dtype=np.float32)
        hand_y = forearm / n                                # palm-up direction
        body_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hand_z = np.cross(hand_y, body_x)
        nz = np.linalg.norm(hand_z)
        if nz < 1e-3:
            # forearm parallel to body_x; pick body_y instead
            body_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            hand_z = np.cross(hand_y, body_y)
            nz = np.linalg.norm(hand_z)
            if nz < 1e-3:
                return np.eye(3, dtype=np.float32)
        hand_z /= nz
        hand_x = np.cross(hand_y, hand_z)
        return np.column_stack([hand_x, hand_y, hand_z]).astype(np.float32)

    for side, wrist_anchor, elbow in [
        ("left",  wrist_l, elbow_l),
        ("right", wrist_r, elbow_r),
    ]:
        key = f"hand_world_{side}"
        if key not in out:
            continue
        hand_world = out[key]                              # (21, 3)
        R = _hand_to_body_R(elbow, wrist_anchor)
        for tip_idx, (bp_l, bp_r) in fingertip_to_bp.items():
            if tip_idx >= len(hand_world):
                continue
            tip_local = hand_world[tip_idx] - hand_world[0]   # wrist-relative
            tip_body  = wrist_anchor + R @ tip_local
            bp = bp_l if side == "left" else bp_r
            bp33_xyz[bp]   = tip_body
            bp33_present[bp] = 1.0
    if bp33_present.sum() > 0:
        out["hand_bp33_xyz_body"]   = bp33_xyz
        out["hand_bp33_present"]    = bp33_present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True,
                    help="Directory with images to cache teachers on. "
                         "Either a flat dir of *.jpg/*.png or a labels.jsonl-pointed source.")
    ap.add_argument("--labels-jsonl", type=Path, default=None,
                    help="If set, iterate from labels.jsonl (synth-style); "
                         "image_rel field is the relative path under source-dir.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--heavy-task", type=Path,
                    default=HERE.parent / "assets" / "pose_landmarker_heavy.task")
    ap.add_argument("--teacher-dir", type=Path,
                    default=HERE.parent / "assets" / "teachers")
    ap.add_argument("--include-hand", action="store_true")
    ap.add_argument("--include-face", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = download_teacher_weights(args.teacher_dir)
    heavy = HeavyTeacher(args.heavy_task)
    hand  = HandTeacher(paths["hand"]) if args.include_hand else None
    face  = FaceTeacher(paths["face"]) if args.include_face else None

    # Build the work list
    if args.labels_jsonl is not None:
        records = []
        with args.labels_jsonl.open() as fh:
            for ln in fh:
                records.append(json.loads(ln))
        if args.limit:
            records = records[: args.limit]
        items = [(r["id"], args.source_dir / r["image_rel"]) for r in records]
    else:
        # Walk all images in source-dir
        exts = {".jpg", ".jpeg", ".png"}
        files = sorted(p for p in args.source_dir.rglob("*") if p.suffix.lower() in exts)
        if args.limit:
            files = files[: args.limit]
        items = [(p.stem, p) for p in files]
    print(f"[cache] {len(items)} frames to process")

    n_done = n_skip = n_no_detect = 0
    t0 = time.time()
    manifest: dict[str, dict] = {}
    for i, (fid, img_path) in enumerate(items):
        out_npz = args.out_dir / f"{fid}.npz"
        if out_npz.exists():
            n_skip += 1
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = cache_one(img_rgb, heavy=heavy, hand=hand, face=face)
        if not out:
            n_no_detect += 1
            continue
        np.savez_compressed(out_npz, **out)
        manifest[fid] = {"teachers": list(out.keys()),
                         "image_path": str(img_path.relative_to(args.source_dir))}
        n_done += 1
        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            rate = n_done / max(dt, 1e-6)
            eta = (len(items) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(items)}] done={n_done} skip={n_skip} "
                  f"no_detect={n_no_detect}  rate={rate:.1f} fps  ETA={eta/60:.1f}min")

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    heavy.close()
    if hand: hand.close()
    if face: face.close()
    print(f"\n[cache] {n_done} cached, {n_skip} skipped (already on disk), "
          f"{n_no_detect} no-detection")
    print(f"[cache] manifest -> {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
