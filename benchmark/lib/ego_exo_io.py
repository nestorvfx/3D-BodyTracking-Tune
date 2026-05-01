"""Loaders for Ego-Exo4D body-pose annotations.

Schema (verified by direct inspection of the v2 release):
- annotations/ego_pose/val/body/annotation/<take_uid>.json:
    { "<frame_idx>": [ { "annotation3D": {<joint_name>: {x,y,z,num_views_for_3d}, ...},
                         "annotation2D": {<cam_name>: {<joint_name>: {x,y,placement}, ...}, ...} } ] }
  Joints absent from a frame's annotation3D are occluded/unannotated (no explicit visibility flag).
- annotations/ego_pose/val/camera_pose/<take_uid>.json:
    { "metadata": {"take_name": str, "take_uid": str},
      "<cam_name>": { "camera_intrinsics": [[3x3]],
                      "camera_extrinsics": [[3x4]],
                      "distortion_coeffs": [k1,k2,p1,p2] } }
  Camera intrinsics/extrinsics are STATIC per take (GoPros are stationary).
  Camera names vary: cam01..cam05, gp01..gp05, plus aria0{1,2}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_body_gt(path: Path) -> dict[int, dict]:
    """Returns {frame_idx (int) -> entry-dict (the [0] of the per-frame list)}."""
    with open(path) as fh:
        raw = json.load(fh)
    out: dict[int, dict] = {}
    for k, v in raw.items():
        # Each frame is a list (single-person body GT → length 1).
        if isinstance(v, list) and len(v) > 0:
            out[int(k)] = v[0]
        elif isinstance(v, dict):
            out[int(k)] = v
    return out


def load_camera_pose(path: Path) -> dict[str, Any]:
    """Returns {take_name, take_uid, cams: {cam_name -> {K(3,3), Rt(3,4), dist}}}.

    Only **static** cameras (exo GoPros) are loaded.  Aria stores
    `camera_extrinsics` as a per-frame dict because the headset moves;
    we skip those because the benchmark uses exo views only.
    """
    with open(path) as fh:
        raw = json.load(fh)
    take_name = raw["metadata"]["take_name"]
    take_uid = raw["metadata"]["take_uid"]
    cams = {}
    for cam_name, cdat in raw.items():
        if cam_name == "metadata":
            continue
        ce = cdat.get("camera_extrinsics")
        # Static-extrinsics cameras have a list-of-lists; moving (aria) ones
        # have a per-frame dict keyed by frame index.
        if not isinstance(ce, list):
            continue
        K  = np.asarray(cdat["camera_intrinsics"], dtype=np.float64)
        Rt = np.asarray(ce,                         dtype=np.float64)
        d  = np.asarray(cdat.get("distortion_coeffs", []), dtype=np.float64)
        cams[cam_name] = {"K": K, "Rt": Rt, "dist": d}
    return {"take_name": take_name, "take_uid": take_uid, "cams": cams}


def is_exo_cam(name: str) -> bool:
    return name.startswith("cam") or name.startswith("gp")


def list_body_takes(annotations_root: Path) -> list[str]:
    """Returns the take_uids that have body GT in the val split."""
    bdir = annotations_root / "ego_pose" / "val" / "body" / "annotation"
    return sorted(p.stem for p in bdir.glob("*.json"))


def load_splits(annotations_root: Path) -> dict[str, str]:
    """splits.json: {take_uid -> 'train'|'val'|'test'}."""
    path = annotations_root / "splits.json"
    with open(path) as fh:
        return json.load(fh)["take_uid_to_split"]
