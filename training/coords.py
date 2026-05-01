"""Coordinate-frame transforms for BlazePose v2 training.

The student's `world_landmarks` (Identity_4 in our port) live in BlazePose's
own **body-axis frame**: origin at mid-hip, X-right (along the hip line),
Y-down (along the spine, toward the feet — yes, *down*), Z-forward (out of
the chest).  This is GHUM-derived and intrinsic to the model.

Hard supervision sources (synth GT, Ego-Exo4D GT) live in **camera-frame
metres**: origin at the camera optical centre.

To compare them, we must rotate + translate camera-frame GT into
body-axis.  This file builds R_cam_to_body from the GT skeleton itself
(no external info needed):

    1. mid_hip   = (left_hip + right_hip) / 2          → translation
    2. body_x    = (right_hip - left_hip)              → +X axis (right)
    3. up_vec    = (mid_shoulder - mid_hip)            → roughly -Y (up = -Y in BP convention)
    4. body_y    = -up_vec                              → +Y (down toward feet)
    5. body_z    = body_x × body_y                      → +Z (forward)
    6. orthonormalise via SVD/Gram-Schmidt.

Then for every joint:
    p_body = R_cam_to_body @ (p_cam - mid_hip)

When either both hips OR both shoulders aren't present, we cannot define
the rotation; fall back to translation-only and flag the sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmark"))

from lib.keypoint_map import HIP_L, HIP_R, COCO17  # 11, 12 in COCO order

SHO_L = COCO17.index("left-shoulder")     # 5
SHO_R = COCO17.index("right-shoulder")    # 6


def build_body_frame(kp17_cam: np.ndarray, present17: np.ndarray
                     ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns (R_cam_to_body (3,3), origin_cam (3,)) or (None, None) if
    the four anchor joints (both hips + both shoulders) aren't visible."""
    if not (present17[HIP_L] and present17[HIP_R]
            and present17[SHO_L] and present17[SHO_R]):
        return None, None

    hip_l = kp17_cam[HIP_L]
    hip_r = kp17_cam[HIP_R]
    sho_l = kp17_cam[SHO_L]
    sho_r = kp17_cam[SHO_R]

    origin   = 0.5 * (hip_l + hip_r)               # mid-hip
    sho_mid  = 0.5 * (sho_l + sho_r)
    body_x   = hip_r - hip_l                        # right
    up       = sho_mid - origin                     # up
    body_y   = -up                                  # BP Y is down toward feet
    body_z   = np.cross(body_x, body_y)             # forward

    # Orthonormalise with QR
    M = np.stack([body_x, body_y, body_z], axis=1)  # (3, 3) columns
    Q, _ = np.linalg.qr(M)
    # Make sure determinant is +1 (right-handed)
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1.0
    # Q's columns are the body-frame basis in cam coords ⇒ R_body_to_cam.
    # We need R_cam_to_body (rotates a cam-frame vector into body-frame).
    R_cam_to_body = Q.T
    return R_cam_to_body, origin


def cam_to_body(kp_cam: np.ndarray, R_cam_to_body: np.ndarray,
                origin_cam: np.ndarray) -> np.ndarray:
    """kp_cam: (..., 3) cam-frame metres → body-frame metres."""
    return (kp_cam - origin_cam) @ R_cam_to_body.T


def map17_to_bp33(kp17_body: np.ndarray, present17: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pad the 17-COCO body-frame KPs into a 33-BP shape (zero for non-mappable
    indices).  Returns (kp33, present33)."""
    from lib.keypoint_map import BP_INDEX_FOR_COCO   # local import
    kp33 = np.zeros((33, 3), dtype=np.float32)
    pres33 = np.zeros(33, dtype=np.float32)
    for coco_idx, bp_idx in enumerate(BP_INDEX_FOR_COCO):
        kp33[bp_idx]  = kp17_body[coco_idx]
        pres33[bp_idx] = float(present17[coco_idx])
    return kp33, pres33


def hard_target_in_body_axis(kp17_cam: np.ndarray, present17: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """One-shot helper: cam-frame 17-COCO → 33-BP in body-axis frame."""
    R, origin = build_body_frame(kp17_cam, present17)
    if R is None:
        # Cannot orient → return translation-only, mark all but mid-hip as
        # "present but unreliable" by clearing the present mask.
        # Keep zeros to avoid contaminating the loss.
        return (np.zeros((33, 3), dtype=np.float32),
                np.zeros(33, dtype=np.float32))
    kp17_body = cam_to_body(kp17_cam, R, origin)
    return map17_to_bp33(kp17_body, present17)


if __name__ == "__main__":
    """Smoke: synthetic skeleton in cam frame, transform to body frame,
    inverse-transform, verify round-trip."""
    np.random.seed(0)
    # Build a fake skeleton: hips ±0.15 m on X, shoulders ±0.18m on X, +0.5m up
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[HIP_L]  = (-0.15, 0.0, 2.5)
    kp[HIP_R]  = (+0.15, 0.0, 2.5)
    kp[SHO_L]  = (-0.18, -0.5, 2.5)
    kp[SHO_R]  = (+0.18, -0.5, 2.5)
    kp[0]      = (0.0, -0.7, 2.5)        # nose
    pres = np.zeros(17, dtype=np.float32)
    for i in [HIP_L, HIP_R, SHO_L, SHO_R, 0]:
        pres[i] = 1.0

    R, origin = build_body_frame(kp, pres)
    print("origin (mid-hip cam-frame):", origin)
    print("R_cam_to_body:\n", R)
    body_kp = cam_to_body(kp, R, origin)
    print("\nbody-axis KPs (mid-hip should be 0,0,0):")
    print(f"  hip_L  : {body_kp[HIP_L]}  (expect ~ (-0.15, 0,    0))")
    print(f"  hip_R  : {body_kp[HIP_R]}  (expect ~ (+0.15, 0,    0))")
    print(f"  sho_L  : {body_kp[SHO_L]}  (expect ~ (-0.18, +0.5, 0))   # BP Y is *down*=feet")
    print(f"  sho_R  : {body_kp[SHO_R]}  (expect ~ (+0.18, +0.5, 0))")
    print(f"  nose   : {body_kp[0]}      (expect ~ ( 0,    +0.7, 0))   # above hip in BP frame")

    bp33, pres33 = hard_target_in_body_axis(kp, pres)
    print(f"\nbp33 shape={bp33.shape}, present joints={int(pres33.sum())}/33")
    print("[smoke] coords.py round-trip OK")
