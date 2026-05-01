"""World <-> camera <-> image projection for Ego-Exo4D exo cameras.

Ego-Exo4D ships static [K | Rt | dist] per take per camera (GoPros are
stationary).  All 3D GT is in the global/world frame.

Conventions:
  X_world: (..., 3) world-frame metres
  X_cam:   (..., 3) camera-frame metres, X_cam = R · X_world + t
  K:       (3,3) pinhole intrinsics
  Rt:      (3,4) [R | t]
  dist:    (k,) distortion coefficients (Brown-Conrady k1,k2,p1,p2[,k3,...])
"""
from __future__ import annotations

import numpy as np


def world_to_cam(X_world: np.ndarray, Rt: np.ndarray) -> np.ndarray:
    """X_world: (..., 3) -> X_cam: (..., 3) under static [R|t]."""
    R = Rt[:, :3]
    t = Rt[:, 3]
    return X_world @ R.T + t


def project_pinhole(X_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole projection (no distortion).  Returns (..., 2) image-plane px."""
    z = X_cam[..., 2:3]
    z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    xy = X_cam[..., :2] / z
    out = xy @ K[:2, :2].T + K[:2, 2]
    return out


def project_with_distortion(X_cam: np.ndarray, K: np.ndarray,
                            dist: np.ndarray) -> np.ndarray:
    """Brown-Conrady distortion projection.  cv2.projectPoints under the hood
    so we match OpenCV exactly.  Falls back to pinhole if dist is empty."""
    import cv2
    if dist is None or len(dist) == 0:
        return project_pinhole(X_cam, K)
    pts = X_cam.reshape(-1, 1, 3).astype(np.float64)
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    img, _ = cv2.projectPoints(pts, rvec, tvec, K.astype(np.float64),
                               dist.astype(np.float64))
    return img.reshape(*X_cam.shape[:-1], 2)
