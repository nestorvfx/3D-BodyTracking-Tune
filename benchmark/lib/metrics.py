"""3D pose metrics: MPJPE (root-relative), PA-MPJPE, PCK3D, AUC.

All inputs are in metres.  Reports in millimetres.
"""
from __future__ import annotations

import numpy as np

from .keypoint_map import HIP_L, HIP_R


def root_center(kp17: np.ndarray) -> np.ndarray:
    """kp17: (..., 17, 3) -> centred copy with mid-hip at origin per-frame.

    NaN-tolerant: if either hip is NaN, the entire frame is left NaN."""
    hip_mid = 0.5 * (kp17[..., HIP_L, :] + kp17[..., HIP_R, :])  # (..., 3)
    return kp17 - hip_mid[..., None, :]


def mpjpe_per_joint(pred17: np.ndarray, gt17: np.ndarray,
                    mask17: np.ndarray) -> np.ndarray:
    """Per-frame, per-joint Euclidean error (metres).  NaN where masked-out
    or where pred is missing.  pred17 / gt17: (N, 17, 3); mask17: (N, 17) bool.
    Returns (N, 17)."""
    err = np.linalg.norm(pred17 - gt17, axis=-1)         # (N, 17)
    err = np.where(mask17, err, np.nan)
    err = np.where(np.isnan(pred17).any(axis=-1), np.nan, err)
    return err


def umeyama(src: np.ndarray, dst: np.ndarray,
            with_scale: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity that aligns `src` -> `dst`.
    Both: (n, 3) with no NaN.

    With `with_scale=True` (default): solve
        argmin_{R,t,s}  || s · R · src + t - dst ||^2
    With `with_scale=False`: rigid-only,
        argmin_{R,t}    ||  R · src + t - dst ||^2
    The rigid-only number tells us what fraction of PA-MPJPE-with-scale was
    actually a scale-fit artefact vs genuine pose error.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = src_c.T @ dst_c / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T)
    if with_scale:
        var_s = (src_c ** 2).sum() / src.shape[0]
        s = (S * np.array([1.0, 1.0, d])).sum() / max(var_s, 1e-12)
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return R, t, s


def pa_mpjpe_per_frame(pred17: np.ndarray, gt17: np.ndarray,
                      mask17: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Procrustes-aligned MPJPE per frame.  Returns (N,) metres or NaN.

    For each frame, fit (R,t,s) on the *visible* joints and apply to the
    full predicted skeleton, then Euclidean error on visible joints.
    `with_scale=False` gives the rigid-only (R+t) variant."""
    N = pred17.shape[0]
    out = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        m = mask17[i] & ~np.isnan(pred17[i]).any(-1)
        if int(m.sum()) < 4:
            continue
        R, t, s = umeyama(pred17[i, m], gt17[i, m], with_scale=with_scale)
        aligned = s * (pred17[i] @ R.T) + t
        err = np.linalg.norm(aligned[m] - gt17[i, m], axis=-1)
        out[i] = err.mean()
    return out


def pa_mpjpe_per_joint_frame(pred17: np.ndarray, gt17: np.ndarray,
                             mask17: np.ndarray,
                             with_scale: bool = True) -> np.ndarray:
    """Per-joint per-frame Procrustes-aligned distance.  Returns (N, 17)
    metres with NaN for masked-out / failed-Procrustes entries."""
    N = pred17.shape[0]
    out = np.full((N, 17), np.nan, dtype=np.float64)
    for i in range(N):
        m = mask17[i] & ~np.isnan(pred17[i]).any(-1)
        if int(m.sum()) < 4:
            continue
        R, t, s = umeyama(pred17[i, m], gt17[i, m], with_scale=with_scale)
        aligned = s * (pred17[i] @ R.T) + t
        err = np.linalg.norm(aligned - gt17[i], axis=-1)
        out[i] = np.where(mask17[i], err, np.nan)
    return out


def pck_curve(errors_mm: np.ndarray, thresholds_mm: np.ndarray) -> np.ndarray:
    """errors_mm: (n,) flattened per-joint errors (NaNs ignored).
    Returns PCK fraction at each threshold."""
    e = errors_mm[~np.isnan(errors_mm)]
    return np.array([(e <= t).mean() for t in thresholds_mm], dtype=np.float64)


def auc(pck_values: np.ndarray, thresholds_mm: np.ndarray) -> float:
    """AUC of PCK curve over [0, max_threshold], normalised to [0,1]."""
    return float(np.trapezoid(pck_values, thresholds_mm) / thresholds_mm[-1])


def aggregate_summary(per_joint_err_m: np.ndarray, mask17: np.ndarray,
                      pa_err_m: np.ndarray) -> dict:
    """per_joint_err_m: (N, 17) metres; pa_err_m: (N,) metres.
    Returns dict with overall + per-joint MPJPE in mm, PCK@50/150 mm, AUC."""
    flat_mm = per_joint_err_m.flatten() * 1000.0
    mpjpe_mm = float(np.nanmean(flat_mm))
    per_joint_mm = (np.nanmean(per_joint_err_m, axis=0) * 1000.0).tolist()
    thresholds_mm = np.linspace(0.0, 200.0, 41)
    pck = pck_curve(flat_mm, thresholds_mm)
    return {
        "mpjpe_mm":       mpjpe_mm,
        "pa_mpjpe_mm":    float(np.nanmean(pa_err_m) * 1000.0),
        "pck50_mm_pct":   100.0 * float(pck_curve(flat_mm, np.array([50.0]))[0]),
        "pck150_mm_pct":  100.0 * float(pck_curve(flat_mm, np.array([150.0]))[0]),
        "auc_0_200":      auc(pck, thresholds_mm),
        "per_joint_mpjpe_mm": dict(zip(
            ["nose","eye_L","eye_R","ear_L","ear_R","shoulder_L","shoulder_R",
             "elbow_L","elbow_R","wrist_L","wrist_R","hip_L","hip_R",
             "knee_L","knee_R","ankle_L","ankle_R"], per_joint_mm)),
        "frames_scored":  int((~np.isnan(per_joint_err_m).all(axis=-1)).sum()),
        "joints_scored":  int(mask17.sum()),
    }
