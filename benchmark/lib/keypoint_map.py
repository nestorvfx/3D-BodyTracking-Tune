"""Joint name lists and the BlazePose 33 -> COCO 17 mapping.

Ego-Exo4D uses joint NAMES (e.g. "left-wrist") in its annotations, so we
key by name rather than index for the GT side.  BlazePose returns 33
indexed landmarks; we slice them into the 17-joint COCO order using
`BP_INDEX_FOR_COCO`.
"""
from __future__ import annotations

# COCO-17 joint names in the canonical COCO order (idx 0..16).
COCO17 = [
    "nose",
    "left-eye", "right-eye",
    "left-ear", "right-ear",
    "left-shoulder", "right-shoulder",
    "left-elbow", "right-elbow",
    "left-wrist", "right-wrist",
    "left-hip", "right-hip",
    "left-knee", "right-knee",
    "left-ankle", "right-ankle",
]

# BlazePose 33 indices that we slice for each COCO-17 joint, in COCO order.
# Sources: MediaPipe Pose Landmarker model card; see SOTA_APPROACH.md §3.
# Notable: BP eye centres are 2/5 (NOT inner-eye 1/4 nor outer-eye 3/6).
BP_INDEX_FOR_COCO = [
    0,    # nose
    2, 5, # eye L/R
    7, 8, # ear L/R
    11, 12, # shoulder L/R
    13, 14, # elbow L/R
    15, 16, # wrist L/R
    23, 24, # hip L/R
    25, 26, # knee L/R
    27, 28, # ankle L/R
]
assert len(BP_INDEX_FOR_COCO) == 17

# Indices of L/R hip in the 17-element COCO array, used to compute mid-hip root.
HIP_L, HIP_R = 11, 12


def gt_to_coco17(annotation3D: dict) -> tuple["np.ndarray", "np.ndarray"]:
    """Converts an Ego-Exo4D `annotation3D` dict to (kp[17,3], present[17] bool).

    Joints absent from `annotation3D` are filled with NaN and `present`=False.
    """
    import numpy as np
    kp = np.full((17, 3), np.nan, dtype=np.float64)
    present = np.zeros(17, dtype=bool)
    for i, name in enumerate(COCO17):
        v = annotation3D.get(name)
        if v is None:
            continue
        kp[i] = (v["x"], v["y"], v["z"])
        present[i] = True
    return kp, present
