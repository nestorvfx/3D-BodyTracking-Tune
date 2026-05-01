"""Teacher wrappers for multi-teacher KD.

Each teacher exposes a uniform `__call__(img_rgb_uint8) -> dict` returning
the supervisory tensors we'll use for distillation.

- BlazePose Heavy: full 33-KP body in body-axis frame + visibility (Apache-2.0)
- MediaPipe Hand Landmarker: 21 KPs/hand (Apache-2.0)        [TODO model dl]
- MediaPipe Face Mesh: 478 face landmarks (Apache-2.0)        [TODO model dl]

For training-time efficiency, the typical flow is:
    teacher = HeavyTeacher(task_path)
    out = teacher(img_uint8)            # one call, ~50-100 ms on CPU
    if out is not None:
        save({"world33": out["world33"], "img33": out["img33"], ...})

i.e., you should pre-cache teacher outputs to disk before training (see
`training/cache_teachers.py`) and just load them in the dataloader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class HeavyTeacher:
    """BlazePose Heavy run via MediaPipe Tasks (no port — we use the .task
    directly through the official runtime).
    """

    def __init__(self, task_path: Path, use_gpu: bool = False):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        delegate = (mp_python.BaseOptions.Delegate.GPU if use_gpu
                    else mp_python.BaseOptions.Delegate.CPU)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(task_path), delegate=delegate),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.lm = mp_vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, img_rgb_uint8: np.ndarray) -> dict | None:
        """img_rgb_uint8: (H, W, 3) uint8 RGB.  Returns dict or None."""
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=img_rgb_uint8)
        res = self.lm.detect(mp_img)
        if not res.pose_world_landmarks:
            return None
        # world_landmarks: 33 KPs in metric metres, BlazePose body-axis frame
        world33 = np.array([[l.x, l.y, l.z, l.visibility, l.presence]
                            for l in res.pose_world_landmarks[0]],
                           dtype=np.float32)
        # pose_landmarks: 33 KPs normalised (0..1) image coords + z (relative)
        img33 = np.array([[l.x, l.y, l.z, l.visibility, l.presence]
                          for l in res.pose_landmarks[0]],
                         dtype=np.float32)
        return {"world33": world33, "img33": img33}

    def close(self):
        self.lm.close()


class HandTeacher:
    """MediaPipe Hand Landmarker (21 KPs/hand, two hands max)."""

    DEFAULT_URL = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/"
                   "hand_landmarker.task")

    def __init__(self, task_path: Path, use_gpu: bool = False):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        delegate = (mp_python.BaseOptions.Delegate.GPU if use_gpu
                    else mp_python.BaseOptions.Delegate.CPU)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(task_path), delegate=delegate),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
        self.lm = mp_vision.HandLandmarker.create_from_options(opts)

    def __call__(self, img_rgb_uint8: np.ndarray) -> dict | None:
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=img_rgb_uint8)
        res = self.lm.detect(mp_img)
        if not res.hand_world_landmarks:
            return None
        # Up to two hands; tag handedness L/R
        out: dict[str, np.ndarray] = {}
        for h_world, h_img, hd in zip(res.hand_world_landmarks,
                                      res.hand_landmarks, res.handedness):
            label = hd[0].category_name.lower()  # 'left' or 'right'
            out[f"world_{label}"] = np.array(
                [[l.x, l.y, l.z] for l in h_world], dtype=np.float32)
            out[f"img_{label}"] = np.array(
                [[l.x, l.y, l.z, l.visibility, l.presence] for l in h_img],
                dtype=np.float32)
        return out

    def close(self):
        self.lm.close()


class FaceTeacher:
    """MediaPipe Face Mesh (Face Landmarker)."""

    DEFAULT_URL = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/latest/"
                   "face_landmarker.task")

    def __init__(self, task_path: Path, use_gpu: bool = False):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        delegate = (mp_python.BaseOptions.Delegate.GPU if use_gpu
                    else mp_python.BaseOptions.Delegate.CPU)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(task_path), delegate=delegate),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.lm = mp_vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, img_rgb_uint8: np.ndarray) -> dict | None:
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=img_rgb_uint8)
        res = self.lm.detect(mp_img)
        if not res.face_landmarks:
            return None
        # 478 normalised image-space landmarks (no world frame for face mesh)
        face = np.array([[l.x, l.y, l.z] for l in res.face_landmarks[0]],
                        dtype=np.float32)
        return {"img_face": face}

    def close(self):
        self.lm.close()


def download_teacher_weights(out_dir: Path) -> dict[str, Path]:
    """Pull Hand + Face .task files from Google's CDN.  Returns paths.

    BlazePose Heavy is already on disk under BlazePose tune/assets/; we
    point at it directly.
    """
    import urllib.request
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "hand": HandTeacher.DEFAULT_URL,
        "face": FaceTeacher.DEFAULT_URL,
    }
    paths: dict[str, Path] = {}
    for name, url in targets.items():
        out = out_dir / f"{name}_landmarker.task"
        if not out.exists() or out.stat().st_size < 1024:
            print(f"[teachers] downloading {name} -> {out}")
            urllib.request.urlretrieve(url, out)
        else:
            print(f"[teachers] cached: {out}  ({out.stat().st_size/1024/1024:.1f} MB)")
        paths[name] = out
    return paths


if __name__ == "__main__":
    """Smoke: load each teacher, run on a single random image."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy",  type=Path,
                    default=Path(__file__).resolve().parent.parent
                            / "assets" / "pose_landmarker_heavy.task")
    ap.add_argument("--teacher-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent
                            / "assets" / "teachers")
    args = ap.parse_args()

    paths = download_teacher_weights(args.teacher_dir)
    print()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"[smoke] image shape: {img.shape}")

    print("--- Heavy ---")
    h = HeavyTeacher(args.heavy)
    out = h(img)
    if out is None:
        print("  no detection (random image, expected on synthetic noise)")
    else:
        print(f"  world33: {out['world33'].shape}, "
              f"sample row 0: {out['world33'][0]}")
    h.close()

    print("--- Hand ---")
    h2 = HandTeacher(paths["hand"])
    out = h2(img)
    print(f"  detection: {None if out is None else list(out.keys())}")
    h2.close()

    print("--- Face ---")
    h3 = FaceTeacher(paths["face"])
    out = h3(img)
    print(f"  detection: {None if out is None else out['img_face'].shape}")
    h3.close()

    print("\n[smoke] all teachers loadable")
