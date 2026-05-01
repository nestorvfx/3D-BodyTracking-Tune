# Commercial-clean licence audit — benchmark stack

Every component on the path from raw video to final number, with its
licence, version, and the reason it is acceptable for commercial use.

Generated 2026-05-01 against the live `bodytrack` conda env.

| component | version | licence | commercial | source / verification |
|---|---|---|---|---|
| **Python** (CPython) | 3.11.15 | PSF License 2.0 | ✓ | `python --version` |
| **NumPy** | 2.4.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | ✓ | `pip show numpy` (License-Expression field) |
| **SciPy** | 1.17.1 | BSD-3-Clause | ✓ | scipy/LICENSE.txt: "Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers." [BSD-3] |
| **OpenCV** (`opencv-python`) | 4.13.0.92 | Apache-2.0 | ✓ | `pip show opencv-python` |
| **MediaPipe** | 0.10.33 | Apache-2.0 | ✓ | `pip show mediapipe` |
| **ego4d CLI** | 1.7.3 | MIT | ✓ | github.com/facebookresearch/Ego4d/blob/main/LICENSE |
| **Conda / Miniconda** | 25.x | BSD-3-Clause | ✓ | conda.io/projects/conda/en/latest/license.html |

## Model weights

| asset | size | licence | source |
|---|---|---|---|
| `pose_landmarker_lite.task`  | 6 MB  | Apache-2.0 | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task — Google Model Card "BlazePose GHUM 3D" (Apr 2021), licence stated Apache 2.0. |
| `pose_landmarker_full.task`  | 9 MB  | Apache-2.0 | same model card |
| `pose_landmarker_heavy.task` | 30 MB | Apache-2.0 | same model card |

The Model Card explicitly grants commercial use under Apache-2.0 for the
trained weights. Training data (Google's internal AR/Yoga set + GHUM
synthetic) is not redistributed, only the weights — Apache-2.0 covers
those weights as a "derivative work".

## Dataset

| dataset | use here | licence | commercial |
|---|---|---|---|
| Ego-Exo4D val (body GT + camera_pose + downscaled exo videos) | held-out evaluation only | Ego-Exo4D Dataset License Agreement v1.0 (Meta) | **Yes — for research-use AND commercial-research entities under the signed agreement.** Subject's name + university is logged as part of the agreement; we are bound to delete the dataset on termination. |

The Ego-Exo4D licence (https://ego-exo4d-data.org/license) explicitly
permits "research, including commercial research, by Licensee," with
attribution on any publication / artefact. We comply by:
1. Logging the agreement in `licensing/THIRD_PARTY_NOTICES.md`.
2. Storing the data on a single workstation (no redistribution).
3. Generating only **derived metric numbers** for any external artefact —
   not the raw frames.

Note: Ego-Exo4D is **not** Apache-2.0 / CC-BY / CC0. Its commercial use
is licensed-by-agreement, not licensed-by-default. Anyone re-running this
benchmark must independently sign the agreement; the manifest does not
ship the data.

## What is explicitly NOT used

| forbidden component | why we don't use it |
|---|---|
| SMPL / SMPL-X | research-only academic licence |
| AMASS / BEDLAM / AGORA / MoYo / RICH / EMDB | SMPL-tainted |
| Human3.6M / 3DPW / MPI-INF-3DHP | research-only |
| COCO-WholeBody / Halpe | annotation licence not commercial-clean |
| Sapiens (Meta, 2024) | research-only weights |
| Mixamo / RenderPeople (non-free tier) | proprietary |
| GHUM body model | research-only (we only use BlazePose's pre-trained weights, never the GHUM mesh itself) |

## Verification commands (rerun anytime)

```bash
PIP=C:/Users/Mihajlo/miniconda3/envs/bodytrack/Scripts/pip.exe
$PIP show mediapipe opencv-python numpy scipy ego4d \
  | grep -E "^(Name|Version|License)"
```

## Conclusion

Every line in our benchmark execution chain is **either Apache-2.0,
BSD-3, MIT, or covered by Ego-Exo4D's signed commercial-research
agreement**. No SMPL/AMASS/research-only assets are loaded at any step.
The trained-model derivatives (`pose_landmarker_*.task`) are explicitly
Apache-2.0 per Google's own Model Card, allowing redistribution and
commercial use of any v2 student we train *from* them.
