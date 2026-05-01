# Third-Party Notices

This project consumes third-party datasets, model weights, and software.
This file aggregates all attribution and licence-preservation requirements
in one place, satisfying the "any reasonable manner" attribution clause
of CC-BY licences and the NOTICE-preservation clause of Apache-2.0.

For a deeper licence audit (per-component status, amber/green ratings,
legal reasoning), see [`dataset/LICENSE_AUDIT.md`](dataset/LICENSE_AUDIT.md).

---

## Datasets

### CMU Motion Capture (cgspeed mirror)
- **Use:** source mocap motions for synthetic body-pose dataset generation
- **Licence:** Public Domain (per CMU Graphics Lab statement)
- **Attribution:** none required; courtesy mention appreciated

### 100STYLE Motion Dataset
- **Use:** stylised walking/running motions for synthetic body-pose dataset generation
- **Licence:** Creative Commons Attribution 4.0 (CC-BY-4.0)
- **Attribution:** © 2022 Mason et al.; "100STYLE Motion Dataset" used under CC-BY-4.0
- **Reference:** Mason et al., SCA 2022

### AIST++ Dance Motion Dataset
- **Use:** dance motion sequences for synthetic body-pose dataset generation
- **Licence:** Creative Commons Attribution 4.0 (CC-BY-4.0)
- **Attribution:** "AIST++ Dance Motion Dataset (Li et al., ICCV 2021), CC-BY-4.0"

### Open Images V7
- **Use:** training-time augmentation only — occluder cutouts (F1) +
  person-free background corpus (F2b). The dataset is NEVER part of the
  validation or test set and NEVER part of the synthetic training source.
- **Licence (images):** Creative Commons Attribution 2.0 (CC-BY-2.0)
- **Licence (annotations):** Apache License 2.0 (Google)
- **Attribution:** "This product uses augmentation imagery derived from
  Open Images V7 (https://storage.googleapis.com/openimages/web/, images
  CC-BY-2.0, annotations Apache-2.0)."
- **Reference:** Kuznetsova et al., IJCV 2020

### Poly Haven HDRIs
- **Use:** environment lighting / world background for synthetic renders
- **Licence:** Creative Commons Zero (CC0)
- **Attribution:** none required; courtesy mention appreciated

### MPFB2 / MakeHuman base meshes & weights
- **Use:** parametric body mesh source for synthetic character generation
- **Licence:** CC0 (base meshes)
- **Attribution:** none required; courtesy mention appreciated

---

## Model Weights

### MobileNetV4-Conv-S (timm Hugging Face)
- **Use:** image-encoder backbone (pretrained on ImageNet-1k)
- **Licence (code/training recipe):** Apache-2.0 (timm)
- **Licence (weights):** Industry-accepted commercial use of ImageNet-1k
  pretrained weights; not formally cleared. See `dataset/LICENSE_AUDIT.md`
  "MobileNetV4 pretrained weights — detail" for the legal analysis and
  fallback options.

### MediaPipe Pose Heavy (`pose_landmarker_heavy.task`)
- **Use:** computing per-sample person mattes for BG-composite augmentation
  (preprocessing only — not embedded in the trained model)
- **Licence:** Apache License 2.0 (Google)
- **NOTICE preservation:** required (this file satisfies it)

---

## Software libraries

The following libraries are used as runtime dependencies. All are commercial-
clean. NOTICE preservation as required by each licence.

- **PyTorch 2.x** — BSD-3-Clause
- **timm 1.0.x** — Apache-2.0
- **Albumentations 2.0.x** — MIT
- **OpenCV** — Apache-2.0
- **TensorBoard** — Apache-2.0
- **Pillow** — MIT-CMU (HPND)
- **Blender 5.x** (dataset generation tool only — not shipped) — GPLv3
  (renders produced by Blender belong to the user per Blender FAQ)
- **MPFB2 / MakeHuman** (dataset generation tool only — not shipped) — GPLv3
  (used to export CC0 mesh/weight assets; no GPL code shipped)

---

## What this file does NOT cover

- Per-image attribution to individual Flickr photographers in Open Images V7.
  Industry-accepted practice for trained-model derivatives is aggregate
  dataset-level attribution (this file). Per-image attribution is not
  required by CC-BY-2.0 for derivatives where individual identification
  is impractical.
- Datasets explicitly excluded from this project (research-only): Pascal VOC,
  COCO images, LVIS, Places365, Pexels, Pixabay, Human3.6M, MPI-INF-3DHP,
  3DPW, AGORA, BEDLAM/BEDLAM2.0, AMASS, EMDB, RICH, MoYo, Ego-Exo4D, SMPL,
  SMPL-X, Mixamo, RenderPeople (non-free).
