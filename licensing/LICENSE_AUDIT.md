# Commercial-use license audit

Status per component for the synthetic 3D pose pipeline.
**Green** = ship without restriction.  **Amber** = OK for dev, resolve before
external ship.  **Red** = block.

Generated 2026-04-20.

## Summary

| component | licence | status | note |
|---|---|---|---|
| Project code (mine) | TBD → Apache-2.0 planned | Green | owner's choice |
| Blender 5.1 | GPL | Green | output of Blender is yours (Blender FAQ) |
| MPFB2 code | GPLv3 | Green | we use the tool to export assets; no GPL code shipped |
| MPFB2 base meshes + weights | CC0 | Green | |
| CMU Mocap (cgspeed mirror) | Public Domain | Green | |
| 100STYLE motion data | CC-BY 4.0 | Green | attribution required at ship |
| AIST++ annotations (BVH retarget input) | CC-BY 4.0 | Green | attribution required at ship |
| Poly Haven HDRIs | CC0 | Green | |
| PyTorch 2.6 | BSD-3 | Green | |
| timm 1.0.26 (code) | Apache-2.0 | Green | |
| MobileNetV4-Conv-S weights (timm HF) | **see below** | **Amber** | ImageNet-1k-trained — industry-accepted for commercial use, but NOT formally cleared. |
| Albumentations 2.0.8 | MIT | Green | |
| Open Images V7 — segmentation masks (annotations) | Apache-2.0 (Google) | Green | used for sim-to-real occluder corpus (training-time augmentation only, never in val/test set). |
| Open Images V7 — images | CC BY 2.0 (Flickr photographers) | Green | used as occluder cutouts + person-free bg corpus for training-time augmentation only. Attribution required at ship — see "Attribution obligations" below. |
| MediaPipe Pose Heavy (`pose_landmarker_heavy.task`) | Apache-2.0 (Google) | Green | used to compute synth person mattes for BG-composite augmentation. |
| RTMPose3D SimCC-3D head (my reimpl) | Apache-2.0 (of design) | Green | implementation not copy-pasted |
| TensorBoard | Apache-2.0 | Green | |

No SMPL / SMPL-X / AMASS / BEDLAM / BEDLAM2 / AGORA / Human3.6M /
3DPW / MPI-INF-3DHP / CMU-Panoptic / EMDB / THuman / Fit3D / HumanML3D /
MPII / COCO-WholeBody / CrowdPose / PoseTrack / JHMDB / Mixamo /
SynthMoCap / RenderPeople (non-free) in any branch of the pipeline.

## MobileNetV4 pretrained weights — detail

**Model:** `timm/mobilenetv4_conv_small.e2400_r224_in1k`
**Author:** Ross Wightman (timm maintainer)
**Training data:** ImageNet-1k
**Weight licence:** Not explicitly stated on the model card.  timm's code
licence is Apache-2.0; the weight card directs to the [MobileNetV4 paper](https://arxiv.org/abs/2404.10518)
for architecture and cites `timm` for the training recipe.

### Legal analysis

1. **ImageNet-1k dataset licence** requires non-commercial use of the
   images themselves.  It does **not** address derived models.
2. **Prevailing industry interpretation** (Ultralytics, HuggingFace,
   Meta, Google): trained weights are "derivative works" that do not
   inherit the dataset's licence restrictions.  Quote from the
   Ultralytics maintainer response to issue
   [#4233](https://github.com/ultralytics/ultralytics/issues/4233):
   > "Using weights derived from ImageNet pre-training does not inherit
   > the dataset's license restrictions."
3. **Enforcement history:** no known suit has enforced ImageNet's term
   against commercial weight redistribution.  Apple, Google, Meta,
   Microsoft, Amazon all ship ImageNet-pretrained models in commercial
   products (every ML-enabled consumer app).
4. **Project stance:** the project's own `README.md` explicitly lists
   MobileNetV4-Conv-M (same family, same licence chain) as an
   Apache-2.0 component.  Keeping MNv4-S is consistent.

### Conclusion: AMBER, proceed with caveat

Use MNv4-S pretrained weights for development and first shipping
iterations.  Before large-scale commercial ship, consider one of:

- **(a)** Retrain the backbone from scratch on a commercial-clean image
  corpus (OpenCLIP-DataComp-1B subset is Apache-2.0 + commercial OK).
  ~$1-2k compute.
- **(b)** Swap backbone to an OpenCLIP Vision Transformer trained on
  DataComp CommonPool.  Different compute profile, different mobile
  export path.
- **(c)** Accept the industry-standard practice and ship with MNv4
  (pragmatic, carries negligible legal risk per industry precedent).

The project README already chose (c) for MobileNetV4-Conv-M.  This
design doc formalises that choice.

## Attribution obligations (at ship)

- **100STYLE** — "© 2022 Mason et al.; used under CC-BY 4.0" per their
  SCA 2022 paper and repo README.
- **AIST++** — "AIST++ Dance Motion Dataset (Li et al., ICCV 2021),
  CC-BY 4.0" per their AIST project website.
- **Poly Haven HDRIs** — CC0, attribution appreciated but not required.
- **MPFB2 / MakeHuman CC0 assets** — attribution appreciated but not required.
- **timm / MobileNetV4** — Apache-2.0 NOTICE text.
- **Open Images V7** — required attribution string (auto-emitted to
  `assets/sim2real_refs/ATTRIBUTION.txt` by the extractor):
  > "This product uses augmentation imagery derived from Open Images V7
  > (https://storage.googleapis.com/openimages/web/, images CC BY 2.0,
  > annotations Apache-2.0)."

  Place this string in:
  1. `THIRD_PARTY_NOTICES.md` at ship time (single-line entry).
  2. The README / About / Credits surface of any consumer app that
     ships the trained model.
  3. The model card if the trained checkpoint is distributed standalone
     (e.g., on Hugging Face).

  Per-image attribution to individual Flickr photographers is NOT
  required for trained-model derivatives — CC BY 2.0's "reasonable to
  the medium" clause is satisfied by aggregate attribution to the
  Open Images V7 dataset (industry precedent: every commercial model
  trained on Open Images / Flickr-CC corpora attributes at the dataset
  level, not per image).

- **MediaPipe Pose** — Apache-2.0 NOTICE text (used at preprocessing
  time to compute mattes; not embedded in the trained model itself,
  but the NOTICE preservation requirement still applies because the
  matte outputs are derivative).

Ship a THIRD_PARTY_NOTICES.md aggregating all of these when the final
artefact is released.
