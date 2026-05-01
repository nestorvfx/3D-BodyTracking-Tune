"""Multi-teacher KD + hard-supervision loss for BlazePose v2 students.

The student model returns a dict with these tensors (matching .tflite names):
  Identity   (B, 195)        — 39 KPs × 5 (x, y, z, visibility, presence)
  Identity_1 (B, 1)          — pose presence (already sigmoid-ed)
  Identity_2 (B, 256, 256, 1)— segmentation logits (FROZEN during fine-tune)
  Identity_3 (B, 64, 64, 39) — heatmap (FROZEN)
  Identity_4 (B, 117)        — 39 × 3 body-axis metres, mid-hip origin

Loss components (all per-joint masked):
    L_total = λ_hard   · L_hard           hard 17-joint GT (synth + EgoExo) in BP body-axis
            + λ_kd_b   · L_kd_body        Heavy teacher KD on world33
            + λ_kd_h   · L_kd_hand        Hand teacher KD on BP idx 17-22
            + λ_kd_f   · L_kd_face        Face teacher KD on BP idx 1,3,4,6,9,10
            + λ_anchor · L_anchor_v1      anti-regression vs frozen v1 student
            + λ_mv     · L_multiview      Ego-Exo4D multi-view 2D consistency
            + λ_vis    · L_vis_BCE        visibility distillation

Each component is gated by per-joint availability masks; if the source isn't
in this batch (e.g. synth-only batch has no multi-view), that loss is zero.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Output decoders ──────────────────────────────────────────────────────

def split_kp_tuple(out_195: torch.Tensor) -> dict[str, torch.Tensor]:
    """Identity (B, 195) = 39 × (x, y, z, visibility, presence)."""
    B = out_195.shape[0]
    t = out_195.view(B, 39, 5)
    return {
        "xyz":        t[..., :3],   # image-normalised x/y + relative z
        "visibility": t[..., 3],
        "presence":   t[..., 4],
    }


def split_world_tuple(out_117: torch.Tensor) -> torch.Tensor:
    """Identity_4 (B, 117) = 39 × (x, y, z) body-axis metres."""
    return out_117.view(out_117.shape[0], 39, 3)


# ─── Loss primitives ──────────────────────────────────────────────────────

def smooth_l1_masked(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor, beta: float = 0.02) -> torch.Tensor:
    """pred/target: (B, K, 3); mask: (B, K) float."""
    err = F.smooth_l1_loss(pred, target, beta=beta, reduction="none").mean(-1)
    err = err * mask
    return err.sum() / mask.sum().clamp_min(1.0)


def bce_masked(pred_logit: torch.Tensor, target_prob: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    err = F.binary_cross_entropy_with_logits(pred_logit, target_prob,
                                             reduction="none")
    err = err * mask
    return err.sum() / mask.sum().clamp_min(1.0)


# ─── Per-joint masks (which BP indices each teacher / source supplies) ───

# Hand teacher: BP indices 17-22 (pinky/index/thumb tips L+R)
HAND_BP_INDICES = list(range(17, 23))

# Face teacher: BP indices 1, 3, 4, 6, 9, 10 (eye corners + mouth)
FACE_BP_INDICES = [1, 3, 4, 6, 9, 10]


# ─── Composite loss ───────────────────────────────────────────────────────

class V2DistillationLoss(nn.Module):
    """All-component distillation loss.

    Each item in the batch is expected to carry per-source masks so that
    samples without a given supervisory signal contribute zero to that
    component (and zero to its denominator).
    """

    def __init__(self,
                 lam_hard:   float = 1.0,
                 lam_kd_b:   float = 0.5,
                 lam_kd_h:   float = 0.5,
                 lam_kd_f:   float = 0.3,
                 lam_anchor: float = 0.1,
                 lam_mv:     float = 0.2,
                 lam_vis:    float = 0.1,
                 beta_hard:  float = 0.05,
                 beta_kd:    float = 0.02):
        super().__init__()
        self.lam_hard   = lam_hard
        self.lam_kd_b   = lam_kd_b
        self.lam_kd_h   = lam_kd_h
        self.lam_kd_f   = lam_kd_f
        self.lam_anchor = lam_anchor
        self.lam_mv     = lam_mv
        self.lam_vis    = lam_vis
        self.beta_hard  = beta_hard
        self.beta_kd    = beta_kd

    def forward(self,
                student_out: dict,
                hard:    dict | None = None,
                teacher_body: dict | None = None,
                teacher_hand: dict | None = None,
                teacher_face: dict | None = None,
                anchor:  dict | None = None,
                multiview: dict | None = None) -> dict:
        """
        student_out: BlazePosePort.forward() output dict.
        hard: {"bp33_xyz_body": (B, 33, 3), "bp33_present": (B, 33)} — body-axis metres.
        teacher_body: same dict format as student_out (Heavy port output).
        teacher_hand: {"bp33_xyz_body": (B, 33, 3), "bp33_present": (B, 33)} —
                      pre-aligned hand-teacher landmarks; only HAND_BP_INDICES
                      are populated (else mask=0).
        teacher_face: same shape; FACE_BP_INDICES populated.
        anchor: dict from frozen v1 forward (.world33 in body-axis).
        multiview: {"K": (B, V, 3, 3), "Rt": (B, V, 3, 4), "kp2d": (B, V, 17, 2),
                    "vis": (B, V, 17)} — V cams per sample.
        """
        s_kp = split_kp_tuple(student_out["Identity"])
        s_world = split_world_tuple(student_out["Identity_4"])     # (B, 39, 3)

        zero = torch.zeros((), device=s_world.device)
        loss_terms: dict[str, torch.Tensor] = {
            "L_hard": zero, "L_kd_body": zero, "L_kd_hand": zero,
            "L_kd_face": zero, "L_anchor": zero, "L_multiview": zero,
            "L_vis": zero,
        }

        # ── 1) Hard supervision (synth + EgoExo train; body-axis frame) ────
        if hard is not None:
            tgt = hard["bp33_xyz_body"].to(s_world.device)        # (B, 33, 3)
            mask = hard["bp33_present"].to(s_world.device)        # (B, 33)
            loss_terms["L_hard"] = smooth_l1_masked(
                s_world[:, :33], tgt, mask, beta=self.beta_hard)

        # ── 2) Body teacher KD (Heavy → first 33 KPs) ──────────────────────
        if teacher_body is not None:
            t_kp = split_kp_tuple(teacher_body["Identity"])
            t_world = split_world_tuple(teacher_body["Identity_4"])
            body_mask = (t_kp["visibility"][:, :33] > 0.1).float()
            loss_terms["L_kd_body"] = smooth_l1_masked(
                s_world[:, :33], t_world[:, :33], body_mask, beta=self.beta_kd)
            # Visibility KD on the same 33 KPs
            loss_terms["L_vis"] = F.binary_cross_entropy_with_logits(
                s_kp["visibility"][:, :33],
                torch.sigmoid(t_kp["visibility"][:, :33]),
                reduction="mean")

        # ── 3) Hand teacher KD (BP idx 17-22) ──────────────────────────────
        if teacher_hand is not None:
            tgt = teacher_hand["bp33_xyz_body"].to(s_world.device)
            mask = teacher_hand["bp33_present"].to(s_world.device)
            # Restrict mask to HAND indices (zero elsewhere).
            full_mask = torch.zeros_like(mask)
            for idx in HAND_BP_INDICES:
                full_mask[:, idx] = mask[:, idx]
            loss_terms["L_kd_hand"] = smooth_l1_masked(
                s_world[:, :33], tgt, full_mask, beta=self.beta_kd)

        # ── 4) Face teacher KD (BP idx 1, 3, 4, 6, 9, 10) ─────────────────
        if teacher_face is not None:
            tgt = teacher_face["bp33_xyz_body"].to(s_world.device)
            mask = teacher_face["bp33_present"].to(s_world.device)
            full_mask = torch.zeros_like(mask)
            for idx in FACE_BP_INDICES:
                full_mask[:, idx] = mask[:, idx]
            loss_terms["L_kd_face"] = smooth_l1_masked(
                s_world[:, :33], tgt, full_mask, beta=self.beta_kd)

        # ── 5) Anchor distillation (anti-regression vs frozen v1) ─────────
        # Per-joint mask: weight is HIGHER on joints with no hard / no
        # teacher signal (= the only thing keeping them from regressing).
        # This is the load-bearing fix for the 20 BP joints not in the 17-COCO
        # mapping — without it, λ_anchor=0.1 is too weak to dominate the
        # side-effect gradients from supervised joints sharing backbone params.
        if anchor is not None:
            a_world = split_world_tuple(anchor["Identity_4"])
            B = s_world.shape[0]
            supervised = torch.zeros(B, 33, device=s_world.device,
                                     dtype=s_world.dtype)
            if hard is not None:
                supervised = torch.maximum(
                    supervised,
                    hard["bp33_present"].to(s_world).float())
            if teacher_body is not None:
                tb_kp = split_kp_tuple(teacher_body["Identity"])
                supervised = torch.maximum(
                    supervised,
                    (tb_kp["visibility"][:, :33] > 0.1).float())
            # 1.0 = unsupervised → strong anchor; 0.2 = supervised → light anchor
            anchor_weight = torch.where(supervised > 0.5,
                                        torch.tensor(0.2, device=supervised.device),
                                        torch.tensor(1.0, device=supervised.device))
            err = F.smooth_l1_loss(s_world[:, :33], a_world[:, :33],
                                   beta=self.beta_kd, reduction="none").mean(-1)
            # Weighted mean: numerator scales with weight, denominator stays B*33
            loss_terms["L_anchor"] = (err * anchor_weight).sum() / max(
                anchor_weight.sum().item(), 1.0)

        # ── 6) Multi-view (single-cam reprojection) consistency ───────────
        # Per-sample: invert the GT body-frame transform to get pred_cam,
        # project via K to native-pixel 2D, compare to GT 2D.  Strong
        # 2D-grounded constraint per sample.  Engaged only when GT supplied
        # all the multi-view fields and >=4 2D anchors are visible.
        if multiview is not None and self.lam_mv > 0:
            valid     = multiview["mv_valid"].to(s_world.device)         # (B,)
            K_norm    = multiview["mv_K_norm"].to(s_world)               # (B, 3, 3) [0,1]
            R_b2c     = multiview["mv_R_body2cam"].to(s_world)           # (B, 3, 3)
            origin    = multiview["mv_origin_cam"].to(s_world)           # (B, 3)
            kp2d_gt   = multiview["mv_kp2d_norm"].to(s_world)            # (B, 17, 2) [0,1]
            present2d = multiview["mv_present_2d"].to(s_world)           # (B, 17)
            from lib.keypoint_map import BP_INDEX_FOR_COCO
            idx = torch.tensor(BP_INDEX_FOR_COCO, device=s_world.device)
            s17_body = s_world[:, :33].index_select(1, idx)              # (B, 17, 3)
            # pred_cam = R_body_to_cam @ pred_body + origin
            s17_cam = (s17_body @ R_b2c.transpose(-1, -2)) + origin[:, None, :]
            z  = s17_cam[..., 2:3].clamp_min(0.05)
            xy = s17_cam[..., :2] / z
            fx = K_norm[:, 0, 0:1].unsqueeze(1)
            fy = K_norm[:, 1, 1:2].unsqueeze(1)
            cx = K_norm[:, 0, 2:3].unsqueeze(1)
            cy = K_norm[:, 1, 2:3].unsqueeze(1)
            s17_2d_norm = torch.stack(
                [fx[..., 0] * xy[..., 0] + cx[..., 0],
                 fy[..., 0] * xy[..., 1] + cy[..., 0]], dim=-1)
            err = F.smooth_l1_loss(s17_2d_norm, kp2d_gt, beta=0.05,
                                   reduction="none").mean(-1)             # (B, 17)
            mask = present2d * valid[:, None]
            denom = mask.sum().clamp_min(1.0)
            loss_terms["L_multiview"] = (err * mask).sum() / denom

        total = (
            self.lam_hard   * loss_terms["L_hard"]
          + self.lam_kd_b   * loss_terms["L_kd_body"]
          + self.lam_kd_h   * loss_terms["L_kd_hand"]
          + self.lam_kd_f   * loss_terms["L_kd_face"]
          + self.lam_anchor * loss_terms["L_anchor"]
          + self.lam_mv     * loss_terms["L_multiview"]
          + self.lam_vis    * loss_terms["L_vis"]
        )
        return {"total": total, **{k: v.detach() for k, v in loss_terms.items()}}


if __name__ == "__main__":
    """Smoke: full loss with all components active."""
    B = 2
    student = {
        "Identity":   torch.randn(B, 195, requires_grad=True),
        "Identity_1": torch.zeros(B, 1, requires_grad=True),
        "Identity_4": torch.randn(B, 117, requires_grad=True),
    }
    hard = {"bp33_xyz_body": torch.randn(B, 33, 3),
            "bp33_present":  torch.zeros(B, 33).index_fill_(1,
                                torch.tensor([0, 11, 12, 23, 24]), 1.0)}
    teacher_body = {"Identity":   torch.randn(B, 195),
                    "Identity_4": torch.randn(B, 117)}
    teacher_hand = {"bp33_xyz_body": torch.randn(B, 33, 3),
                    "bp33_present":  torch.zeros(B, 33).index_fill_(1,
                                torch.tensor([17, 18, 19, 20, 21, 22]), 1.0)}
    teacher_face = {"bp33_xyz_body": torch.randn(B, 33, 3),
                    "bp33_present":  torch.zeros(B, 33).index_fill_(1,
                                torch.tensor([1, 3, 4, 6, 9, 10]), 1.0)}
    anchor = {"Identity": torch.randn(B, 195), "Identity_4": torch.randn(B, 117)}

    loss_fn = V2DistillationLoss()
    out = loss_fn(student, hard=hard, teacher_body=teacher_body,
                  teacher_hand=teacher_hand, teacher_face=teacher_face,
                  anchor=anchor)
    out["total"].backward()
    for k, v in out.items():
        print(f"  {k:14s} = {v.item():.4f}")
    print(f"  grad on student.Identity finite: "
          f"{torch.isfinite(student['Identity'].grad).all().item()}")
