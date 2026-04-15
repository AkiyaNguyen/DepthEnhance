"""
Patch-wise supervised contrastive setup (design A).

- One contrastive sample per feature cell (i, j): embedding is **F[b, :, i, j]** — no avg_pool over
  neighboring feature cells.
- **Labeled:** foreground fraction **f** from full-res GT on the image rectangle aligned to that cell
  (same split as row_b / col_b); then `three_class_after_elimination`.
- **Unlabeled:** teacher probabilities interpolated to **full image** resolution; **mean entropy** on
  the aligned image crop must be ≤ `patch_entropy_max` **before** pseudo-label class assignment;
  then mean rounded teacher in that crop → `three_class_after_elimination`.
- Returns (N, C) and (N,) class ids for MemoryEfficientSupContrastive3ClassLoss.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from utils.loss import binary_entropy


def patch_fold(label: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    fold the label into patches
    label: (B, 1, H, W) or (B, H, W)
    patch_size: int
    Returns:
        (B * H // P * W // P, P * P)
    """
    if label.dim() == 3:
        label = label.unsqueeze(1)
    B, _, H, W = label.shape
    P = int(patch_size)
    if P <= 0:
        raise ValueError(f"patch_size must be positive, got {P}")
    if H % P != 0 or W % P != 0:
        raise ValueError(f"H={H}, W={W} must be divisible by patch_size={P}")

    # Drop the singleton channel before reordering patch axes.
    x = label.squeeze(1).reshape(B, H // P, P, W // P, P)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.reshape(-1, P * P)

def average_entropy(prob: torch.Tensor) -> torch.Tensor:
    """
    mean entropy of the probability over the last dimension (normalize to [0,1])
    """
    ent = binary_entropy(prob)
    ln2 = torch.log(prob.new_tensor(2.0))
    return ent.mean(dim=-1) / ln2

def classify_patch_type(probs: torch.Tensor, low_bd_thresh=0.25, high_bd_thresh=0.75, \
    high_bg_thresh=0, low_fg_thresh=1, eps=1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    probs: (B, H, W, K)
    "foreground", "background", "boundary"
    """
    low_bd_thresh = low_bd_thresh - eps
    high_bd_thresh = high_bd_thresh + eps
    low_fg_thresh = low_fg_thresh - eps
    high_bg_thresh = high_bg_thresh + eps
    
    pseudo_labels = (probs >= 0.5).float()
    average_prob = pseudo_labels.mean(dim=-1)
    fg_mask = average_prob > low_fg_thresh
    bg_mask = average_prob < high_bg_thresh
    boundary_mask = (average_prob >= low_bd_thresh) & (average_prob <= high_bd_thresh)
    
    
    return fg_mask, bg_mask, boundary_mask

def three_class_after_elimination(
    frac: torch.Tensor,
    lo: float = 0.25,
    hi: float = 0.75,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    frac: foreground fraction in [0, 1] per patch.
    Drop lean mixed: (0, lo) ∪ (hi, 1) excluding endpoints → ignore -1.
    """
    out = torch.full_like(frac, -1.0)
    bad = ((frac > eps) & (frac < lo)) | ((frac > hi) & (frac < 1.0 - eps))
    good = ~bad
    out[good & (frac <= eps)] = 0.0
    out[good & (frac >= 1.0 - eps)] = 1.0
    out[good & (frac >= lo) & (frac <= hi)] = 2.0
    return out


def _gt_fg_fraction_patches_fullres(
    label_fullres: torch.Tensor,
    feat_h: int,
    feat_w: int,
    patch_side_d: int,
) -> torch.Tensor:
    """
    For each D×D block on the feature grid, map to image rows/cols via the same split as
    row_b[k]=k*H_img//feat_h, col_b[k]=k*W_img//feat_w, and return mean foreground in that crop.
    Output: (B, Ho, Wo) with Ho=feat_h//D, Wo=feat_w//D.
    """
    B, _, H_img, W_img = label_fullres.shape
    D = int(patch_side_d)
    lab1 = (label_fullres > 0.5).float().squeeze(1)
    P = F.pad(lab1, (1, 0, 1, 0)).cumsum(dim=1).cumsum(dim=2)

    Ho = feat_h // D
    Wo = feat_w // D
    if Ho == 0 or Wo == 0:
        return lab1.new_zeros(B, 0, 0)

    row_b = torch.arange(feat_h + 1, device=lab1.device, dtype=torch.long) * H_img // feat_h
    col_b = torch.arange(feat_w + 1, device=lab1.device, dtype=torch.long) * W_img // feat_w

    pi = torch.arange(Ho, device=lab1.device).view(Ho, 1).expand(Ho, Wo)
    pj = torch.arange(Wo, device=lab1.device).view(1, Wo).expand(Ho, Wo)
    r0 = row_b[pi * D]
    r1 = row_b[(pi + 1) * D]
    c0 = col_b[pj * D]
    c1 = col_b[(pj + 1) * D]

    s = P[:, r1, c1] - P[:, r0, c1] - P[:, r1, c0] + P[:, r0, c0]
    area = (r1 - r0).float() * (c1 - c0).float()
    return s / area.clamp(min=1.0)


def _mean_scalar_in_aligned_crops(
    map_bhw: torch.Tensor,
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    """
    map_bhw: (B, H_img, W_img) per-pixel scalars.
    For each feature cell (i, j), mean over the image rectangle aligned to that cell
    (row_b[i]:row_b[i+1], col_b[j]:col_b[j+1]).
    Returns (B, feat_h, feat_w).
    """
    B, H_img, W_img = map_bhw.shape
    device = map_bhw.device
    x = map_bhw.float()
    P = F.pad(x, (1, 0, 1, 0)).cumsum(dim=1).cumsum(dim=2)
    row_b = torch.arange(feat_h + 1, device=device, dtype=torch.long) * H_img // feat_h
    col_b = torch.arange(feat_w + 1, device=device, dtype=torch.long) * W_img // feat_w
    Ho, Wo = feat_h, feat_w
    pi = torch.arange(Ho, device=device).view(Ho, 1).expand(Ho, Wo)
    pj = torch.arange(Wo, device=device).view(1, Wo).expand(Ho, Wo)
    r0 = row_b[pi]
    r1 = row_b[pi + 1]
    c0 = col_b[pj]
    c1 = col_b[pj + 1]
    s = P[:, r1, c1] - P[:, r0, c1] - P[:, r1, c0] + P[:, r0, c0]
    area = (r1 - r0).float() * (c1 - c0).float()
    return s / area.clamp(min=1.0)


def build_patchwise_supcon(
    labeled_stu_feature: torch.Tensor,
    label_fullres: torch.Tensor,
    unlabeled_stu_feature: torch.Tensor,
    tea_output_unlabeled: torch.Tensor,
    patch_entropy_max: Optional[float],
    elimination_lo: float = 0.25,
    elimination_hi: float = 0.75,
    unlabeled_image_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    one row of (N, C) per feature cell F[:, :, i, j]. Teacher entropy and pseudo-labels use
    full-resolution crops aligned to the same (i, j) grid.

    unlabeled_image_hw: (H, W) of unlabeled RGB inputs. If None, uses label_fullres spatial size
    (ok when all inputs share resolution).

    Returns:
        feat_flat: (N, C)
        cls_flat: (N,) in {0,1,2}; dropped by elimination or unlabeled entropy gate are omitted.
    """
    device = labeled_stu_feature.device
    dtype = labeled_stu_feature.dtype
    C = labeled_stu_feature.shape[1]

    parts_feat = []
    parts_cls = []

    # --- Labeled: one vector per cell; GT fraction on full-res aligned crop (D_feat = 1) ---
    Hf, Wf = labeled_stu_feature.shape[2], labeled_stu_feature.shape[3]
    if Hf > 0 and Wf > 0 and label_fullres.shape[0] > 0:
        frac_l = _gt_fg_fraction_patches_fullres(label_fullres, Hf, Wf, patch_side_d=1)
        cls_l = three_class_after_elimination(frac_l, elimination_lo, elimination_hi)
        feat_l = labeled_stu_feature.permute(0, 2, 3, 1).reshape(-1, C)
        cls_lf = cls_l.reshape(-1)
        m = cls_lf >= 0
        parts_feat.append(feat_l[m])
        parts_cls.append(cls_lf[m])

    # --- Unlabeled: full-res teacher; mean entropy per crop then elimination; certainty first ---
    if patch_entropy_max is not None and unlabeled_stu_feature.shape[0] > 0:
        H_u, W_u = unlabeled_stu_feature.shape[2], unlabeled_stu_feature.shape[3]
        if unlabeled_image_hw is not None:
            H_img, W_img = int(unlabeled_image_hw[0]), int(unlabeled_image_hw[1])
        else:
            H_img, W_img = label_fullres.shape[2], label_fullres.shape[3]

        tea_u = F.interpolate(
            tea_output_unlabeled.float(),
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False,
        )
        ent = binary_entropy(tea_u).squeeze(1)
        mean_e = _mean_scalar_in_aligned_crops(ent, H_u, W_u)

        hard_u = (tea_u >= 0.5).float().squeeze(1)
        frac_u = _mean_scalar_in_aligned_crops(hard_u, H_u, W_u)

        chosen = mean_e <= float(patch_entropy_max)
        cls_u = three_class_after_elimination(frac_u, elimination_lo, elimination_hi)
        valid = chosen & (cls_u >= 0)

        feat_u = unlabeled_stu_feature.permute(0, 2, 3, 1).reshape(-1, C)
        cls_u = cls_u.reshape(-1)
        valid = valid.reshape(-1)

        parts_feat.append(feat_u[valid])
        parts_cls.append(cls_u[valid])

    if not parts_feat:
        z = torch.zeros(0, C, device=device, dtype=dtype)
        return z, z.new_zeros(0)

    feat_flat = torch.cat(parts_feat, dim=0)
    cls_flat = torch.cat(parts_cls, dim=0)
    return feat_flat, cls_flat
