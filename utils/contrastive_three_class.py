"""
3-class contrastive labels on a fixed ph×pw grid on the student feature map.

Class ids (float): 0, 1, 2, and ignore_label (-1).

**Shared rule (labeled GT and unlabeled rounded pseudo):** for each patch, let `frac` be the
fraction of foreground (GT 1, or rounded teacher 1). Then:
  - ~100% fg  -> 1
  - ~100% bg  -> 0
  - frac in [boundary_lo, boundary_hi] -> 2 (boundary)
  - frac > boundary_hi (but not pure fg) -> 1
  - frac < boundary_lo (but not pure bg) -> 0

**Labeled:** Full-res GT is split into one patch per ph×pw **feature** block: image pixels mapped
from each feature row/column range (no interpolation). `frac` = mean GT in that image region.

**Unlabeled:** teacher at feature res, same ph×pw patches; entropy gate → round → same 3-class rule.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from utils.loss import binary_entropy


def assign_three_class_from_fg_fraction(
    frac: torch.Tensor,
    patch_active: torch.Tensor,
    boundary_lo: float,
    boundary_hi: float,
    ignore_label: float = -1.0,
) -> torch.Tensor:
    """
    frac: foreground fraction in [0, 1], arbitrary shape (e.g. B, Ho*Wo).
    patch_active: bool same shape; if False, output ignore_label at those positions.
    """
    cls = torch.full_like(frac, ignore_label)
    ok = patch_active

    pure1 = ok & (frac >= 1.0 - 1e-4)
    pure0 = ok & (frac <= 1e-4)
    boundary = ok & (frac >= boundary_lo) & (frac <= boundary_hi)
    lean_fg = ok & ~pure1 & (frac > boundary_hi)
    lean_bg = ok & ~pure0 & (frac < boundary_lo)

    cls[pure1] = 1.0
    cls[pure0] = 0.0
    cls[boundary] = 2.0
    cls[lean_fg] = 1.0
    cls[lean_bg] = 0.0
    return cls


def build_labeled_three_class_map_fullres(
    label_fullres: torch.Tensor,
    feat_h: int,
    feat_w: int,
    patch_h: int,
    patch_w: int,
    boundary_lo: float = 0.25,
    boundary_hi: float = 0.75,
    ignore_label: float = -1.0,
) -> torch.Tensor:
    """
    label_fullres: (B, 1, H_img, W_img) — GT (binarized at 0.5).
    feat_h, feat_w: student feature map spatial size (Hf, Wf).
    For each ph×pw block on the feature grid, take the aligned **image** rectangle
    (row_b[i*ph]:row_b[(i+1)*ph], col_b[j*pw]:col_b[(j+1)*pw]) and set frac = mean(GT).
    Output: (B, 1, Hf, Wf) with class id broadcast per patch; tail crop if Hf/Wf not multiple.
    """
    B, _, H_img, W_img = label_fullres.shape
    device, dtype = label_fullres.device, label_fullres.dtype
    out = torch.full((B, 1, feat_h, feat_w), ignore_label, device=device, dtype=dtype)

    if patch_h < 1 or patch_w < 1:
        raise ValueError("patch_h and patch_w must be >= 1")

    Ho = feat_h // patch_h
    Wo = feat_w // patch_w
    Hc = Ho * patch_h
    Wc = Wo * patch_w
    if Ho == 0 or Wo == 0:
        return out

    lab1 = (label_fullres > 0.5).float().squeeze(1)
    P = F.pad(lab1, (1, 0, 1, 0)).cumsum(dim=1).cumsum(dim=2)

    row_b = torch.arange(feat_h + 1, device=device, dtype=torch.long) * H_img // feat_h
    col_b = torch.arange(feat_w + 1, device=device, dtype=torch.long) * W_img // feat_w

    pi = torch.arange(Ho, device=device).view(Ho, 1).expand(Ho, Wo)
    pj = torch.arange(Wo, device=device).view(1, Wo).expand(Ho, Wo)
    r0 = row_b[pi * patch_h]
    r1 = row_b[(pi + 1) * patch_h]
    c0 = col_b[pj * patch_w]
    c1 = col_b[(pj + 1) * patch_w]

    s = P[:, r1, c1] - P[:, r0, c1] - P[:, r1, c0] + P[:, r0, c0]
    area = (r1 - r0).float() * (c1 - c0).float()
    frac = s / area.clamp(min=1.0)

    ok = torch.ones_like(frac, dtype=torch.bool)
    cls = assign_three_class_from_fg_fraction(
        frac.reshape(B, Ho * Wo),
        ok.reshape(B, Ho * Wo),
        boundary_lo,
        boundary_hi,
        ignore_label,
    )

    cls_map = cls.view(B, 1, Ho, Wo, 1, 1).expand(B, 1, Ho, patch_h, Wo, patch_w)
    cls_map = cls_map.reshape(B, 1, Hc, Wc)
    out[:, :, :Hc, :Wc] = cls_map
    return out


def build_unlabeled_three_class_map(
    teacher_prob_feat_res: torch.Tensor,
    patch_h: int,
    patch_w: int,
    patch_entropy_max: float,
    boundary_lo: float = 0.25,
    boundary_hi: float = 0.75,
    ignore_label: float = -1.0,
) -> torch.Tensor:
    """
    1) Mean teacher entropy per patch <= patch_entropy_max → patch is in the pool.
    2) Round teacher prob at 0.5 → per-pixel 0/1.
    3) Same 3-class rule as labeled: frac = mean(rounded) in patch.
    """
    B, _, H, W = teacher_prob_feat_res.shape
    device, dtype = teacher_prob_feat_res.device, teacher_prob_feat_res.dtype
    out = torch.full((B, 1, H, W), ignore_label, device=device, dtype=dtype)

    Hc = (H // patch_h) * patch_h
    Wc = (W // patch_w) * patch_w
    if Hc == 0 or Wc == 0:
        return out

    tea = teacher_prob_feat_res[:, :, :Hc, :Wc]
    ent = binary_entropy(tea).squeeze(1)
    Ho, Wo = Hc // patch_h, Wc // patch_w

    hard = (tea >= 0.5).float()
    h_p = hard.reshape(B, Ho, patch_h, Wo, patch_w).permute(0, 2, 4, 3, 5).contiguous()
    h_p = h_p.reshape(B, Ho * Wo, patch_h * patch_w)
    e_p = ent.reshape(B, Ho, patch_h, Wo, patch_w).permute(0, 2, 4, 3, 5).contiguous()
    e_p = e_p.reshape(B, Ho * Wo, patch_h * patch_w)

    mean_e = e_p.mean(dim=1)
    chosen = mean_e <= patch_entropy_max
    frac = h_p.mean(dim=1)

    cls = assign_three_class_from_fg_fraction(
        frac, chosen, boundary_lo, boundary_hi, ignore_label
    )

    cls_map = cls.view(B, 1, Ho, Wo, 1, 1).expand(B, 1, Ho, patch_h, Wo, patch_w)
    cls_map = cls_map.reshape(B, 1, Hc, Wc)
    out[:, :, :Hc, :Wc] = cls_map
    return out


def merge_labeled_unlabeled_for_contrastive(
    labeled_feature: torch.Tensor,
    labeled_three_class: torch.Tensor,
    unlabeled_feature: torch.Tensor,
    unlabeled_three_class: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate batch dimensions: (B_l + B_u, D, H, W) and (B_l + B_u, 1, H, W)."""
    feat = torch.cat([labeled_feature, unlabeled_feature], dim=0)
    lab = torch.cat([labeled_three_class, unlabeled_three_class], dim=0)
    return feat, lab
