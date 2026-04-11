"""
Loss modules for segmentation and consistency training.

Standard interface: pred and target/mask are probabilities in [0, 1]
(same shape). MSELoss is for consistency (input/target can be logits or probs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MSELoss(nn.Module):
    """MSE between probs (e.g. student vs teacher predictions)."""

    def __init__(self):
        super().__init__()

    def forward(self, input_prob: torch.Tensor, target_prob: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(input_prob)
        
        num_classes = input_prob.size(1)
        mse_loss = F.mse_loss(input_prob, target_prob, reduction='none') * mask.float()
        return mse_loss.sum() / (mask.float().sum().clamp(min=1) * num_classes)


class DiceLoss(nn.Module):
    """Dice loss for segmentation (1 - dice score, averaged over batch)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)
        
        size = pred.size(0)
        pred = pred.reshape(size, -1)
        target = target.reshape(size, -1)
        mask = mask.reshape(size, -1)

        intersection = (pred * target * mask).sum(dim=1)
        union = (pred * mask).sum(dim=1) + (target * mask).sum(dim=1)
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()


class ConfidentBCELoss(nn.Module):
    """BCE restricted to confident (high/low) target pixels."""

    def __init__(self, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        mask = ((target > self.threshold) | (target < 1 - self.threshold)).float()
        bce_loss = self.bce(pred, target) * mask
        return bce_loss / mask.float().sum().clamp(min=1)

class BCELoss(nn.Module):
    """BCE loss."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)
        
        bce_loss = self.bce(pred, target) * mask
        return bce_loss.sum() / mask.float().sum().clamp(min=1)

class BCEDiceLoss(nn.Module):
    """Sum of masked BCE and Dice loss."""

    def __init__(self, bce_threshold: float = 0.95, dice_smooth: float = 1.0):
        super().__init__()
        self.mask_bce = ConfidentBCELoss(threshold=bce_threshold)
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.mask_bce(pred, target)
        dice_loss = self.dice(pred, target)
        return bce_loss + dice_loss


class WeightedBCEDiceLoss(nn.Module):   
    """Weighted BCE + weighted Dice loss."""
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, dice_smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.mask_bce = BCELoss()
        self.dice = DiceLoss(smooth=dice_smooth)
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)
        bce_loss = self.mask_bce(pred, target, mask)
        dice_loss = self.dice(pred, target, mask)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
    
class MinimizeFeatureSimilarityLoss(nn.Module):
    """Cosine-similarity-based loss between two feature maps (GAP then normalize)."""

    def __init__(self):
        super().__init__()

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        fea1 = F.adaptive_avg_pool2d(feat1, 1).flatten(1)
        fea2 = F.adaptive_avg_pool2d(feat2, 1).flatten(1)
        fea1 = F.normalize(fea1, dim=1)
        fea2 = F.normalize(fea2, dim=1)
        sim = F.cosine_similarity(fea1, fea2, dim=1)
        return (1 + sim).mean()
class MaximizeFeatureSimilarityLoss(nn.Module):
    """Cosine-similarity-based loss between two feature maps (GAP then normalize)."""

    def __init__(self):
        super().__init__()

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        fea1 = F.adaptive_avg_pool2d(feat1, 1).flatten(1)
        fea2 = F.adaptive_avg_pool2d(feat2, 1).flatten(1)
        fea1 = F.normalize(fea1, dim=1)
        fea2 = F.normalize(fea2, dim=1)
        sim = F.cosine_similarity(fea1, fea2, dim=1)
        return (1 - sim).mean()

class StructureLoss(nn.Module):
    """Weighted BCE + weighted IoU on structure (edge-weighted by avg_pool distance from mask).
    Interface: pred and mask are both probabilities in [0, 1] (same as other losses).
    """

    def __init__(self, kernel_size: int = 31, padding: int = 15, weight_scale: float = 5.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight_scale = weight_scale

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred, mask: probabilities [0, 1]
        weit = 1 + self.weight_scale * torch.abs(
            F.avg_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=self.padding) - mask
        )
        wbce = F.binary_cross_entropy(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

class L2Loss(nn.Module):
    """L2 loss between two feature maps (GAP then normalize)."""
    def __init__(self):
        super().__init__()

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        return torch.norm(feat1 - feat2, p=2, dim=1).mean()


class SupContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feature: torch.Tensor, label: torch.Tensor, ignore_label: int = -1) -> torch.Tensor:
        """
        feature: (B, D, H, W) or (N, D)
        label:   (B, H, W)    or (N,)   — 0/1/-1
        """
        # Flatten if spatial tensor
        if feature.dim() == 4:
            B, D, H, W = feature.shape
            feature = feature.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
            label   = label.reshape(-1)                             # (B*H*W,)

        # Remove ignore pixels
        valid_mask = label != ignore_label
        feature    = feature[valid_mask]
        label      = label[valid_mask]

        M = feature.shape[0]
        if M < 2:
            return torch.tensor(0.0, device=feature.device)

        feature  = F.normalize(feature, dim=1)
        sim      = torch.mm(feature, feature.T) / self.temperature
        diag     = torch.eye(M, dtype=torch.bool, device=feature.device)
        pos_mask = (label.unsqueeze(0) == label.unsqueeze(1)) & ~diag ## mat(i, j) = label i == label j

        if not pos_mask.any():
            return torch.tensor(0.0, device=feature.device)

        denom    = torch.logsumexp(sim.masked_fill(diag, float('-inf')), dim=1)
        pos_mean = (sim * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1)

        valid = pos_mask.any(dim=1)
        return (denom - pos_mean)[valid].mean()


class MemoryEfficientSupContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, max_samples: int = 1024):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples

    def forward(self, feature: torch.Tensor, label: torch.Tensor, ignore_label: int = -1) -> torch.Tensor:
        """
        feature: (B, D, H, W) or (N, D)
        label:   (B, H, W)    or (B, 1, H, W) or (N,) — 0 / 1 / ignore_label
        """
        if feature.dim() == 4:
            B, D, H, W = feature.shape
            feature = feature.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
            label = label.reshape(-1)  # (B*H*W,)

        # Remove ignore pixels
        valid_mask = label != ignore_label
        feature = feature[valid_mask]
        label = label[valid_mask]

        if feature.shape[0] < 2:
            return torch.tensor(0.0, device=feature.device, dtype=feature.dtype)

        feature, label = self._balance_samples(feature, label)

        if feature.shape[0] < 2:
            return torch.tensor(0.0, device=feature.device, dtype=feature.dtype)

        M = feature.shape[0]
        feature = F.normalize(feature, dim=1)
        sim = torch.mm(feature, feature.T) / self.temperature
        diag = torch.eye(M, dtype=torch.bool, device=feature.device)
        pos_mask = (label.unsqueeze(0) == label.unsqueeze(1)) & ~diag

        if not pos_mask.any():
            return torch.tensor(0.0, device=feature.device)

        denom = torch.logsumexp(sim.masked_fill(diag, float("-inf")), dim=1)
        pos_mean = (sim * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1)
        valid = pos_mask.any(dim=1)
        return (denom - pos_mean)[valid].mean()

    def _balance_samples(self, feature: torch.Tensor, label: torch.Tensor):
        """Up to max_samples/2 per class (stratified); if only one class, cap at max_samples."""
        if label.numel() == 0:
            return feature, label
        per_class = max(1, self.max_samples // 2)
        indices = []
        classes = label.unique()
        for cls in classes:
            idx = (label == cls).nonzero(as_tuple=True)[0]
            cap = per_class if len(classes) > 1 else self.max_samples
            if len(idx) > cap:
                perm = torch.randperm(len(idx), device=feature.device)[:cap]
                idx = idx[perm]
            indices.append(idx)
        indices = torch.cat(indices)
        return feature[indices], label[indices]