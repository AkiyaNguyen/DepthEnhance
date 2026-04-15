import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PrototypeNCE_FG_BG", "PrototypeNCE_FG_BG_BD"]


def _zero_scalar(fea: torch.Tensor) -> torch.Tensor:
    return fea.sum() * 0.0


class PrototypeNCE_FG_BG(nn.Module):
    def __init__(self, temperature: float = 0.1, max_samples: int = 256):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples

    def forward(self, fg_fea: torch.Tensor, bg_fea: torch.Tensor, **kwargs) -> torch.Tensor:
        if fg_fea.shape[0] == 0 or bg_fea.shape[0] == 0:
            z = _zero_scalar(fg_fea if fg_fea.numel() else bg_fea)
            return {'loss_fg': z, 'loss_bg': z, 'loss_bd': z, 'loss': z}

        fg = F.normalize(fg_fea, dim=1)
        bg = F.normalize(bg_fea, dim=1)

        ## normalize after take mean
        fg_proto = F.normalize(fg.mean(dim=0, keepdim=True), dim=1).squeeze(0)  # (C,)
        bg_proto = F.normalize(bg.mean(dim=0, keepdim=True), dim=1).squeeze(0)  # (C,)

        fg_s = self._sample(fg)
        bg_s = self._sample(bg)

        loss_fg = self._nce(fg_s, fg_proto, bg_proto)
        loss_bg = self._nce(bg_s, bg_proto, fg_proto)
        z_bd = _zero_scalar(loss_fg)
        return {'loss_fg': loss_fg, 'loss_bg': loss_bg, 'loss_bd': z_bd, 'loss': loss_fg + loss_bg}

    def _sample(self, fea: torch.Tensor) -> torch.Tensor:
        N = fea.shape[0]
        if N <= self.max_samples:
            return fea
        return fea[torch.randperm(N, device=fea.device)[:self.max_samples]]

    def _nce(self, anchors: torch.Tensor, pos_proto: torch.Tensor, neg_proto: torch.Tensor) -> torch.Tensor:
        """
        anchors:   (N, C) normalized
        pos_proto: (C,)   L2-normalized prototype
        neg_proto: (C,)   L2-normalized prototype
        """
        if anchors.shape[0] == 0:
            return anchors.sum()
        sim_pos = (anchors * pos_proto).sum(dim=1, keepdim=True) / self.temperature  # (N, 1)
        sim_neg = (anchors * neg_proto).sum(dim=1, keepdim=True) / self.temperature  # (N, 1)
        logits = torch.cat([sim_pos, sim_neg], dim=1)  # (N, 2)
        targets = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
        return F.cross_entropy(logits, targets)


class PrototypeNCE_FG_BG_BD(nn.Module):
    def __init__(self, temperature: float = 0.1, max_samples: int = 256):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples

    def forward(self, fg_fea: torch.Tensor, bg_fea: torch.Tensor, bd_fea: torch.Tensor, **kwargs):
        if fg_fea.shape[0] == 0 or bg_fea.shape[0] == 0:
            z = _zero_scalar(fg_fea if fg_fea.numel() else bg_fea)
            return {'loss_fg': z, 'loss_bg': z, 'loss_bd': z, 'loss': z}

        fg = F.normalize(fg_fea, dim=1)
        bg = F.normalize(bg_fea, dim=1)

        fg_proto = F.normalize(fg.mean(dim=0, keepdim=True), dim=1).squeeze(0)  # (C,)
        bg_proto = F.normalize(bg.mean(dim=0, keepdim=True), dim=1).squeeze(0)  # (C,)

        loss_fg = self._nce(self._sample(fg), pos_proto=fg_proto, neg_protos=bg_proto.unsqueeze(0))
        loss_bg = self._nce(self._sample(bg), pos_proto=bg_proto, neg_protos=fg_proto.unsqueeze(0))

        z_bd = _zero_scalar(loss_fg)
        if bd_fea.shape[0] == 0:
            loss = loss_fg + loss_bg
            return {'loss_fg': loss_fg, 'loss_bg': loss_bg, 'loss_bd': z_bd, 'loss': loss}

        bd = F.normalize(bd_fea, dim=1)
        bd_proto = F.normalize(bd.mean(dim=0, keepdim=True), dim=1).squeeze(0)  # (C,)
        neg_for_bd = torch.stack([fg_proto, bg_proto], dim=0)  # (2, C)
        loss_bd = self._nce(self._sample(bd), pos_proto=bd_proto, neg_protos=neg_for_bd)

        loss = loss_fg + loss_bg + loss_bd
        return {'loss_fg': loss_fg, 'loss_bg': loss_bg, 'loss_bd': loss_bd, 'loss': loss}

    def _sample(self, fea: torch.Tensor) -> torch.Tensor:
        N = fea.shape[0]
        if N <= self.max_samples:
            return fea
        return fea[torch.randperm(N, device=fea.device)[:self.max_samples]]

    def _nce(self, anchors: torch.Tensor, pos_proto: torch.Tensor, neg_protos: torch.Tensor) -> torch.Tensor:
        """
        anchors:    (N, C)    normalized
        pos_proto:  (C,)      normalized mean
        neg_protos: (M, C)    L2-normalized prototypes, M >= 1
        """
        if anchors.shape[0] == 0:
            return anchors.sum()
        sim_pos = (anchors @ pos_proto).unsqueeze(1) / self.temperature  # (N, 1)
        sim_neg = anchors @ neg_protos.T / self.temperature  # (N, M)
        logits = torch.cat([sim_pos, sim_neg], dim=1)  # (N, 1+M)
        targets = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
        loss = F.cross_entropy(logits, targets)
        return loss