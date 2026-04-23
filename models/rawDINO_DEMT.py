import math
import importlib
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ResUNet import encoder, DecoderBlock, ConvBlock, BiFusionBlock


# Make local Depth-Anything-V2 package importable.
_DEPTH_ANYTHING_ROOT = Path(__file__).resolve().parents[1] / "Depth-Anything-V2"
if _DEPTH_ANYTHING_ROOT.exists():
    import sys
    root_str = str(_DEPTH_ANYTHING_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

DepthAnythingV2 = importlib.import_module("depth_anything_v2.dpt").DepthAnythingV2


class DEMT_DAv2_Extend_RawDINOv2(nn.Module):
    """
    BiFusion teacher that uses raw DINOv2 weights as transformer encoder source.

    This class keeps the same interface/attributes used by existing trainer code:
    - rgb_encoder
    - dav2_encoder
    - decoder blocks + outconv
    """

    _encoder_cfg: Dict[str, Dict[str, object]] = {
        "vits": {"dav2_dim": 384, "features": 64, "out_channels": [48, 96, 192, 384], "layer_indices": [2, 5, 8, 11], "hub_model": "dinov2_vits14"},
        "vitb": {"dav2_dim": 768, "features": 128, "out_channels": [96, 192, 384, 768], "layer_indices": [2, 5, 8, 11], "hub_model": "dinov2_vitb14"},
        "vitl": {"dav2_dim": 1024, "features": 256, "out_channels": [256, 512, 1024, 1024], "layer_indices": [4, 11, 17, 23], "hub_model": "dinov2_vitl14"},
        "vitg": {"dav2_dim": 1536, "features": 384, "out_channels": [1536, 1536, 1536, 1536], "layer_indices": [9, 19, 29, 39], "hub_model": "dinov2_vitg14"},
    }

    _dav2_to_raw_encoder = {
        "depth-anything/Depth-Anything-V2-Small-hf": "vits",
        "depth-anything/Depth-Anything-V2-Base-hf": "vitb",
    }

    def __init__(
        self,
        num_classes,
        dropout=0.1,
        raw_dino_encoder: str = "vitl",
        use_cnn_residual: bool = True,
        bifusion_drop: float = 0.0,
        load_raw_dino: bool = True,
        raw_dino_source: str = "facebookresearch/dinov2",
        raw_dino_model: Optional[str] = None,
        dino_weights_path: Optional[str] = None,
        freeze_dino: bool = True,
        dav2_model_name: Optional[str] = None,
    ):
        super().__init__()

        if dav2_model_name in self._dav2_to_raw_encoder:
            raw_dino_encoder = self._dav2_to_raw_encoder[dav2_model_name]
        if raw_dino_encoder not in self._encoder_cfg:
            raise ValueError(f"Invalid raw_dino_encoder={raw_dino_encoder!r}. Allowed: {list(self._encoder_cfg.keys())}")

        cfg = self._encoder_cfg[raw_dino_encoder]
        self.use_cnn_residual = use_cnn_residual

        self.rgb_encoder = encoder(num_classes=None)

        self.raw_dino_encoder = raw_dino_encoder
        self.depth_anything = DepthAnythingV2(
            encoder=raw_dino_encoder,
            features=int(cfg["features"]),
            out_channels=list(cfg["out_channels"]),
        )
        self._load_raw_dino_weights(
            load_raw_dino=load_raw_dino,
            raw_dino_source=raw_dino_source,
            raw_dino_model=raw_dino_model or str(cfg["hub_model"]),
            dino_weights_path=dino_weights_path,
        )

        # Keep trainer compatibility with existing freeze path and attribute names.
        self.dav2_encoder = self.depth_anything.pretrained
        if freeze_dino:
            for p in self.dav2_encoder.parameters():
                p.requires_grad = False

        self.dav2_dim = int(cfg["dav2_dim"])
        self.dav2_layer_indices = list(cfg["layer_indices"])

        dv2 = self.dav2_dim
        self.pre_align = nn.ModuleList(
            [
                self._make_pre_align(dv2, dv2),
                self._make_pre_align(dv2, dv2),
                self._make_pre_align(dv2, dv2),
                self._make_pre_align(dv2, dv2),
            ]
        )
        self.post_align = nn.ModuleList(
            [
                self._make_post_align(dv2, dv2),
                self._make_post_align(dv2, dv2),
                self._make_post_align(dv2, dv2),
                self._make_post_align(dv2, dv2),
            ]
        )

        self.fusion_block5 = BiFusionBlock(512, dv2, r_2=16, ch_int=256, ch_out=512, drop_rate=bifusion_drop)
        self.fusion_block4 = BiFusionBlock(256, dv2, r_2=16, ch_int=128, ch_out=256, drop_rate=bifusion_drop)
        self.fusion_block3 = BiFusionBlock(128, dv2, r_2=16, ch_int=64, ch_out=128, drop_rate=bifusion_drop)
        self.fusion_block2 = BiFusionBlock(64, dv2, r_2=8, ch_int=32, ch_out=64, drop_rate=bifusion_drop)

        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder2 = DecoderBlock(128 + 64, 64)
        self.decoder1 = DecoderBlock(64 + 64, 64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )

    def _load_raw_dino_weights(
        self,
        load_raw_dino: bool,
        raw_dino_source: str,
        raw_dino_model: str,
        dino_weights_path: Optional[str],
    ) -> None:
        if not load_raw_dino:
            return

        if dino_weights_path:
            state = torch.load(dino_weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                dino_state = state["state_dict"]
            else:
                dino_state = state
        else:
            dino_model = torch.hub.load(raw_dino_source, raw_dino_model)
            dino_state = dino_model.state_dict()

        remapped = {f"pretrained.{k}": v for k, v in dino_state.items()}
        missing, unexpected = self.depth_anything.load_state_dict(remapped, strict=False)
        print(
            f"[DEMT_DAv2_Extend_RawDINOv2] Loaded raw DINOv2 weights from "
            f"{dino_weights_path or (raw_dino_source + ':' + raw_dino_model)} | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    def _make_pre_align(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            ConvBlock(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )

    def _make_post_align(self, in_dim, out_dim):
        return ConvBlock(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def _extract_dino_features(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # DINOv2 patch embed requires H and W divisible by patch size (14 here).
        # We pad only for the transformer branch; fused features are resized back to CNN scales later.
        patch_size = 14
        _, _, h, w = x.shape
        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.no_grad():
            hidden_states = self.dav2_encoder.get_intermediate_layers(
                x,
                self.dav2_layer_indices,
                reshape=False,
                return_class_token=False,
                norm=True,
            )

        feats = []
        for h in hidden_states:
            # h: [B, N, D] where N is patch token count.
            B, N, D = h.shape
            side = int(math.sqrt(N))
            if side * side != N:
                raise ValueError(f"Patch token count {N} is not a perfect square.")
            h = h.permute(0, 2, 1).reshape(B, D, side, side)
            feats.append(h)
        return tuple(feats)

    def forward(self, x, fp=False, feature_layers=1, type='mixed'):
        e1, e2, e3, e4, e5 = self.rgb_encoder(x)

        dino_feats = self._extract_dino_features(x)

        def align_stage(stage_idx, feat, target_spatial):
            y = self.pre_align[stage_idx](feat)
            y = F.interpolate(y, size=target_spatial, mode="bilinear", align_corners=False)
            return self.post_align[stage_idx](y)

        d2 = align_stage(0, dino_feats[0], e2.shape[2:])
        d3 = align_stage(1, dino_feats[1], e3.shape[2:])
        d4 = align_stage(2, dino_feats[2], e4.shape[2:])
        d5 = align_stage(3, dino_feats[3], e5.shape[2:])

        f5 = self.fusion_block5(e5, d5)
        f4 = self.fusion_block4(e4, d4)
        f3 = self.fusion_block3(e3, d3)
        f2 = self.fusion_block2(e2, d2)
        if self.use_cnn_residual:
            f5 = f5 + e5
            f4 = f4 + e4
            f3 = f3 + e3
            f2 = f2 + e2

        dec5 = self.decoder5(f5)
        dec4 = self.decoder4(torch.cat([dec5, f4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, f3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, f2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, e1], dim=1))

        decoder_fea_layers = [dec1, dec2, dec3, dec4, dec5]
        rgb_encoder_fea_layers = [e1, e2, e3, e4, e5]
        mix_encoder_fea_layers = [e1, f2, f3, f4, f5]
        out = self.outconv(dec1)
        final_output = torch.sigmoid(out)

        if fp:
            if type == 'decoder':
                return final_output, decoder_fea_layers[feature_layers - 1]
            elif type == 'rgb_encoder':
                return final_output, rgb_encoder_fea_layers[feature_layers - 1]
            elif type == 'mix_encoder':
                return final_output, mix_encoder_fea_layers[feature_layers - 1]
            else:
                raise ValueError("Invalid type: choose from 'decoder', 'rgb_encoder', 'mix_encoder'")

        return final_output
