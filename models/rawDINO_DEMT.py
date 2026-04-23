import importlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ResUNet import (
    ConvBlock,
    DecoderBlock,
    ResNet34U_f_ExtendDAv2,
    SEFusionBlock,
    encoder,
)


# Local Depth-Anything-V2 package (DPT head + ViT encoder shell).
_DEPTH_ANYTHING_ROOT = Path(__file__).resolve().parents[1] / "Depth-Anything-V2"
if _DEPTH_ANYTHING_ROOT.exists():
    import sys

    root_str = str(_DEPTH_ANYTHING_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

DepthAnythingV2 = importlib.import_module("depth_anything_v2.dpt").DepthAnythingV2


class DEMT_DAv2_Extend_RawDINOv2(nn.Module):
    """
    Same forward path as ``ResNet34U_f_ExtendDAv2`` (proj + SEFusion + U-Net decoder),
    but the frozen geometric backbone is **raw DINOv2** (torch.hub) remapped into the
    Depth-Anything-V2 ViT encoder, instead of the HuggingFace Depth-Anything-V2 backbone.

    Trainer-facing attributes match ``ResNet34U_f_ExtendDAv2``:
    ``rgb_encoder``, ``dav2_encoder``, ``fusion_block*``, ``decoder*``, ``outconv``.
    """

    # DepthAnythingV2 ``encoder=`` id → DPT stem config (must match local depth_anything_v2).
    _encoder_cfg: Dict[str, Dict[str, object]] = {
        "vits": {
            "features": 64,
            "out_channels": [48, 96, 192, 384],
            "hub_model": "dinov2_vits14",
        },
        "vitb": {
            "features": 128,
            "out_channels": [96, 192, 384, 768],
            "hub_model": "dinov2_vitb14",
        },
        "vitl": {
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
            "hub_model": "dinov2_vitl14",
        },
        "vitg": {
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
            "hub_model": "dinov2_vitg14",
        },
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
        load_raw_dino: bool = True,
        raw_dino_source: str = "facebookresearch/dinov2",
        raw_dino_model: Optional[str] = None,
        dino_weights_path: Optional[str] = None,
        freeze_dino: bool = True,
        dav2_model_name: Optional[str] = None,
    ):
        super().__init__()

        if dav2_model_name is not None and dav2_model_name in self._dav2_to_raw_encoder:
            raw_dino_encoder = str(self._dav2_to_raw_encoder[dav2_model_name])
        if raw_dino_encoder not in self._encoder_cfg:
            raise ValueError(
                f"Invalid raw_dino_encoder={raw_dino_encoder!r}. Allowed: {list(self._encoder_cfg.keys())}"
            )

        enc_cfg = self._encoder_cfg[raw_dino_encoder]

        if dav2_model_name is not None and dav2_model_name in ResNet34U_f_ExtendDAv2.dav2_choice:
            fusion_cfg = ResNet34U_f_ExtendDAv2.dav2_choice[dav2_model_name]
            self.dav2_dim = int(fusion_cfg["dav2_dim"])
            self.dav2_layer_indices: List[int] = list(fusion_cfg["layer_indices"])
            self.proj_channels: List[int] = list(fusion_cfg["proj_channels"])
        else:
            # Legacy: infer channel width from encoder variant (no HF id).
            legacy_dim = {k: int(v["out_channels"][-1]) for k, v in self._encoder_cfg.items()}
            self.dav2_dim = legacy_dim[raw_dino_encoder]
            n_blocks = {"vits": 12, "vitb": 12, "vitl": 24, "vitg": 40}[raw_dino_encoder]
            step = max((n_blocks - 1) // 4, 1)
            self.dav2_layer_indices = [step * i for i in range(4)]
            self.proj_channels = [512, 256, 128, 64]

        self.rgb_encoder = encoder(num_classes=None)

        self.raw_dino_encoder = raw_dino_encoder
        self.depth_anything = DepthAnythingV2(
            encoder=raw_dino_encoder,
            features=int(enc_cfg["features"]),
            out_channels=list(enc_cfg["out_channels"]),
        )
        self._load_raw_dino_weights(
            load_raw_dino=load_raw_dino,
            raw_dino_source=raw_dino_source,
            raw_dino_model=raw_dino_model or str(enc_cfg["hub_model"]),
            dino_weights_path=dino_weights_path,
        )

        self.dav2_encoder = self.depth_anything.pretrained
        if freeze_dino:
            for p in self.dav2_encoder.parameters():
                p.requires_grad = False

        n_blocks = len(self.dav2_encoder.blocks)
        self._dino_intermediate_idx = [min(int(i), n_blocks - 1) for i in self.dav2_layer_indices]

        self.proj5 = self._make_proj(self.dav2_dim, self.proj_channels[0])
        self.proj4 = self._make_proj(self.dav2_dim, self.proj_channels[1])
        self.proj3 = self._make_proj(self.dav2_dim, self.proj_channels[2])
        self.proj2 = self._make_proj(self.dav2_dim, self.proj_channels[3])

        self.fusion_block5 = SEFusionBlock(512, self.proj_channels[0], 512)
        self.fusion_block4 = SEFusionBlock(256, self.proj_channels[1], 256)
        self.fusion_block3 = SEFusionBlock(128, self.proj_channels[2], 128)
        self.fusion_block2 = SEFusionBlock(64, self.proj_channels[3], 64)

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

    def _make_proj(self, in_dim, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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

    def _extract_raw_dino_features(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_size = 14
        _, _, h, w = x.shape
        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.no_grad():
            hidden_states = self.dav2_encoder.get_intermediate_layers(
                x,
                self._dino_intermediate_idx,
                reshape=False,
                return_class_token=False,
                norm=True,
            )

        feats = []
        for h in hidden_states:
            b, n, d = h.shape
            side = int(math.sqrt(n))
            if side * side != n:
                raise ValueError(f"Patch token count {n} is not a perfect square.")
            h = h.permute(0, 2, 1).reshape(b, d, side, side)
            feats.append(h)
        return tuple(feats)

    def forward(self, x, fp=False, feature_layers=1, type="mixed"):
        e1, e2, e3, e4, e5 = self.rgb_encoder(x)

        dav2_feats = self._extract_raw_dino_features(x)

        def proj_and_resize(proj_layer, feat, target_feat):
            out = proj_layer(feat)
            return F.interpolate(
                out,
                size=target_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d2 = proj_and_resize(self.proj2, dav2_feats[0], e2)
        d3 = proj_and_resize(self.proj3, dav2_feats[1], e3)
        d4 = proj_and_resize(self.proj4, dav2_feats[2], e4)
        d5 = proj_and_resize(self.proj5, dav2_feats[3], e5)

        f5 = self.fusion_block5(e5, d5)
        f4 = self.fusion_block4(e4, d4)
        f3 = self.fusion_block3(e3, d3)
        f2 = self.fusion_block2(e2, d2)

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
            if type == "decoder":
                return final_output, decoder_fea_layers[feature_layers - 1]
            if type == "rgb_encoder":
                return final_output, rgb_encoder_fea_layers[feature_layers - 1]
            if type == "mix_encoder":
                return final_output, mix_encoder_fea_layers[feature_layers - 1]
            raise ValueError(
                f"Invalid type: {type}, allowed types are 'decoder', 'rgb_encoder', 'mix_encoder'"
            )

        return final_output
