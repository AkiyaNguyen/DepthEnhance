"""
Qualitative visualization: load student ResNet34U_f from a Mean-Teacher checkpoint,
run on Kvasir-SEG test folders, pick top-K samples by Dice, save a grid figure.

Checkpoint: expects a dict saved by ExtendMLFlowLoggerHook / Trainer.get_Trainer_ckpt()
with key 'stu_model' (OrderedDict). Plain stu state_dict also works if you pass --raw_stu_weights.

Run (from repo root meanTeacher/):
  python plot_kvasir_test_qualitative.py

Override paths:
  python plot_kvasir_test_qualitative.py --ckpt save_dir/last_DEMT_addDepthTrainSignal_epoch349.pth \\
    --test_root "D:/.../polypDataset_final1/kvasir_SEG/Test" --depth_dirname depth_v2

Report text (figure caption / methods) — color coding of the rightmost panel:
  The overlay is built by binarizing the predicted mask at 0.5 and comparing to the binarized
  ground truth (same rules as test/eval.evaluate). Pixels are colored as:
  - TP (true positive): predicted foreground and GT foreground — green.
  - TN (true negative): predicted background and GT background — dark blue-gray (subtle, large area).
  - FP (false positive): predicted foreground but GT background — red.
  - FN (false negative): predicted background but GT foreground — orange.
  The panel is a dimmed RGB frame with these colors blended on top so structure remains visible.
  Rows are ordered by descending per-image Dice (student output); depth is shown as the mean
  of the three depth channels after the same resize as training (typically 320x320).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Repo imports (run from meanTeacher/)
from data.transform import Resize, ToTensor
import models
from test.eval import ImageFolderDataset, evaluate


def _device_arg(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _to_numpy_chw(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _dice_single(pred: torch.Tensor, gt: torch.Tensor) -> float:
    m = evaluate(pred, gt)
    return float(m["Dice"])


def _binarize_pred_gt(pred: torch.Tensor, gt: torch.Tensor):
    """pred, gt: [1,1,H,W] float on [0,1] / mask convention from evaluate()."""
    pred_b = (pred >= 0.5).float()
    gt_max = gt.max().item()
    gt_b = (gt > 0.5).float() if gt_max > 0.5 else (gt > 0).float()
    return pred_b, gt_b


def _confusion_overlay_rgb(
    image_chw: torch.Tensor,
    pred_b: torch.Tensor,
    gt_b: torch.Tensor,
) -> np.ndarray:
    """
    image_chw: [3,H,W] 0..1
    pred_b, gt_b: [1,1,H,W] binary float
    Returns HxWx3 float 0..1 for imshow.
    """
    img = image_chw.permute(1, 2, 0).numpy().astype(np.float32)
    pb = pred_b.squeeze().numpy() > 0.5
    gb = gt_b.squeeze().numpy() > 0.5
    tp = pb & gb
    tn = (~pb) & (~gb)
    fp = pb & (~gb)
    fn = (~pb) & gb

    out = img * 0.35
    # BGR order in numpy after permute is RGB — use RGB tuples
    green = np.array([0.0, 0.85, 0.15], dtype=np.float32)
    red = np.array([0.95, 0.15, 0.15], dtype=np.float32)
    orange = np.array([1.0, 0.55, 0.1], dtype=np.float32)
    tn_color = np.array([0.2, 0.22, 0.35], dtype=np.float32)

    for mask, col, w in (
        (tn, tn_color, 0.45),
        (tp, green, 0.72),
        (fp, red, 0.78),
        (fn, orange, 0.78),
    ):
        if mask.any():
            out[mask] = out[mask] * (1.0 - w) + col * w
    return np.clip(out, 0.0, 1.0)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    default_test = os.path.join(
        os.path.dirname(root),
        "polypDataset_final1",
        "kvasir_SEG",
        "Test",
    )
    parser = argparse.ArgumentParser(description="Plot top-K Kvasir-SEG test predictions with TP/TN/FP/FN colors.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(root, "save_dir", "last_DEMT_addDepthTrainSignal_epoch349.pth"),
        help="Trainer checkpoint (.pth) containing 'stu_model' or use --raw_stu_weights for state_dict file.",
    )
    parser.add_argument("--raw_stu_weights", action="store_true", help="Checkpoint is only stu_model.state_dict().")
    parser.add_argument("--test_root", type=str, default=default_test, help="Folder with images/, masks/, depth_*/")
    parser.add_argument("--depth_dirname", type=str, default="depth_v2", help="Subfolder under test_root (or '' to skip depth).")
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--resize_h", type=int, default=320)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--out", type=str, default=os.path.join(root, "kvasir_test_topk_qualitative.png"))
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0 ...")
    args = parser.parse_args()

    device = _device_arg(args.device)
    tfm = transforms.Compose([Resize((args.resize_w, args.resize_h)), ToTensor()])

    depth_dir = args.depth_dirname if args.depth_dirname else None
    ds = ImageFolderDataset(
        dataset_root=args.test_root,
        image_dirname="images",
        mask_dirname="masks",
        depth_dirname=depth_dir,
        transform=tfm,
        list_name=None,
    )
    if len(ds) == 0:
        raise SystemExit(f"No images in {args.test_root}/images")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if args.raw_stu_weights:
        stu_sd = ckpt
    else:
        if "stu_model" not in ckpt:
            raise SystemExit("Checkpoint missing 'stu_model'; pass --raw_stu_weights if file is a plain state_dict.")
        stu_sd = ckpt["stu_model"]

    stu = models.ResNet34U_f(num_classes=1).to(device)
    stu.load_state_dict(stu_sd, strict=True)
    stu.eval()

    scores: list[tuple[float, int]] = []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            img = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            pred = stu(img)
            scores.append((_dice_single(pred, mask), i))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[: args.top_k]
    if not top:
        raise SystemExit("No samples scored.")

    nrows = len(top)
    fig, axes = plt.subplots(nrows, 4, figsize=(4 * 3.2, nrows * 3.0))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    for r, (dice_v, idx) in enumerate(top):
        sample = ds[idx]
        img = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        with torch.no_grad():
            pred = stu(img)

        pred_b, gt_b = _binarize_pred_gt(pred, mask)
        img_cpu = sample["image"]
        mask_vis = mask.squeeze().cpu().numpy()
        pred_vis = pred.squeeze().cpu().numpy()

        # RGB
        axes[r, 0].imshow(_to_numpy_chw(img_cpu).transpose(1, 2, 0))
        axes[r, 0].axis("off")

        # Depth (optional)
        ax_d = axes[r, 1]
        if "depth" in sample:
            d = sample["depth"]  # [3,H,W]
            d_gray = d.float().mean(dim=0).numpy()
            ax_d.imshow(d_gray, cmap="magma")
        else:
            ax_d.text(0.5, 0.5, "no depth", ha="center", va="center", transform=ax_d.transAxes)
        ax_d.axis("off")

        axes[r, 2].imshow(mask_vis, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].axis("off")

        overlay = _confusion_overlay_rgb(img_cpu, pred_b.cpu(), gt_b.cpu())
        axes[r, 3].imshow(overlay)
        axes[r, 3].axis("off")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.04, hspace=0.06)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.out}")
    print("Top rows (Dice, filename):")
    for dice_v, idx in top:
        print(f"  Dice={dice_v:.4f}  {ds.image_files[idx]}")


if __name__ == "__main__":
    main()
