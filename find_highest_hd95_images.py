"""
Script to find top images with highest HD95 scores from a checkpoint.
Usage: python find_highest_hd95_images.py --ckpt <checkpoint_path> --data <data_path>
"""
import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

# Import project modules
import models
from test.eval import ImageFolderDataset
from utils.common import get_proper_device
from torchvision import transforms


class DictToTensor:
    """Transform dict with PIL images to tensors."""
    def __init__(self, resize_size: Optional[int] = None):
        self.resize_size = resize_size

    def __call__(self, data):
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if isinstance(value, Image.Image):
                    if self.resize_size is not None:
                        resample = Image.Resampling.NEAREST if key == 'mask' else Image.Resampling.BILINEAR
                        value = value.resize((self.resize_size, self.resize_size), resample)
                    result[key] = transforms.ToTensor()(value)
                else:
                    result[key] = value
            return result
        if self.resize_size is not None and isinstance(data, Image.Image):
            data = data.resize((self.resize_size, self.resize_size), Image.Resampling.BILINEAR)
        return transforms.ToTensor()(data)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Find top images with highest HD95 scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_highest_hd95_images.py --ckpt ckpt/best_model.pth --data /path/to/data
  python find_highest_hd95_images.py --ckpt ckpt/best_model.pth --data /path/to/data --img-dir Test --mask-dir Test_gt
        """
    )

    parser.add_argument(
        '--ckpt', '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (e.g., ckpt/best_DEMT_DAv2Fusion_addDepthTrainSignal_epoch162.pth)'
    )
    parser.add_argument(
        '--data', '--data-path',
        type=str,
        required=True,
        help='Path to test data root directory'
    )
    parser.add_argument(
        '--img-dir',
        type=str,
        default='Test',
        help='Test image directory name (default: Test)'
    )
    parser.add_argument(
        '--mask-dir',
        type=str,
        default='Test_gt',
        help='Test mask directory name (default: Test_gt)'
    )
    parser.add_argument(
        '--depth-dir',
        type=str,
        default='',
        help='Optional depth directory name under data path. If set, depth is appended in comparison image.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='highest_hd95_images.txt',
        help='Output file path (default: highest_hd95_images.txt)'
    )
    parser.add_argument(
        '--resize',
        type=int,
        default=384,
        help='Resize images to (resize x resize) (default: 384)'
    )
    parser.add_argument(
        '--weights-key',
        type=str,
        default='stu_model',
        choices=['auto', 'stu_model', 'tea_model', 'model', 'raw'],
        help='Which weights key to use from checkpoint (default: stu_model)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='',
        help='Optional explicit model class name in models module'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='Evaluate only first N images (0 = all)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='compare_outputs_hd95',
        help='Directory to save comparison images (default: compare_outputs_hd95)'
    )
    parser.add_argument(
        '--topk-save',
        type=int,
        default=10,
        help='Number of worst HD95 samples to save (default: 10)'
    )

    return parser.parse_args()


def validate_inputs(ckpt_path, data_path, img_dir, mask_dir, depth_dir=None):
    """Validate input paths."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    img_path = os.path.join(data_path, img_dir)
    mask_path = os.path.join(data_path, mask_dir)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image directory not found: {img_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask directory not found: {mask_path}")
    if depth_dir:
        depth_path = os.path.join(data_path, depth_dir)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth directory not found: {depth_path}")

    print(f"✓ Checkpoint: {ckpt_path}")
    print(f"✓ Data path: {data_path}")
    print(f"✓ Image dir: {img_dir}")
    print(f"✓ Mask dir: {mask_dir}")
    if depth_dir:
        print(f"✓ Depth dir: {depth_dir}")


def _pick_state_dict(ckpt, weights_key):
    """Pick weights dict from checkpoint container."""
    if not isinstance(ckpt, dict):
        return ckpt, 'raw'

    if weights_key != 'auto':
        if weights_key == 'raw':
            return ckpt, 'raw'
        if weights_key in ckpt:
            return ckpt[weights_key], weights_key
        raise KeyError(f"weights key '{weights_key}' not found in checkpoint")

    for k in ('stu_model', 'tea_model', 'model'):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k], k
    return ckpt, 'raw'


def _infer_model_name_from_state_dict(state_dict, fallback='ResNet34U_f'):
    """Infer model class from checkpoint parameter names."""
    keys = list(state_dict.keys())
    has_dav2 = any(k.startswith('dav2_encoder.') for k in keys)
    has_rgb_encoder = any(k.startswith('rgb_encoder.') for k in keys)
    has_plain_encoder = any(k.startswith('encoder1.') for k in keys)

    if has_dav2 or has_rgb_encoder:
        return 'DAv2Fusion_ResNet34U_f_EMAEncoderOnly'
    if has_plain_encoder:
        return 'ResNet34U_f'
    return fallback


def _remap_student_to_teacher_if_needed(state_dict, model):
    """Remap encoder keys only when loading student weights into teacher-style model."""
    model_keys = model.state_dict().keys()
    model_expects_rgb = any(k.startswith('rgb_encoder.') for k in model_keys)
    ckpt_has_plain = any(k.startswith('encoder1.') for k in state_dict.keys())
    if not (model_expects_rgb and ckpt_has_plain):
        return state_dict

    print("  Remapping checkpoint keys: encoder1.* -> rgb_encoder.*")
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith('encoder1.'):
            remapped[k.replace('encoder1.', 'rgb_encoder.', 1)] = v
        else:
            remapped[k] = v
    return remapped


def load_checkpoint(ckpt_path, device, weights_key='stu_model'):
    """Load checkpoint and return selected state_dict plus selected key name."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict, selected_key = _pick_state_dict(ckpt, weights_key)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Selected weights '{selected_key}' is not a state_dict")
    return state_dict, selected_key


def create_data_transform(resize_size=384):
    """Create data transform pipeline."""
    return DictToTensor(resize_size=resize_size)


def create_dataset(data_path, test_image_dir, test_mask_dir, resize_size=384):
    """Create test dataset."""
    transform = create_data_transform(resize_size=resize_size)

    img_path = os.path.join(data_path, test_image_dir)
    mask_path = os.path.join(data_path, test_mask_dir)

    if not os.path.exists(img_path):
        raise ValueError(f"Image directory not found: {img_path}")
    if not os.path.exists(mask_path):
        raise ValueError(f"Mask directory not found: {mask_path}")

    dataset = ImageFolderDataset(
        dataset_root=data_path,
        image_dirname=test_image_dir,
        mask_dirname=test_mask_dir,
        transform=transform
    )

    return dataset


def _to_binary_mask(tensor):
    """Convert tensor [1,1,H,W] / [1,H,W] to boolean mask."""
    if tensor.ndim == 4:
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor[0]
    return (tensor >= 0.5).cpu().numpy().astype(bool)


def hd95_score(pred, gt):
    """Compute symmetric HD95 between two binary tensors.

    Convention:
    - both empty: 0
    - one empty, one non-empty: inf
    """
    pred_mask = _to_binary_mask(pred)
    gt_mask = _to_binary_mask(gt)

    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return float('inf')

    pred_surface = pred_mask ^ binary_erosion(pred_mask)
    gt_surface = gt_mask ^ binary_erosion(gt_mask)

    if not pred_surface.any() and not gt_surface.any():
        return 0.0
    if not pred_surface.any() or not gt_surface.any():
        return float('inf')

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    distances = np.concatenate([
        dt_gt[pred_surface],
        dt_pred[gt_surface],
    ])
    if distances.size == 0:
        return float('inf')

    return float(np.percentile(distances, 95))


def evaluate_and_rank_images(model, dataloader, device, max_images=0):
    """Evaluate model and collect per-image HD95 scores."""
    model.eval()

    image_scores = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']

            outputs = model(images)

            for img_idx in range(len(images)):
                if isinstance(outputs, torch.Tensor):
                    single_output = outputs[img_idx:img_idx + 1]
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                    single_output = outputs[0][img_idx:img_idx + 1]
                else:
                    raise TypeError(f"Unsupported model output type: {type(outputs)}")

                single_mask = masks[img_idx:img_idx + 1]
                single_image = images[img_idx:img_idx + 1]
                score = hd95_score(single_output, single_mask)

                image_scores.append({
                    'filename': filenames[img_idx],
                    'hd95': score,
                    'image': single_image.detach().cpu(),
                    'mask': single_mask.detach().cpu(),
                    'pred': single_output.detach().cpu(),
                })
                if max_images > 0 and len(image_scores) >= max_images:
                    break
            if max_images > 0 and len(image_scores) >= max_images:
                break

    image_scores.sort(key=lambda x: x['hd95'], reverse=True)
    return image_scores


def print_results(top_worst):
    """Print results in a formatted table."""
    print('\n' + '=' * 100)
    print('TOP 10 IMAGES WITH HIGHEST HD95 SCORES')
    print('=' * 100)
    print(f"{'Rank':<6} {'Filename':<50} {'HD95':<12}")
    print('-' * 100)

    for rank, item in enumerate(top_worst, 1):
        hd95_text = 'inf' if np.isinf(item['hd95']) else f"{item['hd95']:.6f}"
        print(f"{rank:<6} {item['filename']:<50} {hd95_text:<12}")

    print('=' * 100)
    finite_scores = [x['hd95'] for x in top_worst if np.isfinite(x['hd95'])]
    if finite_scores:
        avg_hd95 = np.mean(finite_scores)
        print(f"\nAverage finite HD95 of top 10 worst: {avg_hd95:.6f}")
    else:
        print('\nAverage finite HD95 of top 10 worst: inf')


def _tensor_to_rgb_uint8(x):
    """Convert tensor [1,C,H,W] or [C,H,W] in [0,1] to RGB uint8 image."""
    if x.ndim == 4:
        x = x[0]
    x = x.float().clamp(0.0, 1.0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return arr


def _tensor_mask_to_uint8(x, threshold=None):
    """Convert mask/pred tensor [1,1,H,W] or [1,H,W] to grayscale uint8."""
    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]
    x = x.float()
    if threshold is not None:
        x = (x >= threshold).float()
    x = x.clamp(0.0, 1.0)
    return (x.numpy() * 255.0).astype(np.uint8)


def _find_depth_file(depth_dir, filename):
    """Return matching depth file path for filename, or None if not found."""
    if not depth_dir:
        return None

    direct_path = os.path.join(depth_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    stem, _ = os.path.splitext(filename)
    for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        candidate = os.path.join(depth_dir, f"{stem}{ext}")
        if os.path.exists(candidate):
            return candidate

    return None


def save_comparison_images(samples, save_dir, depth_dir=None):
    """Save image, pred mask, gt mask, and side-by-side comparison."""
    os.makedirs(save_dir, exist_ok=True)
    for item in samples:
        filename = item['filename']
        stem, _ = os.path.splitext(filename)
        stem = stem.replace(' ', '_')

        image_rgb = _tensor_to_rgb_uint8(item['image'])
        gt_mask = _tensor_mask_to_uint8(item['mask'])
        pred_mask = _tensor_mask_to_uint8(item['pred'], threshold=0.5)

        gt_rgb = np.stack([gt_mask, gt_mask, gt_mask], axis=-1)
        pred_rgb = np.stack([pred_mask, pred_mask, pred_mask], axis=-1)

        panels = [image_rgb]
        depth_file = _find_depth_file(depth_dir, filename)
        if depth_file is not None:
            depth_img = Image.open(depth_file).convert('RGB')
            height, width = image_rgb.shape[0], image_rgb.shape[1]
            depth_img = depth_img.resize((width, height), Image.Resampling.BILINEAR)
            depth_arr = np.array(depth_img, dtype=np.uint8)
            panels.append(depth_arr)

        panels.extend([gt_rgb, pred_rgb])
        compare = np.concatenate(panels, axis=1)

        Image.fromarray(compare).save(os.path.join(save_dir, f"{stem}_compare_hd95_{item['hd95']:.4f}.png"))


def main():
    args = parse_args()

    print('=' * 80)
    print('Find Images with Highest HD95 Scores')
    print('=' * 80)

    try:
        validate_inputs(args.ckpt, args.data, args.img_dir, args.mask_dir, args.depth_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = get_proper_device(device_str)

    print(f"Loading checkpoint: {args.ckpt}")
    try:
        state_dict, selected_key = load_checkpoint(args.ckpt, device, weights_key=args.weights_key)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    print(f"✓ Selected weights key: {selected_key}")

    model_name = args.model_name.strip() or _infer_model_name_from_state_dict(state_dict)
    print(f"Loading model: {model_name}")

    try:
        model = getattr(models, model_name)(num_classes=1).to(device)
    except AttributeError:
        print(f"Warning: Model {model_name} not found. Available models:")
        print(dir(models))
        sys.exit(1)

    state_dict = _remap_student_to_teacher_if_needed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print('✓ Checkpoint loaded successfully')
    print(f"  Missing keys: {len(incompatible_keys.missing_keys)}")
    print(f"  Unexpected keys: {len(incompatible_keys.unexpected_keys)}")

    print(f"Creating dataset from: {args.data}")
    print(f"Resizing images to: {args.resize}x{args.resize}")
    dataset = create_dataset(args.data, args.img_dir, args.mask_dir, resize_size=args.resize)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"✓ Dataset created with {len(dataset)} images")
    if args.max_images > 0:
        print(f"Evaluating first {args.max_images} images")

    print('\nEvaluating...')
    all_scores = evaluate_and_rank_images(model, dataloader, device, max_images=args.max_images)

    top_10_worst = all_scores[:10]
    print_results(top_10_worst)

    num_save = min(args.topk_save, len(all_scores))
    worst_to_save = all_scores[:num_save]
    depth_path = os.path.join(args.data, args.depth_dir) if args.depth_dir else None
    save_comparison_images(worst_to_save, args.save_dir, depth_dir=depth_path)
    print(f"✓ Saved comparison images for {num_save} samples to: {args.save_dir}")

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('TOP 10 IMAGES WITH HIGHEST HD95 SCORES\n')
        f.write('=' * 100 + '\n')
        f.write(f"{'Rank':<6} {'Filename':<50} {'HD95':<12}\n")
        f.write('-' * 100 + '\n')
        for rank, item in enumerate(top_10_worst, 1):
            hd95_text = 'inf' if np.isinf(item['hd95']) else f"{item['hd95']:.6f}"
            f.write(f"{rank:<6} {item['filename']:<50} {hd95_text:<12}\n")
        f.write('=' * 100 + '\n')

        finite_scores = [x['hd95'] for x in top_10_worst if np.isfinite(x['hd95'])]
        if finite_scores:
            f.write(f"\nAverage finite HD95 of top 10 worst: {np.mean(finite_scores):.6f}\n")
        else:
            f.write('\nAverage finite HD95 of top 10 worst: inf\n')
        f.write(f"Total images evaluated: {len(all_scores)}\n")

        finite_all = [x['hd95'] for x in all_scores if np.isfinite(x['hd95'])]
        if finite_all:
            f.write(f"Average finite HD95 (all): {np.mean(finite_all):.6f}\n")
        else:
            f.write('Average finite HD95 (all): inf\n')

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
