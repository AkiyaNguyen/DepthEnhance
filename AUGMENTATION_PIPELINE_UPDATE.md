# Augmentation Pipeline Update

This document contains the updated code blocks for the 3 requested changes:

1. Enable `Normalization` in train/val/test flows.
2. Add `CLAHE` to strong augmentation (`image_s`) branch.
3. Add `GridDistortion` equivalent to strong augmentation (`image_s`) branch.

## 1) Updated `Normalization` class

File: `data/transform.py`

```python
class Normalization(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        normalize = transforms.Normalize(self.mean, self.std)
        result = {}
        for key, value in sample.items():
            # Normalize only image-like tensors and keep labels/masks untouched.
            if key in {'image', 'image_s'} and torch.is_tensor(value) and value.dim() == 3 and value.size(0) == len(self.mean):
                result[key] = normalize(value)
            else:
                result[key] = value
        return result
```

## 2) Added CLAHE + GridDistortion-equivalent for strong branch (`image_s`)

File: `data/dataset.py`

### New helper functions

```python
def apply_clahe_rgb(img: Image.Image, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8), p: float = 0.3) -> Image.Image:
    if random.random() >= p:
        return img
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def apply_grid_distortion_rgb(img: Image.Image, num_steps: int = 5, distort_limit: float = 0.3, p: float = 0.3) -> Image.Image:
    """Approximate Albumentations GridDistortion for RGB PIL images."""
    if random.random() >= p:
        return img

    img_np = np.array(img)
    h, w = img_np.shape[:2]
    x_step = max(1, w // num_steps)
    y_step = max(1, h // num_steps)

    xx = np.zeros(w, np.float32)
    prev = 0
    for i in range(num_steps):
        start = i * x_step
        end = w if i == num_steps - 1 else (i + 1) * x_step
        cur_len = end - start
        scale = 1.0 + random.uniform(-distort_limit, distort_limit)
        step = cur_len * scale
        xx[start:end] = np.linspace(prev, prev + step, cur_len, endpoint=False)
        prev += step

    yy = np.zeros(h, np.float32)
    prev = 0
    for i in range(num_steps):
        start = i * y_step
        end = h if i == num_steps - 1 else (i + 1) * y_step
        cur_len = end - start
        scale = 1.0 + random.uniform(-distort_limit, distort_limit)
        step = cur_len * scale
        yy[start:end] = np.linspace(prev, prev + step, cur_len, endpoint=False)
        prev += step

    map_x = np.tile(np.clip(xx, 0, w - 1), (h, 1)).astype(np.float32)
    map_y = np.tile(np.clip(yy, 0, h - 1).reshape(-1, 1), (1, w)).astype(np.float32)
    out = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return Image.fromarray(out)
```

### Updated strong augmentation branch

```python
if self.mode == 'train':
    img_s1 = Image.fromarray(np.array((data['image'])).astype(np.uint8))
    if random.random() < 0.8:
        img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
    img_s1 = apply_clahe_rgb(img_s1, clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
    img_s1 = apply_grid_distortion_rgb(img_s1, num_steps=5, distort_limit=0.3, p=0.3)
    img_s1 = blur(img_s1, p=0.5)
    data['image_s'] = img_s1  # type: ignore
```

## 3) Applied Normalization after `ToTensor` in dataset returns

File: `data/dataset.py`

### `kvasir_SEG.__init__`

```python
self.transform = transform
self.normalize = Normalization()
```

### `kvasir_SEG.__getitem__`

```python
data = ToTensor()(data)
data = self.normalize(data)

return {**data, 'id': self.id_list[index]}
```

## 4) Validation/Test transforms updated to `Resize + ToTensor + Normalization`

### `utils/build_dataset.py`

```python
from data.transform import Resize, ToTensor, Normalization

val_test_transform = transforms.Compose([
    Resize((resize_w, resize_h)),
    ToTensor(),
    Normalization(),
])
```

### `utils/build_dataset_supervised.py`

```python
from data.transform import Resize, ToTensor, Normalization

val_test_transform = transforms.Compose([
    Resize((resize_w, resize_h)),
    ToTensor(),
    Normalization(),
])
```

### `utils/build_dataset_image_as_depth.py`

```python
from data.transform import Resize, ToTensor, Normalization

val_test_transform = transforms.Compose([
    Resize((resize_w, resize_h)),
    ToTensor(),
    Normalization(),
])
```

### `utils/build_dataset_depth_only.py`

```python
from data.transform import Resize, ToTensor, Normalization

val_test_transform = transforms.Compose([
    Resize((resize_w, resize_h)),
    ToTensor(),
    Normalization(),
])
```

### `inference_kvasir.py`

```python
from data.transform import Resize, ToTensor, Normalization

val_test_transform = transforms.Compose([
    Resize((resize_w, resize_h)),
    ToTensor(),
    Normalization(),
])
```

## Notes

- Teacher branch `image` in train remains without strong augmentations.
- Val/test branches now use only `Resize + ToTensor + Normalization`.
- Strong branch changes are only on `image_s`.
- `GridDistortion` is implemented as an in-repo equivalent using OpenCV remap to avoid introducing a hard Albumentations dependency.
