# BARD Implementation Diff (so với bản gốc Relational)

Tài liệu này chỉ ra các phần **được thêm/sửa** khi nâng từ Relational Distillation sang Boundary-Aware Relational Distillation (BARD).

## 1. Đổi tên class/function chính

### Trước
```python
class DAv2RelationalTrainer(Trainer):
```

```python
class MeanTeacherEvalHook_DAv2_Relational(EvalHook):
```

```python
def training_relational(cfg: Config, trial: typing.Optional[optuna.trial.Trial] = None):
```

### Sau
```python
class DAv2_BARD_Trainer(Trainer):
```

```python
class MeanTeacherEvalHook_DAv2_BARD(EvalHook):
```

```python
def training_bard(cfg: Config, trial: typing.Optional[optuna.trial.Trial] = None):
```

## 2. Thêm hàm trích xuất biên từ mask GT

### Mới thêm trong Trainer
```python
def _get_boundary_map(self, mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Extract boundary map using morphological gradient: dilation - erosion."""
    pad = kernel_size // 2
    mask = (mask > 0.5).float()
    dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilation - erosion).clamp(min=0.0, max=1.0)
    return boundary
```

Ý nghĩa:
- Dùng morphological gradient để lấy vùng biên đối tượng.
- Không cần dependency ngoài, chỉ dùng toán tử pooling của PyTorch.

## 3. Nâng cấp relational loss thành boundary-aware weighted MSE

### Trước (Relational gốc)
```python
stu_relation = self._compute_spatial_relation(stu_labeled_feat)
tea_relation = self._compute_spatial_relation(tea_dav2_feat)
relational_loss = F.mse_loss(stu_relation, tea_relation)
```

### Sau (BARD)
```python
_, _, tea_h, tea_w = tea_dav2_feat.shape
if stu_labeled_feat.shape[-2:] != (tea_h, tea_w):
    stu_labeled_feat = F.interpolate(
        stu_labeled_feat,
        size=(tea_h, tea_w),
        mode='bilinear',
        align_corners=False,
    )

boundary_map = self._get_boundary_map(label[:self.labeled_bs], kernel_size=5)
boundary_map = F.interpolate(boundary_map, size=(tea_h, tea_w), mode='nearest')
boundary_flat = boundary_map.flatten(start_dim=1)
attention_w = torch.maximum(boundary_flat.unsqueeze(2), boundary_flat.unsqueeze(1))

stu_relation = self._compute_spatial_relation(stu_labeled_feat)
tea_relation = self._compute_spatial_relation(tea_dav2_feat)

relation_mse = F.mse_loss(stu_relation, tea_relation, reduction='none')
relational_loss = (relation_mse * attention_w).mean()
```

Điểm mới:
- Lấy boundary từ GT có nhãn: `label[:self.labeled_bs]`.
- Resize boundary về cùng kích thước không gian với teacher feature: `(tea_h, tea_w)`.
- Tạo ma trận chú ý 3D `W` kích thước `(B, N, N)` với:
  - `W_{i,j} = max(boundary_i, boundary_j)`
- Dùng weighted MSE để nhấn mạnh quan hệ có liên quan đến biên.

## 4. Căn chỉnh shape để đảm bảo nhân ma trận attention hợp lệ

Các shape sau khi xử lý:
- `stu_relation`: `(B, N, N)`
- `tea_relation`: `(B, N, N)`
- `relation_mse`: `(B, N, N)`
- `attention_w`: `(B, N, N)`

Điều này đảm bảo phép nhân phần tử:
```python
relation_mse * attention_w
```
luôn hợp lệ theo đúng yêu cầu toán học.

## 5. Cập nhật đường gọi train/eval trong main

### Hook eval dùng class mới
```python
hook_builder(MeanTeacherEvalHook_DAv2_BARD, ...)
```

### Entrypoint gọi function mới
```python
score = training_bard(cfg)
```

## 6. Bổ sung import còn thiếu

Trong file BARD đã thêm:
```python
from torch.optim.lr_scheduler import CosineAnnealingLR
```

Lý do:
- Script sử dụng `CosineAnnealingLR` để tạo scheduler, nên cần import tường minh.

## File liên quan

- File gốc tham chiếu: [DEMT_DAv2_Relational.py](DEMT_DAv2_Relational.py)
- File mới BARD: [DEMT_DAv2_BARD.py](DEMT_DAv2_BARD.py)
