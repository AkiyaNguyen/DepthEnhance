# Structure-Preserving Relational Distillation for DAv2-guided Mean Teacher

## 1. Muc tieu (Objective)

Muc tieu cua phuong phap nay la buoc Student (chi nhan RGB) hoc duoc cau truc khong gian cua doi tuong tu Teacher su dung DAv2 (geometric prior tu depth backbone).

Khac voi feature distillation thong thuong (ep gia tri feature map phai giong nhau), relational distillation chi ep moi quan he giua cac vi tri khong gian trong noi bo feature map phai tuong dong. Nho do, Student van co tu do hoc dac trung 2D, nhung van giu duoc hinh thai cau truc theo neo 3D cua Teacher.

## 2. Chi tiet trien khai (Implementation Details)

### 2.1. Trainer moi

Trainer moi: `DAv2RelationalTrainer`

- Mo rong tu flow Mean Teacher hien co.
- Them loss moi trong PHASE 1: `relational_loss`.
- He so dieu khien: `relational_weight = 0.1`.

### 2.2. Trich xuat feature

Can 2 feature map sau cung de so sanh cau truc:

1. Student feature:
- Lay bang `register_forward_hook` tren encoder sau cung cua student.
- Khong can sua class model goc.

2. Teacher feature:
- DAv2 backbone duoc dong bang hoan toan (`requires_grad=False`).
- Feature depth duoc lay tu nhanh DAv2/projection trong luong forward cua teacher.
- Neu can, chuyen ve dang 4D feature map va dong bo kich thuoc khong gian voi student bang `F.interpolate(..., mode='bilinear')`.

### 2.3. Spatial Relation Matrix

Ham cot loi:

```python
_compute_spatial_relation(feat)
```

Input: `feat` co dang `(B, C, H, W)`.

Buoc tinh:

1. L2 normalization theo channel:

$$
\hat{F} = \text{normalize}(F, p=2, \text{dim}=1)
$$

2. Flatten khong gian:

$$
\hat{F} \in \mathbb{R}^{B \times C \times H \times W} \rightarrow \hat{F}_{flat} \in \mathbb{R}^{B \times C \times N},\; N=H\times W
$$

3. Tinh relation matrix:

$$
R = \hat{F}_{flat}^{\top} \hat{F}_{flat},\; R \in \mathbb{R}^{B \times N \times N}
$$

Phan tu $R_{i,j}$ bieu dien do tuong dong giua pixel $i$ va $j$.

### 2.4. Relational loss va data flow

Relational distillation chi dien ra o PHASE 1.

- Chi tinh tren phan labeled de student-teacher nhin cung noi dung:
	- `stu_feat[:labeled_bs]`
	- `tea_feat[:labeled_bs]`
- Loss:

$$
\mathcal{L}_{rel} = \text{MSE}(R_{stu}, R_{tea})
$$

- Tong loss PHASE 1:

$$
\mathcal{L}_{total} = \mathcal{L}_{sup} + w_c\mathcal{L}_{cons} + \mathcal{L}_{cutmix} + \lambda_{rel}\mathcal{L}_{rel}
$$

voi $\lambda_{rel} = \text{relational\_weight}$.

## 3. Bao toan logic goc

Ban mo rong nay giu nguyen flow Mean Teacher goc:

1. PHASE 1 (Student update)
- Van giu `loss_sup`, `loss_consist_rgbd`, `loss_consist_rgbd_cutmix`.
- Van cap nhat EMA tu student encoder sang teacher rgb encoder.
- Chi them mot thanh phan moi: `relational_loss`.

2. PHASE 2 (Teacher update)
- Khong them distillation vao phase nay.
- Van giu logic `loss_tea_sup` + `depth_learn_from_stu_loss`.
- Van khoa thanh phan backbone depth theo dung thiet ke.

## 4. Monitoring va logging

Can theo doi:

- `phase1_relational_loss`
- Cac metric goc: `*loss*`, `*Dice*`, `*IoU*`, `*ACC_overall*`, `*lr*`

Neu `phase1_relational_loss` lien tuc bang 0.0, can kiem tra lai duong lay teacher feature (hook/module xuat feature) va shape alignment.

## 5. Ket luan

Day la mot clean extension cho Mean Teacher:

- Khong pha vo training pipeline goc.
- Bo sung rang buoc cau truc khong gian tu depth prior cua DAv2.
- Giup Student RGB hoc hinh thai doi tuong ben vung hon so voi chi distill theo gia tri feature truc tiep.