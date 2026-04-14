# Changes Summary (Current Working Tree)

Date: 2026-04-14

## 1) data/transform.py

### RandomZoom refactor and bug fixes
- Replaced per-item random zoom with a single shared zoom factor per sample to keep image and mask spatially aligned.
- Added interpolation order by key:
  - label/mask use nearest neighbor (order=0)
  - other keys use bilinear-like interpolation (order=1)
- Updated zoom tuple handling by dimensionality:
  - 2D arrays use (zoom_factor, zoom_factor)
  - 3D arrays use (zoom_factor, zoom_factor, 1)
- Updated clipped_zoom signature to accept order explicitly and pass it into scipy.ndimage.zoom for both zoom-in and zoom-out branches.

### Resize state
- Resize now pads each sample to a square before resizing:
  - symmetric zero padding is applied on the shorter side
  - `label` and `mask` use nearest-neighbor interpolation
  - other keys use bilinear interpolation