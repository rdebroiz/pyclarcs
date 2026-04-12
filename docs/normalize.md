# `clarcs normalize` — scale and centre-of-mass alignment

Translate and uniformly scale a surface to match a reference's position and size.

---

## Usage

```bash
clarcs normalize INPUT [OUTPUT] --target TARGET [options]
```

**Arguments and options:**

| Argument / Flag | Description |
|---|---|
| `INPUT` | Surface to move / rescale |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-normalized<EXT>` |
| `--target TARGET` | Reference surface (required) |
| `-q / --quiet` | Suppress all output |

---

## Method

The transform combines a uniform scale and a translation:

$$T(\mathbf{x}) = s \cdot \mathbf{x} + (\bar{\mathbf{t}} - \bar{\mathbf{s}})$$

where the scale factor is:

$$s = \frac{\bar{d}_\text{target}}{\bar{d}_\text{source}}$$

and $\bar{d}$ is the mean Euclidean distance from each point to its cloud's
centroid (dispersion).  Applying the scale first then translating ensures that
both the size and the centre of mass of the result match the target.

---

## Examples

```bash
# Normalize skull.vtk to match reference.vtk
clarcs normalize skull.vtk aligned.vtk --target reference.vtk

# Default output name → skull-normalized.vtk
clarcs normalize skull.vtk --target reference.vtk
```

---

## Python API

```python
from pyclarcs.io import load_surface, save_surface
from pyclarcs.alignment import align_rescale

src_pts, src_poly = load_surface("skull.vtk")
tgt_pts, _        = load_surface("reference.vtk")

result = align_rescale(src_pts, tgt_pts)
save_surface("aligned.vtk", result, src_poly)
```
