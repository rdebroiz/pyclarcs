# `clarcs mirror` — symmetry-plane reflection

Reflect a surface across its bilateral symmetry plane.

---

## Usage

```bash
clarcs mirror INPUT [OUTPUT] [--plane PLANE.pl] [--save-plane] [options]
```

**Arguments and options:**

| Argument / Flag | Description |
|---|---|
| `INPUT` | Surface to reflect |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-mirror<EXT>` |
| `--plane PLANE.pl` | Symmetry plane (`.pl`). Estimated automatically if omitted |
| `--save-plane` | Save the (computed or loaded) plane to `<OUTPUT_STEM>.pl` |
| `-q / --quiet` | Suppress all output |

---

## Method

Each vertex $x_i$ is reflected across the plane $(n, d)$:

$$x_i' = x_i - 2\,(x_i \cdot n - d)\,n$$

Because reflection reverses the orientation of every triangle (the right-hand
rule flips sign), each face's vertex list is reversed to restore outward-pointing
normals.

When `--plane` is omitted, the symmetry plane is estimated automatically
(principal axes → coarse ICP → EM-ICP annealing → doubly-stochastic refinement).

---

## Examples

```bash
# Reflect with automatic plane estimation
clarcs mirror brain.vtk

# Use a pre-computed plane (faster)
clarcs mirror brain.vtk brain-mirror.vtk --plane brain.pl

# Estimate, reflect, and save the plane for later
clarcs mirror brain.vtk --save-plane
```

---

## Python API

```python
from pyclarcs.io import load_surface, save_surface
from pyclarcs.symmetry import SymmetryPlane
from pyclarcs.alignment import reflect_surface

pts, polygons = load_surface("brain.vtk")
plane = SymmetryPlane.load("brain.pl")

mirrored = reflect_surface(pts, plane.n, plane.n * plane.d)
flipped_polygons = [f[::-1] for f in polygons]   # restore winding
save_surface("brain-mirror.vtk", mirrored, flipped_polygons)
```
