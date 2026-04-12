# `clarcs recenter` — symmetry-plane recentering

Rigidly align a surface so its symmetry plane coincides with the canonical
plane $n = [1, 0, 0],\ d = 0$ (the YZ plane at $x = 0$).

---

## Usage

```bash
clarcs recenter INPUT [OUTPUT] [--plane PLANE.pl] [--save-plane] [options]
```

**Arguments and options:**

| Argument / Flag | Description |
|---|---|
| `INPUT` | Surface to align |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-recentered<EXT>` |
| `--plane PLANE.pl` | Symmetry plane file (optional). If omitted, the plane is estimated automatically from the surface |
| `--save-plane` | Save the plane parameters (in the canonical frame) to `<OUTPUT_STEM>.pl` |
| `-q / --quiet` | Suppress all output |

---

## Method

Given the symmetry plane $(n, d)$, the transform is a rigid motion (rotation +
translation) that maps $n \to [1, 0, 0]$ and the plane point $n \cdot d \to 0$.
Pairwise distances between all points are preserved exactly.

When `--plane` is omitted, the plane is estimated automatically by running
the full symmetry-plane pipeline (`clarcs symplane` equivalent):

1. Initialisation via principal axes of inertia
2. Coarse ICP with trimmed estimator
3. EM-ICP with simulated annealing
4. Doubly-stochastic EM-ICP refinement

---

## Examples

```bash
# Automatic: estimate the plane and recenter in one step
clarcs recenter skull.vtk

# With a pre-computed plane (faster, skips estimation)
clarcs symplane skull.vtk --save-plane
clarcs recenter skull.vtk --plane skull-symplane.pl

# Estimate, recenter, and save the (canonical) plane for later use
clarcs recenter skull.vtk --save-plane
```

---

## Python API

```python
from pyclarcs.io import load_surface, save_surface
from pyclarcs.symmetry import SymmetryPlane
from pyclarcs.alignment import align_to_symmetry_plane

src_pts, src_poly = load_surface("skull.vtk")
plane = SymmetryPlane.load("skull-sym-plane.pl")

result = align_to_symmetry_plane(src_pts, plane)
save_surface("skull-recentered.vtk", result, src_poly)
```
