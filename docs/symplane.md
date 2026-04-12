# `clarcs symplane` — symmetry plane estimation

Find the best bilateral symmetry plane of a 3-D surface.

---

## Scientific background

The algorithm finds the plane that best "superimposes the left and right parts"
of an approximately bilateral surface.  It is formulated as a MAP problem and
solved with an EM algorithm:

$$\delta^2(X^1, X^2) = \min_{A,\, T} \left[ \sum_{i,j} A_{ij} \| x_i - T(x_j) \|^2 + 2\sigma^2 \sum_{i,j} A_{ij} \log A_{ij} \right]$$

with $X^1 = X^2 = X$ (same surface), $T$ a reflection, and $A$ a fuzzy
correspondence matrix.

The implementation runs four successive stages:

| Stage | Module |
|---|---|
| **Initialisation** — principal axes of inertia tensor | `principal_axes.py` |
| **Coarse** — ICP with trimmed estimator, multi-resolution | `coarse.py` |
| **Fine** — EM-ICP with simulated annealing ($\sigma: 5 \to 0.5$) | `fine.py` |
| **Refinement** — doubly-stochastic EM-ICP at $\sigma = 0.25$ | `fine.py` |

---

## Usage

```bash
clarcs symplane INPUT [OUTPUT] [--save-plane] [options]
```

**Arguments:**

| Argument | Description |
|---|---|
| `INPUT` | Input surface file (any supported format) |
| `OUTPUT` | Output file for the symmetry plane patch. Defaults to `<INPUT_STEM>-symplane<EXT>` |

**Options:**

| Flag | Description |
|---|---|
| `--save-plane` | Also save plane parameters to `<OUTPUT_STEM>.pl` |
| `--init auto\|FILE` | `auto` (principal axes, default) or path to a `.pl` file |
| `--no-coarse` | Skip the coarse ICP stage |
| `--no-fine` | Skip the EM-ICP annealing stage |
| `--no-sym` | Skip the doubly-stochastic refinement |
| `-q / --quiet` | Suppress all output |

---

## Examples

```bash
# Estimate symmetry plane → produces surface-symplane.vtk
clarcs symplane surface.vtk

# Custom output name
clarcs symplane surface.vtk results/plane.vtk

# Also save plane parameters (.pl file)
clarcs symplane surface.vtk --save-plane

# Load a pre-existing initial plane
clarcs symplane surface.vtk --init previous.pl --save-plane

# Coarse stage only (skip EM)
clarcs symplane surface.vtk --no-fine --no-sym

# Works with any supported format
clarcs symplane brain.ply --save-plane
clarcs symplane skull.stl results/skull-plane.vtp

# Batch processing
for f in input/*.vtk; do
    clarcs symplane "$f" "output/$(basename $f .vtk)-plane.vtk" --save-plane -q
done
```

---

## Output files

| File | Content |
|---|---|
| `<OUTPUT>` | Rectangular patch visualising the symmetry plane |
| `<OUTPUT_STEM>.pl` | Plane parameters — normal `n` and point `p` (with `--save-plane`) |

---

## Plane file format (`.pl`)

```
n  0.9998  0.0123  -0.0045
p  83.2000  1.0200  -0.3700
```

- `n` — unit normal vector of the plane
- `p` — a point lying on the plane ($p = n \cdot d$)
- offset $d$ is recovered as $d = n \cdot p$

---

## Typical surfaces

| Surface | # points | # faces | Expected runtime* |
|---|---|---|---|
| Endocranium (CT) | ~10 000 | ~20 000 | ~35 s |
| Skull outer surface (CT) | ~80 000–137 000 | ~160 000–280 000 | 2–5 min |
| Subcortical nucleus (MRI) | ~5 000 | ~10 000 | < 15 s |

\* on a modern multi-core CPU with Numba JIT cache warm.

---

## Python API

```python
from pyclarcs.io import load_surface, save_surface, save_plane_vtk
from pyclarcs.symmetry import SymmetryPlane
from pyclarcs.principal_axes import best_principal_axis_plane
from pyclarcs.coarse import coarse_symmetry
from pyclarcs.fine import em_icp_sym, em_icp_sym_corres

points, polygons = load_surface("surface.vtk")

plane = best_principal_axis_plane(points)
plane = coarse_symmetry(points, plane, verbose=True)
plane = em_icp_sym(points, plane, verbose=True)
plane = em_icp_sym_corres(points, plane, verbose=True)

print(plane)
# SymmetryPlane(n=[0.9998, 0.0123, -0.0045], d=83.2156)

plane.save("plane.pl")
bounds = (points[:, 0].min(), points[:, 0].max(),
          points[:, 1].min(), points[:, 1].max(),
          points[:, 2].min(), points[:, 2].max())
save_plane_vtk("plane.vtk", plane, bounds)
```
