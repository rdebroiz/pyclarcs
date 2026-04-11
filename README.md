# pyclarcs

Python port of the **ZZ_SYMC** tool from the
[CLARCS](https://www.irisa.fr/visages) C++ library.

Automatically finds the **best bilateral symmetry plane** of a 3-D surface
represented as a VTK point cloud or mesh.  Designed for endocranial surfaces
but applicable to any approximately bilateral 3-D object.

---

## Scientific background

The method is described in:

> Combès B., Hennessy R., Waddington J., Roberts N., Prima S.  
> **Automatic symmetry plane estimation of bilateral objects in point clouds.**  
> *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2008).*
> Anchorage, United States.

and is one of the three core algorithms of the CLARCS library, presented in:

> Abadie A., Combès B., Haegelen C., Prima S.  
> **CLARCS, a C++ Library for Automated Registration and Comparison of Surfaces:
> Medical Applications.**  
> *MICCAI Workshop on Mesh Processing in Medical Image Analysis
> (MeshMed'2011).* Toronto, Canada, pp. 117–126.

### Method overview

The algorithm finds the plane that best "superimposes the left and right parts"
of an approximately bilateral surface.  It is formulated as a MAP problem and
solved with an EM algorithm:

```
δ²(X¹, X²) = min_{A, T} [  Σ_{i,j} A_{i,j} ‖x_i − T(x_j)‖²
                           + 2σ² Σ_{i,j} A_{i,j} log A_{i,j}  ]
```

with `X¹ = X² = X` (same surface), `T` a reflection, and `A` a fuzzy
correspondence matrix.

The implementation runs three successive stages:

| Stage | Module | C++ equivalent |
|---|---|---|
| **Initialisation** — principal axes of inertia tensor | `_principal_axes.py` | `principal_axes::optimize()` |
| **Coarse** — ICP with trimmed estimator, multi-resolution | `_coarse.py` | `PointCloudSymmetryPlane::optimize()` |
| **Fine** — EM-ICP with simulated annealing (σ: 5 → 0.5) | `_fine.py::em_icp_sym` | `EM_ICPSym::optimize()` |
| **Refinement** — doubly-stochastic EM-ICP at σ = 0.25 | `_fine.py::em_icp_sym_corres` | `EM_ICPSymCorresSym::optimize()` |

---

## Installation

```bash
pip install pyclarcs
```

Or from source:

```bash
git clone <repo>
cd pyclarcs
pip install -e ".[dev]"
```

**Dependencies** (installed automatically):

| Package | Role |
|---|---|
| `numpy ≥ 1.21` | Linear algebra (replaces newmat, GSL) |
| `scipy ≥ 1.7` | KD-tree neighbour search (replaces kdtree / octree) |
| `pyvtk ≥ 0.5` | VTK legacy file I/O |

---

## Usage

### Command line (equivalent to ZZ_SYMC)

```bash
# Full pipeline (initialisation + coarse + fine + refinement)
pyclarcs -i surface.vtk -O results/plane -o results/symmetric.vtk

# Load a pre-existing initial plane instead of using principal axes
pyclarcs -i surface.vtk --init previous_plane.pl -O results/plane

# Coarse only (skip the EM stages)
pyclarcs -i surface.vtk --no-fine --no-sym -O results/plane_coarse

# Process a batch from a Docker-compatible script
for f in input/*.vtk; do
    pyclarcs -i "$f" -O "output/$(basename $f .vtk).plane" \
             -o "output/$(basename $f .vtk).symmetric.vtk"
done
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `-i / --input` | (required) | Input `.vtk` surface file |
| `-o / --output` | — | Output `.vtk` file: reflected surface |
| `-O / --Output` | — | Output prefix for plane files (`<prefix>.pl` and `<prefix>.vtk`) |
| `--init` | `auto` | `auto` (principal axes) or path to a `.pl` file |
| `--no-coarse` | off | Skip the coarse ICP stage |
| `--no-fine` | off | Skip the EM-ICP annealing stage |
| `--no-sym` | off | Skip the doubly-stochastic refinement |
| `-v / --verbose` | on | Print progress |
| `-q / --quiet` | off | Suppress all output |

### Python API

```python
import numpy as np
from pyclarcs import load_surface, save_surface, save_plane_vtk, SymmetryPlane
from pyclarcs._principal_axes import best_principal_axis_plane
from pyclarcs._coarse import coarse_symmetry
from pyclarcs._fine import em_icp_sym, em_icp_sym_corres

# --- Load surface ---
points, polygons = load_surface("surface.vtk")

# --- Stage 1: principal-axis initialisation ---
plane = best_principal_axis_plane(points)

# --- Stage 2: coarse ICP ---
plane = coarse_symmetry(points, plane, verbose=True)

# --- Stage 3: fine EM-ICP with annealing ---
plane = em_icp_sym(points, plane, sigma_init=5.0, sigma_final=0.5, verbose=True)

# --- Stage 4: doubly-stochastic refinement ---
plane = em_icp_sym_corres(points, plane, sigma=0.25, verbose=True)

print(plane)
# SymmetryPlane(n=[0.9998, 0.0123, -0.0045], d=83.2156)

# --- Save outputs ---
plane.save("plane.pl")                                    # text parameters
bounds = (points[:, 0].min(), points[:, 0].max(),
          points[:, 1].min(), points[:, 1].max(),
          points[:, 2].min(), points[:, 2].max())
save_plane_vtk("plane.vtk", plane, bounds)               # VTK patch
reflected = plane.apply(points)
save_surface("symmetric.vtk", reflected, polygons)        # reflected surface
```

### Plane file format (`.pl`)

```
n  0.9998  0.0123  -0.0045
p  83.2000  1.0200  -0.3700
```

- `n` — unit normal vector of the plane
- `p` — a point lying on the plane (= `n × d`)

The offset `d` is recovered as `d = n · p`.  This format is compatible with
the C++ ZZ_SYMC tool.

---

## Output files

| File | Content |
|---|---|
| `<prefix>.pl` | Symmetry plane parameters (text, compatible with C++ ZZ_SYMC) |
| `<prefix>.vtk` | Rectangular VTK patch visualising the symmetry plane |
| `<output>.vtk` | Input surface reflected through the estimated plane |

---

## Typical surfaces

Based on the experiments described in the CLARCS paper:

| Surface | # points | # faces | Expected runtime* |
|---|---|---|---|
| Endocranium (CT) | ~10 000 | ~20 000 | < 1 min |
| Skull outer surface (CT) | ~80 000–137 000 | ~160 000–280 000 | 2–5 min |
| Subcortical nucleus (MRI) | ~5 000 | ~10 000 | < 30 s |

\* indicative, on a modern CPU.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Repository structure

```
pyclarcs/
├── pyproject.toml              ← packaging configuration (PyPI-ready)
├── README.md
├── src/
│   └── pyclarcs/
│       ├── __init__.py
│       ├── __main__.py         ← python -m pyclarcs
│       ├── _cli.py             ← command-line interface
│       ├── _symmetry.py        ← SymmetryPlane class (reflection, fit, I/O)
│       ├── _principal_axes.py  ← inertia tensor, PA initialisation
│       ├── _io.py              ← VTK read / write via pyvtk
│       ├── _coarse.py          ← ICP + trimmed estimator, multi-resolution
│       └── _fine.py            ← EM-ICP (annealing + doubly-stochastic)
└── tests/
    └── test_symmetry.py
```

---

## Differences from the C++ ZZ_SYMC tool

| Aspect | C++ ZZ_SYMC | pyclarcs |
|---|---|---|
| VTK I/O | VTK C++ library | `pyvtk` (pure Python) |
| Linear algebra | newmat, GSL, LAPACK | `numpy.linalg` |
| KD-tree | custom C kdtree | `scipy.spatial.KDTree` |
| Initialisation | manual or auto (interactive) | auto or from file |
| Interactive mode | yes (VTK window) | no |
| Platform | Linux (Docker) | Linux, macOS, Windows |
| Distribution | Docker image | PyPI (`pip install pyclarcs`) |
| Column-sum bug in EM_ICPSymCorresSym | present (latent, harmless) | fixed |

---

## Licence

MIT
