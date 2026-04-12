# clarcs

Python toolkit for the automated analysis of 3-D surfaces, with a focus on
endocranial and bilateral anatomical structures.

`clarcs` is an extensible command-line tool built around a symmetry plane
estimation algorithm described in:

> Combès B., Hennessy R., Waddington J., Roberts N., Prima S.  
> **Automatic symmetry plane estimation of bilateral objects in point clouds.**  
> *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2008).*
> Anchorage, United States.

and used in the CLARCS research framework, presented in:

> Abadie A., Combès B., Haegelen C., Prima S.  
> **CLARCS, a C++ Library for Automated Registration and Comparison of Surfaces:
> Medical Applications.**  
> *MICCAI Workshop on Mesh Processing in Medical Image Analysis
> (MeshMed'2011).* Toronto, Canada, pp. 117–126.

---

## Installation

```bash
pip install clarcs
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
| `numpy ≥ 1.21` | Linear algebra |
| `scipy ≥ 1.7` | KD-tree neighbour search |
| `vtk ≥ 9.0` | Surface I/O |
| `numba ≥ 0.57` | JIT-compiled kernels (4× speedup on EM stage) |

---

## Supported formats

Both reading and writing support the following formats (format inferred from
file extension):

| Extension | Format |
|---|---|
| `.vtk` | VTK legacy PolyData (ASCII or binary) |
| `.vtp` | VTK XML PolyData |
| `.vtu` | VTK XML UnstructuredGrid (read → converted to PolyData) |
| `.ply` | Stanford PLY |
| `.stl` | STereoLithography |
| `.obj` | Wavefront OBJ |

---

## Commands

| Command | Description | Documentation |
|---|---|---|
| [`clarcs sym-plane`](docs/sym-plane.md) | Estimate the best bilateral symmetry plane | [docs/sym-plane.md](docs/sym-plane.md) |
| [`clarcs centerofmass`](docs/centerofmass.md) | Translate a surface to match a reference's centre of mass | [docs/centerofmass.md](docs/centerofmass.md) |
| [`clarcs rescale`](docs/rescale.md) | Translate and uniformly scale a surface to match a reference | [docs/rescale.md](docs/rescale.md) |
| [`clarcs recenter`](docs/recenter.md) | Rigidly align a surface to the canonical symmetry plane | [docs/recenter.md](docs/recenter.md) |
| [`clarcs orient`](docs/orient.md) | Permute the coordinate axes of a surface | [docs/orient.md](docs/orient.md) |

---

## Python API

```python
from pyclarcs.io import load_surface, save_surface, save_plane_vtk
from pyclarcs.symmetry import SymmetryPlane
from pyclarcs.principal_axes import best_principal_axis_plane
from pyclarcs.coarse import coarse_symmetry
from pyclarcs.fine import em_icp_sym, em_icp_sym_corres
from pyclarcs.alignment import (
    align_center_of_mass, align_rescale,
    align_to_symmetry_plane, reorient_axes,
)

# Load surface (any supported format)
points, polygons = load_surface("surface.vtk")   # or .ply, .stl, .obj, .vtp …

# Run the symmetry plane pipeline
plane = best_principal_axis_plane(points)
plane = coarse_symmetry(points, plane, verbose=True)
plane = em_icp_sym(points, plane, verbose=True)
plane = em_icp_sym_corres(points, plane, verbose=True)

print(plane)
# SymmetryPlane(n=[0.9998, 0.0123, -0.0045], d=83.2156)

# Save outputs
plane.save("plane.pl")
bounds = (points[:, 0].min(), points[:, 0].max(),
          points[:, 1].min(), points[:, 1].max(),
          points[:, 2].min(), points[:, 2].max())
save_plane_vtk("plane.vtk", plane, bounds)
```

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
├── pyproject.toml              ← packaging (PyPI-ready, installs as clarcs)
├── README.md
├── docs/                       ← per-command documentation
│   ├── sym-plane.md
│   ├── centerofmass.md
│   ├── rescale.md
│   ├── recenter.md
│   └── orient.md
├── src/
│   └── pyclarcs/
│       ├── __init__.py
│       ├── __main__.py         ← python -m pyclarcs
│       ├── _cli.py             ← clarcs command + sub-commands (internal)
│       ├── symmetry.py         ← SymmetryPlane class
│       ├── principal_axes.py   ← inertia tensor, PA initialisation
│       ├── io.py               ← multi-format surface I/O via VTK 9+
│       ├── coarse.py           ← ICP + trimmed estimator, multi-resolution
│       ├── fine.py             ← EM-ICP (annealing + doubly-stochastic)
│       ├── alignment.py        ← centerofmass, rescale, recenter, orient
│       └── _numba_kernels.py   ← JIT-compiled kernels (Numba, internal)
└── tests/
    ├── test_symmetry.py
    └── test_alignment.py
```

---

## Licence

MIT
