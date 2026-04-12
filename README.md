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
| [`clarcs register`](docs/register.md) | Non-rigidly register a surface onto a reference (EM-ICP) | [docs/register.md](docs/register.md) |

---

## Typical pipeline

The commands are designed to be chained.  A complete registration workflow
(symmetry-plane alignment → scale normalisation → non-rigid registration):

```bash
# 1. Align target's symmetry plane to x = 0
clarcs recenter  target.vtk  target-recentered.vtk  --save-plane

# 2. Match size and centre of mass to the reference
clarcs rescale   target-recentered.vtk  target-rescaled.vtk  --target ref.vtk

# 3. Non-rigid EM-ICP onto the reference
clarcs register  target-rescaled.vtk  ref.vtk  target-registered.vtk \
                 --deformation target-deformation.vtk
```

`data/run_pipeline.py` automates this sequence on the test surfaces bundled
in `data/` and writes all intermediate results to a directory of your choice:

```bash
python data/generate_samples.py          # create test surfaces (once)
python data/run_pipeline.py  results/    # run full pipeline, save to results/
```

---

## Python API

```python
from pyclarcs.io import load_surface, load_surface_with_normals, save_surface
from pyclarcs.symmetry import SymmetryPlane
from pyclarcs.principal_axes import best_principal_axis_plane
from pyclarcs.coarse import coarse_symmetry
from pyclarcs.fine import em_icp_sym, em_icp_sym_corres
from pyclarcs.alignment import align_to_symmetry_plane, align_rescale
from pyclarcs.mesh import adjacency_csr
from pyclarcs.nonrigid import nonrigid_icp, apply_deformation

# --- Symmetry plane ---
points, polygons = load_surface("surface.vtk")
plane = best_principal_axis_plane(points)
plane = coarse_symmetry(points, plane)
plane = em_icp_sym(points, plane)
plane = em_icp_sym_corres(points, plane)
plane.save("plane.pl")

# --- Non-rigid registration ---
mov_pts, mov_poly, mov_n = load_surface_with_normals("target-rescaled.vtk")
ref_pts, _,        ref_n = load_surface_with_normals("ref.vtk")
adj = adjacency_csr(mov_poly, len(mov_pts))

def_field = nonrigid_icp(mov_pts, mov_n, ref_pts, ref_n, adj)
warped    = apply_deformation(mov_pts, def_field)
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
│   ├── orient.md
│   └── register.md
├── data/                       ← test surfaces and pipeline scripts
│   ├── generate_samples.py     ← generate synthetic + MNI surfaces and test pairs
│   └── run_pipeline.py         ← run recenter → rescale → register end-to-end
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
│       ├── mesh.py             ← mesh adjacency utilities
│       ├── nonrigid.py         ← non-rigid EM-ICP registration
│       └── _numba_kernels.py   ← JIT-compiled kernels (Numba, internal)
└── tests/
    ├── test_symmetry.py
    └── test_alignment.py
```

---

## Licence

MIT
