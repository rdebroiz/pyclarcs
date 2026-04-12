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
| [`clarcs reorient`](docs/reorient.md) | Permute the coordinate axes of a surface | [docs/reorient.md](docs/reorient.md) |
| [`clarcs symplane`](docs/symplane.md) | Estimate the best bilateral symmetry plane | [docs/symplane.md](docs/symplane.md) |
| [`clarcs recenter`](docs/recenter.md) | Rigidly align a surface to the canonical symmetry plane | [docs/recenter.md](docs/recenter.md) |
| [`clarcs centerofmass`](docs/centerofmass.md) | Translate a surface to match a reference's centre of mass | [docs/centerofmass.md](docs/centerofmass.md) |
| [`clarcs normalize`](docs/normalize.md) | Translate and uniformly scale a surface to match a reference | [docs/normalize.md](docs/normalize.md) |
| [`clarcs nlregister`](docs/nlregister.md) | Non-rigidly register a surface onto a reference (EM-ICP) | [docs/nlregister.md](docs/nlregister.md) |

---

## Typical pipeline

The commands are designed to be chained.  A complete registration workflow
(symmetry-plane alignment → scale normalisation → non-rigid registration):

```bash
# 1. Align target's symmetry plane to x = 0
clarcs recenter   target.vtk  target-recentered.vtk  --save-plane

# 2. Match size and centre of mass to the reference
clarcs normalize  target-recentered.vtk  target-normalized.vtk  --target ref.vtk

# 3. Non-rigid EM-ICP onto the reference
clarcs nlregister target-normalized.vtk  ref.vtk  target-nlregistered.vtk \
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
from pyclarcs.nonrigid import nonrigid_icp, apply_deformation, estimate_registration_params

mov_pts, mov_poly, mov_n = load_surface_with_normals("target-normalized.vtk")
ref_pts, _,        ref_n = load_surface_with_normals("ref.vtk")
adj = adjacency_csr(mov_poly, len(mov_pts))

# Auto-estimate sigma, dist_cutoff and period_sigma from the surfaces.
# These three parameters are derived from the nearest-neighbour distance
# distribution between a 2 000-point subsample of the moving surface and
# the reference:
#   sigma        = 50th percentile of NN distances (median gap)
#   dist_cutoff  = max(99th percentile × 1.5,  sigma × 3)
#   period_sigma = max_iter // ceil(log2(sigma / sigma_min))
# Pass explicit values to override any of them.
params = estimate_registration_params(mov_pts, ref_pts)

def_field = nonrigid_icp(mov_pts, mov_n, ref_pts, ref_n, adj, **params)
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
│   ├── reorient.md
│   ├── symplane.md
│   ├── recenter.md
│   ├── centerofmass.md
│   ├── normalize.md
│   └── nlregister.md
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
