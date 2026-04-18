# pyclarcs

Python toolkit for the automated analysis of 3-D anatomical surfaces, with a
focus on endocranial and bilateral structures.

`clarcs` is a command-line tool built around two core algorithms:

- **Symmetry-plane estimation** — described in Combès et al., CVPR 2008
- **Non-rigid surface registration** — EM-ICP with symmetric correspondences,
  TGD geodesic prior, and RKHS Wu-kernel M-step (Combès & Prima, CVIU 2019)

---

## Installation

```bash
pip install pyclarcs
```

Or from source:

```bash
git clone https://github.com/rdebroiz/pyclarcs
cd pyclarcs
pip install -e ".[dev]"
```

**Dependencies** (installed automatically):

| Package | Role |
|---|---|
| `numpy ≥ 1.21` | Numerics |
| `scipy ≥ 1.7` | KD-tree, sparse linear algebra, graph shortest paths |
| `vtk ≥ 9.0` | Surface I/O and mesh processing |
| `numba ≥ 0.57` | JIT-compiled kernels (4× speedup on EM stages) |
| `click ≥ 8.0` | CLI framework |

---

## Supported formats

Format is inferred from the file extension.

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

| Command | Description |
|---|---|
| [`clarcs reorient`](docs/reorient.md) | Permute coordinate axes |
| [`clarcs symplane`](docs/symplane.md) | Estimate the bilateral symmetry plane |
| [`clarcs recenter`](docs/recenter.md) | Rigidly align a surface to its symmetry plane |
| [`clarcs centerofmass`](docs/centerofmass.md) | Translate to match a reference's centre of mass |
| [`clarcs normalize`](docs/normalize.md) | Scale and translate to match a reference |
| [`clarcs nlregister`](docs/nlregister.md) | Non-rigid EM-ICP registration onto a reference |

---

## Typical pipeline

```bash
# 1. Align symmetry plane to x = 0
clarcs recenter   target.vtk  target-recentered.vtk  --save-plane

# 2. Match size and centre of mass to the reference
clarcs normalize  target-recentered.vtk  target-normalized.vtk  --target ref.vtk

# 3. Non-rigid EM-ICP onto the reference
clarcs nlregister target-normalized.vtk  ref.vtk  target-registered.vtk \
                  --deformation target-deformation.vtk
```

`data/run_pipeline.py` automates this sequence on the bundled test surfaces:

```bash
python data/generate_samples.py   # create test surfaces (once)
python data/run_pipeline.py  results/
```

---

## Python API

### Symmetry plane

```python
from pyclarcs.io import load_surface, save_surface, save_plane_vtk
from pyclarcs.principal_axes import best_principal_axis_plane
from pyclarcs.coarse import coarse_symmetry
from pyclarcs.fine import em_icp_sym, em_icp_sym_corres

points, polygons = load_surface("surface.vtk")

plane = best_principal_axis_plane(points)       # initialise
plane = coarse_symmetry(points, plane)           # coarse ICP
plane = em_icp_sym(points, plane)               # EM annealing
plane = em_icp_sym_corres(points, plane)        # doubly-stochastic refinement

plane.save("plane.pl")
bounds = (points.min(0), points.max(0))         # used only for visualisation
save_plane_vtk("plane.vtk", plane, bounds)
```

### Non-rigid registration

```python
from pyclarcs.io import load_surface, load_surface_with_normals, save_surface
from pyclarcs.nonrigid import nonrigid_icp_multires, apply_deformation

mov_pts, mov_poly, mov_normals = load_surface_with_normals("target-normalized.vtk")
ref_pts, ref_poly              = load_surface("ref.vtk")
_, _,    ref_normals           = load_surface_with_normals("ref.vtk")

# All three improvements are on by default:
#   symmetric=True   — symmetric correspondences (Reg2)
#   use_tgd=True     — TGD geodesic shape prior  (Reg3)
#   use_rkhs=True    — RKHS Wu-kernel M-step
def_field = nonrigid_icp_multires(
    mov_pts, mov_normals,
    ref_pts, ref_normals,
    mov_poly, ref_poly,
)

warped = apply_deformation(mov_pts, def_field)
save_surface("registered.vtk", warped, mov_poly)
```

See [`docs/nlregister.md`](docs/nlregister.md) for all parameters and
a benchmark table.

---

## Registration benchmark

`endocranium_mni_pial` 10 k-vertex brain pial surface, synthetic deformation:

| Method | RMS after | Improvement | Time |
|---|---|---|---|
| BCPD (Nyström C++) | 3.82 mm | 37.9 % | 1.7 s |
| clarcs baseline | 3.89 mm | 35.7 % | 80 s |
| + Symmetric correspondences | 2.37 mm | 62.8 % | 74 s |
| + TGD prior | 2.36 mm | 62.9 % | 83 s |
| + RKHS M-step **(v0.2.0)** | **0.99 mm** | **84.0 %** | 97 s |

---

## Repository structure

```
pyclarcs/
├── pyproject.toml
├── README.md
├── docs/                       ← per-command documentation
│   ├── reorient.md
│   ├── symplane.md
│   ├── recenter.md
│   ├── centerofmass.md
│   ├── normalize.md
│   └── nlregister.md
├── data/
│   ├── generate_samples.py     ← generate synthetic + MNI test surfaces
│   ├── benchmark_compare.py    ← compare clarcs / CPD / BCPD
│   └── run_pipeline.py         ← run the full pipeline end-to-end
└── src/pyclarcs/
    ├── _cli.py                 ← CLI entry-point (clarcs command)
    ├── symmetry.py             ← SymmetryPlane class
    ├── principal_axes.py       ← inertia tensor initialisation
    ├── io.py                   ← multi-format surface I/O (VTK 9+)
    ├── coarse.py               ← coarse ICP with trimmed estimator
    ├── fine.py                 ← EM-ICP annealing + doubly-stochastic
    ├── alignment.py            ← rigid transforms (recenter, rescale, …)
    ├── mesh.py                 ← adjacency, TGD, Wu kernel graph
    ├── nonrigid.py             ← non-rigid EM-ICP (Reg2 + Reg3 + RKHS)
    └── _numba_kernels.py       ← JIT-compiled inner loops (internal)
```

---

## Scientific references

> Combès B., Hennessy R., Waddington J., Roberts N., Prima S.
> **Automatic symmetry plane estimation of bilateral objects in point clouds.**
> *CVPR 2008.* Anchorage, United States.

> Abadie A., Combès B., Haegelen C., Prima S.
> **CLARCS, a C++ Library for Automated Registration and Comparison of Surfaces.**
> *MeshMed @ MICCAI 2011.* Toronto, Canada, pp. 117–126.

> Combès B., Prima S.
> **New algorithms for the deformable registration of brain images.**
> *Medical Image Analysis, CVIU 2019.*

---

## Licence

MIT
