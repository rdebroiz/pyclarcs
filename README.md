# 🧠 pyclarcs

Python toolkit for the automated analysis of 3-D anatomical surfaces, with a
focus on endocranial and bilateral structures.

`clarcs` is a command-line tool built around two core algorithms:

- **Symmetry-plane estimation** — Combès et al., CVPR 2008
- **Non-rigid surface registration** — EM-ICP with symmetric correspondences,
  TGD geodesic prior, and RKHS Wu-kernel M-step (Combès & Prima, CVIU 2019)

Combined into a full population-analysis pipeline:
**symmetry → alignment → non-rigid registration → atlas → asymmetry mapping**.

---

## Table of contents

- [Installation](#installation)
- [Supported formats](#supported-formats)
- [Commands](#commands)
- [Typical pipelines](#typical-pipelines)
  - [Preprocessing and registration](#preprocessing-and-registration)
  - [Atlas construction](#atlas-construction)
  - [Asymmetry analysis on an atlas](#asymmetry-analysis-on-an-atlas)
- [Python API](#python-api)
- [Data — PaleoBRAIN dataset](#data--paleobrain-dataset)
- [Registration benchmark](#registration-benchmark)
- [Repository structure](#repository-structure)
- [Scientific references](#scientific-references)
- [Licence](#licence)

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

| Extension | Format | Read | Write |
|---|---|---|---|
| `.vtk` | VTK legacy PolyData (ASCII or binary) | ✓ | ✓ |
| `.vtp` | VTK XML PolyData | ✓ | ✓ |
| `.vtu` | VTK XML UnstructuredGrid (converted to PolyData) | ✓ | — |
| `.ply` | Stanford PLY | ✓ | ✓ |
| `.stl` | STereoLithography | ✓ | ✓ |
| `.obj` | Wavefront OBJ | ✓ | ✓ |

> **Note** — Asymmetry and deformation fields (VECTORS point data) require
> `.vtk` or `.vtp`.

---

## Commands

Commands are grouped by purpose.

### Preprocessing

| Command | Description |
|---|---|
| [`clarcs reorient`](docs/reorient.md) | Permute coordinate axes |
| [`clarcs recenter`](docs/recenter.md) | Align symmetry plane to *x* = 0 |
| [`clarcs centerofmass`](docs/centerofmass.md) | Translate to match a reference's centre of mass |
| [`clarcs normalize`](docs/normalize.md) | Uniformly scale + translate to match a reference |
| [`clarcs downsample`](docs/downsample.md) | Decimate to a target vertex count or ratio |

### Symmetry analysis

| Command | Description |
|---|---|
| [`clarcs symplane`](docs/symplane.md) | Estimate the bilateral symmetry plane |
| [`clarcs mirror`](docs/mirror.md) | Reflect a surface across its symmetry plane |
| [`clarcs asymmetry`](docs/asymmetry.md) | Compute the pointwise asymmetry field |

### Non-rigid registration

| Command | Description |
|---|---|
| [`clarcs nlregister`](docs/nlregister.md) | Non-rigid EM-ICP registration onto a reference |

### Atlas & population analysis

| Command | Description |
|---|---|
| [`clarcs atlas`](docs/atlas.md) | Build a mean shape atlas from a population |
| [`clarcs project-asym`](docs/project-asym.md) | Build atlas + project per-subject asymmetries onto it |

---

## Typical pipelines

### Preprocessing and registration

```bash
# 1. Align symmetry plane to x = 0
clarcs recenter target.vtk target-rc.vtk --save-plane

# 2. Match size and centre of mass to the reference
clarcs normalize target-rc.vtk target-rcs.vtk --target reference.vtk

# 3. Non-rigid EM-ICP registration
clarcs nlregister target-rcs.vtk reference.vtk target-registered.vtk \
                  --deformation target-deformation.vtk
```

### Atlas construction

```bash
# Build a mean shape atlas from a directory of preprocessed surfaces
clarcs atlas subjects/ atlas.vtk --save-registered

# Lower resolution first (useful for large surfaces)
clarcs downsample subjects/B01.ply subjects/B01-5k.ply --target-n 5000
```

### Asymmetry analysis on an atlas

```bash
# All-in-one: build atlas, compute subject asymmetries, project onto atlas
clarcs project-asym subjects/ mean-asym.vtp --save-atlas atlas.vtp \
       --save-stats --save-individual

# Incremental (reuse pre-computed atlas and asymmetry fields)
clarcs asymmetry B01.ply B01-asym.vtk
clarcs project-asym subjects/ mean-asym.vtp \
       --atlas atlas.vtp \
       --registered-dir registered/ \
       --asymmetry-dir  asymmetry/
```

`data/run_pipeline.py` automates the preprocessing + registration sequence
on the bundled test surfaces:

```bash
python data/generate_samples.py   # create test surfaces (once)
python data/run_pipeline.py results/
```

---

## Python API

### Symmetry plane

```python
from pyclarcs.io import load_surface, save_plane_vtk
from pyclarcs.principal_axes import best_principal_axis_plane
from pyclarcs.coarse import coarse_symmetry
from pyclarcs.fine import em_icp_sym, em_icp_sym_corres

points, polygons = load_surface("surface.vtk")

plane = best_principal_axis_plane(points)
plane = coarse_symmetry(points, plane)
plane = em_icp_sym(points, plane)
plane = em_icp_sym_corres(points, plane)

plane.save("plane.pl")
```

### Non-rigid registration

```python
from pyclarcs.io import load_surface, load_surface_with_normals, save_surface
from pyclarcs.nonrigid import register, apply_deformation

mov_pts, mov_poly, mov_normals = load_surface_with_normals("target.vtk")
ref_pts, ref_poly, ref_normals = load_surface_with_normals("reference.vtk")

# Symmetric correspondences + TGD prior + RKHS M-step are all on by default
def_field = register(
    mov_pts, mov_normals,
    ref_pts, ref_normals,
    mov_poly, ref_poly,
)

warped = apply_deformation(mov_pts, def_field)
save_surface("registered.vtk", warped, mov_poly)
```

### Atlas construction

```python
from pyclarcs.io import load_surface_with_normals, save_surface
from pyclarcs.atlas import build_atlas

subjects = [load_surface_with_normals(p) for p in sorted(paths)]

atlas_pts, atlas_poly, registered = build_atlas(subjects, atlas_iter=3)
save_surface("atlas.vtk", atlas_pts, atlas_poly)
```

### Asymmetry projection

```python
from pyclarcs.io import load_vector_field
from pyclarcs.atlas import project_asymmetry_to_atlas
import numpy as np

# registered[i]: atlas vertices in subject i's space (from build_atlas)
# asym_fields[i]: asymmetry field at subject i's vertices (from clarcs asymmetry)
projected = project_asymmetry_to_atlas(registered, asym_fields, subject_pts)

mean_asym = np.stack(projected).mean(axis=0)   # (N, 3) mean field on atlas
```

---

## Data — PaleoBRAIN dataset

75 brain surfaces (B01–B75) and 75 endocasts (E01–E75) from the PaleoBRAIN
project (Balzeau A., 2025, doi:[10.48579/PRO/KZMMLM](https://data.indores.fr/dataset.xhtml?persistentId=doi:10.48579/PRO/KZMMLM)).

```bash
# Download first 10 subjects (brains + endocasts), resample to 5 000 vertices
python data/download_paleobrain.py --n 10 --target-n 5000

# Brains only, all 75
python data/download_paleobrain.py --type brain
```

---

## Registration benchmark

MNI pial endocranium (10 k vertices), synthetic deformation of 8 Gaussian
bumps × 5 mm amplitude. RMS before: **2.28 mm**.

| Configuration | RMS after | Improvement |
|---|---|---|
| Baseline (Laplacian) | 0.333 mm | 85.4 % |
| + Symmetric correspondences + TGD | 0.267 mm | 88.3 % |
| **+ RKHS M-step (default)** | **0.190 mm** | **91.6 %** |

See [`docs/nlregister.md`](docs/nlregister.md) for full details and parameters.

---

## Repository structure

```
pyclarcs/
├── pyproject.toml
├── README.md
├── docs/                           ← per-command documentation
│   ├── reorient.md
│   ├── symplane.md
│   ├── recenter.md
│   ├── centerofmass.md
│   ├── normalize.md
│   ├── mirror.md
│   ├── downsample.md
│   ├── nlregister.md
│   ├── atlas.md
│   ├── asymmetry.md
│   └── project-asym.md
├── data/
│   ├── generate_samples.py         ← generate synthetic + MNI test surfaces
│   ├── download_paleobrain.py      ← download PaleoBRAIN dataset
│   ├── benchmark_compare.py        ← compare clarcs / CPD / BCPD
│   └── run_pipeline.py             ← run the full pipeline end-to-end
└── src/pyclarcs/
    ├── _cli.py                     ← CLI entry-point (clarcs command)
    ├── symmetry.py                 ← SymmetryPlane class
    ├── principal_axes.py           ← inertia tensor initialisation
    ├── io.py                       ← multi-format surface I/O (VTK 9+)
    ├── coarse.py                   ← coarse ICP with trimmed estimator
    ├── fine.py                     ← EM-ICP annealing + doubly-stochastic
    ├── alignment.py                ← rigid transforms (recenter, rescale, …)
    ├── mesh.py                     ← adjacency, TGD, Wu kernel graph
    ├── nonrigid.py                 ← register(), _em_icp(), _build_hierarchy()
    ├── atlas.py                    ← build_atlas(), project_asymmetry_to_atlas()
    └── _numba_kernels.py           ← JIT-compiled inner loops (internal)
```

---

## Scientific references

> Combès B., Hennessy R., Waddington J., Roberts N., Prima S.
> **Automatic symmetry plane estimation of bilateral objects in point clouds.**
> *IEEE CVPR 2008.* Anchorage, United States.

> Abadie A., Combès B., Haegelen C., Prima S.
> **CLARCS, a C++ Library for Automated Registration and Comparison of Surfaces: Medical Applications.**
> *MeshMed @ MICCAI 2011.* Toronto, Canada, pp. 117–126.

> Combès B., Prima S.
> **New algorithms for the deformable registration of brain images.**
> *Medical Image Analysis / CVIU 2019.*

> Balzeau A. (2025).
> **Database of 75 endocasts and 75 brains obtained on the same sample of volunteers.**
> doi:[10.48579/PRO/KZMMLM](https://data.indores.fr/dataset.xhtml?persistentId=doi:10.48579/PRO/KZMMLM)

---

## Licence

MIT
