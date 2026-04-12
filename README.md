# clarcs

Python toolkit for the automated analysis of 3-D surfaces, with a focus on
endocranial and bilateral anatomical structures.

`clarcs` is an extensible command-line tool.  The first available sub-command
is **`sym-plane`** — more tools will be added in future releases.

---

## Scientific background

The symmetry plane algorithm is described in:

> Combès B., Hennessy R., Waddington J., Roberts N., Prima S.  
> **Automatic symmetry plane estimation of bilateral objects in point clouds.**  
> *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2008).*
> Anchorage, United States.

and is used in the CLARCS research framework, presented in:

> Abadie A., Combès B., Haegelen C., Prima S.  
> **CLARCS, a C++ Library for Automated Registration and Comparison of Surfaces:
> Medical Applications.**  
> *MICCAI Workshop on Mesh Processing in Medical Image Analysis
> (MeshMed'2011).* Toronto, Canada, pp. 117–126.

### Method overview

The algorithm finds the plane that best "superimposes the left and right parts"
of an approximately bilateral surface.  It is formulated as a MAP problem and
solved with an EM algorithm:

$$
\delta^2(X^1, X^2) = \min_{A,\, T} \left[
  \sum_{i,j} A_{ij} \| x_i - T(x_j) \|^2
  + 2\sigma^2 \sum_{i,j} A_{ij} \log A_{ij}
\right]
$$

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

### `clarcs sym-plane` — symmetry plane estimation

Find the best bilateral symmetry plane of a 3-D surface.

```bash
clarcs sym-plane INPUT [OUTPUT] [--save-plane] [options]
```

**Arguments:**

| Argument | Description |
|---|---|
| `INPUT` | Input surface file (any supported format) |
| `OUTPUT` | Output file for the symmetry plane patch. Defaults to `<INPUT_STEM>-sym-plane<EXT>` |

**Options:**

| Flag | Description |
|---|---|
| `--save-plane` | Also save plane parameters to `<OUTPUT_STEM>.pl` |
| `--init auto\|FILE` | `auto` (principal axes, default) or path to a `.pl` file |
| `--no-coarse` | Skip the coarse ICP stage |
| `--no-fine` | Skip the EM-ICP annealing stage |
| `--no-sym` | Skip the doubly-stochastic refinement |
| `-v / --verbose` | Print progress (default: on) |
| `-q / --quiet` | Suppress all output |

**Examples:**

```bash
# Estimate symmetry plane → produces surface-sym-plane.vtk
clarcs sym-plane surface.vtk

# Custom output name
clarcs sym-plane surface.vtk results/plane.vtk

# Also save plane parameters (.pl file)
clarcs sym-plane surface.vtk --save-plane

# Load a pre-existing initial plane
clarcs sym-plane surface.vtk --init previous.pl --save-plane

# Coarse stage only (skip EM)
clarcs sym-plane surface.vtk --no-fine --no-sym

# Works with any supported format
clarcs sym-plane brain.ply --save-plane
clarcs sym-plane skull.stl results/skull-plane.vtp

# Batch processing
for f in input/*.vtk; do
    clarcs sym-plane "$f" "output/$(basename $f .vtk)-plane.vtk" --save-plane -q
done
```

**Output files:**

| File | Content |
|---|---|
| `<OUTPUT>` | Rectangular patch visualising the symmetry plane |
| `<OUTPUT_STEM>.pl` | Plane parameters — normal `n` and point `p` (with `--save-plane`) |

---

### `clarcs centerofmass` — centre-of-mass alignment

Translate a surface so its centre of mass coincides with a reference surface.

```bash
clarcs centerofmass INPUT [OUTPUT] --target TARGET
```

| Argument / Flag | Description |
|---|---|
| `INPUT` | Surface to move |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-centerofmass<EXT>` |
| `--target TARGET` | Reference surface (required) |

---

### `clarcs rescale` — scale and centre-of-mass alignment

Translate and uniformly scale a surface to match a reference's position and size.

```bash
clarcs rescale INPUT [OUTPUT] --target TARGET
```

| Argument / Flag | Description |
|---|---|
| `INPUT` | Surface to move / rescale |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-rescale<EXT>` |
| `--target TARGET` | Reference surface (required) |

The scale factor is $s = \bar{d}_\text{target} / \bar{d}_\text{source}$, where $\bar{d}$ is the mean Euclidean distance from each point to the centroid (dispersion).

---

### `clarcs recenter` — symmetry-plane recentering

Rigidly align a surface so its symmetry plane coincides with the canonical plane $n = [1, 0, 0],\ d = 0$ (the YZ plane at $x = 0$).

```bash
clarcs recenter INPUT [OUTPUT] [--plane PLANE.pl] [--save-plane]
```

| Argument / Flag | Description |
|---|---|
| `INPUT` | Surface to align |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-recentered<EXT>` |
| `--plane PLANE.pl` | Symmetry plane file (optional). If omitted, the plane is estimated automatically from the surface |
| `--save-plane` | Save the plane parameters to `<OUTPUT_STEM>.pl` |

```bash
# Automatic: estimate the plane and recenter in one step
clarcs recenter skull.vtk

# With a pre-computed plane (faster, skips estimation)
clarcs sym-plane skull.vtk --save-plane
clarcs recenter skull.vtk --plane skull-sym-plane.pl

# Estimate, recenter, and save the plane for later use
clarcs recenter skull.vtk --save-plane
```

---

### `clarcs orient` — axis permutation

Permute the coordinate axes of a surface.

```bash
clarcs orient INPUT [OUTPUT] --axes X Y Z
```

| Argument / Flag | Description |
|---|---|
| `INPUT` | Input surface |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-oriented<EXT>` |
| `--axes X Y Z` | Destination indices for the current x, y, z axes (default: `0 1 2` = identity) |

```bash
# Swap x and z axes
clarcs orient surface.vtk --axes 2 1 0

# Cyclic permutation x→1, y→2, z→0
clarcs orient surface.vtk --axes 1 2 0
```

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

# Run the pipeline
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

### Plane file format (`.pl`)

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
    └── test_symmetry.py
```

---

## Licence

MIT
