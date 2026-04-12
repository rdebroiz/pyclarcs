# `clarcs register` — non-rigid surface registration

Non-rigidly register a moving surface onto a reference using EM-ICP with a
graph-Laplacian regularisation on the deformation field.

---

## Usage

```bash
clarcs register INPUT REF [OUTPUT] [--deformation FIELD] [options]
```

**Arguments and options:**

| Argument / Flag | Description |
|---|---|
| `INPUT` | Moving surface (any supported format) |
| `REF` | Reference surface |
| `OUTPUT` | Warped output surface. Defaults to `<INPUT_STEM>-registered<EXT>` |
| `--deformation FIELD` | Save the per-vertex deformation field to this VTK file (VECTORS point data) |
| `--sigma F` | Initial bandwidth of the correspondence kernel (default: `3.0`) |
| `--beta F` | Regularisation weight — higher = smoother field (default: `100.0`) |
| `--dist-cutoff F` | Search radius for candidate correspondences (default: `15.0`) |
| `--max-iter N` | Number of outer EM iterations (default: `80`) |
| `--icm-iter N` | Jacobi ICM steps per outer iteration (default: `120`) |
| `--period-sigma N` | Halve sigma every N iterations (default: `40`) |
| `-q / --quiet` | Suppress all output |

---

## Method

The algorithm is a non-rigid EM-ICP variant where the unknown is a
per-vertex deformation field $d_i \in \mathbb{R}^3$ (one vector per vertex of
the moving surface, initialised to zero).

**E-step** — doubly-stochastic fuzzy correspondences

For each transformed point $T_i = x_i + d_i$, candidate reference points
$y_j$ within radius $r$ with a compatible normal ($n_i \cdot m_j \geq 0$)
receive a weight:

$$w_{ij} = \exp\!\left(-\|T_i - y_j\| / \sigma\right)$$

The correspondence matrix is doubly normalised (row sums *and* column sums),
which symmetrises the matching and suppresses outliers.  The resulting fuzzy
target $\bar{y}_i$ is the weighted barycentre of the matched reference points.

**M-step** — Jacobi ICM

Minimises the sum of a data term and a graph-Laplacian regulariser:

$$E = \sum_i W_i \|x_i + d_i - \bar{y}_i\|^2 + \beta \sum_{(i,j)\in\mathcal{E}} \|d_i - d_j\|^2$$

The closed-form Jacobi update (repeated `icm_iter` times) is:

$$d_i \leftarrow \frac{W_i(\bar{y}_i - x_i) + \beta \sum_{j \in \mathcal{N}_i} d_j}{\beta\,|\mathcal{N}_i| + W_i}$$

where $\mathcal{N}_i$ is the set of mesh-edge neighbours of $i$ and
$W_i = \sum_j \tilde{w}_{ij}$ is the total correspondence weight.

**Annealing** — $\sigma$ is halved every `period_sigma` iterations, sharpening
the correspondences as the field converges.

Vertex normals are computed automatically from the input mesh via VTK.
The mesh adjacency graph is derived from the polygon connectivity of the moving
surface.

---

## Examples

```bash
# Basic registration
clarcs register target.vtk reference.vtk registered.vtk

# Also save the deformation field for ParaView visualisation
clarcs register target.vtk reference.vtk registered.vtk \
               --deformation field.vtk

# Fewer iterations for a quick preview
clarcs register target.vtk reference.vtk --max-iter 20 --icm-iter 60

# Stronger regularisation (smoother deformation)
clarcs register target.vtk reference.vtk --beta 500
```

### Typical full pipeline

```bash
clarcs recenter  target.vtk  target-rc.vtk  --save-plane
clarcs rescale   target-rc.vtk  target-rcs.vtk  --target reference.vtk
clarcs register  target-rcs.vtk  reference.vtk  target-registered.vtk \
                 --deformation target-deformation.vtk
```

The helper script `data/run_pipeline.py` automates this sequence on the
bundled test surfaces:

```bash
python data/generate_samples.py          # build test pairs (once)
python data/run_pipeline.py  results/    # run end-to-end, write to results/
```

---

## Output files

| File | Content |
|---|---|
| `OUTPUT` | Warped moving surface (same connectivity as INPUT) |
| `FIELD` | Original surface + deformation vectors as VTK VECTORS point data |

The deformation field VTK file can be visualised in ParaView with the
*Warp By Vector* or *Glyph* filter applied to the VECTORS array.

---

## Python API

```python
from pyclarcs.io import load_surface_with_normals, save_surface, save_deformation_vtk
from pyclarcs.mesh import adjacency_csr
from pyclarcs.nonrigid import nonrigid_icp, apply_deformation

mov_pts, mov_poly, mov_normals = load_surface_with_normals("target.vtk")
ref_pts, _,        ref_normals = load_surface_with_normals("reference.vtk")

adj = adjacency_csr(mov_poly, len(mov_pts))

def_field = nonrigid_icp(
    mov_pts, mov_normals,
    ref_pts, ref_normals,
    adj,
    sigma=3.0, beta=100.0, dist_cutoff=15.0,
    max_iter=80, icm_iter=120,
)

warped = apply_deformation(mov_pts, def_field)
save_surface("registered.vtk", warped, mov_poly)
save_deformation_vtk("field.vtk", mov_pts, mov_poly, def_field)
```
