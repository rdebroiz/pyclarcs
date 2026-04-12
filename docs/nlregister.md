# `clarcs nlregister` — non-rigid surface registration

Non-rigidly register a moving surface onto a reference using EM-ICP with a
graph-Laplacian regularisation on the deformation field.

---

## Usage

```bash
clarcs nlregister INPUT REF [OUTPUT] [--deformation FIELD] [options]
```

**Arguments and options:**

| Argument / Flag | Description |
|---|---|
| `INPUT` | Moving surface (any supported format) |
| `REF` | Reference surface |
| `OUTPUT` | Warped output surface. Defaults to `<INPUT_STEM>-nlregistered<EXT>` |
| `--deformation FIELD` | Save the per-vertex deformation field to this VTK file (VECTORS point data) |
| `--sigma F` | Initial bandwidth of the correspondence kernel — **auto-estimated if omitted** |
| `--beta F` | Regularisation weight — higher = smoother field (default: `100.0`) |
| `--dist-cutoff F` | Search radius for candidate correspondences — **auto-estimated if omitted** |
| `--max-iter N` | Number of outer EM iterations (default: `80`) |
| `--icm-iter N` | Jacobi ICM steps per outer iteration (default: `120`) |
| `--period-sigma N` | Halve sigma every N iterations — **auto-estimated if omitted** |
| `--sigma-min F` | Annealing floor for sigma (default: `0.1`) |
| `--e-chunk N` | Vertices per KDTree batch in the E-step (default: `2000`). Reduce to lower peak RAM. |
| `-q / --quiet` | Suppress all output |

### Auto-estimation of sigma, dist_cutoff and period_sigma

When `--sigma`, `--dist-cutoff` or `--period-sigma` are omitted, the command
subsamples 2 000 vertices from the moving surface, queries their nearest
neighbour on the reference, and derives the three parameters from the resulting
distance distribution:

| Parameter | Formula | Rationale |
|---|---|---|
| `sigma` | $\tilde{d}_{50}$ — median NN distance | At $\sigma$ = median gap, roughly half the point pairs have weight $\geq e^{-1} \approx 0.37$: broad but informative correspondences to start with. |
| `dist_cutoff` | $\max\!\bigl(\tilde{d}_{99} \times 1.5,\; \sigma \times 3\bigr)$ | The 99th-percentile covers near-outlier points; ×1.5 adds a safety margin. The $3\sigma$ floor ensures the search radius is always meaningful relative to the kernel width. |
| `period_sigma` | $\bigl\lfloor \text{max\_iter} / \lceil \log_2(\sigma / \sigma_{\min}) \rceil \bigr\rfloor$ | Computes the number of halvings needed to bring $\sigma$ from its initial value down to $\sigma_{\min}$, then spreads them evenly across the outer iterations. |

where $\tilde{d}_p$ denotes the $p$-th percentile of the nearest-neighbour
distances measured from the subsample to the reference.

The auto-estimated values are printed at the start of the run (unless `--quiet`)
so they can be reused or overridden in subsequent calls.

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
clarcs nlregister target.vtk reference.vtk registered.vtk

# Also save the deformation field for ParaView visualisation
clarcs nlregister target.vtk reference.vtk registered.vtk \
                  --deformation field.vtk

# Fewer iterations for a quick preview
clarcs nlregister target.vtk reference.vtk --max-iter 20 --icm-iter 60

# Stronger regularisation (smoother deformation)
clarcs nlregister target.vtk reference.vtk --beta 500
```

### Typical full pipeline

```bash
clarcs recenter   target.vtk  target-rc.vtk  --save-plane
clarcs normalize  target-rc.vtk  target-rcs.vtk  --target reference.vtk
clarcs nlregister target-rcs.vtk  reference.vtk  target-registered.vtk \
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
from pyclarcs.nonrigid import nonrigid_icp, apply_deformation, estimate_registration_params

mov_pts, mov_poly, mov_normals = load_surface_with_normals("target.vtk")
ref_pts, _,        ref_normals = load_surface_with_normals("reference.vtk")

# Auto-estimate parameters from the surfaces (optional — pass explicit values
# to override any of the three).
params = estimate_registration_params(mov_pts, ref_pts)
print(params)
# → {"sigma": 3.6, "dist_cutoff": 20.4, "period_sigma": 13}

adj = adjacency_csr(mov_poly, len(mov_pts))

def_field = nonrigid_icp(
    mov_pts, mov_normals,
    ref_pts, ref_normals,
    adj,
    **params,          # unpack auto-estimated params; add beta/max_iter/... as needed
)

warped = apply_deformation(mov_pts, def_field)
save_surface("registered.vtk", warped, mov_poly)
save_deformation_vtk("field.vtk", mov_pts, mov_poly, def_field)
```
