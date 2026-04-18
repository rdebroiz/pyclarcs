# `clarcs nlregister` — non-rigid surface registration

Non-rigidly register a moving surface onto a reference using EM-ICP with three
algorithmic improvements over the baseline (Combès & Prima 2019, CVIU):

1. **Symmetric correspondences** (Reg2) — prevents many-to-one mappings
2. **TGD shape prior** (Reg3) — penalises cross-sulcus matches via geodesic depth
3. **RKHS M-step** — Wu C4 kernel replaces graph-Laplacian for topology-independent regularisation

---

## Usage

```bash
clarcs nlregister INPUT REF [OUTPUT] [--deformation FIELD] [options]
```

**Arguments and options:**

| Argument / Flag | Default | Description |
|---|---|---|
| `INPUT` | — | Moving surface (any supported format) |
| `REF` | — | Reference surface |
| `OUTPUT` | auto | Warped output. Defaults to `<INPUT_STEM>-nlregistered<EXT>` |
| `--deformation FIELD` | — | Save per-vertex deformation field to this VTK file |
| `--sigma F` | auto | Initial Gaussian kernel bandwidth [mm] |
| `--beta F` | auto | Laplacian regularisation weight (used only with `--no-rkhs`) |
| `--dist-cutoff F` | auto | Correspondence search radius [mm] |
| `--max-iter N` | `80` | Outer EM iterations |
| `--icm-iter N` | `50` | Max CG iterations per outer iteration |
| `--period-sigma N` | auto | Halve sigma every N iterations |
| `--sigma-min F` | auto | Annealing floor for sigma |
| `--outlier-weight F` | `0.1` | Prior probability of outlier vertex (CPD-style) |
| `--normal-min-dot F` | `0.0` | Min source/reference normal dot-product (0=same hemisphere) |
| `--n-levels N` | auto | Resolution levels (1=single-res) |
| `--coarsest-n N` | `2000` | Target vertex count at coarsest level |
| `--beta-coarse-factor F` | `1.0` | Per-level beta multiplier toward coarser levels |
| `--e-chunk N` | `2000` | Vertices per KDTree batch (lower = less RAM) |
| `--no-symmetric` | off | Disable symmetric correspondences (A+B) |
| `--no-tgd` | off | Disable TGD geodesic shape prior |
| `--no-rkhs` | off | Use Laplacian M-step instead of RKHS Wu kernel |
| `--rkhs-lambda F` | `0.01` | RKHS regularisation weight |
| `-q / --quiet` | — | Suppress all output |

All registration parameters (`sigma`, `dist-cutoff`, `period-sigma`, `beta`,
`sigma-min`, `n-levels`) are **auto-estimated from the surfaces** when omitted.

---

## Method

The algorithm is a non-rigid EM-ICP variant.  The unknown is a per-vertex
deformation field $d_i \in \mathbb{R}^3$ initialised to zero.

### E-step — Gaussian kernel + symmetric correspondences + TGD prior

For each transformed point $T_i = x_i + d_i$, candidate reference points
$y_j$ within radius $r$ with compatible normal ($n_i \cdot m_j \geq \delta$)
receive a Gaussian weight:

$$w_{ij} = \exp\!\left(-\frac{\|T_i - y_j\|^2}{2\sigma^2}\right) \cdot \pi_{ij}$$

where $\pi_{ij} = \exp\!\left(-\frac{|\mathrm{TGD}_i - \mathrm{TGD}_j|^2}{2\sigma_{\mathrm{tgd}}^2}\right)$
is the **TGD shape prior** (disabled with `--no-tgd`).  TGD (Total Geodesic
Distance) is a scalar per vertex that encodes sulcal depth: high on gyral
crests, low in deep sulci.  The prior down-weights matches between vertices at
different depths, reducing cross-sulcus false correspondences.

**Symmetric correspondences** (disabled with `--no-symmetric`):

Instead of row-normalising alone, the combined weight is

$$\tilde{w}_{ij} = w_{ij} / s^R_i + w_{ij} / s^C_j$$

where $s^R_i = \sum_j w_{ij}$ and $s^C_j = \sum_i w_{ij}$ are the row and
column sums.  This prevents multiple moving vertices from collapsing onto the
same reference vertex.

**Outlier term** (CPD-style, $\sigma$-invariant):

$$W_i = \frac{\sum_j \tilde{w}_{ij}}{\sum_j \tilde{w}_{ij} + c},
\quad c = w_{\mathrm{out}} \cdot M/N$$

The fuzzy target: $\bar{y}_i = \sum_j \tilde{w}_{ij}\,y_j / \sum_j \tilde{w}_{ij}$.

### M-step — RKHS Wu kernel (or Laplacian fallback)

**RKHS mode** (default, disable with `--no-rkhs`):

The deformation is expressed as $d = K\alpha$ where $K$ is a sparse kernel
matrix with the Wu C4 compactly-supported kernel:

$$K_{ij} = \left(1 - \frac{r_{ij}}{b}\right)^4\!\left(\frac{4r_{ij}}{b} + 1\right)
\cdot \mathbf{1}_{r_{ij} < b}, \quad r_{ij} = \|x_i - x_j\|$$

$K$ is computed once on the original moving-mesh positions ($b = 8\sigma_\mathrm{min}$).
The M-step solves a symmetric sparse system by CG:

$$\bigl(\mathrm{diag}(W) \cdot K + \lambda I\bigr)\,\alpha = \mathrm{diag}(W)\,(\bar{y} - x)$$

then applies $d = K\alpha$.  The RKHS norm $\alpha^\top K \alpha$ provides
smooth, topology-independent regularisation that scales with point density.

**Laplacian mode** (`--no-rkhs`):

$$\bigl(\mathrm{diag}(W + \beta|\mathcal{N}_i|) - \beta A\bigr)\,d_{:,k} = W \cdot (\bar{y} - x)_{:,k}$$

### Annealing

$\sigma$ is halved every `period_sigma` iterations down to `sigma_min`.

### Multi-resolution

A coarse-to-fine hierarchy of decimated moving meshes is built (unless
`n-levels=1`).  Registration runs from the coarsest level to the finest,
warm-starting each finer level from the interpolated coarser deformation
(inverse-distance-weighted).  TGD and the RKHS kernel are computed once on
the finest mesh and interpolated to coarser levels.

---

## Benchmark (endocranium_mni_pial 10k, synthetic deformation)

| Configuration | RMS after | Improvement | Time |
|---|---|---|---|
| BCPD (Nyström C++) | 3.82 mm | 37.9 % | 1.7 s |
| clarcs baseline | 3.89 mm | 35.7 % | 80 s |
| + Symmetric (Reg2) | 2.37 mm | 62.8 % | 74 s |
| + TGD prior (Reg3) | 2.36 mm | 62.9 % | 83 s |
| + RKHS M-step | **0.99 mm** | **84.0 %** | 97 s |

---

## Examples

```bash
# Basic registration (all improvements enabled by default)
clarcs nlregister target.vtk reference.vtk registered.vtk

# Also save the deformation field for ParaView visualisation
clarcs nlregister target.vtk reference.vtk registered.vtk \
                  --deformation field.vtk

# Faster preview: fewer iterations, disable TGD computation
clarcs nlregister target.vtk reference.vtk --max-iter 20 --no-tgd

# Disable all new features (baseline Laplacian EM-ICP)
clarcs nlregister target.vtk reference.vtk --no-symmetric --no-tgd --no-rkhs
```

### Typical full pipeline

```bash
clarcs recenter   target.vtk  target-rc.vtk  --save-plane
clarcs normalize  target-rc.vtk  target-rcs.vtk  --target reference.vtk
clarcs nlregister target-rcs.vtk  reference.vtk  target-registered.vtk \
                  --deformation target-deformation.vtk
```

---

## Output files

| File | Content |
|---|---|
| `OUTPUT` | Warped moving surface (same connectivity as INPUT) |
| `FIELD` | Original surface + deformation vectors as VTK VECTORS point data |

---

## Python API

```python
from pyclarcs.io import load_surface, load_surface_with_normals, save_surface
from pyclarcs.nonrigid import nonrigid_icp_multires, apply_deformation

mov_pts, mov_poly, mov_normals = load_surface_with_normals("target.vtk")
ref_pts, ref_poly              = load_surface("reference.vtk")
_, _,    ref_normals           = load_surface_with_normals("reference.vtk")

def_field = nonrigid_icp_multires(
    mov_pts, mov_normals,
    ref_pts, ref_normals,
    mov_poly, ref_poly,   # ref_poly enables mesh-based TGD for reference
    # All algorithmic improvements are on by default:
    # symmetric=True, use_tgd=True, use_rkhs=True
)

warped = apply_deformation(mov_pts, def_field)
save_surface("registered.vtk", warped, mov_poly)
```
