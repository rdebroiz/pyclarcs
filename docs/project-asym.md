# `clarcs project-asym` — atlas construction + asymmetry projection

Build a mean shape atlas from a population and project each subject's
pointwise asymmetry field onto it.  The output is a population-level
asymmetry map on the common atlas surface.

This command combines [`clarcs atlas`](atlas.md) and
[`clarcs asymmetry`](asymmetry.md) into a single pipeline, with support for
reusing pre-computed intermediates at any stage.

---

## Usage

```bash
clarcs project-asym SUBJECTS_DIR OUTPUT [options]
```

**Arguments:**

| Argument | Description |
|---|---|
| `SUBJECTS_DIR` | Directory of bilateral subject surfaces (sorted alphabetically) |
| `OUTPUT` | Mean asymmetry field (VTK VECTORS on atlas geometry, `.vtk` or `.vtp`) |

**Atlas options** (ignored when `--atlas` is provided):

| Flag | Default | Description |
|---|---|---|
| `--atlas PATH` | — | Pre-existing atlas surface (skip atlas build) |
| `--save-atlas PATH` | — | Save the built atlas surface |
| `--atlas-iter N` | `3` | Atlas construction cycles |
| `--no-prealign` | off | Disable sym-plane + CoM pre-alignment in atlas build |

**Pre-computed intermediates:**

| Flag | Description |
|---|---|
| `--registered-dir DIR` | Pre-computed registered atlas surfaces (from `clarcs atlas --save-registered`). Only used with `--atlas`. Files matched by alphabetical sort order |
| `--asymmetry-dir DIR` | Pre-computed asymmetry fields (from `clarcs asymmetry`). Files matched by alphabetical sort order |

**Output options:**

| Flag | Description |
|---|---|
| `--save-stats` | Save per-vertex norm statistics: `<OUTPUT_STEM>-std/min/max<EXT>` |
| `--save-individual` | Save per-subject projected fields: `<OUTPUT_STEM>-<subject><EXT>` |

**Registration options:**

| Flag | Default | Description |
|---|---|---|
| `--max-iter N` | `80` | EM iterations per registration |
| `--n-levels N` | auto | Resolution levels |
| `--no-symmetric` | off | Disable symmetric correspondences |
| `--no-tgd` | off | Disable TGD geodesic shape prior |
| `--no-rkhs` | off | Use Laplacian M-step instead of RKHS |
| `-q / --quiet` | — | Suppress all output |

---

## Pipeline

For each subject:

1. **Asymmetry** — compute or load the deformation field $T^i$ mapping the
   mirror of subject $i$ onto subject $i$ itself.
2. **Registered surface** — compute or load the atlas vertices deformed into
   subject $i$'s coordinate space (from atlas construction).
3. **Projection** — for each atlas vertex $k$, interpolate $T^i$ by
   inverse-distance weighting from the subject vertices nearest to the
   registered atlas position.

$$\hat{T}^i_k = \sum_{j \in \mathcal{N}(k)} w_{kj}\, T^i_j$$

After collecting all projected fields, the command computes:

$$\bar{T}_k = \frac{1}{n} \sum_{i=1}^n \hat{T}^i_k \quad \text{(mean, saved to OUTPUT)}$$

$$\sigma_k = \text{std}_i\!\left(\|\hat{T}^i_k\|\right) \quad \text{(norm std, saved with --save-stats)}$$

---

## Usage modes

### One-pass (build everything)

```bash
clarcs project-asym subjects/ mean-asym.vtp \
       --save-atlas atlas.vtp \
       --save-stats \
       --save-individual
```

Builds the atlas, computes subject asymmetries, projects, aggregates.

### Reuse pre-computed atlas

```bash
# Atlas already built with:
#   clarcs atlas subjects/ atlas.vtp --save-registered
# Asymmetries already computed with:
#   clarcs asymmetry B01.ply B01-asym.vtk  (for each subject)

clarcs project-asym subjects/ mean-asym.vtp \
       --atlas       atlas.vtp \
       --registered-dir registered/ \
       --asymmetry-dir  asymmetry/
```

### Build atlas, reuse asymmetries

```bash
clarcs project-asym subjects/ mean-asym.vtp \
       --asymmetry-dir asymmetry/ \
       --save-atlas atlas.vtp
```

---

## Output files

| File | Content |
|---|---|
| `OUTPUT` | Mean asymmetry VECTORS on atlas geometry |
| `<OUTPUT_STEM>-std<EXT>` | Per-vertex std of asymmetry norm (`--save-stats`) |
| `<OUTPUT_STEM>-min<EXT>` | Per-vertex min of asymmetry norm (`--save-stats`) |
| `<OUTPUT_STEM>-max<EXT>` | Per-vertex max of asymmetry norm (`--save-stats`) |
| `<OUTPUT_STEM>-<subject><EXT>` | Per-subject projected field (`--save-individual`) |

All files use the atlas surface geometry.  Open in ParaView and colour by
`asymmetry_mean` norm for the population asymmetry map; use `asymmetry_std`
to identify regions of high inter-subject variability.

---

## Python API

```python
from pyclarcs.io import load_surface_with_normals, save_deformation_vtk
from pyclarcs.atlas import build_atlas, project_asymmetry_to_atlas
import numpy as np

subjects = [load_surface_with_normals(p) for p in sorted(paths)]

# Build atlas → registered[i] = atlas in subject i's space
atlas_pts, atlas_poly, registered = build_atlas(subjects, atlas_iter=3)

# Per-subject asymmetry fields (ndarray (M_i, 3)) at subject vertex positions
# ... compute or load asym_fields[i] and subject_pts[i] ...

projected = project_asymmetry_to_atlas(registered, asym_fields, subject_pts)

arr = np.stack(projected)           # (n, N, 3)
mean_asym = arr.mean(axis=0)        # (N, 3)  → save as VECTORS
std_norms = np.linalg.norm(arr, axis=2).std(axis=0)   # (N,) → save as scalars

save_deformation_vtk("mean-asym.vtk", atlas_pts, atlas_poly, mean_asym,
                     deformation_name="asymmetry_mean")
```
