# `clarcs atlas` — mean shape atlas construction

Build a mean surface atlas from a population of surfaces.

---

## Scientific background

The atlas $M$ is defined as the Fréchet mean of the population
$\{X^1, \dots, X^n\}$ under the pseudo-distance $\delta$:

$$M = \underset{X}{\arg\min} \sum_{i=1}^{n} \delta(X, X^i)$$

The iterative algorithm (Abadie et al., MeshMed 2011) alternates between:

1. **Non-rigid registration** — register the current mean shape $M$
   (moving) toward each subject $X^i$ (reference).
2. **Averaging** — replace $M$ with the coordinate-wise mean of all
   warped copies.

Using $M$ as the moving surface ensures that all warped copies share the
same topology and vertex count, so the mean is well-defined regardless
of the subjects' individual topologies.

---

## Usage

```bash
clarcs atlas SUBJECTS_DIR OUTPUT [--save-registered] [options]
```

**Arguments and options:**

| Argument / Flag | Default | Description |
|---|---|---|
| `SUBJECTS_DIR` | — | Directory of subject surfaces (sorted alphabetically; first = initial template) |
| `OUTPUT` | — | Output atlas surface |
| `--save-registered` | off | Save the atlas warped toward each subject as `<OUTPUT_STEM>-registered-<name><EXT>` |
| `--atlas-iter N` | `3` | Number of register-all → average cycles |
| `--no-prealign` | off | Disable symmetry-plane + CoM pre-alignment before each registration |
| `--max-iter N` | `80` | EM iterations per registration |
| `--n-levels N` | auto | Resolution levels (auto from surface size) |
| `--no-symmetric` | off | Disable symmetric correspondences (Reg2) |
| `--no-tgd` | off | Disable TGD geodesic shape prior (Reg3) |
| `--no-rkhs` | off | Use Laplacian M-step instead of RKHS Wu kernel |
| `-q / --quiet` | — | Suppress all output |

---

## Pre-alignment (default)

Before each registration pair, the command:

1. Estimates the bilateral symmetry plane of both the mean shape and the
   subject via the **principal-axis inertia tensor** (fast, no ICP).
2. Aligns both to the canonical plane ($x = 0$).
3. Translates the mean shape so its centre of mass matches the subject's.

This rigid initialisation reduces the gap before non-rigid registration and
makes the atlas robust to varying subject positions.

A warning is printed when the symmetry residual exceeds **5 %** of the
bounding-box diagonal (surface may not be approximately bilateral).

The **final** mean shape is recentered to its own canonical plane at the end.

Disable with `--no-prealign`.

---

## Output — `--save-registered`

Each `<OUTPUT_STEM>-registered-<subject_name><EXT>` file is the atlas surface
deformed to match the corresponding subject.  These surfaces all share the
atlas topology (same vertex count, same connectivity) and are the natural
input for downstream population statistics or PCA.

See also [`clarcs project-asym`](project-asym.md) which builds the atlas
and projects per-subject asymmetry fields in a single command.

---

## Examples

```bash
# Build atlas from all .ply files in subjects/
clarcs atlas subjects/ atlas.vtk

# Also save per-subject registered surfaces
clarcs atlas subjects/ atlas.vtk --save-registered

# Fewer iterations, lower resolution (faster preview)
clarcs atlas subjects/ atlas.vtk --atlas-iter 1 --n-levels 1 --max-iter 20

# Skip pre-alignment (if subjects are already roughly aligned)
clarcs atlas subjects/ atlas.vtk --no-prealign
```

---

## RMS output

At the end, the command always prints:

```
RMS (atlas→subjects): 3.2100 mm → 1.4350 mm  (+55.3%)  max=2.1400 mm
```

- **Before** — nearest-neighbour distance from the initial template to each subject.
- **After** — nearest-neighbour distance from each registered atlas surface to the corresponding subject.

---

## Python API

```python
from pyclarcs.io import load_surface_with_normals, save_surface
from pyclarcs.atlas import build_atlas

subjects = [load_surface_with_normals(p) for p in sorted(paths)]

atlas_pts, atlas_poly, registered = build_atlas(
    subjects,
    atlas_iter=3,
    prealign=True,   # symmetry-plane + CoM pre-alignment (default)
    n_levels=2,
    max_iter=80,
)

save_surface("atlas.vtk", atlas_pts, atlas_poly)

# registered[i]: atlas vertices deformed toward subject i
# → same topology as atlas, suitable for PCA
```
