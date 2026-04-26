# `clarcs asymmetry` — pointwise asymmetry field

Quantify left-right asymmetry at every vertex of a bilateral surface.

---

## Scientific background

For an approximately bilateral surface $X$, let $S(X)$ denote the mirror
of $X$ across its symmetry plane.  Non-rigidly registering $S(X)$ onto
$X$ yields a deformation field $T$ such that $S(X) + T \approx X$.

The **pointwise asymmetry** at vertex $i$ is the vector $T_i$: its
direction encodes which way the surface deviates from bilateral symmetry,
and its norm $\|T_i\|$ gives the local asymmetry magnitude (in mm).

This approach follows Combès & Prima (MICCAI 2008) and Abadie et al.
(MeshMed 2011).

---

## Usage

```bash
clarcs asymmetry INPUT [OUTPUT] [--plane PLANE.pl] [--save-warped] [options]
```

**Arguments and options:**

| Argument / Flag | Default | Description |
|---|---|---|
| `INPUT` | — | Bilateral surface |
| `OUTPUT` | auto | VTK file with VECTORS point data. Defaults to `<INPUT_STEM>-asymmetry.vtk` |
| `--plane PLANE.pl` | auto | Pre-computed symmetry plane. Estimated if omitted |
| `--save-warped` | off | Also save the registered mirror as `<OUTPUT_STEM>-warped<EXT>` |
| `--max-iter N` | `80` | EM iterations |
| `--n-levels N` | auto | Resolution levels |
| `--no-symmetric` | off | Disable symmetric correspondences |
| `--no-tgd` | off | Disable TGD geodesic shape prior |
| `--no-rkhs` | off | Use Laplacian M-step instead of RKHS |
| `-q / --quiet` | — | Suppress all output |

> **Output format** — the asymmetry field is stored as VTK **VECTORS** point
> data on the **original** surface geometry.  The output must be `.vtk` or
> `.vtp`; other extensions are coerced to `.vtk`.

---

## Method

1. **Symmetry plane** — estimated (full pipeline) or loaded from `--plane`.
2. **Mirror** — reflect the surface across the plane; reverse face winding to
   restore outward normals.
3. **Non-rigid registration** — `register(mirror, original)` gives deformation
   field $T$.
4. **Save** — $T$ stored as VECTORS on the original surface geometry.

Since vertex $i$ of the mirror corresponds to vertex $i$ of the original
(same topology, just reflected), displaying $T$ on the original surface gives
the intuitive interpretation: "at this anatomical location, the surface is this
many mm away from bilateral symmetry".

---

## Visualisation in ParaView

Open the `.vtk` output in ParaView:

- **Colour by** `asymmetry` norm → asymmetry magnitude map.
- Apply **Glyph** filter → oriented asymmetry vectors.
- Apply **Warp By Vectors** (`asymmetry`) → exaggerated deformation visualisation.

---

## Examples

```bash
# Compute asymmetry with automatic plane estimation
clarcs asymmetry brain.ply brain-asymmetry.vtk

# Use a pre-computed plane (faster)
clarcs asymmetry brain.ply brain-asymmetry.vtk --plane brain.pl

# Also save the registered mirror (for quality check)
clarcs asymmetry brain.ply brain-asymmetry.vtk --save-warped

# Batch processing
for f in subjects/*.ply; do
    stem=$(basename $f .ply)
    clarcs asymmetry "$f" "asym/${stem}-asymmetry.vtk" --plane "planes/${stem}.pl" -q
done
```

---

## RMS output (always printed)

```
RMS mirror→original: 4.2100 mm → 0.8350 mm  (+80.2%)
```

- **Before** — distance between the raw mirror and the original (global asymmetry scale).
- **After** — residual after registration (how well the mirror fits after non-rigid correction).

---

## Python API

```python
from pyclarcs.io import load_surface_with_normals, save_deformation_vtk
from pyclarcs.io import compute_surface_normals
from pyclarcs.alignment import reflect_surface
from pyclarcs.nonrigid import register, apply_deformation

pts, polys, normals = load_surface_with_normals("brain.ply")

# Mirror
from pyclarcs.symmetry import SymmetryPlane
plane = SymmetryPlane.load("brain.pl")
mir_pts = reflect_surface(pts, plane.n, plane.n * plane.d)
mir_polys   = [f[::-1] for f in polys]
mir_normals = compute_surface_normals(mir_pts, mir_polys)

# Register mirror → original
def_field = register(mir_pts, mir_normals, pts, normals, mir_polys, polys)

# Save asymmetry field on original geometry
save_deformation_vtk("brain-asymmetry.vtk", pts, polys, def_field,
                     deformation_name="asymmetry")
```

---

## See also

[`clarcs project-asym`](project-asym.md) — project per-subject asymmetry
fields from this command onto a common atlas surface for population analysis.
