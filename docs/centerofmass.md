# `clarcs centerofmass` — centre-of-mass alignment

Translate a surface so its centre of mass coincides with a reference surface.

---

## Usage

```bash
clarcs centerofmass INPUT [OUTPUT] --target TARGET [options]
```

**Arguments and options:**

| Argument / Flag | Description |
|---|---|
| `INPUT` | Surface to move |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-centerofmass<EXT>` |
| `--target TARGET` | Reference surface (required) |
| `-q / --quiet` | Suppress all output |

---

## Method

The transform is a pure translation:

$$
T(\mathbf{x}) = \mathbf{x} + (\bar{\mathbf{t}} - \bar{\mathbf{s}})
$$

where $\bar{\mathbf{s}}$ is the centroid of the input surface and
$\bar{\mathbf{t}}$ is the centroid of the target surface.
Relative distances between all points are preserved exactly.

---

## Examples

```bash
# Move skull.vtk so its centre of mass matches reference.vtk
clarcs centerofmass skull.vtk aligned.vtk --target reference.vtk

# Default output name → skull-centerofmass.vtk
clarcs centerofmass skull.vtk --target reference.vtk
```

---

## Python API

```python
from pyclarcs.io import load_surface, save_surface
from pyclarcs.alignment import align_center_of_mass

src_pts, src_poly = load_surface("skull.vtk")
tgt_pts, _        = load_surface("reference.vtk")

result = align_center_of_mass(src_pts, tgt_pts)
save_surface("aligned.vtk", result, src_poly)
```
