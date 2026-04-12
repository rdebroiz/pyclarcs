# `clarcs orient` — axis permutation

Permute the coordinate axes of a surface.

---

## Usage

```bash
clarcs orient INPUT [OUTPUT] --axes X Y Z [options]
```

**Arguments and options:**

| Argument / Flag | Description |
|---|---|
| `INPUT` | Input surface |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-oriented<EXT>` |
| `--axes X Y Z` | Destination indices for the current x, y, z axes (default: `0 1 2` = identity) |
| `-q / --quiet` | Suppress all output |

---

## Method

Each axis of the input is sent to the destination index given by `--axes`.
`--axes A B C` means:

- current x-axis → output axis A
- current y-axis → output axis B
- current z-axis → output axis C

`A B C` must be a permutation of `{0, 1, 2}`; any repeated index raises an
error.

---

## Examples

```bash
# Identity (no change)
clarcs orient surface.vtk --axes 0 1 2

# Swap x and z axes
clarcs orient surface.vtk --axes 2 1 0

# Cyclic permutation x→1, y→2, z→0
clarcs orient surface.vtk --axes 1 2 0
```

---

## Python API

```python
from pyclarcs.io import load_surface, save_surface
from pyclarcs.alignment import reorient_axes

pts, poly = load_surface("surface.vtk")

# Swap x and z
result = reorient_axes(pts, x_to=2, y_to=1, z_to=0)
save_surface("surface-oriented.vtk", result, poly)
```
