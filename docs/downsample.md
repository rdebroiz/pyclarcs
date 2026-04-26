# `clarcs downsample` — mesh decimation

Reduce the vertex count of a surface using quadric error decimation.

---

## Usage

```bash
clarcs downsample INPUT [OUTPUT] (--target-n N | --ratio R) [options]
```

Exactly one of `--target-n` or `--ratio` is required.

**Arguments and options:**

| Argument / Flag | Description |
|---|---|
| `INPUT` | Input surface (any supported format) |
| `OUTPUT` | Output file. Defaults to `<INPUT_STEM>-downsampled<EXT>` |
| `--target-n N` | Target vertex count |
| `--ratio R` | Target fraction of the original count (e.g. `0.1` = 10 %) |
| `-q / --quiet` | Suppress all output |

---

## Method

Uses VTK's `vtkQuadricDecimation`, which minimises the quadric error metric
at each edge collapse.  The resulting vertex count may differ slightly from
the target due to topological constraints.

This is the same decimation used internally by `clarcs atlas` to build its
multi-resolution hierarchy.

---

## Examples

```bash
# Reduce to approximately 5 000 vertices
clarcs downsample brain.ply brain-5k.ply --target-n 5000

# Keep 10 % of the original vertices
clarcs downsample skull.vtk skull-10pct.vtk --ratio 0.1

# Batch processing
for f in subjects/*.ply; do
    clarcs downsample "$f" "downsampled/$(basename $f)" --target-n 5000 -q
done
```

---

## Python API

```python
from pyclarcs.io import load_surface, save_surface
from pyclarcs.mesh import decimate_surface

pts, polygons = load_surface("brain.ply")

pts_d, polygons_d = decimate_surface(pts, polygons, target_n=5000)
save_surface("brain-5k.ply", pts_d, polygons_d)
```
