# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**pyclarcs** is a Python toolkit for automated analysis of 3D anatomical surfaces (endocranial and bilateral structures). It implements:
- **Symmetry-plane estimation** (Combès et al., CVPR 2008) — bilateral symmetry of 3D surfaces
- **Non-rigid surface registration** — EM-ICP with symmetric correspondences, TGD geodesic prior, and RKHS Wu-kernel M-step (Combès & Prima, CVIU 2019)

Python ≥3.9, version 0.3.0. Entry point: the `clarcs` CLI (alias `pyclarcs`).

## Development setup and commands

```bash
# Development install
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test file
pytest tests/test_symmetry.py

# Run a single test
pytest tests/test_symmetry.py::test_reflection_identity

# Generate sample data for manual testing
python data/generate_samples.py
python data/run_pipeline.py results/
```

No linter or formatter is configured.

## Architecture

All source code lives in `src/pyclarcs/`.

### Module map

| Module | Role |
|--------|------|
| `_cli.py` | Click CLI group with 7 commands (`symplane`, `recenter`, `centerofmass`, `normalize`, `nlregister`, `mirror`, `reorient`) |
| `io.py` | Surface I/O — format inferred from extension (`.vtk`, `.vtp`, `.vtu`, `.ply`, `.stl`, `.obj`) |
| `symmetry.py` | `SymmetryPlane` class: plane geometry, reflection, M-step fitting, `.pl` serialization |
| `principal_axes.py` | Inertia-tensor-based initialization for symmetry estimation |
| `coarse.py` | Coarse ICP with trimmed estimator |
| `fine.py` | Fine EM-ICP annealing + doubly-stochastic refinement |
| `alignment.py` | Rigid transforms (recenter, rescale, center-of-mass alignment) |
| `mesh.py` | Adjacency lists, decimation, TGD geodesic prior, Wu RKHS kernel |
| `nonrigid.py` | Non-rigid EM-ICP: `register` (public orchestrator), `_em_icp` (single-level EM loop), `_build_hierarchy`, `_prepare_tgd` |
| `_numba_kernels.py` | Numba JIT-compiled hot loops; `_warmup()` is called at import time |

### Surface representation

A surface is always a `(points, polygons)` pair:
- `points`: `ndarray (N, 3)` float64
- `polygons`: list of lists (variable polygon sizes — not necessarily triangles)

Normals are computed on demand via `vtkPolyDataNormals` (with consistency enforcement).

### Symmetry estimation pipeline

```
principal_axes → coarse ICP (trimmed) → EM-ICP annealing → doubly-stochastic refinement
```

`SymmetryPlane` is parametrized by unit normal `n` and offset `d`. Reflection: `p' = (I − 2nnᵀ)p + 2dn`. Serialized as `.pl` (compatible with C++ ZZ_SYMC tool).

### Non-rigid registration pipeline

`register()` runs EM-ICP at multiple resolution levels (auto-computed: ≤5k vertices → 1 level, ≤30k → 2, >30k → 3). Three algorithmic improvements are layered in:
- **Reg2** — symmetric correspondences (prevents many-to-one mappings)
- **Reg3** — TGD geodesic prior (shape constraint)
- **RKHS** — Wu-kernel M-step (topology-independent regularization)

Key parameters auto-estimated from geometry when not provided by the user:
- `sigma` — 75th percentile of nearest-neighbour distances to reference
- `beta` — derived from mesh spacing
- `sigma_min` — mesh spacing / 2

### Performance

Hot inner loops (E-step Gaussian weights, M-step pair collection, subsampling) are Numba JIT-compiled in `_numba_kernels.py` with explicit `float64`/`int64` types for predictable performance. BLAS threading must be limited to 1 thread to avoid contention with Numba's thread pool.

### CLI output naming

When the output path is omitted, files are named `<INPUT_STEM>-<COMMAND><EXT>`. All commands accept `-q/--quiet` to suppress progress messages.

## Key references

- Combès & Prima, CVIU 2019 — non-rigid registration theory behind `nlregister`
- Combès et al., CVPR 2008 — symmetry-plane estimation theory behind `symplane`
- C++ original: ZZ_SYMC (CLARCS project)
