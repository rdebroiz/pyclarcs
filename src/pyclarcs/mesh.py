"""
Mesh topology utilities.

ROLE
====
The non-rigid registration algorithm requires two pieces of connectivity
information that are not needed by the symmetry-plane pipeline:

1. **Adjacency lists** — for each vertex i, the list of its direct
   neighbours in the mesh (vertices that share an edge with i).  Used
   in the ICM update to compute the Laplacian regularisation term
   Σ_{j ∈ N_i} d_j.

2. **CSR adjacency matrix** — the same information in scipy sparse format,
   enabling vectorised sparse matrix–vector products  A @ def_field  instead
   of a Python loop over neighbours.

Both are derived purely from the polygon list returned by ``load_surface``.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mesh_adjacency(
    polygons: list[list[int]],
    n_points: int,
) -> list[list[int]]:
    """Build per-vertex neighbour lists from a polygon mesh.

    For each face (triangle or quad), every pair of consecutive vertices
    (including the wrap-around edge) is recorded as a symmetric edge.
    Self-loops are discarded.

    Parameters
    ----------
    polygons : list of face index lists
        As returned by ``load_surface``.
    n_points : int
        Total number of vertices (upper bound on vertex indices).

    Returns
    -------
    list of length n_points
        ``adj[i]`` is a sorted list of vertex indices adjacent to vertex i.
    """
    adj: list[set[int]] = [set() for _ in range(n_points)]
    for face in polygons:
        n = len(face)
        for k in range(n):
            i = face[k]
            j = face[(k + 1) % n]
            if i != j:
                adj[i].add(j)
                adj[j].add(i)
    return [sorted(s) for s in adj]


def adjacency_csr(
    polygons: list[list[int]],
    n_points: int,
) -> csr_matrix:
    """Build a symmetric adjacency matrix in CSR format.

    Entry (i, j) is 1.0 if vertices i and j share a mesh edge.
    The matrix is symmetric and has zero diagonal.

    Parameters
    ----------
    polygons : list of face index lists
    n_points : int

    Returns
    -------
    csr_matrix of shape (n_points, n_points)
        Symmetric, binary adjacency matrix.
    """
    adj = mesh_adjacency(polygons, n_points)
    rows: list[int] = []
    cols: list[int] = []
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
