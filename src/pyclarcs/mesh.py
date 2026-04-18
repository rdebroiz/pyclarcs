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
# Internal VTK helpers
# ---------------------------------------------------------------------------

def _to_polydata(points: np.ndarray, faces: list[list[int]]):
    """Build a vtkPolyData from numpy arrays (lazy VTK import)."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(points.astype("float32"), deep=True))
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for idx in face:
            cells.InsertCellPoint(idx)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetPolys(cells)
    return poly


def _from_polydata(poly) -> tuple[np.ndarray, list[list[int]]]:
    """Extract (points, faces) from a vtkPolyData."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    faces: list[list[int]] = []
    poly.GetPolys().InitTraversal()
    id_list = vtk.vtkIdList()
    while poly.GetPolys().GetNextCell(id_list):
        faces.append([id_list.GetId(i) for i in range(id_list.GetNumberOfIds())])
    return pts, faces


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decimate_surface(
    points: np.ndarray,
    faces: list[list[int]],
    target_n: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Decimate a mesh to approximately *target_n* vertices.

    Uses ``vtkQuadricDecimation`` which reliably achieves high reduction
    ratios (unlike ``vtkDecimatePro`` with topology preservation, which
    stalls on complex meshes like brain surfaces).

    Parameters
    ----------
    points : ndarray (N, 3)
    faces  : list of face index lists
    target_n : int
        Desired number of vertices in the output mesh.

    Returns
    -------
    (points, faces) of the decimated mesh.
        The vertex count may differ slightly from *target_n* due to
        topological constraints.
    """
    import vtk

    reduction = max(0.0, min(0.99, 1.0 - target_n / len(points)))
    deci = vtk.vtkQuadricDecimation()
    deci.SetInputData(_to_polydata(points, faces))
    deci.SetTargetReduction(reduction)
    deci.Update()
    return _from_polydata(deci.GetOutput())


def compute_vertex_normals(
    points: np.ndarray,
    faces: list[list[int]],
) -> np.ndarray:
    """Compute smooth per-vertex normals using VTK.

    Parameters
    ----------
    points : ndarray (N, 3)
    faces  : list of face index lists

    Returns
    -------
    normals : ndarray (N, 3)
        Unit outward normals at each vertex.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(_to_polydata(points, faces))
    nf.ComputePointNormalsOn()
    nf.ComputeCellNormalsOff()
    nf.SplittingOff()   # preserve point count — no edge splitting
    nf.Update()
    return vtk_to_numpy(
        nf.GetOutput().GetPointData().GetNormals()
    ).astype(float)

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


def mesh_edge_graph(
    polygons: list[list[int]],
    points: np.ndarray,
) -> csr_matrix:
    """Build an edge-weighted CSR graph from a mesh for geodesic computations.

    Edge weights are Euclidean distances between vertices.

    Parameters
    ----------
    polygons : list of face index lists
    points   : ndarray (N, 3)

    Returns
    -------
    csr_matrix (N, N)  — symmetric, zero diagonal, weight = edge length
    """
    n_pts = len(points)
    adj = mesh_adjacency(polygons, n_pts)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            rows.append(i)
            cols.append(j)
            data.append(float(np.linalg.norm(points[i] - points[j])))
    return csr_matrix((data, (rows, cols)), shape=(n_pts, n_pts))


def compute_tgd(
    points: np.ndarray,
    polygons: list[list[int]],
    n_seeds: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Approximate normalised Total Geodesic Distance (TGD) for each vertex.

    TGD(i) = Σ_j d_g(i, j) / max_k Σ_j d_g(k, j)

    Full all-pairs geodesic is O(N²) so we approximate by running Dijkstra
    from *n_seeds* random vertices and summing.  Because the normalisation
    factor (the maximum row sum) does not need to be exact, the approximation
    is robust: relative TGD values converge quickly as n_seeds grows.

    On brain pial surfaces the TGD separates sulcal depths (high TGD) from
    gyral crests (low TGD), providing a strong shape prior that prevents
    cross-sulcus correspondences in the E-step.

    Parameters
    ----------
    points   : ndarray (N, 3)
    polygons : list of face index lists
    n_seeds  : int  — number of random Dijkstra sources (200 gives <1 % error)
    seed     : int  — RNG seed

    Returns
    -------
    tgd : ndarray (N,) in [0, 1]
    """
    from scipy.sparse.csgraph import dijkstra

    graph = mesh_edge_graph(polygons, points)
    n_pts = len(points)
    rng   = np.random.default_rng(seed)
    sources = rng.choice(n_pts, size=min(n_seeds, n_pts), replace=False)

    # dijkstra returns (n_seeds, N) distance matrix from each source
    dist = dijkstra(graph, indices=sources, directed=False)  # (n_seeds, N)

    # Replace inf (unreachable vertices) with the largest finite value so that
    # isolated / disconnected vertices are treated as maximally far away rather
    # than propagating NaN through the normalisation step.
    finite_max = float(dist[np.isfinite(dist)].max()) if np.isfinite(dist).any() else 1.0
    dist = np.where(np.isinf(dist), finite_max, dist)

    # Sum over seed distances: proportional to true TGD (scale by N/n_seeds)
    tgd = dist.sum(axis=0)  # (N,)

    max_val = tgd.max()
    if max_val > 0:
        tgd /= max_val
    return tgd.astype(float)


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
