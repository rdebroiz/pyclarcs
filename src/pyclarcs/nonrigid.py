"""
Non-rigid EM-ICP surface registration.

THEORETICAL BACKGROUND
=======================
This module implements a non-rigid registration algorithm from the CLARCS
framework.  It is a variant of the EM-ICP method where the transform is a
smooth, per-vertex deformation field regularised by a first-order graph
Laplacian prior on the mesh topology.

ALGORITHM
=========

Notation
--------
  x_i  : vertex i of the moving surface (i = 0 … N-1)
  n_i  : unit normal at x_i
  y_j  : vertex j of the reference surface (j = 0 … M-1)
  m_j  : unit normal at y_j
  d_i  : deformation vector attached to x_i  (unknown, initialised to 0)
  N_i  : set of mesh-edge neighbours of x_i

Outer loop (max_iter iterations)
---------------------------------

E-step — doubly-stochastic fuzzy correspondences
  For each vertex i:
      T_i = x_i + d_i                       (current transformed position)

      For each y_j within radius r of T_i whose normal is compatible
      (n_i · m_j ≥ 0):
          w_ij = exp( −‖T_i − y_j‖ / σ )

      Row sum:  sR_i = Σ_j w_ij
      Col sum:  sC_j = Σ_i w_ij

      Doubly-stochastic weight:
          ṽ_ij = w_ij / sC_j  +  w_ij / sR_i

      Total weight:  W_i  = Σ_j ṽ_ij
      Target point:  ȳ_i  = (Σ_j ṽ_ij y_j) / W_i

  Implementation: a single chunked KDTree pass builds the COO arrays
  (rows, cols, weights).  Row/col sums and barycentres are then computed
  with numpy bincount — no N×M sparse matrix is ever constructed.

E-step — Gaussian kernel + row-normalised weights + outlier term

  For each vertex i:
      T_i = x_i + d_i

      For each y_j within radius r of T_i, normal compatible
      (n_i · m_j ≥ normal_min_dot):
          w_ij = exp( −‖T_i − y_j‖² / (2σ²) )

      Row sum:  sR_i = Σ_j w_ij

      Outlier-adjusted weight (CPD-style):
          W_i = sR_i / (sR_i + c_outlier)

      Barycentre:  ȳ_i = (Σ_j w_ij · y_j) / sR_i

  The outlier constant is:
      c_outlier = w/(1−w) · (2πσ²)^(3/2) · M/N
  where w = outlier_weight ∈ [0, 1).

  Using raw row sums (not doubly-stochastic) means W_i scales with
  the number and quality of correspondences.  This ensures the data
  term dominates when correspondences are dense (early/coarse) and
  the regularisation takes over when they thin out (fine/late).

M-step — preconditioned conjugate gradient
  Solves the sparse linear system  M · d[:,k] = b[:,k]  independently
  for each coordinate k ∈ {0, 1, 2}:

      M   =  diag(W_i + β·|N_i|)  −  β · A          (symmetric PSD)
      b_i =  W_i · (ȳ_i − x_i)_k

  Preconditioning with the diagonal of M (Jacobi preconditioner) gives
  O(√κ) convergence vs. O(κ) for unaccelerated Jacobi.  The previous
  outer-iteration solution warm-starts the solve.

Annealing
  σ ← max(σ / 2,  σ_min)   every `period_sigma` outer iterations.

DEFAULT PARAMETERS
  sigma        = 3.0    initial bandwidth (std dev of Gaussian kernel)
  beta         = 0.5    regularisation weight
  dist_cutoff  = 15.0   search radius (typically 3–4 σ)
  max_iter     = 80
  icm_iter     = 50     max CG iterations per outer iteration
  period_sigma = 20     sigma halved every 20 outer iterations
  sigma_min    = 0.1
  outlier_weight = 0.1  fraction of points expected to be outliers
  e_chunk      = 2000   vertices per KDTree query batch
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, diags as sp_diags
from scipy.sparse.linalg import cg as sp_cg


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------

def estimate_registration_params(
    mov_pts: np.ndarray,
    ref_pts: np.ndarray,
    *,
    max_iter: int = 80,
    sigma_min: float | None = None,
    n_sample: int = 2000,
    seed: int = 0,
) -> dict:
    """Estimate good EM-ICP parameters from the two input surfaces.

    A random subsample of the moving surface is queried against the
    reference KDTree to obtain the nearest-neighbour distance distribution,
    which directly characterises the initial surface-to-surface gap.  A
    self-NN query on the same subsample gives the local mesh spacing, which
    drives the annealing floor and regularisation weight.

    Parameters
    ----------
    mov_pts : ndarray (N, 3)
    ref_pts : ndarray (M, 3)
    max_iter : int
        Outer iterations (needed to compute period_sigma).
    sigma_min : float or None
        Annealing floor.  None → auto from mesh spacing
        (``max(0.1, mesh_spacing / 2)``).
    n_sample : int
        Number of moving points to subsample for the distance estimate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``sigma``, ``dist_cutoff``, ``period_sigma``,
    ``sigma_min``, ``beta``, ``n_levels``, ``mesh_spacing``.

    Notes
    -----
    Heuristics
      mesh_spacing = median 1-NN distance within the moving surface sample
      sigma_min    = max(0.1, mesh_spacing / 2)   [if not overridden]
      beta         = 0.5  [regularisation weight — tuned for Gaussian kernel + row sums]
      n_levels     = 1 if N≤5 000 else 2 if N≤30 000 else 3
      sigma        = 75th percentile of NN-to-ref distances
                     Using the 75th percentile (not the median) ensures that the
                     initial kernel is wide enough for non-convex surfaces (e.g.
                     brain pial) where the nearest reference vertex can be an
                     anatomically wrong match on an adjacent fold.  A broader
                     initial sigma lets the algorithm "see past" such false
                     candidates and then progressively tighten via annealing.
      dist_cutoff  = 99th percentile of NN-to-ref distances × 1.5
                     (floor: sigma × 3)
      period_sigma = max_iter // ceil(log2(sigma / sigma_min))
                     (halvings from sigma to sigma_min spread evenly)
    """
    rng = np.random.default_rng(seed)
    mov_pts = np.asarray(mov_pts, dtype=float)
    ref_pts = np.asarray(ref_pts, dtype=float)

    idx = rng.choice(len(mov_pts), size=min(n_sample, len(mov_pts)), replace=False)
    sample = mov_pts[idx]

    # Nearest-neighbour to reference: characterises the surface-to-surface gap.
    nn_dists, _ = KDTree(ref_pts).query(sample, k=1, workers=-1)

    # Self-NN within the moving surface: characterises the mesh spacing.
    # k=2 because k=1 is the point itself (distance 0).
    nn_self, _ = KDTree(mov_pts).query(sample, k=2, workers=-1)
    mesh_spacing = float(np.median(nn_self[:, 1]))

    # Annealing floor: stop at half the mesh spacing so the last iterations
    # still form meaningful correspondences.
    if sigma_min is None:
        sigma_min = max(0.1, mesh_spacing / 2.0)

    # Regularisation weight β.
    # With row-sum weights W ≈ n_neighbours × exp(−0.5) at fine scale and
    # mean degree ≈ 6, we target ~30 % data trust at sigma_min:
    #   W / (W + β × deg) ≈ 0.3  →  β ≈ W × (1/0.3 − 1) / deg
    # At sigma_min, n_neighbours ≈ 4 within 2σ_min, mean weight ≈ 0.4
    # → W ≈ 1.6, β ≈ 1.6 × 2.33 / 6 ≈ 0.62.
    # We use β = 0.5 as a safe default that keeps the deformation smooth
    # without over-constraining large-scale corrections at coarse levels.
    beta = 0.5

    # Recommended number of resolution levels.
    N = len(mov_pts)
    n_levels = 1 if N <= 5_000 else (2 if N <= 30_000 else 3)

    # Use the 75th percentile so that the initial kernel is wide enough to
    # form meaningful correspondences even on non-convex surfaces (e.g. brain
    # pial) where the nearest reference vertex can be a false match on an
    # adjacent fold.  With the 50th percentile the kernel is too tight
    # (~1.5× mesh spacing), which causes the algorithm to lock onto wrong
    # correspondences early and converge to a local minimum.
    sigma = float(np.percentile(nn_dists, 75))
    sigma = max(sigma, sigma_min * 4)          # floor: at least two halvings

    dist_cutoff = float(np.percentile(nn_dists, 99)) * 1.5
    dist_cutoff = max(dist_cutoff, sigma * 3)  # always at least 3σ

    n_halvings = max(1, math.ceil(math.log2(sigma / sigma_min)))
    # Use (n_halvings + 1) slots so that after all halvings complete, at least
    # one full slot of iterations runs at sigma_min — the tightest resolution.
    # Without this, when sigma/sigma_min ≈ 2 the halving happens on the very
    # last iteration and no refinement at sigma_min occurs.
    period_sigma = max(1, max_iter // (n_halvings + 1))

    return {
        "sigma":        round(sigma,        4),
        "dist_cutoff":  round(dist_cutoff,  4),
        "period_sigma": period_sigma,
        "sigma_min":    round(sigma_min,    4),
        "beta":         round(beta,         4),
        "n_levels":     n_levels,
        "mesh_spacing": round(mesh_spacing, 4),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nonrigid_icp(
    mov_pts: np.ndarray,
    mov_normals: np.ndarray,
    ref_pts: np.ndarray,
    ref_normals: np.ndarray,
    adjacency: csr_matrix,
    *,
    init_def_field: np.ndarray | None = None,
    sigma: float = 3.0,
    beta: float = 0.5,
    dist_cutoff: float = 15.0,
    max_iter: int = 80,
    icm_iter: int = 50,
    period_sigma: int = 20,
    sigma_min: float = 0.1,
    outlier_weight: float = 0.1,
    normal_min_dot: float = 0.0,
    e_chunk: int = 2000,
    tgd_mov: np.ndarray | None = None,
    tgd_ref: np.ndarray | None = None,
    sigma_tgd: float = 0.2,
    symmetric: bool = True,
    use_rkhs: bool = True,
    rkhs_radius: float | None = None,
    rkhs_lambda: float = 0.01,
    verbose: bool = True,
) -> np.ndarray:
    """Register a moving surface onto a reference using non-rigid EM-ICP.

    The algorithm iterates between:
      - computing doubly-stochastic fuzzy correspondences (E-step), and
      - solving for the deformation field with Laplacian regularisation
        via preconditioned conjugate gradient (M-step).

    Parameters
    ----------
    mov_pts : ndarray (N, 3)
        Vertices of the moving surface.
    mov_normals : ndarray (N, 3)
        Unit normals at each moving vertex.
    ref_pts : ndarray (M, 3)
        Vertices of the reference surface.
    ref_normals : ndarray (M, 3)
        Unit normals at each reference vertex.
    adjacency : csr_matrix (N, N)
        Symmetric mesh adjacency matrix (from ``mesh.adjacency_csr``).
    init_def_field : ndarray (N, 3) or None
        Initial deformation field.  None initialises to zero (default).
        Used by ``nonrigid_icp_multires`` to warm-start the finest level
        from the interpolated coarse solution.
    sigma : float
        Initial bandwidth (std dev) of the Gaussian correspondence kernel
        ``exp(−d²/(2σ²))``.  Auto-estimated as the 75th percentile of NN
        distances if called via the CLI without ``--sigma``.
    beta : float
        Laplacian regularisation weight.  Higher → smoother deformation.
        Must be small relative to the expected per-vertex correspondence
        weight (typically 0.1–2.0); the default 0.5 gives a ~30 % data /
        ~70 % regularisation split at the finest scale.
    dist_cutoff : float
        Maximum search radius for candidate correspondences (typically 3–4 σ).
    max_iter : int
        Number of outer EM iterations.
    icm_iter : int
        Maximum number of conjugate gradient iterations per outer iteration.
    period_sigma : int
        Number of outer iterations between each halving of sigma.
    sigma_min : float
        Minimum value of sigma (annealing floor).
    outlier_weight : float
        Prior probability of a moving vertex being an outlier (0 = disabled).
        Outlier constant: ``c = w × M/N`` (density-normalised, σ-invariant).
        A vertex is effectively an outlier when its total correspondence weight
        ``row_sum < c``.  Reduces the influence of vertices with sparse/poor
        correspondences so they are dominated by the Laplacian prior.
    normal_min_dot : float
        Minimum dot product of moving and reference normals for a
        correspondence to be accepted.  0.0 = same hemisphere (default);
        increase toward 1.0 for stricter orientation filtering.
    e_chunk : int
        Number of vertices processed per KDTree query batch in the E-step.
        Has no effect on results; lower values reduce peak memory.
    tgd_mov : ndarray (N,) or None
        Normalised Total Geodesic Distance for each moving vertex,
        pre-computed via ``mesh.compute_tgd``.  If provided together with
        ``tgd_ref``, a TGD shape prior (Reg3, Combès & Prima 2019) is applied
        in the E-step: ``π_ij = exp(−|tgd_mov[i]−tgd_ref[j]|²/(2σ_tgd²))``
        is multiplied into each correspondence weight, penalising matches
        between vertices at different sulcal depths.
    tgd_ref : ndarray (M,) or None
        Normalised TGD for each reference vertex.
    sigma_tgd : float
        Bandwidth of the TGD prior kernel (both TGD arrays are in [0, 1]).
        Smaller values apply a tighter shape prior; default 0.2 rejects
        pairs whose normalised TGD differs by more than ~0.4 (2σ cutoff).
    use_rkhs : bool
        If True (default), replace the Laplacian M-step with an RKHS
        deformation model using the Wu C4 compactly-supported kernel
        (Combès & Prima 2019, Reg3).  The deformation is expressed as
        d = K α, where K_ij = wu(||x_i−x_j||/r) is a sparse kernel matrix.
        The M-step solves (diag(W)·K + λ·I)·α = W·target', then d = K·α.
        This gives a smoother, topology-independent regularisation than the
        Laplacian graph prior.
    rkhs_radius : float or None
        Compact support radius of the Wu kernel (in mesh units).
        None (default) → set by the caller (``nonrigid_icp_multires``)
        to ``mesh_spacing * 2``, giving ~10 neighbours/vertex at any
        resolution.  Fallback if called directly: ``sigma_min * 8``.
    rkhs_lambda : float
        RKHS regularisation weight.  Default 0.01; smaller values allow
        larger deformations but risk instability.
    symmetric : bool
        If True (default), use symmetric correspondences (Reg2 from Combès &
        Prima 2019): each edge weight is normalised by both the row sum
        (mov→ref) *and* the column sum (ref→mov).  The combined weight is
        ``ṽ_ij = w_ij/sR_i + w_ij/sC_j``.  This prevents many-to-one
        mappings where multiple moving vertices collapse onto the same
        reference vertex, which is especially harmful on non-convex surfaces
        such as brain pial where sulcal folds are topologically close in
        Euclidean space but anatomically distinct.
    verbose : bool
        Print iteration progress.

    Returns
    -------
    def_field : ndarray (N, 3)
        Per-vertex deformation field.
        The warped surface is ``mov_pts + def_field``.
    """
    mov_pts = np.asarray(mov_pts, dtype=float)
    mov_normals = np.asarray(mov_normals, dtype=float)
    ref_pts = np.asarray(ref_pts, dtype=float)
    ref_normals = np.asarray(ref_normals, dtype=float)

    N = len(mov_pts)
    M = len(ref_pts)

    use_tgd = (tgd_mov is not None) and (tgd_ref is not None)
    if use_tgd:
        tgd_mov = np.asarray(tgd_mov, dtype=float)
        tgd_ref = np.asarray(tgd_ref, dtype=float)
        inv_two_sigma_tgd2 = 1.0 / (2.0 * sigma_tgd * sigma_tgd)

    if init_def_field is not None:
        def_field = np.asarray(init_def_field, dtype=float).copy()
    else:
        def_field = np.zeros((N, 3), dtype=float)

    # Precompute adjacency statistics (fixed throughout, Laplacian mode)
    neigh_count = np.asarray(adjacency.sum(axis=1), dtype=float).ravel()  # (N,)

    # RKHS Wu kernel pre-computation (fixed on original mesh positions)
    K_mat: csr_matrix | None = None
    c_field: np.ndarray | None = None    # RKHS coefficient field (N, 3)
    if use_rkhs:
        r_k = rkhs_radius if rkhs_radius is not None else sigma_min * 8.0
        mov_tree_k = KDTree(mov_pts)
        # sparse_distance_matrix is C-backed: returns COO with all pairs
        # (i, j) whose Euclidean distance < r_k in one vectorised call.
        _coo = mov_tree_k.sparse_distance_matrix(
            mov_tree_k, r_k, output_type="coo_matrix"
        )
        _r_norm = _coo.data / r_k                               # in [0, 1)
        _kv     = (1.0 - _r_norm) ** 4 * (4.0 * _r_norm + 1.0)
        K_mat   = csr_matrix((_kv, (_coo.row, _coo.col)), shape=(N, N))
        c_field = np.zeros((N, 3), dtype=float)
        if verbose:
            nnz = K_mat.nnz
            print(f"  [RKHS] Wu kernel built: r={r_k:.2f}  nnz={nnz} ({nnz/N:.0f}/vertex)")

    ref_tree = KDTree(ref_pts)

    for it in range(max_iter):
        transformed = mov_pts + def_field  # (N, 3)

        # ------------------------------------------------------------
        # E-step: build COO arrays in a single chunked pass, then
        # compute doubly-stochastic barycentres without a N×M matrix.
        # ------------------------------------------------------------
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        wvals_parts: list[np.ndarray] = []

        inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

        for start in range(0, N, e_chunk):
            end = min(start + e_chunk, N)
            nbrs_chunk = ref_tree.query_ball_point(
                transformed[start:end], dist_cutoff,
                return_sorted=False, workers=-1,
            )
            for local_i, nbrs in enumerate(nbrs_chunk):
                if not nbrs:
                    continue
                i = start + local_i
                nbrs_arr = np.asarray(nbrs, dtype=np.int32)
                diffs    = ref_pts[nbrs_arr] - transformed[i]
                dists2   = np.einsum("ij,ij->i", diffs, diffs)   # squared distances
                valid    = (ref_normals[nbrs_arr] @ mov_normals[i]) >= normal_min_dot
                nbrs_v   = nbrs_arr[valid]
                if len(nbrs_v) == 0:
                    continue
                wv = np.exp(-dists2[valid] * inv_two_sigma2)
                if use_tgd:
                    tgd_diff2 = (tgd_mov[i] - tgd_ref[nbrs_v]) ** 2
                    wv *= np.exp(-tgd_diff2 * inv_two_sigma_tgd2)
                rows_parts.append(np.full(len(nbrs_v), i, dtype=np.int32))
                cols_parts.append(nbrs_v)
                wvals_parts.append(wv)

        if not rows_parts:
            if verbose:
                print(f"  iter {it:3d}: no correspondences — stopping early.")
            break

        rows  = np.concatenate(rows_parts)
        cols  = np.concatenate(cols_parts)
        wvals = np.concatenate(wvals_parts)
        del rows_parts, cols_parts, wvals_parts

        # Row sums (= total correspondence strength per moving vertex)
        row_sums = np.bincount(rows, weights=wvals, minlength=N)

        if symmetric:
            # Symmetric correspondences (Reg2 — Combès & Prima 2019):
            # Combined weight  ṽ_ij = w_ij/sR_i + w_ij/sC_j
            # normalises by both the row sum and the column sum, preventing
            # many-to-one mappings where multiple moving vertices compete for
            # the same reference vertex.
            col_sums = np.bincount(cols, weights=wvals, minlength=M)
            row_inv_per_edge = np.zeros(len(rows))
            col_inv_per_edge = np.zeros(len(rows))
            nz_r_mask = row_sums[rows] > 1e-10
            nz_c_mask = col_sums[cols] > 1e-10
            row_inv_per_edge[nz_r_mask] = 1.0 / row_sums[rows[nz_r_mask]]
            col_inv_per_edge[nz_c_mask] = 1.0 / col_sums[cols[nz_c_mask]]
            comb_w = wvals * (row_inv_per_edge + col_inv_per_edge)
            W_sums = np.bincount(rows, weights=comb_w, minlength=N)
            W_inv  = np.zeros(N)
            nz_W   = W_sums > 1e-10
            W_inv[nz_W] = 1.0 / W_sums[nz_W]
            corresBary = np.empty((N, 3), dtype=float)
            for k in range(3):
                corresBary[:, k] = np.bincount(
                    rows, weights=comb_w * ref_pts[cols, k], minlength=N
                ) * W_inv
            # Use W_sums for the outlier term (scaled to same range as row_sums
            # by multiplying back by an effective row_sum scale)
            eff_weight = W_sums
            inlier_mask = W_sums > 0.0
        else:
            eff_weight = row_sums
            # Barycentre normalised by raw row_sums.
            row_inv = np.zeros(N)
            nz_r = row_sums > 1e-10
            row_inv[nz_r] = 1.0 / row_sums[nz_r]
            corresBary = np.empty((N, 3), dtype=float)
            for k in range(3):
                corresBary[:, k] = np.bincount(
                    rows, weights=wvals * ref_pts[cols, k], minlength=N
                ) * row_inv
            inlier_mask = row_sums > 0.0

        del wvals, rows, cols

        corresBary[~inlier_mask] = 0.0   # unused vertices: barycentre irrelevant

        # Outlier term: down-weight vertices whose total correspondence strength
        # is below a density-normalised threshold.
        # We use  c = outlier_weight × M/N  instead of the full CPD formula
        # (which contains (2πσ²)^{3/2} and blows up when σ is large, killing
        # all correspondences in the coarse/early iterations).
        # Interpretation: a moving vertex is an "outlier" when its eff_weight is
        # below  outlier_weight × (M/N).  Because eff_weight scales with M,
        # this threshold is density-normalised and σ-invariant.
        if outlier_weight > 0.0:
            c_outlier = outlier_weight * M / N
            weight_out = eff_weight / (eff_weight + c_outlier)
        else:
            weight_out = eff_weight.copy()

        # ------------------------------------------------------------
        # M-step: preconditioned conjugate gradient
        # Two modes:
        #  RKHS (use_rkhs=True):  deformation d = K α, solves
        #    (diag(W) K + λ I) α = diag(W) target'   for each coord
        #    d = K α
        #  Laplacian (use_rkhs=False):  solves
        #    (diag(W + β|N_i|) - β A) d = diag(W) target'
        # ------------------------------------------------------------
        target_offset = corresBary - mov_pts  # (N, 3)

        if use_rkhs and K_mat is not None:
            # RKHS M-step: (W K + λ I) α = W target'
            WK = sp_diags(weight_out) @ K_mat             # (N, N) sparse
            lhs = WK + sp_diags(np.full(N, rkhs_lambda))  # + λ I
            prec_rkhs = sp_diags(1.0 / np.maximum(lhs.diagonal(), 1e-10))
            rhs = weight_out[:, np.newaxis] * target_offset

            for k in range(3):
                sol, _ = sp_cg(
                    lhs, rhs[:, k],
                    x0=c_field[:, k],
                    M=prec_rkhs,
                    rtol=1e-5,
                    maxiter=icm_iter,
                )
                if np.all(np.isfinite(sol)):
                    c_field[:, k] = sol

            d_new = K_mat @ c_field          # d = K α
            max_deform = dist_cutoff * 3.0
            if np.all(np.isfinite(d_new)) and float(np.max(np.abs(d_new))) < max_deform:
                # Laplacian smoothing on d_new: removes high-frequency folding
                # artefacts introduced by RKHS (which is topology-independent
                # and can invert triangles on complex surfaces).  2 passes at
                # α=0.3 damp one-vertex spikes without eroding large-scale fit.
                deg = np.asarray(adjacency.sum(axis=1), dtype=float).ravel()
                inv_deg = sp_diags(1.0 / np.maximum(deg, 1.0))
                A_norm = inv_deg @ adjacency
                d_smooth = d_new
                for _ in range(2):
                    d_smooth = 0.7 * d_smooth + 0.3 * (A_norm @ d_smooth)
                def_field[:] = d_smooth
            else:
                # Unstable update: reset c_field so next CG starts from 0
                # rather than the diverged state, preventing runaway growth.
                _bad = float(np.max(np.abs(d_new))) if np.all(np.isfinite(d_new)) else float("nan")
                if verbose:
                    print(f"  iter {it+1}: RKHS M-step rejected (max|d|={_bad:.1f}, limit={max_deform:.1f})")
                c_field[:] = 0.0
        else:
            # Laplacian M-step
            diag_vals = weight_out + beta * neigh_count          # (N,)
            M_mat  = sp_diags(diag_vals) - beta * adjacency     # (N, N) PSD
            M_prec = sp_diags(1.0 / np.maximum(diag_vals, 1e-10))
            rhs    = weight_out[:, np.newaxis] * target_offset  # (N, 3)

            for k in range(3):
                sol, _ = sp_cg(
                    M_mat, rhs[:, k],
                    x0=def_field[:, k],
                    M=M_prec,
                    rtol=1e-5,
                    maxiter=icm_iter,
                )
                if np.all(np.isfinite(sol)):
                    def_field[:, k] = sol

        # ------------------------------------------------------------
        # Annealing: halve sigma every period_sigma iterations
        # ------------------------------------------------------------
        if (it + 1) % period_sigma == 0:
            sigma = max(sigma / 2.0, sigma_min)
            # Reset RKHS coefficients: accumulated c_field from the previous
            # sigma level is a poor warm-start after a halving and causes CG
            # to diverge when the new rhs is much smaller in scale.
            if use_rkhs and c_field is not None:
                c_field[:] = 0.0

        if verbose:
            n_inliers = int(inlier_mask.sum())
            print(
                f"  iter {it + 1:3d}/{max_iter}"
                f"  σ={sigma:.3f}"
                f"  inliers={n_inliers}/{N}"
            )

    return def_field


# ---------------------------------------------------------------------------
# Multi-resolution helpers
# ---------------------------------------------------------------------------

def _interpolate_field(
    field_coarse: np.ndarray,
    pts_coarse: np.ndarray,
    pts_fine: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Inverse-distance weighted interpolation of a deformation field.

    For each vertex in *pts_fine*, locates the *k* nearest vertices in
    *pts_coarse* and computes a weighted average of their deformation
    vectors.  Weights are proportional to 1/distance, giving exact
    transfer when a fine vertex coincides with a coarse vertex.

    Parameters
    ----------
    field_coarse : ndarray (N_c, 3)
    pts_coarse   : ndarray (N_c, 3)
    pts_fine     : ndarray (N_f, 3)
    k            : int — number of neighbours (4 is typically sufficient)

    Returns
    -------
    ndarray (N_f, 3)
    """
    dists, idxs = KDTree(pts_coarse).query(pts_fine, k=k, workers=-1)
    w = 1.0 / np.maximum(dists, 1e-10)   # (N_f, k)
    w /= w.sum(axis=1, keepdims=True)
    # einsum: for each fine vertex sum  w[i,k] * field_coarse[idxs[i,k]]
    return np.einsum("nk,nkd->nd", w, field_coarse[idxs])


def _smooth_field(
    field: np.ndarray,
    adjacency: csr_matrix,
    n_iter: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """Apply n_iter steps of explicit Laplacian smoothing to a vector field.

    Each step: field ← (1-α)·field + α · (A_norm @ field)
    where A_norm is the row-normalised adjacency (each row sums to 1).

    This removes high-frequency cross-sulcus artefacts introduced by IDW
    interpolation in 3D without mesh-topology awareness.  A small number of
    iterations (3) with α=0.5 is sufficient to damp one-vertex spikes while
    preserving the large-scale deformation from the coarser level.

    Parameters
    ----------
    field      : ndarray (N, D)
    adjacency  : csr_matrix (N, N) — symmetric, unweighted
    n_iter     : int
    alpha      : float in (0, 1)
    """
    field = field.copy()
    deg = np.asarray(adjacency.sum(axis=1), dtype=float).ravel()
    deg = np.maximum(deg, 1.0)
    inv_deg = sp_diags(1.0 / deg)
    A_norm = inv_deg @ adjacency          # row-normalised
    for _ in range(n_iter):
        field = (1.0 - alpha) * field + alpha * (A_norm @ field)
    return field


def _build_level(
    pts: np.ndarray,
    faces: list,
    target_n: int,
) -> tuple[np.ndarray, np.ndarray, list, csr_matrix]:
    """Decimate *pts/faces* to *target_n* vertices and build adjacency.

    Returns
    -------
    (pts_l, normals_l, faces_l, adjacency_l)
    """
    from pyclarcs.mesh import decimate_surface, compute_vertex_normals, adjacency_csr
    d_pts, d_faces = decimate_surface(pts, faces, target_n)
    d_normals = compute_vertex_normals(d_pts, d_faces)
    d_adj = adjacency_csr(d_faces, len(d_pts))
    return d_pts, d_normals, d_faces, d_adj


# ---------------------------------------------------------------------------
# Multi-resolution registration
# ---------------------------------------------------------------------------

def nonrigid_icp_multires(
    mov_pts: np.ndarray,
    mov_normals: np.ndarray,
    ref_pts: np.ndarray,
    ref_normals: np.ndarray,
    mov_polygons: list,
    ref_polygons: list | None = None,
    *,
    n_levels: int = 3,
    target_n_coarsest: int = 2000,
    sigma: float | None = None,
    beta: float | None = None,
    beta_coarse_factor: float = 1.0,
    dist_cutoff: float | None = None,
    max_iter: int = 80,
    icm_iter: int = 50,
    period_sigma: int | None = None,
    sigma_min: float | None = None,
    outlier_weight: float = 0.1,
    normal_min_dot: float = 0.0,
    e_chunk: int = 2000,
    symmetric: bool = True,
    use_tgd: bool = True,
    tgd_n_seeds: int = 200,
    sigma_tgd: float = 0.5,
    use_rkhs: bool = True,
    rkhs_radius: float | None = None,
    rkhs_lambda: float = 0.01,
    verbose: bool = True,
) -> np.ndarray:
    """Multi-resolution non-rigid EM-ICP surface registration.

    Builds a hierarchy of *n_levels* progressively decimated copies of the
    moving surface (always decimated from the original, not recursively).
    Registration runs from the coarsest level to the finest:

      coarsest (target_n_coarsest pts)
          → run nonrigid_icp  (max_iter iterations)
          → interpolate deformation field to next finer mesh (IDW, k=4)
      …
      finest (original mesh, len(mov_pts) pts)
          → run nonrigid_icp  (max_iter // 2 iterations, warm-started)

    At each level the KDTree is always queried against the **full-resolution
    reference**, so the hierarchy only affects the moving surface.

    Sigma, dist_cutoff and period_sigma are re-estimated at each level from
    the current residual (transformed minus reference) unless explicit values
    are provided.  Because the residual decreases as the hierarchy progresses,
    the kernel automatically narrows at finer scales.

    Parameters
    ----------
    mov_pts : ndarray (N, 3)
    mov_normals : ndarray (N, 3)
    ref_pts : ndarray (M, 3)
    ref_normals : ndarray (M, 3)
    mov_polygons : list of face index lists
        Polygon connectivity of the moving surface (needed for decimation
        and adjacency construction at each level).
    n_levels : int
        Number of resolution levels including the finest.  With
        ``n_levels=1`` the method is identical to ``nonrigid_icp``.
    target_n_coarsest : int
        Target vertex count at the coarsest level.  Intermediate levels
        are placed geometrically between this value and ``len(mov_pts)``.
    sigma, dist_cutoff, period_sigma : float or None
        Override the auto-estimated values at every level.  None (default)
        triggers per-level re-estimation from the current residual.
    beta : float or None
        Regularisation weight.  None (default) → auto-estimated per level
        from the level's mesh spacing (``4 × mesh_spacing``).  Because
        coarser decimated meshes have larger spacing, this naturally yields
        a higher β at coarse levels without needing ``beta_coarse_factor``.
    beta_coarse_factor : float
        Multiplier applied to *explicit* beta values at each coarser level
        (finest idx=0, coarsest idx=L-1).  Ignored when beta is None.
    sigma_min : float or None
        Annealing floor.  None (default) → auto-estimated per level from
        the level's mesh spacing (``max(0.1, mesh_spacing / 2)``).
    max_iter : int
        Outer iterations at every level.  Coarse levels are cheap (few
        vertices); the finest level warm-starts from the interpolated
        coarse solution so it needs fewer corrections, but the full
        budget is kept so sigma annealing runs to completion.
    icm_iter : int
        Maximum CG iterations per outer iteration (same at all levels).
    e_chunk : int
        KDTree batch size (same at all levels).
    verbose : bool

    Returns
    -------
    def_field : ndarray (N, 3)
        Per-vertex deformation field at the finest (original) resolution.
        The warped surface is ``mov_pts + def_field``.
    """
    from pyclarcs.mesh import adjacency_csr

    mov_pts     = np.asarray(mov_pts,     dtype=float)
    mov_normals = np.asarray(mov_normals, dtype=float)
    ref_pts     = np.asarray(ref_pts,     dtype=float)
    ref_normals = np.asarray(ref_normals, dtype=float)

    N = len(mov_pts)

    # ------------------------------------------------------------------
    # Build hierarchy: level 0 = finest (original), level L-1 = coarsest
    # ------------------------------------------------------------------
    adj_finest = adjacency_csr(mov_polygons, N)
    # Each entry: (pts, normals, faces, adjacency)
    hierarchy = [(mov_pts, mov_normals, mov_polygons, adj_finest)]

    for lev in range(1, n_levels):
        # Geometrically-spaced target size in log space
        t = lev / (n_levels - 1) if n_levels > 1 else 1.0
        target_n = max(
            target_n_coarsest,
            int(N * (target_n_coarsest / N) ** t),
        )
        if target_n >= len(hierarchy[0][0]) * 0.85:
            if verbose:
                print(
                    f"  [multires] level {lev}: target {target_n} too close "
                    f"to finest ({N}), stopping hierarchy here."
                )
            break
        if verbose:
            print(f"  [multires] building level {lev}: {N} → ~{target_n} vertices…")
        hierarchy.append(_build_level(mov_pts, mov_polygons, target_n))

    n_actual = len(hierarchy)

    # Pre-compute sigma_min from the FINEST-level mesh so that all levels
    # share the same annealing floor.  Without this, coarser decimated meshes
    # produce larger mesh_spacing → larger sigma_min → the sigma_min * 4 floor
    # in estimate_registration_params forces the starting sigma much too high
    # at coarse levels (e.g. sigma_min_coarse * 4 >> initial gap).
    # By pinning sigma_min to the finest-level spacing throughout, coarse
    # levels keep a fine sigma_min (→ more halvings, gradual annealing) and
    # the warm-started fine level still benefits from sigma_min * 4 forcing
    # at least two halvings even when the warm-start residual is small.
    if sigma_min is None:
        _finest_sample = min(2000, N)
        _rng = np.random.default_rng(0)
        _idx = _rng.choice(N, size=_finest_sample, replace=False)
        _nn, _ = KDTree(mov_pts).query(mov_pts[_idx], k=2, workers=-1)
        sigma_min = max(0.1, float(np.median(_nn[:, 1])) / 2.0)

    # ------------------------------------------------------------------
    # Compute TGD once on finest mesh and reference, interpolate per level
    # ------------------------------------------------------------------
    tgd_mov_fine: np.ndarray | None = None
    tgd_ref_arr:  np.ndarray | None = None
    if use_tgd:
        from pyclarcs.mesh import compute_tgd
        if verbose:
            print("  [multires] computing TGD on moving surface…")
        tgd_mov_fine = compute_tgd(mov_pts, mov_polygons, n_seeds=tgd_n_seeds)
        if verbose:
            print("  [multires] computing TGD on reference surface…")
        if ref_polygons is not None:
            tgd_ref_arr = compute_tgd(ref_pts, ref_polygons, n_seeds=tgd_n_seeds)
        else:
            # No polygon connectivity supplied — approximate via KNN graph
            from scipy.spatial import KDTree as _KDTree
            from scipy.sparse import csr_matrix as _csr
            from scipy.sparse.csgraph import dijkstra as _dijkstra
            _ref_tree = _KDTree(ref_pts)
            _dists, _idxs = _ref_tree.query(ref_pts, k=9, workers=-1)
            M_ref = len(ref_pts)
            _rrows, _rcols, _rdata = [], [], []
            for _i in range(M_ref):
                for _k in range(1, 9):
                    _j = _idxs[_i, _k]
                    _rrows.append(_i); _rcols.append(_j)
                    _rdata.append(float(_dists[_i, _k]))
            _ref_graph = _csr((_rdata, (_rrows, _rcols)), shape=(M_ref, M_ref))
            _rng2 = np.random.default_rng(0)
            _ref_seeds = _rng2.choice(M_ref, size=min(tgd_n_seeds, M_ref), replace=False)
            _ref_dist = _dijkstra(_ref_graph, indices=_ref_seeds, directed=False)
            _fin_max = float(_ref_dist[np.isfinite(_ref_dist)].max()) \
                       if np.isfinite(_ref_dist).any() else 1.0
            _ref_dist = np.where(np.isinf(_ref_dist), _fin_max, _ref_dist)
            tgd_ref_arr = _ref_dist.sum(axis=0)
            _mx = tgd_ref_arr.max()
            if _mx > 0:
                tgd_ref_arr /= _mx
            tgd_ref_arr = tgd_ref_arr.astype(float)

    # ------------------------------------------------------------------
    # Register coarsest → finest
    # ------------------------------------------------------------------
    def_field_prev: np.ndarray | None = None   # result from coarser level
    pts_prev: np.ndarray | None       = None

    for idx in range(n_actual - 1, -1, -1):
        pts_l, normals_l, _, adj_l = hierarchy[idx]
        N_l = len(pts_l)
        is_finest = (idx == 0)

        # Warm-start from interpolated coarser field.
        # IDW interpolation in 3D can map a fine vertex to a coarse vertex on
        # the opposite wall of a narrow sulcus, introducing a cross-sulcus
        # deformation discontinuity that immediately folds the fine mesh.
        # A few steps of Laplacian smoothing on the fine mesh removes these
        # one-vertex spikes while preserving the large-scale warm-start.
        if def_field_prev is None:
            init_l = None
        else:
            init_l = _interpolate_field(def_field_prev, pts_prev, pts_l)
            init_l = _smooth_field(init_l, adj_l, n_iter=3, alpha=0.5)

        # All levels run max_iter outer iterations.  Coarse levels are
        # cheap (few vertices) and need to do the heavy lifting; the fine
        # level warm-starts from the interpolated coarse solution so it
        # only needs to refine, but still benefits from the full budget.
        max_iter_l = max_iter

        # β: if an explicit value was provided, scale geometrically toward
        # coarser levels; if None, auto-estimate per level (below).
        beta_l = (beta * (beta_coarse_factor ** idx)) if beta is not None else None

        # Auto-estimate params from current residual (and mesh spacing).
        transformed_l = pts_l if init_l is None else pts_l + init_l
        sigma_l    = sigma
        cutoff_l   = dist_cutoff
        period_l   = period_sigma
        sigma_min_l = sigma_min

        if sigma_l is None or cutoff_l is None or period_l is None \
                or beta_l is None or sigma_min_l is None:
            auto = estimate_registration_params(
                transformed_l, ref_pts,
                max_iter=max_iter_l,
                sigma_min=sigma_min,   # None → auto from mesh spacing
            )
            if sigma_l    is None: sigma_l    = auto["sigma"]
            if cutoff_l   is None: cutoff_l   = auto["dist_cutoff"]
            if period_l   is None: period_l   = auto["period_sigma"]
            if beta_l     is None: beta_l     = auto["beta"]
            if sigma_min_l is None: sigma_min_l = auto["sigma_min"]

        # TGD for this level: interpolate fine-level TGD to coarser mesh
        if use_tgd and tgd_mov_fine is not None:
            if idx == 0:
                tgd_mov_l = tgd_mov_fine
            else:
                tgd_mov_l = _interpolate_field(
                    tgd_mov_fine[:, np.newaxis], mov_pts, pts_l
                ).ravel()
        else:
            tgd_mov_l = None

        # Per-level RKHS radius: 2× local mesh spacing → ~10 neighbours/vertex.
        rkhs_radius_l = rkhs_radius
        if use_rkhs and rkhs_radius_l is None:
            rkhs_radius_l = auto["mesh_spacing"] * 2.0

        if verbose:
            label = "finest" if is_finest else f"level {idx}"
            tgd_tag = f"  TGD={'on' if tgd_mov_l is not None else 'off'}"
            rkhs_tag = f"  rkhs_r={rkhs_radius_l:.2f}" if use_rkhs else ""
            print(
                f"\n  [multires] {label}  {N_l} vertices"
                f"  {max_iter_l} outer iterations"
                f"  β={beta_l:.2f}  σ_min={sigma_min_l:.3f}"
                f"  σ={sigma_l:.3f}  r={cutoff_l:.2f}"
                f"  period_σ={period_l}{tgd_tag}{rkhs_tag}"
            )

        def_field_l = nonrigid_icp(
            pts_l, normals_l,
            ref_pts, ref_normals,
            adj_l,
            init_def_field=init_l,
            sigma=sigma_l,
            beta=beta_l,
            dist_cutoff=cutoff_l,
            max_iter=max_iter_l,
            icm_iter=icm_iter,
            period_sigma=period_l,
            sigma_min=sigma_min_l,
            outlier_weight=outlier_weight,
            normal_min_dot=normal_min_dot,
            e_chunk=e_chunk,
            tgd_mov=tgd_mov_l,
            tgd_ref=tgd_ref_arr,
            sigma_tgd=sigma_tgd,
            symmetric=symmetric,
            use_rkhs=use_rkhs,
            rkhs_radius=rkhs_radius_l,
            rkhs_lambda=rkhs_lambda,
            verbose=verbose,
        )

        def_field_prev = def_field_l
        pts_prev       = pts_l

    return def_field_prev  # finest-level result


def apply_deformation(
    points: np.ndarray,
    def_field: np.ndarray,
) -> np.ndarray:
    """Apply a deformation field to a point cloud.

    Parameters
    ----------
    points : ndarray (N, 3)
    def_field : ndarray (N, 3)

    Returns
    -------
    ndarray (N, 3) — warped coordinates  ``points + def_field``.
    """
    return np.asarray(points, dtype=float) + np.asarray(def_field, dtype=float)
