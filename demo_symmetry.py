#!/usr/bin/env python3
"""
demo_symmetry.py
================
Démonstration du calcul de plan de symétrie sur une surface cérébrale humaine.

Données
-------
Surface piale du cerveau MNI standard issue du projet "Brain for Blender"
(brainder.org, A. Winkler).  La surface piale est la surface externe du
cortex ; elle épouse directement l'endocrâne et constitue donc un excellent
proxy de surface endocranienne.

Licence des données : Creative Commons Attribution 4.0 International.

Algorithme
----------
Pipeline pyclarcs (port Python de ZZ_SYMC / CLARCS) :
  1. Initialisation par axes principaux (tenseur d'inertie)
  2. Optimisation grossière ICP + estimateur tronqué (multi-résolution)
  3. Raffinement fin EM-ICP avec recuit simulé  (σ : 5 → 0.5 mm)
  4. Raffinement final EM-ICP doublement stochastique  (σ = 0.25 mm)

Référence :
  "Automatic symmetry plane estimation of bilateral objects in point clouds",
  CVPR 2008.

Visualisation
-------------
  vedo (https://vedo.embl.es/)

Usage
-----
  python demo_symmetry.py                     # télécharge + calcule + affiche
  python demo_symmetry.py --no-download       # utilise le cache local
  python demo_symmetry.py --no-fine           # ICP grossier seulement (rapide)
  python demo_symmetry.py --quiet             # sans messages de progression
  python demo_symmetry.py --save plane.pl     # sauvegarde le plan estimé
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DATA_URL = (
    "https://s3.us-east-2.amazonaws.com/brainder/software/"
    "brain4blender/smallfiles/pial_Full_ply.tar.bz2"
)
CACHE_DIR = Path.home() / ".cache" / "pyclarcs"
ARCHIVE_CACHE = CACHE_DIR / "pial_Full_ply.tar.bz2"
LH_PLY = CACHE_DIR / "pial_Full_ply" / "lh.pial.ply"
RH_PLY = CACHE_DIR / "pial_Full_ply" / "rh.pial.ply"


# ---------------------------------------------------------------------------
# Téléchargement et extraction
# ---------------------------------------------------------------------------

def _progress_hook(block_count: int, block_size: int, total: int) -> None:
    """Affiche la progression du téléchargement sur une seule ligne."""
    downloaded = block_count * block_size
    if total > 0:
        pct = min(100, 100 * downloaded / total)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:5.1f}%  {downloaded/1e6:.1f}/{total/1e6:.1f} MB",
              end="", flush=True)


def download_surface(verbose: bool = True) -> None:
    """Télécharge et extrait les surfaces PLY si elles ne sont pas en cache.

    Les fichiers sont stockés dans ~/.cache/pyclarcs/ pour éviter de
    re-télécharger à chaque exécution.
    """
    if LH_PLY.exists() and RH_PLY.exists():
        if verbose:
            print(f"  Cache trouvé : {LH_PLY.parent}")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Téléchargement ---
    if not ARCHIVE_CACHE.exists():
        if verbose:
            print(f"Téléchargement de la surface piale MNI depuis brainder.org…")
            print(f"  Source : {DATA_URL}")
        urllib.request.urlretrieve(
            DATA_URL, ARCHIVE_CACHE,
            reporthook=_progress_hook if verbose else None,
        )
        if verbose:
            print()  # newline après la barre de progression

    # --- Extraction ---
    if verbose:
        print("Extraction de l'archive…")
    # bz2 peut ne pas être disponible dans Python : on utilise la commande système
    import subprocess
    result = subprocess.run(
        ["tar", "xjf", str(ARCHIVE_CACHE), "-C", str(CACHE_DIR)],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Échec de l'extraction : {result.stderr.decode()}\n"
            "Vérifiez que 'tar' et 'bzip2' sont installés sur votre système."
        )
    if verbose:
        print(f"  → {LH_PLY.parent}")


# ---------------------------------------------------------------------------
# Chargement PLY via vedo
# ---------------------------------------------------------------------------

def load_ply(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    """Charge un fichier PLY avec vedo et retourne (points, faces).

    Parameters
    ----------
    path : Path – chemin vers le fichier .ply

    Returns
    -------
    points : ndarray (N, 3)
    faces  : list of [i, j, k]  (triangles)
    """
    import vedo
    mesh = vedo.load(str(path))
    # In recent vedo versions, .points and .cells are properties, not methods
    pts = mesh.points
    if callable(pts):
        pts = pts()
    points = np.array(pts, dtype=float)
    cls = mesh.cells
    if callable(cls):
        cls = cls()
    faces = [list(c) for c in cls]
    return points, faces


# ---------------------------------------------------------------------------
# Pipeline pyclarcs
# ---------------------------------------------------------------------------

def compute_symmetry_plane(
    points: np.ndarray,
    do_coarse: bool = True,
    do_fine: bool = True,
    do_sym: bool = True,
    verbose: bool = True,
) -> "SymmetryPlane":
    """Lance les 4 étapes du pipeline pyclarcs sur le nuage de points.

    Parameters
    ----------
    points    : ndarray (N, 3) – surface complète (LH + RH fusionnés)
    do_coarse : bool – activer l'étape ICP grossière
    do_fine   : bool – activer l'EM-ICP avec recuit
    do_sym    : bool – activer le raffinement doublement stochastique
    verbose   : bool – afficher la progression

    Returns
    -------
    SymmetryPlane – plan de symétrie estimé
    """
    from pyclarcs.principal_axes import best_principal_axis_plane
    from pyclarcs.coarse import coarse_symmetry
    from pyclarcs.fine import em_icp_sym, em_icp_sym_corres

    t0 = time.time()

    # --- Étape 1 : initialisation par axes principaux ---
    if verbose:
        print("\n[1/4] Initialisation par axes principaux…")
    plane = best_principal_axis_plane(points)
    if verbose:
        print(f"      {plane}  ({time.time()-t0:.1f}s)")

    # --- Étape 2 : ICP grossier multi-résolution ---
    if do_coarse:
        if verbose:
            print("[2/4] Optimisation grossière ICP…")
        t1 = time.time()
        plane = coarse_symmetry(points, plane, verbose=False)
        if verbose:
            print(f"      {plane}  ({time.time()-t1:.1f}s)")
    else:
        if verbose:
            print("[2/4] ICP grossier ignoré (--no-coarse)")

    # --- Étape 3 : EM-ICP avec recuit simulé ---
    if do_fine:
        if verbose:
            print("[3/4] Raffinement fin EM-ICP (recuit σ : 5→0.5 mm)…")
        t2 = time.time()
        plane = em_icp_sym(points, plane, sigma_init=5.0, sigma_final=0.5,
                           verbose=False)
        if verbose:
            print(f"      {plane}  ({time.time()-t2:.1f}s)")
    else:
        if verbose:
            print("[3/4] EM-ICP ignoré (--no-fine)")

    # --- Étape 4 : EM-ICP doublement stochastique ---
    if do_sym:
        if verbose:
            print("[4/4] Raffinement doublement stochastique (σ = 0.25 mm)…")
        t3 = time.time()
        plane = em_icp_sym_corres(points, plane, sigma=0.25, verbose=False)
        if verbose:
            print(f"      {plane}  ({time.time()-t3:.1f}s)")
    else:
        if verbose:
            print("[4/4] Raffinement doublement stochastique ignoré (--no-sym)")

    if verbose:
        print(f"\nTemps total : {time.time()-t0:.1f}s")

    return plane


# ---------------------------------------------------------------------------
# Visualisation vedo
# ---------------------------------------------------------------------------

def visualize(
    lh_pts: np.ndarray,
    lh_faces: list[list[int]],
    rh_pts: np.ndarray,
    rh_faces: list[list[int]],
    plane: "SymmetryPlane",
) -> None:
    """Affiche la surface cérébrale et le plan de symétrie dans une fenêtre 3D.

    Représentation :
    - Hémisphère gauche  : couleur chaude (saumon)
    - Hémisphère droit   : couleur froide (bleu acier)
    - Points sur le plan : blanc (distance signée ≈ 0)
    - Plan de symétrie   : rectangle semi-transparent cyan
    - Vecteur normal     : flèche verte

    Contrôles de la caméra :
        clic + glisser  → rotation
        molette         → zoom
        clic droit      → déplacement
        q               → quitter
    """
    import vedo

    all_pts = np.vstack([lh_pts, rh_pts])
    bounds = (
        float(all_pts[:, 0].min()), float(all_pts[:, 0].max()),
        float(all_pts[:, 1].min()), float(all_pts[:, 1].max()),
        float(all_pts[:, 2].min()), float(all_pts[:, 2].max()),
    )

    # --- Calcul de la distance signée pour la colorisation ---
    lh_dist = plane.signed_distance(lh_pts)   # > 0 : même côté que n
    rh_dist = plane.signed_distance(rh_pts)

    # Plage de couleur symétrique autour de 0
    d_max = max(abs(lh_dist).max(), abs(rh_dist).max()) * 0.8

    # --- Maillages vedo ---
    # Hémisphère gauche : dégradé rouge-blanc-bleu selon distance au plan
    lh_mesh = vedo.Mesh([lh_pts.tolist(), lh_faces])
    lh_mesh.cmap("RdBu_r", lh_dist, vmin=-d_max, vmax=d_max)
    lh_mesh.alpha(0.92)
    lh_mesh.name = "Hémisphère gauche"

    # Hémisphère droit : même colorisation
    rh_mesh = vedo.Mesh([rh_pts.tolist(), rh_faces])
    rh_mesh.cmap("RdBu_r", rh_dist, vmin=-d_max, vmax=d_max)
    rh_mesh.alpha(0.92)
    rh_mesh.name = "Hémisphère droit"

    # --- Plan de symétrie : rectangle semi-transparent ---
    n = plane.n
    centre = plane.project(all_pts.mean(axis=0)).ravel()

    # Demi-taille du rectangle = diagonale de la boite englobante
    diag = np.linalg.norm([bounds[1]-bounds[0],
                           bounds[3]-bounds[2],
                           bounds[5]-bounds[4]]) * 0.55

    # Deux vecteurs orthonormaux dans le plan
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref);  u /= np.linalg.norm(u)
    v = np.cross(n, u);    v /= np.linalg.norm(v)

    corners = [
        (centre + diag*u + diag*v).tolist(),
        (centre - diag*u + diag*v).tolist(),
        (centre - diag*u - diag*v).tolist(),
        (centre + diag*u - diag*v).tolist(),
    ]
    sym_plane_mesh = vedo.Mesh([corners, [[0, 1, 2], [0, 2, 3]]])
    sym_plane_mesh.c("cyan").alpha(0.25)
    sym_plane_mesh.name = "Plan de symétrie"

    # Contour du plan (pour le rendre visible)
    sym_plane_outline = vedo.Line(
        corners + [corners[0]], lw=3, c="deepskyblue"
    )

    # --- Vecteur normal (flèche verte) ---
    arrow_start = centre.tolist()
    arrow_end = (centre + n * diag * 0.35).tolist()
    normal_arrow = vedo.Arrow(arrow_start, arrow_end, s=0.015, c="limegreen")
    normal_arrow.name = "Normale au plan"

    # --- Barre de couleur et légende ---
    cbar = lh_mesh.add_scalarbar(
        title="Distance au plan (mm)",
        c="white",
    )

    # --- Texte informatif ---
    info = (
        f"Plan de symétrie estimé\n"
        f"n = [{plane.n[0]:+.4f}, {plane.n[1]:+.4f}, {plane.n[2]:+.4f}]\n"
        f"d = {plane.d:.4f} mm\n"
        f"\n"
        f"Rouge  = côté droit du plan\n"
        f"Bleu   = côté gauche du plan\n"
        f"Blanc  = sur le plan"
    )
    text_box = vedo.Text2D(
        info,
        pos="bottom-left",
        font="Calco",
        s=0.72,
        c="white",
        bg="k3",
        alpha=0.7,
    )

    title = vedo.Text2D(
        "Calcul du plan de symétrie – Surface piale MNI (Brain for Blender)",
        pos="top-center",
        font="Calco",
        s=0.85,
        c="white",
    )

    # --- Fenêtre vedo ---
    plt = vedo.Plotter(
        title="pyclarcs — Plan de symétrie",
        size=(1280, 800),
        bg="k1",           # fond gris très sombre
        bg2="midnightblue" # dégradé en haut
    )

    plt.show(
        lh_mesh,
        rh_mesh,
        sym_plane_mesh,
        sym_plane_outline,
        normal_arrow,
        text_box,
        title,
        viewup="z",     # axe Z vers le haut (convention neuroimagerie)
        zoom=1.1,
        interactive=True,
    )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Démo pyclarcs : plan de symétrie sur surface cérébrale MNI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Ne pas télécharger les données (utilise le cache existant).",
    )
    parser.add_argument(
        "--no-coarse", action="store_true",
        help="Ignorer l'étape ICP grossière.",
    )
    parser.add_argument(
        "--no-fine", action="store_true",
        help="Ignorer les étapes EM-ICP fines (beaucoup plus rapide).",
    )
    parser.add_argument(
        "--no-sym", action="store_true",
        help="Ignorer le raffinement doublement stochastique.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Supprimer les messages de progression.",
    )
    parser.add_argument(
        "--save", metavar="FILE",
        help="Sauvegarder le plan estimé dans un fichier .pl.",
    )
    args = parser.parse_args(argv)

    verbose = not args.quiet

    # --- Téléchargement ---
    if not args.no_download:
        download_surface(verbose=verbose)
    else:
        if not (LH_PLY.exists() and RH_PLY.exists()):
            print(
                f"ERREUR : fichiers PLY introuvables dans {CACHE_DIR}.\n"
                "Relancez sans --no-download pour les télécharger.",
                file=sys.stderr,
            )
            return 1

    # --- Chargement ---
    if verbose:
        print("\nChargement des surfaces PLY…")
    lh_pts, lh_faces = load_ply(LH_PLY)
    rh_pts, rh_faces = load_ply(RH_PLY)
    if verbose:
        print(f"  LH : {len(lh_pts):,} sommets, {len(lh_faces):,} faces")
        print(f"  RH : {len(rh_pts):,} sommets, {len(rh_faces):,} faces")

    # On fusionne les deux hémisphères en un seul nuage de points
    # pour le calcul du plan (la connectivité n'est pas nécessaire ici)
    all_pts = np.vstack([lh_pts, rh_pts])

    # --- Calcul ---
    plane = compute_symmetry_plane(
        all_pts,
        do_coarse=not args.no_coarse,
        do_fine=not args.no_fine,
        do_sym=not args.no_sym,
        verbose=verbose,
    )

    # --- Sauvegarde optionnelle ---
    if args.save:
        plane.save(args.save)
        if verbose:
            print(f"\nPlan sauvegardé : {args.save}")

    # --- Visualisation ---
    if verbose:
        print("\nOuverture de la fenêtre de visualisation…")
        print("  (clic + glisser = rotation, molette = zoom, q = quitter)")

    visualize(lh_pts, lh_faces, rh_pts, rh_faces, plane)
    return 0


if __name__ == "__main__":
    sys.exit(main())
