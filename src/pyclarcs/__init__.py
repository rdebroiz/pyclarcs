"""
pyclarcs - Automatic symmetry plane estimation for 3D surfaces.

Python rewrite of the ZZ_SYMC tool from the CLARCS project.
"""

from pyclarcs.symmetry import SymmetryPlane
from pyclarcs.io import (
    load_surface,
    load_surface_with_normals,
    save_surface,
    save_plane_vtk,
    save_deformation_vtk,
)
from pyclarcs.mesh import mesh_adjacency, adjacency_csr
from pyclarcs.nonrigid import register, apply_deformation
from pyclarcs.atlas import build_atlas

__all__ = [
    "SymmetryPlane",
    "load_surface",
    "load_surface_with_normals",
    "save_surface",
    "save_plane_vtk",
    "save_deformation_vtk",
    "mesh_adjacency",
    "adjacency_csr",
    "register",
    "apply_deformation",
    "build_atlas",
]
__version__ = "0.1.0"
