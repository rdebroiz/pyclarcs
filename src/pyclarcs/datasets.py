"""Synthetic data generation and reference dataset download commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click


_verbose_option = click.option(
    "-q", "--quiet", "quiet",
    is_flag=True, default=False,
    help="Suppress all output.",
)


# ---------------------------------------------------------------------------
# generate-synth-data helpers
# ---------------------------------------------------------------------------

def _make_ellipsoid(
    a: float,
    b: float,
    c: float,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_lat: int = 100,
    n_lon: int = 180,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> "tuple[np.ndarray, list[list[int]]]":
    import numpy as np

    rng = np.random.default_rng(seed)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

    pts: list = []
    for la in lat:
        for lo in lon:
            pts.append([
                a * np.cos(la) * np.cos(lo),
                b * np.cos(la) * np.sin(lo),
                c * np.sin(la),
            ])

    points = np.array(pts, dtype=float) + np.array(centre)
    if noise_sigma > 0:
        points += rng.standard_normal(points.shape) * noise_sigma

    faces: list[list[int]] = []
    total_rings = n_lat + 2
    for ring in range(total_rings - 1):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            v00 = ring * n_lon + j
            v01 = ring * n_lon + j2
            v10 = (ring + 1) * n_lon + j
            v11 = (ring + 1) * n_lon + j2
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return points, faces


def _synth_affine_perturb(points: "np.ndarray") -> "np.ndarray":
    import numpy as np

    axis = np.array([0.3, 0.8, 0.5])
    axis /= np.linalg.norm(axis)
    theta = np.radians(12.0)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    centroid = points.mean(axis=0)
    return 1.03 * ((points - centroid) @ R.T) + centroid + np.array([10.0, 7.0, -5.0])


def _synth_smooth_deformation(points: "np.ndarray") -> "np.ndarray":
    import numpy as np

    rng = np.random.default_rng(99)
    N = len(points)
    radius = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) * 0.15
    field = np.zeros((N, 3), dtype=float)
    ctrl_pts = points[rng.choice(N, size=8, replace=False)]
    for k in range(8):
        disp = rng.standard_normal(3) * 5.0
        weights = np.exp(-(np.linalg.norm(points - ctrl_pts[k], axis=1) / radius) ** 2)
        field += weights[:, np.newaxis] * disp
    return field


# ---------------------------------------------------------------------------
# generate-synth-data command
# ---------------------------------------------------------------------------

@click.command("generate-synth-data")
@click.argument("output_dir", metavar="OUTPUT_DIR", default=".", required=False)
@click.option("--no-reg", is_flag=True,
              help="Skip registration test pairs (*_target.vtk).")
@_verbose_option
def generate_synth_data(output_dir, no_reg, quiet):
    """Generate synthetic test surfaces in OUTPUT_DIR.

    \b
    Surfaces produced:
      ellipsoid_skull_noisy.vtk    skull-shaped (a=85 b=65 c=70 mm), true plane x=85
      ellipsoid_a15_b60_c45.vtk   unit-test ellipsoid (a=15 b=60 c=45 mm), true plane x=5
      ellipsoid_oblate.vtk         oblate (a=40 b=90 c=90 mm), true plane x=40

    Unless --no-reg, a matching *_target.vtk registration pair is created for
    each available reference surface (including MNI surfaces if already present).
    """
    verbose = not quiet
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from pyclarcs.io import load_surface, save_surface

    specs = [
        ("ellipsoid_skull_noisy.vtk",  85.0, 65.0, 70.0, (85.0, 0.0, 0.0), 0.5, 0,
         "skull-shaped (a=85 b=65 c=70 mm), true plane x=85"),
        ("ellipsoid_a15_b60_c45.vtk",  15.0, 60.0, 45.0, ( 5.0, 0.0, 0.0), 0.3, 1,
         "unit-test (a=15 b=60 c=45 mm), true plane x=5"),
        ("ellipsoid_oblate.vtk",        40.0, 90.0, 90.0, (40.0, 0.0, 0.0), 0.4, 2,
         "oblate (a=40 b=90 c=90 mm), true plane x=40"),
    ]

    if verbose:
        click.echo("Generating synthetic surfaces…")
    for fname, a, b, c, centre, sigma, seed, desc in specs:
        pts, faces = _make_ellipsoid(a, b, c, centre=centre, noise_sigma=sigma, seed=seed)
        out = out_dir / fname
        save_surface(str(out), pts, faces)
        if verbose:
            click.echo(f"  → {out.name}  ({len(pts):,} pts)  [{desc}]")

    if no_reg:
        if verbose:
            click.echo("Skipping registration pairs (--no-reg).")
    else:
        refs = [
            "ellipsoid_skull_noisy.vtk",
            "endocranium_mni_pial_10k.vtk",
            "endocranium_mni_pial.vtk",
        ]
        if verbose:
            click.echo("\nGenerating registration test pairs…")
        for ref_fname in refs:
            ref_path = out_dir / ref_fname
            if not ref_path.exists():
                if verbose:
                    click.echo(f"  SKIP {ref_fname} (not found)")
                continue
            pts, faces = load_surface(str(ref_path))
            perturbed = _synth_affine_perturb(pts)
            target_pts = perturbed + _synth_smooth_deformation(perturbed)
            out = out_dir / f"{ref_path.stem}_target.vtk"
            save_surface(str(out), target_pts, faces)
            if verbose:
                click.echo(
                    f"  {ref_fname}  →  {out.name}"
                    f"  ({len(target_pts):,} pts)"
                    f"  [t=(10,7,-5) mm | R=12° | s=1.03 | 8 bumps ×5 mm]"
                )

    if verbose:
        click.echo("\nDone.")


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------

def _decimate_surface_inplace(path: Path, target_n: int, verbose: bool) -> None:
    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.mesh import decimate_surface

    pts, faces = load_surface(str(path))
    n_orig = len(pts)
    if n_orig <= target_n:
        if verbose:
            click.echo(f"  {path.name}: {n_orig} ≤ {target_n} vertices, skipped.")
        return
    pts_d, faces_d = decimate_surface(pts, faces, target_n)
    save_surface(str(path), pts_d, faces_d)
    if verbose:
        click.echo(f"  {path.name}: {n_orig} → {len(pts_d)} vertices.")


_MNI152_URLS = {
    "L": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_hemi-L_desc-pial_surf.surf.gii"
    ),
    "R": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_hemi-R_desc-pial_surf.surf.gii"
    ),
}


def _load_gifti(path: Path) -> "tuple[np.ndarray, list[list[int]]]":
    try:
        import nibabel as nib
    except ImportError:
        raise click.ClickException(
            "nibabel is required to read MNI152 GIfTI surfaces:\n"
            "  pip install nibabel"
        )
    import numpy as np

    img = nib.load(str(path))
    pts  = img.darrays[0].data.astype(float)
    tris = img.darrays[1].data.astype(int)
    return pts, tris.tolist()


# ---------------------------------------------------------------------------
# download group
# ---------------------------------------------------------------------------

@click.group("download")
def download():
    """Download reference datasets."""


@download.command("mni")
@click.argument("output_dir", metavar="OUTPUT", default=".", required=False)
@click.option("--target-n", default=10_000, show_default=True, type=int, metavar="N",
              help="Decimate the merged surface to ~N vertices. Pass 0 to skip.")
@click.option("--atlas", "with_atlas", is_flag=True,
              help="Also download the MNI152NLin2009cAsym pial atlas (TemplateFlow). "
                   "Requires nibabel (pip install nibabel).")
@click.option("--force", is_flag=True,
              help="Re-download even if files are already cached.")
@_verbose_option
def download_mni(output_dir, target_n, with_atlas, force, quiet):
    """Download MNI pial surfaces to OUTPUT.

    \b
    Without --atlas (Brain for Blender, A. Winkler — CC BY 4.0):
      endocranium_mni_pial.vtk       full-resolution LH+RH merged surface
      endocranium_mni_pial_10k.vtk   decimated to ~10 000 vertices

    \b
    With --atlas (MNI152NLin2009cAsym pial, TemplateFlow — CC0):
      mni152_pial_L.vtk   left hemisphere pial surface
      mni152_pial_R.vtk   right hemisphere pial surface
      mni152_pial.vtk     merged LH+RH atlas surface
    """
    import subprocess
    import urllib.request
    import numpy as np

    verbose = not quiet
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _MNI_URL = (
        "https://s3.us-east-2.amazonaws.com/brainder/software/"
        "brain4blender/smallfiles/pial_Full_ply.tar.bz2"
    )
    _cache_dir = Path.home() / ".cache" / "pyclarcs"
    _archive   = _cache_dir / "pial_Full_ply.tar.bz2"
    _cache     = _cache_dir / "pial_Full_ply"
    lh_ply     = _cache / "lh.pial.ply"
    rh_ply     = _cache / "rh.pial.ply"

    def _reporthook(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            click.echo(f"\r  {pct:3d}%", nl=False)

    if lh_ply.exists() and rh_ply.exists() and not force:
        if verbose:
            click.echo(f"Cache found: {_cache}")
    else:
        _cache_dir.mkdir(parents=True, exist_ok=True)
        if not _archive.exists() or force:
            if verbose:
                click.echo(f"Downloading MNI pial surface from brainder.org…\n  {_MNI_URL}")
            urllib.request.urlretrieve(
                _MNI_URL, _archive,
                reporthook=_reporthook if verbose else None,
            )
            if verbose:
                click.echo("")
        if verbose:
            click.echo("Extracting archive…")
        result = subprocess.run(
            ["tar", "xjf", str(_archive), "-C", str(_cache_dir)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise click.ClickException(
                f"Extraction failed: {result.stderr.decode()}\n"
                "Make sure 'tar' and 'bzip2' are installed."
            )
        if verbose:
            click.echo(f"  → {_cache}")

    from pyclarcs.io import load_surface, save_surface

    if verbose:
        click.echo("Loading LH + RH pial surfaces…")
    lh_pts, lh_faces = load_surface(str(lh_ply))
    rh_pts, rh_faces = load_surface(str(rh_ply))

    offset = len(lh_pts)
    rh_faces_shifted = [[v + offset for v in f] for f in rh_faces]
    all_pts   = np.vstack([lh_pts, rh_pts])
    all_faces = lh_faces + rh_faces_shifted

    out = out_dir / "endocranium_mni_pial.vtk"
    save_surface(str(out), all_pts, all_faces)
    if verbose:
        click.echo(f"  → {out.name}  ({len(all_pts):,} pts, {len(all_faces):,} faces)")

    if target_n:
        from pyclarcs.mesh import decimate_surface
        n = target_n
        suffix = f"{n // 1000}k" if n % 1000 == 0 else str(n)
        if verbose:
            click.echo(f"Decimating to ~{n} vertices…")
        dec_pts, dec_faces = decimate_surface(all_pts, all_faces, n)
        out2 = out_dir / f"endocranium_mni_pial_{suffix}.vtk"
        save_surface(str(out2), dec_pts, dec_faces)
        if verbose:
            click.echo(f"  → {out2.name}  ({len(dec_pts):,} pts, {len(dec_faces):,} faces)")

    if with_atlas:
        _atlas_cache = _cache_dir / "mni152"
        _atlas_cache.mkdir(parents=True, exist_ok=True)

        hemi_pts:   dict[str, np.ndarray]     = {}
        hemi_faces: dict[str, list[list[int]]] = {}

        for hemi, url in _MNI152_URLS.items():
            dest = _atlas_cache / f"mni152_pial_{hemi}.surf.gii"
            if not dest.exists() or force:
                if verbose:
                    click.echo(f"Downloading MNI152 {hemi} pial from TemplateFlow…\n  {url}")
                urllib.request.urlretrieve(
                    url, dest,
                    reporthook=_reporthook if verbose else None,
                )
                if verbose:
                    click.echo("")
            elif verbose:
                click.echo(f"Cache found: {dest.name}")

            pts, faces = _load_gifti(dest)
            hemi_pts[hemi]   = pts
            hemi_faces[hemi] = faces

            out_h = out_dir / f"mni152_pial_{hemi}.vtk"
            save_surface(str(out_h), pts, faces)
            if verbose:
                click.echo(f"  → {out_h.name}  ({len(pts):,} pts, {len(faces):,} faces)")

        offset_atlas = len(hemi_pts["L"])
        r_shifted = [[v + offset_atlas for v in f] for f in hemi_faces["R"]]
        atlas_pts   = np.vstack([hemi_pts["L"], hemi_pts["R"]])
        atlas_faces = hemi_faces["L"] + r_shifted
        out_atlas = out_dir / "mni152_pial.vtk"
        save_surface(str(out_atlas), atlas_pts, atlas_faces)
        if verbose:
            click.echo(
                f"  → {out_atlas.name}  "
                f"({len(atlas_pts):,} pts, {len(atlas_faces):,} faces)"
            )

    click.echo(f"Done. Files ready in {out_dir}.")


@download.command("paleobrain")
@click.argument("output_dir", metavar="OUTPUT", default="paleobrain", required=False)
@click.option("--n", default=75, show_default=True, type=click.IntRange(1, 75), metavar="N",
              help="Number of subjects to download (1–75).")
@click.option("--type", "kind",
              type=click.Choice(["brain", "endocast", "all"]), default="all",
              show_default=True,
              help="Which surfaces to download.")
@click.option("--target-n", default=None, type=int, metavar="N",
              help="After download, decimate each surface to ~N vertices.")
@click.option("--force", is_flag=True,
              help="Re-download files that already exist.")
@_verbose_option
def download_paleobrain(output_dir, n, kind, target_n, force, quiet):
    """Download the PaleoBRAIN surface dataset to OUTPUT.

    \b
    75 brain surfaces (B01–B75.ply) and 75 endocasts (E01–E75.ply).
    Reference: Balzeau A. (2025). doi:10.48579/PRO/KZMMLM

    \b
    Examples:
      clarcs download paleobrain data/paleobrain
      clarcs download paleobrain data/paleobrain --n 10 --type brain
      clarcs download paleobrain data/paleobrain --n 10 --target-n 5000
    """
    import urllib.request
    import urllib.error

    verbose = not quiet

    _BASE_URL = "https://data.indores.fr/api/access/datafile/"
    _BRAIN_IDS: dict[str, int] = {
        "B01.ply": 25532, "B02.ply": 25611, "B03.ply": 25636, "B04.ply": 25646,
        "B05.ply": 25648, "B06.ply": 25607, "B07.ply": 25608, "B08.ply": 25580,
        "B09.ply": 25618, "B10.ply": 25619, "B11.ply": 25519, "B12.ply": 25556,
        "B13.ply": 25585, "B14.ply": 25574, "B15.ply": 25510, "B16.ply": 25605,
        "B17.ply": 25538, "B18.ply": 25591, "B19.ply": 25517, "B20.ply": 25638,
        "B21.ply": 25520, "B22.ply": 25613, "B23.ply": 25554, "B24.ply": 25612,
        "B25.ply": 25551, "B26.ply": 25557, "B27.ply": 25634, "B28.ply": 25629,
        "B29.ply": 25600, "B30.ply": 25576, "B31.ply": 25547, "B32.ply": 25560,
        "B33.ply": 25620, "B34.ply": 25578, "B35.ply": 25575, "B36.ply": 25540,
        "B37.ply": 25569, "B38.ply": 25584, "B39.ply": 25609, "B40.ply": 25573,
        "B41.ply": 25524, "B42.ply": 25643, "B43.ply": 25653, "B44.ply": 25587,
        "B45.ply": 25606, "B46.ply": 25604, "B47.ply": 25637, "B48.ply": 25601,
        "B49.ply": 25610, "B50.ply": 25539, "B51.ply": 25533, "B52.ply": 25603,
        "B53.ply": 25602, "B54.ply": 25617, "B55.ply": 25571, "B56.ply": 25526,
        "B57.ply": 25635, "B58.ply": 25566, "B59.ply": 25647, "B60.ply": 25625,
        "B61.ply": 25531, "B62.ply": 25509, "B63.ply": 25595, "B64.ply": 25577,
        "B65.ply": 25657, "B66.ply": 25568, "B67.ply": 25514, "B68.ply": 25645,
        "B69.ply": 25616, "B70.ply": 25511, "B71.ply": 25642, "B72.ply": 25624,
        "B73.ply": 25630, "B74.ply": 25552, "B75.ply": 25558,
    }
    _ENDOCAST_IDS: dict[str, int] = {
        "E01.ply": 25581, "E02.ply": 25599, "E03.ply": 25621, "E04.ply": 25516,
        "E05.ply": 25628, "E06.ply": 25632, "E07.ply": 25543, "E08.ply": 25582,
        "E09.ply": 25541, "E10.ply": 25650, "E11.ply": 25555, "E12.ply": 25631,
        "E13.ply": 25596, "E14.ply": 25523, "E15.ply": 25570, "E16.ply": 25525,
        "E17.ply": 25655, "E18.ply": 25542, "E19.ply": 25588, "E20.ply": 25652,
        "E21.ply": 25534, "E22.ply": 25594, "E23.ply": 25586, "E24.ply": 25537,
        "E25.ply": 25598, "E26.ply": 25615, "E27.ply": 25590, "E28.ply": 25513,
        "E29.ply": 25545, "E30.ply": 25656, "E31.ply": 25521, "E32.ply": 25527,
        "E33.ply": 25658, "E34.ply": 25579, "E35.ply": 25639, "E36.ply": 25614,
        "E37.ply": 25640, "E38.ply": 25654, "E39.ply": 25593, "E40.ply": 25512,
        "E41.ply": 25562, "E42.ply": 25651, "E43.ply": 25553, "E44.ply": 25589,
        "E45.ply": 25535, "E46.ply": 25518, "E47.ply": 25563, "E48.ply": 25529,
        "E49.ply": 25550, "E50.ply": 25561, "E51.ply": 25544, "E52.ply": 25641,
        "E53.ply": 25508, "E54.ply": 25564, "E55.ply": 25528, "E56.ply": 25592,
        "E57.ply": 25623, "E58.ply": 25548, "E59.ply": 25597, "E60.ply": 25626,
        "E61.ply": 25567, "E62.ply": 25522, "E63.ply": 25549, "E64.ply": 25644,
        "E65.ply": 25546, "E66.ply": 25559, "E67.ply": 25530, "E68.ply": 25572,
        "E69.ply": 25649, "E70.ply": 25622, "E71.ply": 25565, "E72.ply": 25633,
        "E73.ply": 25536, "E74.ply": 25583, "E75.ply": 25515,
    }

    entries: list[tuple[str, int]] = []
    if kind in ("brain", "all"):
        entries += [(k, _BRAIN_IDS[k]) for k in sorted(_BRAIN_IDS)[:n]]
    if kind in ("endocast", "all"):
        entries += [(k, _ENDOCAST_IDS[k]) for k in sorted(_ENDOCAST_IDS)[:n]]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo(f"PaleoBRAIN — {len(entries)} file(s) → {out_dir}")
        click.echo("doi:10.48579/PRO/KZMMLM")
        if target_n:
            click.echo(f"Resampling to ~{target_n} vertices after download.")

    def _reporthook(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            click.echo(f"\r  {pct:3d}%", nl=False)

    errors: list[str] = []
    for i, (name, file_id) in enumerate(entries, 1):
        dest = out_dir / name
        prefix = f"[{i}/{len(entries)}] {name}"

        if dest.exists() and not force:
            if verbose:
                click.echo(f"{prefix}  (exists, skipped)")
            if target_n:
                _decimate_surface_inplace(dest, target_n, verbose)
            continue

        if verbose:
            click.echo(f"{prefix}  downloading…")
        tmp = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(
                f"{_BASE_URL}{file_id}", tmp,
                reporthook=_reporthook if verbose else None,
            )
            if verbose:
                click.echo("")
            tmp.rename(dest)
        except urllib.error.URLError as exc:
            tmp.unlink(missing_ok=True)
            click.echo(f"  ERROR: {exc}", err=True)
            errors.append(name)
            continue

        if target_n:
            _decimate_surface_inplace(dest, target_n, verbose)

    click.echo(
        f"\nDone. {len(entries) - len(errors)}/{len(entries)} file(s) ready in {out_dir}."
    )
    if errors:
        click.echo(f"Failed: {', '.join(errors)}", err=True)
        sys.exit(1)
