#!/usr/bin/env python3
"""
Download the PaleoBRAIN surface dataset (doi:10.48579/PRO/KZMMLM).

75 brain surfaces (B01-B75.ply) and 75 endocasts (E01-E75.ply) obtained by
MRI at Pitié-Salpêtrière Hospital (2022-2023, PaleoBRAIN project).

Reference: Balzeau A. (2025). doi:10.48579/PRO/KZMMLM

Usage examples
--------------
# Download everything (150 PLY files, ~750 MB total)
python data/download_paleobrain.py

# Download only the first 10 subjects
python data/download_paleobrain.py --n 10

# Download only brains for subjects 1-5
python data/download_paleobrain.py --n 5 --type brain

# Download first 10 subjects and downsample to ~5000 vertices
python data/download_paleobrain.py --n 10 --target-n 5000

# Re-download even if files exist
python data/download_paleobrain.py --n 5 --force
"""

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset manifest (filename → Dataverse numeric file ID)
# Persistent dataset DOI: doi:10.48579/PRO/KZMMLM
# Base download URL: https://data.indores.fr/api/access/datafile/{id}
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subject_manifest(n: int, kind: str) -> list[tuple[str, int]]:
    """Return list of (filename, file_id) for the requested subset."""
    entries: list[tuple[str, int]] = []
    if kind in ("brain", "all"):
        keys = sorted(_BRAIN_IDS)[:n]
        entries += [(k, _BRAIN_IDS[k]) for k in keys]
    if kind in ("endocast", "all"):
        keys = sorted(_ENDOCAST_IDS)[:n]
        entries += [(k, _ENDOCAST_IDS[k]) for k in keys]
    return entries


def _download_file(file_id: int, dest: Path) -> None:
    """Download a single file from the Dataverse repository."""
    url = f"{_BASE_URL}{file_id}"
    tmp = dest.with_suffix(".tmp")
    try:
        def _reporthook(count, block_size, total):
            if total > 0:
                pct = min(100, count * block_size * 100 // total)
                print(f"\r  {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        print()
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _downsample(path: Path, target_n: int) -> None:
    """Decimate *path* in-place to ~target_n vertices using pyclarcs."""
    try:
        from pyclarcs.io import load_surface, save_surface
        from pyclarcs.mesh import decimate_surface
    except ImportError:
        print("  [downsample] pyclarcs not installed — skipping resampling.")
        return

    pts, faces = load_surface(str(path))
    n_orig = len(pts)
    if n_orig <= target_n:
        print(f"  [downsample] {path.name}: {n_orig} vertices ≤ {target_n}, skipped.")
        return
    pts_d, faces_d = decimate_surface(pts, faces, target_n)
    save_surface(str(path), pts_d, faces_d)
    print(f"  [downsample] {path.name}: {n_orig} → {len(pts_d)} vertices.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", default="data/paleobrain", metavar="DIR",
        help="Directory to save PLY files (default: data/paleobrain).",
    )
    parser.add_argument(
        "--n", type=int, default=75, metavar="N",
        help="Number of subjects to download (1–75, default: all 75).",
    )
    parser.add_argument(
        "--type", choices=("brain", "endocast", "all"), default="all",
        dest="kind",
        help="Which surfaces to download (default: all).",
    )
    parser.add_argument(
        "--target-n", type=int, default=None, metavar="N",
        help="After download, downsample each surface to ~N vertices using "
             "VTK QuadricDecimation.  Requires pyclarcs.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files that already exist.",
    )
    args = parser.parse_args()

    n = max(1, min(75, args.n))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _subject_manifest(n, args.kind)
    print(
        f"PaleoBRAIN dataset — {len(manifest)} file(s) → {out_dir}\n"
        f"doi:10.48579/PRO/KZMMLM"
    )
    if args.target_n:
        print(f"Resampling to ~{args.target_n} vertices after download.\n")

    errors: list[str] = []
    for i, (name, file_id) in enumerate(manifest, 1):
        dest = out_dir / name
        prefix = f"[{i}/{len(manifest)}] {name}"

        if dest.exists() and not args.force:
            print(f"{prefix}  (already exists, skipped)")
            if args.target_n:
                _downsample(dest, args.target_n)
            continue

        print(f"{prefix}  downloading…")
        try:
            _download_file(file_id, dest)
        except urllib.error.URLError as exc:
            print(f"  ERROR: {exc}")
            errors.append(name)
            continue

        if args.target_n:
            _downsample(dest, args.target_n)

    print(f"\nDone. {len(manifest) - len(errors)}/{len(manifest)} file(s) ready in {out_dir}.")
    if errors:
        print("Failed:", ", ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
