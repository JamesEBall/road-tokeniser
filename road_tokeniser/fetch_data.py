"""Download UK STATS19 and NZ Waka Kotahi CAS crash data to data/raw/.

Idempotent — re-running skips files already present with the right size.
Writes a SHA-256 manifest at data/raw/manifest.json so changes between fetches
are visible.

CLI:
    rt-fetch --site uk
    rt-fetch --site nz
    rt-fetch --site both
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class DataSource:
    site: str
    name: str
    url: str
    filename: str
    licence: str

    @property
    def relpath(self) -> str:
        return f"{self.site}/{self.filename}"


SOURCES: list[DataSource] = [
    # NZ — Waka Kotahi Crash Analysis System (all crashes since 2000, geocoded)
    DataSource(
        site="nz",
        name="CAS (all crashes, geocoded WGS84)",
        url=(
            "https://opendata.arcgis.com/api/v3/datasets/"
            "8d684f1841fa4dbea6afaefc8a1ba0fc_0/downloads/data"
            "?format=csv&spatialRefId=4326"
        ),
        filename="CAS_Data_public.csv",
        licence="CC-BY 4.0 — © Waka Kotahi / NZ Transport Agency",
    ),
    # UK — DfT STATS19 5-year rolling (denser signal in small bboxes)
    DataSource(
        site="uk",
        name="STATS19 collisions last 5 years",
        url=(
            "https://data.dft.gov.uk/road-accidents-safety-data/"
            "dft-road-casualty-statistics-collision-last-5-years.csv"
        ),
        filename="stats19_collision_last_5_years.csv",
        licence="OGL v3.0 — © Crown copyright, UK Department for Transport",
    ),
    DataSource(
        site="uk",
        name="STATS19 casualties last 5 years",
        url=(
            "https://data.dft.gov.uk/road-accidents-safety-data/"
            "dft-road-casualty-statistics-casualty-last-5-years.csv"
        ),
        filename="stats19_casualty_last_5_years.csv",
        licence="OGL v3.0 — © Crown copyright, UK Department for Transport",
    ),
    DataSource(
        site="uk",
        name="STATS19 vehicles last 5 years",
        url=(
            "https://data.dft.gov.uk/road-accidents-safety-data/"
            "dft-road-casualty-statistics-vehicle-last-5-years.csv"
        ),
        filename="stats19_vehicle_last_5_years.csv",
        licence="OGL v3.0 — © Crown copyright, UK Department for Transport",
    ),
]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def project_root() -> Path:
    """Return the project root (parent of the package directory)."""
    return Path(__file__).resolve().parent.parent


def raw_dir() -> Path:
    d = project_root() / "data" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _expected_size(url: str, timeout: float = 30.0) -> int | None:
    """HEAD request to get content-length. Returns None if unavailable."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.ok and "content-length" in r.headers:
            return int(r.headers["content-length"])
    except requests.RequestException:
        return None
    return None


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def _download(src: DataSource, dest: Path, *, force: bool = False) -> tuple[bool, int]:
    """Download `src` to `dest` if missing or size-mismatched. Returns (downloaded?, size_bytes)."""
    expected = _expected_size(src.url)
    if dest.exists() and not force:
        actual = dest.stat().st_size
        if expected is None or actual == expected:
            return False, actual
        print(
            f"[fetch] size mismatch for {dest.name}: {actual} vs expected {expected} — refetching",
            file=sys.stderr,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(src.url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or expected or 0
        with tmp.open("wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=src.filename,
            disable=not sys.stderr.isatty(),
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return True, dest.stat().st_size


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"files": {}, "updated_at": None}
    return json.loads(path.read_text())


def _save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def fetch(sites: list[str], *, force: bool = False) -> dict:
    """Download all sources for the given sites. Returns the manifest."""
    chosen = [s for s in SOURCES if s.site in sites]
    if not chosen:
        raise ValueError(f"No sources for sites={sites}; known sites: {{nz, uk}}")

    base = raw_dir()
    manifest_path = base / "manifest.json"
    manifest = _load_manifest(manifest_path)

    for src in chosen:
        dest = base / src.relpath
        downloaded, size = _download(src, dest, force=force)
        digest = _sha256_of(dest)
        manifest["files"][src.relpath] = {
            "name": src.name,
            "url": src.url,
            "licence": src.licence,
            "size_bytes": size,
            "sha256": digest,
            "downloaded_now": downloaded,
        }
        status = "downloaded" if downloaded else "cached"
        print(f"[fetch] {status:>10}  {src.relpath}  ({size/1e6:.1f} MB)  sha256:{digest[:12]}")

    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_manifest(manifest_path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download crash data for road-tokeniser.")
    p.add_argument(
        "--site",
        choices=["uk", "nz", "both"],
        default="both",
        help="Which country's crash data to download.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached file matches expected size.",
    )
    args = p.parse_args(argv)

    sites = ["uk", "nz"] if args.site == "both" else [args.site]
    fetch(sites, force=args.force)
    print(f"[fetch] manifest at {raw_dir() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
