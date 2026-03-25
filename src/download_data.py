"""
src/download_data.py

Downloads real CMS Medicare Part D Prescriber data from:
https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers

This is the actual dataset CMS publishes annually. It covers every
provider who wrote Medicare Part D prescriptions — name, NPI, specialty,
every drug they prescribed, total claims, total cost, beneficiary count.

2022 file: ~1.2 million providers, 25 million drug-provider rows, ~2.5GB uncompressed
We use the summarized version (~500MB) which aggregates by provider+drug.

No API key needed. No registration. Public domain government data.
"""

import requests
import os
import hashlib
from pathlib import Path
from tqdm import tqdm

# CMS data URLs — Part D Prescribers by Provider and Drug
# These are the actual CMS download links
CMS_URLS = {
    "2022": "https://data.cms.gov/sites/default/files/2024-04/bae27a41-d5af-47ef-b3eb-28895e3eb7be/MUP_DPR_RY24_P04_V10_DY22_NPIBN.csv",
    "2021": "https://data.cms.gov/sites/default/files/2023-04/c0e7d59f-e3da-459f-a0ef-bce1b3c6a749/MUP_DPR_RY23_P04_V10_DY21_NPIBN.csv",
    "2020": "https://data.cms.gov/sites/default/files/2022-07/MUP_DPR_RY22_P04_V10_DY20_NPIBN.csv",
}

# OIG Exclusions List — all providers excluded from federal programs
OIG_URL = "https://oig.hhs.gov/exclusions/downloadables/LEIE.csv"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, chunk_size: int = 1024*1024) -> Path:
    """Stream-download a file with progress bar."""
    print(f"\nDownloading: {dest.name}")
    print(f"URL: {url}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  Saved → {dest}  ({size_mb:.1f} MB)")
    return dest


def get_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_cms_partd(year: str = "2022") -> Path:
    url  = CMS_URLS[year]
    dest = RAW_DIR / f"cms_partd_{year}.csv"
    if dest.exists():
        print(f"  {dest.name} already exists ({dest.stat().st_size/1e6:.0f} MB) — skipping")
        return dest
    return download_file(url, dest)


def download_oig_exclusions() -> Path:
    dest = RAW_DIR / "oig_exclusions.csv"
    if dest.exists():
        print(f"  oig_exclusions.csv already exists — skipping")
        return dest
    return download_file(OIG_URL, dest)


def validate_downloads():
    """Check downloaded files look sane."""
    import pandas as pd
    results = {}

    for year in ["2022", "2021"]:
        f = RAW_DIR / f"cms_partd_{year}.csv"
        if f.exists():
            df = pd.read_csv(f, nrows=5)
            results[f"cms_partd_{year}"] = {
                "exists": True,
                "columns": list(df.columns),
                "size_mb": f.stat().st_size / 1e6,
            }
            print(f"  ✓ cms_partd_{year}.csv — {results[f'cms_partd_{year}']['size_mb']:.0f}MB")
            print(f"    Columns: {list(df.columns)[:6]}...")

    oig = RAW_DIR / "oig_exclusions.csv"
    if oig.exists():
        df = pd.read_csv(oig, nrows=5, encoding="latin-1")
        results["oig"] = {"exists": True, "columns": list(df.columns)}
        print(f"  ✓ oig_exclusions.csv — {oig.stat().st_size/1e6:.0f}MB")

    return results


if __name__ == "__main__":
    print("CMS Medicare Part D Data Downloader")
    print("Source: https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers")
    print("=" * 60)
    print("\nDownloading 2022 Part D data (largest/most recent)...")
    download_cms_partd("2022")
    print("\nDownloading OIG exclusions list...")
    download_oig_exclusions()
    print("\nValidating downloads...")
    validate_downloads()
    print("\nDone. Run src/clean.py next.")
