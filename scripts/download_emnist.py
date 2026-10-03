"""Download the EMNIST ByClass files from NIST and verify them.

Usage (from the repo root):
    python scripts/download_emnist.py

What it does:
1. Downloads NIST's gzip.zip (~0.5 GB, all EMNIST splits) into data/raw/.
2. Extracts only the five ByClass files we use into data/emnist/.
3. Checks SHA-256 checksums against configs/emnist_sha256.json so every
   teammate trains on byte-identical data. On the very first run that file
   does not exist yet, so the script creates it; commit it to git.

Uses only the Python standard library, so it runs before anything is installed.
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

NIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

# The only files we need from the zip: ByClass train/test images and labels,
# plus the mapping from label index to ASCII character code.
BYCLASS_FILES = [
    "emnist-byclass-train-images-idx3-ubyte.gz",
    "emnist-byclass-train-labels-idx1-ubyte.gz",
    "emnist-byclass-test-images-idx3-ubyte.gz",
    "emnist-byclass-test-labels-idx1-ubyte.gz",
    "emnist-byclass-mapping.txt",
]

# NIST's server answers 403 Forbidden to Python's default User-Agent, so
# identify as a regular browser.
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "emnist"
CHECKSUM_FILE = REPO_ROOT / "configs" / "emnist_sha256.json"


def sha256_of(path):
    """Hash a file in 1 MB chunks so large files don't fill memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    """Download url to dest, printing progress. Skips if dest already exists."""
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")  # avoid leaving a half file if interrupted
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        resp = urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        sys.exit(
            f"ERROR: download refused ({e.code} {e.reason}).\n"
            f"Download {url} in your browser instead, unzip it, copy the five\n"
            f"emnist-byclass-* files into {OUT_DIR}, and re-run this script."
        )
    with resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def extract_byclass(zip_path, out_dir):
    """Copy just the ByClass files out of the zip (they sit in a gzip/ folder)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        by_name = {Path(n).name: n for n in zf.namelist()}
        for name in BYCLASS_FILES:
            if name not in by_name:
                sys.exit(f"ERROR: {name} not found in {zip_path}")
            target = out_dir / name
            if target.exists():
                continue
            with zf.open(by_name[name]) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"Extracted {name}")


def verify_checksums(out_dir, checksum_file):
    """Compare file hashes to the committed record, or create the record."""
    actual = {name: sha256_of(out_dir / name) for name in BYCLASS_FILES}
    if not checksum_file.exists():
        checksum_file.parent.mkdir(parents=True, exist_ok=True)
        checksum_file.write_text(json.dumps(actual, indent=2) + "\n")
        print(f"No checksum record found; created {checksum_file.relative_to(REPO_ROOT)}.")
        print("Commit that file so teammates can verify they have identical data.")
        return
    expected = json.loads(checksum_file.read_text())
    bad = [n for n in BYCLASS_FILES if expected.get(n) != actual[n]]
    if bad:
        sys.exit(f"ERROR: checksum mismatch for {bad}. Delete data/ and re-download.")
    print("All checksums match the committed record.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=NIST_URL,
                        help="Override the download URL (e.g. a mirror).")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the 0.5 GB zip after extracting.")
    args = parser.parse_args()

    zip_path = RAW_DIR / "emnist_gzip.zip"
    if not all((OUT_DIR / n).exists() for n in BYCLASS_FILES):
        download(args.url, zip_path)
        extract_byclass(zip_path, OUT_DIR)
        if not args.keep_zip:
            zip_path.unlink()
    verify_checksums(OUT_DIR, CHECKSUM_FILE)
    print(f"EMNIST ByClass ready in {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
