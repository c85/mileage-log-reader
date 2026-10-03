"""Form ML-7 mileage-log reader: reusable code imported by the scripts/ commands.

Project-wide paths live here, once, so every module and script agrees on them.
This file must stay standard-library only: scripts/download_emnist.py imports
it before numpy is installed.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"            # downloaded / generated data (not in git)
EMNIST_DIR = DATA_DIR / "emnist"
CONFIG_DIR = REPO_ROOT / "configs"       # small committed settings and checksums
OUTPUT_DIR = REPO_ROOT / "outputs"       # reports and figures (not in git)
