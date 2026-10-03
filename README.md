# ML-7 Mileage Log Reader

ISM 6642 final project: reads handwritten Form ML-7 mileage logs and decides
which rows to auto-post and which to route to an AP clerk.

## Setup

Requires Python 3.9+.

```bash
python scripts/download_emnist.py   # downloads EMNIST ByClass (~0.5 GB) into data/emnist/
```

Data files are not committed; the script recreates them and verifies their
checksums against `configs/emnist_sha256.json`.
