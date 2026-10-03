# ML-7 Mileage Log Reader

ISM 6642 final project: reads handwritten Form ML-7 mileage logs and decides
which rows to auto-post and which to route to an AP clerk.

## Folder layout

| Folder | What goes in it |
|---|---|
| `mlreader/` | Reusable code (loaders, models, pipeline steps), one module per topic. Shared paths live in `mlreader/__init__.py`. |
| `scripts/` | Short runnable commands (`python scripts/<name>.py`) that call `mlreader/`. No logic that another script would need. |
| `configs/` | Small committed settings and checksums. |
| `data/`, `outputs/` | Created by the scripts; never committed. |

Add to an existing module before creating a new one, and check open branches
and pull requests first so two people don't build the same thing.

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python scripts/download_emnist.py   # downloads EMNIST ByClass (~0.5 GB) into data/emnist/
python scripts/audit_emnist.py      # checks labels/orientation, writes outputs/emnist_audit/
```

Data files are not committed; the script recreates them and verifies their
checksums against `configs/emnist_sha256.json`.

## Character data (`mlreader/emnist.py`)

`load_split("train" | "val" | "test")` returns upright 28x28 images (white ink
on black) and labels 0-35, decoded with `CLASSES = "0-9A-Z"`. Lowercase
ByClass classes are dropped because Form ML-7 only holds digits and capitals.
Validation is a stratified 10% of EMNIST train (seed 6642); test is EMNIST's
own test split, which is also the only source of characters for synthetic
test logs.
