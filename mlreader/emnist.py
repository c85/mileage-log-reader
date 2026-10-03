"""Load EMNIST ByClass and reduce it to the 36 characters Form ML-7 uses.

Form ML-7 boxes only ever hold digits 0-9 or capital letters A-Z, so we keep
those 36 classes and drop the 26 lowercase ones.

Three things this module guarantees (each one is checked, not assumed):
1. Label mapping: ByClass labels 0-9 are '0'-'9' and 10-35 are 'A'-'Z'. We
   read NIST's own mapping file and assert this, so a wrong file fails loudly.
2. Orientation: EMNIST stores every image transposed (rows and columns swapped)
   relative to MNIST. We transpose back and assert the result looks upright.
3. Splits: EMNIST ships train and test only. We carve a validation set out of
   train with a fixed seed, so every teammate gets the identical split.

Usage:
    from mlreader.emnist import load_split
    x_train, y_train = load_split("train")   # x: (N, 28, 28) uint8, y: (N,) int64
"""

import gzip
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "emnist"

# Our class index i <-> CLASSES[i]. Identical to ByClass labels 0-35.
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CLASSES)  # 36
DIGIT_LABELS = range(0, 10)
LETTER_LABELS = range(10, 36)

SPLIT_SEED = 6642          # fixed seed for the train/validation split
VAL_FRACTION = 0.10        # share of each class moved from train to validation

# Which raw file pair each of our splits is read from.
_SOURCE_FILES = {
    "train": "emnist-byclass-train",
    "val": "emnist-byclass-train",   # validation is carved out of train
    "test": "emnist-byclass-test",
}


def read_idx(path):
    """Read a gzipped IDX file (the MNIST/EMNIST binary format) into a NumPy array.

    IDX layout: 2 zero bytes, 1 byte data type (0x08 = unsigned byte),
    1 byte number of dimensions, then one big-endian uint32 per dimension,
    then the raw data.
    """
    with gzip.open(path, "rb") as f:
        raw = f.read()
    zero, dtype_code, ndim = raw[0:2], raw[2], raw[3]
    if zero != b"\x00\x00" or dtype_code != 0x08:
        raise ValueError(f"{path} is not an unsigned-byte IDX file")
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: header says {shape} but file holds {data.size} values")
    return data.reshape(shape)


def read_mapping(data_dir=DATA_DIR):
    """Read NIST's mapping file: each line is '<label> <ASCII code>'."""
    mapping = {}
    for line in (Path(data_dir) / "emnist-byclass-mapping.txt").read_text().split("\n"):
        if line.strip():
            label, code = line.split()
            mapping[int(label)] = chr(int(code))
    return mapping


def check_mapping(mapping):
    """Assert ByClass labels 0-35 are exactly 0-9 then A-Z (guarantee 1)."""
    ours = "".join(mapping[i] for i in range(NUM_CLASSES))
    assert ours == CLASSES, f"Unexpected label mapping for 0-35: {ours!r}"


def fix_orientation(images):
    """Undo EMNIST's storage transpose: swap rows and columns of every image."""
    return images.transpose(0, 2, 1)


def check_orientation(images, labels):
    """Assert images are upright, using the digit 7 (guarantee 2).

    An upright 7 has its horizontal bar along the TOP. In a transposed 7 that
    bar runs down the LEFT side instead. Transposing swaps those two bands
    exactly, so on the average '7' the top band must hold more ink than the
    left band; if it doesn't, the images are still transposed.
    """
    sevens = images[labels == CLASSES.index("7")]
    if len(sevens) == 0:
        return  # nothing to check against (e.g. a tiny test subset)
    mean7 = sevens.astype(np.float64).mean(axis=0)
    top_ink = mean7[:9, :].sum()
    left_ink = mean7[:, :9].sum()
    assert top_ink > left_ink, (
        f"Images look transposed: top-band ink {top_ink:.0f} vs left-band ink "
        f"{left_ink:.0f} on the average '7'. Check fix_orientation()."
    )


def load_raw(prefix, data_dir=DATA_DIR):
    """Load one raw EMNIST file pair, without filtering or orientation fix."""
    data_dir = Path(data_dir)
    images = read_idx(data_dir / f"{prefix}-images-idx3-ubyte.gz")
    labels = read_idx(data_dir / f"{prefix}-labels-idx1-ubyte.gz").astype(np.int64)
    if len(images) != len(labels):
        raise ValueError(f"{prefix}: {len(images)} images but {len(labels)} labels")
    return images, labels


def train_val_indices(labels, seed=SPLIT_SEED, val_fraction=VAL_FRACTION):
    """Split train indices into (train, val), stratified by class (guarantee 3).

    Each class contributes the same fraction to validation, so rare letters
    such as Q or Z are not missing from it by chance. Same seed -> same split.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def load_split(split, data_dir=DATA_DIR):
    """Return (images, labels) for 'train', 'val' or 'test'.

    images: (N, 28, 28) uint8, upright, white ink on black like EMNIST.
    labels: (N,) int64 in 0..35; decode with CLASSES[label].
    """
    if split not in _SOURCE_FILES:
        raise ValueError(f"split must be one of {list(_SOURCE_FILES)}, got {split!r}")
    check_mapping(read_mapping(data_dir))

    images, labels = load_raw(_SOURCE_FILES[split], data_dir)
    keep = labels < NUM_CLASSES                 # drop lowercase a-z (labels 36-61)
    images, labels = fix_orientation(images[keep]), labels[keep]
    check_orientation(images, labels)

    if split in ("train", "val"):
        train_idx, val_idx = train_val_indices(labels)
        pick = train_idx if split == "train" else val_idx
        images, labels = images[pick], labels[pick]
    return images, labels
