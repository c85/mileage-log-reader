"""SCRUM-10 audit: prove the EMNIST input path is correct and report class coverage.

Usage (from the repo root, after scripts/download_emnist.py):
    python scripts/audit_emnist.py

Writes to outputs/emnist_audit/:
    orientation_before_after.png  raw (as stored) vs. fixed images, same samples
    samples_per_class.png         8 examples of each of the 36 classes, after the fix
    class_counts.csv              images per class in train / val / test
    class_counts.png              bar chart of the train counts (shows the imbalance)
    split_summary.json            split sizes, seed, and imbalance figures
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no window needed
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import emnist  # noqa: E402

OUT_DIR = emnist.REPO_ROOT / "outputs" / "emnist_audit"
SEED = 0  # only picks which examples appear in the figures

# Chart colors: light surface, one blue series, neutral text and grid.
SURFACE, INK, INK_2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def show(ax, img):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])


def orientation_figure(path):
    """Same 12 training images, as stored on disk (top) and after the fix (bottom)."""
    images, labels = emnist.load_raw("emnist-byclass-train")
    rng = np.random.default_rng(SEED)
    chars = "0237AEFGJLRZ"  # asymmetric shapes make a transpose easy to spot
    picks = [rng.choice(np.flatnonzero(labels == emnist.CLASSES.index(c))) for c in chars]
    raw = images[picks]
    fixed = emnist.fix_orientation(raw)

    fig, axes = plt.subplots(2, len(chars), figsize=(len(chars) * 0.9, 2.6), facecolor=SURFACE)
    for j, c in enumerate(chars):
        show(axes[0, j], raw[j])
        show(axes[1, j], fixed[j])
        axes[0, j].set_title(c, color=INK, fontsize=10)
    axes[0, 0].set_ylabel("as stored", color=INK_2, fontsize=9)
    axes[1, 0].set_ylabel("fixed", color=INK_2, fontsize=9)
    fig.suptitle("EMNIST orientation: raw files are transposed; the loader swaps rows/columns back",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def samples_figure(path, images, labels, per_class=8):
    """A 36-row grid: one row per class, `per_class` random examples each."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(emnist.NUM_CLASSES, per_class,
                             figsize=(per_class * 0.55, emnist.NUM_CLASSES * 0.55), facecolor=SURFACE)
    for c in range(emnist.NUM_CLASSES):
        idx = rng.choice(np.flatnonzero(labels == c), size=per_class, replace=False)
        for j, i in enumerate(idx):
            show(axes[c, j], images[i])
        axes[c, 0].set_ylabel(emnist.CLASSES[c], rotation=0, labelpad=10, va="center",
                              color=INK, fontsize=9)
    fig.suptitle("Training samples per class after loading (0-9, A-Z)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def counts_chart(path, train_counts):
    """Bar chart of training images per class. One series, bars start at zero."""
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(emnist.NUM_CLASSES)
    ax.bar(x, train_counts, width=0.8, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, list(emnist.CLASSES), color=INK, fontsize=9)
    ax.tick_params(axis="y", colors=INK_2, labelsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)

    # Divider and labels for the two groups, plus direct labels on the extremes only.
    ax.axvline(9.5, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    top = train_counts.max()
    ax.text(4.5, top * 1.13, "digits", ha="center", color=INK_2, fontsize=9)
    ax.text(22.5, top * 1.13, "capital letters", ha="center", color=INK_2, fontsize=9)
    for i in (int(train_counts.argmax()), int(train_counts.argmin())):
        ax.text(i, train_counts[i] + top * 0.015, f"{train_counts[i]:,}", ha="center",
                va="bottom", color=INK, fontsize=8)
    ax.set_ylim(0, top * 1.2)
    ax.set_title("Training images per class, EMNIST ByClass (after removing lowercase)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = emnist.read_mapping()
    emnist.check_mapping(mapping)
    print(f"Label mapping OK: {len(mapping)} ByClass labels; we keep 0-35 = {emnist.CLASSES}")

    splits = {name: emnist.load_split(name) for name in ("train", "val", "test")}
    print("Orientation check passed on every split (asserted inside load_split).")

    counts = {name: np.bincount(y, minlength=emnist.NUM_CLASSES) for name, (_, y) in splits.items()}

    with open(OUT_DIR / "class_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "type", "train", "val", "test", "train_share_pct"])
        total_train = counts["train"].sum()
        for c, ch in enumerate(emnist.CLASSES):
            w.writerow([ch, "digit" if c < 10 else "letter", counts["train"][c], counts["val"][c],
                        counts["test"][c], round(100 * counts["train"][c] / total_train, 2)])

    tr = counts["train"]
    digits, letters = tr[:10], tr[10:]
    summary = {
        "source": "NIST EMNIST ByClass (gzip.zip), checksums in configs/emnist_sha256.json",
        "classes_kept": emnist.CLASSES,
        "lowercase_dropped": True,
        "orientation_fix": "transpose rows/columns of each 28x28 image",
        "split_seed": emnist.SPLIT_SEED,
        "val_fraction_of_train_per_class": emnist.VAL_FRACTION,
        "sizes": {name: int(y.size) for name, (_, y) in splits.items()},
        "digits_share_of_train_pct": round(100 * digits.sum() / tr.sum(), 1),
        "largest_class": [emnist.CLASSES[tr.argmax()], int(tr.max())],
        "smallest_class": [emnist.CLASSES[tr.argmin()], int(tr.min())],
        "largest_to_smallest_ratio": round(float(tr.max() / tr.min()), 1),
        "smallest_digit": [emnist.CLASSES[digits.argmin()], int(digits.min())],
        "smallest_letter": [emnist.CLASSES[10 + letters.argmin()], int(letters.min())],
        "largest_letter": [emnist.CLASSES[10 + letters.argmax()], int(letters.max())],
    }
    (OUT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    orientation_figure(OUT_DIR / "orientation_before_after.png")
    samples_figure(OUT_DIR / "samples_per_class.png", *splits["train"])
    counts_chart(OUT_DIR / "class_counts.png", tr)

    print(json.dumps(summary["sizes"]), "images in train / val / test")
    print(f"Digits are {summary['digits_share_of_train_pct']}% of train; largest class "
          f"{summary['largest_class'][0]} has {summary['largest_to_smallest_ratio']}x the "
          f"images of smallest class {summary['smallest_class'][0]}.")
    print(f"Wrote audit files to {OUT_DIR.relative_to(emnist.REPO_ROOT)}/")


if __name__ == "__main__":
    main()
