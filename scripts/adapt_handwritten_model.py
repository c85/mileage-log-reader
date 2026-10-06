"""Adapt the EMNIST model's output layer using labeled handwritten development forms."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import DATA_DIR, REPO_ROOT  # noqa: E402
from mlreader.classifier import ModelNotAvailable  # noqa: E402
from mlreader.emnist import CLASSES  # noqa: E402
from mlreader.registration import extract_cells, register_page  # noqa: E402
from scripts.evaluate_team_filled_forms import _label_fields  # noqa: E402


def _read_examples(forms_dir, ground_truth):
    grouped = defaultdict(list)
    with ground_truth.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            grouped[row["FORM_ID"].strip()].append(row)
    if not grouped:
        raise ValueError(f"No labeled forms found in {ground_truth}.")

    images = []
    labels = []
    allowed_kinds = []
    used_forms = []
    for form_id, rows in sorted(grouped.items()):
        image_path = forms_dir / f"{form_id}.png"
        if not image_path.is_file():
            raise ValueError(f"Missing labeled form image: {image_path}")
        registration = register_page(image_path)
        if not registration.ok:
            raise ValueError(f"Could not register {image_path}: {registration.reason}")
        crops = extract_cells(registration.image)
        field_labels, _ = _label_fields(form_id, rows)
        used_forms.append(form_id)
        for field, expected in field_labels:
            for position, character in enumerate(expected):
                if character is None:
                    continue
                cell = crops[field][position]
                if cell.normalized is None:
                    continue
                images.append(cell.normalized.reshape(-1).astype(np.float32) / 255.0)
                labels.append(CLASSES.index(character))
                allowed_kinds.append(0 if character.isdigit() else 1)

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(allowed_kinds, dtype=np.int8),
        used_forms,
    )


def _forward_hidden(images, w1, b1):
    return np.maximum(images @ w1 + b1, 0)


def _adapt_output_layer(hidden, labels, allowed_kinds, w2, b2, args):
    original_w2 = w2.copy()
    w2 = w2.copy()
    b2 = b2.copy()
    first_w = np.zeros_like(w2)
    second_w = np.zeros_like(w2)
    first_b = np.zeros_like(b2)
    second_b = np.zeros_like(b2)
    rng = np.random.default_rng(args.seed)
    step = 0
    digit_classes = np.arange(0, 10)
    letter_classes = np.arange(10, 36)

    for epoch in range(args.epochs):
        order = rng.permutation(len(labels))
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset : offset + args.batch_size]
            batch_hidden = hidden[indices]
            batch_labels = labels[indices]
            batch_kinds = allowed_kinds[indices]
            logits = batch_hidden @ w2 + b2
            probabilities = np.zeros_like(logits)

            for kind, class_ids in ((0, digit_classes), (1, letter_classes)):
                rows = np.flatnonzero(batch_kinds == kind)
                if not len(rows):
                    continue
                restricted = logits[np.ix_(rows, class_ids)]
                restricted -= restricted.max(axis=1, keepdims=True)
                exp = np.exp(restricted)
                exp /= exp.sum(axis=1, keepdims=True)
                probabilities[np.ix_(rows, class_ids)] = exp

            gradient_logits = probabilities
            gradient_logits[np.arange(len(batch_labels)), batch_labels] -= 1.0
            gradient_logits /= len(batch_labels)
            gradient_w = batch_hidden.T @ gradient_logits
            gradient_w += args.regularization * (w2 - original_w2)
            gradient_b = gradient_logits.sum(axis=0)

            step += 1
            first_w = 0.9 * first_w + 0.1 * gradient_w
            second_w = 0.999 * second_w + 0.001 * gradient_w * gradient_w
            first_b = 0.9 * first_b + 0.1 * gradient_b
            second_b = 0.999 * second_b + 0.001 * gradient_b * gradient_b
            corrected_first_w = first_w / (1 - 0.9**step)
            corrected_second_w = second_w / (1 - 0.999**step)
            corrected_first_b = first_b / (1 - 0.9**step)
            corrected_second_b = second_b / (1 - 0.999**step)
            w2 -= args.learning_rate * corrected_first_w / (np.sqrt(corrected_second_w) + 1e-8)
            b2 -= args.learning_rate * corrected_first_b / (np.sqrt(corrected_second_b) + 1e-8)

    return w2, b2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--forms-dir", type=Path, default=REPO_ROOT / "examples/development_forms"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REPO_ROOT / "examples/development_forms/ground_truth.csv",
    )
    parser.add_argument("--base-model", type=Path, default=DATA_DIR / "models/emnist_mlp.npz")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DATA_DIR / "models/emnist_mlp_handwritten.npz",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=6642)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive.")
    if args.regularization < 0:
        parser.error("regularization cannot be negative.")
    if not args.base_model.is_file():
        raise ModelNotAvailable(f"No base model found at {args.base_model}.")

    images, labels, allowed_kinds, forms = _read_examples(
        args.forms_dir, args.ground_truth
    )
    with np.load(args.base_model, allow_pickle=False) as source:
        if str(source["classes"].item()) != CLASSES:
            raise ValueError("Base model label map does not match the current EMNIST loader.")
        w1 = source["w1"].astype(np.float32)
        b1 = source["b1"].astype(np.float32)
        w2 = source["w2"].astype(np.float32)
        b2 = source["b2"].astype(np.float32)
        metadata = json.loads(str(source["metadata"].item())) if "metadata" in source else {}

    hidden = _forward_hidden(images, w1, b1)
    adapted_w2, adapted_b2 = _adapt_output_layer(
        hidden, labels, allowed_kinds, w2, b2, args
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    metadata.update(
        {
            "adaptation_method": "EMNIST MLP output-layer fine-tuning",
            "adaptation_source": str(args.forms_dir),
            "adaptation_forms": forms,
            "adaptation_samples": int(len(labels)),
            "adaptation_epochs": args.epochs,
            "adaptation_learning_rate": args.learning_rate,
            "adaptation_regularization": args.regularization,
            "adaptation_seed": args.seed,
            "base_model": str(args.base_model),
        }
    )
    np.savez_compressed(
        args.output_model,
        w1=w1,
        b1=b1,
        w2=adapted_w2,
        b2=adapted_b2,
        classes=np.asarray(CLASSES),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"Adapted {len(labels)} character cells from {len(forms)} development forms.")
    print(f"Saved optional checkpoint: {args.output_model}")


if __name__ == "__main__":
    main()
