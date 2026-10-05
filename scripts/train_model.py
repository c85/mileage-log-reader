"""Train the from-scratch NumPy MLP and report held-out ByClass performance.

Usage (after scripts/download_emnist.py):
    python scripts/train_model.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import DATA_DIR, OUTPUT_DIR  # noqa: E402
from mlreader.emnist import CLASSES, class_weights, duplicate_indices, load_split  # noqa: E402
from mlreader.registration import normalize_emnist  # noqa: E402


def _select_training_indices(labels, max_images, seed):
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError("The training split must contain all 36 classes.")
    if max_images >= len(labels):
        return np.arange(len(labels))
    minimum_per_class = min(500, max_images // len(CLASSES))
    if minimum_per_class < 1:
        raise ValueError("--max-train-images must be at least 36.")
    quotas = np.minimum(counts, minimum_per_class)
    remaining = max_images - int(quotas.sum())
    if remaining > 0:
        proportions = counts / counts.sum()
        additions = np.floor(proportions * remaining).astype(int)
        quotas = np.minimum(counts, quotas + additions)
        while quotas.sum() < max_images:
            room = np.flatnonzero(quotas < counts)
            if not len(room):
                break
            choices = room[np.argsort(counts[room] - quotas[room])[::-1]]
            for label in choices:
                if quotas.sum() >= max_images:
                    break
                quotas[label] += 1
    selected = []
    for label, quota in enumerate(quotas):
        indices = np.flatnonzero(labels == label)
        selected.extend(rng.choice(indices, size=int(quota), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _forward(x, w1, b1, w2, b2):
    z1 = x @ w1 + b1
    hidden = np.maximum(z1, 0)
    logits = hidden @ w2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True), z1, hidden


def _predict(images, weights, batch_size=512):
    w1, b1, w2, b2 = weights
    output = []
    for start in range(0, len(images), batch_size):
        x = normalize_emnist(images[start : start + batch_size]).reshape(-1, 784).astype(np.float32) / 255.0
        probs, _, _ = _forward(x, w1, b1, w2, b2)
        output.append(probs)
    return np.concatenate(output, axis=0)


def _restricted_prediction(probabilities, labels):
    prediction = np.argmax(probabilities, axis=1)
    digit_rows = np.flatnonzero(labels < 10)
    letter_rows = np.flatnonzero(labels >= 10)
    if len(digit_rows):
        prediction[digit_rows] = np.argmax(probabilities[digit_rows, :10], axis=1)
    if len(letter_rows):
        prediction[letter_rows] = 10 + np.argmax(probabilities[letter_rows, 10:36], axis=1)
    return prediction


def _report(name, probabilities, labels):
    raw = np.argmax(probabilities, axis=1)
    restricted = _restricted_prediction(probabilities, labels)
    digit = labels < 10
    letter = ~digit
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, predicted in zip(labels, restricted):
        confusion[int(truth), int(predicted)] += 1
    per_class = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(len(CLASSES), dtype=float),
        where=confusion.sum(axis=1) > 0,
    )
    return {
        "split": name,
        "sample_count": int(len(labels)),
        "unrestricted_36_class_accuracy": float(np.mean(raw == labels)),
        "field_restricted_accuracy": float(np.mean(restricted == labels)),
        "digit_accuracy_unrestricted": float(np.mean(raw[digit] == labels[digit])) if digit.any() else None,
        "letter_accuracy_unrestricted": float(np.mean(raw[letter] == labels[letter])) if letter.any() else None,
        "digit_accuracy_field_restricted": float(np.mean(restricted[digit] == labels[digit])) if digit.any() else None,
        "letter_accuracy_field_restricted": float(np.mean(restricted[letter] == labels[letter])) if letter.any() else None,
        "macro_class_accuracy_field_restricted": float(per_class.mean()),
        "per_class_accuracy_field_restricted": {char: float(per_class[i]) for i, char in enumerate(CLASSES)},
        "confusion_matrix": confusion,
    }


def _train(x_images, y, args):
    rng = np.random.default_rng(args.seed)
    weights = class_weights(y).astype(np.float32)
    weights /= np.mean(weights[y])
    hidden_size = args.hidden_size
    w1 = rng.normal(0, np.sqrt(2 / 784), (784, hidden_size)).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    w2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, len(CLASSES))).astype(np.float32)
    b2 = np.zeros(len(CLASSES), dtype=np.float32)
    parameters = [w1, b1, w2, b2]
    first_moments = [np.zeros_like(value) for value in parameters]
    second_moments = [np.zeros_like(value) for value in parameters]
    step = 0

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y))
        total_loss = 0.0
        seen = 0
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset : offset + args.batch_size]
            x = normalize_emnist(x_images[indices]).reshape(-1, 784).astype(np.float32) / 255.0
            target = y[indices]
            probs, z1, hidden = _forward(x, *parameters)
            sample_weight = weights[target]
            normalizer = max(float(sample_weight.sum()), 1e-8)
            total_loss += float((-np.log(np.maximum(probs[np.arange(len(target)), target], 1e-8)) * sample_weight).sum())
            seen += len(target)
            delta = probs
            delta[np.arange(len(target)), target] -= 1.0
            delta *= (sample_weight / normalizer)[:, None]
            grad_w2 = hidden.T @ delta
            grad_b2 = delta.sum(axis=0)
            grad_hidden = delta @ parameters[2].T
            grad_hidden[z1 <= 0] = 0
            grad_w1 = x.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            gradients = [grad_w1, grad_b1, grad_w2, grad_b2]
            step += 1
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                first_moments[index] = 0.9 * first_moments[index] + 0.1 * gradient
                second_moments[index] = 0.999 * second_moments[index] + 0.001 * gradient * gradient
                m_hat = first_moments[index] / (1 - 0.9**step)
                v_hat = second_moments[index] / (1 - 0.999**step)
                parameter -= args.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        print(f"epoch {epoch:02d}/{args.epochs}: weighted loss {total_loss / max(seen, 1):.4f}", flush=True)
    return parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-train-images", type=int, default=90000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=6642)
    args = parser.parse_args()

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")
    selected = _select_training_indices(train_y, args.max_train_images, args.seed)
    x_train, y_train = train_x[selected], train_y[selected]
    del selected
    print(f"Training on {len(y_train):,} images from {len(set(y_train.tolist()))} classes.")
    parameters = _train(x_train, y_train, args)

    val_probabilities = _predict(val_x, parameters)
    val_report = _report("validation", val_probabilities, val_y)
    leaked = set(duplicate_indices(train_x, test_x).tolist())
    leaked.update(duplicate_indices(val_x, test_x).tolist())
    test_keep = np.array([i for i in range(len(test_y)) if i not in leaked], dtype=np.int64)
    test_x, test_y = test_x[test_keep], test_y[test_keep]
    test_probabilities = _predict(test_x, parameters)
    test_report = _report("leak-free-test", test_probabilities, test_y)

    model_dir = DATA_DIR / "models"
    output_dir = OUTPUT_DIR / "model_eval"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": f"784-{args.hidden_size}-ReLU-36-softmax",
        "training_source": "EMNIST ByClass train; lowercase classes removed; orientation fixed by mlreader.emnist",
        "preprocessing": "inference-matched: threshold ink, fit 20x20, center mass in 28x28",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "training_samples": int(len(y_train)),
        "training_class_counts": {char: int(np.sum(y_train == i)) for i, char in enumerate(CLASSES)},
        "inverse_frequency_weighting": True,
        "excluded_test_duplicates": int(len(leaked)),
        "classes": CLASSES,
        "numpy_version": np.__version__,
        "validation_field_restricted_accuracy": val_report["field_restricted_accuracy"],
        "test_field_restricted_accuracy": test_report["field_restricted_accuracy"],
    }
    model_path = model_dir / "emnist_mlp.npz"
    np.savez_compressed(model_path, w1=parameters[0], b1=parameters[1], w2=parameters[2], b2=parameters[3], classes=np.asarray(CLASSES), metadata=np.asarray(json.dumps(metadata)))
    for report, filename in ((val_report, "validation_confusion.csv"), (test_report, "test_confusion.csv")):
        matrix = report.pop("confusion_matrix")
        with (output_dir / filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["truth/prediction", *CLASSES])
            for label, row in zip(CLASSES, matrix):
                writer.writerow([label, *row.tolist()])
    metrics = {
        "conditions": "isolated EMNIST characters normalized with the same 20x20 center-of-mass transform as inference",
        "model": metadata,
        "validation": val_report,
        "test": test_report,
        "note": "Character metrics are not form-cell extraction or row auto-post metrics.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Leak-free held-out accuracy (all 36 classes): {test_report['unrestricted_36_class_accuracy']:.3%}")
    print(f"Leak-free held-out accuracy (field restricted): {test_report['field_restricted_accuracy']:.3%}")
    print(f"Model: {model_path}")
    print(f"Metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
