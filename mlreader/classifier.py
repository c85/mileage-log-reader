"""NumPy MLP inference for the 36 Form ML-7 character classes."""

from pathlib import Path

import numpy as np

from mlreader.emnist import CLASSES


class ModelNotAvailable(FileNotFoundError):
    pass


class EMNISTMLP:
    def __init__(self, weights_path):
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise ModelNotAvailable(
                f"No trained reader found at {self.weights_path}. Run scripts/train_model.py first."
            )
        with np.load(self.weights_path, allow_pickle=False) as model:
            if str(model["classes"].item()) != CLASSES:
                raise ValueError("Model label map does not match the current EMNIST loader.")
            self.w1 = model["w1"].astype(np.float32)
            self.b1 = model["b1"].astype(np.float32)
            self.w2 = model["w2"].astype(np.float32)
            self.b2 = model["b2"].astype(np.float32)
            self.metadata = model["metadata"].item() if "metadata" in model else "{}"

    def _probabilities(self, image):
        vector = np.asarray(image, dtype=np.float32).reshape(1, 784) / 255.0
        hidden = np.maximum(vector @ self.w1 + self.b1, 0)
        logits = hidden @ self.w2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return (exp / exp.sum(axis=1, keepdims=True))[0]

    def predict(self, image, allowed="digit", top_k=3):
        """Return a top class and probabilities restricted by the field type."""
        if allowed == "digit":
            labels = np.arange(0, 10)
        elif allowed == "letter":
            labels = np.arange(10, 36)
        else:
            labels = np.arange(0, 36)
        all_probabilities = self._probabilities(image)
        restricted = all_probabilities[labels].astype(np.float64)
        total = restricted.sum()
        if total <= 0 or not np.isfinite(total):
            return {
                "character": None,
                "confidence": None,
                "class_probabilities": {},
                "top_alternatives": [],
            }
        restricted /= total
        order = np.argsort(restricted)[::-1][:top_k]
        return {
            "character": CLASSES[int(labels[order[0]])],
            "confidence": float(restricted[order[0]]),
            "class_probabilities": {
                CLASSES[int(label)]: float(restricted[index])
                for index, label in enumerate(labels)
            },
            "top_alternatives": [
                {"character": CLASSES[int(labels[i])], "confidence": float(restricted[i])}
                for i in order
            ],
        }
