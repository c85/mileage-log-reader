"""Synthetic Form ML-7 generation and photo-quality perturbations."""

from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

from mlreader import CONFIG_DIR, REPO_ROOT
from mlreader.layout import CANVAS_HEIGHT, CANVAS_WIDTH, FIELDS


TEMPLATE_PATH = REPO_ROOT / "assets" / "form_ml7_blank.png"


def load_template(path=TEMPLATE_PATH):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"ML-7 blank template not found: {path}")
    if image.shape[1] != CANVAS_WIDTH or image.shape[0] != CANVAS_HEIGHT:
        raise ValueError(f"Template must be {CANVAS_WIDTH}x{CANVAS_HEIGHT}; got {image.shape[1]}x{image.shape[0]}")
    return image


def _date_raw(day):
    return day.strftime("%m%d")


def build_log_truth(references=None, week_ending=date(2026, 9, 27), rows=6, seed=6642, discrepancy_row=None, total_offset=0):
    """Create a made-up valid log; optionally insert a known arithmetic error."""
    if references is None:
        import json

        references = json.loads((CONFIG_DIR / "reference_data.json").read_text())
    rng = np.random.default_rng(seed)
    employee = str(rng.choice(references["employees"]))
    schedule = references["visit_schedule"][employee]
    days = [week_ending - timedelta(days=n) for n in range(6, -1, -1)]
    available = [day for day in days if day.isoformat() in schedule]
    if not available:
        raise ValueError(f"No synthetic visits are configured for {employee} in the requested week.")
    chosen_days = sorted(rng.choice(available, size=min(rows, len(available)), replace=False))
    start = int(rng.integers(48000, 70000))
    output_rows = []
    for index, day in enumerate(chosen_days, start=1):
        gap = int(rng.integers(0, 6))
        start += gap
        distance = int(rng.integers(8, 95))
        end = start + distance
        clients = schedule[day.isoformat()]
        client = str(rng.choice(clients))
        written_miles = distance
        if discrepancy_row == index:
            written_miles = max(0, distance - 5)
        output_rows.append(
            {
                "row_number": index,
                "date": _date_raw(day),
                "trip_date_iso": day.isoformat(),
                "client": client,
                "odometer_start": f"{start:06d}",
                "odometer_end": f"{end:06d}",
                "miles": f"{written_miles:03d}",
                "actual_distance": distance,
            }
        )
        start = end
    total = sum(int(row["miles"]) for row in output_rows) + total_offset
    fields = {
        "header.employee_id": employee,
        "header.week_ending": week_ending.strftime("%m%d%y"),
        "footer.total_miles": f"{total:04d}",
    }
    for row in output_rows:
        row_number = row["row_number"]
        for name in ("date", "client", "odometer_start", "odometer_end", "miles"):
            fields[f"rows.{row_number}.{name}"] = row[name]
    return {
        "employee_id": employee,
        "week_ending": week_ending.isoformat(),
        "rows": output_rows,
        "total_miles": f"{total:04d}",
        "fields": fields,
    }


def draw_test_characters(truth, test_images, test_labels, rng, source_indices=None):
    """Render held-out EMNIST test images into the blank printed boxes."""
    from mlreader.emnist import CLASSES

    image = load_template()
    choices = {label: np.flatnonzero(test_labels == label) for label in range(len(CLASSES))}
    if any(not len(indices) for indices in choices.values()):
        raise ValueError("Held-out EMNIST test split must contain all 36 classes.")
    sources = []
    for field, value in truth["fields"].items():
        cells = FIELDS[field]
        if len(value) != len(cells):
            raise ValueError(f"Truth field {field} has {len(value)} characters for {len(cells)} cells.")
        for char, location in zip(value, cells):
            label = CLASSES.index(char)
            source_index = int(rng.choice(choices[label]))
            source = test_images[source_index]
            ys, xs = np.where(source > 12)
            if not len(xs):
                continue
            glyph = source[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            left, top, right, bottom = location.rect
            box_w = right - left - 2 * 11
            box_h = bottom - top - 2 * 10
            scale = min(box_w / glyph.shape[1], box_h / glyph.shape[0])
            new_size = (max(1, round(glyph.shape[1] * scale)), max(1, round(glyph.shape[0] * scale)))
            glyph = cv2.resize(glyph, new_size, interpolation=cv2.INTER_CUBIC)
            x = left + (right - left - new_size[0]) // 2
            y = top + (bottom - top - new_size[1]) // 2
            strength = glyph.astype(np.float32) / 255.0
            region = image[y : y + new_size[1], x : x + new_size[0]].astype(np.float32)
            rendered = region * (1.0 - 0.88 * strength[:, :, None]) + 12.0 * (0.88 * strength[:, :, None])
            image[y : y + new_size[1], x : x + new_size[0]] = np.clip(rendered, 0, 255).astype(np.uint8)
            sources.append(
                {
                    "field": field,
                    "position": location.position,
                    "label": char,
                    "source_split": "EMNIST ByClass test",
                    "test_image_index": int(source_indices[source_index]) if source_indices is not None else source_index,
                }
            )
    return image, sources


def capture_page(page, condition="clean", seed=6642, jpeg_quality=88):
    """Put a canonical form on a neutral background with a controlled capture.

    Returns the captured image and the source-image page corners used for the
    synthetic ground truth.
    """
    rng = np.random.default_rng(seed)
    frame_w, frame_h = 2500, 1700
    background = np.full((frame_h, frame_w, 3), 122, dtype=np.uint8)
    src = np.float32(
        [[0, 0], [CANVAS_WIDTH - 1, 0], [CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1], [0, CANVAS_HEIGHT - 1]]
    )
    if condition == "clean":
        dst = np.float32([[150, 110], [2350, 110], [2350, 1510], [150, 1510]])
    elif condition == "rotated":
        dst = np.float32([[180, 150], [2320, 150], [2320, 1510], [180, 1510]])
        angle = float(rng.choice([-7.0, -4.0, 4.0, 7.0]))
        matrix = cv2.getRotationMatrix2D((1250, 850), angle, 1.0)
        dst = cv2.transform(dst.reshape(1, -1, 2), matrix).reshape(4, 2)
    elif condition in {"perspective_shadow", "shadow"}:
        inset = int(rng.integers(0, 80))
        dst = np.float32(
            [[150 + inset, 80], [2320, 140 + inset // 2], [2240 - inset // 2, 1560], [220, 1470 - inset]]
        )
    else:
        raise ValueError("condition must be clean, rotated, or perspective_shadow")
    transform = cv2.getPerspectiveTransform(src, dst)
    captured = cv2.warpPerspective(page, transform, (frame_w, frame_h), borderValue=(122, 122, 122))
    if condition in {"perspective_shadow", "shadow"}:
        yy, xx = np.mgrid[0:frame_h, 0:frame_w]
        shadow = np.exp(-(((xx - 1780) / 520) ** 2 + ((yy - 850) / 650) ** 2) / 2.0) * 0.22
        light = (1.0 - shadow[:, :, None]).astype(np.float32)
        captured = np.clip(captured.astype(np.float32) * light, 0, 255).astype(np.uint8)
        noise = rng.normal(0, 1.5, captured.shape).astype(np.float32)
        captured = np.clip(captured.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if condition != "clean":
        quality = jpeg_quality if condition == "perspective_shadow" else 94
        ok, encoded = cv2.imencode(".jpg", captured, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if ok:
            captured = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return captured, dst.tolist()


def load_leak_free_test_split():
    """Load only held-out test examples absent byte-for-byte from train/val."""
    from mlreader.emnist import duplicate_indices, load_split

    train_images, _ = load_split("train")
    val_images, _ = load_split("val")
    test_images, test_labels = load_split("test")
    excluded = set(duplicate_indices(train_images, test_images).tolist())
    excluded.update(duplicate_indices(val_images, test_images).tolist())
    keep = np.array([i for i in range(len(test_labels)) if i not in excluded], dtype=np.int64)
    return test_images[keep], test_labels[keep], keep, len(excluded)
