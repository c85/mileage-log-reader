"""Form registration, fixed-grid cell extraction, and EMNIST normalization."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mlreader.layout import CANVAS_HEIGHT, CANVAS_WIDTH, CELL_MARGIN, FIELDS


@dataclass
class RegistrationResult:
    ok: bool
    image: np.ndarray | None
    method: str
    corners: list[list[float]] | None
    reason: str | None = None


@dataclass
class CellCrop:
    field: str
    position: int
    allowed: str
    row_number: int | None
    rect: tuple[int, int, int, int]
    image: np.ndarray
    normalized: np.ndarray | None
    extraction_status: str
    ink_pixels: int
    extraction_reason: str | None = None


def load_image(source):
    """Load an image path or BGR/RGB array as a BGR NumPy image."""
    if isinstance(source, np.ndarray):
        image = source
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("Image array must be grayscale, BGR, RGB, or BGRA.")
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _order_quad(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right
    ordered[1] = points[np.argmin(diffs)]  # top-right
    ordered[3] = points[np.argmax(diffs)]  # bottom-left
    return ordered


def _page_quad(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(gray.shape[0] * gray.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.20:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = _order_quad(approx)
        top = np.linalg.norm(quad[1] - quad[0])
        bottom = np.linalg.norm(quad[2] - quad[3])
        left = np.linalg.norm(quad[3] - quad[0])
        right = np.linalg.norm(quad[2] - quad[1])
        width = (top + bottom) / 2
        height = (left + right) / 2
        ratio = width / max(height, 1)
        if 1.25 <= ratio <= 1.9:
            return quad
    return None


def register_page(source):
    """Find a paper boundary and warp it to the supplied ML-7 template size.

    An aspect-ratio fallback supports flatbed scans that have already been
    cropped to the page. Phone images without a recoverable page boundary are
    rejected so they can be reviewed instead of silently mis-cropped.
    """
    image = load_image(source)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quad = _page_quad(gray)
    target = np.float32(
        [[0, 0], [CANVAS_WIDTH - 1, 0], [CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1], [0, CANVAS_HEIGHT - 1]]
    )
    if quad is not None:
        transform = cv2.getPerspectiveTransform(quad, target)
        warped = cv2.warpPerspective(
            image,
            transform,
            (CANVAS_WIDTH, CANVAS_HEIGHT),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return RegistrationResult(True, warped, "paper-boundary-perspective", quad.tolist())

    input_ratio = width / max(height, 1)
    target_ratio = CANVAS_WIDTH / CANVAS_HEIGHT
    if abs(input_ratio - target_ratio) <= 0.035:
        resized = cv2.resize(image, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_AREA)
        corners = [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]
        return RegistrationResult(True, resized, "aspect-ratio-flat-scan", corners)

    return RegistrationResult(
        False,
        None,
        "rejected",
        None,
        "Could not locate a page quadrilateral and the image aspect ratio does not match Form ML-7.",
    )


def normalize_cell(cell):
    """Convert a dark-ink cell crop into centered, white-on-black 28x28 data."""
    if cell is None or cell.size == 0:
        return None, "empty", 0, "Cell crop has no image pixels."
    if cell.ndim == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell
    # Coordinates are inset from the printed grid, so the remaining dark
    # pixels should be handwriting. Fixed thresholding is repeatable at inference.
    ink = (gray < 165).astype(np.uint8) * 255
    ys, xs = np.where(ink > 0)
    count = int(len(xs))
    if count < 5:
        return None, "empty", count, "No handwriting pixels found inside the cell."
    if count > int(ink.size * 0.55):
        return None, "unreadable", count, "Most of the cell is dark; crop or lighting is not supported."
    return center_ink(ink), "ok", count, None


def center_ink(ink):
    """Scale an ink mask into 20x20 and center its mass in a 28x28 frame."""
    ink = np.asarray(ink, dtype=np.uint8)
    ys, xs = np.where(ink > 0)
    if not len(xs):
        return np.zeros((28, 28), dtype=np.uint8)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    glyph = ink[y0:y1, x0:x1]
    h, w = glyph.shape
    scale = 20 / max(w, h)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    glyph = cv2.resize(glyph, (resized_w, resized_h), interpolation=interpolation)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x = (28 - resized_w) // 2
    y = (28 - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = glyph

    # Center of mass, matching the conventional EMNIST/MNIST character frame.
    moment = cv2.moments(canvas)
    if moment["m00"] > 0:
        cx = moment["m10"] / moment["m00"]
        cy = moment["m01"] / moment["m00"]
        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (28, 28), borderValue=0)
    return canvas


def normalize_emnist(images):
    """Apply the same 20x20, center-of-mass normalization used at inference."""
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Expected (N, 28, 28) EMNIST images, got {images.shape}.")
    output = np.zeros(images.shape, dtype=np.uint8)
    for index, image in enumerate(images):
        # In a dark-ink/light-paper crop, the inference threshold (<165)
        # corresponds to EMNIST intensities above 90 after inversion.
        ink = (image >= 90).astype(np.uint8) * 255
        output[index] = center_ink(ink)
    return output


def extract_cells(registered_image):
    """Crop all known character boxes from the canonical ML-7 canvas."""
    if registered_image.shape[1] != CANVAS_WIDTH or registered_image.shape[0] != CANVAS_HEIGHT:
        raise ValueError(
            f"Expected registered canvas {CANVAS_WIDTH}x{CANVAS_HEIGHT}; got "
            f"{registered_image.shape[1]}x{registered_image.shape[0]}."
        )
    crops = {}
    for field, locations in FIELDS.items():
        cells = []
        for location in locations:
            left, top, right, bottom = location.rect
            margin = CELL_MARGIN
            crop = registered_image[top + margin : bottom - margin, left + margin : right - margin]
            normalized, status, ink_pixels, reason = normalize_cell(crop)
            cells.append(
                CellCrop(
                    field=field,
                    position=location.position,
                    allowed=location.allowed,
                    row_number=location.row_number,
                    rect=location.rect,
                    image=crop,
                    normalized=normalized,
                    extraction_status=status,
                    ink_pixels=ink_pixels,
                    extraction_reason=reason,
                )
            )
        crops[field] = cells
    return crops
