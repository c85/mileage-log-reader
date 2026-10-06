"""Form registration, fixed-grid cell extraction, and EMNIST normalization."""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from mlreader import REPO_ROOT
from mlreader.layout import CANVAS_HEIGHT, CANVAS_WIDTH, CELL_MARGIN, FIELDS

PREPROCESSING_VERSION = "ml7-cell-lighting-v3"
BORDER_RECOVERY_MARGIN = 2
CROP_BOUNDARY_WARNING = "Writing touches the crop boundary and may be clipped."
BORDER_RECOVERY_WARNING = "Expanded crop requires verification against the source."
LIGHTING_WARNING = "Lighting-adjusted ink requires verification against the source."


@dataclass
class RegistrationResult:
    ok: bool
    image: np.ndarray | None
    method: str
    corners: list[list[float]] | None
    reason: str | None = None
    template_alignment: dict | None = None


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
    preprocessing_issues: tuple[str, ...] = ()
    crop_rect: tuple[int, int, int, int] | None = None
    border_recovery: dict | None = None
    lighting_adjustment: dict | None = None


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


@lru_cache(maxsize=1)
def _template_features():
    template = cv2.imread(str(REPO_ROOT / "assets/form_ml7_blank.png"), cv2.IMREAD_GRAYSCALE)
    if template is None:
        return (), None
    keypoints, descriptors = cv2.ORB_create(nfeatures=6000).detectAndCompute(template, None)
    return keypoints, descriptors


def _align_template(image):
    """Align the printed ML-7 features, independently of the handwritten answers."""
    template_points, template_descriptors = _template_features()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points, descriptors = cv2.ORB_create(nfeatures=6000).detectAndCompute(gray, None)
    details = {"verified": False, "applied": False, "matches": 0, "inliers": 0}
    if template_descriptors is None or descriptors is None:
        details["reason"] = "Printed form features could not be matched to the ML-7 template."
        return image, details
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(template_descriptors, descriptors, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
    details["matches"] = len(matches)
    if len(matches) < 25:
        details["reason"] = "Too few printed form features matched the ML-7 template."
        return image, details
    source = np.float32([points[match.trainIdx].pt for match in matches])
    target = np.float32([template_points[match.queryIdx].pt for match in matches])
    transform, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if transform is None or mask is None or not np.isfinite(transform).all():
        details["reason"] = "Printed form alignment could not be estimated reliably."
        return image, details
    accepted = mask.ravel().astype(bool)
    details["inliers"] = int(accepted.sum())
    coverage = cv2.contourArea(cv2.convexHull(target[accepted])) / (CANVAS_WIDTH * CANVAS_HEIGHT)
    corners = np.float32([[0, 0], [CANVAS_WIDTH - 1, 0], [CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1], [0, CANVAS_HEIGHT - 1]])
    aligned_corners = cv2.perspectiveTransform(corners[None], transform)[0]
    displacement = float(np.linalg.norm(aligned_corners - corners, axis=1).max())
    details["maximum_corner_displacement"] = displacement
    details["matched_template_area_share"] = float(coverage)
    spans = np.ptp(target[accepted], axis=0)
    if (
        details["inliers"] < 25
        or accepted.mean() < 0.45
        or coverage < 0.05
        or spans[0] < CANVAS_WIDTH * 0.45
        or spans[1] < CANVAS_HEIGHT * 0.1
        or not np.isfinite(aligned_corners).all()
        or displacement > CANVAS_WIDTH * 0.1
        or np.linalg.det(transform[:2, :2]) <= 0
        or not cv2.isContourConvex(aligned_corners)
    ):
        details["reason"] = "Printed form alignment failed the coverage or geometry checks."
        return image, details
    details.update(verified=True, reason=None, transform=transform.tolist())
    # Preserve an already aligned scan rather than resampling its characters.
    if displacement <= 1.0:
        return image, details
    details["applied"] = True
    aligned = cv2.warpPerspective(
        image, transform, (CANVAS_WIDTH, CANVAS_HEIGHT),
        flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255),
    )
    return aligned, details


def _registered_result(image, method, corners):
    image, alignment = _align_template(image)
    if alignment["applied"]:
        method += "+printed-template"
    return RegistrationResult(True, image, method, corners, template_alignment=alignment)


def register_page(source):
    """Find a paper boundary and warp it to the supplied ML-7 template size.

    Printed template features then correct offsets caused by printing margins.
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
        return _registered_result(warped, "paper-boundary-perspective", quad.tolist())

    input_ratio = width / max(height, 1)
    target_ratio = CANVAS_WIDTH / CANVAS_HEIGHT
    if abs(input_ratio - target_ratio) <= 0.035:
        resized = cv2.resize(image, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_AREA)
        corners = [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]
        return _registered_result(resized, "aspect-ratio-flat-scan", corners)

    return RegistrationResult(
        False,
        None,
        "rejected",
        None,
        "Could not locate a page quadrilateral and the image aspect ratio does not match Form ML-7.",
    )


def _clean_cell_ink(gray):
    """Keep handwriting while removing isolated specks and tiny frame fragments."""
    binary = (gray < 165).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ink = np.zeros_like(binary)
    height, width = binary.shape
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        touches_edge = x == 0 or y == 0 or x + component_width == width or y + component_height == height
        if area < 3 or (touches_edge and area < 30 and min(component_width, component_height) <= 2):
            continue
        ink[labels == index] = 255
    return ink


def _prepare_cell_gray(gray):
    """Flatten paper illumination and recover faint ink only when warranted.

    This chooses a mask from image measurements, without model predictions.
    The raw source is kept separately; the model's 28x28 transform is unchanged.
    """
    if gray.size == 0 or min(gray.shape) < 21:
        return gray, None
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    paper_low, paper_high = np.percentile(background, [5, 95])
    if paper_low < 60:
        return gray, None  # Too dark to infer the paper reliably.
    corrected = np.clip(
        np.rint(gray.astype(np.float32) * 255 / np.maximum(background, 1)), 0, 255,
    ).astype(np.uint8)
    foreground = corrected[corrected < 225]
    faint = len(foreground) >= 25 and np.percentile(foreground, 25) >= 100
    uneven = paper_low < 225 or paper_high - paper_low >= 20
    if not faint and not uneven:
        return gray, None
    threshold = 185 if faint else 165
    prepared = np.where(corrected < threshold, 0, 255).astype(np.uint8)
    original = _clean_cell_ink(gray)
    candidate = _clean_cell_ink(prepared)
    original_pixels = int(np.count_nonzero(original))
    candidate_pixels = int(np.count_nonzero(candidate))
    changed = int(np.count_nonzero((original > 0) != (candidate > 0)))
    if changed < max(5, original_pixels * 0.03) or candidate_pixels > candidate.size * 0.55:
        return gray, None
    if original_pixels < 5 and candidate_pixels:
        ys, xs = np.where(candidate > 0)
        if candidate_pixels < 25 or np.ptp(ys) < 8 or np.ptp(xs) < 2:
            return gray, None  # Do not turn a few pale specks into a character.
    return prepared, {
        "applied": True,
        "method": "local-paper-background",
        "background_kernel": 21,
        "paper_intensity_p05": float(paper_low),
        "paper_intensity_p95": float(paper_high),
        "faint_ink": bool(faint),
        "uneven_lighting": bool(uneven),
        "relative_ink_threshold": threshold,
        "changed_ink_pixels": changed,
    }


def normalize_cell(cell):
    """Convert a dark-ink cell crop into centered, white-on-black 28x28 data."""
    if cell is None or cell.size == 0:
        return None, "unreadable", 0, "Cell crop has no image pixels."
    if cell.ndim == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell
    # Retain the ink threshold and 28x28 transform used by the existing model.
    ink = _clean_cell_ink(gray)
    ys, xs = np.where(ink > 0)
    count = int(len(xs))
    if count < 5:
        return None, "empty", count, "No handwriting pixels found inside the cell."
    if count > int(ink.size * 0.55):
        return None, "unreadable", count, "Most of the cell is dark; crop or lighting is not supported."
    return center_ink(ink), "ok", count, None


def _recover_border_ink(image, cell):
    """Recover connected strokes inside a verified box without changing the model.

    Keep the original cleaned ink as a seed. Only new ink connected to that
    seed can be retained; disconnected frame fragments and neighboring marks
    cannot create a character. Long printed lines are removed only in the
    newly exposed margin. Every applied recovery requires human verification.
    """
    if cell.extraction_status != "ok" or any(
        issue not in (CROP_BOUNDARY_WARNING, LIGHTING_WARNING) for issue in cell.preprocessing_issues
    ):
        return cell
    left, top, right, bottom = cell.rect
    margin = BORDER_RECOVERY_MARGIN
    crop_rect = (left + margin, top + margin, right - margin, bottom - margin)
    x0, y0, x1, y1 = crop_rect
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        return cell
    crop = image[y0:y1, x0:x1]
    gray, lighting = _prepare_cell_gray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    ink = (gray < 165).astype(np.uint8) * 255
    height, width = ink.shape
    band = CELL_MARGIN - margin
    old_gray = cv2.cvtColor(cell.image, cv2.COLOR_BGR2GRAY)
    if band <= 0 or old_gray.shape != (height - 2 * band, width - 2 * band):
        return cell
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((1, max(20, int(width * 0.8))), np.uint8),
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((max(20, int(height * 0.8)), 1), np.uint8),
    )
    lines = np.zeros_like(ink)
    lines[:band] = horizontal[:band]
    lines[-band:] = horizontal[-band:]
    lines[:, :band] |= vertical[:, :band]
    lines[:, -band:] |= vertical[:, -band:]
    cleaned_gray = gray.copy()
    cleaned_gray[lines > 0] = 255
    cleaned = _clean_cell_ink(cleaned_gray)
    seed = np.zeros_like(cleaned)
    old_gray, _ = _prepare_cell_gray(old_gray)
    seed[band:-band, band:-band] = _clean_cell_ink(old_gray)
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8), connectivity=8,
    )
    retained = np.zeros_like(cleaned)
    for index in range(1, count):
        component = labels == index
        if np.any(component & (seed > 0)):
            retained[component] = 255
    outside = retained.copy()
    outside[band:-band, band:-band] = 0
    added = int(np.count_nonzero(outside))
    if added < 5:
        return cell
    pixels = int(np.count_nonzero(retained))
    if np.any((seed > 0) & (retained == 0)) or pixels > int(retained.size * 0.55):
        # Keep the original prediction and flag a recovery that cannot safely
        # preserve its ink. Model confidence is never used to select a crop.
        return replace(cell, preprocessing_issues=tuple(dict.fromkeys(
            cell.preprocessing_issues + ("Border ink could not be recovered reliably.",)
        )))
    return replace(
        cell, image=crop, normalized=center_ink(retained), ink_pixels=pixels,
        crop_rect=crop_rect,
        lighting_adjustment=lighting or cell.lighting_adjustment,
        border_recovery={
            "applied": True,
            "original_crop_rect": list(cell.crop_rect or (
                left + CELL_MARGIN, top + CELL_MARGIN,
                right - CELL_MARGIN, bottom - CELL_MARGIN,
            )),
            "added_ink_pixels": added,
            "removed_frame_pixels": int(np.count_nonzero(lines)),
        },
        preprocessing_issues=tuple(dict.fromkeys(
            cell.preprocessing_issues + (BORDER_RECOVERY_WARNING,)
            + ((LIGHTING_WARNING,) if lighting else ())
        )),
    )


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


def _printed_grid(gray):
    # Local thresholding recovers pale box lines even under uneven illumination.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10,
    )
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, 5), dtype=np.uint8))
    horizontal = cv2.morphologyEx(joined, cv2.MORPH_OPEN, np.ones((1, 40), dtype=np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((40, 1), dtype=np.uint8))
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours], vertical


def _refine_group(locations, segments, vertical):
    """Locate a printed group of equally spaced boxes near its template location."""
    left, top, _, bottom = locations[0].rect
    right = locations[-1].rect[2]
    count = len(locations)
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    nearby = [
        (x, y, width, height) for x, y, width, height in segments
        if height <= 12 and count * 62 * 0.45 <= width <= count * 67
        and abs(x + width / 2 - center_x) < 110
        and abs(y + height / 2 - center_y) < 85
    ]
    candidates = []
    for upper in nearby:
        for lower in nearby:
            ux, uy, uw, uh = upper
            lx, ly, lw, lh = lower
            upper_y, lower_y = uy + uh / 2, ly + lh / 2
            if not 57 <= lower_y - upper_y <= 68:
                continue
            full_x, _, full_width, _ = upper if uw >= lw else lower
            if full_width < count * 57 or abs(full_x + full_width / 2 - center_x) > 55:
                continue
            overlap = max(0, min(ux + uw, lx + lw) - max(ux, lx))
            if overlap < min(uw, lw) * 0.85 or abs((upper_y + lower_y) / 2 - center_y) > 45:
                continue
            y0, y1 = int(round(upper_y)), int(round(lower_y))
            x1 = full_x + full_width - 1
            cost = (
                abs((full_x + x1) / 2 - center_x) + abs((y0 + y1) / 2 - center_y)
                + abs(x1 - full_x - count * 62) + 2 * abs(y1 - y0 - 62)
            )
            candidates.append((cost, full_x, y0, x1, y1))
    if not candidates:
        return [location.rect for location in locations], ("Printed field frame could not be located.",)
    _, x0, y0, x1, y1 = min(candidates)
    edges = np.rint(np.linspace(x0, x1, count + 1)).astype(int)
    rects = [(int(edges[i]), y0, int(edges[i + 1]), y1) for i in range(count)]
    support = [
        float((vertical[y0 + 3:y1 - 3, max(0, x - 3):x + 4] > 0).sum(axis=0).max()) / max(y1 - y0 - 6, 1)
        for x in edges
    ]
    issues = () if min(support) >= 0.4 and np.mean(support) >= 0.6 else ("Printed cell borders could not all be verified.",)
    nominal = [location.rect for location in locations]
    if np.max(np.abs(np.asarray(rects) - np.asarray(nominal))) <= 2:
        rects = nominal
    return rects, issues


def extract_cells(registered_image):
    """Crop all known character boxes from the canonical ML-7 canvas."""
    if registered_image.shape[1] != CANVAS_WIDTH or registered_image.shape[0] != CANVAS_HEIGHT:
        raise ValueError(
            f"Expected registered canvas {CANVAS_WIDTH}x{CANVAS_HEIGHT}; got "
            f"{registered_image.shape[1]}x{registered_image.shape[0]}."
        )
    gray = cv2.cvtColor(registered_image, cv2.COLOR_BGR2GRAY)
    segments, vertical = _printed_grid(gray)
    crops = {}
    for field, locations in FIELDS.items():
        cells = []
        groups = []
        for location in locations:
            if groups and groups[-1][-1].rect[2] == location.rect[0]:
                groups[-1].append(location)
            else:
                groups.append([location])
        refined = []
        for group in groups:
            rects, issues = _refine_group(group, segments, vertical)
            refined.extend((location, rect, issues) for location, rect in zip(group, rects))
        for location, rect, issues in refined:
            left, top, right, bottom = rect
            margin = CELL_MARGIN
            crop = registered_image[top + margin : bottom - margin, left + margin : right - margin]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else np.empty((0, 0), np.uint8)
            prepared, lighting = _prepare_cell_gray(crop_gray) if not issues else (crop_gray, None)
            normalized, status, ink_pixels, reason = normalize_cell(prepared)
            if lighting:
                issues += (LIGHTING_WARNING,)
            if status == "ok":
                cleaned = _clean_cell_ink(prepared)
                edge_pixels = int(np.count_nonzero(cleaned[0]) + np.count_nonzero(cleaned[-1])
                                  + np.count_nonzero(cleaned[:, 0]) + np.count_nonzero(cleaned[:, -1]))
                if edge_pixels >= max(5, ink_pixels * 0.05):
                    issues += (CROP_BOUNDARY_WARNING,)
            cells.append(
                _recover_border_ink(registered_image, CellCrop(
                    field=field,
                    position=location.position,
                    allowed=location.allowed,
                    row_number=location.row_number,
                    rect=rect,
                    image=crop,
                    normalized=normalized,
                    extraction_status=status,
                    ink_pixels=ink_pixels,
                    extraction_reason=reason,
                    preprocessing_issues=issues,
                    crop_rect=(left + margin, top + margin, right - margin, bottom - margin),
                    lighting_adjustment=lighting,
                ))
            )
        crops[field] = cells
    return crops
