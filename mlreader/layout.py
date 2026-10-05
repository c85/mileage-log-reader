"""Canonical Form ML-7 geometry and character-level field definitions."""

from dataclasses import dataclass


CANVAS_WIDTH = 2200
CANVAS_HEIGHT = 1400
LAYOUT_VERSION = "ml7-reference-2200x1400-v1"
CELL_MARGIN = 5
MAX_ROWS = 9


@dataclass(frozen=True)
class CellLocation:
    field: str
    position: int
    allowed: str  # "digit" or "letter"
    rect: tuple[int, int, int, int]  # left, top, right, bottom, canonical pixels
    row_number: int | None = None


def _cells(field, boxes, allowed, row_number=None):
    return [CellLocation(field, i, allowed, box, row_number) for i, box in enumerate(boxes)]


def _horizontal_cells(xs, y0, y1):
    return [(x0, y0, x1, y1) for x0, x1 in xs]


def _span_cells(x0, y0, width, height, count):
    return [
        (x0 + i * width, y0, x0 + (i + 1) * width, y0 + height)
        for i in range(count)
    ]


FIELDS: dict[str, list[CellLocation]] = {}
FIELDS["header.employee_id"] = [
    CellLocation(
        "header.employee_id",
        position,
        "letter" if position < 2 else "digit",
        rect,
    )
    for position, rect in enumerate(_span_cells(290, 190, 62, 62, 6))
]
FIELDS["header.week_ending"] = _cells(
    "header.week_ending",
    _horizontal_cells(
        [(1190, 1252), (1252, 1314), (1344, 1406), (1406, 1468), (1498, 1560), (1560, 1622)],
        190,
        252,
    ),
    "digit",
)
FIELDS["footer.total_miles"] = _cells(
    "footer.total_miles", _span_cells(1578, 1194, 62, 62, 4), "digit"
)

ROW_Y = [420, 506, 592, 678, 764, 850, 936, 1022, 1108]
_row_boxes = {
    "date": [(170, 232), (232, 294), (324, 386), (386, 448)],
    "client": [(520, 582), (582, 644), (644, 706)],
    "odometer_start": [(760 + i * 62, 822 + i * 62) for i in range(6)],
    "odometer_end": [(1200 + i * 62, 1262 + i * 62) for i in range(6)],
    "miles": [(1640 + i * 62, 1702 + i * 62) for i in range(3)],
}
_row_allowed = {
    "date": "digit",
    "client": "letter",
    "odometer_start": "digit",
    "odometer_end": "digit",
    "miles": "digit",
}
for _row, _y in enumerate(ROW_Y, start=1):
    for _field, _xs in _row_boxes.items():
        _key = f"rows.{_row}.{_field}"
        FIELDS[_key] = _cells(
            _key,
            _horizontal_cells(_xs, _y, _y + 62),
            _row_allowed[_field],
            _row,
        )

FIELD_LENGTHS = {key: len(value) for key, value in FIELDS.items()}
FIELD_KIND = {
    "header.employee_id": "employee_id",
    "header.week_ending": "week_ending",
    "footer.total_miles": "total_miles",
}
for _row in range(1, MAX_ROWS + 1):
    FIELD_KIND.update(
        {
            f"rows.{_row}.date": "trip_date",
            f"rows.{_row}.client": "client_code",
            f"rows.{_row}.odometer_start": "odometer",
            f"rows.{_row}.odometer_end": "odometer",
            f"rows.{_row}.miles": "miles",
        }
    )


def all_cells():
    """Yield every expected cell in stable document order."""
    yield from FIELDS["header.employee_id"]
    yield from FIELDS["header.week_ending"]
    for row in range(1, MAX_ROWS + 1):
        for field in ("date", "client", "odometer_start", "odometer_end", "miles"):
            yield from FIELDS[f"rows.{row}.{field}"]
    yield from FIELDS["footer.total_miles"]
