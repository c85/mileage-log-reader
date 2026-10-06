# Handwritten development forms

This folder contains three full-page hand-filled forms, converted to PNG:
`form_01.png`, `form_02.png` and `form_03.png`. All records are fictional.

`ground_truth.csv` contains manually transcribed answers for the header, each
filled trip row and the total miles. Fixed-width values preserve leading zeros
as written. There are 16 labeled trip rows across the three forms.
Form 2 row 5 is recorded as written: 278 miles, although the odometer change is
268 miles. This discrepancy is in the source form, not a transcription change.

Use this small set to inspect errors and guide reader changes. Keep it separate
from `../team_filled_forms/`, which remains the held-out evaluation set; do not
tune against those 12 forms. After development changes are settled, evaluate the
held-out set separately for a final comparison.

To score this development set, run:

```bash
.venv/bin/python scripts/evaluate_team_filled_forms.py \
  --forms-dir examples/development_forms \
  --ground-truth examples/development_forms/ground_truth.csv \
  --output-dir outputs/development_forms_eval \
  --as-of 2026-10-05
```

The results are development measurements only, not held-out performance.
