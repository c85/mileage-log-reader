# Handwritten development forms

This folder contains three full-page hand-filled forms, converted to PNG:
`form_01.png`, `form_02.png` and `form_03.png`. All records are fictional.

`ground_truth.csv` contains manually transcribed answers for the header, each
filled trip row and the total miles. Fixed-width values preserve leading zeros
as written. There are 16 labeled trip rows across the three forms.
Form 2 row 5 is recorded as written: 278 miles, although the odometer change is
268 miles. This discrepancy is in the source form, not a transcription change.

Use this small set to inspect errors and guide reader changes. Keep it separate
from `../team_filled_forms/`; do not tune against those 12 forms. They have
already been evaluated across versions and now serve as a comparison benchmark.
The current v3 reader is frozen. Use new forms from different writers for a
fresh evaluation, following `docs/version_freeze.md`.

To score this development set, run:

```bash
.venv/bin/python scripts/evaluate_team_filled_forms.py \
  --forms-dir examples/development_forms \
  --ground-truth examples/development_forms/ground_truth.csv \
  --output-dir outputs/development_forms_lighting_eval \
  --dataset-name "handwritten development forms" \
  --as-of 2026-10-05
```

The results are development measurements only, not held-out performance.
Printed-template alignment, local box detection and small-artifact cleanup
raised character accuracy from 35/400 (8.8%) to 359/400 (89.8%) using the same
EMNIST checkpoint. Border-stroke recovery then raised accuracy to 365/400
(91.3%), exact fields from 58/89 to 63/89, and exact trip rows from 2/16 to 5/16.
No blank rows were treated as filled, compared with 11 in the initial baseline.
All 16 rows still require review under the image-quality and business checks.
The v2 preprocessing (`ml7-border-recovery-v2`) expanded 259 labeled
cell crops; every applied recovery requires source verification. There are
280 cells with preprocessing warnings, compared with 55 before recovery.

Current preprocessing is `ml7-cell-lighting-v3`. Targeted faint-ink and
lighting correction raises the score to 369/400 characters (92.3%) and 65/89
exact fields (73.0%); exact trip rows remain 5/16. All expected cells are
extracted, with no extra blank rows. There are 99 lighting-adjusted cells and
318 cells with preprocessing warnings. Adjustments require source verification,
and all 16 rows still need review. Model weights and routing thresholds remain
unchanged.

The earlier output-layer adaptation experiment used the old fixed crops and
did not reliably improve the held-out result. It is recorded in
`docs/results_summary.md`; the current reader uses the original model.
