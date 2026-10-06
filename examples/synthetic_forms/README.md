# Synthetic mileage-log examples

This folder contains 18 computer-generated ML-7 forms and their correct answers.
All employee, client, trip and mileage records are made up. The characters come
from the public EMNIST ByClass test data. Photographed, hand-completed forms are
kept separately in `../team_filled_forms/`.

## Contents

- **18 PNG images:** 15 valid images and three deliberately invalid images.
- **`ground_truth.csv`:** correct written values for all 108 filled trip rows,
  using the same column names as the handwritten examples' answer file.
- **`manifest.json`:** image filenames, case details, random seeds, image
  conditions, related-form groups and expected business-check failures.
- **`reference_data.json`:** the fictional employee records and visit
  schedules used by all valid forms in this pack. The invalid schedule case
  deliberately uses the unscheduled client `XYZ`.

In the answer CSV, `FORM_ID` is the image filename without `.png`. Each form has
six filled rows; rows 7-9 are blank. Employee ID, week ending and total miles
repeat on each trip row. Read CSV columns as text to preserve leading zeros.
Intentional errors are recorded exactly as written on the images.

## Cases

The 15 valid images show five distinct forms, each in clean, rotated and
perspective/shadow conditions. Related versions share the same underlying form.

| Invalid image | Intentional change in row 1 | Expected result when read correctly |
|---|---|---|
| `invalid_date.png` | Date is 09/31 | Invalid date; the visit-schedule lookup also fails |
| `invalid_schedule.png` | Client is XYZ, which is not scheduled for that employee and date | Failed visit-schedule check |
| `invalid_odometer.png` | Ending odometer 068252 is below starting odometer 068253; written miles remain 037 | Invalid odometer order and mileage mismatch |

These invalid cases share `form_01_clean.png` as their valid control.

## Use for testing

Use this folder's reference records and a fixed evaluation date of
**2026-10-05**. The forms' week-ending date is **2026-09-27**. This keeps the
expected date checks consistent when testing later. For example, to inspect a
fixture with the reader, run:

```bash
.venv/bin/python scripts/demo.py examples/synthetic_forms/form_01_clean.png \
  --references examples/synthetic_forms/reference_data.json \
  --as-of 2026-10-05
```

Compare the reader's output with `ground_truth.csv`; use `manifest.json` for
expected business-check failures. Valid written values do not guarantee
automatic approval: confidence and the team's review policy still apply.
Keep these evaluation characters out of training, and keep related image
variants together when grouping results. Run
`.venv/bin/python scripts/evaluate_synthetic_forms.py` to score the pack; the
separate summary is in `docs/results_summary.md`.

These are the same 18 images as the complete dataset package. Their answers
have been arranged into one CSV. The fixture score is a small QA check over
five underlying valid forms, not a broad handwriting-accuracy estimate.

## Reproduce the images

The full reproduction package, including the script, saved character bank,
rendering support and original JSON answers, is attached to
[SCRUM-9](https://fiumsis.atlassian.net/browse/SCRUM-9) as
`SCRUM9_Synthetic_Data.zip`. Follow its included README to verify or regenerate
the images. The seed values in this folder identify the cases in that package.
