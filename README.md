# SCRUM-26 controlled synthetic QA fixtures

This folder contains all 36 computer-generated ML-7 test images and their correct
written answers. All employee, client, trip and mileage records are fictional.
The 12 logical cases each have clean, rotated and perspective/shadow versions.
There are 15 valid images, 21 intentionally invalid images and 216 filled trip
rows. Rows 7–9 are blank.

## Contents

- 36 PNG images: the complete SCRUM-26 test set, kept together for convenience.
- `ground_truth.csv`: correct written values for all 216 trip rows. Read columns
  as text to preserve leading zeros. Intentional errors remain as written.
- `manifest.json`: filenames, seeds, conditions, related-case groups, intended
  changes, expected validation flags and expected row/document routes.
- `reference_data.json`: fictional HR records and schedules for ordinary cases.
- `reference_data_invalid_employee.json`: references for the three unknown-employee
  images only. Its schedule includes ZZ9999 while its HR list excludes ZZ9999,
  isolating the employee-lookup failure.
- `reader_policy.json`: the saved prototype settings behind the expected routes.

In the CSV, FORM_ID is the image filename without .png. WEEK_ENDING uses MM/DD/YY;
trip dates use MMDD. Header and total values repeat on each filled row. Every
manifest path is relative to this folder, and each case names its reference file.

## Coverage

| Case group | Intended defect | Expected review rows with correctly read fields |
| --- | --- | --- |
| valid_form_01 through valid_form_05 | None | None under the controlled assumptions below |
| invalid_date | Row 1 has September 31; date and dependent schedule checks fail | Row 1 |
| invalid_schedule | Row 1 names unscheduled client XYZ | Row 1 |
| invalid_odometer | Row 1 ends below its start; mileage equality also fails | Row 1 |
| invalid_employee | ZZ9999 has no HR entry | All six rows |
| invalid_continuity | Row 2 starts below the previous end | Row 2 |
| invalid_row_miles | Row 1 written miles disagree with odometer difference | Row 1 |
| invalid_weekly_total | Footer disagrees with the sum of written miles | All six rows |

Every error group uses valid_form_01 as its control. Single-fault cases have one
intended defect; dependent failures are recorded explicitly. The continuity case
preserves row mileage, and the row-mile case adjusts the footer to isolate its
intended check.

## Use for testing

Use the fixed validation date **2026-10-05** and the supplied reference and policy
files. The form week ends on **2026-09-27**. Compare the reader's output with the
CSV and manifest. The unknown-employee cases must use their special reference
file; all other cases use reference_data.json.

Expected routes assume correctly read, complete fields, character confidence
0.99, successful extraction and no preprocessing warnings. Under those conditions,
valid rows are auto_post candidates; specified failures force needs_review.
A review row makes the document needs_review. These are controlled expectations,
not a claim that the reader recognizes every image correctly. Actual recognition,
confidence and image warnings can change the result. No classifier was trained
or evaluated as part of SCRUM-26.

The original package records known-value validation and pixel reproduction for
all 36 fixtures. Its validator revision is
`1f1d9a9df45b38361f946da3b76e4bc1ffd1d110`; see manifest provenance for source
revisions and hashes. This GitHub packaging step does not constitute a new reader
run or certify a later reader revision.

## Relationship to SCRUM-9 and training data

Eighteen images also exist in ../synthetic_forms/. They are retained here so the
complete SCRUM-26 set can be used from one folder. SCRUM-26 adds degraded versions
of the original three error cases and four new error groups.

All images and their character sources are QA-only. Do not train or tune the model
on these images or their crops. Related variants are not independent samples:
keep each manifest leakage_group together, including the form-01 control and all
of its error variants. The saved test-character bank excludes previously identified
exact duplicate indices 75979 and 83597. The prior full-data duplicate audit was
reused, not repeated; this does not establish absence of near-duplicates.

## Full reproduction package

The complete package is attached to [SCRUM-26](https://fiumsis.atlassian.net/browse/SCRUM-26)
as `SCRUM26_QA_Fixtures.zip`. It contains per-image source-character records, the
saved character bank, scripts, pinned dependencies, source snapshots, checksums
and the detailed verification record. Use its README to verify or regenerate
the fixtures. Those reproduction materials are retained in Jira, while this
folder contains ready-to-use test inputs and expected results.
