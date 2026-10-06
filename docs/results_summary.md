# Results summary

## Executive result

Printed-template alignment, local box detection and small-artifact cleanup
improved the 12-form handwritten score from 17.7% to 52.6% with the existing
EMNIST model. Exact fields increased from 11 to 76 out of 371, but no complete
trip row was read exactly and all 67 rows require review. On the generated
test set, the configured policy auto-posted 7 of 90 rows (7.8%) with no errors
among those 7; none of the 15 logs had every row eligible for auto-post. Under
the conservative assumption that AP still keys any log containing a review
row, the measured gross keying-cost saving is $0/month. This is a useful
fail-closed result, not a production savings claim.

Current form scores use `ml7-template-grid-cleanup-v1`, with validation as of
2026-10-05. Preprocessing was developed on the three new handwritten forms and
EMNIST-validation synthetic logs, then frozen before the held-out comparison.
The original model weights and policy thresholds were retained. A model's
high probability cannot override an unverified layout or a suspect crop.

## Registration spike

The SCRUM-12 spike evaluated 15 synthetic blank-form captures: five clean,
five rotated, and five with moderate perspective and shadow. All 15 pages
registered; all 214 mapped character boxes per sample passed the four-edge
grid check, and both odometer fields retained exactly six mapped cells. Mean
corner error was 1.0 pixels for clean, 1.1 for rotated, and 0.4 for
perspective/shadow. The full table and per-photo evidence are in
`docs/spike_results.md` and `outputs/spike_registration/results.json`.

This supports the tested blank-template capture conditions only. It does not
measure handwriting segmentation or recognition.

## Character classifier

The model is a NumPy MLP trained from scratch for 10 epochs on 90,000 EMNIST
ByClass training images (90,000 selected across all 36 classes). It uses
192 ReLU hidden units, batch size 256, learning rate 0.001, inverse-frequency
loss weights, and the same threshold/20 × 20/center-of-mass transform as
inference. The final test set contains 89,262 images after excluding two
exact duplicates shared with train or validation.

| Held-out isolated-character metric | Result |
|---|---:|
| Unrestricted 36-class accuracy | 85.07% |
| Unrestricted digit accuracy | 85.04% |
| Unrestricted capital-letter accuracy | 85.14% |
| Field-restricted accuracy | 95.73% |
| Field-restricted digit accuracy | 97.44% |
| Field-restricted capital-letter accuracy | 92.56% |
| Macro accuracy across 36 restricted classes | 93.88% |

The largest restricted test confusions were O→D (199), O→Q (112), U→V (107),
9→4 (79), and S→J (75). B, E, U, Y, and N were the lowest-accuracy classes.
Detailed results are in `outputs/model_eval/metrics.json` and
`outputs/model_eval/test_confusion.csv`.

## Synthetic end-to-end logs

The results in this section come from the command-generated set in
`outputs/synthetic_logs` (15 six-row logs, five per capture condition, with a
deliberate row-mile fault). They do not include the separate
`examples/synthetic_forms/` fixture pack, which has 15 image variants from
five valid underlying forms and three invalid controls. The fixture pack has
its own score in the next section; its repeated image conditions should not be
counted as independent forms or folded into the generated-log metrics below.

Fifteen six-row forms were rendered from leak-free EMNIST test characters,
with five logs per capture condition. Every log deliberately has row 5 miles
five miles away from its odometer difference; the total still matches the
written mileage column. The set contains 2,220 non-empty cells, 495 fields,
and 90 trip rows.

| Metric | Result |
|---|---:|
| Character accuracy across generated forms | 97.34% (2,161 / 2,220) |
| Digit accuracy | 97.71% (1,876 / 1,920) |
| Capital-letter accuracy | 95.00% (285 / 300) |
| Exact field accuracy | 89.70% (444 / 495) |
| Exact row accuracy | 57.78% (52 / 90) |
| Empty/unreadable cell crops among expected cells | 0 / 2,220 |
| Rows routed to auto-post candidate | 7 / 90 (7.78%) |
| Errors among auto-post candidates | 0 / 7 (0%; small synthetic sample) |
| Logs with every row auto-post eligible | 0 / 15 |

| Capture condition | Character accuracy | Exact row accuracy | Auto-post row share | Residual error among auto rows |
|---|---:|---:|---:|---:|
| Clean | 98.24% | 63.33% | 10.00% | 0 / 3 |
| Rotated | 96.76% | 56.67% | 0.00% | n/a |
| Perspective and shadow | 97.03% | 53.33% | 13.33% | 0 / 4 |

The most frequent exact-field errors were client codes and dates (12 of 15
logs had at least one of each), odometer starts (8), and odometer ends (7). A cell
crop with no usable ink is counted separately from a classifier read; these
generated boxes had no crop failures. The classifier still made wrong reads,
so `recognition_failures: 0` means it returned a class for every usable crop,
not that every read was correct. The earlier fixed-crop baseline scored
97.48% characters and 54/90 exact rows; the current changes traded a few
synthetic reads for substantially better handwritten cropping. The current
metrics are in `outputs/synthetic_preprocessed_eval/`.

## Pre-rendered synthetic QA fixtures

`scripts/evaluate_synthetic_forms.py` scored the separate 18-image fixture
pack using the existing model, policy, fixture reference records, and
2026-10-05 validation date. All 15 valid images registered and all 2,220
expected character cells were extracted. Across the 15 paired image variants,
character accuracy was 96.8% (2,150 / 2,220), exact-field accuracy was 87.3%
(432 / 495), and exact trip-row accuracy was 52.2% (47 / 90). Thirteen rows
were auto-post candidates, with no incorrect rows among those 13; this small,
controlled sample does not establish a safe error rate.

| Capture condition | Character accuracy | Exact trip rows |
|---|---:|---:|
| Clean | 97.0% (718 / 740) | 17 / 30 (56.7%) |
| Rotated | 96.6% (715 / 740) | 14 / 30 (46.7%) |
| Perspective and shadow | 96.9% (717 / 740) | 16 / 30 (53.3%) |

All three invalid controls registered. The reader detected 4 of the 5 expected
failed checks, with all expected failures found in 2 of the 3 controls. In the
invalid-odometer image it detected the mileage mismatch but missed the
ending-odometer-order failure: the starting value `068253` was read as
`062253`, making the predicted ending value appear larger. This shows the
limits of relying on arithmetic checks when recognition is wrong.

These are five underlying valid forms, each shown in three image conditions;
the 15 images are paired robustness views, not 15 independent forms. The
characters are assembled from EMNIST test images, so these results are a small
synthetic QA check and do not estimate performance on human handwriting.
They are reported separately from the generated-log and team-filled results.
Detailed metrics and error files are written under the Git-ignored
`outputs/synthetic_forms_preprocessed_eval/` directory. The original fixture
baseline was 97.1% characters and 50/90 exact rows.

## Synthetic development comparison

To support iteration without using the held-out test characters, a separate
set of 15 six-row logs was generated from the EMNIST validation split. Five
logs use each capture condition, and every log has the same deliberate row-5
mileage fault. Its initial baseline was 96.7% character accuracy
(2,147 / 2,220), 85.7% exact-field accuracy (424 / 495), and 41.1% exact-row
accuracy (37 / 90). The original policy run routed 3 of 90 rows to auto-post, with
0 observed errors among those 3.

This is a development score for comparing changes, not held-out performance
or a business-savings estimate. It does not use the team-filled forms. The
images, labels, and metrics are reproducible under the Git-ignored
`outputs/synthetic_dev/` and `outputs/synthetic_dev_eval/` directories using
the development command in `README.md`. With current preprocessing, the same
batch scores 96.6% characters (2,145/2,220), 85.3% fields (422/495) and 40.0%
rows (36/90), with 4 auto-post candidates and no observed errors among those
4. Current metrics are in `outputs/synthetic_dev_preprocessed_eval/`.

## Handwritten development forms

Three additional handwritten forms provide 16 trip rows and 400 character
labels. The initial fixed-crop baseline scored 35/400 characters (8.8%), no
exact fields or rows, and 11 invented blank rows. Development inspection
showed that accepting a page boundary did not align the printed character
boxes: printing offsets and local box displacement contaminated the crops.

Current preprocessing scores 359/400 characters (89.8%), 58/89 exact fields
(65.2%) and 2/16 exact rows (12.5%). All 400 expected cells are extracted,
all three template alignments are verified, and no blank row is treated as
filled. Fifty-five labeled cells carry border-verification or clipping
warnings; their predictions are retained for review. All 16 rows require
review under the image-quality, confidence and business checks. The invented
dates and employee records also affect business routing, so this result is
not a standalone calibration of review safety.

These are development measurements only. Labels preserve the source writing,
including form 2 row 5's `278` miles although its odometer difference is `268`.
Detailed current errors are in `outputs/development_forms_preprocessed_eval/`.

Before the cropping fix, an output-layer adaptation trial scored 18.0% on
the 12 held-out forms versus 17.7% originally, while exact fields fell from
11 to 7. That historical trial was not adopted. Baselines used the old
preprocessing at revision `61e9e4a`; rerunning adaptation with current crops
would be a new experiment. The current results use the original checkpoint.

## Team-filled handwritten forms

The independent set has 12 photographed forms, 67 trip rows, and 1,665
handwritten characters. Ground truth preserves the six-digit odometers; form
5's total is `172` with its first box blank, which is scored as a blank cell.
The frozen preprocessing and original model were run without corrections.
These 12 forms were not used to tune the preprocessing changes.

| Metric | Original fixed crops | Current preprocessing |
|---|---:|---:|
| Page transforms accepted | 12 / 12 | 12 / 12 |
| Printed template alignment verified | Not measured | 8 / 12 |
| Expected character cells extracted | 1,478 / 1,665 (88.8%) | 1,510 / 1,665 (90.7%) |
| Character accuracy, all labels | 295 / 1,665 (17.7%) | 876 / 1,665 (52.6%) |
| Character accuracy on extracted cells | 20.0% | 58.0% |
| Digit accuracy | 18.7% | 771 / 1,440 (53.5%) |
| Capital-letter accuracy | 11.6% | 105 / 225 (46.7%) |
| Exact fields | 11 / 371 (3.0%) | 76 / 371 (20.5%) |
| Exact trip rows | 0 / 67 | 0 / 67 |
| Unlabeled blank rows treated as active | 36 | 12 |

All 67 labeled rows require review. Four template alignments are unverified,
155 expected cells are not extracted successfully, and there are 573 crop
warnings across the detected fields, including extra rows. These issues block
automatic routing even when a character has high model probability. Common
usable-cell confusions include 0→6 (62), 0→2 (39) and 9→4 (37). This remains
a small sample of team handwriting and captures, insufficient for production
threshold calibration. Detailed current metrics are in
`outputs/team_filled_preprocessed_eval/`.

The supplied figures are synthetic examples too. Use their current raw
predictions, source crops and review reasons in the demo; coffee rings,
shadows, crowded writing and uncertain borders remain important failure
cases. Registration success alone does not establish readable cells.

## Reproduction and limits

After setup and dataset generation in `README.md`, use the original model:

```bash
.venv/bin/python scripts/evaluate_synthetic_logs.py --data-dir outputs/synthetic_preprocessed_eval
.venv/bin/python scripts/evaluate_synthetic_forms.py --output-dir outputs/synthetic_forms_preprocessed_eval
.venv/bin/python scripts/evaluate_team_filled_forms.py --output-dir outputs/team_filled_preprocessed_eval --as-of 2026-10-05
.venv/bin/python -m unittest discover -s tests
```

Generated logs use held-out EMNIST test images rather than train images, but
they still share EMNIST's centered stroke style. The field-restricted result
is not a phone-photo or handwritten-form accuracy claim. Thresholds are
prototype settings, and the 0/7 residual result is too small to establish a
safe error rate. The 12-form handwritten evaluation is a small initial check;
more writers and capture conditions are needed before calibrating or raising
the straight-through rate. The regression suite has 24 passing tests covering
crop cleanup and preservation, printing offsets, box counts, failed layout
verification, high-confidence uncertain crops and reviewed corrections. The
browser warning display and escaping were also checked. The existing model
SHA-256 is `9da5b69c57476cae69a266553c3b1d39c35cd0ae1306716ed586270a92d3dfa9`.
