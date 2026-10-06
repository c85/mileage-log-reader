# Results summary

## Executive result

The prototype registers the tested synthetic page captures and learns
EMNIST characters, but the independent team-filled set shows a large gap on
actual handwriting: no complete trip row was read exactly. On the generated
test set, the configured policy auto-posted 7 of 90 rows (7.8%) with no errors
among those 7; none of the 15 logs had every row eligible for auto-post. Under
the conservative assumption that AP still keys any log containing a review
row, the measured gross keying-cost saving is $0/month. This is a useful
fail-closed result, not a production savings claim.

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
| Character accuracy across generated forms | 97.48% (2,164 / 2,220) |
| Digit accuracy | 97.92% (1,880 / 1,920) |
| Capital-letter accuracy | 94.67% (284 / 300) |
| Exact field accuracy | 90.10% (446 / 495) |
| Exact row accuracy | 60.00% (54 / 90) |
| Empty/unreadable cell crops among expected cells | 0 / 2,220 |
| Rows routed to auto-post candidate | 7 / 90 (7.78%) |
| Errors among auto-post candidates | 0 / 7 (0%; small synthetic sample) |
| Logs with every row auto-post eligible | 0 / 15 |

| Capture condition | Character accuracy | Exact row accuracy | Auto-post row share | Residual error among auto rows |
|---|---:|---:|---:|---:|
| Clean | 98.24% | 63.33% | 10.00% | 0 / 3 |
| Rotated | 97.16% | 63.33% | 0.00% | n/a |
| Perspective and shadow | 97.03% | 53.33% | 13.33% | 0 / 4 |

The most frequent exact-field errors were client codes (14 of 15 logs had at
least one), dates (9), odometer starts (8), and odometer ends (6). A cell
crop with no usable ink is counted separately from a classifier read; these
generated boxes had no crop failures. The classifier still made wrong reads,
so `recognition_failures: 0` means it returned a class for every usable crop,
not that every read was correct.

## Pre-rendered synthetic QA fixtures

`scripts/evaluate_synthetic_forms.py` scored the separate 18-image fixture
pack using the existing model, policy, fixture reference records, and
2026-10-05 validation date. All 15 valid images registered and all 2,220
expected character cells were extracted. Across the 15 paired image variants,
character accuracy was 97.1% (2,155 / 2,220), exact-field accuracy was 88.3%
(437 / 495), and exact trip-row accuracy was 55.6% (50 / 90). Twelve rows
were auto-post candidates, with no incorrect rows among those 12; this small,
controlled sample does not establish a safe error rate.

| Capture condition | Character accuracy | Exact trip rows |
|---|---:|---:|
| Clean | 97.0% (718 / 740) | 17 / 30 (56.7%) |
| Rotated | 97.3% (720 / 740) | 17 / 30 (56.7%) |
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
`outputs/synthetic_forms_eval/` directory.

## Team-filled handwritten forms

The independent set has 12 photographed forms, 67 trip rows, and 1,665
handwritten characters. Ground truth preserves the six-digit odometers; form
5's total is `172` with its first box blank, which is scored as a blank cell.
The fixed model and policy were run without corrections or tuning on this set.

| Metric | Result |
|---|---:|
| Page transforms accepted by the reader | 12 / 12 |
| Expected character cells extracted | 1,478 / 1,665 (88.8%) |
| Character accuracy, all labeled characters | 295 / 1,665 (17.7%) |
| Character accuracy on extracted cells | 295 / 1,478 (20.0%) |
| Digit accuracy | 269 / 1,440 (18.7%) |
| Capital-letter accuracy | 26 / 225 (11.6%) |
| Exact fields | 11 / 371 (3.0%) |
| Exact trip rows | 0 / 67 |

The reader marked 36 unlabeled blank rows as active, and all 67 labeled rows
were routed to review under the current confidence and business rules, with
validation evaluated as of 2026-10-05. Among usable crops, common confusions
included 0→5 (65) and 0→6 (61). The accepted page transform alone did not
ensure useful crops. These results show that the
synthetic character and form metrics do not transfer to this handwriting and
capture set. The sample meets the adjusted 12-form target, but it represents
only the team's made-up entries and is too small to calibrate production
thresholds.

The bundled clean scan (`examples/figure1_clean_scan.png`) is a harder
handwriting-domain example. In a live run, the model read employee ID
RC5107 as RC6107 and week ending 09/27/26 as 07/27/26. It also misread
multiple odometer digits, including row 5's end value. All seven rows were
sent to review even though there were no empty crops. This is the failure the
demo should show: segmentation succeeded, while recognition and the business
checks did not support a safe post.

The supplied field-condition phone photo
(`examples/figure2_phone_photo.jpg`) registers to the page, but the coffee
ring and uneven shadow cause blank lower rows to look occupied. The run had
nine cell-extraction failures and no model-output failures; all rows were
routed to review. This capture is outside the successful synthetic blank-page
spike conditions and demonstrates why registration success alone does not
mean a form is readable.

## Reproduction and limits

```bash
python scripts/run_spike.py
python scripts/train_model.py
python scripts/generate_synthetic_logs.py --count 15 --quality mixed --fault-row 5
python scripts/evaluate_synthetic_logs.py
python scripts/evaluate_synthetic_forms.py
python scripts/evaluate_team_filled_forms.py --as-of 2026-10-05
```

Generated logs use held-out EMNIST test images rather than train images, but
they still share EMNIST's centered stroke style. The field-restricted result
is not a phone-photo or handwritten-form accuracy claim. Thresholds are
prototype settings, and the 0/7 residual result is too small to establish a
safe error rate. The 12-form handwritten evaluation is a small initial check;
more writers and capture conditions are needed before calibrating or raising
the straight-through rate.
