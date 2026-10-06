# Data record: EMNIST character data (SCRUM-10, SCRUM-11)

What data the model is trained and tested on, exactly how it is prepared, and
how to reproduce it. Each item names the one file that is the source of truth;
values are not copied elsewhere, so they cannot drift apart.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_emnist.py   # fetch + verify data
python scripts/audit_emnist.py      # re-check everything below, write outputs/emnist_audit/
```

## Source and version

- **Dataset:** EMNIST ByClass (Cohen et al., 2017), public, from NIST:
  `https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip`, binary (IDX) format.
- **Version:** pinned by SHA-256 checksums of the five files we use, in
  `configs/emnist_sha256.json`. `download_emnist.py` refuses to continue if a
  teammate's files differ.
- **Privacy:** public character images only. No real mileage logs, employee or
  patient data anywhere in the pipeline.

## Classes and label mapping

- Form ML-7 boxes hold only digits 0-9 and capital letters A-Z, so we keep
  ByClass labels 0-35 and **drop the 26 lowercase classes** (labels 36-61).
- Our label `i` is the character `CLASSES[i]`, with
  `CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"` (`mlreader/emnist.py`).
  This equals NIST's own mapping file, and `check_mapping()` asserts it on
  every load.

## Transforms (training data)

Applied in `load_split()` in `mlreader/emnist.py`, in this order:

1. Read the raw IDX files (own reader, `read_idx()`).
2. Drop lowercase classes.
3. **Transpose each 28x28 image**: EMNIST is stored with rows and columns
   swapped. `check_orientation()` asserts the result is upright on every load
   (evidence: `orientation_before_after.png`, attached to SCRUM-10).
4. For model training, apply the same threshold, 20x20 fit, and center-of-mass
   normalization used on extracted form cells (`normalize_emnist()` in
   `mlreader/registration.py`). Divide by 255 before the MLP forward pass.

Output from the loader is 28x28 `uint8`, white ink on black. Model training
and inference both apply the further normalization above.

## Splits

| Split | Source | How chosen |
|---|---|---|
| train | EMNIST train | everything not in val |
| val | EMNIST train | 10% of **each class**, random with seed `SPLIT_SEED = 6642` |
| test | EMNIST test | EMNIST's own test split, untouched |

Seed and fraction live in `mlreader/emnist.py` (`SPLIT_SEED`, `VAL_FRACTION`).
Stratifying by class keeps rare letters (e.g. K) represented in validation.
Measured sizes: train 480,595 / val 53,398 / test 89,264
(`outputs/emnist_audit/split_summary.json`, attached to SCRUM-10).
The audit found two exact test images also present in train, no test images
shared with validation, and no validation images shared with train. The
model's final test metrics and the log generator both exclude those two test
images.

**Synthetic test logs use characters from the test split only**, so no
character a model trained on can appear in a log it is scored on.

## Class imbalance and field restrictions

**Finding** (`class_counts.png`): digits are 64.6% of training images; the
most common class (1) has 15.6x the images of the rarest (K). On Form ML-7,
letters in employee IDs and client codes are close to uniform, so the
training distribution does not match the forms.

**Decision, two parts:**

1. **Restrict each box to its allowed set.** The form says which boxes hold
   digits and which hold letters, so at read time the model picks among the
   10 digits (labels 0-9) or the 26 letters (labels 10-35) only. This removes
   digit-vs-letter confusion (O/0, I/1, S/5, Z/2, B/8) but not confusion
   between characters in the same group.
2. **Inverse-frequency class weights in the training loss.**
   `class_weights(labels)` returns `N / (36 * count[c])` per class, computed
   from the selected training sample. The MLP applies these weights in its
   cross-entropy loss so rare classes contribute more than their raw frequency
   implies. Full-split values are in `class_weights.csv`.

**Not chosen:** oversampling rare letters (repeats the same images, slower
epochs). Fallback if per-letter validation accuracy shows rare letters (K, Q,
Z, ...) lagging despite the weights.

## Model and synthetic form test data

`scripts/train_model.py` trains a 784-input, one-hidden-layer NumPy MLP on
the train split, validates on the stratified train-derived validation split,
and reports on the untouched test split. Exact test images also present in
train or validation are excluded from the final test metrics.

The recorded run selected 90,000 training images from all 36 classes. On the
89,262 leak-free test images, unrestricted accuracy was 85.07%; using the
field's digit-versus-letter restriction raised it to 95.73%. Digits were
97.44% and capital letters 92.56% under that restriction. These values are
isolated-character metrics, not form-reading accuracy.

`scripts/generate_synthetic_logs.py` creates made-up trip fields, renders
characters from leak-free EMNIST test images into the blank Form ML-7 asset,
and saves the field strings and source indices beside each image. Those
characters are not used for training or model selection.

### Pre-rendered synthetic QA forms

`examples/synthetic_forms/` is a separate fixture pack: 15 valid image
variants represent five underlying forms (clean, rotated, and
perspective/shadow), and three more images show invalid date, schedule, and
odometer cases. The accompanying CSV contains 108 labeled rows; the manifest
records seeds, related-form groups, image conditions, and expected failed
checks. It identifies the characters as coming from the EMNIST ByClass test
subset. These fixtures are not included in the generated-log metrics in
`docs/results_summary.md`; count the valid data as five underlying forms, not
15 independent examples, and keep each group's variants together.

The folder's `reference_data.json` contains the two fictional employees and
their visit schedules used by all five valid form groups. The valid
ground-truth employee/date/client combinations match these schedules. The
invalid-date, invalid-schedule, and invalid-odometer controls deliberately
exercise their documented validation failures. The broader
`configs/reference_data.json` also contains unrelated fictional records used
elsewhere in the project.

## Leakage check

- `duplicate_indices()` finds images whose pixels exactly match across
  splits (hash, then byte-for-byte confirmation). The audit checks
  test-vs-train, test-vs-val and val-vs-train; results in
  `outputs/emnist_audit/leakage_check.json`.
- **Policy:** test images that also occur in train or val are never used in
  synthetic test logs or final test metrics. The generator records every
  source test index it used.

## Independent handwritten evaluation set

The 12 team-filled photos in `examples/team_filled_forms/` have made-up values
and hand-labeled fields in `ground_truth.csv` (67 trip rows). They are scored
separately from EMNIST and the generated-log test set with
`scripts/evaluate_team_filled_forms.py`. Fixed-width odometer and mileage
labels retain written leading zeroes; a blank leading box remains blank. The
sample is an initial check of this team's handwriting and phone captures, not
a representative population for threshold calibration.
