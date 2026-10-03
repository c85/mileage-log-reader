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

Output: 28x28 `uint8`, white ink on black, character centered, as EMNIST
provides it. Any further scaling (e.g. dividing by 255) belongs to the model's
training code and must be mirrored at inference.

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

**Synthetic test logs use characters from the test split only**, so no
character a model trained on can appear in a log it is scored on.

## Class imbalance: decision

**Finding** (`class_counts.png`): digits are 64.6% of training images; the
most common class (1) has 15.6x the images of the rarest (K). On Form ML-7,
letters in employee IDs and client codes are close to uniform, so the
training distribution does not match the forms.

**Decision, two parts:**

1. **Restrict each box to its allowed set.** The form says which boxes hold
   digits and which hold letters, so at read time the model picks among the
   10 digits (labels 0-9) or the 26 letters (labels 10-35) only. This removes
   digit-vs-letter confusion (O/0, I/1, S/5, Z/2, B/8) and most of the effect
   of the digit-heavy imbalance. *(Model-side decision: to be confirmed with the
   model owner, Christopher.)*
2. **Inverse-frequency class weights in the training loss.**
   `class_weights(labels)` returns `N / (36 * count[c])` per class, computed
   from the training split. Each class then contributes equally to the loss,
   so rare letters are not under-learned. Values: `class_weights.csv`.

**Not chosen:** oversampling rare letters (repeats the same images, slower
epochs). Fallback if per-letter validation accuracy shows rare letters (K, Q,
Z, ...) lagging despite the weights.

## Leakage check

- `duplicate_indices()` finds images whose pixels exactly match across
  splits (hash, then byte-for-byte confirmation). The audit checks
  test-vs-train, test-vs-val and val-vs-train; results in
  `outputs/emnist_audit/leakage_check.json`.
- **Policy:** test images that also occur in train or val are never used in
  synthetic test logs. The SCRUM-9 generator enforces this and records, for
  every generated log, which test images it used.

## Open items

- **Inference preprocessing** (photo cell → EMNIST-style 28x28) must match the
  training format above exactly. Owner: pipeline/model, with data. Track in
  SCRUM-13/14.
- **Synthetic-log leakage** is enforced once the SCRUM-9 generator exists.
