# Mileage log reader

## Project reference documents

- [Project overview and requirements](OVERVIEW.md), converted from the supplied final-project brief.
- [Team contract](TEAM_CONTRACT.md), converted from the signed team contract.

This repository contains a from-scratch prototype for reading the supplied
synthetic Sabal Coast Form ML-7. It accepts an image path at demo time,
aligns the printed template, locates the actual box borders, cleans small ink
artifacts, recovers border strokes and adjusts supported lighting problems,
classifies uppercase letters and
digits with an EMNIST-trained NumPy MLP, checks the assembled fields, and
routes each row to auto-post or clerk review. It does not send reimbursements
or connect to an ERP.

## Set up a clean clone

Use Python 3.10 or newer.

The commands below create a local virtual environment and use its Python for
every install and run command. The paths shown are for macOS/Linux; on Windows
replace .venv/bin/python with .venv\Scripts\python.exe.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_emnist.py
.venv/bin/python scripts/audit_emnist.py
```

For the frozen baseline, restore the saved `data/models/emnist_mlp.npz`
checkpoint and confirm its SHA-256 against [version_freeze.md](docs/version_freeze.md).
The checkpoint is Git-ignored and must be retained separately. A clean clone
without that saved file cannot reproduce the frozen scores exactly.

The training command creates a new checkpoint at that same path:

```bash
.venv/bin/python scripts/train_model.py
```

Preserve the frozen checkpoint before an explicitly authorized new training
experiment. Training is not required when the recorded checkpoint is available.

The data download is about 0.5 GB. It is kept under `data/`, which is ignored
by Git. The checksum file in `configs/` pins the five ByClass files used by the
loader. Training writes `data/models/emnist_mlp.npz`; measured model outputs
go under `outputs/model_eval/`.

## Run the browser demo

After the setup above, start the presentation interface with:

~~~bash
.venv/bin/python scripts/web_demo.py
~~~

Open the printed address in a browser, then choose **Take a photo** on a
supported mobile browser or **Choose a photo** to upload a JPEG or PNG. The
page shows registration, field reads and confidence, cell crops, business
checks, row routing, and the correction audit flow. It is a local prototype;
it does not post reimbursements.

To use a phone camera while the server runs on a laptop, connect both devices
to the same trusted Wi-Fi network and start the server on the laptop with:

~~~bash
.venv/bin/python scripts/web_demo.py --host 0.0.0.0
~~~

Open the laptop's Wi-Fi address printed by the server on the phone. The
camera button requests the rear camera where the browser supports it; the
file-picker button remains available as a fallback. Use synthetic or
made-up forms only. Uploaded images and corrections are stored under the
Git-ignored outputs/web_demo/ directory.

## Terminal fallback

After the setup above, this is the one command to try a new image:

```bash
.venv/bin/python scripts/demo.py /path/to/form_ml7_photo.jpg
```

The bundled clean sample demonstrates the known row-mile discrepancy from
the project brief:

```bash
.venv/bin/python scripts/demo.py examples/figure1_clean_scan.png
```

The phone-photo sample is available at `examples/figure2_phone_photo.jpg`.
Run it with `.venv/bin/python scripts/demo.py examples/figure2_phone_photo.jpg` to see
how the current prototype routes a difficult capture.
The default policy and reference records are explicitly synthetic and live in
`configs/reader_policy.json` and `configs/reference_data.json`. Unknown
employees, unscheduled clients, invalid dates, incomplete crops, low
confidence, and arithmetic inconsistencies route to review. The confidence
thresholds are configurable prototype values, not production approvals.

Each run writes a JSON result and cell crops under `outputs/demo/`. To record
a reviewer correction, use `--correction FIELD.PATH=VALUE --reason "..."`;
the original read, corrected value, reason, crop evidence, and validation
outcome are appended to `correction_audit.jsonl`.

## Reproduce the registration spike

```bash
.venv/bin/python scripts/run_spike.py
```

This creates 15 synthetic blank-form captures (clean, rotated, and
perspective/shadow), records field-box detection and registration results,
and updates `docs/spike_results.md`. It does not train a model or claim
handwriting accuracy.

## Generate and measure synthetic logs

```bash
.venv/bin/python scripts/generate_synthetic_logs.py --count 15 --quality mixed --fault-row 5
.venv/bin/python scripts/evaluate_synthetic_logs.py --data-dir outputs/synthetic_lighting_eval
```

By default, the generator draws characters only from the EMNIST ByClass test
split and excludes exact duplicates shared with the train or validation split.
It saves ground truth and source-image indices beside each generated log.
Evaluation reports raw character, field, and row accuracy, extraction and
recognition failures separately, and auto-post residual error by capture
condition.

The `--fault-row 5` option deliberately makes the written mileage in row 5
disagree with its odometers while keeping the weekly total aligned with the
written-mile column. To inspect one of those failures in the demo, run:

```bash
.venv/bin/python scripts/demo.py outputs/synthetic_logs/log_001_clean.jpg
```

To create separate **development** examples from the EMNIST validation split,
use a different output folder and score it separately:

```bash
.venv/bin/python scripts/generate_synthetic_logs.py \
  --count 15 --quality mixed --fault-row 5 --seed 7621 \
  --source-split val --output outputs/synthetic_dev
.venv/bin/python scripts/evaluate_synthetic_logs.py \
  --manifest outputs/synthetic_dev/manifest.json \
  --data-dir outputs/synthetic_dev_lighting_eval --as-of 2026-10-05
```

This makes 15 new logs, five per capture condition. The evaluator labels this
run as development data and omits the business-savings scenario. Use these
scores to compare explicitly authorized future reader versions; do not report
them as held-out test results. The current v3 reader is frozen.

## Pre-rendered synthetic QA forms

The separate `examples/synthetic_forms/` pack contains 15 valid image
variants from five underlying forms, plus three deliberately invalid images.
Its CSV has 108 labeled trip rows, and its manifest records image conditions,
seeds, related-form groups, and expected failed checks. See the folder's
README for case details. Score the pack separately with:

```bash
.venv/bin/python scripts/evaluate_synthetic_forms.py --output-dir outputs/synthetic_forms_lighting_eval
```

This uses the trained model checkpoint and writes detailed scores under
`outputs/synthetic_forms_lighting_eval/`. The fixtures are separate from the 15
command-generated logs scored above. The valid images are three paired views
of five forms, not 15 independent forms; the score report keeps them separate.

## Frozen baseline

The current reader is frozen at `ml7-cell-lighting-v3` as of 2026-10-06.
[Version freeze](docs/version_freeze.md) records the code revision, model and
configuration hashes, baseline scores and the plan for evaluating new forms.
Keep this version as the baseline for any explicitly authorized future work.

## Handwritten development forms

The three newly labeled handwritten forms in `examples/development_forms/`
provide a small development set for inspecting recognition errors and guiding
reader changes. Score them separately from the 12 held-out forms with:

```bash
.venv/bin/python scripts/evaluate_team_filled_forms.py \
  --forms-dir examples/development_forms \
  --ground-truth examples/development_forms/ground_truth.csv \
  --output-dir outputs/development_forms_lighting_eval \
  --dataset-name "handwritten development forms" \
  --as-of 2026-10-05
```

See that folder's README for the labels and a note about a mileage discrepancy
written on form 2. These development results are for iteration only. Do not
change `examples/team_filled_forms/` or tune against its labels. Those 12 forms
have been evaluated across several versions and now serve as a comparison
benchmark. A fresh generalization check needs new forms, labeled before
inspecting predictions and read with the frozen version. Current preprocessing keeps
the original EMNIST model weights and confidence thresholds. Unverified form
alignment, uncertain box borders, clipped writing, failed extraction, low
confidence, or failed business checks require review. Source crops and the
reason for review are available in the browser and CLI result JSON.
Border recovery preserves connected strokes near verified box edges and flags
every expanded crop for review. On the three development forms, character
accuracy rose from 89.8% to 91.3%, exact fields from 58/89 to 63/89, and exact
rows from 2/16 to 5/16. This uses the existing model and unchanged confidence thresholds.
Targeted cell lighting correction then raises character accuracy to 92.3%
(369/400) and exact fields to 65/89; exact rows remain 5/16. Adjusted ink also
requires review, with its measurements retained in the result JSON. The model
and its 28 × 28 normalization remain unchanged.

## Evaluate the team-filled forms

```bash
.venv/bin/python scripts/evaluate_team_filled_forms.py \
  --output-dir outputs/team_filled_lighting_eval --as-of 2026-10-05
```

This scores the 12 hand-filled photos in `examples/team_filled_forms/` against
`ground_truth.csv`. The labels preserve written leading zeroes; a blank leading
box in the form 5 total is scored as blank. The script does not train or tune
the model. It writes summary metrics, per-form scores, field errors, and
character confusion tables under the Git-ignored
`outputs/team_filled_lighting_eval/` directory. The current held-out result
is 69.5% character accuracy, 158/371 exact fields, and 0/67 exact trip rows;
all 67 rows require review. The earlier 17.7% baseline is preserved in the
results summary, along with the 52.6% cleanup and 54.1% border-recovery scores.
Keep these handwritten results separate from the synthetic-log evaluation above.

## Automated regression suite

```bash
.venv/bin/python -m unittest discover -s tests
```

It uses made-up reference data only. No real mileage logs, employee records,
expense forms, client codes, or patient information belong in this project.
See `docs/approach.md`, `docs/results_summary.md`, and `docs/business_note.md`
for the design and business interpretation. `docs/demo_run_of_show.md` has
the demo pacing. The private contribution-statement template and Christopher
Martin's draft are in `docs/contribution_statements/`.
