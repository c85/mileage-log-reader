# Mileage log reader

## Project reference documents

- [Project overview and requirements](OVERVIEW.md), converted from the supplied final-project brief.
- [Team contract](TEAM_CONTRACT.md), converted from the signed team contract.

This repository contains a from-scratch prototype for reading the supplied
synthetic Sabal Coast Form ML-7. It accepts an image path at demo time,
registers the page, crops each defined box, classifies uppercase letters and
digits with an EMNIST-trained NumPy MLP, checks the assembled fields, and
routes each row to auto-post or clerk review. It does not send reimbursements
or connect to an ERP.

## Set up a clean clone

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python scripts/download_emnist.py
python scripts/audit_emnist.py
python scripts/train_model.py
```

The data download is about 0.5 GB. It is kept under `data/`, which is ignored
by Git. The checksum file in `configs/` pins the five ByClass files used by the
loader. Training writes `data/models/emnist_mlp.npz`; measured model outputs
go under `outputs/model_eval/`.

## Run the demo

After the setup above, this is the one command to try a new image:

```bash
python scripts/demo.py /path/to/form_ml7_photo.jpg
```

The bundled clean sample demonstrates the known row-mile discrepancy from
the project brief:

```bash
python scripts/demo.py examples/figure1_clean_scan.png
```

The phone-photo sample is available at `examples/figure2_phone_photo.jpg`.
Run it with `python scripts/demo.py examples/figure2_phone_photo.jpg` to see
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
python scripts/run_spike.py
```

This creates 15 synthetic blank-form captures (clean, rotated, and
perspective/shadow), records field-box detection and registration results,
and updates `docs/spike_results.md`. It does not train a model or claim
handwriting accuracy.

## Generate and measure synthetic logs

```bash
python scripts/generate_synthetic_logs.py --count 15 --quality mixed --fault-row 5
python scripts/evaluate_synthetic_logs.py
```

The generator draws characters only from the EMNIST ByClass test split and
excludes exact duplicates shared with the train or validation split. It saves
ground truth and source-image indices beside each generated log. Evaluation
reports raw character, field, and row accuracy, extraction and recognition
failures separately, and auto-post residual error by capture condition.

The `--fault-row 5` option deliberately makes the written mileage in row 5
disagree with its odometers while keeping the weekly total aligned with the
written-mile column. To inspect one of those failures in the demo, run:

```bash
python scripts/demo.py outputs/synthetic_logs/log_001_clean.jpg
```

## Automated regression suite

```bash
python -m unittest discover -s tests
```

It uses made-up reference data only. No real mileage logs, employee records,
expense forms, client codes, or patient information belong in this project.
See `docs/approach.md`, `docs/results_summary.md`, and `docs/business_note.md`
for the design and business interpretation. `docs/demo_run_of_show.md` has
the demo pacing and `docs/contribution_statement_template.md` is a private
personal-statement template to complete in your own words.
