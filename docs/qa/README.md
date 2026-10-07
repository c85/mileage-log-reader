# Frozen v3 QA package

This package records the **2026-10-07** QA run of `ml7-cell-lighting-v3`.
Reader code, model weights and policy match the frozen baseline. The earlier
run remains in [archive/ff674684](archive/ff674684/README.md).

## Start here

- [QA matrix and execution log](QA_Test_Matrix.xlsx).
- [Executed notebook](MileageLogReader_QA.ipynb).
- [Execution record, source hashes and data checksums](evidence/QA_Execution_Record.json).
- [Visual inspection](evidence/QA_Visual_Review.json).
- [Frozen-checkpoint evaluation](evidence/Model_Evaluation_Run.json).
- [18-image SCRUM-9 results](evidence/Synthetic_Form_Metrics.json).
- [36-image SCRUM-26 results](evidence/QA_Fixture_Metrics.json).
- [36-case known-value validation](evidence/QA_Fixture_Known_Value_Check.json).
- [QA-21 acceptance scope](evidence/QA21_Status.json).
- [Regression test log](evidence/Regression_Test_Log.txt).

## Version and acceptance scope

| Item | Recorded identity |
| --- | --- |
| Frozen reader implementation | `5e95a2cbdbe7d86d32d3aa9d88d3db77d7a9db2c` |
| Source/fixture snapshot | `208a29981c65ce8121302a5fc05ee9e326aa2f44` |
| Preprocessing | `ml7-cell-lighting-v3` |
| Model SHA-256 | `9da5b69c57476cae69a266553c3b1d39c35cd0ae1306716ed586270a92d3dfa9` |
| Business-validation date | `2026-10-05` |
| Training during QA | None |

The snapshot contains the fixture uploads; its reader, assets and configurations
match the frozen implementation. The new notebook and runner are QA harness
changes in the working tree. The execution record preserves the notebook input
hash, reader source hashes, data checksums, environment, timestamps and executed
steps. Saved notebook outputs belong to this run.

QA-01 through QA-21 passed their stated prototype checks. QA-22 is **Evaluated**:
it reports measurements without assuming an accuracy pass threshold. All 41
regression tests passed. Codex performed the automated run and inspected the
QA-02/QA-04 images. A separate team QA sign-off has not been recorded.

QA-21 matches [the frozen routing policy](../routing_policy.md): minimum
confidence 0.82 ordinarily and 0.92 for each odometer's three highest-place
digits. All 21 boundary variants and six standalone cost checks pass.
The script's $100 exposure example is a sensitivity scenario, not a reader
routing requirement. Passing these checks does not establish confidence
calibration or production payment approval.

## Measurements

| Dataset | Correct characters | Exact fields | Exact trip rows | Auto-post candidates |
| --- | ---: | ---: | ---: | ---: |
| SCRUM-9: 18 images, including 3 invalid controls | 2,590/2,664 (97.22%) | 526/594 | 60/108 | 15 |
| SCRUM-26: 36 images, including 21 invalid controls | 5,222/5,328 (98.01%) | 1,089/1,188 | 139/216 | 26 |
| Shared 15 valid images: five forms in three conditions | 2,152/2,220 (96.94%) | 433/495 | 47/90 | 8 |

Known-value validation passes 18/18 and 36/36 cases respectively. These checks
supply written answers at confidence 0.99 and do not measure recognition.
Image reading registered every image with no missing expected cells in these
packs. The 18-image run has 74 recognition mismatches; the 36-image run has 106.
An intended source defect can be misread into different checks while the row
still goes to review; inspect each case outcome.

Neither pack auto-posted an invalid-source row or an extra predicted row.
The 36-image report additionally checks header, week, footer and trip-row
transcription together: zero errors among 26 accepted candidates. These small
samples do not establish a safe population error rate.

**Eighteen images overlap between the packs.** Do not sum their sample counts.
Paired variants and invalid controls repeat the same underlying handwriting.
The higher expanded-pack score reflects its case mix; the reader did not
improve. Conditions and valid/invalid subsets are reported separately.
The shared valid-image scores match [the frozen results summary](../results_summary.md).
This is a repeated synthetic benchmark, not a fresh handwriting-generalization test.

## Reproduce locally

Follow the repository README setup and restore the exact frozen model. Then run:

```bash
.venv/bin/python scripts/run_qa.py --output-dir outputs/qa_v3_reproduction
```

Use a new output directory each time. The runner verifies the model and all
five EMNIST checksums, checks the reader against the frozen revision, and
executes notebook Steps 3–22 in an isolated copy. It saves new evidence and an
executed notebook. It does not train or overwrite the model, repository evidence
or recorded baseline outputs. Inspect the QA-02/QA-04 images before accepting
those visual checks. Colab-only setup/upload and archive-download cells are
skipped locally and identified in the execution record.

For Colab, run the notebook in order in a fresh session. Step 1 checks out the
recorded source/fixture snapshot; Step 2 installs dependencies, verifies public
data and asks for the saved checkpoint. Step 20 evaluates it without training.
Step 22 reads all 36 fixtures using each case's reference file, including the
special unknown-employee references.

The SCRUM-26 manifest differs from the older evaluator's schema. Use this
notebook/runner for the 36-image pack; `scripts/evaluate_synthetic_forms.py`
still targets the 18-image pack.

## Evidence and archived run

Workbook evidence filenames resolve under `evidence/`. Individual results are
in `evidence/synthetic_form_results/` and `evidence/qa_fixture_results/`.
Configuration and fixture metadata are snapshots. Temporary paths in saved
responses identify the execution environment; input images remain in the
repository's examples folders with hashes recorded in the evidence.

The checkpoint and public data remain Git-ignored. Preserve the checkpoint
separately; retraining creates a new artifact. The historical run used revision
`ff674684` and checkpoint hash beginning `a8fd1498`. Its results and pending
policy statements belong to that run. Its original notebook, workbook and
evidence remain under `archive/ff674684/`.
