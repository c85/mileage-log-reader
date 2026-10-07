# SCRUM-24 QA matrix and recorded evidence

This folder contains the project’s QA test matrix and the supporting evidence
for its recorded execution log. The matrix defines expected outcomes, evidence
requirements, dependencies and actual results for QA-01 through QA-22.

## Start here

- [QA test matrix](QA_Test_Matrix.xlsx): the submitted workbook, including both tabs.
- [Executed QA notebook](MileageLogReader_QA.ipynb): the matching code and saved output.
- [Execution record](evidence/QA_Execution_Record.json): tested revision, executed steps and timestamps.
- [Visual review](evidence/QA_Visual_Review.json): recorded assessments of QA-02 and QA-04.
- [Synthetic form metrics](evidence/Synthetic_Form_Metrics.json): the 18-image evaluation.
- [Remaining QA-21 work](evidence/QA21_Status.json): the component and policy limitations.

## Version and scope

Tested reader revision: `ff674684e8834c1de4b0a105c773444605336d9d`.

The evidence records a run starting at 2026-10-06 03:55:59 UTC and ending at
2026-10-06 04:03:48 UTC. In America/New_York, that crosses midnight from
October 5 to October 6. The fixed business-validation date is 2026-10-05.
The matrix dates and saved evidence are preserved as supplied.

These results belong to that revision and its freshly trained model. They
do not certify a later reader revision or replace the repository's separate
frozen baseline and results summary. No new QA run was performed when this
upload package was assembled.

The matrix records 20 Passed cases, QA-21 Pending - component, and QA-22
Evaluated. Available QA-21 confidence and reimbursement calculations passed;
high-impact routing integration, calibration and final policy approval remained
pending under SCRUM-22 at the recorded revision.

QA-22 evaluated all 18 synthetic images and 108 trip rows: 2,591/2,664 correct
characters, 527/594 exact fields and 62/108 exact rows. Known-value validation
passed 18/18 cases separately from image recognition. These are sample results,
with no assumed accuracy pass threshold. Photographed handwritten forms were
not evaluated in this run. SCRUM-18 model evaluation remains separate.

## Evidence locations

| Original run location | Location in this folder |
| --- | --- |
| `outputs/qa/` | `evidence/` |
| `outputs/qa/synthetic_form_results/` | `evidence/synthetic_form_results/` |
| `outputs/qa/qa03_generated_logs/` | `evidence/qa03_generated_logs/` |
| `outputs/model_eval/` | `evidence/model_eval/` |
| Policy, public-data checksums and requirements | `evidence/run_configuration/` |
| Saved form answers, manifest and references | `evidence/fixture_metadata/` |

Evidence filenames in the workbook resolve under `evidence/`. The 18 individual
reader responses are under `evidence/synthetic_form_results/`. Configuration
and fixture metadata are historical snapshots, not replacements for live files.
Original paths inside JSON and notebook output describe the execution environment.

The 18 synthetic input images already exist in
[`examples/synthetic_forms`](../../examples/synthetic_forms/). Their bytes were
checked against the saved QA archive when this package was assembled. The
small answer/reference snapshots are included here to preserve the run context.

## Reproduction

Open the notebook in Google Colab and run the numbered cells in order in a
fresh session. Step 1 checks out the exact revision above in a separate working
folder. Step 20 trains the model; Step 21 evaluates the 18 saved images. Review
the orientation and character-crop images before accepting those visual checks.

The notebook already has recorded outputs; rerunning is not needed merely to
upload or review this evidence. A new run produces new observations. Training
in a different environment may produce different model weights and results.

The model hash used for this recorded run is
`a8fd14989fdb0040b2f476675c6e8e2de92ffda181a8174b852ef59518a67510`.
The original `QA_Evidence.zip` preserves that exact model checkpoint and
the complete run package. Retain it separately. Model weights and downloaded
datasets are omitted from this GitHub upload, consistent with the repository's
generated-artifact convention. The training record and environment versions
are included in `evidence/`.

This upload organizes the SCRUM-24 matrix and its existing evidence. A separate
SCRUM-27 results report is not included.
