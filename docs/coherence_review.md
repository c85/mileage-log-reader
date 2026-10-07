# Repository coherence and deliverables review

Reviewed on **2026-10-07**, starting from repository revision `208a299` after
fetching origin. The checkout matched the remote branch and initially had no
tracked changes. This review also includes the authorized frozen-v3 QA refresh
made in the working tree.

## Assessment

The implementation and main business recommendation are coherent. The latest
pushes added documentation, QA evidence and fixtures; they did not change the
frozen reader. Current baseline evaluations reproduce the published scores.
The main remaining work concerns submission packaging, backup demonstration
evidence, and access to the exact saved checkpoint.

## Alignment with OVERVIEW.md

| Requirement | Repository evidence | Assessment / remaining work |
| --- | --- | --- |
| Approach document, 3–4 pages | [approach.md](approach.md) | Covers pipeline, data, model/fallback, costs, evaluation, scope and risks. Page compliance is unverified; the earlier DOCX was deleted. Prepare a paginated submission. |
| Spike that changes the plan | [spike_results.md](spike_results.md) | Has 15 cases, conditions, measurements and a GO decision. Explicitly state the original assumption and what changed in the execution plan; the current text does not clearly record that change. |
| Repository runnable from a clean clone | [README](../README.md), own loader/training loop and interfaces | Documented setup downloads public data and trains a model. Exact frozen scores additionally need the separately retained checkpoint. A fresh dependency install/download/training run was not repeated during this review. |
| New image at demo time, deliberate failure | [web demo](../scripts/web_demo.py), [CLI](../scripts/demo.py), [run of show](demo_run_of_show.md) | Upload/path inputs and known-failure examples exist. Rehearse the actual unseen-image live run on the presentation machine. |
| Results summary, 2 pages | [results_summary.md](results_summary.md) | Metrics, conditions, confusion and failure modes are present and baseline numbers reproduce. At roughly 3,016 words, this is a detailed technical record; prepare a concise two-page submission. |
| Business note, 1 page | [business_note.md](business_note.md) | Covers human-review recommendation, $34,000/month keying baseline, hypothetical savings and error costs. Roughly 491 words; verify the final page layout. |
| Backup recording | [demo_run_of_show.md](demo_run_of_show.md) | Explicitly says a narrated video is not in the repository. Record a rehearsal or identify an existing external recording. |
| Private individual statements | [contribution statements](contribution_statements/) | One draft and a template are present. Each person still needs their private submission; others' private statements need not be added to Git. |
| Team contract and sprint workbook | [TEAM_CONTRACT.md](../TEAM_CONTRACT.md) | Roles and coordination are documented. Jira is identified as the tracking source; a sprint workbook or submission link was not found locally. External completion was not verified. |
| Own neural network; no prohibited reader | [model training](../scripts/train_model.py), [AI disclosure](ai_assistance.md) | From-scratch NumPy MLP, own extraction/normalization/validation and generator. No prohibited OCR/document/LLM reader dependency found in runtime code. |
| Made-up business records only | Fixture provenance, handwritten set documentation | Repository documentation identifies the records as fictional. The review did not independently investigate how every hand-filled image was collected. |

## Verification

The frozen checkpoint, policy and default references match all three hashes in
[version_freeze.md](version_freeze.md). Reader code, assets and configurations
remain unchanged from that baseline. Dependency consistency passes and the
41-test regression suite passes.

Fresh evaluation outputs are under the Git-ignored
`outputs/coherence_review_20261007/` directory, with validation fixed at
2026-10-05:

| Dataset | Reproduced character score | Exact fields | Exact rows |
| --- | ---: | ---: | ---: |
| 12 photographed team forms | 1,157/1,665 (69.49%) | 158/371 | 0/67 |
| 15 generated test logs | 2,159/2,220 (97.25%) | 440/495 | 50/90 |
| 15 valid synthetic image variants | 2,152/2,220 (96.94%) | 433/495 | 47/90 |

All 67 labeled handwritten rows require review. Generated logs have three
auto-post candidates, zero observed row-transcription errors among those
three, and no fully automatic logs. The repeated handwriting benchmark and
small synthetic accepted-row counts do not establish production readiness.

## QA gaps addressed in this update

The uploaded SCRUM-24 evidence tested revision `ff674684` with a different
checkpoint. It was labeled honestly as historical, but could not certify v3.
It is now preserved under [qa/archive/ff674684](qa/archive/ff674684/README.md).
The primary [QA package](qa/README.md) contains a newly executed v3 notebook,
updated workbook, exact checkpoint identity, environment/source hashes and
current results. No model training or reader changes were made.

The original notebook referenced a removed rendering function and an old
generator metadata key. Those calls now match the frozen implementation.
The newer 36-image manifest also differs from the generic evaluator's schema;
the dedicated [QA runner](../scripts/run_qa.py) consumes it through the notebook
and applies each case's reference file. The older generic evaluator remains
the command for the 18-image pack.

All 36 new fixture image hashes match their manifest. Known-value validation
passes all 36 cases. Actual image reading reports 5,222/5,328 characters,
1,089/1,188 exact fields, 139/216 exact rows and 26 auto-post candidates, with
zero invalid-source rows or payment-context transcription errors accepted.
Eighteen images overlap the older pack, and all variants repeat underlying
forms; these counts must not be combined or treated as new independent writers.

QA-21 now checks the documented 0.82/0.92 prototype confidence policy. Its
21 confidence variants and six standalone cost checks pass. The $100 exposure
example is not an operational routing requirement. This scope correction does
not calibrate confidence or approve payments. Codex's automated run and visual
inspection are recorded separately from team QA sign-off, which is not recorded.

## Priorities before submission

1. Prepare and verify the required page-limited approach, results and business
   documents; preserve these technical Markdown records as supporting evidence.
2. Retain/share the exact frozen checkpoint with a clear restore location and
   rehearse setup and the unseen-image demo on another machine.
3. Record the backup demonstration, explicitly document the spike's plan change,
   and identify the sprint-workbook submission artifact.
4. Have the team QA owner review the refreshed package, and complete each
   member's private contribution submission.
