# Frozen reader baseline

Status: **frozen at the user's request on 2026-10-06**.

## Version identity

- Reader code revision: `5e95a2cbdbe7d86d32d3aa9d88d3db77d7a9db2c`
  (`Enhance preprocessing with lighting correction and update documentation`).
- Preprocessing: `ml7-cell-lighting-v3`.
- Model: the existing NumPy EMNIST MLP, `data/models/emnist_mlp.npz`.
- Policy: `configs/reader_policy.json`, `prototype-conservative`.
- Default reference records: `configs/reference_data.json`.
- Recorded evaluation validation date: **2026-10-05**.

The code revision identifies the implementation, dependencies, tracked source
forms, labels and fixture reference records. This note records the freeze;
it does not change recognition behavior. The checkpoint and detailed generated
outputs remain Git-ignored, so retain the checkpoint matching the hash below.

| Artifact | SHA-256 |
|---|---|
| Model checkpoint | `9da5b69c57476cae69a266553c3b1d39c35cd0ae1306716ed586270a92d3dfa9` |
| Reader policy | `fd167744a3bddd1e606f011e129e93ea4b6164926fd9d7f581c6437bf2f5fcb2` |
| Default reference records | `f3e2ceae1b546310a403ada70128c62979fbff75378b9711f45578e096399928` |

## Recorded baseline results

| Dataset | Character accuracy | Exact fields | Exact trip rows |
|---|---:|---:|---:|
| Three handwritten development forms | 369/400 (92.25%) | 65/89 | 5/16 |
| Twelve team-filled evaluation forms | 1,157/1,665 (69.49%) | 158/371 | 0/67 |
| Generated synthetic development logs | 2,143/2,220 (96.53%) | 421/495 | 36/90 |
| Generated synthetic test logs | 2,159/2,220 (97.25%) | 440/495 | 50/90 |
| Synthetic QA valid image variants | 2,152/2,220 (96.94%) | 433/495 | 47/90 |

All 67 team-form rows require review. Generated synthetic test logs have
3 auto-post candidates, with zero observed incorrect candidates and no
fully automatic logs. The QA set has 8 candidates with zero observed errors;
its three invalid controls expose 4 of 5 expected failed checks. These small
samples do not establish a safe production error rate. The last implementation
test run passed **41 tests**; recording this freeze did not rerun them.

Detailed metrics are in the corresponding `outputs/*_lighting_eval/`
directories. The comparisons, capture variations and limits are in
[results_summary.md](results_summary.md).

## Evaluation after the freeze

The freeze covers model weights, registration, cell extraction, lighting and
border recovery, normalization, confidence policy and business validation.
Keep this version as the comparison baseline. Future recognition or policy
changes require an explicit decision to start a new version.

The three development forms guided the implementation. The twelve team forms
were evaluated repeatedly across versions, though the latest lighting rule
was settled before its evaluation. Treat their score as a benchmark; use
new forms from different writers for a fresh generalization check.

For that check, label the new forms from their source images before inspecting
reader predictions, run the frozen reader and report its uncorrected results.
Keep raw predictions separate from human corrections. If those results guide
a later change, that batch becomes development data for that later version;
a further unseen batch is needed for a fresh evaluation.
