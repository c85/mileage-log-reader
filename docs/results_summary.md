# Mileage log reader results summary

We recommend human review of every reimbursement. Frozen `ml7-cell-lighting-v3`
reads 69.49% of characters on 12 photographed forms, with 0/67 exact trip rows;
all require review. Generated logs yield three candidates out of 90 rows and
no fully automatic log. With whole-log handling at $3.40 per log, measured
gross keying savings are $0/month. All business records are fictional.

The reader was frozen on October 6, 2026. All routing comparisons below use
the October 5, 2026 validation date, the unchanged EMNIST checkpoint and the
0.82 ordinary / 0.92 high-place odometer confidence thresholds.

## Isolated character classifier

The from-scratch 784-192-ReLU-36 NumPy MLP trained for ten epochs on 90,000
EMNIST ByClass images, using class weights, batch size 256 and learning rate
0.001. It excludes lowercase and shares the 20 x 20/28 x 28 normalization
with inference. Test data contains 89,262 images after two duplicate exclusions.

| Isolated EMNIST test group | Unrestricted accuracy | Field-restricted accuracy |
| --- | ---: | ---: |
| All 36 classes | 85.07% | 95.73% |
| Digits | 85.04% | 97.44% |
| Capital letters | 85.14% | 92.56% |

Restrictions use the known field type; these are isolated-character scores.
Largest confusions: O to D (199), O to Q (112), U to V (107), 9 to 4 (79), S to J (75).

## End to end measurements

| Dataset and conditions | Character accuracy | Exact fields | Exact trip rows |
| --- | ---: | ---: | ---: |
| 12 photographed team forms | 1,157/1,665 (69.49%) | 158/371 (42.59%) | 0/67 (0%) |
| 15 generated test logs, mixed captures | 2,159/2,220 (97.25%) | 440/495 (88.89%) | 50/90 (55.56%) |
| Five valid QA forms in three paired captures | 2,152/2,220 (96.94%) | 433/495 (87.47%) | 47/90 (52.22%) |
| Three handwritten development forms | 369/400 (92.25%) | 65/89 (73.03%) | 5/16 (31.25%) |

Generated logs use leak-free EMNIST test characters and a deliberate row-5
mileage fault. QA variants reuse five forms; development forms guided
preprocessing and are not held-out results. Digit/letter scores are
70.14%/65.33% on photographed forms, 97.60%/95.00% on generated logs and
97.24%/95.00% on valid QA variants.

| Generated test capture | Character accuracy | Exact rows | Auto-post candidates |
| --- | ---: | ---: | ---: |
| Clean, five logs | 98.24% | 19/30 (63.33%) | 3/30 |
| Rotated, five logs | 96.76% | 17/30 (56.67%) | 0/30 |
| Perspective and shadow, five logs | 96.76% | 14/30 (46.67%) | 0/30 |

## Extraction and failure modes

The registration spike accepted 15/15 synthetic blank captures and all 214
mapped boxes per capture; it did not test handwriting recognition. On the
photographed benchmark, 12/12 pages registered but only eight had verified
template alignment. Extraction succeeded for 1,634/1,665 expected characters
(98.14%); 31 were unavailable. Recognition on usable cells was 70.81%, showing
that successful cropping alone does not ensure a correct read. Eight extra
rows were detected. Generated logs and valid QA variants had no missing crops.

Photographed-form exact fields were employee ID 5/12, week ending 5/12,
total 5/12, trip date 29/67, client 32/67, odometer start 23/67, odometer end
22/67 and miles 37/67. The most frequent usable-cell confusions were 0 to 6
(39), 9 to 4 (21) and 9 to 3 (20). Blank cells, faint ink, border strokes,
crowded characters, lighting and occupied-row detection remain failure modes.

Preprocessing improved benchmark character accuracy from 17.7% to 52.6%
after alignment/cleanup, 54.1% after border recovery and 69.5% after lighting
correction, without changing model weights. V3 produced 473 recovered and
1,254 lighting-adjusted cells across detected fields; warnings include extra
rows and do not share the labeled-character denominator. Generated test
accuracy slipped from v2's 97.34% to 97.25%, so the change is not a gain on
every dataset.

## Validation and routing

Raw recognition and post-rule errors are separate measures. Header/reference,
date, odometer, mileage and total checks combine with confidence and crop
evidence. Unverified alignment or an adjusted/uncertain crop requires review.
Generated test logs have 3/90 auto-post candidates (3.33%) with zero observed
trip-row transcription errors among those three. Valid QA variants have 8/90
candidates with zero observed errors. Neither small count establishes a safe
production error rate; row-exact scoring excludes header/footer transcription.

Of five expected failed checks in the three original invalid QA controls,
four were detected. In one odometer case, 068253 was read as 062253, masking
the end-before-start failure; the mileage mismatch still required review.
Arithmetic cannot identify every recognition error.

The refreshed October 7 QA run separately reads 36 SCRUM-26 images: 36/36
known-value cases pass, while actual images score 5,222/5,328 characters,
1,089/1,188 exact fields and 139/216 exact rows. Its 26 candidates contain
no invalid-source rows or header/week/footer/row transcription errors.
Eighteen images overlap the older QA pack; these are repeated controls,
not 36 independent handwritten forms or evidence of reader improvement.

## Limits and reproducibility

The 12 photographed forms were evaluated across versions and are a repeated
benchmark. Three development forms and controlled EMNIST glyphs do not
represent field clinicians. Evaluate new writers with v3 fixed and labels
prepared before predictions; use another unseen batch if their results guide
a later version. No production labor savings or confidence calibration is claimed.

The 41 regression tests pass. Reproduction commands and the frozen checkpoint
SHA-256 are in README.md and docs/version_freeze.md; current QA evidence is
in docs/qa/. Full comparisons, field errors and capture experiments are in
[the supporting results record](results_detail.md). Codex assisted with
implementation, analysis and report drafting; the team owns and must explain
the submitted work, as disclosed in docs/ai_assistance.md.
