# Ten-minute demonstration run of show

## Recommendation — 1 minute

Recommend using this build as a review-triage prototype, not as a payment
poster. It auto-posted 7 of 90 synthetic rows but cleared no complete log;
therefore it produced no measured whole-log keying savings. The bundled
handwriting example sent every row to review.

## Approach — 3 minutes

Show the pipeline in `docs/approach.md`: page registration, the fixed ML-7
cell map, the shared EMNIST normalization, the trained MLP, field assembly,
reference/arithmetic checks, and confidence routing. Defend three choices:

1. Restrict a character by box type and remove lowercase classes.
2. Keep cell extraction and character recognition failures separate.
3. Require both business checks and configured confidence before a row can
   become an auto-post candidate.

## Live run — 3 minutes

Generate the known failing input and run it:

```bash
python scripts/generate_synthetic_logs.py --count 1 --quality clean --fault-row 5
python scripts/demo.py outputs/synthetic_logs/log_001_clean.jpg
```

Point out rows that pass and rows sent to review. The generator deliberately
makes row 5's written miles disagree with its odometer difference while the
weekly total still matches the written column. Explain every displayed
review reason. Then, if time allows, run
`python scripts/demo.py examples/figure1_clean_scan.png` to show how a clean
looking handwriting sample can still produce confident but wrong reads.
The field-condition phone sample is another useful failure:
`python scripts/demo.py examples/figure2_phone_photo.jpg`. Registration
works, but the coffee ring and shadows trigger spurious occupied cells and
the values are routed to review.

## Numbers and limits — 2 minutes

Use `docs/results_summary.md`. Keep isolated EMNIST accuracy (85.07%
unrestricted, 95.73% field restricted), generated-form accuracy (97.48%
characters, 60.00% exact rows), and the 15-image registration spike separate.
State the set sizes and quality conditions. Emphasize that 0 residual errors
among 7 auto-post candidates is far too small to establish a safe error rate.

## Four-week next step — 1 minute

1. Collect team-filled forms with made-up employee/client values and hand-label
   every field.
2. Measure page/cell failure and character/field/row error on that independent
   set, including actual phone captures.
3. Calibrate confidence by field and odometer position; compare residual cost
   with clerk review before changing policy thresholds.
4. Add reviewer feedback and retain the source crop and correction history.

## Backup and Q&A

Capture a separate rehearsal on a team laptop as the backup recording; this
repository contains the reproducible commands and the synthetic inputs but
not a narrated video. Be ready to explain the 784–192–36 network, ten epochs,
learning rate 0.001, batch size 256, the 90,000 selected train images, the
53,398 validation and 89,262 leak-free test characters, and why the form
field restricts digit versus capital-letter outputs.
