# Ten-minute demonstration run of show

## Recommendation — 1 minute

Recommend using this build as a review-triage prototype, not as a payment
poster. It auto-posted 7 of 90 synthetic rows but cleared no complete log;
the independent team-filled evaluation also had 0 exact rows out of 67.
Therefore the prototype produced no measured whole-log keying savings.

## Approach — 3 minutes

Show the pipeline in `docs/approach.md`: page registration, the fixed ML-7
cell map, the shared EMNIST normalization, the trained MLP, field assembly,
reference/arithmetic checks, and confidence routing. Defend three choices:

1. Restrict a character by box type and remove lowercase classes.
2. Keep cell extraction and character recognition failures separate.
3. Require both business checks and configured confidence before a row can
   become an auto-post candidate.

## Live run — 3 minutes

For the classroom presentation, start the browser interface:

~~~bash
.venv/bin/python scripts/web_demo.py
~~~

Generate a known-failing synthetic log, select its image in the browser, and
expand row 5 to show the cell crops, confidence, odometer mismatch, and review
route:

~~~bash
.venv/bin/python scripts/generate_synthetic_logs.py --count 1 --quality clean --fault-row 5
~~~

On a phone connected to the laptop's trusted Wi-Fi, start the server with
~~~bash
.venv/bin/python scripts/web_demo.py --host 0.0.0.0
~~~
and open the laptop's Wi-Fi address on the phone. Use **Take a photo** for a
made-up form; the regular image picker is the fallback. The correction panel
can demonstrate SCRUM-28 and show where its audit record is saved.

For a handwritten file-upload example, choose
`examples/team_filled_forms/form_04.png`, which had the strongest partial
field score in the set. Show the raw result first, then use
`ground_truth.csv` to point out correct reads and errors. The full-set
handwriting metrics belong in the results section, not in this single-image
demo.

If the browser is unavailable, use the CLI fallback:

Generate the known failing input and run it:

```bash
.venv/bin/python scripts/generate_synthetic_logs.py --count 1 --quality clean --fault-row 5
.venv/bin/python scripts/demo.py outputs/synthetic_logs/log_001_clean.jpg
```

Point out rows that pass and rows sent to review. The generator deliberately
makes row 5's written miles disagree with its odometer difference while the
weekly total still matches the written column. Explain every displayed
review reason. Then, if time allows, run
`.venv/bin/python scripts/demo.py examples/figure1_clean_scan.png` to show how a clean
looking handwriting sample can still produce confident but wrong reads.
The field-condition phone sample is another useful failure:
`.venv/bin/python scripts/demo.py examples/figure2_phone_photo.jpg`. Registration
works, but the coffee ring and shadows trigger spurious occupied cells and
the values are routed to review.

## Numbers and limits — 2 minutes

Use `docs/results_summary.md`. Keep isolated EMNIST accuracy (85.07%
unrestricted, 95.73% field restricted), generated-log accuracy (97.48%
characters, 60.00% exact rows), the 15-image registration spike, the synthetic
fixture score, and the team-filled result (17.7% character accuracy, 0/67
exact rows) separate. The fixture score was 97.1% character accuracy and
55.6% exact rows across 15 paired images of five underlying forms. State the
set sizes and quality conditions. Emphasize that 0 residual errors among 7
generated-log auto-post candidates is far too small to establish a safe error
rate.

## Four-week next step — 1 minute

1. Expand the handwritten set with more writers and lighting, angle, and focus
   conditions; label every field.
2. Improve occupied-row detection and cell extraction, then measure character,
   field, and row error on a separate evaluation set.
3. Calibrate confidence by field and odometer position on development data;
   keep the held-out forms out of tuning.
4. Compare residual cost with clerk review and retain source crops and
   correction history.

## Backup and Q&A

Capture a separate rehearsal on a team laptop as the backup recording; this
repository contains the reproducible commands and the synthetic inputs but
not a narrated video. Be ready to explain the 784–192–36 network, ten epochs,
learning rate 0.001, batch size 256, the 90,000 selected train images, the
53,398 validation and 89,262 leak-free test characters, and why the form
field restricts digit versus capital-letter outputs.
