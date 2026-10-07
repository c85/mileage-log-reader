# Ten-minute demonstration run of show

## Recommendation — 1 minute

Recommend using this build as a review-triage prototype, not as a payment
poster. It auto-posted 3 of 90 synthetic rows but cleared no complete log;
the repeated team-filled benchmark also had 0 exact rows out of 67.
Therefore the prototype produced no measured whole-log keying savings.

## Approach — 3 minutes

Show the pipeline in `docs/approach.md`: page and printed-template alignment,
box detection near the ML-7 cell map, border recovery, lighting correction,
ink cleanup, the shared EMNIST
normalization, the existing MLP, field assembly,
reference/arithmetic checks, and confidence routing. Defend three choices:

1. Restrict a character by box type and remove lowercase classes.
2. Keep cell extraction and character recognition failures separate.
3. Require verified crops, business checks and configured confidence before a row can
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
.venv/bin/python scripts/generate_synthetic_logs.py --count 1 --quality clean --fault-row 5 --source-split val --output outputs/demo_synthetic
~~~

Select `outputs/demo_synthetic/log_001_clean.jpg`. The demo uses validation
characters and a separate output folder to preserve the recorded test batch.

On a phone connected to the laptop's trusted Wi-Fi, start the server with
~~~bash
.venv/bin/python scripts/web_demo.py --host 0.0.0.0
~~~
and open the laptop's Wi-Fi address on the phone. Use **Take a photo** for a
made-up form; the regular image picker is the fallback. The correction panel
can demonstrate SCRUM-28 and show where its audit record is saved.

For a handwritten file-upload example, choose
`examples/development_forms/form_02.png`. Show the raw result first, then use
that folder's `ground_truth.csv` to point out correct reads, remaining errors
and crop warnings. The full-set
handwriting metrics belong in the results section, not in this single-image
demo.

If the browser is unavailable, use the CLI fallback:

Generate the known failing input and run it:

```bash
.venv/bin/python scripts/generate_synthetic_logs.py --count 1 --quality clean --fault-row 5 --source-split val --output outputs/demo_synthetic
.venv/bin/python scripts/demo.py outputs/demo_synthetic/log_001_clean.jpg
```

Point out rows that pass and rows sent to review. The generator deliberately
makes row 5's written miles disagree with its odometer difference while the
weekly total still matches the written column. Explain every displayed
review reason. Then, if time allows, run
`.venv/bin/python scripts/demo.py examples/figure1_clean_scan.png` to show how a clean
looking handwriting sample can still produce confident but wrong reads.
The field-condition phone sample is another useful failure:
`.venv/bin/python scripts/demo.py examples/figure2_phone_photo.jpg`. Registration
may work while coffee rings and shadows still produce uncertain cells.
Show its current crops and review reasons.

## Numbers and limits — 2 minutes

Use `docs/results_summary.md`. Keep isolated EMNIST accuracy (85.07%
unrestricted, 95.73% field restricted), generated-log accuracy (97.25%
characters, 55.56% exact rows), the 15-image registration spike, the synthetic
fixture score, and the team-filled result (69.5% character accuracy, 0/67
exact rows) separate. The fixture score is 96.9% character accuracy and
52.2% exact rows across 15 paired images of five underlying forms. The three
development forms score 92.3% characters; that is a development result, not
independent performance. Explain that preprocessing improved the handwritten
baseline from 17.7% using the same model. Explain the
border-recovery gain separately: 52.6% to 54.1% on team forms, with expanded
crops still requiring verification. Lighting correction then raised the team
score to 69.5%, while generated synthetic accuracy slipped from 97.34% to
97.25%. Adjusted ink requires review too. State the
set sizes and quality conditions. Emphasize that 0 residual errors among 3
generated-log auto-post candidates is far too small to establish a safe error
rate. State that the team set was evaluated repeatedly; 69.5% is a benchmark
score, and a fresh check needs new writers. Show `docs/version_freeze.md`
to identify the frozen v3 code and checkpoint.

## Four-week next step — 1 minute

1. Evaluate the frozen v3 reader on new writers and capture conditions; label
   every field before inspecting predictions and report raw results separately.
2. If a further version is explicitly authorized, use development examples to
   improve occupied-row detection or extraction and preserve v3 for comparison.
3. For that future version, calibrate confidence by field and odometer position
   on development data and reserve a separate unseen evaluation batch.
4. Compare residual cost with clerk review and retain source crops and
   correction history.

## Backup and Q&A

Christopher's approximately four-minute **How it works / Live demo** segment
is assigned in SCRUM-31.

Capture a separate rehearsal on a team laptop as the backup recording; this
repository contains the reproducible commands and the synthetic inputs but
not a narrated video. Be ready to explain the 784–192–36 network, ten epochs,
learning rate 0.001, batch size 256, the 90,000 selected train images, the
53,398 validation and 89,262 leak-free test characters, and why the form
field restricts digit versus capital-letter outputs.
