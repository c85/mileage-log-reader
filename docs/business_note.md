# Business note for Accounts Payable

## Recommended policy

Keep a person checking and approving every reimbursement. The prototype can
read mileage forms and flag problems, but it is not ready to approve payments
automatically. After improving cell preprocessing, the team's test of 12
handwritten forms read **69.5% of characters correctly** and did not read any
complete trip row correctly
(**0 of 67 rows**). Every trip row needed review. This improved the earlier
17.7% character baseline using the existing model, but still requires human
checking of every reimbursement.
Recovering strokes near box borders added a modest gain over the previous
52.6% score. Expanded crops remain flagged for a person to verify.
Cell lighting correction subsequently raised the score from 54.1% to 69.5%
and exact fields from 84 to 158 out of 371. Adjusted ink also needs source
verification; this gain does not yet support automatic reimbursement approval.

Before a row could qualify for automatic processing, the system would need
a properly aligned image, reliable character readings, valid employee and
client details, dates within the stated week, and matching odometer readings,
written miles and weekly totals. Digits that could cause large payment errors
need stricter checks. Keep the original image and check results for review.
The current settings still need testing on more handwriting.

## Monthly cost frame

The assignment estimates **10,000 logs per month at $3.40 each** for manual
entry: **$34,000 per month**, or **$408,000 per year**.

The examples below assume staff still enter the entire log whenever any row
needs review, at the same $3.40 cost per log.

| Logs needing no manual entry | Logs still keyed | Monthly manual-entry cost| Possible savings before other costs |
|---:|---:|---:|---:|
| 0% | 10,000 | $34,000 | $0 |
| 50% | 5,000 | $17,000 | $17,000 |
| 80% | 2,000 | $6,800 | $27,200 |

**These are examples, not achieved savings.** Software, staff checking and
error-correction costs must also be measured and included.

In a separate test of 15 computer-generated logs, three of 90 rows passed
the checks, but no complete log passed. Every log deliberately included a
mileage mistake, so this test does not predict how many normal AP logs could
be processed automatically. No labor savings have been demonstrated.
These figures refer to the command-generated logs, not the
separate pre-rendered fixtures in `examples/synthetic_forms/`. Those fixtures
have a separate QA score in `docs/results_summary.md`; their small, paired set
does not change the generated-log auto-post share or savings estimate.

## Why errors matter

Incorrect mileage can cause overpayments or underpayments. For example,
misreading one odometer value by 1,000 miles could change reimbursement by
**$620** at $0.62 per mile if the mistake goes undetected. Underpayments also
create correction work: if 2% of the 70,000 monthly rows were underpaid and
each caused one $28 correction ticket, that work would cost **$39,200**.
These examples show potential costs; they are not measured losses.

## Decision after the prototype

Improve handwriting accuracy, test forms from more people and different
phone photos, and have another person verify the correct answers. Measure
how long AP staff take with and without the tool. Use those results to decide
whether it saves time after checking and correction costs. The Product Lead
should review and approve any proposal to expand automatic processing.
