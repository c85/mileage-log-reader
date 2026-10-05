# Business note for Accounts Payable

## Recommended policy

Do not post a row just because its characters look confident. Make a row an
auto-post candidate only when the image is registered, every required
character has confidence above the configured threshold, the employee and
date-specific client exist in reference data, the trip falls within the
stated week, odometers move forward, written miles equal the odometer
difference, and the weekly total reconciles. Require a stricter threshold for
the first three digits of either six-digit odometer. Route every failed or
unsupported case to AP and preserve the original crop and checks.

The committed thresholds are prototype settings in
`configs/reader_policy.json`, not approved production cutoffs. They must be
chosen on an independently labeled set of team-filled, made-up logs by
comparing residual error against the cost of review. Until that evidence
exists, the model is a sorting aid and a clerk remains responsible for
reimbursement approval.

## Monthly cost frame

At 10,000 logs per month and $3.40 of keying per log, today's direct keying
cost is about **$34,000 per month** or **$408,000 per year**. The brief's
70,000 monthly rows mean each one percentage point of row error is about 700
wrong rows. If each underpayment creates a $28 correction ticket, a 2% error
rate across all rows could mean about $39,200 of ticket work per month before
any clinician-retention effect. Overpayments need a separate calculation:
the same wrong odometer digit may change mileage by 100,000, 10,000, 1,000,
100, 10, or 1 mile depending on position, or $62,000, $6,200, $620, $62,
$6.20, or $0.62 at the reimbursement rate. A project-level error rate must
not be converted to dollars without the position and direction of each error.

Let `q` be the measured share of logs for which **every** row passes policy.
If any row needs review and AP still keys the full log, the rough handling
cost is:

| Fully auto-posted log share `q` | Logs still keyed | Keying cost | Gross keying cost avoided |
|---:|---:|---:|---:|
| 0% | 10,000 | $34,000 | $0 |
| 50% | 5,000 | $17,000 | $17,000 |
| 80% | 2,000 | $6,800 | $27,200 |

These are scenarios, not measured savings. They assume no partial-log review
and exclude software, monitoring, exception handling, audit, payroll tax,
false-post, and correction costs. `scripts/evaluate_synthetic_logs.py` writes
the same transparent scenario using the observed fully auto-posted log
share. It reports residual errors among posted rows separately; those errors
must be costed by field and odometer position before a CFO business case is
made.

In the current 15-log synthetic evaluation, **0 of 15 logs** passed every
row, so the conservative gross keying cost avoided is $0 per month. Seven of
90 individual rows met the row policy (7.8%), with 0 observed errors among
those 7. That small controlled sample cannot establish a safe false-post
rate. It does show that a promising row-level character score may still
produce no fully automatic logs when every row must pass. If AP can review
only individual rows, an actual per-row handling cost is needed before
estimating savings.

## Decision after the prototype

Compare manual review cost with the cost-weighted false-post rate at several
configured confidence thresholds. Prefer a low straight-through rate with
measured near-zero residual errors over a high rate whose errors cannot be
reproduced. Expand automation only after phone-photo tests, team-filled
handwriting, and independent review establish that the thresholds work for
the forms AP will actually receive.
