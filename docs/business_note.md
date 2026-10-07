# Business note for Accounts Payable

## Recommended policy

Keep a person checking and approving every reimbursement. On 12 photographed
handwritten forms, the prototype read **69.5% of characters correctly**, with
**0 of 67 complete trip rows correct**; every row required review. Alignment,
cropping and lighting changes improved the earlier 17.7% baseline while
keeping the trained model unchanged. The reader is not ready to approve payments.

An auto-post candidate must have verified alignment and crops, reliable
readings, valid employee/client references and dates, and matching odometers,
miles and total. The frozen minimum confidence is 0.82 ordinarily and 0.92
for each odometer's three highest-place digits. Failed checks or crop warnings
require review regardless of confidence. Retain images and check results.
These thresholds need calibration on new writers; the 12 forms are a repeated benchmark.


## Monthly cost frame

The assignment estimates **10,000 logs per month at $3.40 each** for manual
entry: **$34,000 per month**, or **$408,000 per year**.

The examples below assume staff still enter the entire log whenever any row
needs review, at the same $3.40 cost per log.

| Fully automatic logs | Logs still keyed | Monthly keying cost | Gross savings before other costs |
|---:|---:|---:|---:|
| 0% | 10,000 | $34,000 | $0 |
| 50% | 5,000 | $17,000 | $17,000 |
| 80% | 2,000 | $6,800 | $27,200 |

**These are examples, not achieved savings.** Software, staff checking and
error-correction costs must also be measured and included.

In 15 generated logs, three of 90 rows qualified but no complete log did.
Every log deliberately contained a mileage fault, so this sample cannot
predict normal AP automation. No labor savings have been demonstrated.


## Why errors matter

Incorrect mileage can cause overpayments or underpayments. For example,
misreading one odometer value by 1,000 miles could change reimbursement by
**$620** at $0.62 per mile if the mistake goes undetected. Underpayments also
create correction work: if 2% of the 70,000 monthly rows were underpaid and
each caused one $28 correction ticket, that work would cost **$39,200**.
These examples show potential costs; they are not measured losses.


## Decision after the prototype

Evaluate frozen v3 on new writers and phone conditions, with independently
checked answers prepared before inspecting predictions. Reserve a fresh test
batch if those results guide a later version. Compare AP time with and without
the tool, including review and correction costs. The Product Lead should
approve any proposed expansion of automatic processing.
