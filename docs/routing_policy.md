# Routing Policy

## Purpose

The routing policy determines whether a mileage-log row qualifies as an **AUTO-POST candidate** or must be sent for human review. These are prototype routing labels; the reader does not issue payments or connect to accounting. The goal is to identify uncertain reads and failed reimbursement checks for review.

## Decision Rules

A row is eligible as an **AUTO-POST candidate** only when:

- The form layout and registration are verified.
- All required fields are successfully extracted.
- OCR confidence meets the configured prototype thresholds, including the stricter threshold for the three highest-place odometer digits.
- All business and arithmetic validation checks pass.
- No preprocessing or image-quality warning requires verification.

A row is routed to **NEEDS REVIEW** when any of these conditions fail.

## Reimbursement Impact

Mileage errors can create different levels of financial exposure depending on the size of the recognition error. Small digit errors may have limited impact, while place-value errors can substantially increase reimbursement amounts.

The reimbursement impact analysis is implemented in:

`scripts/reimbursement_impact.py`

The analysis illustrates potential reimbursement errors by digit position at the assignment's $0.62-per-mile rate. For example, a 1,000-mile odometer error could change reimbursement by $620 if it goes undetected.

The script's $100 maximum-exposure cutoff is a hypothetical sensitivity scenario. It is not a limit specified on Form ML-7, an approved business rule, or a threshold enforced by the reader. The reader does not estimate dollar exposure for each row or use this cutoff for routing.

## Current Policy

The system follows a conservative, fail-closed approach. A high model-confidence score alone cannot override failed business checks, uncertain image preprocessing, or unverified layout and registration.

Operational prototype routing uses `configs/reader_policy.json` and the validation checks. The configured minimum character confidence is 0.82, with a stricter minimum of 0.92 for the three highest-place digits in each odometer field. These confidence controls reflect the greater potential impact of place-value errors; they do not calculate financial exposure or enforce a dollar cutoff. This documentation clarification leaves the frozen v3 reader and its settings unchanged.

Thresholds should be calibrated using held-out validation data before production use. Until sufficient evidence exists, uncertain cases remain routed to human review.

## Business Rationale

This policy prioritizes reimbursement accuracy and financial control over maximizing automation. The objective is not to auto-post every mileage record, but to automate only cases supported by sufficient evidence while preserving human oversight for exceptions.
