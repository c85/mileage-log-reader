# Routing Policy

## Purpose

The routing policy determines whether a mileage-log row can be automatically posted for reimbursement or must be sent for human review. The goal is to prevent uncertain or financially significant OCR errors from flowing directly into accounting.

## Decision Rules

A row is eligible for **AUTO-POST** only when:

- The form layout and registration are verified.
- All required fields are successfully extracted.
- OCR confidence meets the approved threshold.
- All business and arithmetic validation checks pass.
- No preprocessing or image-quality warning requires verification.
- The estimated reimbursement exposure is below the review threshold.

A row is routed to **NEEDS REVIEW** when any of these conditions fail.

## Reimbursement Impact

Mileage errors can create different levels of financial exposure depending on the size of the recognition error. Small digit errors may have limited impact, while place-value errors can substantially increase reimbursement amounts.

The reimbursement impact analysis is implemented in:

`scripts/reimbursement_impact.py`

The analysis demonstrates why high-impact or uncertain cases should require human review rather than being automatically posted.

## Current Policy

The system follows a conservative, fail-closed approach. A high model-confidence score alone cannot override failed business checks, uncertain image preprocessing, or significant reimbursement exposure.

Thresholds should be calibrated using held-out validation data before production use. Until sufficient evidence exists, uncertain cases remain routed to human review.

## Business Rationale

This policy prioritizes reimbursement accuracy and financial control over maximizing automation. The objective is not to auto-post every mileage record, but to automate only cases supported by sufficient evidence while preserving human oversight for exceptions.
