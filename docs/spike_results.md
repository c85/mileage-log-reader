# Registration spike results

The SCRUM-12 spike evaluated 15 synthetic blank-form captures and did not train or score handwriting recognition.

| Capture condition | Photos | Registered | Box-grid accepted | Mean corner error | Mean box success |
|---|---:|---:|---:|---:|---:|
| clean | 5 | 5 | 5 | 1.0px | 100.0% |
| perspective_shadow | 5 | 5 | 5 | 0.4px | 100.0% |
| rotated | 5 | 5 | 5 | 1.1px | 100.0% |

**Recommendation:** GO for the measured clean, small-rotation and moderate-perspective synthetic conditions; unsupported page boundaries remain needs-review cases.

Cell-box success requires a registered page, at least 80% of printed character boxes to pass four-edge coverage, and exactly six mapped cells in each odometer field. The capture set is programmatically warped from the provided blank ML-7 template; it is not evidence about handwritten-cell segmentation or model accuracy.

Failure cases, corner coordinates and per-box scores are in `outputs/spike_registration/results.json`.
