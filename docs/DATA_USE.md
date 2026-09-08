# Data Use

Two PhysioNet **demo** datasets are used. Both are the openly-available demo
subsets, not the full credentialed databases — no PhysioNet credentialing was sought
or is required for the data actually used in this project (see §1 of `PROJECT_PLAN.md`
for why: demo data only, clinical narrative synthesised).

## Datasets and licenses

| Dataset | Version | Local path | License | Terms |
|---|---|---|---|---|
| MIMIC-IV Clinical Database Demo | 2.2 | `mimic-iv-clinical-database-demo-2.2/` | [Open Database License (ODbL) v1.0](https://physionet.org/content/mimic-iv-demo/view-license/2.2/) | Free to share, modify, and use with attribution; share-alike for derivative databases |
| Wearable Device Dataset from Induced Stress and Structured Exercise Sessions | 1.0.1 | `wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/` | [Open Data Commons Attribution License (ODC-By) v1.0](https://physionet.org/content/wearable-device-dataset/view-license/1.0.1/) | Free to share, modify, and use with attribution; no share-alike requirement |

Full license text is vendored alongside each dataset (`LICENSE.txt` in each dataset
folder) and is authoritative over this summary.

## What this means for the project

- **Attribution is required** for both datasets in any publication, report, or
  slide deck derived from them. Cite:
  - Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Badawi, O.
    (2023). *MIMIC-IV Clinical Database Demo* (version 2.2). PhysioNet.
    https://doi.org/10.13026/dp1f-ex47
  - Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
    *Circulation* 101(23), e215–e220 (the standard PhysioNet acknowledgment).
- **No PHI re-identification.** MIMIC-IV is already de-identified per HIPAA Safe
  Harbor; this project must not attempt to re-link, re-identify, or combine these
  records with any external identifying data.
- **Demo scope is a stated limitation, not a workaround.** 100 MIMIC-IV patients, 20 ICU deaths, 53 readmissions (E6 in `PROJECT_PLAN.md`). Every
  report carries the honest-reporting statement (§17 of `PROJECT_PLAN.md`).
- **The wearable dataset has no link to the ICU cohort** (healthy volunteers, median
  age ~21 vs ICU median 63 — E10). It is used only as a transport/DSP testbed and for
  documented, watermarked signal morphing (R7) — never presented as measured patient
  data.

## Rule: no raw data enters git

Raw dataset files (`*.csv`, `*.csv.gz`, WFDB `.dat`/`.hea`, and the three top-level
dataset directories) are excluded via `.gitignore`. `data/raw/` holds symlinks to the
three dataset folders for pipeline code to reference at a stable relative path — the
symlinks themselves are also gitignored so the repository never depends on where a
given machine happens to keep the datasets. Anyone cloning this repository must
download the three demo datasets from PhysioNet independently:

- https://physionet.org/content/mimic-iv-demo/2.2/
- https://physionet.org/content/wearable-device-dataset/1.0.1/

Derived artefacts that are small, non-identifying, and needed for reproducibility
(e.g. `eda_outputs/*.csv`, figures) may be tracked via DVC or git as appropriate —
never as raw PHI-shaped tables.
