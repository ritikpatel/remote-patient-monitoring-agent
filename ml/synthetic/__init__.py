"""EMR-WGAN synthetic EHR generation, following Yan et al. (JMIR AI 2024;3:e52615).

`mimic-synthetic-generator-tutorial.pdf` in the repo root is the tutorial this
package implements: preprocessing, EMR-WGAN training, generation with
postprocessing, and the paper's data-quality evaluation battery (Figure 3).

Read `ml/synthetic/README.md` before using any of it. The short version is that
the tutorial's premise is 181,294 real patients and this cohort has 100, and
that difference is not a detail -- it decides what the output may legitimately
be used for. `evaluate.py` measures the consequence rather than asserting it.
"""
