# Release notes

## v1.1.0 (submission-synchronised release, 2026-09-13)

- Replaced the earlier eight-analyte B+BD-versus-M CA125 analysis with the
  manuscript-aligned six-analyte-plus-CA125 Step 2 model for B versus BD+M.
- Added the fixed Step 2 endpoint, C = 0.01 regularisation setting, repeated
  five-fold development OOF procedure, 100-model temporal-validation ensemble,
  fixed threshold rule and calibration-transport audit.
- Added sample-level paired single-cell pathway analysis used for Fig. 6d.
- Updated the GSE184880 scFEA analysis to the seven clinically annotated cancer
  samples used in the manuscript.
- Removed superseded Model B scripts from the submission archive; historical
  versions remain recoverable from Git history.
- Refreshed the code-to-analysis manifest, privacy audit and release checksums.

## v1.0.1 (fixed public release, 2026-09-03)

- Removed absolute local style paths and bundled the publication style needed
  by public figure scripts.
- Standardised reader-facing cohort labels to discovery cohort and temporal
  same-centre validation cohort without changing model inputs, scores,
  thresholds or numerical results.
- Refreshed release metadata, privacy checks and file-level checksums.
- Zenodo concept DOI: https://doi.org/10.5281/zenodo.21594512.

## v1.0.0 (fixed public release, 2026-07-26)

- Fixed manuscript-associated analysis and figure-generation scripts.
- Public input requirements and controlled-access instructions for the core
  untargeted and targeted models; participant-level processed inputs are not
  included in the public release.
- Environment specifications, script manifest and release checksums.
- Public-release metadata, citation file, MIT source-code licence and privacy
  audit record.
- Zenodo archival DOI: https://doi.org/10.5281/zenodo.21594513.
