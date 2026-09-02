# Public-release privacy audit

Release assessed: `v1.0.1`, 2026-09-02

## Included

- analysis and figure-generation scripts;
- environment specifications and dependency lists;
- code-to-analysis manifest;
- public-data accession instructions; and
- documentation describing controlled study-input requirements.

## Excluded before public deposit

- raw LC-MS and targeted MRM vendor files;
- participant-level clinical records;
- de-identified participant-level processed matrices, model scores, ages and
  CA125 values from the confidential reviewer-input package;
- direct identifiers, dates, free text and internal linkage files; and
- generated analysis outputs, caches and local project data directories.

## Verification record

- tracked text files were screened for absolute local paths, credentials,
  personal-identification labels and email addresses;
- no credentials, direct-identification fields or absolute local paths were
  retained in the public release;
- no direct participant identifiers, contact details or local project paths were
  retained in the public release; and
- Python source files were parsed for syntax after release preparation.

The separate confidential peer-review package is retained for editorial review
and is not part of this public GitHub or Zenodo release.
