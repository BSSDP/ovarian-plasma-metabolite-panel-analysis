# Machine-learning-guided development and temporal validation of a targeted plasma metabolite panel for ovarian cancer detection

Fixed public release: `v1.0.0`

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21594513.svg)](https://doi.org/10.5281/zenodo.21594513)

This repository contains the analysis and figure-generation code for the
manuscript **Machine-learning-guided development and temporal validation of a
targeted plasma metabolite panel for ovarian cancer detection**. It is organised as a fixed,
auditable public release for GitHub deposit at
`https://github.com/BSSDP/ovarian-plasma-metabolite-panel-analysis` and Zenodo
archival.

## Scope

The package covers untargeted LC-MS preprocessing and QC, feature-level
differential analysis, discovery normal-control versus malignant-tumour model
development, targeted MRM panel development, fixed-score temporal validation,
normal-control-derived age-residualization sensitivity analyses, exploratory
targeted-panel-score plus CA125 lesion triage, single-cell/scFEA analyses, and
CPTAC/TCGA public-omics support.

## Data access and execution boundary

Raw LC-MS vendor files, targeted MRM files, participant-level processed inputs,
full clinical records and large public-omics objects are not redistributed in
this public repository. They are subject to participant consent, institutional
data-sharing controls or source-database access conditions. The scripts retain
project-relative path assumptions so they can be run in an authorised project
mirror with controlled inputs placed at the documented locations.
`processed_input_examples/README.md` defines the required input classes and
access boundaries. A separate minimal pseudonymized input package is available
to editors and peer reviewers under confidential access, as described in the
manuscript Data availability statement.

The approved public source-data scope is limited to the figure-specific Source
Data released with the article. Direct identifiers, dates, free text, raw vendor
files and complete individual-level clinical data are excluded from this
repository.

## Important model provenance

The manuscript-targeted score is the **weighted probability ensemble** recorded
in the current targeted-panel outputs and Supplementary Table 6. The current
scripts added in this release supersede earlier exploratory variants retained in
the historical package. Use the current scripts whose filenames end in
`_current.py` or refer to the targeted-panel-score plus CA125 lesion-triage
workflow.

## Environment

Python package requirements are specified in `environment.yml` and
`requirements.txt`; R dependencies are listed in `renv_packages.txt`. A pinned
full runtime may additionally be reconstructed from the supplied specifications.
Where data paths differ, configure an authorised project mirror rather than
modifying source data.

## Reproducing the core models

1. Create the Python environment from `environment.yml` or `requirements.txt`;
   install the listed R packages if running the R components.
2. Review `processed_input_examples/README.md` and
   `DATA_ACCESS_AND_REPRODUCTION.md`; obtain controlled study inputs only
   through the manuscript data-access route where permitted.
3. Use `CODE_PACKAGE_MANIFEST.tsv` to identify the current scripts for each
   manuscript analysis and figure. Scripts ending in `_current.py` supersede
   historical exploratory variants.
4. Verify the release after download with `RELEASE_MANIFEST.sha256`.

## Integrity and citation

`CODE_PACKAGE_MANIFEST.tsv` lists the script-to-analysis mapping.
`RELEASE_MANIFEST.sha256` records checksums for every package file. Cite the
manuscript and fixed-release DOI
[`10.5281/zenodo.21594513`](https://doi.org/10.5281/zenodo.21594513);
`CITATION.cff` provides version-specific repository metadata.

## Licence

Source code is released under the MIT License. Public database data remain
subject to their original repository terms. This repository must not be used to
attempt re-identification or to link public outputs with external clinical
records.
