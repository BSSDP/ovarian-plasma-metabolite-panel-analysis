# Processed input requirements and regeneration notes

This public manuscript code package is designed to run from processed inputs
rather than raw vendor files or restricted clinical source documents. No
participant-level study input files are included in this repository.

## Required input classes

1. Untargeted LC-MS abundance matrices
   - positive- and negative-ion cleaned abundance matrices;
   - sample metadata with lesion group labels and clinical covariates; and
   - metabolite annotation tables with feature identifiers and pathway labels.
2. Targeted MRM input tables
   - eight-analyte area-ratio matrix;
   - analytical batch labels;
   - normal-control and malignant-tumour labels; and
   - CA125 values and lesion labels for the exploratory CA125 integration.
3. Single-cell inputs
   - processed or raw count matrices for GSE217517 and GSE184880;
   - per-cell sample identifiers and tumour/normal source labels; and
   - cell-type annotation and CNV-score outputs.
4. scFEA inputs
   - epithelial-cell count matrices;
   - tumour-like versus normal-like epithelial metadata; and
   - scFEA module annotations and metabolite-balance mappings.
5. CPTAC/TCGA support inputs
   - CPTAC-OV gene-level proteomics matrices and sample annotations; and
   - TCGA-OV expression and clinical survival tables.

## Files not included in the public release

- raw LC-MS and targeted MRM vendor files;
- de-identified participant-level processed matrices, model scores, ages and
  CA125 values;
- individual patient-level clinical source workbooks;
- large `.h5ad` single-cell objects and cached public-portal downloads; and
- trained model objects containing patient-level matrices.

The manuscript Data availability statement identifies public Source Data,
public dataset accessions and the controlled-access route for restricted study
data. A minimal pseudonymized processed-input package is available to editors
and peer reviewers through confidential manuscript review, subject to the
stated conditions.

