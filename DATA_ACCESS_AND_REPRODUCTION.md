# Data access and reproduction notes

## Restricted study data

This public package intentionally excludes raw LC-MS vendor files, targeted MRM
raw files, participant-level processed inputs and full clinical metadata. These
materials are restricted by consent, ethics approval and institutional
data-sharing requirements. Processed source tables and figure-source summaries
are supplied separately as article Source Data and Supplementary Tables. A
minimal pseudonymized processed-input package is supplied only to editors and
peer reviewers through the confidential manuscript-submission route.

## Public data

- GSE217517 and GSE184880: Gene Expression Omnibus.
- TCGA-OV: NCI Genomic Data Commons.
- CPTAC ovarian proteomic resources: PDC000110 and, where applicable,
  ProteomeXchange PXD015903.

## Reproduction sequence

1. Create an authorised project mirror with the relative directory layout
   described in the script manifests.
2. Review `processed_input_examples/README.md` for the required input classes.
   Obtain controlled study inputs only through the manuscript data-access route;
   do not attempt to infer or reconstruct participant-level inputs from public
   outputs.
3. Create the Python/R environment using the supplied specifications.
4. Run the LC-MS QC and discovery scripts, then the current ModelA and ModelC
   scripts.
5. Run the targeted-panel-score plus CA125 workflow only as the exploratory
   ovarian-lesion triage analysis described in the manuscript.
6. Compare regenerated summaries against the supplied Source Data and
   Supplementary Table 6.

No script in this package should be interpreted as supporting unrestricted
release of patient-level data.

