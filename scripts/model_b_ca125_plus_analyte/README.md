# Step 2 panel-CA125 clinical-application analysis

`run_two_step_clinical_application_current.py` is the manuscript-aligned
analysis entry point. It compares benign lesions with borderline or malignant
lesions and combines CA125 with six targeted analytes: 3-GPA,
acetylcarnitine, creatine, arginine, carnitine and tryptophan.

The model is an L2-regularised logistic regression with balanced class weights
and `C = 0.01`. Development predictions are averaged from 20 repeats of
stratified five-fold out-of-fold prediction. The 100 fitted development-fold
models form the fixed temporal-validation prediction ensemble. The decision
threshold is selected from development OOF predictions only and is applied to
temporal validation without refitting or threshold re-optimisation.

Set `OV_PROJECT_ROOT` to an authorised project mirror containing the controlled
clinical and targeted-MRM inputs documented in the repository. Participant-level
study inputs are not distributed in the public release.

`audit_step2_calibration_transport.py` generates Brier score,
calibration-in-the-large, calibration slope and threshold-transport summaries
from the participant-level prediction table produced inside the authorised
project environment.
