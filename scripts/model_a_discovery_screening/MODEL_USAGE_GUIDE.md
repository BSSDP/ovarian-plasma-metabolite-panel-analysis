# ModelA locked pipeline usage

`final_model.pkl` contains the fitted full-data pipeline and its reporting metadata:

- median imputation, standardisation, RFE and the selected classifier;
- required feature names and order;
- the selected model family and feature count;
- an internally derived Youden threshold;
- the repeated nested cross-validation protocol.

Use `load_and_use_model.py` to apply the pipeline to a data frame containing the
eight required input features.

The model was evaluated using repeated nested internal cross-validation on 222
normal-control and malignant-tumour samples. The stored threshold is intended for
future locked evaluation and is **not** an externally validated clinical threshold.
