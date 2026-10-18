# Models & Artifacts

This folder contains pre-trained artifacts produced during training:

1. **`scaler.pkl`**:
   - `sklearn.preprocessing.StandardScaler` object.
   - Fitted on the 2,309-dimensional feature representation (2,304 semantic interaction features + 5 metadata features).

2. **`specter2_nn_model.h5`**:
   - 4-layer Keras Sequential Neural Network with Batch Normalization, LeakyReLU, and Dropout regularization.
   - Input dimension: 2,309.
   - Output: Sigmoid probability of citation relationship.
   - Recommended decision threshold: **0.35** (optimized via Matthews Correlation Coefficient, yielding MCC ~0.5306).
