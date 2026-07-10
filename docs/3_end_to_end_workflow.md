# End-to-End Workflow: MoCha2026 CPU-Optimized Strategy

This document summarizes the exact mapping of inputs to outputs across our localized environment, outlining the flow for Phase 1 and beyond.

## Flow 1: Offline Training (Local Machine)
Since we have strict hardware constraints (Ryzen 3 CPU, 8GB RAM, No GPU), we decouple feature extraction from model optimization.

### Step 1: Pre-extraction (Phase 1)
1. **Input:** The `Canonicalized_SMPL_pickles` directory.
2. **Process:** A standalone Python script reads one `.pkl` file at a time. For every sequence, it routes `data["pose"]` and `data["trans"]` through the baseline's `MotionPreprocessor` (which normalizes the sequence and fixes the FPS) and the frozen `EvaluatorModelWrapper` (which generates the 512-dimension embedding using PyTorch CPU inference).
3. **Memory Management:** Memory is explicitly cleared and garbage collected (`gc.collect()`) after each sequence and file to prevent RAM usage from spiking above ~2GB.
4. **Output:** A compiled, single file (e.g., `train_features.csv` or `.npy`) storing `[subject_id, walk_id, 512_features..., label]`.

### Step 2: Classifier Training (Phase 2 & 3)
1. **Input:** The compiled `train_features.csv` dataset.
2. **Process:** Utilizing `LightGBM` (or Scikit-Learn's `HistGradientBoostingClassifier`), we execute a 5-Fold Stratified Cross-Validation on the tabular dataset. We optimize hyperparameters using `Optuna` or `GridSearchCV` to maximize **Quadratic Weighted Kappa (QWK)** and **Macro F1**.
3. **Output:** A serialized `model.joblib` or `model.txt` representing the absolute best decision boundary for predicting MDS-UPDRS severity.

## Flow 2: Inference (CodaBench Server)
The CodaBench server executes our submitted zip file against a hidden dataset of canonicalized `.pkl` files exactly like our local training data. 

### Step 1: Setup
1. **Input:** The zip file is unpacked inside CodaBench's isolated Docker container.
2. **Initialization:** The evaluation script imports our `run.py`. We initialize the frozen PyTorch `EvaluatorModelWrapper` (for feature extraction) and our custom `LightGBM` model (for classification).

### Step 2: Prediction loop
1. **Input:** The evaluator passes a dictionary containing raw, unlabelled `pose` and `trans` arrays via `predict(data)`.
2. **Process:** 
   - `predict` loops through each sample.
   - The sample is mapped to a 512D feature using the PyTorch wrapper.
   - The 512D feature is passed to `lightgbm_model.predict()`.
3. **Output:** The script immediately returns an integer `0, 1, 2, or 3`. CodaBench aggregates these answers and computes the final leaderboard Macro F1.

## Conclusion
By splitting the data flow into these two completely isolated processes (Offline Extraction vs Fast ML Training), we turn a 3D Human Pose deep learning problem into a traditional tabular data problem, completely neutralizing the disadvantages of our hardware constraints.
