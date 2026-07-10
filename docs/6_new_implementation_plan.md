# New Implementation Plan: Advanced MoCha2026 Strategy

This document outlines the updated phase-by-phase implementation plan, integrating the findings from the `TaatiTeam/CARE-PD` repository while strictly adhering to our Ryzen 3 / 8GB RAM hardware constraints.

## Overview of Strategy Update
While the baseline bundle provided a BiGRU `EvaluatorModelWrapper` for 512D feature extraction, the new CARE-PD repository provides weights for state-of-the-art Transformer models (like `MotionBERT`, `MoMask`, `MixSTE`). 

**The new plan:** We will still use CPU-friendly LightGBM for classification, but we will upgrade our feature extraction pipeline to test BOTH the baseline BiGRU and a pre-trained Transformer (e.g., `MoMask` or `MotionCLIP`) from the new repo.

---

### Phase 1: Environment Setup & Pre-Trained Weights
- **Action:** Install dependencies (`lightgbm`, `scikit-learn`, `optuna`).
- **Action:** Download the pre-trained checkpoints using `bash CARE-PD_github/scripts/download_models.sh`.
- **Goal:** Ensure all deep learning backbones are available locally on the SSD.

### Phase 2: Memory-Efficient Offline Feature Extraction (Dual-Track)
- **Action:** Write `extract_features.py`.
- **Track A (Baseline Features):** Loop through `Canonicalized_SMPL_pickles`, passing them through `EvaluatorModelWrapper` (BiGRU) to generate `train_features_baseline.csv`.
- **Track B (SOTA Features):** Loop through the same pickles, passing them through `MoMask` (or `MotionCLIP`) loaded via `backbone_loader.py` to generate `train_features_sota.csv`.
- **Constraint Check:** Both tracks will be run purely on the CPU, loading one pickle at a time, clearing memory (`gc.collect()`) after every sequence to avoid exceeding 8GB RAM.

### Phase 3: LightGBM / XGBoost Model Training
- **Action:** Write `train_lgbm.py`.
- **Method:** We will train and evaluate Gradient Boosting models on both `train_features_baseline.csv` and `train_features_sota.csv`.
- **Validation:** 5-Fold Stratified Cross-Validation evaluating `Macro F1` and `Quadratic Weighted Kappa (QWK)`. We will apply class weights or SMOTE to handle the ordinal class imbalance (UPDRS 0, 1, 2, 3).
- **Goal:** Identify which feature set (Baseline vs SOTA) yields a Macro F1 score consistently > 0.70.

### Phase 4: Constructing the Final CodaBench Submission
- **Action:** Overwrite `run.py` and `submission/baseline_model.py` in the `MoCha_baseline_bundle`.
- **Method:** 
  1. The `predict()` function will load the winning feature extractor (either BiGRU or the SOTA transformer) into CPU memory.
  2. The extracted features are passed into our optimized `LightGBM` model (`model.joblib`).
  3. The integer prediction is returned.
- **Packaging:** We will zip our modified `run.py`, the LightGBM weights, and the winning PyTorch checkpoint. (Note: CodaBench submissions must be <1GB, so we will only bundle the exact weights we need).

### Phase 5: Submission & Iteration
- **Action:** Upload the zip to CodaBench.
- **Goal:** Verify that our local Cross-Validation Macro F1 generalizes to the hidden test server. If not, increase LightGBM regularization (e.g., max_depth reduction, higher L1/L2 penalty) and submit again.
