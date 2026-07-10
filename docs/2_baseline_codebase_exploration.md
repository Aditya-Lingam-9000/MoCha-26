# Baseline Codebase Exploration: MoCha Baseline Bundle

This document details the exact mechanics of the provided `MoCha_baseline_bundle` and how CodaBench evaluates submissions.

## 1. Submission Structure and `run.py`
When a submission zip is uploaded, the CodaBench execution server looks specifically for `run.py`.

Inside `run.py`, the only required function for the API is:
```python
def predict(data: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, int]]:
```
- **Input:** The hidden data dictionary mirroring the exact same structure as the CARE-PD `.pkl` files (except labels are withheld and `beta` values are zeroes).
- **Process:** The baseline iterates over each `subject_id` and `walk_id`, delegating feature extraction and class prediction to a unified `Model` class located in `submission/baseline_model.py`.
- **Output:** A nested dictionary returning `{subject_id: {walk_id: predicted_class}}`, where `predicted_class` is an integer `0, 1, 2, or 3`.

## 2. Feature Extraction Mechanics
In `submission/baseline_model.py`, the processing pipeline operates in two distinct steps per sample:

1. **`MotionPreprocessor` (`submission/preprocess.py`):**
   - **FPS Standardization:** Uses `resample_array` to linearly interpolate the variable FPS sequences to a target static FPS (usually `20.0` or `30.0`).
   - **HumanML3D Conversion:** Converts the `pose` and `trans` arrays into an intermediate 263-dimensional canonical motion representation using SMPL body math (`data.preprocessing.humanml3d.py`).
   - **Normalization:** Standardizes the array using a provided `mean.npy` and `std.npy` derived from the training set. Pads the sequence to a maximum fixed length (e.g., 196 frames).

2. **`EvaluatorModelWrapper` (`model/t2m_eval_wrapper.py`):**
   - Implements a bi-directional GRU model over the 263D representation.
   - Outputs a fixed **512-dimensional continuous feature vector** representing the whole walk sequence (`get_motion_embeddings_ordered`).

## 3. Classifier Mechanics
The baseline uses a lightweight PyTorch neural network (`MotionClassifier`) on top of the 512D vector:
```python
nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 4)
)
```
- While functional, small MLPs on heavily tabular/low-dimensional data tend to underperform compared to Decision Trees and Boosting variants.
- Furthermore, PyTorch MLPs require batch iteration, data loaders, and gradients, making local experimentation slower on a dual-core CPU compared to `scikit-learn`/`xgboost`.

## 4. Path Forward
We will keep the `MotionPreprocessor` and `EvaluatorModelWrapper` completely intact, as they provide an excellent, robust 512D representation of the complex 3D human motion. 

We will entirely remove `MotionClassifier`, extracting the 512D representations *offline* and subsequently training a `LightGBM` model. Our final CodaBench submission will simply invoke `LightGBM.predict()` after running the sequence through the frozen `EvaluatorModelWrapper`.
