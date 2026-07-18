# MoCha Baseline Bundle

This repository is a submit-ready baseline for **MoCha2026 - Benchmark and
Challenge on Parkinsonian Gait @ ECCV2026**.

The challenge evaluates methods that predict MDS-UPDRS gait severity from
canonicalized SMPL motion sequences. The CodaBench evaluator imports the
submission entry point, calls `predict(data)`, and handles writing prediction
files for scoring.

## Repository Layout

```text
run.py                    # Required CodaBench entry point
metadata.yaml             # No-op command required by CodaBench
submission/               # Baseline model and preprocessing wrapper
model/                    # Motion encoder architecture
data/preprocessing/       # SMPL-to-motion-feature conversion utilities
human_body_prior/         # Minimal SMPL body model utilities used by preprocessing
chumpy/                   # Compatibility shim for legacy SMPL pickles
weights/                  # Baseline checkpoints and preprocessing statistics
```

Only `run.py` is part of the challenge API. The other folders are one example
implementation and can be replaced by your own model code.

## Expected Input Format

CodaBench calls:

```python
predict(data)
```

where the unseen challenge data follows this format:

```python
data[subject_id][walk_id] = {
    "pose": np.ndarray,   # shape (T, 72), SMPL axis-angle pose
    "trans": np.ndarray,  # shape (T, 3), global translation
    "beta": np.ndarray,   # shape (1, 10), all zeros for privacy
    "fps": int,           # original collection frame rate
}
```

The baseline handles FPS conversion internally before inference. If you replace
the model, make sure your preprocessing converts each sample to the frame rate
expected by your method.

## Required Output Format

Your `predict` function must return a nested Python `dict`:

```python
predictions[subject_id][walk_id] = predicted_class
```

where `predictions` is a dictionary of dictionaries and `predicted_class` is
an integer in `{0, 1, 2, 3}`.

You do not need to write `predictions.json` inside CodaBench. The official
ingestion program serializes the dictionary returned by `predict(data)`.

## Preparing a CodaBench Submission

Create a zip whose root contains `run.py`, `metadata.yaml`, and any files or
folders needed by your method. The zip should contain the **contents** of your
submission folder, not the submission folder itself.

From inside your submission folder, run:

```bash
cd /path/to/your_submission_folder
zip -r ../your_submission.zip . \
  -x "*/__pycache__/*" "*.DS_Store" ".git/*"
```

Before uploading, you can check the zip root:

```bash
unzip -l ../your_submission.zip | head
```

You should see `run.py` and `metadata.yaml` at the top level of the archive,
not nested under a project folder.

Upload the resulting zip to the MoCha2026 CodaBench challenge.

## Adapting This Baseline

- Keep `run.py` as a small CodaBench wrapper.
- Replace `submission/baseline_model.py` with your own model loading and
  prediction logic.
- Replace `submission/preprocess.py` if your method uses a different motion
  representation.
- Include all model weights and pure-Python helper code needed at inference
  time.
- Do not rely on downloading packages, weights, or data during evaluation.

## Model Files

The baseline weights are stored with Git LFS because one checkpoint is larger
than GitHub's regular file-size limit. After cloning, run:

```bash
git lfs pull
```

if the large weight files were not downloaded automatically.

## Training Data

Participants can train on the CARE-PD dataset:

- Website: <https://neurips2025.care-pd.ca/>
- Hugging Face: <https://huggingface.co/datasets/vida-adl/CARE-PD>
- Canonicalized SMPL pickles:
  <https://huggingface.co/datasets/vida-adl/CARE-PD/tree/main/Canonicalized_SMPL_pickles>

## Contact

For challenge questions, use the CodaBench forum or contact the organizers:

<mocha.eccv@gmail.com>
