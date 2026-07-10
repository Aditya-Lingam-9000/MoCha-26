# Dataset Exploration: CARE-PD

This document outlines the detailed structure of the dataset provided in `CARE-PD/Canonicalized_SMPL_pickles`, which acts as our training dataset for the MoCha2026 challenge.

## 1. Directory Structure and Files
The directory contains the following `*_canonical.pkl` files corresponding to 9 distinct cohorts:
1. **`3DGait_canonical.pkl`** (~8 MB)
2. **`BMCLab_canonical.pkl`** (~127 MB)
3. **`DNE_canonical.pkl`** (~23 MB)
4. **`E-LC_canonical.pkl`** (~437 MB)
5. **`KUL-DT-T_canonical.pkl`** (~116 MB)
6. **`PD-GaM_canonical.pkl`** (~84 MB)
7. **`T-LTC_canonical.pkl`** (~106 MB)
8. **`T-SDU-PD_canonical.pkl`** (~26 MB)
9. **`T-SDU_canonical.pkl`** (~185 MB)

**Total Size:** ~1.1 GB. This is extremely manageable.

## 2. Data Schema per Pickle
Each pickle file is a nested Python dictionary adhering to the following structure:
```python
{
    "anonymized_subject_id": {
        "anonymized_walk_id": {
            "pose": np.ndarray,      # Shape: (T, 72) - SMPL axis-angle poses
            "trans": np.ndarray,     # Shape: (T, 3) - Global translation (x, y, z)
            "beta": np.ndarray,      # Shape: (1, 10) - All zeros (anonymized shape)
            "fps": int,              # Original frames per second
            "UPDRS_GAIT": int,       # Target Label: 0, 1, 2, or 3
            # Other metadata may exist but is mostly for research.
        }
    }
}
```

## 3. Preprocessing Characteristics
The prefix `Canonicalized` means:
- The data is rotated into a shared coordinate system: `x=lateral, y=up, z=forward`.
- Translation in the first frame starts at `x=0, z=0`.
- The body stands on the floor plane `y=0`.
- Missing target labels (`UPDRS_GAIT`) might appear as `None`. We must filter these out during our offline feature extraction since we are doing supervised learning.

## 4. Dataset Constraints & Hardware Implications
If we loaded all 1.1 GB of pickle dictionaries, along with NumPy arrays expanded into memory, it might spike to 3-4 GB. However, passing these through a PyTorch model dynamically during training epochs would cause massive memory duplication and likely exceed our 8GB limit. **This reinforces the strategy of processing one `.pkl` file at a time, running the feature extractor, deleting the raw arrays from memory, and saving the aggregated lightweight 512D feature vectors to disk.**
