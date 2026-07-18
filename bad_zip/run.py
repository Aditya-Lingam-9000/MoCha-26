"""CodaBench submission entry point.

CodaBench imports this file and calls `predict(data)`. Participants can keep
this wrapper mostly unchanged and replace the model implementation underneath.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from submission.baseline_model import Model


_MODEL: Model | None = None


def get_model() -> Model:
    """Load model weights once, then reuse the model for all samples."""
    global _MODEL
    if _MODEL is None:
        _MODEL = Model()
    return _MODEL


def predict(data: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, int]]:
    """Return predictions[subject_id][walk_id] = UPDRS class in {0, 1, 2, 3}."""
    model = get_model()
    return model.predict_dataset(data)


def main() -> None:
    """Optional local runner; CodaBench uses predict(data) directly."""
    parser = argparse.ArgumentParser(description="Run local MoCha baseline inference.")
    parser.add_argument("--input", required=True, type=Path, help="Challenge input .pkl file")
    parser.add_argument("--output", required=True, type=Path, help="Where to save predictions.json")
    args = parser.parse_args()

    with args.input.open("rb") as f:
        data = pickle.load(f)
    predictions = predict(data)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"Saved predictions for {sum(len(walks) for walks in predictions.values())} samples.")


if __name__ == "__main__":
    main()
