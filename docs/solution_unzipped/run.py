"""
Submission entry point for MoCha2026.

CodaBench imports this file and calls:
    predictions = predict(data)

The returned object must be a nested dictionary:
    predictions[subject_id][walk_id] = predicted_UPDRS_GAIT_class

Valid prediction classes are integers 0, 1, 2, and 3.
"""

import argparse
import json
import pickle
from pathlib import Path

from model import Model


def predict(data):
    model = Model()
    predictions = {}

    for subject_id, subject_data in data.items():
        predictions[str(subject_id)] = {}
        for walk_id, sample in subject_data.items():
            pred = model.predict(sample)
            predictions[str(subject_id)][str(walk_id)] = int(round(float(pred)))

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Local test runner for MoCha2026 submissions.')
    parser.add_argument('--input', type=Path, required=True, help='Path to a challenge input .pkl file')
    parser.add_argument('--output', type=Path, default=None, help='Optional JSON file for local predictions')
    args = parser.parse_args()

    with args.input.open('rb') as f:
        data = pickle.load(f)
    predictions = predict(data)

    if args.output is not None:
        with args.output.open('w', encoding='utf-8') as f:
            json.dump(predictions, f)
    else:
        print(json.dumps(predictions, indent=2))


if __name__ == '__main__':
    main()
