import argparse
import json
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd


EXAMPLE_RECORD = {
    "course_difficulty": "medium",
    "course_type": "video",
    "age_group": "26-35",
    "past_course_history": 3,
    "weekly_login_frequency": 5,
    "avg_time_per_session_minutes": 35,
    "device_used": "desktop",
    "course_length_hours": 6,
    "has_certification": "Y",
}


def load_records(path: str | None) -> List[Dict[str, Any]]:
    if path is None:
        return [EXAMPLE_RECORD]
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("input_json must contain an object or an array of objects")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with course completion model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, default=None, help="Path to JSON file with records")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    model = joblib.load(args.model_path)

    records = load_records(args.input_json)
    df = pd.DataFrame.from_records(records)

    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= args.threshold).astype(int)

    outputs = []
    for record, p, y in zip(records, proba, preds):
        outputs.append(
            {
                "input": record,
                "prob_complete": float(p),
                "prediction_label": "Y" if y == 1 else "N",
                "threshold": args.threshold,
            }
        )

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(outputs, f, indent=2)
        print(f"Wrote predictions to {args.output_path}")
    else:
        print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()