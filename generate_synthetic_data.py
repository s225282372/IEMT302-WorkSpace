import argparse
import os
from typing import List
import numpy as np
import pandas as pd

from src.config import ALLOWED_VALUES


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_choices(choices: List[str], size: int, p: List[float] | None = None) -> List[str]:
    return list(np.random.choice(choices, size=size, p=p))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic course completion dataset")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--out_path", type=str, default="data/training_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.rows

    # Features
    course_difficulty = sample_choices(ALLOWED_VALUES["course_difficulty"], n, p=[0.4, 0.4, 0.2])
    course_type = sample_choices(ALLOWED_VALUES["course_type"], n, p=[0.45, 0.2, 0.2, 0.15])
    age_group = sample_choices(ALLOWED_VALUES["age_group"], n, p=[0.35, 0.35, 0.18, 0.10, 0.02])
    device_used = sample_choices(ALLOWED_VALUES["device_used"], n, p=[0.55, 0.45])
    has_certification = sample_choices(ALLOWED_VALUES["has_certification"], n, p=[0.3, 0.7])

    past_course_history = rng.integers(0, 21, size=n)
    weekly_login_frequency = rng.integers(0, 8, size=n)
    avg_time_per_session_minutes = rng.normal(loc=30, scale=12, size=n)
    avg_time_per_session_minutes = np.clip(avg_time_per_session_minutes, 2, 120)
    course_length_hours = rng.choice([1, 3, 5, 8, 10, 15, 20, 30, 40], size=n, p=[0.05, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.07, 0.03])

    # Map effects into a latent score z
    diff_effect = np.array([{"easy": 0.7, "medium": 0.0, "hard": -0.6}[d] for d in course_difficulty])
    type_effect = np.array([{"video": 0.2, "reading": -0.05, "coding": 0.15, "combined": 0.25}[t] for t in course_type])
    age_effect = np.array([
        {"18-25": -0.05, "26-35": 0.15, "36-45": 0.1, "46-60": -0.05, "60+": -0.25}[a]
        for a in age_group
    ])
    device_effect = np.array([{"mobile": -0.08, "desktop": 0.08}[d] for d in device_used])
    cert_effect = np.array([{"Y": 0.2, "N": 0.0}[c] for c in has_certification])

    past_effect = 0.05 * np.sqrt(past_course_history)
    login_effect = 0.12 * np.minimum(weekly_login_frequency, 7)
    session_effect = 0.015 * np.minimum(avg_time_per_session_minutes, 90)
    length_effect = -0.015 * np.minimum(course_length_hours, 40)

    # Combine with noise and base rate
    base = -0.2
    noise = rng.normal(0, 0.5, size=n)

    z = (
        base
        + diff_effect
        + type_effect
        + age_effect
        + device_effect
        + cert_effect
        + past_effect
        + login_effect
        + session_effect
        + length_effect
        + noise
    )

    prob = logistic(z)
    completion_numeric = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "course_difficulty": course_difficulty,
            "course_type": course_type,
            "age_group": age_group,
            "past_course_history": past_course_history,
            "weekly_login_frequency": weekly_login_frequency,
            "avg_time_per_session_minutes": avg_time_per_session_minutes,
            "device_used": device_used,
            "course_length_hours": course_length_hours,
            "has_certification": has_certification,
            "completion": ["Y" if x == 1 else "N" for x in completion_numeric],
        }
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_csv(args.out_path, index=False)
    print(f"Wrote {len(df)} rows to {args.out_path}")


if __name__ == "__main__":
    main()