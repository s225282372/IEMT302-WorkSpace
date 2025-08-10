## Course Completion Predictor

This project trains a model to predict whether a student will complete an online course based on behavior and course details. It outputs calibrated probabilities (e.g., 0.50 means 50% chance of completion) and a Y/N label at a configurable threshold.

### Features
- course_difficulty: easy, medium, hard
- course_type: video, reading, coding, combined
- age_group: 18-25, 26-35, 36-45, 46-60, 60+
- past_course_history: integer (number of courses completed)
- weekly_login_frequency: integer (0-7)
- avg_time_per_session_minutes: numeric (e.g., 5, 20, 45)
- device_used: mobile, desktop
- course_length_hours: numeric (e.g., 1, 5, 10)
- has_certification: Y or N

### Label
- completion: Y or N

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate synthetic data
```bash
python generate_synthetic_data.py --rows 5000 --out_path data/training_data.csv
```

### Train and evaluate
```bash
python train.py --data_path data/training_data.csv --output_dir artifacts
```
Artifacts saved to `artifacts/`:
- `model.joblib`: trained pipeline
- `metrics.json`: evaluation metrics

### Inference
Create a JSON file with one or more records matching the schema, e.g. `sample.json`:
```json
[
  {
    "course_difficulty": "medium",
    "course_type": "video",
    "age_group": "26-35",
    "past_course_history": 3,
    "weekly_login_frequency": 5,
    "avg_time_per_session_minutes": 35,
    "device_used": "desktop",
    "course_length_hours": 6,
    "has_certification": "Y"
  }
]
```
Run inference:
```bash
python infer.py --model_path artifacts/model.joblib --input_json sample.json --threshold 0.5 --output_path predictions.json
```
Output file contains probabilities and Y/N predictions.

### Notes
- The model outputs probabilities; adjust `--threshold` based on your precision/recall tradeoff.
- For real data, ensure the columns and allowed values match the schema above.