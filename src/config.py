from typing import List

CATEGORICAL_FEATURES: List[str] = [
    "course_difficulty",
    "course_type",
    "age_group",
    "device_used",
    "has_certification",
]

NUMERIC_FEATURES: List[str] = [
    "past_course_history",
    "weekly_login_frequency",
    "avg_time_per_session_minutes",
    "course_length_hours",
]

LABEL_COLUMN: str = "completion"

ALLOWED_VALUES = {
    "course_difficulty": ["easy", "medium", "hard"],
    "course_type": ["video", "reading", "coding", "combined"],
    "age_group": ["18-25", "26-35", "36-45", "46-60", "60+"],
    "device_used": ["mobile", "desktop"],
    "has_certification": ["Y", "N"],
}