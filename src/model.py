from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_model(random_state: int = 42, calibrate: bool = True) -> Pipeline:
    base_model = GradientBoostingClassifier(random_state=random_state)

    if calibrate:
        classifier = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
    else:
        classifier = base_model

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", classifier),
        ]
    )
    return pipeline