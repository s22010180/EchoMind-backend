"""
Optional data-driven scorer.

After you have collected labelled vivas (examiner rubric score per session), run
`python -m scripts.train_scorer` to fit a regularised regression on the session
feature vectors. If models/scorer.joblib exists, its prediction is added to each
report as `predicted_rating`. Until then the rule-based indicators are used alone.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LearnedScorer:
    def __init__(self, path: str):
        self.model = None
        self.feature_names = []
        self.label_range = (0.0, 100.0)
        if not path or not os.path.exists(path):
            return
        try:
            import joblib

            blob = joblib.load(path)
            self.model = blob["model"]
            self.feature_names = list(blob["feature_names"])
            self.label_range = tuple(blob.get("label_range", (0.0, 100.0)))
            logger.info("Loaded learned scorer (%d features)", len(self.feature_names))
        except Exception:
            logger.exception("Could not load learned scorer from %s", path)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, features: Dict[str, Optional[float]]) -> Optional[float]:
        if not self.available:
            return None
        x = np.array([[np.nan if features.get(n) is None else float(features[n]) for n in self.feature_names]])
        try:
            y = float(self.model.predict(x)[0])
        except Exception:
            logger.exception("Learned scorer prediction failed")
            return None
        lo, hi = self.label_range
        return round(float(np.clip(y, lo, hi)), 1)
