"""
Train a data-driven scorer from YOUR collected, labelled vivas.

1. Run vivas, then for each one call  PATCH /api/sessions/{id}/label  with the examiner's
   rubric score (0-100).
2. Download  GET /api/sessions/export/dataset.csv  (as the lecturer) -> data/dataset.csv
3. Train:     python -m scripts.train_scorer --csv data/dataset.csv
4. Restart the API. Reports now include `predicted_rating` next to the rule-based indicators.

With few sessions the model is a regularised linear regression (RidgeCV) on standardised
features with median imputation - deliberately simple so it is hard to overfit.
Cross-validated MAE / R^2 are printed: report them honestly in your thesis, and use
student-wise splits if the same student appears in several sessions.
"""
import argparse
import csv
import os

import numpy as np

from app.services.scoring import FEATURE_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="models/scorer.joblib")
    ap.add_argument("--min-rows", type=int, default=20)
    args = ap.parse_args()

    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("lecturer_label", "") != ""]
    if len(rows) < args.min_rows:
        raise SystemExit(f"Only {len(rows)} labelled sessions; need at least {args.min_rows}.")

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    X_all = np.array([[num(r.get(n)) for n in FEATURE_NAMES] for r in rows])
    y = np.array([float(r["lecturer_label"]) for r in rows])
    keep = [i for i in range(X_all.shape[1]) if not np.all(np.isnan(X_all[:, i]))
            and np.nanstd(X_all[:, i]) > 1e-9]
    names = [FEATURE_NAMES[i] for i in keep]
    X = X_all[:, keep]

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 3, 20)))
    cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    pred = cross_val_predict(model, X, y, cv=cv)
    print(f"Sessions: {len(y)} | features used: {len(names)}")
    print(f"Cross-validated MAE: {mean_absolute_error(y, pred):.2f} | R^2: {r2_score(y, pred):.3f}")
    print(f"Baseline MAE (always predict the mean): {np.mean(np.abs(y - y.mean())):.2f}")

    model.fit(X, y)
    coefs = model[-1].coef_
    print("\nMost influential features (standardised coefficients):")
    for i in np.argsort(-np.abs(coefs))[:10]:
        print(f"  {names[i]:28s} {coefs[i]:+.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"model": model, "feature_names": names,
                 "label_range": (float(y.min()), float(y.max())), "n_train": int(len(y))}, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
