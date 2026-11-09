"""
Do the indicators agree with the examiner?  (This is what your own dataset is FOR.)

Reads the CSVs exported from the Research page (all *.csv in data/datasets/, merged, duplicates dropped) and,
for each student performance indicator, reports how strongly it moves with the examiner's own marks:
  * overall mark (lecturer_label)            - from the rubric
  * the matching rubric criterion rating      - e.g. promptness vs "Responsiveness to questions"
using Spearman rank correlation with a bootstrap 95% confidence interval. Nothing is trained or tuned here -
it simply tells you, honestly, how far each indicator can be trusted as evidence.

    python -m scripts.validate_metrics            # uses data/datasets/*.csv
    python -m scripts.validate_metrics --csv a.csv b.csv
"""
import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

KEYS = ["promptness", "fluency", "depth", "technical", "consistency"]
# which examiner rubric criterion each indicator is supposed to reflect
CRITERION = {"promptness": "responsiveness", "fluency": "communication", "depth": "depth",
             "technical": "technical_knowledge", "consistency": "responsiveness"}
MIN_PAIRS = 8


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    i = 0
    xs = x[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 7) -> Tuple[Optional[float], Optional[float]]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        v = spearman(a[idx], b[idx])
        if v is not None:
            vals.append(v)
    if len(vals) < n // 2:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def strength(rho: Optional[float]) -> str:
    if rho is None:
        return "not computable"
    a = abs(rho)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak"


def load_rows(files: List[str]) -> List[Dict[str, str]]:
    rows, seen = [], set()
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r.get("student_pseudo_id"), r.get("duration_s"), r.get("lecturer_label"),
                       tuple(r.get(f"perf_{k}") for k in KEYS))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
    return rows


def pairs(rows: List[Dict[str, str]], x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        try:
            x, y = float(r[x_col]), float(r[y_col])
        except (KeyError, TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def analyse(rows: List[Dict[str, str]]) -> Dict[str, object]:
    out: Dict[str, object] = {"n_sessions": len(rows), "indicators": {}}
    for k in KEYS:
        entry: Dict[str, object] = {}
        for name, y_col in (("vs_overall_mark", "lecturer_label"), (f"vs_{CRITERION[k]}_rating", f"rating_{CRITERION[k]}")):
            x, y = pairs(rows, f"perf_{k}", y_col)
            if len(x) < MIN_PAIRS:
                entry[name] = {"n": int(len(x)), "rho": None, "ci95": None, "verdict": f"needs at least {MIN_PAIRS} pairs"}
                continue
            rho = spearman(x, y)
            lo, hi = bootstrap_ci(x, y)
            entry[name] = {"n": int(len(x)), "rho": None if rho is None else round(rho, 3),
                           "ci95": None if lo is None else [round(lo, 3), round(hi, 3)], "verdict": strength(rho)}
        out["indicators"][k] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+")
    ap.add_argument("--dir", default="data/datasets")
    ap.add_argument("--out", default="models/validation_report.json")
    args = ap.parse_args()

    files = args.csv or sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {args.dir}/ (export them from the Research page).")
    rows = load_rows(files)
    result = analyse(rows)

    print(f"Sessions: {result['n_sessions']} (from {len(files)} file(s))\n")
    print(f"{'Indicator':<13}{'Compared with':<34}{'n':>4}{'rho':>8}   95% CI          Agreement")
    for k, entry in result["indicators"].items():
        for name, e in entry.items():
            ci = f"[{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]" if e["ci95"] else "-"
            rho = f"{e['rho']:+.2f}" if e["rho"] is not None else "-"
            label = "overall examiner mark" if name == "vs_overall_mark" else name[3:-7].replace("_", " ") + " rating"
            print(f"{k:<13}{label:<34}{e['n']:>4}{rho:>8}   {ci:<16}{e['verdict']}")
    print("\nReport these numbers as they are. A weak result is a legitimate finding (it means the indicator should be")
    print("treated as supporting evidence only); a strong one needs a sample large enough to be believed.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
