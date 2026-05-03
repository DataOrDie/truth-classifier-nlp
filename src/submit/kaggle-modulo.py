#!/usr/bin/env python
"""Create Kaggle submission CSVs from saved LR training artifacts."""

from pathlib import Path
import argparse
import sys

import joblib
import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
default_project_root = script_dir.parents[1]
src_path = default_project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing.one_step import preprocess_one_step
from submit.save_model import load_model


def generate_submission_csv(
    model_name: str,
    project_root: Path | None = None,
    test_data_path: Path | None = None,
    submissions_dir: Path | None = None,
    is_tree_model: bool = False,  # accepted for call-site compatibility, not used
    verbose: bool = True,
) -> Path:
    root = project_root or default_project_root
    test_path = test_data_path or (root / "data" / "test_nolabel.csv")
    out_dir = submissions_dir or (root / "submissions")
    model_dir = root / "models" / model_name

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_path}")

    df_test = pd.read_csv(test_path)
    if verbose:
        print(f"[INFO] Test rows: {len(df_test):,}  columns: {df_test.columns.tolist()}")

    artifacts = load_model(project_root=root, model_name=model_name)
    model_pipeline = artifacts["model"]
    options = artifacts["options"]
    feature_names = artifacts["features"]

    if verbose:
        print(f"[INFO] Model type: {type(model_pipeline).__name__}")
        print(f"[INFO] Training features: {len(feature_names)}")

    # Load threshold (default 0.5 if not saved).
    threshold_path = model_dir / f"{model_name}-threshold.joblib"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5
    if verbose:
        print(f"[INFO] Decision threshold: {threshold:.2f}")

    # Load fitted vectorizer and inject into options so test text is transformed
    # with the exact same vocabulary that was built on training data.
    vectorizer_path = model_dir / f"{model_name}-vectorizer.joblib"
    if vectorizer_path.exists():
        fitted_vec = joblib.load(vectorizer_path)
        options.statement_fitted_vectorizer = fitted_vec
        if verbose:
            print(f"[INFO] Vectorizer loaded  (vocab: {len(fitted_vec.vocabulary_):,})")
    else:
        if verbose:
            print("[WARNING] No saved vectorizer found — test text will be fit independently (vocabulary mismatch risk)")

    # Test data has no label column — skip the label preprocessing step entirely.
    options.label_option = "skip"

    if verbose:
        print("[SECTION] Preprocessing test data")

    df_test_processed = preprocess_one_step(df_test, options=options)

    if verbose:
        print(f"  Processed shape: {df_test_processed.shape}")

    # Select only numeric columns, then align to the exact feature set used at training.
    X_test = (
        df_test_processed
        .select_dtypes(exclude="object")
        .reindex(columns=feature_names, fill_value=0)
        .fillna(0)
    )

    if verbose:
        missing = set(feature_names) - set(df_test_processed.select_dtypes(exclude="object").columns)
        if missing:
            print(f"  [WARNING] {len(missing)} features missing from test — filled with 0: {sorted(missing)[:5]}...")
        print(f"  Feature matrix: {X_test.shape}")

    # Predict with probability threshold.
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    if verbose:
        unique, counts = np.unique(y_pred, return_counts=True)
        print("[INFO] Prediction distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count:,} ({100.0 * count / len(y_pred):.1f}%)")

    out_dir.mkdir(exist_ok=True)
    submission_df = pd.DataFrame({"id": df_test["id"].values, "label": y_pred})
    submission_path = out_dir / f"submission-{model_name}.csv"
    submission_df.to_csv(submission_path, index=False)

    if verbose:
        print(f"[INFO] Submission saved: {submission_path}  ({len(submission_df):,} rows)")

    return submission_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle submission CSV from a saved LR model artifact.",
    )
    parser.add_argument("model_name", help="Model artifact folder name under models/")
    parser.add_argument("--project-root", dest="project_root", default=None)
    parser.add_argument("--test-data-path", dest="test_data_path", default=None)
    parser.add_argument("--submissions-dir", dest="submissions_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_submission_csv(
        model_name=args.model_name,
        project_root=Path(args.project_root).resolve() if args.project_root else None,
        test_data_path=Path(args.test_data_path).resolve() if args.test_data_path else None,
        submissions_dir=Path(args.submissions_dir).resolve() if args.submissions_dir else None,
        verbose=True,
    )


if __name__ == "__main__":
    main()
