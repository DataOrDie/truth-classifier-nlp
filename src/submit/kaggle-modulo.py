#!/usr/bin/env python
"""Create Kaggle submission CSVs from saved training artifacts."""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

# Resolve paths from this file so running from any cwd still works.
script_dir = Path(__file__).resolve().parent
default_project_root = script_dir.parents[1]
src_path = default_project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing.one_step import preprocess_one_step
from submit.save_model import load_model


def _compute_model_scores(model_pipeline, X_test: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return a numeric score vector for diagnostics if the model supports it."""

    if hasattr(model_pipeline, "decision_function"):
        try:
            scores = np.asarray(model_pipeline.decision_function(X_test), dtype=float)
            return scores, "decision_function"
        except Exception:
            pass

    if hasattr(model_pipeline, "predict_proba"):
        probabilities = np.asarray(model_pipeline.predict_proba(X_test), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1], "predict_proba[:, 1]"
        return probabilities.reshape(-1), "predict_proba"

    return np.full(len(X_test), np.nan, dtype=float), "unavailable"


def generate_submission_csv(
    model_name: str,
    project_root: Path | None = None,
    test_data_path: Path | None = None,
    submissions_dir: Path | None = None,
    is_tree_model: bool = False,
    verbose: bool = True,
) -> Path:
    """Generate a Kaggle submission CSV from a saved model.

    Args:
        model_name: Artifact folder name under models/.
        project_root: Repository root. Defaults to this file's repo root.
        test_data_path: Optional explicit test CSV path.
        submissions_dir: Optional explicit submission output directory.
        is_tree_model: Whether to apply tree-focused engineered features.
        verbose: Whether to print progress/details.

    Returns:
        Path to the generated submission CSV.
    """

    root = project_root or default_project_root
    test_path = test_data_path or (root / "data" / "test_nolabel.csv")
    out_dir = submissions_dir or (root / "submissions")

    if verbose:
        print(f"[INFO] Model name: {model_name}")
        print("[SECTION] Loading test dataset")

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_path}")

    df_test = pd.read_csv(test_path)
    if verbose:
        print(f"Test dataset shape: {df_test.shape}")
        print(f"Columns: {df_test.columns.tolist()}")
        print(f"Sample IDs: {df_test['id'].head().tolist()}")

    artifacts = load_model(
        project_root=root,
        model_name=model_name,
    )

    model_pipeline = artifacts["model"]
    options = artifacts["options"]
    feature_names = artifacts["features"]

    # Backward compatibility for older saved options artifacts.
    if not hasattr(options, "accept_option"):
        options.accept_option = "skip"

    # Force row-preserving NewExist preprocessing for submissions.
    options.newexist_option = "A"

    if verbose:
        print(f"\nModel type: {type(model_pipeline)}")
        print(f"Number of features: {len(feature_names)}\n")
        print("[SECTION] Preprocessing test data")
        print("Preprocessing options:")
        print(f"  noemp_option: {options.noemp_option}")
        print(f"  newexist_option: {options.newexist_option}")
        print(f"  createjob_option: {options.createjob_option}")
        print(f"  retainedjob_option: {options.retainedjob_option}")
        print(f"  approvaldate_option: {options.approvaldate_option}")
        print(f"  approvalfy_option: {options.approvalfy_option}")
        print(f"  franchise_option: {options.franchise_option}")
        print(f"  urbanrural_option: {options.urbanrural_option}")
        print(f"  revlinecr_option: {options.revlinecr_option}")
        print(f"  lowdoc_option: {options.lowdoc_option}")
        print(f"  disbursementgross_option: {options.disbursementgross_option}")
        print(f"  accept_option: {options.accept_option}")
        print(f"  local_state: {options.local_state}")

    df_test_processed = preprocess_one_step(
        df_test,
        options=options,
        is_tree_model=is_tree_model,
    )
    if verbose:
        print(f"Processed test dataset shape: {df_test_processed.shape}")
        print(f"Processed features: {df_test_processed.shape[1]}")

    # Some preprocessing options can drop rows.
    if len(df_test_processed) != len(df_test):
        dropped_count = len(df_test) - len(df_test_processed)
        if verbose:
            print(
                f"WARNING: Preprocessing changed row count by {dropped_count}. "
                "Restoring original row index for full-length submission."
            )
        df_test_processed = df_test_processed.reindex(df_test.index)

    if verbose:
        print("[SECTION] Verifying feature alignment")
    processed_features = df_test_processed.columns.tolist()

    if set(processed_features) != set(feature_names):
        missing_in_test = set(feature_names) - set(processed_features)
        extra_in_test = set(processed_features) - set(feature_names)

        if verbose and missing_in_test:
            print(f"WARNING: Missing features in test data: {missing_in_test}")
        if verbose and extra_in_test:
            print(f"WARNING: Extra features in test data: {extra_in_test}")

        # Robust alignment: add missing training columns with 0 and drop extras.
        df_test_processed = df_test_processed.reindex(columns=feature_names, fill_value=0)
        if verbose:
            print("Features realigned to match training schema (missing -> 0, extras dropped)")
    else:
        # Ensure same order as training.
        df_test_processed = df_test_processed[feature_names]
        if verbose:
            print("Features match training data perfectly")

    # Fill any missing values introduced by row restoration/alignment.
    df_test_processed = df_test_processed.fillna(0)

    if verbose:
        print("[DEBUG] Final df_test_processed columns:")
        for idx, col in enumerate(df_test_processed.columns.tolist(), start=1):
            print(f"  {idx:02d}. {col}")

    if verbose:
        print("[SECTION] Generating predictions")
    X_test = df_test_processed
    if verbose:
        print(f"Input shape for model: {X_test.shape}")

    y_pred = model_pipeline.predict(X_test)
    if verbose:
        print(f"Predictions generated. Shape: {y_pred.shape}")
        print("Prediction distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    if verbose:
        for label, count in zip(unique, counts):
            pct = 100.0 * count / len(y_pred)
            print(f"  Class {label}: {count:6d} ({pct:6.2f}%)")

    y_score, score_source = _compute_model_scores(model_pipeline, X_test)
    if verbose:
        if np.isnan(y_score).all():
            print("Decision score unavailable for this model pipeline")
        else:
            print(
                f"Decision scores ({score_source}) - "
                f"Mean: {np.nanmean(y_score):.4f}, Std: {np.nanstd(y_score):.4f}"
            )

    if verbose:
        print("[SECTION] Creating submission file")
    out_dir.mkdir(exist_ok=True)

    submission_ids = df_test["id"].values
    if verbose:
        print(f"Submission IDs: {len(submission_ids)} | Predictions: {len(y_pred)}")

    if len(submission_ids) != len(y_pred):
        raise ValueError(
            "Submission length mismatch: "
            f"ids={len(submission_ids)}, preds={len(y_pred)}"
        )

    submission_df = pd.DataFrame(
        {
            "id": submission_ids,
            "Accept": y_pred.astype(int),
        }
    )

    submission_path = out_dir / f"submission-{model_name}.csv"
    submission_df.to_csv(submission_path, index=False)

    if verbose:
        print(f"Submission file created: {submission_path}")
        print(f"Submission shape: {submission_df.shape}")
        # print("\nFirst 10 submissions:")
        # print(submission_df.head(10).to_string(index=False))

    return submission_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle submission CSV from a saved model artifact.",
    )
    parser.add_argument("model_name", help="Model artifact folder name under models/")
    parser.add_argument(
        "--project-root",
        dest="project_root",
        default=None,
        help="Optional project root path (defaults to repo root)",
    )
    parser.add_argument(
        "--test-data-path",
        dest="test_data_path",
        default=None,
        help="Optional explicit path to test CSV",
    )
    parser.add_argument(
        "--submissions-dir",
        dest="submissions_dir",
        default=None,
        help="Optional output directory for submission CSV",
    )
    parser.add_argument(
        "--is-tree-model",
        dest="is_tree_model",
        action="store_true",
        help="Apply tree-focused engineered features in preprocessing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_submission_csv(
        model_name=args.model_name,
        project_root=Path(args.project_root).resolve() if args.project_root else None,
        test_data_path=Path(args.test_data_path).resolve() if args.test_data_path else None,
        submissions_dir=Path(args.submissions_dir).resolve() if args.submissions_dir else None,
        is_tree_model=args.is_tree_model,
        verbose=True,
    )


if __name__ == "__main__":
    main()


