#!/usr/bin/env python
"""Create Kaggle submission CSVs from saved tree model artifacts.

Handles the full inference pipeline for tree models trained with rfc.py or
any equivalent script that saves an OrdinalEncoder alongside the model.

Key difference from kaggle-modulo.py: after preprocessing this script applies
the saved OrdinalEncoder to the grouped categorical and interaction-key string
columns before assembling the feature matrix. Without this step those columns
would be silently dropped, and the model would predict on a mismatched feature
set.
"""

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


def generate_tree_submission_csv(
    model_name: str,
    project_root: Path | None = None,
    test_data_path: Path | None = None,
    submissions_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """
    Generate a Kaggle submission CSV from a saved tree model.

    Parameters
    ----------
    model_name : str
        Artifact folder name under models/ (e.g. "rfc").
    project_root : Path, optional
        Project root directory. Defaults to two levels above this file.
    test_data_path : Path, optional
        Path to test CSV. Defaults to data/test_nolabel.csv.
    submissions_dir : Path, optional
        Output directory for the CSV. Defaults to submissions/.
    verbose : bool
        Print progress information.

    Returns
    -------
    Path
        Path to the written submission CSV.

    Raises
    ------
    FileNotFoundError
        If the test data or OrdinalEncoder artifact is missing.
    RuntimeError
        If the OrdinalEncoder was not fit on a DataFrame (feature_names_in_ missing).
    """
    root     = project_root or default_project_root
    test_path = test_data_path or (root / "data" / "test_nolabel.csv")
    out_dir  = submissions_dir or (root / "submissions")
    model_dir = root / "models" / model_name

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_path}")

    # -------------------------------------------------------------------------
    # Load model artifacts
    # -------------------------------------------------------------------------
    if verbose:
        print(f"[INFO] Loading model artifacts from: {model_dir}")

    artifacts      = load_model(project_root=root, model_name=model_name)
    model_pipeline = artifacts["model"]
    options        = artifacts["options"]
    feature_names  = artifacts["features"]

    if verbose:
        print(f"  Model type        : {type(model_pipeline).__name__}")
        print(f"  Training features : {len(feature_names)}")

    # Load decision threshold (default 0.5 if the artifact is missing)
    threshold_path = model_dir / f"{model_name}-threshold.joblib"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5
    if verbose:
        print(f"  Decision threshold: {threshold:.2f}")

    # Load OrdinalEncoder — required for tree models
    enc_path = model_dir / f"{model_name}-ordinal-encoder.joblib"
    if not enc_path.exists():
        raise FileNotFoundError(
            f"OrdinalEncoder artifact not found at {enc_path}.\n"
            "Ensure the model was trained with rfc.py or an equivalent tree training script."
        )
    ordinal_enc = joblib.load(enc_path)

    # feature_names_in_ is set by sklearn when the encoder is fit on a DataFrame.
    # It records the column names in the exact order used during fit — required for
    # safe transform on test data.
    if not hasattr(ordinal_enc, "feature_names_in_"):
        raise RuntimeError(
            "OrdinalEncoder.feature_names_in_ is not available. "
            "The encoder must have been fit on a pandas DataFrame (requires sklearn >= 1.0)."
        )
    train_cat_cols = ordinal_enc.feature_names_in_.tolist()
    if verbose:
        print(f"  OrdinalEncoder    : {len(train_cat_cols)} categorical columns")

    # Load the fitted vectorizer and inject it into options so the test statements
    # are projected into the exact vocabulary built on training data.
    # SentenceTransformer (embeddings) is stateless — no saved file needed.
    vectorizer_path = model_dir / f"{model_name}-vectorizer.joblib"
    if options.statement_vectorizer_type not in ("embeddings", "none") and vectorizer_path.exists():
        fitted_vec = joblib.load(vectorizer_path)
        options.statement_fitted_vectorizer = fitted_vec
        if verbose:
            print(f"  Vectorizer        : loaded (vocab: {len(fitted_vec.vocabulary_):,})")
    elif options.statement_vectorizer_type == "embeddings":
        if verbose:
            print(f"  Vectorizer        : embeddings '{options.statement_embedding_model}' (stateless)")
    else:
        if verbose:
            print("[WARNING] No saved vectorizer found — vocabulary may not match training")

    # Test data has no label column; 'skip' leaves the column alone if it exists
    # and does nothing if it does not.
    options.label_option = "skip"

    # Load true-rate maps: per-speaker / per-subject / per-party false-claim rates
    # computed from the training set. Applied after preprocessing so that test rows
    # get the credibility score of their speaker/subject/party without seeing labels.
    rate_maps_path = model_dir / f"{model_name}-true-rate-maps.joblib"
    _true_rate_artifacts = None
    if rate_maps_path.exists():
        _true_rate_artifacts = joblib.load(rate_maps_path)
        if verbose:
            print(f"  True-rate maps    : {list(_true_rate_artifacts['rate_maps'].keys())}")

    # -------------------------------------------------------------------------
    # Load and preprocess test data
    # -------------------------------------------------------------------------
    df_test = pd.read_csv(test_path)
    if verbose:
        print(f"\n[INFO] Test rows: {len(df_test):,}  columns: {df_test.columns.tolist()}")
        print("[SECTION] Preprocessing test data")

    df_test_processed = preprocess_one_step(df_test, options=options)
    if verbose:
        print(f"  Processed shape: {df_test_processed.shape}")

    # -------------------------------------------------------------------------
    # Apply true-rate maps
    # The source columns are string categoricals still present in df_test_processed.
    # Must happen BEFORE OrdinalEncoding so the added float columns end up in the
    # numeric portion of the final feature matrix.
    # -------------------------------------------------------------------------
    if _true_rate_artifacts is not None:
        _rate_maps  = _true_rate_artifacts["rate_maps"]
        _group_cols = _true_rate_artifacts["group_cols"]
        _fallback   = _true_rate_artifacts["fallback"]
        for _feat, _src_col in _group_cols.items():
            if _src_col in df_test_processed.columns:
                df_test_processed[_feat] = (
                    df_test_processed[_src_col].map(_rate_maps[_feat]).fillna(_fallback)
                )
            else:
                df_test_processed[_feat] = _fallback
                if verbose:
                    print(f"  [WARNING] Source column '{_src_col}' missing from test — using fallback {_fallback}")

    # -------------------------------------------------------------------------
    # OrdinalEncoding
    # Replicate the encoding step from the training script.
    # The encoder was fit on train_cat_cols in a specific order; transform must
    # receive exactly those columns in that order.
    # Columns present in training but absent from test data are filled with a
    # dummy string; OrdinalEncoder maps any unseen string to unknown_value=-1.
    # -------------------------------------------------------------------------
    if verbose:
        print("[SECTION] Applying OrdinalEncoder")

    _test_obj_cols = set(df_test_processed.select_dtypes(include="object").columns)

    _test_cat_df = pd.DataFrame(index=df_test_processed.index)
    _missing_cat = []
    for col in train_cat_cols:
        if col in _test_obj_cols:
            _test_cat_df[col] = df_test_processed[col]
        else:
            # Column was in training but not in test; encoder maps "__MISSING__" to -1
            _test_cat_df[col] = "__MISSING__"
            _missing_cat.append(col)

    if verbose and _missing_cat:
        print(f"  [WARNING] {len(_missing_cat)} categorical cols absent from test → encoded as -1: {_missing_cat[:5]}")

    _cat_encoded = pd.DataFrame(
        ordinal_enc.transform(_test_cat_df),
        columns=train_cat_cols,
        index=df_test_processed.index,
    )
    if verbose:
        print(f"  Encoded {len(train_cat_cols)} categorical columns")

    # -------------------------------------------------------------------------
    # Assemble feature matrix
    # Numeric columns (including true-rate floats from above) + OrdinalEncoded
    # categoricals, then reindex to the exact feature list from training.
    # fill_value=0 handles any column that preprocessing produced in training
    # but is entirely absent from this test batch.
    # -------------------------------------------------------------------------
    _numeric_test = df_test_processed.select_dtypes(exclude="object")
    df_test_features = pd.concat([_numeric_test, _cat_encoded], axis=1)

    X_test = (
        df_test_features
        .reindex(columns=feature_names, fill_value=0)
        .fillna(0)
    )

    if verbose:
        _missing_feats = set(feature_names) - set(df_test_features.columns)
        if _missing_feats:
            print(
                f"  [WARNING] {len(_missing_feats)} features missing from test — "
                f"filled with 0: {sorted(_missing_feats)[:5]}..."
            )
        print(f"  Final feature matrix: {X_test.shape}")

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    if verbose:
        unique, counts = np.unique(y_pred, return_counts=True)
        print("[INFO] Prediction distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count:,} ({100.0 * count / len(y_pred):.1f}%)")

    # -------------------------------------------------------------------------
    # Write submission CSV
    # -------------------------------------------------------------------------
    out_dir.mkdir(exist_ok=True)
    submission_df  = pd.DataFrame({"id": df_test["id"].values, "label": y_pred})
    submission_path = out_dir / f"submission-{model_name}.csv"
    submission_df.to_csv(submission_path, index=False)

    if verbose:
        print(f"[INFO] Submission saved: {submission_path}  ({len(submission_df):,} rows)")

    return submission_path


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle submission CSV from a saved tree model artifact.",
    )
    parser.add_argument(
        "model_name",
        help="Model artifact folder name under models/ (e.g. rfc)",
    )
    parser.add_argument("--project-root",    dest="project_root",    default=None)
    parser.add_argument("--test-data-path",  dest="test_data_path",  default=None)
    parser.add_argument("--submissions-dir", dest="submissions_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_tree_submission_csv(
        model_name=args.model_name,
        project_root=Path(args.project_root).resolve() if args.project_root else None,
        test_data_path=Path(args.test_data_path).resolve() if args.test_data_path else None,
        submissions_dir=Path(args.submissions_dir).resolve() if args.submissions_dir else None,
        verbose=True,
    )


if __name__ == "__main__":
    main()
