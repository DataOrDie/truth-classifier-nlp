#!/usr/bin/env python
"""
Kaggle Submission Script for Vehicle Emissions Classification Challenge

This script loads a trained model and generates predictions for the test dataset.

Usage:
    python kaggle.py <model_name> [--is-tree-model]
    
Arguments:
    model_name: Name of the trained model (corresponds to models/{model_name}/ directory)
    --is-tree-model: Apply tree-focused engineered features during preprocessing
    
Examples:
    python kaggle.py geom-svm
    python kaggle.py geom-svm-thresholdTuneClassifier
    python kaggle.py bagging-etrees-tree-first --is-tree-model
    
Output:
    - Predictions CSV will be saved to: submissions/submission-{model_name}.csv
    - The model and preprocessing artifacts must exist in: models/{model_name}/
    
Instructions:
    1. Train a model using a training script (e.g., geom-svm.py, geom-svm-thresholdTuneClassifier.py)
    2. Run this script with the model name: python kaggle.py <model_name>
    3. The submission CSV will be created in the submissions/ directory
    4. Upload the CSV to Kaggle competition manually or use kaggle CLI
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Resolve paths from this file so running from any cwd still works
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing.one_step import preprocess_one_step
from submit.save_model import load_model


# =============================================================================
# SECTION: Parse command-line arguments
# =============================================================================
if len(sys.argv) < 2:
    print("Usage: python kaggle.py <model_name> [--is-tree-model]")
    print("Example: python kaggle.py geom-svm-thresholdTuneClassifier")
    sys.exit(1)

model_name = sys.argv[1]
is_tree_model = "--is-tree-model" in sys.argv[2:]
print(f"[INFO] Model name: {model_name}")
print(f"[INFO] is_tree_model: {is_tree_model}")


# =============================================================================
# SECTION: Load test data
# =============================================================================
print("[SECTION] Loading test dataset")
test_path = project_root / "data" / "test_nolabel.csv"
if not test_path.exists():
    raise FileNotFoundError(f"Test data not found at: {test_path}")

df_test = pd.read_csv(test_path)
print(f"Test dataset shape: {df_test.shape}")
print(f"Columns: {df_test.columns.tolist()}")
print(f"Sample IDs: {df_test['id'].head().tolist()}")


# =============================================================================
# SECTION: Load saved model and preprocessing options
# =============================================================================
artifacts = load_model(
    project_root=project_root,
    model_name=model_name,
)

svm_pipeline = artifacts["model"]
options = artifacts["options"]
feature_names = artifacts["features"]

# Backward compatibility for older saved options artifacts.
if not hasattr(options, "accept_option"):
    options.accept_option = "skip"

# Force row-preserving NewExist preprocessing for submissions.
options.newexist_option = "A"

print(f"\nModel type: {type(svm_pipeline)}")
print(f"Number of features: {len(feature_names)}\n")


# =============================================================================
# SECTION: Apply preprocessing to test data
# =============================================================================
print("[SECTION] Preprocessing test data")
print(f"Preprocessing options:")
print(f"  noemp_option: {options.noemp_option}")
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
print(f"Processed test dataset shape: {df_test_processed.shape}")
print(f"Processed features: {df_test_processed.shape[1]}")

# Some preprocessing options (e.g., NewExist option B) can drop rows.
# Restore original test index so submission always has one prediction per input row.
if len(df_test_processed) != len(df_test):
    dropped_count = len(df_test) - len(df_test_processed)
    print(
        f"WARNING: Preprocessing changed row count by {dropped_count}. "
        "Restoring original row index for full-length submission."
    )
    df_test_processed = df_test_processed.reindex(df_test.index)


# =============================================================================
# SECTION: Verify feature alignment
# =============================================================================
print("[SECTION] Verifying feature alignment")
processed_features = df_test_processed.columns.tolist()

if set(processed_features) != set(feature_names):
    missing_in_test = set(feature_names) - set(processed_features)
    extra_in_test = set(processed_features) - set(feature_names)
    
    if missing_in_test:
        print(f"WARNING: Missing features in test data: {missing_in_test}")
    if extra_in_test:
        print(f"WARNING: Extra features in test data: {extra_in_test}")

    # Robust alignment: add missing training columns with 0 and drop extras.
    df_test_processed = df_test_processed.reindex(columns=feature_names, fill_value=0)
    print("Features realigned to match training schema (missing -> 0, extras dropped)")
else:
    # Ensure same order as training
    df_test_processed = df_test_processed[feature_names]
    print("Features match training data perfectly")

# Fill any missing values introduced by row restoration/alignment.
df_test_processed = df_test_processed.fillna(0)

print("[DEBUG] Final df_test_processed columns:")
for idx, col in enumerate(df_test_processed.columns.tolist(), start=1):
    print(f"  {idx:02d}. {col}")


# =============================================================================
# SECTION: Generate predictions
# =============================================================================
print("[SECTION] Generating predictions")
X_test = df_test_processed
print(f"Input shape for model: {X_test.shape}")

# Get predictions
y_pred = svm_pipeline.predict(X_test)
print(f"Predictions generated. Shape: {y_pred.shape}")
print(f"Prediction distribution:")
unique, counts = np.unique(y_pred, return_counts=True)
for label, count in zip(unique, counts):
    pct = 100.0 * count / len(y_pred)
    print(f"  Class {label}: {count:6d} ({pct:6.2f}%)")

# Get decision function scores for reference
y_score = svm_pipeline.decision_function(X_test)
print(f"Decision scores - Mean: {y_score.mean():.4f}, Std: {y_score.std():.4f}")


# =============================================================================
# SECTION: Create submission file
# =============================================================================
print("[SECTION] Creating submission file")
submissions_dir = project_root / "submissions"
submissions_dir.mkdir(exist_ok=True)

# Align IDs with the exact rows that survived preprocessing.
submission_ids = df_test["id"].values
print(f"Submission IDs: {len(submission_ids)} | Predictions: {len(y_pred)}")

if len(submission_ids) != len(y_pred):
    raise ValueError(
        "Submission length mismatch: "
        f"ids={len(submission_ids)}, preds={len(y_pred)}"
    )

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': submission_ids,
    'Accept': y_pred.astype(int)
})

# Save to CSV
submission_path = submissions_dir / f"submission-{model_name}.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(f"Submission shape: {submission_df.shape}")
print(f"\nFirst 10 submissions:")
print(submission_df.head(10).to_string(index=False))


# =============================================================================
# SECTION: Kaggle submission
# =============================================================================
# print("\n" + "=" * 80)
# print("KAGGLE SUBMISSION")
# print("=" * 80)


