# Kaggle Submission Guide

## Overview
This guide explains how to prepare and submit your model predictions to the Kaggle competition.

## Process

### Step 1: Train the Model
Before submitting, you must have trained the model:

```bash
cd src/training
python geom-svm.py
```

This script will:
- Load and preprocess the training data
- Train the LinearSVC model using stratified k-fold cross-validation
- Evaluate on a holdout test set
- **Save the model and preprocessing options to `models/` directory**

### Step 2: Generate Predictions
After training, run the submission script:

```bash
cd src/submit  
python kaggle.py
```

This script will:
- Load the trained model from `models/geom-svm-model.joblib`
- Load preprocessing options from `models/geom-svm-options.joblib`
- Load test data from `data/test_nolabel.csv`
- Apply the same preprocessing as training
- Generate predictions using the trained model
- Create submission CSV file

### Step 3: Upload to Kaggle

#### Option A: Web Interface (Recommended)
1. Go to: https://www.kaggle.com/c/vehicle-emissions-classification/submit
2. Click "Submit Predictions"
3. Upload the generated CSV file: `submissions/submission-geom-svm.csv`
4. Add a description (e.g., "Geom-SVM model")
5. Click "Submit"

#### Option B: Kaggle CLI (if installed)
```bash
cd submissions
kaggle competitions submit -c vehicle-emissions-classification \
  -f submission-geom-svm.csv \
  -m "Geom-SVM model submission"
```

To install Kaggle CLI:
```bash
pip install kaggle
```

## Submission Format

The submission file must be a CSV with exactly two columns:
- `id`: The loan ID from the test dataset
- `Accept`: Binary prediction (0 = Reject, 1 = Accept)

Example:
```csv
id,Accept
bae908d5352,0
9260b4c0f25,1
2c4e5bbee21,1
```

## Important Notes

### Submission Limits
- **Before deadline**: Up to 100 submissions per day
- **Mark best submissions**: You must mark your 2 best submissions before the deadline
  - These will be used for final leaderboard ranking

### Data Handling
- **DO NOT** modify test data rows - keep all rows as-is
- **MUST** apply the same preprocessing as training data
- **MUST** use the same feature order as training

### Features Used
The model uses the following preprocessing options:
- noemp_option: C
- newexist_option: B
- createjob_option: C
- retainedjob_option: B
- approvaldate_option: A
- approvalfy_option: A
- franchise_option: binary
- urbanrural_option: onehot
- revlinecr_option: C
- lowdoc_option: C
- disbursementgross_option: C
- local_state: IL

## Troubleshooting

### Model Not Found Error
If you get "Model not found at: models/geom-svm-model.joblib":
- Run the training script first: `python src/training/geom-svm.py`
- Ensure the training script completed successfully
- Check that models/ directory was created with the .joblib files

### Preprocessing Errors
If you get preprocessing errors:
- Ensure test data (`data/test_nolabel.csv`) exists
- Check that all preprocessing modules are available in `src/preprocessing/`
- Verify the same Python environment is used for training and submission

### Feature Mismatch
If features don't align:
- This is automatically handled by the submission script
- The script reorders and validates features
- Check output logs for any warnings

## Output Files

After running the submission script:
- **Submission CSV**: `submissions/submission-geom-svm.csv`
- **Model**: `models/geom-svm-model.joblib` (pipeline with scaler and SVM)
- **Options**: `models/geom-svm-options.joblib` (preprocessing configuration)
- **Features**: `models/geom-svm-feature-names.joblib` (feature names and order)

## Next Steps

1. ✅ Train the model: `python geom-svm.py`
2. ✅ Generate predictions: `python kaggle.py`
3. ✅ Submit to Kaggle: Upload CSV or use Kaggle CLI
4. ✅ Monitor leaderboard: Check your score
5. ✅ Mark best submissions: Select your 2 best before deadline
6. ✅ Submit final deliverables: Notebooks and documentation to Moodle after deadline



kaggle CLI

kaggle competitions list --group entered
ref                                                                            deadline             category   reward  teamCount  userHasEntered
-----------------------------------------------------------------------------  -------------------  ---------  ------  ---------  --------------
https://www.kaggle.com/competitions/cdaw-loan-approval-prediction-in-illinois  2026-04-08 21:50:00  Community   Kudos          4            True




### kaggle competitions submissions

Shows your past submissions for a competition.

Usage:

kaggle competitions submissions <COMPETITION> [options]

Arguments:

    <COMPETITION>: Competition URL suffix (e.g., house-prices-advanced-regression-techniques).

Options:

    -v, --csv: Print results in CSV format.
    -q, --quiet: Suppress verbose output.

Example:

Show submissions for "house-prices-advanced-regression-techniques" in CSV format, quietly:

kaggle competitions submissions house-prices-advanced-regression-techniques -v -q

COMMAND: 

```
kaggle competitions submit -c cdaw-loan-approval-prediction-in-illinois -f submissions/submission-geom-svm.csv -m "Geom-SVM model submission"
```