"""
Model persistence module for saving trained models and preprocessing configurations.

This module provides utilities to save trained models, preprocessing options, and
feature names in a standardized way that's reproducible for submission.
"""

from pathlib import Path
from typing import Any, List, Dict
import joblib


def save_model(
    model_pipeline: Any,
    preprocessing_options: Any,
    feature_names: List[str],
    project_root: Path,
    model_name: str,
) -> Dict[str, Path]:
    """
    Save a trained model pipeline, preprocessing options, and feature names.

    This function saves all artifacts needed for reproducible inference, organized
    by model type within the models/ directory.

    Parameters
    ----------
    model_pipeline : sklearn Pipeline
        The trained model pipeline (e.g., sklearn Pipeline with scaler and classifier)
    preprocessing_options : Any
        The preprocessing configuration object used during training
    feature_names : List[str]
        List of feature names in the order used by the model
    project_root : Path
        Root directory of the project
    model_name : str
        Name/type of the model (e.g., 'geom-svm', 'bagging', 'trees')
        Subdirectory will be created: models/{model_name}/

    Returns
    -------
    Dict[str, Path]
        Dictionary with keys:
        - 'model': Path to saved model pipeline
        - 'options': Path to saved preprocessing options
        - 'features': Path to saved feature names
        - 'model_dir': Directory where all artifacts were saved

    Examples
    --------
    >>> saved_paths = save_model(
    ...     model_pipeline=svm_pipeline,
    ...     preprocessing_options=options,
    ...     feature_names=X_trainval.columns.tolist(),
    ...     project_root=Path(__file__).parents[2],
    ...     model_name='geom-svm'
    ... )
    >>> print(f"Model saved to: {saved_paths['model']}")
    """

    print("[SECTION] Saving model and preprocessing options")

    # Create model-specific subdirectory
    models_dir = project_root / "models"
    model_dir = models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    model_path = model_dir / f"{model_name}-model.joblib"
    options_path = model_dir / f"{model_name}-options.joblib"
    features_path = model_dir / f"{model_name}-feature-names.joblib"

    # # Save artifacts
    # print(f"Saving {model_name} model artifacts to: {model_dir}")
    joblib.dump(model_pipeline, model_path)
    # print(f"  ✓ Model pipeline saved to: {model_path}")

    joblib.dump(preprocessing_options, options_path)
    # print(f"  ✓ Preprocessing options saved to: {options_path}")

    joblib.dump(feature_names, features_path)
    # print(f"  ✓ Feature names ({len(feature_names)} features) saved to: {features_path}")

    # Return paths for reference
    return {
        "model": model_path,
        "options": options_path,
        "features": features_path,
        "model_dir": model_dir,
    }


def load_model(
    project_root: Path,
    model_name: str,
) -> Dict[str, Any]:
    """
    Load a previously saved model, preprocessing options, and feature names.

    Parameters
    ----------
    project_root : Path
        Root directory of the project
    model_name : str
        Name/type of the model (must match save_model call)

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'model': Loaded model pipeline
        - 'options': Loaded preprocessing options
        - 'features': Loaded feature names

    Raises
    ------
    FileNotFoundError
        If any of the required files are not found

    Examples
    --------
    >>> artifacts = load_model(
    ...     project_root=Path(__file__).parents[2],
    ...     model_name='geom-svm'
    ... )
    >>> model = artifacts['model']
    >>> options = artifacts['options']
    >>> features = artifacts['features']
    """

    model_dir = project_root / "models" / model_name

    model_path = model_dir / f"{model_name}-model.joblib"
    options_path = model_dir / f"{model_name}-options.joblib"
    features_path = model_dir / f"{model_name}-feature-names.joblib"

    # Check all files exist
    missing_files = []
    if not model_path.exists():
        missing_files.append(str(model_path))
    if not options_path.exists():
        missing_files.append(str(options_path))
    if not features_path.exists():
        missing_files.append(str(features_path))

    if missing_files:
        raise FileNotFoundError(
            f"Missing model artifacts for '{model_name}':\n"
            + "\n".join(f"  - {f}" for f in missing_files)
            + f"\nPlease run the training script first to generate these files."
        )

    # print("[SECTION] Loading model artifacts")
    # print(f"Loading {model_name} model artifacts from: {model_dir}")

    model_pipeline = joblib.load(model_path)
    # print(f"  ✓ Model pipeline loaded")

    preprocessing_options = joblib.load(options_path)
    # print(f"  ✓ Preprocessing options loaded")

    feature_names = joblib.load(features_path)
    # print(f"  ✓ Feature names loaded ({len(feature_names)} features)")

    return {
        "model": model_pipeline,
        "options": preprocessing_options,
        "features": feature_names,
    }


def list_saved_models(project_root: Path) -> Dict[str, List[Path]]:
    """
    List all saved models and their artifacts.

    Parameters
    ----------
    project_root : Path
        Root directory of the project

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary where keys are model names and values are lists of artifact files

    Examples
    --------
    >>> models = list_saved_models(Path(__file__).parents[2])
    >>> for model_name, artifacts in models.items():
    ...     print(f"{model_name}: {len(artifacts)} artifacts")
    """

    models_dir = project_root / "models"
    if not models_dir.exists():
        return {}

    saved_models = {}
    for model_subdir in models_dir.iterdir():
        if model_subdir.is_dir():
            artifacts = list(model_subdir.glob("*.joblib"))
            if artifacts:
                saved_models[model_subdir.name] = artifacts

    return saved_models
