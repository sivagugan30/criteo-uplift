"""
Utility script to load saved model artifacts and visualization data.
Use this in Streamlit dashboard or other analysis scripts.
"""

import joblib
import pandas as pd
from pathlib import Path


def get_project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


def load_model(model_name):
    """
    Load a saved model by name.
    
    Available models:
    - 't_learner' -> T-Learner (XGBoost)
    - 's_learner' -> S-Learner (XGBoost)
    - 'x_learner' -> X-Learner (XGBoost)
    - 'uplift_rf' -> Uplift Random Forest (CausalML)
    - 'causal_forest' -> Causal Forest (EconML)
    """
    models_dir = get_project_root() / 'models'
    
    model_files = {
        't_learner': 't_learner_model.joblib',
        's_learner': 's_learner_model.joblib',
        'x_learner': 'x_learner_model.joblib',
        'uplift_rf': 'uplift_random_forest.joblib',
        'causal_forest': 'causal_forest_econml.joblib'
    }
    
    if model_name not in model_files:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_files.keys())}")
    
    model_path = models_dir / model_files[model_name]
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run the corresponding notebook first.")
    
    return joblib.load(model_path)


def load_all_models():
    """Load all available models."""
    models = {}
    for name in ['t_learner', 's_learner', 'x_learner', 'uplift_rf', 'causal_forest']:
        try:
            models[name] = load_model(name)
            print(f"✅ Loaded: {name}")
        except FileNotFoundError:
            print(f"⚠️ Not found: {name}")
    return models


def load_visualization_data(filename):
    """
    Load visualization data CSV.
    
    Available files:
    - 'nb02_predictions.csv' -> Test predictions from notebook 02
    - 'nb02_model_metrics.csv' -> Model comparison metrics
    - 'nb02_qini_curves.csv' -> Qini curve data
    - 'nb03_predictions.csv' -> Causal Forest predictions
    - 'nb03_permutation_importance.csv' -> Feature importance
    - 'nb03_feature_uplift_correlation.csv' -> Feature correlations
    - 'nb04_decile_analysis.csv' -> Decile analysis data
    - 'nb04_calibration_data.csv' -> Calibration data
    - 'nb04_bootstrap_results.csv' -> Bootstrap test results
    """
    data_dir = get_project_root() / 'visualizations' / 'data'
    file_path = data_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)


def get_image_path(filename):
    """
    Get path to a saved visualization image.
    
    Available images:
    - 'nb02_qini_curves.png'
    - 'nb02_uplift_distributions.png'
    - 'nb02_model_comparison.png'
    - 'nb03_permutation_importance.png'
    - 'nb03_feature_correlation.png'
    - 'nb03_qini_curves.png'
    - 'nb04_decile_lift.png'
    - 'nb04_calibration.png'
    - 'nb04_bootstrap_distributions.png'
    """
    images_dir = get_project_root() / 'visualizations' / 'images'
    return images_dir / filename


def list_available_data():
    """List all available data files."""
    data_dir = get_project_root() / 'visualizations' / 'data'
    if data_dir.exists():
        files = list(data_dir.glob('*.csv'))
        print("Available data files:")
        for f in sorted(files):
            print(f"  - {f.name}")
        return [f.name for f in files]
    return []


def list_available_images():
    """List all available image files."""
    images_dir = get_project_root() / 'visualizations' / 'images'
    if images_dir.exists():
        files = list(images_dir.glob('*.png'))
        print("Available image files:")
        for f in sorted(files):
            print(f"  - {f.name}")
        return [f.name for f in files]
    return []


if __name__ == "__main__":
    print("=" * 60)
    print("UPLIFT MODEL ARTIFACTS LOADER")
    print("=" * 60)
    
    print("\n📁 Available Models:")
    print("-" * 40)
    models_dir = get_project_root() / 'models'
    if models_dir.exists():
        for f in sorted(models_dir.glob('*.joblib')):
            print(f"  - {f.name}")
    
    print("\n📊 Available Data Files:")
    print("-" * 40)
    list_available_data()
    
    print("\n🖼️ Available Images:")
    print("-" * 40)
    list_available_images()
    
    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("=" * 60)
    print("""
from scripts.load_artifacts import load_model, load_visualization_data

# Load a model
t_learner = load_model('t_learner')

# Load visualization data
qini_data = load_visualization_data('nb02_qini_curves.csv')

# Get image path for Streamlit
from scripts.load_artifacts import get_image_path
img_path = get_image_path('nb02_qini_curves.png')
""")

