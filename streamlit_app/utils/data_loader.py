"""
Data loading utilities for the Streamlit dashboard.
Centralizes all data access to avoid redundant loading.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, IMAGES_DIR


@st.cache_data
def load_eda_summary():
    """Load EDA summary statistics."""
    return pd.read_csv(DATA_DIR / "nb01_eda_summary.csv")


@st.cache_data
def load_feature_statistics():
    """Load feature-level statistics."""
    return pd.read_csv(DATA_DIR / "nb01_feature_statistics.csv")


@st.cache_data
def load_outcome_distribution():
    """Load outcome distribution data."""
    return pd.read_csv(DATA_DIR / "nb01_outcome_distribution.csv")


@st.cache_data
def load_treatment_distribution():
    """Load treatment/control distribution."""
    return pd.read_csv(DATA_DIR / "nb01_treatment_distribution.csv")


@st.cache_data
def load_uplift_by_quartile():
    """Load uplift analysis by feature quartiles."""
    return pd.read_csv(DATA_DIR / "nb01_uplift_by_quartile.csv")


@st.cache_data
def load_model_metrics():
    """Load model performance metrics (Qini, AUUC)."""
    return pd.read_csv(DATA_DIR / "nb02_model_metrics.csv")


@st.cache_data
def load_predictions():
    """Load model predictions for test set."""
    return pd.read_csv(DATA_DIR / "nb02_predictions.csv")


@st.cache_data
def load_qini_curves():
    """Load Qini curve data for all models."""
    return pd.read_csv(DATA_DIR / "nb02_qini_curves.csv")


@st.cache_data
def load_causal_forest_metrics():
    """Load causal forest model metrics."""
    return pd.read_csv(DATA_DIR / "nb03_model_metrics.csv")


@st.cache_data
def load_permutation_importance():
    """Load feature importance from permutation analysis."""
    return pd.read_csv(DATA_DIR / "nb03_permutation_importance.csv")


@st.cache_data
def load_feature_uplift_correlation():
    """Load feature-uplift correlation data."""
    return pd.read_csv(DATA_DIR / "nb03_feature_uplift_correlation.csv")


@st.cache_data
def load_bootstrap_results():
    """Load bootstrap confidence interval results."""
    return pd.read_csv(DATA_DIR / "nb04_bootstrap_results.csv")


@st.cache_data
def load_calibration_data():
    """Load model calibration analysis data."""
    return pd.read_csv(DATA_DIR / "nb04_calibration_data.csv")


@st.cache_data
def load_decile_analysis():
    """Load decile-level performance analysis."""
    return pd.read_csv(DATA_DIR / "nb04_decile_analysis.csv")


@st.cache_data
def load_customer_segments():
    """Load customer segment summary."""
    return pd.read_csv(DATA_DIR / "nb05_customer_segments.csv")


@st.cache_data
def load_shap_importance():
    """Load SHAP feature importance for uplift."""
    return pd.read_csv(DATA_DIR / "nb05_shap_importance.csv")


@st.cache_data
def load_predictions_with_segments():
    """Load full predictions with customer segments."""
    return pd.read_csv(DATA_DIR / "nb05_predictions_with_segments.csv")


def get_image_path(image_name: str) -> Path:
    """Get full path to a visualization image."""
    return IMAGES_DIR / image_name


def load_image(image_name: str):
    """Load an image file for display."""
    image_path = get_image_path(image_name)
    if image_path.exists():
        return str(image_path)
    return None


