"""
Configuration for the Uplift Modeling Dashboard
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "visualizations" / "data"
IMAGES_DIR = PROJECT_ROOT / "visualizations" / "images"
MODELS_DIR = PROJECT_ROOT / "models"

# Color scheme - professional, muted palette
COLORS = {
    "persuadables": "#2E7D32",      # Dark green
    "sure_things": "#1565C0",        # Dark blue
    "lost_causes": "#616161",        # Gray
    "sleeping_dogs": "#C62828",      # Dark red
    "primary": "#1A1A2E",            # Dark navy
    "secondary": "#16213E",          # Slightly lighter navy
    "accent": "#0F3460",             # Blue accent
    "highlight": "#E94560",          # Coral highlight
    "text": "#EAEAEA",               # Light text
    "background": "#0F0F1A",         # Very dark background
}

# Segment display names and colors
SEGMENTS = {
    "Persuadables": {
        "color": COLORS["persuadables"],
        "action": "Target",
        "description": "Users who convert because of treatment"
    },
    "Sure Things": {
        "color": COLORS["sure_things"],
        "action": "Save budget",
        "description": "Users who convert regardless of treatment"
    },
    "Lost Causes": {
        "color": COLORS["lost_causes"],
        "action": "Skip",
        "description": "Users who do not convert regardless of treatment"
    },
    "Sleeping Dogs": {
        "color": COLORS["sleeping_dogs"],
        "action": "Avoid",
        "description": "Users where treatment hurts conversion"
    }
}

# Model names
MODELS = ["T-Learner", "S-Learner", "X-Learner"]

# Page configuration
PAGE_CONFIG = {
    "page_title": "Uplift Modeling Dashboard",
    "page_icon": None,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Feature names (anonymized in original dataset)
FEATURE_NAMES = [f"f{i}" for i in range(12)]


