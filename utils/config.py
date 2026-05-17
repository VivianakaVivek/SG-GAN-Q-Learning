"""
utils/config.py
Loads and validates the YAML config file.
"""

import yaml
import os


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(config: dict):
    """Create all output directories specified in config."""
    for key, path in config["output"].items():
        if key.endswith("_dir"):
            os.makedirs(path, exist_ok=True)
