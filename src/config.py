# src/config.py
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Default configurations
DEFAULT_CONFIG = {
    # Data paths
    "data": {
        "raw_dir": os.path.join(BASE_DIR, "data", "raw"),
        "processed_dir": os.path.join(BASE_DIR, "data", "processed"),
        "models_dir": os.path.join(BASE_DIR, "data", "models"),
        "train_file": "train.csv",
        "val_file": "val.csv",
        "test_file": "test.csv",
        "amazon_products_file": "Amazon-Products.csv"
    },

    # Preprocessing settings
    "preprocessing": {
        "max_length": 128,
        "remove_stopwords": True,
        "lowercase": True,
        "remove_special_chars": True,
        "lemmatize": True,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42
    },

    # Model settings
    "model": {
        "model_type": "bert",  # Options: bert, roberta, distilbert
        "model_name": "bert-base-uncased",
        "dropout_rate": 0.1,
        "num_labels": None,  # Will be determined during preprocessing
    },

    # Training settings
    "training": {
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 3,
        "checkpoint_dir": os.path.join(BASE_DIR, "data", "models", "checkpoints")
    },

    # API settings
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "model_path": os.path.join(BASE_DIR, "data", "models", "best_model.pt"),
        "log_level": "INFO"
    },

    # Logging settings
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_dir": os.path.join(BASE_DIR, "logs")
    }
}


class Config:
    """
    Configuration class for managing application settings.

    Loads settings from a YAML file and/or environment variables,
    with fallback to default values.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file (optional)
        """
        self.config = DEFAULT_CONFIG.copy()

        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            self._load_from_yaml(config_path)

        # Override with environment variables
        self._load_from_env()

    def _load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                self._update_config(self.config, yaml_config)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")

    def _load_from_env(self) -> None:
        """
        Override configuration with environment variables.

        Environment variables should be in the format:
        SHOPFULLY_SECTION_KEY=value

        For example:
        SHOPFULLY_MODEL_DROPOUT_RATE=0.2
        """
        prefix = "SHOPFULLY_"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Remove prefix and split by underscore
                key_parts = env_key[len(prefix):].lower().split("_")

                # Navigate through config dictionary
                current = self.config
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the value, converting to appropriate type
                current[key_parts[-1]] = self._convert_type(env_value)

    def _update_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update configuration dictionary.

        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_config(target[key], value)
            else:
                target[key] = value

    def _convert_type(self, value: str) -> Any:
        """
        Convert string value to appropriate Python type.

        Args:
            value: String value to convert

        Returns:
            Converted value
        """
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass

        # Handle boolean values
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Return as string
        return value

    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Get configuration value.

        Args:
            section: Configuration section
            key: Configuration key (optional)

        Returns:
            Configuration value or section dictionary
        """
        if section not in self.config:
            return None

        if key is None:
            return self.config[section]

        return self.config[section].get(key)

    def __getitem__(self, key: str) -> Any:
        """
        Get configuration section using dictionary syntax.

        Args:
            key: Section name

        Returns:
            Configuration section
        """
        return self.config.get(key, {})


# Create global config instance
config = Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config instance
    """
    global config
    config = Config(config_path)
    return config
