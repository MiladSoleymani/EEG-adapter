"""Configuration management for ECoG-ATCNet-LoRA project."""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """Configuration class for managing experiment settings."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Parameters
        ----------
        config_path : str, optional
            Path to custom config file. If None, uses default_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / 'default_config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._validate_config()

    def _validate_config(self):
        """Validate configuration values."""
        # Validate frequency band
        if self.config['data']['freq_band'] is not None:
            assert len(self.config['data']['freq_band']) == 2, "freq_band must have 2 values"
            assert self.config['data']['freq_band'][0] < self.config['data']['freq_band'][1], \
                "freq_band[0] must be less than freq_band[1]"

        # Validate time range
        assert len(self.config['data']['time_range']) == 2, "time_range must have 2 values"
        assert self.config['data']['time_range'][0] < self.config['data']['time_range'][1], \
            "time_range[0] must be less than time_range[1]"

        # Validate LoRA rank
        if self.config['lora']['enabled']:
            assert self.config['lora']['rank'] > 0, "LoRA rank must be positive"
            assert self.config['lora']['alpha'] > 0, "LoRA alpha must be positive"

        # Create directories if they don't exist
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['results_dir'], exist_ok=True)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Parameters
        ----------
        key_path : str
            Dot-separated path to config value (e.g., 'data.freq_band')
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Parameters
        ----------
        key_path : str
            Dot-separated path to config value
        value : Any
            Value to set
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def update_from_dict(self, updates: Dict[str, Any]):
        """
        Update configuration from a dictionary.

        Parameters
        ----------
        updates : dict
            Dictionary with configuration updates (supports nested dicts)
        """
        def recursive_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    recursive_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        recursive_update(self.config, updates)
        self._validate_config()

    def save(self, save_path: str):
        """
        Save configuration to YAML file.

        Parameters
        ----------
        save_path : str
            Path to save configuration
        """
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)

    # Convenience properties for common config values
    @property
    def data_config(self) -> Dict:
        """Get data configuration."""
        return self.config['data']

    @property
    def model_config(self) -> Dict:
        """Get model configuration."""
        return self.config['model']

    @property
    def lora_config(self) -> Dict:
        """Get LoRA configuration."""
        return self.config['lora']

    @property
    def training_config(self) -> Dict:
        """Get training configuration."""
        return self.config['training']

    @property
    def base_training_config(self) -> Dict:
        """Get base model training configuration."""
        return self.config['training']['base']

    @property
    def lora_training_config(self) -> Dict:
        """Get LoRA training configuration."""
        return self.config['training']['lora']

    @property
    def hardware_config(self) -> Dict:
        """Get hardware configuration."""
        return self.config['hardware']

    @property
    def logging_config(self) -> Dict:
        """Get logging configuration."""
        return self.config['logging']

    @property
    def seed(self) -> int:
        """Get random seed."""
        return self.config['seed']


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file

    Returns
    -------
    Config
        Configuration object
    """
    return Config(config_path)
