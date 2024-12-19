"""
Configuration management for NNE Strategy
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        """Initialize configuration"""
        self.config_dir = Path(__file__).parent
        self.project_root = self.config_dir.parent.parent
        
        # Default configuration
        self._config = {
            'market': {
                'hours': {
                    'start': '09:30',
                    'end': '16:00'
                },
                'data_interval': '1m'
            },
            'data': {
                'directories': {
                    'raw': 'nne_strategy/data/raw',              # Update path
                    'processed': 'data/processed',   # Processed/cleaned data
                    'analysis': 'nne_strategy/data/analysis'      # Update path
                },
                'file_prefix': 'NNE_data_',
                'file_extension': '.csv',
                'required_columns': ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            },
            'trading': {
                'symbol': 'NNE'
            }
        }
        
        # Load custom config if exists
        self._load_custom_config()
        
    def get(self, *keys: str) -> Any:
        """Get configuration value
        
        Args:
            *keys: Configuration key path
            
        Returns:
            Configuration value
        """
        value = self._config
        for key in keys:
            value = value.get(key)
            if value is None:
                return None
        return value
        
    def get_path(self, *keys: str) -> Path:
        """Get configuration path
        
        Args:
            *keys: Configuration key path
            
        Returns:
            Path object
        """
        value = self.get(*keys)
        if value is None:
            return None
        return Path(value)
        
    def _load_custom_config(self):
        """Load custom configuration from file"""
        config_file = self.config_dir / 'config.json'
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    custom_config = json.load(f)
                self._update_config(custom_config)
                logger.info("Loaded custom configuration")
            except Exception as e:
                logger.error(f"Error loading custom config: {str(e)}")
                
    def _update_config(self, custom_config: Dict):
        """Update configuration with custom values"""
        def update_dict(base: Dict, update: Dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict):
                    update_dict(base[key], value)
                else:
                    base[key] = value
                    
        update_dict(self._config, custom_config)

# Create singleton instance
config = Config() 