"""
Configuration management for Claude Code Context Plugin
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.util

# Default configuration
DEFAULT_CONFIG = {
    "token_limit": 200000,
    "burn_rate_threshold": 150,
    "auto_compact_threshold": 0.85,
    "checkpoint_interval": 15,  # minutes
    "monitoring_enabled": True,
    "smart_suggestions": True,
    "session_persistence": True,
    "learning_enabled": True,
    "dashboard_enabled": True,
    "log_level": "INFO",
}

WORKFLOW_CONFIG = {
    "coding_session_detection": True,
    "debugging_mode_optimization": True,
    "architecture_session_persistence": True,
    "learning_from_patterns": True,
    "proactive_recommendations": True,
}


def get_config_path() -> Path:
    """Get the configuration file path"""
    return Path.home() / ".claude" / "claude-code-plugin" / "config.py"


def load_config() -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    config_path = get_config_path()

    if not config_path.exists():
        # Return default configuration
        return {**DEFAULT_CONFIG, **WORKFLOW_CONFIG}

    try:
        # Load configuration from Python file
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Merge with defaults
        config = DEFAULT_CONFIG.copy()

        # Update with plugin config
        if hasattr(config_module, "PLUGIN_CONFIG"):
            config.update(config_module.PLUGIN_CONFIG)

        # Update with workflow config
        if hasattr(config_module, "WORKFLOW_CONFIG"):
            config.update(config_module.WORKFLOW_CONFIG)

        return config

    except Exception as e:
        print(f"Warning: Error loading configuration from {config_path}: {e}")
        print("Using default configuration")
        return {**DEFAULT_CONFIG, **WORKFLOW_CONFIG}


def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file"""
    try:
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Separate plugin and workflow configs
        plugin_config = {}
        workflow_config = {}

        for key, value in config.items():
            if key in DEFAULT_CONFIG:
                plugin_config[key] = value
            elif key in WORKFLOW_CONFIG:
                workflow_config[key] = value

        # Generate configuration file content
        config_content = f"""# Claude Code Context Plugin Configuration
# This file is automatically generated

PLUGIN_CONFIG = {plugin_config!r}

WORKFLOW_CONFIG = {workflow_config!r}
"""

        with open(config_path, "w") as f:
            f.write(config_content)

        return True

    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific configuration value"""
    config = load_config()
    return config.get(key, default)


def set_config_value(key: str, value: Any) -> bool:
    """Set a specific configuration value"""
    config = load_config()
    config[key] = value
    return save_config(config)


def validate_config(config: Dict[str, Any]) -> tuple:
    """Validate configuration and return (is_valid, errors)"""
    errors = []

    # Validate token limit
    token_limit = config.get("token_limit", 0)
    if not isinstance(token_limit, int) or token_limit <= 0:
        errors.append("token_limit must be a positive integer")

    # Validate burn rate threshold
    burn_rate = config.get("burn_rate_threshold", 0)
    if not isinstance(burn_rate, (int, float)) or burn_rate <= 0:
        errors.append("burn_rate_threshold must be a positive number")

    # Validate auto compact threshold
    auto_compact = config.get("auto_compact_threshold", 0)
    if not isinstance(auto_compact, (int, float)) or not (0 < auto_compact <= 1):
        errors.append("auto_compact_threshold must be between 0 and 1")

    # Validate checkpoint interval
    checkpoint_interval = config.get("checkpoint_interval", 0)
    if not isinstance(checkpoint_interval, int) or checkpoint_interval <= 0:
        errors.append("checkpoint_interval must be a positive integer")

    # Validate boolean settings
    boolean_settings = [
        "monitoring_enabled",
        "smart_suggestions",
        "session_persistence",
        "learning_enabled",
        "dashboard_enabled",
    ]

    for setting in boolean_settings:
        if setting in config and not isinstance(config[setting], bool):
            errors.append(f"{setting} must be a boolean (True/False)")

    # Validate log level
    log_level = config.get("log_level", "INFO")
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_log_levels:
        errors.append(f"log_level must be one of: {', '.join(valid_log_levels)}")

    return len(errors) == 0, errors


def create_default_config() -> bool:
    """Create default configuration file"""
    config_path = get_config_path()

    if config_path.exists():
        return False  # Don't overwrite existing config

    return save_config({**DEFAULT_CONFIG, **WORKFLOW_CONFIG})


def reset_config() -> bool:
    """Reset configuration to defaults"""
    return save_config({**DEFAULT_CONFIG, **WORKFLOW_CONFIG})


def get_config_info() -> Dict[str, Any]:
    """Get information about configuration"""
    config_path = get_config_path()
    config = load_config()

    return {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "config_valid": validate_config(config)[0],
        "config_size": len(config),
        "plugin_settings": len([k for k in config.keys() if k in DEFAULT_CONFIG]),
        "workflow_settings": len([k for k in config.keys() if k in WORKFLOW_CONFIG]),
    }
