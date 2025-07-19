from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict
import yaml
import os


class Thresholds(BaseModel):
    confidence: float = Field(..., ge=0, le=1)
    risk: float = Field(..., ge=0, le=1)


class Limits(BaseModel):
    max_actions: int = Field(..., ge=1)
    max_depth: int = Field(..., ge=1)


class Preferences(BaseModel):
    allow_external_commands: bool = False
    log_level: str = "INFO"


class Config(BaseModel):
    thresholds: Thresholds
    limits: Limits
    preferences: Preferences

    @classmethod
    def load(cls, path: str = None) -> "Config":
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Usage:
# config = Config.load()
