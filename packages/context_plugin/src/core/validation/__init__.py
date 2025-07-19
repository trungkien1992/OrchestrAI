"""
Enhanced EAEPT Validation Logging Module
Comprehensive system for tracking AI work effects and validation during EAEPT workflow execution
"""

from .eaept_validation_logger import (
    EAEPTValidationLogger,
    ValidationEvent,
    ValidationLevel,
    ValidationMetric,
    PhaseValidationSummary,
    init_validation_logger,
    get_validation_logger,
    log_phase_start,
    log_phase_end,
    log_effectiveness,
)

__all__ = [
    "EAEPTValidationLogger",
    "ValidationEvent",
    "ValidationLevel",
    "ValidationMetric",
    "PhaseValidationSummary",
    "init_validation_logger",
    "get_validation_logger",
    "log_phase_start",
    "log_phase_end",
    "log_effectiveness",
]

__version__ = "1.0.0"
__author__ = "Enhanced EAEPT Team"
__description__ = (
    "Validation logging system for tracking AI work effectiveness in EAEPT workflows"
)
