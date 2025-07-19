"""
Claude Code Context Plugin

A practical plugin that integrates directly with Claude Code for intelligent
context management during AI development workflows.
"""

from .core.plugin_manager import PluginManager
from .core.monitor import ContextMonitor
from .core.orchestrator import PluginOrchestrator
from .cli.commands import main as cli_main

__version__ = "1.0.0"
__author__ = "Claude Context Management Team"

# Plugin manager instance
plugin_manager = PluginManager()

# Main exports
__all__ = [
    "PluginManager",
    "ContextMonitor",
    "PluginOrchestrator",
    "cli_main",
    "plugin_manager",
]
