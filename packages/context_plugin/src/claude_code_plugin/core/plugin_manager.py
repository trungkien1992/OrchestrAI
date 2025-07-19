"""
Plugin Manager for Claude Code Context Plugin

Manages the overall plugin lifecycle, integration with Claude Code,
and coordination between different components.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from .monitor import ContextMonitor
from .orchestrator import PluginOrchestrator
from ..config import load_config

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Main plugin manager that coordinates all plugin functionality
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_config()
        self.storage_path = Path.home() / ".claude" / "claude-code-plugin" / "data"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.monitor = ContextMonitor(self.config)
        self.orchestrator = PluginOrchestrator(self.config)

        # Plugin state
        self.is_running = False
        self.current_session = None
        self.session_history = []

        logger.info("Plugin Manager initialized")

    async def start_plugin(self):
        """Start the plugin and all monitoring"""
        try:
            logger.info("Starting Claude Code Context Plugin...")

            # Start monitoring
            await self.monitor.start_monitoring()

            # Initialize session
            self.current_session = {
                "id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "start_time": datetime.now(),
                "status": "active",
            }

            self.is_running = True
            logger.info("Plugin started successfully")

        except Exception as e:
            logger.error(f"Failed to start plugin: {e}")
            raise

    async def stop_plugin(self):
        """Stop the plugin and save session data"""
        try:
            logger.info("Stopping Claude Code Context Plugin...")

            # Stop monitoring
            await self.monitor.stop_monitoring()

            # Save current session
            if self.current_session:
                await self._save_session(self.current_session)

            self.is_running = False
            logger.info("Plugin stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping plugin: {e}")
            raise

    async def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get recent session history"""
        try:
            history_file = self.storage_path / "session_history.json"

            if not history_file.exists():
                return []

            with open(history_file, "r") as f:
                all_sessions = json.load(f)

            # Return most recent sessions
            return all_sessions[-limit:] if all_sessions else []

        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []

    async def handle_claude_code_event(self, event_type: str, event_data: Dict):
        """Handle events from Claude Code"""
        try:
            logger.debug(f"Handling Claude Code event: {event_type}")

            if event_type == "file_modified":
                await self._handle_file_modified(event_data)
            elif event_type == "command_executed":
                await self._handle_command_executed(event_data)
            elif event_type == "session_started":
                await self._handle_session_started(event_data)
            elif event_type == "session_ended":
                await self._handle_session_ended(event_data)
            elif event_type == "token_usage_updated":
                await self._handle_token_usage_updated(event_data)
            else:
                logger.warning(f"Unknown event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling Claude Code event: {e}")

    async def get_plugin_status(self) -> Dict[str, Any]:
        """Get current plugin status"""
        try:
            monitor_status = await self.monitor.get_status()

            return {
                "plugin_running": self.is_running,
                "current_session": self.current_session,
                "monitor_status": monitor_status,
                "last_update": datetime.now().isoformat(),
                "config_loaded": bool(self.config),
            }

        except Exception as e:
            logger.error(f"Error getting plugin status: {e}")
            return {"error": str(e)}

    async def _handle_file_modified(self, event_data: Dict):
        """Handle file modification events"""
        file_path = event_data.get("file_path")
        modification_type = event_data.get("type", "unknown")

        logger.debug(f"File modified: {file_path} ({modification_type})")

        # Update session data
        if self.current_session:
            if "files_modified" not in self.current_session:
                self.current_session["files_modified"] = []

            self.current_session["files_modified"].append(
                {
                    "path": file_path,
                    "type": modification_type,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    async def _handle_command_executed(self, event_data: Dict):
        """Handle command execution events"""
        command = event_data.get("command")
        success = event_data.get("success", True)

        logger.debug(f"Command executed: {command} (success: {success})")

        # Update session data
        if self.current_session:
            if "commands_executed" not in self.current_session:
                self.current_session["commands_executed"] = []

            self.current_session["commands_executed"].append(
                {
                    "command": command,
                    "success": success,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Check if we should suggest optimization
        if command in ["/compact", "/digest"]:
            await self._analyze_command_timing(command, success)

    async def _handle_session_started(self, event_data: Dict):
        """Handle session start events"""
        session_type = event_data.get("session_type", "unknown")

        logger.info(f"New session started: {session_type}")

        # Create new session
        self.current_session = {
            "id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now(),
            "session_type": session_type,
            "status": "active",
        }

    async def _handle_session_ended(self, event_data: Dict):
        """Handle session end events"""
        logger.info("Session ended")

        if self.current_session:
            self.current_session["end_time"] = datetime.now()
            self.current_session["status"] = "ended"

            # Save session
            await self._save_session(self.current_session)

            # Reset current session
            self.current_session = None

    async def _handle_token_usage_updated(self, event_data: Dict):
        """Handle token usage updates"""
        token_count = event_data.get("token_count", 0)
        token_limit = event_data.get("token_limit", 200000)

        logger.debug(f"Token usage updated: {token_count}/{token_limit}")

        # Update monitor
        await self.monitor.update_token_usage(token_count, token_limit)

        # Check if we should suggest optimization
        usage_ratio = token_count / token_limit
        if usage_ratio > self.config.get("auto_compact_threshold", 0.85):
            await self._suggest_optimization("token_pressure")

    async def _analyze_command_timing(self, command: str, success: bool):
        """Analyze command timing for learning"""
        logger.debug(f"Analyzing command timing: {command} (success: {success})")

        # In a real implementation, this would learn from timing patterns
        # and suggest optimal times for commands
        pass

    async def _suggest_optimization(self, trigger: str):
        """Suggest optimization based on trigger"""
        logger.info(f"Suggesting optimization due to: {trigger}")

        try:
            # Use orchestrator to get recommendation
            session_data = {
                "trigger": trigger,
                "current_session": self.current_session,
                "timestamp": datetime.now().isoformat(),
            }

            recommendation = await self.orchestrator.make_decision(session_data)

            # In a real implementation, this would display the recommendation
            # to the user through Claude Code's interface
            logger.info(f"Recommendation: {recommendation}")

        except Exception as e:
            logger.error(f"Error suggesting optimization: {e}")

    async def _save_session(self, session: Dict):
        """Save session to history"""
        try:
            history_file = self.storage_path / "session_history.json"

            # Load existing history
            if history_file.exists():
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                history = []

            # Add current session
            session_copy = session.copy()

            # Convert datetime objects to strings
            for key in ["start_time", "end_time"]:
                if key in session_copy and isinstance(session_copy[key], datetime):
                    session_copy[key] = session_copy[key].isoformat()

            history.append(session_copy)

            # Keep only last 100 sessions
            if len(history) > 100:
                history = history[-100:]

            # Save updated history
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

            logger.debug(f"Session saved: {session['id']}")

        except Exception as e:
            logger.error(f"Error saving session: {e}")
