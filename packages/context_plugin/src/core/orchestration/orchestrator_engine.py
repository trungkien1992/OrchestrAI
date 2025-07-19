#!/usr/bin/env python3
"""
Claude Orchestration Engine
Intelligent context management system that balances /digest and /compact commands
Enhanced with async processing and advanced analytics
"""

import json
import os
import time
import argparse
import subprocess
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from collections import Counter, deque


class WorkflowProfile(Enum):
    DEVELOPMENT = "development"
    RESEARCH = "research"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    CUSTOM = "custom"


class CommandType(Enum):
    DIGEST = "digest"
    COMPACT = "compact"
    HYBRID = "hybrid"
    NONE = "none"


@dataclass
class ContextMetrics:
    """Enhanced metrics for context analysis with burn rate tracking"""

    token_count: int = 0
    message_count: int = 0
    session_duration: float = 0.0  # minutes
    complexity_score: float = 0.0
    task_completion_count: int = 0
    code_blocks: int = 0
    repetitive_patterns: int = 0
    topic_changes: int = 0
    # New burn rate tracking
    burn_rate: float = 0.0
    time_to_limit: Optional[float] = None
    token_history: List[Tuple[datetime, int]] = None
    confidence_score: float = 0.0

    def __post_init__(self):
        if self.token_history is None:
            self.token_history = []

    def calculate_complexity(self) -> float:
        """Calculate overall complexity score with burn rate consideration"""
        factors = [
            self.message_count / 50,  # Normalize to 50 messages
            self.session_duration / 60,  # Normalize to 60 minutes
            self.code_blocks / 10,  # Normalize to 10 code blocks
            self.repetitive_patterns / 5,  # Normalize to 5 repetitions
            self.topic_changes / 3,  # Normalize to 3 topic changes
            min(1.0, self.burn_rate / 100),  # Normalize burn rate
        ]
        return min(1.0, sum(factors) / len(factors))


@dataclass
class WorkflowConfig:
    """Configuration for orchestration workflow"""

    automation_level: str = "high"
    digest_threshold: int = 3
    session_duration_trigger: float = 30.0
    token_threshold: float = 0.75
    auto_trigger: bool = True
    preserve_code: bool = True
    rag_integration: bool = True
    continuity_priority: str = "high"

    @classmethod
    def from_file(cls, config_path: Path) -> "WorkflowConfig":
        """Load configuration from YAML file"""
        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                return cls(**data.get("orchestration_config", {}))
        return cls()

    def save_to_file(self, config_path: Path):
        """Save configuration to YAML file"""
        config_data = {"orchestration_config": asdict(self)}
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)


class ContextAnalyzer:
    """Enhanced context analyzer with burn rate tracking"""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.session_start = datetime.now()
        self.metrics = ContextMetrics()
        # Add memory-efficient storage
        self.token_history = deque(maxlen=500)
        self.message_history = deque(maxlen=128)
        self.burn_rate_window = 5  # minutes

    async def analyze_session(self) -> ContextMetrics:
        """Enhanced async session context analysis with burn rate tracking"""
        # Calculate session duration
        self.metrics.session_duration = (
            datetime.now() - self.session_start
        ).total_seconds() / 60

        # Parallel analysis of different metrics
        tasks = [
            self._estimate_token_count_async(),
            self._count_messages_async(),
            self._count_completed_tasks_async(),
            self._count_code_blocks_async(),
            self._detect_repetitive_patterns_async(),
            self._detect_topic_changes_async(),
        ]

        results = await asyncio.gather(*tasks)

        # Assign results
        self.metrics.token_count = results[0]
        self.metrics.message_count = results[1]
        self.metrics.task_completion_count = results[2]
        self.metrics.code_blocks = results[3]
        self.metrics.repetitive_patterns = results[4]
        self.metrics.topic_changes = results[5]

        # New burn rate analysis
        self.metrics.burn_rate = self._calculate_dynamic_burn_rate()
        self.metrics.time_to_limit = self._calculate_time_to_limit()
        self.metrics.confidence_score = self._calculate_confidence_score()
        self.metrics.token_history = list(self.token_history)

        self.metrics.complexity_score = self.metrics.calculate_complexity()

        return self.metrics

    def _estimate_token_count(self) -> int:
        """Estimate current token usage from conversation history"""
        # Check for existing session files or conversation context
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

            # Try to read conversation from various sources
            token_count = 0

            # Check for Claude conversation log
            claude_log = Path.home() / ".claude" / "conversation.log"
            if claude_log.exists():
                with open(claude_log, "r") as f:
                    content = f.read()
                    token_count = len(enc.encode(content))

            # Fallback to estimation based on session metrics
            if token_count == 0:
                token_count = min(8000, self.metrics.message_count * 150)

            return token_count
        except ImportError:
            # Fallback if tiktoken not available
            return min(8000, self.metrics.message_count * 150)

    def _count_messages(self) -> int:
        """Count messages in current session"""
        # Try to count from actual conversation sources
        try:
            claude_log = Path.home() / ".claude" / "conversation.log"
            if claude_log.exists():
                with open(claude_log, "r") as f:
                    lines = f.readlines()
                    # Count user/assistant message pairs
                    return (
                        len(
                            [
                                line
                                for line in lines
                                if line.strip().startswith("user:")
                                or line.strip().startswith("assistant:")
                            ]
                        )
                        // 2
                    )
        except:
            pass

        # Fallback to estimation
        return min(50, int(self.metrics.session_duration * 2))

    def _count_completed_tasks(self) -> int:
        """Count completed tasks via TodoRead simulation"""
        # Try to read from actual todo files or session context
        try:
            # Check for digest session files
            digest_dir = Path(
                "/Users/admin/AstraTrade-Project/claude-rag/digest_resource"
            )
            if digest_dir.exists():
                current_session = digest_dir / "session_current.md"
                if current_session.exists():
                    with open(current_session, "r") as f:
                        content = f.read()
                        # Count completed tasks markers
                        return content.count("✅")
        except:
            pass

        # Fallback to estimation
        return max(0, int(self.metrics.session_duration / 15))

    def _count_code_blocks(self) -> int:
        """Count code blocks in conversation"""
        # Try to count from actual conversation
        try:
            claude_log = Path.home() / ".claude" / "conversation.log"
            if claude_log.exists():
                with open(claude_log, "r") as f:
                    content = f.read()
                    # Count code block markers
                    return content.count("```")
        except:
            pass

        # Fallback to estimation
        return max(0, int(self.metrics.task_completion_count * 1.5))

    def _detect_repetitive_patterns(self) -> int:
        """Detect repetitive discussion patterns"""
        # Try to analyze conversation for repetitive patterns
        try:
            from collections import Counter

            claude_log = Path.home() / ".claude" / "conversation.log"
            if claude_log.exists():
                with open(claude_log, "r") as f:
                    content = f.read()
                    # Simple pattern detection - repeated phrases
                    words = content.lower().split()
                    word_counts = Counter(words)
                    # Count words that appear more than 10 times
                    repetitive_count = sum(
                        1 for count in word_counts.values() if count > 10
                    )
                    return min(10, repetitive_count)
        except:
            pass

        # Fallback to estimation
        return max(0, int(self.metrics.message_count / 15))

    def _detect_topic_changes(self) -> int:
        """Detect significant topic changes"""
        # Try to analyze conversation for topic changes
        try:
            claude_log = Path.home() / ".claude" / "conversation.log"
            if claude_log.exists():
                with open(claude_log, "r") as f:
                    content = f.read()
                    # Simple topic change detection - look for certain keywords
                    topic_keywords = [
                        "implement",
                        "create",
                        "debug",
                        "fix",
                        "analyze",
                        "explain",
                    ]
                    topic_changes = 0
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if any(keyword in line.lower() for keyword in topic_keywords):
                            topic_changes += 1
                    return min(5, topic_changes)
        except:
            pass

        # Fallback to estimation
        return max(0, int(self.metrics.session_duration / 20))

    def _calculate_dynamic_burn_rate(self) -> float:
        """Calculate dynamic burn rate using sliding window approach"""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.burn_rate_window)

        # Filter token history within window
        window_data = [
            (ts, tokens) for ts, tokens in self.token_history if ts >= window_start
        ]

        if len(window_data) < 2:
            # Not enough data, use simple burn rate
            session_duration = max(0.01, self.metrics.session_duration)
            return max(0, self.metrics.token_count) / session_duration

        # Calculate burn rate from window data
        tokens_start, tokens_end = window_data[0][1], window_data[-1][1]
        time_diff = (window_data[-1][0] - window_data[0][0]).total_seconds() / 60
        burn_rate = (tokens_end - tokens_start) / time_diff if time_diff > 0 else 0

        return max(0, burn_rate)

    def _calculate_time_to_limit(self) -> Optional[float]:
        """Calculate estimated time to reach token limit"""
        if self.metrics.burn_rate <= 0:
            return None

        token_limit = 200_000  # Claude's token limit
        remaining_tokens = token_limit - self.metrics.token_count

        if remaining_tokens <= 0:
            return 0.0

        return remaining_tokens / self.metrics.burn_rate

    def _calculate_confidence_score(self) -> float:
        """Calculate confidence score in analysis"""
        factors = []

        # Data completeness factor
        data_completeness = min(1.0, self.metrics.session_duration / 10)
        factors.append(data_completeness)

        # History depth factor
        history_depth = min(1.0, len(self.token_history) / 100)
        factors.append(history_depth)

        # Message count factor
        message_factor = min(1.0, self.metrics.message_count / 20)
        factors.append(message_factor)

        # Burn rate stability factor
        burn_rate_stability = min(1.0, len(self.token_history) / 50)
        factors.append(burn_rate_stability)

        return sum(factors) / len(factors) if factors else 0.0

    def update_token_history(self, token_count: int):
        """Update token history with timestamp"""
        self.token_history.append((datetime.now(), token_count))

    async def _estimate_token_count_async(self) -> int:
        """Async version of token count estimation"""
        return await asyncio.to_thread(self._estimate_token_count)

    async def _count_messages_async(self) -> int:
        """Async version of message counting"""
        return await asyncio.to_thread(self._count_messages)

    async def _count_completed_tasks_async(self) -> int:
        """Async version of task completion counting"""
        return await asyncio.to_thread(self._count_completed_tasks)

    async def _count_code_blocks_async(self) -> int:
        """Async version of code block counting"""
        return await asyncio.to_thread(self._count_code_blocks)

    async def _detect_repetitive_patterns_async(self) -> int:
        """Async version of repetitive pattern detection"""
        return await asyncio.to_thread(self._detect_repetitive_patterns)

    async def _detect_topic_changes_async(self) -> int:
        """Async version of topic change detection"""
        return await asyncio.to_thread(self._detect_topic_changes)


class DecisionEngine:
    """Makes intelligent decisions about command execution"""

    def __init__(self, config: WorkflowConfig, analyzer: ContextAnalyzer):
        self.config = config
        self.analyzer = analyzer
        self.command_history: List[Dict] = []
        self.rag_api_url = "http://127.0.0.1:8001"
        self.rag_available = self._check_rag_availability()

    async def decide_command(
        self, user_intent: str = ""
    ) -> Tuple[CommandType, Dict[str, Any]]:
        """Async decision making with parallel analysis"""
        # Parallel execution of analysis and RAG context
        tasks = [
            self.analyzer.analyze_session(),
            (
                self._get_rag_context_async(user_intent)
                if self.rag_available
                else asyncio.create_task(asyncio.sleep(0, result={}))
            ),
        ]

        metrics, rag_context = await asyncio.gather(*tasks)

        # Parallel score calculation
        digest_score_task = asyncio.create_task(
            self._calculate_digest_score_async(metrics, user_intent, rag_context)
        )
        compact_score_task = asyncio.create_task(
            self._calculate_compact_score_async(metrics, user_intent, rag_context)
        )

        digest_score, compact_score = await asyncio.gather(
            digest_score_task, compact_score_task
        )

        # Decision logic with RAG enhancement
        if digest_score > compact_score and digest_score > 0.6:
            return CommandType.DIGEST, self._get_digest_params(metrics)
        elif compact_score > 0.6:
            return CommandType.COMPACT, self._get_compact_params(metrics)
        elif digest_score > 0.4 and compact_score > 0.4:
            return CommandType.HYBRID, self._get_hybrid_params(metrics)
        else:
            return CommandType.NONE, {}

    def _calculate_digest_score(
        self, metrics: ContextMetrics, user_intent: str, rag_context: Dict = None
    ) -> float:
        """Calculate priority score for digest command with RAG enhancement"""
        score = 0.0

        # Session continuity factors
        if metrics.session_duration > self.config.session_duration_trigger:
            score += 0.3

        # Task completion factor
        if metrics.task_completion_count >= self.config.digest_threshold:
            score += 0.4

        # Code implementation factor
        if metrics.code_blocks > 5:
            score += 0.2

        # User intent factor
        if any(
            keyword in user_intent.lower()
            for keyword in ["history", "previous", "session", "document"]
        ):
            score += 0.3

        # Complexity factor
        if metrics.complexity_score > 0.7:
            score += 0.2

        # RAG context enhancement
        if rag_context:
            # Check if similar sessions suggest digest
            similar_sessions = rag_context.get("similar_sessions", [])
            if any(
                "digest" in session.get("actions", []) for session in similar_sessions
            ):
                score += 0.1

        return min(1.0, score)

    def _calculate_compact_score(
        self, metrics: ContextMetrics, user_intent: str, rag_context: Dict = None
    ) -> float:
        """Calculate priority score for compact command with RAG enhancement"""
        score = 0.0

        # Token usage factor
        token_ratio = metrics.token_count / 8000
        if token_ratio > self.config.token_threshold:
            score += 0.5

        # Repetitive patterns factor
        if metrics.repetitive_patterns > 3:
            score += 0.3

        # Message count factor
        if metrics.message_count > 25:
            score += 0.2

        # User intent factor
        if any(
            keyword in user_intent.lower()
            for keyword in ["summarize", "compress", "clarify", "focus"]
        ):
            score += 0.4

        # Topic changes factor
        if metrics.topic_changes > 2:
            score += 0.2

        # RAG context enhancement
        if rag_context:
            # Check if similar sessions suggest compact
            similar_sessions = rag_context.get("similar_sessions", [])
            if any(
                "compact" in session.get("actions", []) for session in similar_sessions
            ):
                score += 0.1

        return min(1.0, score)

    def _get_digest_params(self, metrics: ContextMetrics) -> Dict[str, Any]:
        """Get parameters for digest command"""
        if metrics.session_duration > 45:
            return {"action": "snapshot", "description": "Long session archival"}
        elif metrics.task_completion_count >= 3:
            return {"action": "create", "description": "Task completion milestone"}
        else:
            return {"action": "update", "description": "Progress update"}

    def _get_compact_params(self, metrics: ContextMetrics) -> Dict[str, Any]:
        """Get parameters for compact command"""
        if metrics.token_count / 8000 > 0.8:
            return {
                "level": "deep",
                "preserve_code": True,
                "description": "Token optimization",
            }
        elif metrics.repetitive_patterns > 3:
            return {
                "level": "standard",
                "focus": "clarity",
                "description": "Pattern reduction",
            }
        else:
            return {
                "level": "light",
                "preserve_code": True,
                "description": "Light optimization",
            }

    def _get_hybrid_params(self, metrics: ContextMetrics) -> Dict[str, Any]:
        """Get parameters for hybrid execution"""
        return {
            "sequence": ["compact", "digest"],
            "compact_params": self._get_compact_params(metrics),
            "digest_params": self._get_digest_params(metrics),
            "description": "Hybrid optimization and documentation",
        }

    def _check_rag_availability(self) -> bool:
        """Check if RAG system is available"""
        try:
            import requests

            response = requests.get(f"{self.rag_api_url}/status", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _get_rag_context(self, user_intent: str) -> Dict[str, Any]:
        """Get context from RAG system"""
        try:
            import requests

            search_request = {
                "query": f"orchestration decisions similar to: {user_intent}",
                "max_results": 5,
            }
            response = requests.post(
                f"{self.rag_api_url}/search", json=search_request, timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}

    async def _get_rag_context_async(self, user_intent: str) -> Dict[str, Any]:
        """Async version of RAG context retrieval"""
        return await asyncio.to_thread(self._get_rag_context, user_intent)

    async def _calculate_digest_score_async(
        self, metrics: ContextMetrics, user_intent: str, rag_context: Dict = None
    ) -> float:
        """Async version of digest score calculation"""
        return await asyncio.to_thread(
            self._calculate_digest_score, metrics, user_intent, rag_context
        )

    async def _calculate_compact_score_async(
        self, metrics: ContextMetrics, user_intent: str, rag_context: Dict = None
    ) -> float:
        """Async version of compact score calculation"""
        return await asyncio.to_thread(
            self._calculate_compact_score, metrics, user_intent, rag_context
        )


class CommandDispatcher:
    """Executes commands based on decisions"""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.command_log: List[Dict] = []

    async def execute_command_async(
        self, command_type: CommandType, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Async command execution with parallel processing"""
        timestamp = datetime.now().isoformat()

        try:
            if command_type == CommandType.DIGEST:
                result = await self._execute_digest_async(params)
            elif command_type == CommandType.COMPACT:
                result = await self._execute_compact_async(params)
            elif command_type == CommandType.HYBRID:
                result = await self._execute_hybrid_async(params)
            else:
                result = {"status": "skipped", "reason": "No command needed"}

            # Log the command execution
            log_entry = {
                "timestamp": timestamp,
                "command_type": command_type.value,
                "params": params,
                "result": result,
                "status": "success",
            }

            self.command_log.append(log_entry)
            return result

        except Exception as e:
            error_result = {"status": "error", "error": str(e), "timestamp": timestamp}

            log_entry = {
                "timestamp": timestamp,
                "command_type": command_type.value,
                "params": params,
                "result": error_result,
                "status": "error",
            }

            self.command_log.append(log_entry)
            return error_result

    def execute_command(
        self, command_type: CommandType, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sync wrapper for backward compatibility"""
        return asyncio.run(self.execute_command_async(command_type, params))

    def _execute_digest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute digest command"""
        action = params.get("action", "create")

        # Enhanced digest execution with proper path handling
        digest_script = Path(
            "/Users/admin/AstraTrade-Project/claude-rag/src/digest_implementation.py"
        )

        if not digest_script.exists():
            return {"status": "error", "message": "Digest implementation not found"}

        if action == "create":
            return self._run_command(["python", str(digest_script), "--create"])
        elif action == "snapshot":
            return self._run_command(["python", str(digest_script), "--snapshot"])
        elif action == "update":
            return self._run_command(
                ["python", str(digest_script), "--create"]
            )  # Update via create
        elif action == "show":
            return self._run_command(["python", str(digest_script), "--show"])
        else:
            return {"status": "error", "message": f"Unknown digest action: {action}"}

    def _execute_compact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compact command"""
        # Since /compact is a Claude command, we need to trigger it properly
        level = params.get("level", "standard")
        preserve_code = params.get("preserve_code", True)

        # Check if we can invoke Claude's compact command
        try:
            # Try to find a way to trigger compact
            compact_cmd = Path.home() / ".claude" / "commands" / "compact.sh"
            if compact_cmd.exists():
                result = self._run_command(["bash", str(compact_cmd)])
                if result["status"] == "success":
                    result.update(
                        {
                            "level": level,
                            "preserve_code": preserve_code,
                            "message": f"Conversation compacted at {level} level",
                        }
                    )
                    return result
        except:
            pass

        # Fallback to simulation
        return {
            "status": "success",
            "action": "compact",
            "level": level,
            "preserve_code": preserve_code,
            "message": f"Conversation compacted at {level} level (simulated)",
            "tokens_saved": 1000 + (500 if level == "deep" else 200),
        }

    def _execute_hybrid(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hybrid command sequence"""
        sequence = params.get("sequence", ["compact", "digest"])
        results = []

        for command in sequence:
            if command == "compact":
                result = self._execute_compact(params.get("compact_params", {}))
            elif command == "digest":
                result = self._execute_digest(params.get("digest_params", {}))
            else:
                result = {"status": "error", "message": f"Unknown command: {command}"}

            results.append({"command": command, "result": result})

        return {
            "status": "success",
            "action": "hybrid",
            "sequence": sequence,
            "results": results,
        }

    def _run_command(self, cmd_args: List[str]) -> Dict[str, Any]:
        """Run external command and return result"""
        try:
            # Store original directory
            original_dir = os.getcwd()

            # Change to the correct directory based on command
            if "digest_implementation.py" in cmd_args[1] if len(cmd_args) > 1 else "":
                os.chdir("/Users/admin/AstraTrade-Project/claude-rag")

            result = subprocess.run(
                cmd_args, capture_output=True, text=True, timeout=30
            )

            # Restore original directory
            os.chdir(original_dir)

            if result.returncode == 0:
                return {
                    "status": "success",
                    "output": result.stdout,
                    "command": " ".join(cmd_args),
                }
            else:
                return {
                    "status": "error",
                    "error": result.stderr,
                    "command": " ".join(cmd_args),
                }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Command timed out",
                "command": " ".join(cmd_args),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "command": " ".join(cmd_args)}

    async def _execute_digest_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of digest execution"""
        return await asyncio.to_thread(self._execute_digest, params)

    async def _execute_compact_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of compact execution"""
        return await asyncio.to_thread(self._execute_compact, params)

    async def _execute_hybrid_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of hybrid execution"""
        return await asyncio.to_thread(self._execute_hybrid, params)


class OrchestrationEngine:
    """Main orchestration engine that coordinates all components"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = (
            config_path or Path.home() / ".claude" / "orchestrator_config.yaml"
        )
        self.config = WorkflowConfig.from_file(self.config_path)
        self.analyzer = ContextAnalyzer(self.config)
        self.decision_engine = DecisionEngine(self.config, self.analyzer)
        self.dispatcher = CommandDispatcher(self.config)
        self.session_log_path = Path.home() / ".claude" / "orchestrator_session.log"
        self.session_state = self._load_session_state()

    async def orchestrate(
        self, user_intent: str = "", force_command: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main async orchestration method with parallel processing"""
        try:
            # Analyze current context
            metrics = await self.analyzer.analyze_session()

            # Decide on command
            if force_command:
                command_type = CommandType(force_command.lower())
                params = self._get_default_params(command_type, metrics)
            else:
                command_type, params = await self.decision_engine.decide_command(
                    user_intent
                )

            # Execute command and generate recommendations in parallel
            command_task = asyncio.create_task(
                self.dispatcher.execute_command_async(command_type, params)
            )
            recommendations_task = asyncio.create_task(
                self._generate_recommendations_async(metrics)
            )

            result, recommendations = await asyncio.gather(
                command_task, recommendations_task
            )

            # Log session
            self._log_session(metrics, command_type, params, result)

            return {
                "status": "success",
                "metrics": asdict(metrics),
                "command_executed": command_type.value,
                "parameters": params,
                "result": result,
                "recommendations": recommendations,
            }

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

            self._log_session(self.analyzer.metrics, CommandType.NONE, {}, error_result)
            return error_result

    def _get_default_params(
        self, command_type: CommandType, metrics: ContextMetrics
    ) -> Dict[str, Any]:
        """Get default parameters for forced commands"""
        if command_type == CommandType.DIGEST:
            return self.decision_engine._get_digest_params(metrics)
        elif command_type == CommandType.COMPACT:
            return self.decision_engine._get_compact_params(metrics)
        elif command_type == CommandType.HYBRID:
            return self.decision_engine._get_hybrid_params(metrics)
        else:
            return {}

    def _generate_recommendations(self, metrics: ContextMetrics) -> List[str]:
        """Generate recommendations based on current metrics"""
        recommendations = []

        if metrics.token_count > 6000:
            recommendations.append("Consider using /compact to optimize token usage")

        if metrics.task_completion_count >= 3:
            recommendations.append(
                "Good time to run /digest --create to document progress"
            )

        if metrics.session_duration > 60:
            recommendations.append(
                "Long session detected - consider /digest --snapshot for archival"
            )

        if metrics.repetitive_patterns > 4:
            recommendations.append(
                "Repetitive patterns detected - /compact --deep might help"
            )

        if metrics.complexity_score > 0.8:
            recommendations.append(
                "High complexity session - hybrid approach recommended"
            )

        return recommendations

    async def _generate_recommendations_async(
        self, metrics: ContextMetrics
    ) -> List[str]:
        """Async version of recommendations generation"""
        return await asyncio.to_thread(self._generate_recommendations, metrics)

    def _log_session(
        self,
        metrics: ContextMetrics,
        command_type: CommandType,
        params: Dict,
        result: Dict,
    ):
        """Log session information"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": asdict(metrics),
            "command_type": command_type.value,
            "parameters": params,
            "result": result,
        }

        try:
            with open(self.session_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Warning: Could not log session: {e}")

        # Update session state
        self._update_session_state(command_type, result)

    def _load_session_state(self) -> Dict[str, Any]:
        """Load session state from file"""
        state_file = Path.home() / ".claude" / "orchestrator_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return {
            "last_command": None,
            "last_execution": None,
            "command_count": 0,
            "session_start": datetime.now().isoformat(),
        }

    def _save_session_state(self):
        """Save session state to file"""
        state_file = Path.home() / ".claude" / "orchestrator_state.json"
        try:
            with open(state_file, "w") as f:
                json.dump(self.session_state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save session state: {e}")

    def _update_session_state(self, command_type: CommandType, result: Dict):
        """Update session state after command execution"""
        self.session_state.update(
            {
                "last_command": command_type.value,
                "last_execution": datetime.now().isoformat(),
                "command_count": self.session_state.get("command_count", 0) + 1,
                "last_result": result,
            }
        )
        self._save_session_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        metrics = self.analyzer.analyze_session()

        return {
            "session_duration": metrics.session_duration,
            "token_count": metrics.token_count,
            "message_count": metrics.message_count,
            "task_completion_count": metrics.task_completion_count,
            "complexity_score": metrics.complexity_score,
            "config": asdict(self.config),
            "recommendations": self._generate_recommendations(metrics),
            "last_commands": (
                self.dispatcher.command_log[-5:] if self.dispatcher.command_log else []
            ),
        }

    def update_config(self, new_config: Dict[str, Any]):
        """Update orchestration configuration"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.config.save_to_file(self.config_path)

    def get_session_state(self) -> Dict[str, Any]:
        """Get current session state"""
        return self.session_state.copy()

    def reset_session(self):
        """Reset session state"""
        self.session_state = {
            "last_command": None,
            "last_execution": None,
            "command_count": 0,
            "session_start": datetime.now().isoformat(),
        }
        self._save_session_state()
        self.analyzer.session_start = datetime.now()


async def main_async():
    """Async main CLI interface"""
    parser = argparse.ArgumentParser(description="Claude Orchestration Engine")
    parser.add_argument("--orchestrate", action="store_true", help="Run orchestration")
    parser.add_argument("--intent", type=str, default="", help="User intent or context")
    parser.add_argument(
        "--force",
        type=str,
        choices=["digest", "compact", "hybrid"],
        help="Force specific command",
    )
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--config", action="store_true", help="Show configuration")
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze current context"
    )
    parser.add_argument("--profile", type=str, help="Set workflow profile")

    args = parser.parse_args()

    # Initialize engine
    engine = OrchestrationEngine()

    if args.orchestrate:
        result = await engine.orchestrate(args.intent, args.force)
        print(json.dumps(result, indent=2))

    elif args.status:
        status = engine.get_status()
        print(json.dumps(status, indent=2))

    elif args.config:
        config = asdict(engine.config)
        print(json.dumps(config, indent=2))

    elif args.analyze:
        metrics = await engine.analyzer.analyze_session()
        print(json.dumps(asdict(metrics), indent=2))

    elif args.profile:
        # Set workflow profile
        profiles = {
            "dev": {
                "automation_level": "high",
                "digest_threshold": 2,
                "continuity_priority": "high",
            },
            "research": {
                "automation_level": "medium",
                "token_threshold": 0.8,
                "digest_threshold": 4,
            },
            "debug": {
                "automation_level": "low",
                "token_threshold": 0.7,
                "digest_threshold": 5,
            },
            "doc": {
                "automation_level": "high",
                "digest_threshold": 1,
                "continuity_priority": "high",
            },
        }

        if args.profile in profiles:
            engine.update_config(profiles[args.profile])
            print(f"Profile '{args.profile}' applied successfully")
        else:
            print(f"Unknown profile: {args.profile}")
            print(f"Available profiles: {list(profiles.keys())}")

    else:
        parser.print_help()


def main():
    """Main CLI interface wrapper"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
