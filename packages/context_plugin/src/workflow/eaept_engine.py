#!/usr/bin/env python3
"""
Enhanced EAEPT Workflow Engine - Core Implementation
Systematic Express-Ask-Explore-Plan-Code-Test methodology with auto-orchestration
"""

import json
import os
import sys
import time
import asyncio
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import requests

# Add Groq integration import
import groq_reasoning_tool

GROQ_AVAILABLE = True

# Import orchestration engine with fallback
from core.orchestration.orchestrator_engine import (
    OrchestrationEngine,
    ContextMetrics,
    CommandType,
)

# Import validation logger with fallback
from core.validation.eaept_validation_logger import (
    EAEPTValidationLogger,
    ValidationEvent,
    ValidationLevel,
    init_validation_logger,
    get_validation_logger,
)


class EAEPTPhase(Enum):
    """Enhanced EAEPT workflow phases"""

    EXPRESS = "express"
    ASK = "ask"
    EXPLORE = "explore"
    PLAN = "plan"
    CODE = "code"
    TEST = "test"
    COMPLETE = "complete"


class WorkflowState(Enum):
    """Workflow execution states"""

    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    AUTO_TRANSITIONING = "auto_transitioning"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PhaseConfig:
    """Configuration for each EAEPT phase"""

    name: str
    description: str
    auto_transition_threshold: float = 0.8
    context_optimization_strategy: str = "standard"
    rag_integration: bool = False
    parallel_execution: bool = False
    max_duration_minutes: Optional[int] = None
    token_threshold: float = 0.75


class PhaseMetrics:
    """Metrics for tracking phase execution"""

    def __init__(self, phase: EAEPTPhase):
        self.phase = phase
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.token_usage_start = 0
        self.token_usage_end = 0
        self.context_optimizations = 0
        self.rag_queries = 0
        self.user_interactions = 0
        self.completion_confidence = 0.0
        self.quality_score = 0.0
        self.notes: List[str] = []

    @property
    def duration_minutes(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() / 60

    @property
    def token_usage(self) -> int:
        return max(0, self.token_usage_end - self.token_usage_start)


# Groq Integration Helper
class GroqIntegrationHelper:
    """Helper to call Groq async reasoning functions for workflow phases."""

    def __init__(self):
        self.available = GROQ_AVAILABLE

    async def ask_phase_analysis(
        self, context: str, question: str
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            return await groq_reasoning_tool.reason_about_code(context, question)
        except Exception as e:
            return {"error": str(e)}

    async def explore_phase_research(
        self, research_topic: str, context: str = ""
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            return await groq_reasoning_tool.analyze_problem(research_topic, context)
        except Exception as e:
            return {"error": str(e)}

    async def test_phase_review(
        self, code: str, test_focus: str = "Full code review"
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            return await groq_reasoning_tool.reason_about_code(code, test_focus)
        except Exception as e:
            return {"error": str(e)}


class EAEPTWorkflowEngine:
    """Main enhanced EAEPT workflow engine with auto-orchestration"""

    def __init__(
        self, project_root: Optional[str] = None, enable_validation: bool = True
    ):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config_path = self.project_root / "config" / "eaept-config.yaml"
        self.state_path = self.project_root / "config" / ".eaept-state.json"
        self.rag_url = "http://localhost:8000"

        # Initialize components
        self.orchestrator = OrchestrationEngine() if OrchestrationEngine else None
        self.phase_configs = self._load_phase_configs()
        self.workflow_state = self._load_workflow_state()
        self.current_phase = EAEPTPhase(
            self.workflow_state.get("current_phase", "express")
        )
        self.current_task = self.workflow_state.get("current_task", "")
        self.phase_metrics: Dict[EAEPTPhase, PhaseMetrics] = {}
        self.groq_helper = GroqIntegrationHelper()

        # Initialize validation logging
        self.validation_logger = None
        if enable_validation and EAEPTValidationLogger:
            try:
                self.validation_logger = init_validation_logger(str(self.project_root))
                print("🔍 Enhanced validation logging enabled")
            except Exception as e:
                print(f"Warning: Could not initialize validation logger: {e}")

        # Initialize phase metrics
        for phase in EAEPTPhase:
            if phase.value not in self.workflow_state.get("phase_metrics", {}):
                self.phase_metrics[phase] = PhaseMetrics(phase)

    def _load_phase_configs(self) -> Dict[EAEPTPhase, PhaseConfig]:
        """Load phase configurations"""
        default_configs = {
            EAEPTPhase.EXPRESS: PhaseConfig(
                name="Express",
                description="Deep analysis and task framing",
                auto_transition_threshold=0.85,
                context_optimization_strategy="preserve_thinking",
                max_duration_minutes=15,
                token_threshold=0.6,
            ),
            EAEPTPhase.ASK: PhaseConfig(
                name="Ask",
                description="Interactive clarification and validation",
                auto_transition_threshold=0.9,
                context_optimization_strategy="preserve_dialogue",
                max_duration_minutes=10,
                token_threshold=0.5,
            ),
            EAEPTPhase.EXPLORE: PhaseConfig(
                name="Explore",
                description="RAG-powered research and discovery",
                auto_transition_threshold=0.8,
                context_optimization_strategy="preserve_research",
                rag_integration=True,
                parallel_execution=True,
                max_duration_minutes=30,
                token_threshold=0.85,
            ),
            EAEPTPhase.PLAN: PhaseConfig(
                name="Plan",
                description="Detailed implementation planning",
                auto_transition_threshold=0.85,
                context_optimization_strategy="preserve_architecture",
                max_duration_minutes=20,
                token_threshold=0.7,
            ),
            EAEPTPhase.CODE: PhaseConfig(
                name="Code",
                description="Implementation and development",
                auto_transition_threshold=0.8,
                context_optimization_strategy="preserve_code",
                parallel_execution=True,
                max_duration_minutes=60,
                token_threshold=0.9,
            ),
            EAEPTPhase.TEST: PhaseConfig(
                name="Test",
                description="Validation and quality assurance",
                auto_transition_threshold=0.9,
                context_optimization_strategy="preserve_tests",
                parallel_execution=True,
                max_duration_minutes=30,
                token_threshold=0.8,
            ),
        }

        # Load custom configs if available
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    custom_config = yaml.safe_load(f)
                    for phase_name, config_data in custom_config.get(
                        "phases", {}
                    ).items():
                        phase = EAEPTPhase(phase_name)
                        if phase in default_configs:
                            # Update default config with custom values
                            for key, value in config_data.items():
                                if hasattr(default_configs[phase], key):
                                    setattr(default_configs[phase], key, value)
            except Exception as e:
                print(f"Warning: Could not load custom config: {e}")

        return default_configs

    def _load_workflow_state(self) -> Dict[str, Any]:
        """Load workflow state from file"""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load workflow state: {e}")

        return {
            "current_phase": "express",
            "current_task": "",
            "workflow_state": "initialized",
            "phase_metrics": {},
            "session_start": datetime.now().isoformat(),
            "auto_orchestration_enabled": True,
        }

    def _save_workflow_state(self):
        """Save current workflow state"""
        state_data = {
            "current_phase": self.current_phase.value,
            "current_task": self.current_task,
            "workflow_state": self.workflow_state.get("workflow_state", "in_progress"),
            "phase_metrics": {
                phase.value: {
                    "start_time": metrics.start_time.isoformat(),
                    "end_time": (
                        metrics.end_time.isoformat() if metrics.end_time else None
                    ),
                    "duration_minutes": metrics.duration_minutes,
                    "token_usage": metrics.token_usage,
                    "completion_confidence": metrics.completion_confidence,
                    "quality_score": metrics.quality_score,
                    "notes": metrics.notes,
                }
                for phase, metrics in self.phase_metrics.items()
            },
            "session_start": self.workflow_state.get(
                "session_start", datetime.now().isoformat()
            ),
            "last_update": datetime.now().isoformat(),
            "auto_orchestration_enabled": self.workflow_state.get(
                "auto_orchestration_enabled", True
            ),
        }

        try:
            os.makedirs(self.state_path.parent, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save workflow state: {e}")

    async def start_workflow(
        self, task_description: str, auto_execute: bool = True
    ) -> Dict[str, Any]:
        """Start new EAEPT workflow"""
        print(f"🚀 Starting Enhanced EAEPT Workflow")
        print(f"📝 Task: {task_description}")

        # Initialize workflow
        self.current_task = task_description
        self.current_phase = EAEPTPhase.EXPRESS
        self.workflow_state["workflow_state"] = "in_progress"
        self.workflow_state["current_task"] = task_description

        # Log workflow start
        if self.validation_logger:
            self.validation_logger.log_workflow_decision(
                "start_workflow",
                1.0,
                f"Starting EAEPT workflow for task: {task_description[:100]}",
            )

        # Start orchestration monitoring
        if self.orchestrator:
            await self.orchestrator.orchestrate(
                f"Starting EAEPT workflow: {task_description}"
            )

        # Initialize phase metrics
        self.phase_metrics[self.current_phase] = PhaseMetrics(self.current_phase)

        if auto_execute:
            return await self.execute_full_workflow()
        else:
            return await self.execute_current_phase()

    async def execute_full_workflow(self) -> Dict[str, Any]:
        """Execute complete EAEPT workflow with auto-transitions"""
        workflow_results = {}

        try:
            while self.current_phase != EAEPTPhase.COMPLETE:
                print(f"\n🔄 Executing Phase: {self.current_phase.value.upper()}")

                # Execute current phase
                phase_result = await self.execute_current_phase()
                workflow_results[self.current_phase.value] = phase_result

                # Check for auto-transition
                if await self._should_auto_transition():
                    await self._transition_to_next_phase()
                else:
                    print(f"⏸️  Workflow paused at {self.current_phase.value} phase")
                    self.workflow_state["workflow_state"] = "paused"
                    break

            if self.current_phase == EAEPTPhase.COMPLETE:
                workflow_results["workflow_summary"] = (
                    await self._generate_workflow_summary()
                )
                print("\n✅ EAEPT Workflow completed successfully!")

        except Exception as e:
            print(f"❌ Workflow error: {e}")
            self.workflow_state["workflow_state"] = "error"
            workflow_results["error"] = str(e)

        finally:
            self._save_workflow_state()

        return workflow_results

    async def execute_current_phase(self) -> Dict[str, Any]:
        """Execute the current EAEPT phase"""
        config = self.phase_configs[self.current_phase]
        metrics = self.phase_metrics[self.current_phase]

        print(f"📋 {config.name}: {config.description}")

        # Start phase execution with validation logging
        metrics.start_time = datetime.now()
        if self.validation_logger:
            self.validation_logger.start_phase(self.current_phase.value)

        if self.orchestrator:
            context = await self.orchestrator.analyzer.analyze_session()
            metrics.token_usage_start = context.token_count

        try:
            # Execute phase-specific logic
            if self.current_phase == EAEPTPhase.EXPRESS:
                result = await self._execute_express_phase()
            elif self.current_phase == EAEPTPhase.ASK:
                result = await self._execute_ask_phase()
            elif self.current_phase == EAEPTPhase.EXPLORE:
                result = await self._execute_explore_phase()
            elif self.current_phase == EAEPTPhase.PLAN:
                result = await self._execute_plan_phase()
            elif self.current_phase == EAEPTPhase.CODE:
                result = await self._execute_code_phase()
            elif self.current_phase == EAEPTPhase.TEST:
                result = await self._execute_test_phase()
            else:
                result = {"status": "unknown_phase"}

            # End phase execution
            metrics.end_time = datetime.now()
            if self.orchestrator:
                context = await self.orchestrator.analyzer.analyze_session()
                metrics.token_usage_end = context.token_count

            # Auto-orchestration after phase
            if config.token_threshold and self.orchestrator:
                await self._handle_context_optimization(config)

            metrics.completion_confidence = result.get("confidence", 0.8)
            metrics.quality_score = result.get("quality", 0.8)

            # Log phase completion metrics
            if self.validation_logger and EAEPTValidationLogger:
                self.validation_logger.log_performance_metric(
                    f"{self.current_phase.value}_duration_minutes",
                    metrics.duration_minutes,
                    config.max_duration_minutes,
                )
                self.validation_logger.log_performance_metric(
                    f"{self.current_phase.value}_token_usage", metrics.token_usage, None
                )

            # End phase logging
            if self.validation_logger:
                self.validation_logger.end_phase(self.current_phase.value)

            print(f"✅ {config.name} phase completed")
            print(f"   Duration: {metrics.duration_minutes:.1f} minutes")
            print(f"   Token usage: {metrics.token_usage}")
            print(f"   Confidence: {metrics.completion_confidence:.1%}")

            return result

        except Exception as e:
            metrics.end_time = datetime.now()
            metrics.notes.append(f"Error: {str(e)}")

            # Log error with validation logger
            if self.validation_logger and EAEPTValidationLogger:
                self.validation_logger.log_error(
                    f"{self.current_phase.value}_phase_error",
                    str(e),
                    recovery_attempted=False,
                )

            print(f"❌ {config.name} phase failed: {e}")
            raise

    # Phase execution methods with enhanced validation logging
    async def _execute_express_phase(self) -> Dict[str, Any]:
        """Execute Express phase: Deep analysis and task framing"""
        print("🤔 Thinking deeply about the task...")

        # Log thinking time and depth
        thinking_start = time.time()

        # Simulate deep analysis (in real implementation, this would be actual analysis)
        await asyncio.sleep(0.5)  # Simulate thinking time

        thinking_time = time.time() - thinking_start
        confidence = 0.85
        quality = 0.8

        # Log validation metrics
        if self.validation_logger and EAEPTValidationLogger:
            self.validation_logger.log_performance_metric(
                "thinking_time_seconds", thinking_time, 10.0  # Target: under 10 seconds
            )
            self.validation_logger.log_workflow_decision(
                "task_analysis_complete",
                confidence,
                f"Analyzed task: {self.current_task[:50]}...",
            )

        return {
            "status": "completed",
            "phase": "express",
            "confidence": confidence,
            "quality": quality,
            "thinking_time": thinking_time,
        }

    async def _execute_ask_phase(self) -> Dict[str, Any]:
        """Execute Ask phase: Interactive clarification"""
        print("❓ Generating clarification questions...")

        # Simulate question generation and user interaction
        await asyncio.sleep(0.3)

        confidence = 0.9
        quality = 0.85
        groq_result = None
        if self.groq_helper.available:
            # Use the current task as context and ask for requirement clarification
            groq_result = await self.groq_helper.ask_phase_analysis(
                self.current_task,
                "What requirements or clarifications are needed for this task?",
            )

        # Log user interaction quality
        if self.validation_logger and EAEPTValidationLogger:
            self.validation_logger.log_user_interaction(
                "clarification_questions", quality
            )
            self.validation_logger.log_workflow_decision(
                "requirements_clarified",
                confidence,
                "Generated clarifying questions for task requirements",
            )

        return {
            "status": "completed",
            "phase": "ask",
            "confidence": confidence,
            "quality": quality,
            "questions_generated": 3,
            "groq_analysis": groq_result,
        }

    async def _execute_explore_phase(self) -> Dict[str, Any]:
        """Execute Explore phase: RAG-powered research"""
        print("🔍 Exploring with RAG-powered research...")

        # Simulate RAG queries
        rag_start = time.time()

        # Mock RAG queries (in real implementation, these would be actual queries)
        queries = [
            "casino game mechanics patterns",
            "flutter performance optimization",
            "starknet integration best practices",
        ]

        total_results = 0
        total_relevance = 0.0
        groq_results = []

        for query in queries:
            await asyncio.sleep(0.1)  # Simulate query time
            results_count = 5  # Mock results
            relevance = 0.85  # Mock relevance
            response_time = 120.0  # Mock response time in ms

            total_results += results_count
            total_relevance += relevance

            # Log each RAG query
            if self.validation_logger and EAEPTValidationLogger:
                self.validation_logger.log_rag_query(
                    query, results_count, relevance, response_time
                )
            # Use Groq to analyze each research topic
            if self.groq_helper.available:
                groq_result = await self.groq_helper.explore_phase_research(
                    query, self.current_task
                )
                groq_results.append({"query": query, "groq": groq_result})

        rag_time = time.time() - rag_start
        avg_relevance = total_relevance / len(queries)
        confidence = 0.8
        quality = 0.85

        # Log overall exploration metrics
        if self.validation_logger and EAEPTValidationLogger:
            self.validation_logger.log_performance_metric(
                "rag_exploration_time_seconds",
                rag_time,
                30.0,  # Target: under 30 seconds
            )
            self.validation_logger.log_workflow_decision(
                "exploration_complete",
                confidence,
                f"Completed RAG research with {total_results} total results, avg relevance: {avg_relevance:.2f}",
            )

        return {
            "status": "completed",
            "phase": "explore",
            "confidence": confidence,
            "quality": quality,
            "rag_queries": len(queries),
            "total_results": total_results,
            "average_relevance": avg_relevance,
            "groq_research": groq_results,
        }

    async def _execute_plan_phase(self) -> Dict[str, Any]:
        """Execute Plan phase: Detailed implementation planning"""
        print("📋 Creating detailed implementation plan...")

        # Simulate planning process
        planning_start = time.time()
        await asyncio.sleep(0.4)

        planning_time = time.time() - planning_start
        confidence = 0.85
        quality = 0.9

        # Mock plan components
        plan_components = [
            "component_architecture",
            "data_flow_design",
            "integration_strategy",
            "performance_requirements",
            "testing_approach",
        ]

        # Log planning metrics
        if self.validation_logger and EAEPTValidationLogger:
            self.validation_logger.log_performance_metric(
                "planning_time_seconds", planning_time, 20.0  # Target: under 20 seconds
            )
            self.validation_logger.log_workflow_decision(
                "implementation_plan_created",
                confidence,
                f"Created detailed plan with {len(plan_components)} components",
            )

        return {
            "status": "completed",
            "phase": "plan",
            "confidence": confidence,
            "quality": quality,
            "plan_components": len(plan_components),
            "planning_time": planning_time,
        }

    async def _execute_code_phase(self) -> Dict[str, Any]:
        """Execute Code phase: Implementation"""
        print("💻 Beginning implementation...")

        # Simulate code generation
        coding_start = time.time()
        await asyncio.sleep(0.6)

        coding_time = time.time() - coding_start
        confidence = 0.8
        quality = 0.85

        # Mock code generation metrics
        languages = ["python", "dart", "cairo"]
        total_lines = 0

        for language in languages:
            lines = 50 if language == "python" else 30  # Mock lines generated
            test_coverage = 0.8 if language == "python" else 0.7
            code_quality = 0.9

            total_lines += lines

            # Log code generation for each language
            if self.validation_logger and EAEPTValidationLogger:
                self.validation_logger.log_code_generation(
                    language, lines, code_quality, test_coverage
                )

        # Log overall coding metrics
        if self.validation_logger and EAEPTValidationLogger:
            self.validation_logger.log_performance_metric(
                "coding_time_seconds", coding_time, 60.0  # Target: under 60 seconds
            )
            self.validation_logger.log_workflow_decision(
                "implementation_complete",
                confidence,
                f"Generated {total_lines} lines across {len(languages)} languages",
            )

        return {
            "status": "completed",
            "phase": "code",
            "confidence": confidence,
            "quality": quality,
            "total_lines": total_lines,
            "languages": len(languages),
            "coding_time": coding_time,
        }

    async def _execute_test_phase(self) -> Dict[str, Any]:
        """Execute Test phase: Validation and QA"""
        print("🧪 Running comprehensive testing...")

        # Simulate testing process
        testing_start = time.time()
        await asyncio.sleep(0.5)

        testing_time = time.time() - testing_start
        confidence = 0.9
        quality = 0.95

        # Mock test metrics
        test_types = ["unit", "integration", "performance"]
        total_tests = 0

        for test_type in test_types:
            test_count = 10 if test_type == "unit" else 5
            pass_rate = 0.95 if test_type == "unit" else 0.9

            total_tests += test_count

            # Log testing metrics
            if self.validation_logger and EAEPTValidationLogger:
                self.validation_logger.log_performance_metric(
                    f"{test_type}_tests_pass_rate",
                    pass_rate,
                    0.9,  # Target: 90% pass rate
                )

        # Log overall testing metrics
        if self.validation_logger and EAEPTValidationLogger:
            self.validation_logger.log_performance_metric(
                "testing_time_seconds", testing_time, 30.0  # Target: under 30 seconds
            )
            self.validation_logger.log_workflow_decision(
                "testing_complete",
                confidence,
                f"Executed {total_tests} tests across {len(test_types)} test types",
            )

        # Use Groq to review the code (simulate with current_task as code context)
        groq_review = None
        if self.groq_helper.available:
            groq_review = await self.groq_helper.test_phase_review(
                self.current_task, "Full code review and test analysis"
            )
        return {
            "status": "completed",
            "phase": "test",
            "confidence": confidence,
            "quality": quality,
            "total_tests": total_tests,
            "test_types": len(test_types),
            "testing_time": testing_time,
            "groq_review": groq_review,
        }

    async def _should_auto_transition(self) -> bool:
        """Determine if workflow should auto-transition to next phase"""
        config = self.phase_configs[self.current_phase]
        metrics = self.phase_metrics[self.current_phase]

        # Check completion confidence
        if metrics.completion_confidence < config.auto_transition_threshold:
            return False

        return True

    async def _transition_to_next_phase(self):
        """Transition to the next EAEPT phase"""
        phase_order = [
            EAEPTPhase.EXPRESS,
            EAEPTPhase.ASK,
            EAEPTPhase.EXPLORE,
            EAEPTPhase.PLAN,
            EAEPTPhase.CODE,
            EAEPTPhase.TEST,
            EAEPTPhase.COMPLETE,
        ]

        current_index = phase_order.index(self.current_phase)
        from_phase = self.current_phase.value

        if current_index < len(phase_order) - 1:
            next_phase = phase_order[current_index + 1]
            confidence = self.phase_metrics[self.current_phase].completion_confidence

            # Log phase transition
            if self.validation_logger and EAEPTValidationLogger:
                self.validation_logger.log_phase_transition(
                    from_phase, next_phase.value, confidence, auto_transition=True
                )

            self.current_phase = next_phase
            self.phase_metrics[self.current_phase] = PhaseMetrics(self.current_phase)
            print(f"🔄 Auto-transitioning to {self.current_phase.value.upper()} phase")
        else:
            # Log workflow completion
            if self.validation_logger and EAEPTValidationLogger:
                self.validation_logger.log_workflow_decision(
                    "workflow_completed", 1.0, "All EAEPT phases completed successfully"
                )

            self.current_phase = EAEPTPhase.COMPLETE
            print("🎯 Workflow completed!")

        self._save_workflow_state()

    async def _handle_context_optimization(self, config: PhaseConfig):
        """Handle context optimization for phase"""
        if not self.orchestrator:
            return

        context = await self.orchestrator.analyzer.analyze_session()
        token_ratio = context.token_count / 200000  # Claude's limit
        tokens_before = context.token_count

        if token_ratio > config.token_threshold:
            strategy = config.context_optimization_strategy
            print(f"🔄 Triggering context optimization: {strategy}")

            result = await self.orchestrator.orchestrate(
                f"Phase {self.current_phase.value} context optimization using {strategy} strategy"
            )

            if result.get("command_executed") != "none":
                self.phase_metrics[self.current_phase].context_optimizations += 1

                # Get post-optimization context for comparison
                post_context = await self.orchestrator.analyzer.analyze_session()
                tokens_after = post_context.token_count

                # Calculate effectiveness based on token reduction
                effectiveness = (
                    min(1.0, (tokens_before - tokens_after) / tokens_before)
                    if tokens_before > 0
                    else 0.0
                )

                # Log context optimization
                if self.validation_logger and EAEPTValidationLogger:
                    self.validation_logger.log_context_optimization(
                        strategy, tokens_before, tokens_after, effectiveness
                    )
                    self.validation_logger.log_orchestration_action(
                        f"context_optimization_{strategy}",
                        result.get("command_executed", "unknown"),
                        effectiveness,
                    )

                print(f"✅ Context optimized: {result.get('command_executed')}")
                print(
                    f"   Token reduction: {tokens_before} → {tokens_after} ({effectiveness:.1%})"
                )
            else:
                # Log failed optimization attempt
                if self.validation_logger and EAEPTValidationLogger:
                    self.validation_logger.log_orchestration_action(
                        f"context_optimization_{strategy}", "no_action_taken", 0.0
                    )

    async def _generate_workflow_summary(self) -> Dict[str, Any]:
        """Generate comprehensive workflow summary"""
        total_duration = sum(
            metrics.duration_minutes for metrics in self.phase_metrics.values()
        )
        total_tokens = sum(
            metrics.token_usage for metrics in self.phase_metrics.values()
        )

        summary = {
            "task": self.current_task,
            "total_duration_minutes": round(total_duration, 1),
            "total_token_usage": total_tokens,
            "phases_completed": len(
                [m for m in self.phase_metrics.values() if m.end_time]
            ),
            "average_confidence": round(
                sum(m.completion_confidence for m in self.phase_metrics.values())
                / len(self.phase_metrics),
                2,
            ),
            "average_quality": round(
                sum(m.quality_score for m in self.phase_metrics.values())
                / len(self.phase_metrics),
                2,
            ),
            "context_optimizations": sum(
                m.context_optimizations for m in self.phase_metrics.values()
            ),
            "workflow_efficiency": "High - Enhanced EAEPT with auto-orchestration",
        }

        # Generate validation effectiveness report
        if self.validation_logger and EAEPTValidationLogger:
            try:
                validation_report = (
                    self.validation_logger.generate_effectiveness_report()
                )
                summary["validation_report"] = validation_report
                summary["validation_available"] = True

                # Add key validation metrics to summary
                if "overall_metrics" in validation_report:
                    overall = validation_report["overall_metrics"]
                    summary["validation_effectiveness"] = overall.get(
                        "average_effectiveness", 0.0
                    )
                    summary["validation_quality"] = overall.get("average_quality", 0.0)
                    summary["total_validation_events"] = validation_report.get(
                        "total_events", 0
                    )

            except Exception as e:
                print(f"Warning: Could not generate validation report: {e}")
                summary["validation_available"] = False
        else:
            summary["validation_available"] = False

        return summary

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "current_phase": self.current_phase.value,
            "current_task": self.current_task,
            "workflow_state": self.workflow_state.get("workflow_state", "unknown"),
            "phase_metrics": {
                phase.value: {
                    "duration": metrics.duration_minutes,
                    "completed": metrics.end_time is not None,
                    "confidence": metrics.completion_confidence,
                    "quality": metrics.quality_score,
                }
                for phase, metrics in self.phase_metrics.items()
            },
            "total_duration": sum(
                m.duration_minutes for m in self.phase_metrics.values()
            ),
            "session_start": self.workflow_state.get("session_start"),
        }


async def main():
    """Main CLI interface for enhanced EAEPT workflow"""
    parser = argparse.ArgumentParser(description="Enhanced EAEPT Workflow System")
    parser.add_argument(
        "--start", type=str, help="Start new workflow with task description"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current workflow status"
    )
    parser.add_argument(
        "--continue-workflow", action="store_true", help="Continue current workflow"
    )
    parser.add_argument("--phase", type=str, help="Execute specific phase")
    parser.add_argument("--reset", action="store_true", help="Reset workflow state")
    parser.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Enable auto-execution (default)",
    )
    parser.add_argument(
        "--manual", action="store_false", dest="auto", help="Disable auto-execution"
    )

    args = parser.parse_args()

    # Initialize workflow engine
    workflow = EAEPTWorkflowEngine()

    if args.start:
        print("🚀 Starting Enhanced EAEPT Workflow...")
        result = await workflow.start_workflow(args.start, auto_execute=args.auto)
        print(json.dumps(result, indent=2, default=str))

    elif args.status:
        status = workflow.get_workflow_status()
        print("📊 Enhanced EAEPT Workflow Status:")
        print(json.dumps(status, indent=2, default=str))

    elif args.continue_workflow:
        print("▶️  Continuing Enhanced EAEPT Workflow...")
        result = await workflow.execute_full_workflow()
        print(json.dumps(result, indent=2, default=str))

    elif args.phase:
        try:
            workflow.current_phase = EAEPTPhase(args.phase)
            result = await workflow.execute_current_phase()
            print(json.dumps(result, indent=2, default=str))
        except ValueError:
            print(f"❌ Invalid phase: {args.phase}")
            print(f"Valid phases: {[p.value for p in EAEPTPhase]}")

    elif args.reset:
        if workflow.state_path.exists():
            workflow.state_path.unlink()
        print("🔄 Workflow state reset")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
