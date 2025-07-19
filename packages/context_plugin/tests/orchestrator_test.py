#!/usr/bin/env python3
"""
Test Suite for Claude Orchestration Engine
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
from datetime import datetime

from orchestrator_engine import (
    OrchestrationEngine,
    ContextAnalyzer,
    DecisionEngine,
    CommandDispatcher,
    WorkflowConfig,
    ContextMetrics,
    CommandType,
)


class TestContextAnalyzer(unittest.TestCase):
    """Test suite for ContextAnalyzer"""

    def setUp(self):
        self.config = WorkflowConfig()
        self.analyzer = ContextAnalyzer(self.config)

    def test_analyze_session_basic(self):
        """Test basic session analysis"""
        metrics = self.analyzer.analyze_session()

        self.assertIsInstance(metrics, ContextMetrics)
        self.assertGreaterEqual(metrics.session_duration, 0)
        self.assertGreaterEqual(metrics.token_count, 0)
        self.assertGreaterEqual(metrics.message_count, 0)

    def test_complexity_calculation(self):
        """Test complexity score calculation"""
        metrics = ContextMetrics(
            message_count=25,
            session_duration=30,
            code_blocks=5,
            repetitive_patterns=2,
            topic_changes=1,
        )

        complexity = metrics.calculate_complexity()
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)


class TestDecisionEngine(unittest.TestCase):
    """Test suite for DecisionEngine"""

    def setUp(self):
        self.config = WorkflowConfig()
        self.analyzer = ContextAnalyzer(self.config)
        self.decision_engine = DecisionEngine(self.config, self.analyzer)

    def test_digest_score_calculation(self):
        """Test digest score calculation"""
        metrics = ContextMetrics(
            session_duration=45,
            task_completion_count=3,
            code_blocks=6,
            complexity_score=0.8,
        )

        score = self.decision_engine._calculate_digest_score(
            metrics, "document progress"
        )
        self.assertGreater(score, 0.5)

    def test_compact_score_calculation(self):
        """Test compact score calculation"""
        metrics = ContextMetrics(
            token_count=7000, repetitive_patterns=4, message_count=30, topic_changes=3
        )

        score = self.decision_engine._calculate_compact_score(
            metrics, "summarize conversation"
        )
        self.assertGreater(score, 0.5)

    def test_command_decision_logic(self):
        """Test command decision logic"""
        # Test digest preference
        with patch.object(
            self.decision_engine, "_calculate_digest_score", return_value=0.7
        ):
            with patch.object(
                self.decision_engine, "_calculate_compact_score", return_value=0.4
            ):
                command_type, params = self.decision_engine.decide_command(
                    "document session"
                )
                self.assertEqual(command_type, CommandType.DIGEST)

        # Test compact preference
        with patch.object(
            self.decision_engine, "_calculate_digest_score", return_value=0.4
        ):
            with patch.object(
                self.decision_engine, "_calculate_compact_score", return_value=0.7
            ):
                command_type, params = self.decision_engine.decide_command(
                    "compress conversation"
                )
                self.assertEqual(command_type, CommandType.COMPACT)


class TestCommandDispatcher(unittest.TestCase):
    """Test suite for CommandDispatcher"""

    def setUp(self):
        self.config = WorkflowConfig()
        self.dispatcher = CommandDispatcher(self.config)

    @patch("subprocess.run")
    def test_execute_digest_command(self, mock_run):
        """Test digest command execution"""
        mock_run.return_value = Mock(returncode=0, stdout="Success", stderr="")

        params = {"action": "create"}
        result = self.dispatcher._execute_digest(params)

        self.assertEqual(result["status"], "success")
        self.assertIn("output", result)

    def test_execute_compact_command(self):
        """Test compact command execution"""
        params = {"level": "standard", "preserve_code": True}
        result = self.dispatcher._execute_compact(params)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["level"], "standard")
        self.assertTrue(result["preserve_code"])

    def test_execute_hybrid_command(self):
        """Test hybrid command execution"""
        params = {
            "sequence": ["compact", "digest"],
            "compact_params": {"level": "standard"},
            "digest_params": {"action": "create"},
        }

        with patch.object(
            self.dispatcher, "_execute_compact", return_value={"status": "success"}
        ):
            with patch.object(
                self.dispatcher, "_execute_digest", return_value={"status": "success"}
            ):
                result = self.dispatcher._execute_hybrid(params)

                self.assertEqual(result["status"], "success")
                self.assertEqual(result["action"], "hybrid")
                self.assertEqual(len(result["results"]), 2)


class TestOrchestrationEngine(unittest.TestCase):
    """Test suite for OrchestrationEngine"""

    def setUp(self):
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

        self.engine = OrchestrationEngine(self.config_path)

    def tearDown(self):
        # Clean up temporary files
        if self.config_path.exists():
            self.config_path.unlink()
        os.rmdir(self.temp_dir)

    def test_orchestrate_with_intent(self):
        """Test orchestration with user intent"""
        with patch.object(self.engine.dispatcher, "execute_command") as mock_execute:
            mock_execute.return_value = {"status": "success", "action": "digest"}

            result = self.engine.orchestrate("I need to document my progress")

            self.assertEqual(result["status"], "success")
            self.assertIn("metrics", result)
            self.assertIn("command_executed", result)

    def test_orchestrate_with_force(self):
        """Test orchestration with forced command"""
        with patch.object(self.engine.dispatcher, "execute_command") as mock_execute:
            mock_execute.return_value = {"status": "success", "action": "compact"}

            result = self.engine.orchestrate("", force_command="compact")

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["command_executed"], "compact")

    def test_get_status(self):
        """Test status retrieval"""
        status = self.engine.get_status()

        self.assertIn("session_duration", status)
        self.assertIn("token_count", status)
        self.assertIn("config", status)
        self.assertIn("recommendations", status)

    def test_session_state_management(self):
        """Test session state management"""
        # Test initial state
        initial_state = self.engine.get_session_state()
        self.assertIsNone(initial_state["last_command"])
        self.assertEqual(initial_state["command_count"], 0)

        # Test state update
        self.engine._update_session_state(CommandType.DIGEST, {"status": "success"})
        updated_state = self.engine.get_session_state()
        self.assertEqual(updated_state["last_command"], "digest")
        self.assertEqual(updated_state["command_count"], 1)

        # Test state reset
        self.engine.reset_session()
        reset_state = self.engine.get_session_state()
        self.assertIsNone(reset_state["last_command"])
        self.assertEqual(reset_state["command_count"], 0)


class TestWorkflowConfig(unittest.TestCase):
    """Test suite for WorkflowConfig"""

    def test_config_creation(self):
        """Test configuration creation"""
        config = WorkflowConfig()

        self.assertEqual(config.automation_level, "high")
        self.assertEqual(config.digest_threshold, 3)
        self.assertTrue(config.auto_trigger)

    def test_config_from_file(self):
        """Test configuration loading from file"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
orchestration_config:
  automation_level: "medium"
  digest_threshold: 5
  auto_trigger: false
"""
            )
            config_path = Path(f.name)

        try:
            config = WorkflowConfig.from_file(config_path)

            self.assertEqual(config.automation_level, "medium")
            self.assertEqual(config.digest_threshold, 5)
            self.assertFalse(config.auto_trigger)
        finally:
            config_path.unlink()


class TestIntegration(unittest.TestCase):
    """Integration tests for the orchestration system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        self.engine = OrchestrationEngine(self.config_path)

    def tearDown(self):
        if self.config_path.exists():
            self.config_path.unlink()
        os.rmdir(self.temp_dir)

    def test_full_orchestration_flow(self):
        """Test complete orchestration flow"""
        with patch.object(self.engine.dispatcher, "_run_command") as mock_run:
            mock_run.return_value = {"status": "success", "output": "Command executed"}

            # Test with digest intent
            result = self.engine.orchestrate("I need to document my progress")
            self.assertEqual(result["status"], "success")

            # Test with compact intent
            result = self.engine.orchestrate("The conversation is getting too long")
            self.assertEqual(result["status"], "success")

    def test_rag_integration(self):
        """Test RAG system integration"""
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200)

            # Test RAG availability check
            decision_engine = DecisionEngine(self.engine.config, self.engine.analyzer)
            self.assertTrue(decision_engine._check_rag_availability())

    def test_error_handling(self):
        """Test error handling in orchestration"""
        with patch.object(self.engine.dispatcher, "execute_command") as mock_execute:
            mock_execute.side_effect = Exception("Test error")

            result = self.engine.orchestrate("Test error scenario")

            self.assertEqual(result["status"], "error")
            self.assertIn("error", result)


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
