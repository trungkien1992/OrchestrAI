"""
Command-line interface for Claude Code Context Plugin
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional
import os
import click
from ai_providers.provider_factory import get_provider_from_env, get_provider

from ..core.monitor import ContextMonitor
from ..core.orchestrator import PluginOrchestrator
from ..core.plugin_manager import PluginManager
from ..config import load_config
import tempfile
import subprocess


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        prog="claude-code-plugin",
        description="Claude Code Context Plugin - Smart context management for AI development",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start context monitoring")
    monitor_parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    monitor_parser.add_argument(
        "--interval", type=int, default=30, help="Monitoring interval (seconds)"
    )

    # Orchestrate command
    orchestrate_parser = subparsers.add_parser(
        "orchestrate", help="Manual orchestration"
    )
    orchestrate_parser.add_argument(
        "--intent", required=True, help="Session intent description"
    )
    orchestrate_parser.add_argument(
        "--dry-run", action="store_true", help="Show recommendation without executing"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show current context status")
    status_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed information"
    )

    # History command
    history_parser = subparsers.add_parser("history", help="Show session history")
    history_parser.add_argument(
        "--limit", type=int, default=10, help="Number of sessions to show"
    )

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize plugin configuration")
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing configuration"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage plugin configuration")
    config_parser.add_argument(
        "--show", action="store_true", help="Show current configuration"
    )
    config_parser.add_argument(
        "--edit", action="store_true", help="Edit configuration file"
    )

    return parser


async def monitor_command(args):
    """Start context monitoring"""
    try:
        config = load_config()
        monitor = ContextMonitor(config)

        print("🔄 Starting Claude Code Context Monitor...")
        print(f"📊 Monitoring interval: {args.interval} seconds")
        print(f"🎯 Token limit: {config.get('token_limit', 200000)}")
        print("Press Ctrl+C to stop monitoring")

        if args.daemon:
            print("🔧 Running as daemon...")
            # In a real implementation, this would fork to background

        await monitor.start_monitoring(interval=args.interval)

    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        return 1


async def orchestrate_command(args):
    """Handle manual orchestration"""
    try:
        config = load_config()
        orchestrator = PluginOrchestrator(config)

        print(f"🎯 Analyzing session intent: {args.intent}")

        # Get session data (in real implementation, this would collect from Claude Code)
        session_data = {
            "intent": args.intent,
            "timestamp": "now",
            "context_size": "estimated",
        }

        decision = await orchestrator.make_decision(session_data)

        print(f"🤖 Decision: {decision.get('action', 'No action needed')}")
        print(f"💡 Reasoning: {decision.get('reasoning', 'Standard analysis')}")
        print(f"⏱️  Estimated time: {decision.get('estimated_time', 'Unknown')}")

        if args.dry_run:
            print("🧪 Dry run mode - no commands executed")
        else:
            print("✅ Recommendation ready for execution")

    except Exception as e:
        print(f"❌ Error during orchestration: {e}")
        return 1


async def status_command(args):
    """Show current context status"""
    try:
        config = load_config()
        monitor = ContextMonitor(config)

        status = await monitor.get_status()

        print("📊 Claude Code Context Status")
        print("=" * 40)
        print(f"🔋 Token Usage: {status.get('token_usage', 'Unknown')}")
        print(f"📈 Burn Rate: {status.get('burn_rate', 'Unknown')}")
        print(f"⏰ Time to Limit: {status.get('time_to_limit', 'Unknown')}")
        print(f"🎯 Session Type: {status.get('session_type', 'Unknown')}")
        print(f"📂 Context Files: {status.get('context_files', 'Unknown')}")

        if args.verbose:
            print("\n🔍 Detailed Information:")
            print(f"📅 Session Start: {status.get('session_start', 'Unknown')}")
            print(f"🔄 Last Update: {status.get('last_update', 'Unknown')}")
            print(f"💾 Sessions Saved: {status.get('sessions_saved', 'Unknown')}")
            print(f"🎯 Confidence: {status.get('confidence', 'Unknown')}")

    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return 1


async def history_command(args):
    """Show session history"""
    try:
        config = load_config()
        manager = PluginManager(config)

        sessions = await manager.get_session_history(limit=args.limit)

        print(f"📅 Recent Sessions (last {args.limit}):")
        print("=" * 50)

        for i, session in enumerate(sessions, 1):
            print(
                f"{i}. {session.get('name', 'Unknown')} "
                f"({session.get('duration', 'Unknown')} - "
                f"{session.get('date', 'Unknown')})"
            )

        if not sessions:
            print("No sessions found")

    except Exception as e:
        print(f"❌ Error getting history: {e}")
        return 1


def init_command(args):
    """Initialize plugin configuration"""
    try:
        config_path = Path.home() / ".claude" / "claude-code-plugin" / "config.py"

        if config_path.exists() and not args.force:
            print(f"❌ Configuration already exists at {config_path}")
            print("Use --force to overwrite")
            return 1

        # Create config directory
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create default configuration
        default_config = """# Claude Code Context Plugin Configuration

PLUGIN_CONFIG = {
    'token_limit': 200000,
    'burn_rate_threshold': 150,
    'auto_compact_threshold': 0.85,
    'checkpoint_interval': 15,  # minutes
    'monitoring_enabled': True,
    'smart_suggestions': True
}

WORKFLOW_CONFIG = {
    'coding_session_detection': True,
    'debugging_mode_optimization': True,
    'architecture_session_persistence': True,
    'learning_from_patterns': True,
    'proactive_recommendations': True
}
"""

        with open(config_path, "w") as f:
            f.write(default_config)

        print(f"✅ Configuration initialized at {config_path}")
        print("🔧 Edit the configuration file to customize settings")

    except Exception as e:
        print(f"❌ Error initializing configuration: {e}")
        return 1


def config_command(args):
    """Manage plugin configuration"""
    try:
        config_path = Path.home() / ".claude" / "claude-code-plugin" / "config.py"

        if args.show:
            if config_path.exists():
                print(f"📄 Configuration file: {config_path}")
                print("=" * 50)
                with open(config_path, "r") as f:
                    print(f.read())
            else:
                print("❌ Configuration file not found")
                print("Run 'claude-code-plugin init' to create one")
                return 1

        elif args.edit:
            if config_path.exists():
                import os

                editor = os.environ.get("EDITOR", "nano")
                os.system(f"{editor} {config_path}")
            else:
                print("❌ Configuration file not found")
                print("Run 'claude-code-plugin init' to create one")
                return 1
        else:
            print("❌ Use --show or --edit with config command")
            return 1

    except Exception as e:
        print(f"❌ Error managing configuration: {e}")
        return 1


@click.group()
def cli():
    """AI Orchestrator CLI"""
    pass


@cli.command()
@click.option(
    "--provider", prompt="AI Provider (openai/anthropic/groq)", help="AI provider name"
)
@click.option(
    "--api-key", prompt="API Key", hide_input=True, help="API key for the provider"
)
def set_key(provider, api_key):
    """Set the AI provider and API key (saved to environment variables)."""
    os.environ["AI_PROVIDER"] = provider
    os.environ["AI_API_KEY"] = api_key
    click.echo(f"Set provider to {provider} and API key.")


@cli.command()
@click.option("--prompt", prompt="Prompt", help="Prompt for completion")
def complete(prompt):
    """Run a test completion with the configured provider."""
    import asyncio

    provider = get_provider_from_env()
    result = asyncio.run(provider.complete(prompt))
    click.echo(f"Completion result:\n{result}")


@cli.command()
@click.option(
    "--task", prompt="Coding Task", help="Describe the coding task to orchestrate"
)
@click.option("--max-iterations", default=3, help="Maximum feedback/fix iterations")
def run_workflow(task, max_iterations):
    """Run an AI-powered orchestration workflow for a coding task, with validation and test feedback loop."""
    import asyncio

    provider = get_provider_from_env()
    click.echo(f"Starting workflow for: {task}")
    # Step 1: Plan
    prompt = (
        f"You are an expert AI coding assistant. {task}\nProvide a step-by-step plan."
    )
    plan = asyncio.run(provider.complete(prompt))
    click.echo(f"\nStep-by-step plan:\n{plan}\n")
    # Step 2: Code generation
    code_prompt = f"{plan}\nNow, write the Python code for this task."
    code = asyncio.run(provider.complete(code_prompt))
    click.echo(f"\nGenerated code (iteration 1):\n{code}\n")
    # Step 3: Linting Feedback Loop
    for iteration in range(1, max_iterations + 1):
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp.flush()
            tmp_path = tmp.name
        result = subprocess.run(["flake8", tmp_path], capture_output=True, text=True)
        lint_output = result.stdout.strip()
        if not lint_output:
            click.echo(f"\n✅ Code passed linting on iteration {iteration}!")
            break
        click.echo(f"\nLinting issues (iteration {iteration}):\n{lint_output}\n")
        fix_prompt = f"The following Python code has these linting issues:\n{lint_output}\nPlease fix the code and return the corrected version only.\nCode:\n{code}"
        code = asyncio.run(provider.complete(fix_prompt))
        click.echo(f"\nAI-provided fix (iteration {iteration+1}):\n{code}\n")
    else:
        click.echo("\n⚠️  Max iterations reached. Code may still have issues.")
    # Step 4: Generate tests
    test_prompt = f"{code}\nNow, write unit tests for this code."
    tests = asyncio.run(provider.complete(test_prompt))
    click.echo(f"\nGenerated tests (iteration 1):\n{tests}\n")
    # Step 5: Test Execution & Feedback Loop
    for test_iter in range(1, max_iterations + 1):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "main_code.py")
            test_path = os.path.join(tmpdir, "test_code.py")
            with open(code_path, "w") as f:
                f.write(code)
            with open(test_path, "w") as f:
                f.write(tests)
            # Patch sys.path for import
            test_code = (
                f"import sys\nsys.path.insert(0, '{tmpdir}')\n" + open(test_path).read()
            )
            with open(test_path, "w") as f:
                f.write(test_code)
            # Run pytest
            result = subprocess.run(
                ["pytest", test_path, "--tb=short", "-q"],
                capture_output=True,
                text=True,
            )
            test_output = result.stdout.strip() + "\n" + result.stderr.strip()
            if "failed" not in test_output.lower():
                click.echo(f"\n✅ All tests passed on iteration {test_iter}!")
                break
            click.echo(f"\nTest failures (iteration {test_iter}):\n{test_output}\n")
            # Ask AI to fix code and/or tests
            fix_prompt = f"The following code and tests have these test failures:\n{test_output}\nPlease fix the code and/or tests so all tests pass.\nCode:\n{code}\n\nTests:\n{tests}"
            fixed = asyncio.run(provider.complete(fix_prompt))
            # Heuristically split code and tests (AI should return both)
            if "class Test" in fixed or "def test_" in fixed:
                # Try to split at first test function/class
                split_idx = fixed.find("def test_")
                if split_idx == -1:
                    split_idx = fixed.find("class Test")
                code, tests = fixed[:split_idx], fixed[split_idx:]
            else:
                code = fixed
            click.echo(f"\nAI-provided fix (iteration {test_iter+1}):\n{fixed}\n")
    else:
        click.echo("\n⚠️  Max iterations reached. Tests may still be failing.")
    click.echo("Workflow complete.")


@cli.command()
@click.option(
    "--project-dir",
    prompt="Project Directory",
    type=click.Path(exists=True, file_okay=False),
    help="Path to the project directory",
)
@click.option(
    "--task",
    prompt="Project Task",
    help="Describe the project-level task to orchestrate",
)
def run_project_workflow(project_dir, task):
    """Run a project-aware AI orchestration workflow: summarize all code, send to AI with the task."""
    import asyncio

    provider = get_provider_from_env()
    click.echo(f"Summarizing project in: {project_dir}")
    # Step 1: Read and summarize all Python files
    file_summaries = []
    for root, _, files in os.walk(project_dir):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                # Summarize each file (first N lines + filename)
                summary = f"File: {os.path.relpath(fpath, project_dir)}\n" + "\n".join(
                    code.splitlines()[:30]
                )
                file_summaries.append(summary)
    project_context = "\n\n".join(file_summaries)
    click.echo(
        f"Read {len(file_summaries)} Python files. Sending summary to AI provider..."
    )
    # Step 2: Send summary and task to AI
    prompt = f"You are an expert AI project orchestrator. Here is the current project codebase (summarized):\n{project_context}\n\nTask: {task}\n\nWhat changes or steps should be taken to accomplish this task? Be specific and reference files as needed."
    plan = asyncio.run(provider.complete(prompt))
    click.echo(f"\nAI Project Plan:\n{plan}\n")


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "monitor":
            return asyncio.run(monitor_command(args))
        elif args.command == "orchestrate":
            return asyncio.run(orchestrate_command(args))
        elif args.command == "status":
            return asyncio.run(status_command(args))
        elif args.command == "history":
            return asyncio.run(history_command(args))
        elif args.command == "init":
            return init_command(args)
        elif args.command == "config":
            return config_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
