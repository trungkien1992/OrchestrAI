#!/usr/bin/env python3
"""
Orchestration Enforcer - Ensures proper orchestration usage
"""

import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta


class OrchestrationEnforcer:
    def __init__(self):
        self.log_file = Path.home() / ".claude" / "orchestration_compliance.log"
        self.last_orchestration_time = None
        self.session_start = datetime.now()

    def log_orchestration_usage(self, command, result):
        """Log orchestration usage for compliance tracking"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "result": result,
            "session_duration": (datetime.now() - self.session_start).total_seconds()
            / 60,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def check_orchestration_compliance(self):
        """Check if orchestration should be used based on time/activity"""
        if not self.last_orchestration_time:
            return True, "No orchestration used yet in this session"

        time_since_last = datetime.now() - self.last_orchestration_time

        if time_since_last > timedelta(minutes=30):
            return (
                True,
                f"Last orchestration was {time_since_last.total_seconds()/60:.1f} minutes ago",
            )

        return False, "Orchestration recently used"

    def enforce_orchestration(self, intent):
        """Enforce orchestration usage"""
        should_use, reason = self.check_orchestration_compliance()

        if should_use:
            print(f"🎯 ORCHESTRATION REQUIRED: {reason}")
            print(f"Running: orchestrator_engine.py --orchestrate --intent '{intent}'")

            try:
                result = subprocess.run(
                    [
                        "python",
                        "/Users/admin/.claude/commands/orchestrator_engine.py",
                        "--orchestrate",
                        "--intent",
                        intent,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                self.last_orchestration_time = datetime.now()
                self.log_orchestration_usage(intent, result.stdout)

                print("✅ Orchestration completed")
                print(result.stdout)

                return True

            except subprocess.CalledProcessError as e:
                print(f"❌ Orchestration failed: {e}")
                return False
        else:
            print(f"⏳ Orchestration not required: {reason}")
            return False

    def get_compliance_report(self):
        """Generate compliance report"""
        if not self.log_file.exists():
            return "No orchestration usage logged"

        with open(self.log_file, "r") as f:
            logs = [json.loads(line) for line in f.readlines()]

        total_sessions = len(logs)
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60

        return f"""
Orchestration Compliance Report:
- Total orchestration uses: {total_sessions}
- Session duration: {session_duration:.1f} minutes
- Last orchestration: {self.last_orchestration_time or 'Never'}
- Compliance rate: {'Good' if total_sessions > 0 else 'Poor'}
"""


def main():
    enforcer = OrchestrationEnforcer()

    import sys

    if len(sys.argv) > 1:
        intent = " ".join(sys.argv[1:])
        enforcer.enforce_orchestration(intent)
    else:
        print("Usage: python orchestration_enforcer.py <intent>")
        print(
            "Example: python orchestration_enforcer.py 'Starting Phase 2 implementation'"
        )


if __name__ == "__main__":
    main()
