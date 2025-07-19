"""
Basic usage example for Claude Code Context Plugin
"""

import asyncio
from claude_code_plugin import PluginManager, ContextMonitor


async def main():
    """Example of basic plugin usage"""

    # Initialize plugin manager
    plugin_manager = PluginManager()

    print("🚀 Starting Claude Code Context Plugin...")

    # Start the plugin
    await plugin_manager.start_plugin()

    # Get initial status
    status = await plugin_manager.get_plugin_status()
    print(f"📊 Plugin Status: {status}")

    # Simulate some activity
    print("\n🔄 Simulating development activity...")

    # Simulate file modifications
    await plugin_manager.handle_claude_code_event(
        "file_modified", {"file_path": "/path/to/main.py", "type": "modified"}
    )

    await plugin_manager.handle_claude_code_event(
        "file_modified", {"file_path": "/path/to/utils.py", "type": "created"}
    )

    # Simulate command execution
    await plugin_manager.handle_claude_code_event(
        "command_executed", {"command": "Edit", "success": True}
    )

    # Simulate token usage update
    await plugin_manager.handle_claude_code_event(
        "token_usage_updated", {"token_count": 45000, "token_limit": 200000}
    )

    # Get updated status
    status = await plugin_manager.get_plugin_status()
    print(f"📊 Updated Status: {status}")

    # Get session history
    history = await plugin_manager.get_session_history()
    print(f"📅 Session History: {len(history)} sessions")

    # Stop the plugin
    print("\n⏹️  Stopping plugin...")
    await plugin_manager.stop_plugin()

    print("✅ Plugin stopped successfully")


if __name__ == "__main__":
    asyncio.run(main())
