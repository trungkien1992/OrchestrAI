"""
Cursor Agent Groq API Configuration
Setup and configuration for integrating Groq's fast inference with Cursor Agent
"""

import os
import json
from pathlib import Path


class CursorGroqConfig:
    """
    Configuration manager for Cursor Agent's Groq API integration.
    """

    def __init__(self):
        self.config_dir = Path.home() / ".cursor" / "groq"
        self.config_file = self.config_dir / "config.json"
        self.env_file = self.config_dir / ".env"

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def setup_environment(self, api_key: str = None):
        """
        Set up environment variables for Groq API.

        Args:
            api_key: Groq API key (if not provided, will prompt for it)
        """
        if not api_key:
            api_key = input("Enter your Groq API key: ").strip()

        if not api_key:
            print("❌ No API key provided. Setup cancelled.")
            return False

        # Set environment variable
        os.environ["GROQ_API_KEY"] = api_key

        # Save to .env file
        env_content = f"GROQ_API_KEY={api_key}\n"
        with open(self.env_file, "w") as f:
            f.write(env_content)

        # Save to config file
        config = {
            "api_key_configured": True,
            "setup_date": str(Path().cwd()),
            "config_version": "1.0",
        }

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        print("✅ Groq API key configured successfully!")
        print(f"   Config saved to: {self.config_file}")
        print(f"   Environment file: {self.env_file}")

        return True

    def load_environment(self):
        """Load environment variables from .env file."""
        if self.env_file.exists():
            with open(self.env_file, "r") as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
            return True
        return False

    def get_api_key(self) -> str:
        """Get the configured Groq API key."""
        # Try environment variable first
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            return api_key

        # Try loading from .env file
        if self.load_environment():
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                return api_key

        # No default key
        return ""

    def is_configured(self) -> bool:
        """Check if Groq API is properly configured."""
        api_key = self.get_api_key()
        return bool(api_key)

    def get_config_info(self) -> dict:
        """Get configuration information."""
        return {
            "configured": self.is_configured(),
            "config_file": str(self.config_file),
            "env_file": str(self.env_file),
            "api_key_set": bool(self.get_api_key()),
        }


def setup_cursor_groq_integration():
    """
    Interactive setup for Cursor Agent Groq integration.
    """
    print("🚀 Cursor Agent Groq API Integration Setup")
    print("=" * 50)

    config = CursorGroqConfig()

    if config.is_configured():
        print("✅ Groq API is already configured!")
        info = config.get_config_info()
        print(f"   Config file: {info['config_file']}")
        print(f"   Environment file: {info['env_file']}")

        choice = input("\nWould you like to reconfigure? (y/N): ").strip().lower()
        if choice != "y":
            return True

    print("\n📋 Setup Instructions:")
    print("1. Get your Groq API key from: https://console.groq.com/")
    print("2. The API key will be stored securely in your local config")
    print("3. Cursor Agent will use this key for fast reasoning tasks")

    print("\n🔑 API Key Setup:")
    success = config.setup_environment()

    if success:
        print("\n🎉 Setup Complete!")
        print("\n📖 Usage Examples:")
        print("   - Code analysis and reasoning")
        print("   - Problem breakdown and solutions")
        print("   - Implementation planning")
        print("   - Debug assistance")
        print("\n💡 The Groq tool is now available to Cursor Agent!")

        return True
    else:
        print("\n❌ Setup failed. Please try again.")
        return False


def test_groq_integration():
    """
    Test the Groq integration to ensure it's working properly.
    """
    print("🧪 Testing Groq API Integration")
    print("=" * 40)

    config = CursorGroqConfig()

    if not config.is_configured():
        print("❌ Groq API not configured. Run setup first.")
        return False

    print("✅ Configuration found")

    # Test the API
    try:
        import asyncio
        from groq_reasoning_tool import check_groq_health

        async def test():
            result = await check_groq_health()
            return result

        result = asyncio.run(test())

        if result.get("status") == "healthy":
            print("✅ Groq API connection successful!")
            print(f"   Response time: {result.get('response_time_seconds', 'N/A')}s")
            print(f"   Model: {result.get('model', 'N/A')}")
            return True
        else:
            print("❌ Groq API connection failed:")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "setup":
            setup_cursor_groq_integration()
        elif command == "test":
            test_groq_integration()
        elif command == "status":
            config = CursorGroqConfig()
            info = config.get_config_info()
            print(json.dumps(info, indent=2))
        else:
            print("Usage: python cursor_groq_config.py [setup|test|status]")
    else:
        # Interactive mode
        print("Cursor Agent Groq Integration")
        print("1. Setup - Configure Groq API key")
        print("2. Test - Test the integration")
        print("3. Status - Check configuration status")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == "1":
            setup_cursor_groq_integration()
        elif choice == "2":
            test_groq_integration()
        elif choice == "3":
            config = CursorGroqConfig()
            info = config.get_config_info()
            print(json.dumps(info, indent=2))
        else:
            print("Invalid choice.")
