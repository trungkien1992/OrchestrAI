# OrchestrAI

**OrchestrAI** is a provider-agnostic AI coding copilot and project orchestrator. It generates, tests, and iteratively improves code and tests across entire projects—using any major AI API.

> AI-powered project orchestration, from code to tests.

---

## Features
- Provider-agnostic: works with OpenAI, Anthropic, Groq, and more
- Multi-file/project awareness: reads and reasons about entire codebases
- Automated code and test generation
- Linting and test feedback loops for production-ready code
- Extensible CLI for advanced orchestration workflows
- Secure: never stores or pushes your API keys

---

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -e ./packages/context_plugin
   ```
2. **Set your provider and API key**
   ```bash
   python -m claude_code_plugin.cli.commands set-key
   # Or with options:
   python -m claude_code_plugin.cli.commands set-key --provider openai --api-key sk-...
   ```
3. **Run a workflow**
   ```bash
   python -m claude_code_plugin.cli.commands run-workflow --task "Build a REST API in Python to manage a todo list"
   ```
4. **Project-wide orchestration**
   ```bash
   python -m claude_code_plugin.cli.commands run-project-workflow --project-dir /path/to/your/project --task "Add JWT authentication to all API endpoints"
   ```

---

## Security
- API keys are loaded from environment variables or local `.env` files (never committed)
- `.gitignore` protects all secrets and local config
- No secrets in code or git history

---

## Contributing
Pull requests and issues are welcome! Please ensure no secrets are included in your contributions.

---

## License
MIT 