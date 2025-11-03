# Installation and Setup Guide

## For End Users (Once Published)

Once the package is published to PyPI:

```bash
uv pip install health-universe-a2a
```

## For Development (Current State)

### Option 1: Local Development in This Repo

```bash
cd healthuniverse-a2a-sdk-python

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "from health_universe_a2a import A2AAgent; print('✅ SDK installed!')"
```

### Option 2: Clone Fresh Repository

```bash
# Clone the repository
git clone https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git
cd healthuniverse-a2a-sdk-python

# Install for development
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Option 3: Install from Git (Before PyPI)

Others can install directly from your git repo:

```bash
uv pip install git+https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git
```

## Verify Installation

```bash
# Run verification script
uv run python verify_package.py

# Run tests
uv run pytest

# Check code quality
uv run ruff check src/
uv run mypy src/
```

## Development Workflow

### 1. Make Changes

Edit files in `src/health_universe_a2a/`

### 2. Run Tests

```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_base.py # Run specific test file
uv run pytest -v                 # Verbose output
uv run pytest --cov              # With coverage report
```

### 3. Check Code Quality

```bash
# Linting
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Formatting
uv run ruff format src/

# Type checking
uv run mypy src/
```

### 4. Test Examples

```bash
# Run example
python examples/simple_agent.py
```

## Publishing to PyPI

### First Time Setup

1. **Create PyPI account**: https://pypi.org/account/register/
2. **Install build tools**:
   ```bash
   uv pip install build twine
   ```

### Build and Publish

```bash
# Update version in pyproject.toml and __init__.py

# Build distribution
python -m build

# Upload to PyPI (test first!)
python -m twine upload --repository testpypi dist/*

# If test looks good, upload to production
python -m twine upload dist/*
```

### Using uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# Much faster than pip!
```

## Integration with Existing Projects

### Add as Dependency

In your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "health-universe-a2a>=0.1.0",
]
```

Or with uv:

```bash
uv pip install health-universe-a2a
```

### Use in Your Code

```python
from health_universe_a2a import StreamingAgent, MessageContext

class MyAgent(StreamingAgent):
    def get_agent_name(self) -> str:
        return "My Agent"

    def get_agent_description(self) -> str:
        return "Does something useful"

    async def process_message(self, message: str, context: MessageContext) -> str:
        return "Hello from my agent!"
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'health_universe_a2a'`:

1. Make sure you're in the right virtual environment
2. Run `uv pip install -e .` (for development)
3. Check `uv pip list | grep health-universe-a2a`

### Type Checking Errors

If mypy complains about missing imports:

```bash
# Install type stubs
uv pip install types-all

# Or ignore specific errors in pyproject.toml
```

### Test Failures

If tests fail:

```bash
# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_base.py::TestA2AAgent::test_agent_initialization

# Show print statements
uv run pytest -s
```

## Next Steps

1. ✅ Verify installation works
2. ✅ Run all tests
3. ✅ Read [QUICKSTART.md](QUICKSTART.md) for usage
4. ✅ Check [examples/](examples/) for reference
5. ✅ Read [README.md](README.md) for full documentation

## Support

- 📚 [Full Documentation](README.md)
- 💬 [Discord Community](https://discord.gg/healthuniverse)
- 🐛 [Issue Tracker](https://github.com/Health-Universe/healthuniverse-a2a-sdk-python/issues)
- 📧 [Email Support](mailto:support@healthuniverse.com)
