# Installation and Setup Guide

## Installing the SDK

Install directly from the public GitHub repository:

```bash
uv pip install git+https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git
```

> **Note:** Using [uv](https://github.com/astral-sh/uv) is recommended for faster dependency management. You can also use `pip install git+...` if preferred.

## Development Installation

### Option 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git
cd healthuniverse-a2a-sdk-python

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "from health_universe_a2a import Agent; print('SDK installed!')"
```

### Option 2: Add to Existing Project

In your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "health-universe-a2a @ git+https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git",
]
```

Then install:

```bash
uv pip install -e .
```

## Verify Installation

```python
import health_universe_a2a
print(health_universe_a2a.__version__)
```

Or run the full verification:

```bash
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
uv run python examples/simple_agent.py
```

## Using uv (Recommended)

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

## Use in Your Code

```python
from health_universe_a2a import Agent, AgentContext

class MyAgent(Agent):
    def get_agent_name(self) -> str:
        return "My Agent"

    def get_agent_description(self) -> str:
        return "Does something useful"

    async def process_message(self, message: str, context: AgentContext) -> str:
        return "Hello from my agent!"

if __name__ == "__main__":
    MyAgent().serve()
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

1. Verify installation works
2. Run all tests
3. Read [QUICKSTART.md](QUICKSTART.md) for usage
4. Check [examples/](examples/) for reference
5. Read [README.md](README.md) for full documentation

## Support

- [Full Documentation](README.md)
- [Issue Tracker](https://github.com/Health-Universe/healthuniverse-a2a-sdk-python/issues)
- [Email Support](mailto:support@healthuniverse.com)
