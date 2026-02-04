# Contributing to Health Universe A2A SDK

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Health-Universe/healthuniverse-a2a-sdk-python
   cd healthuniverse-a2a-sdk-python
   ```

2. **Install uv** (if not already installed)
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or with pip
   pip install uv
   ```

3. **Install dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```

   Note: uv automatically creates and manages a virtual environment for you.

## Code Quality

### Linting

We use [Ruff](https://github.com/astral-sh/ruff) for linting:

```bash
uv run ruff check src/
```

Fix auto-fixable issues:
```bash
uv run ruff check --fix src/
```

### Formatting

Format code with Ruff:

```bash
uv run ruff format src/
```

### Type Checking

We use [mypy](https://mypy-lang.org/) for type checking:

```bash
uv run mypy src/
```

### Running All Checks

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format --check src/

# Type check
uv run mypy src/

# Test
uv run pytest
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=health_universe_a2a

# Run specific test file
uv run pytest tests/test_base.py

# Run with verbose output
uv run pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Use pytest fixtures for setup/teardown
- Aim for >90% code coverage

Example:
```python
import pytest
from health_universe_a2a import Agent, AgentContext

def test_agent_basic_configuration():
    """Agent should have correct basic configuration."""
    class TestAgent(Agent):
        def get_agent_name(self) -> str:
            return "Test"

        def get_agent_description(self) -> str:
            return "Test agent"

        async def process_message(self, message: str, context: AgentContext) -> str:
            return "done"

    agent = TestAgent()
    assert agent.get_agent_name() == "Test"
    assert agent.supports_push_notifications() is True
```

## Pull Request Process

1. **Fork the repository** and create a branch from `main`

2. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   - Use clear, descriptive commit messages
   - Reference issue numbers if applicable
   ```
   feat: Add validation hook to base agent

   - Add validate_message() method to Agent
   - Update Agent to call validation before processing
   - Add example showing validation usage

   Closes #123
   ```

4. **Run all checks locally**
   ```bash
   uv run ruff check src/
   uv run ruff format src/
   uv run mypy src/
   uv run pytest
   ```

5. **Push to your fork** and submit a pull request

6. **PR Review Process**
   - Maintainers will review your PR
   - Address any feedback or requested changes
   - Once approved, your PR will be merged

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Examples:
```
feat: Add document operations to Agent
fix: Handle cancellation in Agent properly
docs: Update README with Agent examples
test: Add tests for validation hook
refactor: Extract context building to separate method
chore: Update dependencies
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for all functions
- Maximum line length: 100 characters
- Use docstrings for classes and public methods

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    More detailed description if needed. Can span multiple lines
    and include examples.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided

    Example:
        >>> example_function("hello", 42)
        True
    """
    pass
```

### Error Handling

- Use specific exception types
- Include helpful error messages
- Don't catch exceptions silently

```python
# Good
try:
    result = process_data(data)
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    raise

# Bad
try:
    result = process_data(data)
except:
    pass
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new classes and methods
- Include examples for new features
- Update CHANGELOG.md

## Release Process

(For maintainers)

1. Update version in `pyproject.toml` and `src/health_universe_a2a/__init__.py`
2. Update CHANGELOG.md
3. Create a git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. Build and publish:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## Questions?

- Open an issue for bugs or feature requests
- Join our [Discord](https://discord.gg/healthuniverse) for questions
- Email [support@healthuniverse.com](mailto:support@healthuniverse.com)

Thank you for contributing!
