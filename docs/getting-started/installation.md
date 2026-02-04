# Installation

## Requirements

- Python 3.10 or higher
- uv or pip package manager

## Installing from GitHub

The SDK is distributed via the public GitHub repository:

```bash
uv pip install git+https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git
```

> **Note:** Using [uv](https://github.com/astral-sh/uv) is recommended for faster dependency management. You can also use `pip install git+...` if preferred.

## Development Installation

For contributing to the SDK:

```bash
git clone https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git
cd healthuniverse-a2a-sdk-python
uv sync --extra dev
```

## Verifying Installation

```python
import health_universe_a2a
print(health_universe_a2a.__version__)
```

## Next Steps

- [Quick Start](quickstart.md) - Build your first agent
