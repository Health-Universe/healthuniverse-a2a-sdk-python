# Publishing Guide

This document describes how to publish and consume the Health Universe A2A SDK.

## Current Distribution: Public GitHub Repository

The SDK is distributed via the public GitHub repository.

### Installing from GitHub

```bash
uv pip install git+https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git
```

### In pyproject.toml

For projects using the SDK, add to your `pyproject.toml`:

```toml
[project]
dependencies = [
    "health-universe-a2a @ git+https://github.com/Health-Universe/healthuniverse-a2a-sdk-python.git",
]
```

## Publishing New Versions

### Automated Release (Recommended)

1. Update the version in `src/health_universe_a2a/__init__.py`:
   ```python
   __version__ = "0.2.1"
   ```

2. Commit the version bump:
   ```bash
   git add src/health_universe_a2a/__init__.py
   git commit -m "Bump version to 0.2.1"
   git push
   ```

3. Create a GitHub Release:
   - Go to the repository's Releases page
   - Click "Draft a new release"
   - Create a new tag (e.g., `v0.2.1`)
   - Add release notes
   - Click "Publish release"

4. The GitHub Actions workflow will automatically:
   - Run tests, linting, and type checks
   - Build the package

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes to the public API
- **MINOR** (0.2.0): New features, backwards compatible
- **PATCH** (0.1.1): Bug fixes, backwards compatible

### Pre-release Versions

For alpha/beta releases, use suffixes:

```python
__version__ = "0.3.0a1"  # Alpha 1
__version__ = "0.3.0b1"  # Beta 1
__version__ = "0.3.0rc1" # Release candidate 1
```

## Future: Public PyPI

When the SDK is ready for public release on PyPI:

1. Create a PyPI account and API token
2. Add `PYPI_API_TOKEN` to GitHub repository secrets
3. Configure the publish workflow in `.github/workflows/publish.yml`
4. Create a non-prerelease GitHub Release

The workflow will then publish to public PyPI.

## Documentation

Documentation is automatically generated and deployed to GitHub Pages on every push to `main`.

View the docs at: https://health-universe.github.io/healthuniverse-a2a-sdk-python/

To build docs locally:

```bash
uv sync --extra docs
uv run mkdocs serve
```

## Troubleshooting

### Build Failures

If the CI build fails:

1. Check the Actions tab for detailed logs
2. Ensure all tests pass locally: `uv run pytest`
3. Ensure linting passes: `uv run ruff check src/`
4. Ensure type checking passes: `uv run mypy src/`
