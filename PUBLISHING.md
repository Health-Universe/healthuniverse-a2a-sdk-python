# Publishing Guide

This document describes how to publish and consume the Health Universe A2A SDK.

## Current Distribution: GitHub Packages

The SDK is currently distributed via GitHub Packages as a private package.

### Installing from GitHub Packages

#### 1. Create a Personal Access Token (PAT)

1. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Generate a new token with `read:packages` scope
3. Copy the token value

#### 2. Configure pip

Create or edit `~/.pip/pip.conf` (Linux/macOS) or `%APPDATA%\pip\pip.ini` (Windows):

```ini
[global]
extra-index-url = https://__token__:YOUR_PAT_HERE@ghcr.io/Health-Universe/healthuniverse-a2a-sdk-python/simple/
```

Or use environment variables:

```bash
export PIP_EXTRA_INDEX_URL="https://__token__:${GITHUB_TOKEN}@ghcr.io/Health-Universe/healthuniverse-a2a-sdk-python/simple/"
```

#### 3. Install the package

```bash
pip install health-universe-a2a
```

Or with uv:

```bash
uv add health-universe-a2a --extra-index-url "https://__token__:${GITHUB_TOKEN}@ghcr.io/Health-Universe/healthuniverse-a2a-sdk-python/simple/"
```

### In pyproject.toml

For projects using the SDK, add to your `pyproject.toml`:

```toml
[tool.uv]
extra-index-url = ["https://ghcr.io/Health-Universe/healthuniverse-a2a-sdk-python/simple/"]

# Or for pip-based projects in requirements.txt:
# --extra-index-url https://ghcr.io/Health-Universe/healthuniverse-a2a-sdk-python/simple/
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
   - Publish to GitHub Packages

### Manual Release

If needed, you can trigger the publish workflow manually:

1. Go to Actions > "Publish to GitHub Packages"
2. Click "Run workflow"
3. Optionally specify a version override
4. Click "Run workflow"

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

When the SDK is ready for public release:

1. Create a PyPI account and API token
2. Add `PYPI_API_TOKEN` to GitHub repository secrets
3. Uncomment the `publish-pypi` job in `.github/workflows/publish.yml`
4. Create a non-prerelease GitHub Release

The workflow will then publish to both GitHub Packages and public PyPI.

## Documentation

Documentation is automatically generated and deployed to GitHub Pages on every push to `main`.

View the docs at: https://health-universe.github.io/healthuniverse-a2a-sdk-python/

To build docs locally:

```bash
uv sync --extra docs
uv run mkdocs serve
```

## Troubleshooting

### Authentication Errors

If you get 401/403 errors when installing:

1. Verify your PAT has `read:packages` scope
2. Check the PAT hasn't expired
3. Ensure the token is correctly set in pip config or environment

### Version Conflicts

If the published version already exists, the upload will fail. You must:

1. Increment the version number
2. Create a new release

GitHub Packages doesn't allow overwriting existing versions.

### Build Failures

If the CI build fails:

1. Check the Actions tab for detailed logs
2. Ensure all tests pass locally: `uv run pytest`
3. Ensure linting passes: `uv run ruff check src/`
4. Ensure type checking passes: `uv run mypy src/`
