# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Installing from GitHub Packages

The SDK is distributed via GitHub Packages. You'll need a GitHub Personal Access Token with `read:packages` scope.

### Step 1: Create a Personal Access Token

1. Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Generate a new token (classic) with `read:packages` scope
3. Copy and save the token

### Step 2: Configure Authentication

**Option A: Environment Variable**

```bash
export GITHUB_TOKEN="your_token_here"
```

**Option B: pip Configuration**

Create `~/.pip/pip.conf`:

```ini
[global]
extra-index-url = https://__token__:YOUR_TOKEN@ghcr.io/Health-Universe/healthuniverse-a2a-sdk-python/simple/
```

### Step 3: Install

**Using pip:**

```bash
pip install health-universe-a2a
```

**Using uv:**

```bash
uv add health-universe-a2a
```

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
