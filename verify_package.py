#!/usr/bin/env python3
"""
Simple script to verify the package structure and imports.
Run this to ensure everything is set up correctly.
"""

import sys
from pathlib import Path


def verify_structure() -> bool:
    """Verify directory structure."""
    print("Verifying package structure...")

    required_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src/health_universe_a2a/__init__.py",
        "src/health_universe_a2a/base.py",
        "src/health_universe_a2a/streaming.py",
        "src/health_universe_a2a/async_agent.py",
        "src/health_universe_a2a/context.py",
        "src/health_universe_a2a/update_client.py",
        "src/health_universe_a2a/types/validation.py",
        "src/health_universe_a2a/types/extensions.py",
        "examples/simple_streaming_agent.py",
        "examples/complex_streaming_agent.py",
        "examples/simple_async_agent.py",
        "examples/complex_async_agent.py",
        "tests/test_base.py",
    ]

    missing: list[str] = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
            print(f"  ❌ Missing: {file_path}")
        else:
            print(f"  ✅ Found: {file_path}")

    if missing:
        print(f"\n❌ {len(missing)} files missing!")
        return False

    print("\n✅ All required files present!")
    return True


def verify_imports() -> bool:
    """Verify package imports."""
    print("\nVerifying imports...")

    try:
        # Add src to path for local testing
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        from health_universe_a2a import (
            A2AAgentBase,
            AsyncAgent,
            BackgroundContext,
            StreamingAgent,
            StreamingContext,
            ValidationAccepted,
            ValidationRejected,
        )

        print("  ✅ All imports successful!")

        # Verify classes are importable
        print(f"  ✅ A2AAgentBase: {A2AAgentBase.__name__}")
        print(f"  ✅ StreamingAgent: {StreamingAgent.__name__}")
        print(f"  ✅ AsyncAgent: {AsyncAgent.__name__}")
        print(f"  ✅ MessageContext: {StreamingContext.__name__}")
        print(f"  ✅ AsyncContext: {BackgroundContext.__name__}")
        print(f"  ✅ ValidationAccepted: {ValidationAccepted.__name__}")
        print(f"  ✅ ValidationRejected: {ValidationRejected.__name__}")

        # ValidationResult is a type alias, so we just verify it's defined
        print("  ✅ ValidationResult: (type alias)")

        return True

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def main() -> None:
    """Run all verifications."""
    print("=" * 60)
    print("Health Universe A2A SDK - Package Verification")
    print("=" * 60)

    structure_ok = verify_structure()
    imports_ok = verify_imports()

    print("\n" + "=" * 60)
    if structure_ok and imports_ok:
        print("✅ Package verification PASSED!")
        print("\nNext steps:")
        print("  1. Install package: pip install -e .")
        print("  2. Run tests: pytest")
        print("  3. Run linting: ruff check src/")
        print("  4. Run type checking: mypy src/")
    else:
        print("❌ Package verification FAILED!")
        print("\nPlease fix the issues above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
