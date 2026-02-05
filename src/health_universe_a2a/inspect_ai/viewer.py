"""
Inspect View Subprocess Management.

This module provides utilities for starting and stopping the Inspect AI
viewer as a subprocess, allowing agents to have built-in log visualization.

Based on the pattern from a2a-grant-reviewer.
"""

import atexit
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Global viewer process reference
_viewer_process: subprocess.Popen | None = None


def get_viewer_port() -> int:
    """
    Get the port for Inspect View subprocess.

    Returns:
        Port number from INSPECT_VIEW_PORT env var or default 7575
    """
    return int(os.getenv("INSPECT_VIEW_PORT", "7575"))


def get_viewer_host() -> str:
    """
    Get the host for Inspect View subprocess.

    Returns:
        Host from INSPECT_VIEW_HOST env var or default "127.0.0.1"
    """
    return os.getenv("INSPECT_VIEW_HOST", "127.0.0.1")


def get_log_dir() -> str:
    """
    Get the log directory path.

    Returns:
        Log directory from INSPECT_LOG_DIR env var or default "./inspect_logs"
    """
    return os.getenv("INSPECT_LOG_DIR", "./inspect_logs")


def is_viewer_running() -> bool:
    """
    Check if the viewer subprocess is running.

    Returns:
        True if viewer process exists and is running
    """
    global _viewer_process
    if _viewer_process is None:
        return False
    # Check if process is still running
    return _viewer_process.poll() is None


def start_inspect_view(
    log_dir: str | None = None,
    port: int | None = None,
    host: str | None = None,
) -> bool:
    """
    Launch inspect view as a subprocess.

    Args:
        log_dir: Directory containing .eval files (default: INSPECT_LOG_DIR or ./inspect_logs)
        port: Port to run viewer on (default: INSPECT_VIEW_PORT or 7575)
        host: Host to bind to (default: INSPECT_VIEW_HOST or 127.0.0.1)

    Returns:
        True if viewer started successfully, False otherwise

    Example:
        from health_universe_a2a.inspect_ai import start_inspect_view

        # Start with defaults
        start_inspect_view()

        # Start with custom settings
        start_inspect_view(log_dir="./logs", port=8080)
    """
    global _viewer_process

    # Don't start if already running
    if is_viewer_running():
        logger.warning("Inspect View is already running")
        return True

    viewer_port = port or get_viewer_port()
    viewer_host = host or get_viewer_host()
    viewer_log_dir = log_dir or get_log_dir()

    # Ensure log dir exists
    Path(viewer_log_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Inspect View on {viewer_host}:{viewer_port}")
    logger.info(f"Log directory: {viewer_log_dir}")

    try:
        # Set environment for FastAPI server mode
        viewer_env = os.environ.copy()
        viewer_env["INSPECT_VIEW_FASTAPI_SERVER"] = "true"

        _viewer_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "inspect_ai._cli.main",
                "view",
                "start",
                "--host",
                viewer_host,
                "--port",
                str(viewer_port),
                "--log-dir",
                viewer_log_dir,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=viewer_env,
        )

        logger.info(f"Inspect View started (PID: {_viewer_process.pid})")
        logger.info(f"Access at: http://{viewer_host}:{viewer_port}")
        return True

    except FileNotFoundError:
        logger.warning(
            "Could not start Inspect View: inspect-ai not installed or not in PATH"
        )
        logger.warning("Install with: pip install inspect-ai")
        return False
    except Exception as e:
        logger.warning(f"Could not start Inspect View: {e}")
        return False


def stop_inspect_view() -> None:
    """
    Stop the inspect view subprocess.

    This is registered as an atexit handler when start_inspect_view() is called,
    but can also be called manually.
    """
    global _viewer_process

    if _viewer_process is None:
        return

    logger.info("Stopping Inspect View...")

    try:
        # Send SIGTERM for graceful shutdown
        _viewer_process.send_signal(signal.SIGTERM)
        _viewer_process.wait(timeout=5)
        logger.info("Inspect View stopped gracefully")
    except subprocess.TimeoutExpired:
        # Force kill if graceful shutdown takes too long
        logger.warning("Inspect View did not stop gracefully, killing...")
        _viewer_process.kill()
        _viewer_process.wait()
    except Exception as e:
        logger.warning(f"Error stopping Inspect View: {e}")
    finally:
        _viewer_process = None


def get_viewer_url() -> str | None:
    """
    Get the URL for the running Inspect View.

    Returns:
        URL string if viewer is running, None otherwise

    Example:
        from health_universe_a2a.inspect_ai import start_inspect_view, get_viewer_url

        start_inspect_view()
        url = get_viewer_url()  # "http://127.0.0.1:7575"
    """
    if not is_viewer_running():
        return None
    return f"http://{get_viewer_host()}:{get_viewer_port()}"


def get_viewer_pid() -> int | None:
    """
    Get the PID of the running Inspect View process.

    Returns:
        Process ID if viewer is running, None otherwise
    """
    global _viewer_process
    if _viewer_process is None:
        return None
    return _viewer_process.pid


def get_viewer_status() -> dict:
    """
    Get status information about the Inspect View process.

    Returns:
        Dict with running status, URL, PID, and log directory

    Example:
        status = get_viewer_status()
        # {
        #     "running": True,
        #     "url": "http://127.0.0.1:7575",
        #     "pid": 12345,
        #     "log_dir": "./inspect_logs"
        # }
    """
    running = is_viewer_running()
    return {
        "running": running,
        "url": get_viewer_url() if running else None,
        "pid": get_viewer_pid() if running else None,
        "log_dir": get_log_dir(),
        "port": get_viewer_port(),
        "host": get_viewer_host(),
    }


# Register cleanup on exit
atexit.register(stop_inspect_view)
