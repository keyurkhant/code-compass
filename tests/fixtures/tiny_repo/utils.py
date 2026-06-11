"""Utility functions for the tiny test repo."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test app")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def validate_config(config) -> bool:
    """Validate that the config has sensible values."""
    if not config.host:
        return False
    return 1 <= config.port <= 65535
